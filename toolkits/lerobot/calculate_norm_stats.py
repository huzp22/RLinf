# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import os
import pathlib

import numpy as np
import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms
import tqdm
import tyro
from openpi.training.config import DataConfig, TrainConfig

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config


def _default_norm_num_workers() -> int:
    """CPU 并行加载：上限避免进程过多导致 IPC 抖动。"""
    cpu = os.cpu_count() or 16
    return max(8, min(cpu, 64))


def _apply_norm_num_workers(config: TrainConfig, num_workers: int | None) -> TrainConfig:
    nw = num_workers if num_workers is not None else _default_norm_num_workers()
    return dataclasses.replace(config, num_workers=nw)


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {
            k: v
            for k, v in x.items()
            if not np.issubdtype(np.asarray(v).dtype, np.str_)
        }


def create_torch_dataloader(
    data_config: DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.TorchDataLoader, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(
        data_config, action_horizon, model_config
    )
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(
        data_config, action_horizon, batch_size, shuffle=False
    )
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str,
    repo_id: str = "",
    model_path: str | None = None,
    output_dir: str | None = None,
    num_workers: int | None = None,
    batch_size: int | None = None,
):
    """Compute OpenPI norm stats for ``state`` and ``actions``.

    Args:
        config_name: Registered TrainConfig name (e.g. ``restock_goods_beijing_chengdu_sm2sm_norm``).
        repo_id: Override dataset repo id(s). Comma-separated for multi-repo. Empty = use config default.
        model_path: If set, write stats under ``<model_path>/assets/<asset_id>/`` (training expects this).
        output_dir: Explicit output directory (overrides ``model_path`` default layout).
        num_workers: DataLoader worker 数；默认按 CPU 核数在 [8, 64] 内自动选取。
        batch_size: 覆盖 TrainConfig.batch_size；适当增大可加速扫库（占内存更多）。
    """
    if not os.environ.get("HF_LEROBOT_HOME"):
        raise EnvironmentError(
            "HF_LEROBOT_HOME must be set before running this script. "
            "Export it manually, for example: "
            "export HF_LEROBOT_HOME=/path/to/lerobot_root"
        )
    dk = {"repo_id": repo_id} if repo_id.strip() else None
    config = get_openpi_config(
        config_name,
        data_kwargs=dk,
        batch_size=batch_size,
    )
    config = _apply_norm_num_workers(config, num_workers)
    print(
        f"[calculate_norm_stats] batch_size={config.batch_size}, "
        f"num_workers={config.num_workers}"
    )
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    # 多库时优先用稳定 ``asset_id`` 作为子目录名。
    norm_subdir = data_config.asset_id or data_config.repo_id
    if norm_subdir is None:
        raise ValueError("Data config must have asset_id or repo_id for norm output path")

    if output_dir is not None:
        output_path = pathlib.Path(output_dir)
    elif model_path is not None:
        mp = pathlib.Path(model_path)
        assets_root = mp / "assets"
        assets_root.mkdir(parents=True, exist_ok=True)
        output_path = assets_root / str(norm_subdir)
    else:
        output_path = config.assets_dirs / str(norm_subdir)

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)

    # 兼容 openpi 加载：有时在同目录还期望重复的一份命名（仅在有 model_path 时提示）
    if model_path is not None:
        print(
            "Training will load norms from:",
            pathlib.Path(model_path) / "assets" / str(norm_subdir),
        )


if __name__ == "__main__":
    tyro.cli(main)
