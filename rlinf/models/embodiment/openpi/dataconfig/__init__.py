# Copyright 2025 The RLinf Authors.
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
# openpi model configs

import dataclasses
import difflib
import os
import pathlib
from typing import Any, Optional

import openpi.models.pi0_config as pi0_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
from omegaconf import DictConfig, OmegaConf
from openpi.training.config import (
    AssetsConfig,
    DataConfig,
    TrainConfig,
)

from rlinf.models.embodiment.openpi.dataconfig.x2square_sm2sm import (
    REPO_BUNDLES_BY_ASSETS_ID,
    RESTOCK_SM2SM_ASSET_ID,
    _RESTOCK_GOODS_BEIJING_CHENGDU_SM2SM_REPO_ID,
    x2square_sm2sm_train_configs,
)
from rlinf.models.embodiment.openpi.openpi_multirepo_patch import (
    apply_openpi_multirepo_patch,
)

from rlinf.models.embodiment.openpi.dataconfig.behavior_dataconfig import (
    LeRobotBehaviorDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.calvin_dataconfig import (
    LeRobotCalvinDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.franka_co_training_dataconfig import (
    LeRobotFrankaEEDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.franka_dataconfig import (
    CustomDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.gsenv_dataconfig import (
    LeRobotGSEnvDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.isaaclab_dataconfig import (
    LeRobotIsaacLabStackCubeDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.libero_dataconfig import (
    LeRobotLiberoDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.maniskill_dataconfig import (
    LeRobotManiSkillDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.metaworld_dataconfig import (
    LeRobotMetaworldDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.realworld_dataconfig import (
    LeRobotRealworldDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.robocasa_dataconfig import (
    LeRobotRobocasaDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.robotwin_aloha_dataconfig import (
    LeRobotAlohaDataConfig,
)

_CONFIGS = [
    TrainConfig(
        name="pi0_libero",
        model=pi0_config.Pi0Config(),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_libero/assets"),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi0_base/params"
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_libero/assets"),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi05_base"
        ),
        pytorch_weight_path="checkpoints/torch/pi05_base",
    ),
    TrainConfig(
        name="pi0_maniskill",
        model=pi0_config.Pi0Config(),
        data=LeRobotManiSkillDataConfig(
            repo_id="physical-intelligence/maniskill",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_base"),
            extra_delta_transform=False,
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
        seed=0,
        batch_size=32,
        num_workers=8,
        num_train_steps=200,  # 1_000, #30_000
        log_interval=5,  # 25,
        save_interval=50,  # 200,
    ),
    TrainConfig(
        name="pi05_maniskill",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False
        ),  # discrete_state_input=False: stateless policy, True: with state policy
        data=LeRobotManiSkillDataConfig(
            repo_id="physical-intelligence/maniskill",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi05_maniskill/assets"),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi05_base"
        ),
        pytorch_weight_path="checkpoints/torch/pi05_base",
        seed=0,
        batch_size=256,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_workers=8,
        num_train_steps=5_000,
        log_interval=5,
        save_interval=250,
    ),
    TrainConfig(
        name="pi05_franka",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=8, discrete_state_input=False
        ),  # discrete_state_input=False: stateless policy, True: with state policy
        data=LeRobotFrankaEEDataConfig(
            repo_id="physical-intelligence/real_rl",  # Not important
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="checkpoints/torch/pi05_franka_pretrained/assets"
            ),
            output_action_dim=6,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi05_base"
        ),
        pytorch_weight_path="checkpoints/torch/pi05_base",
        seed=0,
        batch_size=16,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_workers=8,
        num_train_steps=5_000,
        log_interval=5,
        save_interval=250,
    ),
    TrainConfig(
        name="pi05_maniskill_sim_real_co_training",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=8, discrete_state_input=False
        ),  # discrete_state_input=False: stateless policy, True: with state policy
        data=LeRobotFrankaEEDataConfig(
            repo_id="physical-intelligence/pick_and_place_real",
            default_prompt="default prompt",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="checkpoints/torch/pi05_maniskill_sim_real_co_training/assets"
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi05_base"
        ),
        pytorch_weight_path="checkpoints/torch/pi05_base",
        seed=0,
        batch_size=16,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_workers=8,
        num_train_steps=5_000,
        log_interval=5,
        save_interval=250,
    ),
    TrainConfig(
        name="pi0_metaworld",
        model=pi0_config.Pi0Config(action_horizon=5),
        data=LeRobotMetaworldDataConfig(
            repo_id="lerobot/metaworld_mt50",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_metaworld/assets"),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi0_base/params"
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
    ),
    TrainConfig(
        name="pi05_metaworld",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=5, discrete_state_input=False
        ),
        data=LeRobotMetaworldDataConfig(
            repo_id="lerobot/metaworld_mt50",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_metaworld/assets"),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi05_base/params"
        ),
        pytorch_weight_path="checkpoints/torch/pi05_base",
    ),
    TrainConfig(
        name="pi0_calvin",
        model=pi0_config.Pi0Config(action_horizon=5),
        data=LeRobotCalvinDataConfig(
            repo_id="InternRobotics/InternData-Calvin_ABC",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_calvin/assets"),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi0_base/params"
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_calvin",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=5, discrete_state_input=False
        ),
        data=LeRobotCalvinDataConfig(
            repo_id="InternRobotics/InternData-Calvin_ABC",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_calvin/assets"),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi05_base/params"
        ),
        pytorch_weight_path="checkpoints/torch/pi05_base",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_robocasa_human",
        model=pi0_config.Pi0Config(action_horizon=5),
        data=LeRobotRobocasaDataConfig(
            repo_id="daixianjie/robocasa_human_lerobot",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(asset_id="physical-intelligence/robocasa_all_human"),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi0_base/params"
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
        num_train_steps=100_000,
    ),
    TrainConfig(
        name="pi0_aloha_robotwin",
        model=pi0_config.Pi0Config(discrete_state_input=False),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/robotwin",
            adapt_to_pi=False,
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="checkpoints/torch/pi0_aloha_robotwin/assets"
            ),
            extra_delta_transform=True,  # True for delta action, False for abs_action
        ),
        freeze_filter=pi0_config.Pi0Config().get_freeze_filter(),
        pytorch_weight_path="checkpoints/torch/pi0_base",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_aloha_robotwin",
        model=pi0_config.Pi0Config(pi05=True, discrete_state_input=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/robotwin",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                assets_dir="checkpoints/torch/pi05_aloha_robotwin/assets"
            ),
            extra_delta_transform=True,  # True for delta action, False for abs_action
        ),
        pytorch_weight_path="checkpoints/torch/pi05_base",
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi0_behavior",
        model=pi0_config.Pi0Config(),
        data=LeRobotBehaviorDataConfig(
            repo_id="physical-intelligence/behavior",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_behavior/assets"),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi0_base/params"
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_behavior",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotBehaviorDataConfig(
            repo_id="physical-intelligence/behavior",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi05_behavior/assets"),
            extra_delta_transform=False,
            extract_state_from_proprio=True,
            use_all_wrist_images=True,
            use_quantile_norm=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi0_base/params"
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_gsenv",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=5, discrete_state_input=False
        ),
        data=LeRobotGSEnvDataConfig(
            repo_id="RLinf/GSEnv-PutCubeOnPlate-v0",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_r2s2r/assets"),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi05_base/params"
        ),
        pytorch_weight_path="checkpoints/torch/pi05_base",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_realworld",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=LeRobotRealworldDataConfig(
            repo_id="realworld_franka_bin_relocation",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_base/assets"),
            extra_delta_transform=False,
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
    ),
    TrainConfig(
        name="pi0_custom",
        model=pi0_config.Pi0Config(),
        data=CustomDataConfig(
            repo_id="physical-intelligence/custom_dataset",
            base_config=DataConfig(
                prompt_from_task=True
            ),  # we need language instruction
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_base/assets"),
            extra_delta_transform=False,  # True for delta action, False for abs_action
            action_train_with_rotation_6d=False,  # User can add extra config in custom dataset
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
    ),
    *x2square_sm2sm_train_configs(),
    TrainConfig(
        name="pi05_isaaclab_stack_cube",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False
        ),
        data=LeRobotIsaacLabStackCubeDataConfig(
            repo_id="RLinf/IsaacLab-Stack-Cube-Data",
            base_config=DataConfig(prompt_from_task=False),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_isaaclab/assets"),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi05_base/params"
        ),
        pytorch_weight_path="checkpoints/torch/pi05_base",
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}

apply_openpi_multirepo_patch()


def _to_plain_mapping(obj: Any) -> Any:
    if isinstance(obj, DictConfig):
        return OmegaConf.to_container(obj, resolve=True)
    return obj


def _coerce_openpi_data_kwargs(data_kwargs: dict) -> dict:
    """将 Hydra / 嵌套 dict 转为 ``dataclasses.replace`` 可用的字段。"""
    out = {k: _to_plain_mapping(v) for k, v in data_kwargs.items()}
    if "assets" in out and isinstance(out["assets"], dict):
        ad = out["assets"]
        out["assets"] = AssetsConfig(
            assets_dir=ad.get("assets_dir"),
            asset_id=ad.get("asset_id"),
        )
    return out


def _apply_assets_id_to_data_kwargs(data_kwargs: dict | None) -> dict | None:
    """``assets_id`` / ``dataset_bundle`` 展开为多库 ``repo_id`` 与稳定 ``assets.asset_id``。"""
    if not data_kwargs:
        return data_kwargs
    dk = dict(_coerce_openpi_data_kwargs(data_kwargs))
    bundle_key = dk.pop("dataset_bundle", None) or dk.pop("assets_id", None)
    if bundle_key is None:
        return dk
    if bundle_key not in REPO_BUNDLES_BY_ASSETS_ID:
        known = ", ".join(sorted(REPO_BUNDLES_BY_ASSETS_ID))
        raise ValueError(
            f"Unknown OpenPI data bundle or assets_id {bundle_key!r}. Known ids: {known}"
        )
    bundle = REPO_BUNDLES_BY_ASSETS_ID[bundle_key]
    merged = {**bundle, **dk}
    base_assets = bundle["assets"]
    user_assets = dk.get("assets")
    if isinstance(user_assets, AssetsConfig):
        merged["assets"] = dataclasses.replace(
            base_assets,
            assets_dir=user_assets.assets_dir or base_assets.assets_dir,
            asset_id=user_assets.asset_id or base_assets.asset_id,
        )
    elif isinstance(user_assets, dict):
        merged["assets"] = dataclasses.replace(
            base_assets,
            assets_dir=user_assets.get("assets_dir") or base_assets.assets_dir,
            asset_id=user_assets.get("asset_id") or base_assets.asset_id,
        )
    else:
        merged["assets"] = base_assets
    return merged


def _resolve_norm_assets_dir(model_path: str, asset_id: str | None) -> pathlib.Path:
    """选择含 ``<asset_id>/norm_stats.json`` 的 ``assets`` 父目录，供 OpenPI ``DataConfig`` 加载。

    常见情况：权重在 ``.../pi0_base_pytorch``（safetensors），而归一化写在原 JAX 目录
    ``.../pi0_base/assets/<asset_id>/``。此时应仍从 JAX 侧 ``assets`` 读 norm。

    查找顺序（后者仅在前序未找到 ``norm_stats.json`` 时尝试）：

    1. 环境变量 ``OPENPI_NORM_ASSETS_DIR``（若设置且为目录）
    2. ``<model_path>/assets``
    3. 若 ``model_path`` 目录名以 ``_pytorch`` 结尾，再试 ``<strip>/assets``
    """
    mp = pathlib.Path(model_path)
    default = mp / "assets" if (mp / "assets").is_dir() else mp
    if not asset_id:
        return default

    def _has_norm(assets_parent: pathlib.Path) -> bool:
        return (assets_parent / asset_id / "norm_stats.json").is_file()

    candidates: list[pathlib.Path] = []
    env_dir = os.environ.get("OPENPI_NORM_ASSETS_DIR")
    if env_dir:
        p = pathlib.Path(env_dir).expanduser()
        if p.is_dir():
            candidates.append(p)

    pt_assets = mp / "assets"
    if pt_assets.is_dir():
        candidates.append(pt_assets)

    suffix = "_pytorch"
    if mp.name.endswith(suffix):
        sibling = mp.with_name(mp.name[: -len(suffix)])
        jax_assets = sibling / "assets"
        if jax_assets.is_dir():
            candidates.append(jax_assets)

    seen: set[pathlib.Path] = set()
    for c in candidates:
        key = c.resolve()
        if key in seen:
            continue
        seen.add(key)
        if _has_norm(c):
            return c
    return default


def _override_with_model_path(config: TrainConfig, model_path: str) -> TrainConfig:
    """Return a copy of the config with assets/weight paths set from model_path.

    OpenPI PyTorch checkpoints usually store normalization stats under
    ``<model_path>/assets/<asset_id>/``. Older code pointed ``assets_dir`` at
    ``model_path`` itself, which breaks loading and yields "Normalization stats not found".
    """
    mp = pathlib.Path(model_path)
    asset_id = None
    if dataclasses.is_dataclass(config.data) and hasattr(config.data, "assets"):
        asset_id = getattr(config.data.assets, "asset_id", None)
    assets_dir_for_norm = str(_resolve_norm_assets_dir(model_path, asset_id))

    data_config = config.data
    if (
        dataclasses.is_dataclass(data_config)
        and hasattr(data_config, "assets")
        and dataclasses.is_dataclass(data_config.assets)
    ):
        data_config = dataclasses.replace(
            data_config,
            assets=dataclasses.replace(
                data_config.assets, assets_dir=assets_dir_for_norm
            ),
        )

    replace_kwargs = {
        "data": data_config,
        "pytorch_weight_path": model_path,
    }
    if dataclasses.is_dataclass(config) and any(
        field.name == "assets_dirs" for field in dataclasses.fields(config)
    ):
        replace_kwargs["assets_dirs"] = model_path

    return dataclasses.replace(config, **replace_kwargs)


def _override_with_data_kwargs(config: TrainConfig, data_kwargs: dict) -> TrainConfig:
    """Return a copy of the config with data_config set from openpi_data."""
    data_config = dataclasses.replace(config.data, **data_kwargs)
    replace_kwargs = {"data": data_config}
    return dataclasses.replace(config, **replace_kwargs)


def apply_hydra_openpi_to_train_model(config: TrainConfig, openpi_hydra: Any) -> TrainConfig:
    """把 Hydra ``actor.model.openpi`` 里与 ``TrainConfig.model`` 同名的字段合并进 ``model``。

    OpenPI 的 ``data.create(..., model_config)`` 用 ``model_config.max_token_len`` 等构造
    tokenizer；若仅改 yaml 而不合并，DataLoader 仍会用注册表里默认的 ``Pi0Config``（例如
    ``max_token_len=48``），与 ``get_model`` 里对 ``OpenPi0Config`` 的覆盖不一致。
    """
    if openpi_hydra is None:
        return config
    raw = (
        OmegaConf.to_container(openpi_hydra, resolve=True)
        if isinstance(openpi_hydra, DictConfig)
        else openpi_hydra
    )
    if not isinstance(raw, dict):
        return config
    model = config.model
    if not dataclasses.is_dataclass(model):
        return config
    skip = {"config_name"}
    field_names = {f.name for f in dataclasses.fields(model)}
    updates = {k: v for k, v in raw.items() if k not in skip and k in field_names}
    if not updates:
        return config
    new_model = dataclasses.replace(model, **updates)
    return dataclasses.replace(config, model=new_model)


def get_openpi_config(
    config_name: str,
    model_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    repo_id: Optional[str] = None,
    data_kwargs: Optional[dict] = None,
    openpi_hydra: Optional[Any] = None,
    num_workers: Optional[int] = None,
    seed: Optional[int] = None,
) -> TrainConfig:
    """Get a config by name.

    Args:
        config_name: Name of the config to load.
        model_path: Optional path to override model weights and assets.
        batch_size: Optional batch size override.
        repo_id: Optional LeRobot repo_id or local data path to override.
            When using a local path, the template ``assets.asset_id`` is kept when
            set (e.g. sm2sm 多库)，否则沿用原 ``repo_id`` 作为 norm 资源键。
        data_kwargs: Optional overrides for the data config factory. Supports
            ``assets_id`` / ``dataset_bundle`` to expand bundled multi-repo ids.
        openpi_hydra: Optional Hydra 节点 ``actor.model.openpi``；其中与 ``Pi0Config`` 等
            ``model`` 字段同名的项（如 ``max_token_len``、``action_horizon``）会合并进
            返回的 ``TrainConfig.model``，供 DataLoader 与权重侧一致。
        num_workers: 若设置，覆盖 ``TrainConfig.num_workers``（OpenPI DataLoader 子进程数）。
        seed: 若设置，覆盖 ``TrainConfig.seed``（OpenPI ``create_data_loader`` shuffle 等）。
    """
    if config_name in _CONFIGS_DICT:
        config = _CONFIGS_DICT[config_name]
    else:
        import openpi.training.config as openpi_train_cfg

        fallback = getattr(openpi_train_cfg, "_CONFIGS_DICT", None)
        if fallback is not None and config_name in fallback:
            config = fallback[config_name]
        else:
            closest = difflib.get_close_matches(
                config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0
            )
            closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
            raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    if model_path is not None:
        config = _override_with_model_path(config, model_path)
    if data_kwargs is not None:
        raw = (
            OmegaConf.to_container(data_kwargs, resolve=True)
            if isinstance(data_kwargs, DictConfig)
            else data_kwargs
        )
        if not isinstance(raw, dict):
            raise TypeError(f"openpi data_kwargs must be a mapping, got {type(raw)}")
        dk = _apply_assets_id_to_data_kwargs(_coerce_openpi_data_kwargs(dict(raw)))
        config = _override_with_data_kwargs(config, dk)
    if batch_size is not None:
        config = dataclasses.replace(config, batch_size=batch_size)

    if repo_id is not None:
        template_asset_id = config.data.assets.asset_id
        original_repo_id = config.data.repo_id
        preserved = template_asset_id if template_asset_id is not None else original_repo_id
        new_assets = dataclasses.replace(config.data.assets, asset_id=preserved)
        new_data = dataclasses.replace(config.data, repo_id=repo_id, assets=new_assets)
        config = dataclasses.replace(config, data=new_data)

    if openpi_hydra is not None:
        config = apply_hydra_openpi_to_train_model(config, openpi_hydra)

    if num_workers is not None:
        config = dataclasses.replace(config, num_workers=int(num_workers))

    if seed is not None:
        config = dataclasses.replace(config, seed=int(seed))

    return config
