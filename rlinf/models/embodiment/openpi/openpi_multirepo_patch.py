# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import openpi.models.model as _model
import openpi.training.data_loader as openpi_data_loader
import openpi.transforms as _transforms

if TYPE_CHECKING:
    from collections.abc import Callable

    from openpi.training import config as _config
    from torch.utils.data import Dataset


def _create_multi_lerobot_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
) -> Dataset:
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    repo_ids = [r.strip() for r in str(data_config.repo_id).split(",") if r.strip()]
    if not repo_ids:
        raise ValueError("repo_id split by comma produced an empty repo id list")

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_ids[0])
    delta_timestamps = {
        key: [t / dataset_meta.fps for t in range(action_horizon)]
        for key in data_config.action_sequence_keys
    }
    dataset = lerobot_dataset.MultiLeRobotDataset(
        repo_ids=repo_ids,
        delta_timestamps=delta_timestamps,
    )
    if data_config.prompt_from_task:
        dataset = openpi_data_loader.TransformedDataset(
            dataset,
            [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)],
        )
    return dataset


@functools.cache
def apply_openpi_multirepo_patch() -> None:
    """将 ``openpi.training.data_loader.create_torch_dataset`` 替换为支持逗号分隔 ``repo_id`` 的实现。"""
    orig: Callable[..., Dataset] = openpi_data_loader.create_torch_dataset

    def create_torch_dataset(
        data_config: _config.DataConfig,
        action_horizon: int,
        model_config: _model.BaseModelConfig,
    ) -> Dataset:
        repo_id = data_config.repo_id
        if (
            repo_id is not None
            and repo_id != "fake"
            and "," in str(repo_id)
        ):
            return _create_multi_lerobot_torch_dataset(
                data_config, action_horizon, model_config
            )
        return orig(data_config, action_horizon, model_config)

    openpi_data_loader.create_torch_dataset = create_torch_dataset
