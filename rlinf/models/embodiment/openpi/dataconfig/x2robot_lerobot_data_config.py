# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""LeRobot x2robot / XSquare：``DataConfigFactory`` 子类，管线使用 RLinf ``arx_policy``。"""

from __future__ import annotations

import dataclasses
import logging
import pathlib

import etils.epath as epath
import numpy as np
import openpi.models.model as _model
import openpi.shared.normalize as _normalize
import openpi.transforms as _transforms
import tyro
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import arx_policy


def _individual_keys_repack() -> _transforms.Group:
    mapping: dict = {
        "images": {
            "left_wrist_view": "left_wrist_view",
            "face_view": "face_view",
            "right_wrist_view": "right_wrist_view",
        },
        "actions": "actions",
        "actions_is_pad": "actions_is_pad",
        "prompt": "task",
    }
    for k in arx_policy.ALL_COMPONENT_KEYS:
        mapping[k] = k
    return _transforms.Group(inputs=[_transforms.RepackTransform(mapping)])


@dataclasses.dataclass(frozen=True)
class RlinfLeRobotX2robotDataConfig(DataConfigFactory):
    mode: str | None = None
    use_delta_actions: bool = False
    mask_history_slave_states: bool = False
    action_dim: int = 14
    state_history_size: int = 0
    state_future_size: int = 0
    state_step: int = 1
    slave_state_dim: int = 14
    random_drop_master: float = 0.0
    random_drop_history: float = 0.0
    random_drop_future: float = 0.0
    random_pos_offset: float = 0.0
    only_right_obs: bool = False
    unified_input: bool = False
    individual_keys: bool = False
    prompt_from_task: bool = False
    prompt_meta_key: str | None = "meta"
    prompt_meta_dropout_p: float = 0.0
    prompt_meta_dropout_seed: int = 0

    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "left_wrist_view": "left_wrist_view",
                            "face_view": "face_view",
                            "right_wrist_view": "right_wrist_view",
                        },
                        "state": "state",
                        "actions": "actions",
                        "actions_is_pad": "actions_is_pad",
                        "prompt": "task",
                    }
                )
            ]
        )
    )

    @property
    def state_sequence_length(self) -> int:
        return self.state_history_size + 1 + self.state_future_size

    @override
    def create_base_config(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        if repo_id is not None and self.assets.asset_id is not None:
            asset_id = self.assets.asset_id
        elif repo_id is not None:
            asset_id = repo_id.replace(",", "_")
        else:
            asset_id = self.assets.asset_id

        merged_base = self.base_config or DataConfig()
        data_field_names = {f.name for f in dataclasses.fields(DataConfig)}

        kwargs: dict = {
            "repo_id": repo_id,
            "asset_id": asset_id,
            "norm_stats": self._load_norm_stats(
                epath.Path(self.assets.assets_dir or assets_dirs), asset_id
            ),
            "use_quantile_norm": model_config.model_type != _model.ModelType.PI0,
        }
        if "prompt_from_task" in data_field_names:
            kwargs["prompt_from_task"] = merged_base.prompt_from_task or self.prompt_from_task
        for meta_key in (
            "prompt_meta_key",
            "prompt_meta_dropout_p",
            "prompt_meta_dropout_seed",
        ):
            if meta_key in data_field_names:
                kwargs[meta_key] = getattr(self, meta_key)

        return dataclasses.replace(merged_base, **kwargs)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        assert self.mode in ("s2s", "s2m", "sm2m", "sm2sm"), f"Invalid mode: {self.mode}"

        repack = _individual_keys_repack() if self.individual_keys else self.repack_transforms

        data_transforms = _transforms.Group(
            inputs=[
                arx_policy.ArxInputs(
                    mode=self.mode,
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    state_history_size=self.state_history_size,
                    state_future_size=self.state_future_size,
                    slave_state_dim=self.slave_state_dim,
                    mask_history_slave_states=self.mask_history_slave_states,
                    random_drop_master=self.random_drop_master,
                    random_drop_history=self.random_drop_history,
                    random_drop_future=self.random_drop_future,
                    random_pos_offset=self.random_pos_offset,
                    only_right_obs=self.only_right_obs,
                    unified_input=self.unified_input,
                    individual_keys=self.individual_keys,
                )
            ],
            outputs=[arx_policy.ArxOutputs(action_dim=self.action_dim)],
        )
        if self.use_delta_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)
        base_config = self.create_base_config(assets_dirs, model_config)

        if (self.random_drop_master > 0.0 or self.random_drop_future > 0.0) and base_config.norm_stats is not None:
            norm_stats = dict(base_config.norm_stats)
            state_stats = norm_stats["state"]
            new_std = np.array(state_stats.std, copy=True)
            zero_var_indices = np.where(new_std == 0)[0]
            if len(zero_var_indices) > 0:
                new_std[zero_var_indices] = 1.0
                logging.info(
                    "Fixed %s zero-variance state dimensions: %s",
                    len(zero_var_indices),
                    zero_var_indices.tolist(),
                )
                norm_stats["state"] = _normalize.NormStats(
                    mean=state_stats.mean,
                    std=new_std,
                    q01=state_stats.q01,
                    q99=state_stats.q99,
                )
                base_config = dataclasses.replace(base_config, norm_stats=norm_stats)

        return dataclasses.replace(
            base_config,
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
