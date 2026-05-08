# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""XSquare x2robot sm2sm（restock 多库等）OpenPI ``TrainConfig`` 注册。

``repo_id`` 可为逗号分隔的多数据集 ID；``assets.asset_id`` 保持稳定短名，
以便 ``norm_stats`` 与 checkpoint 内 ``assets/<asset_id>`` 对齐。
"""

from __future__ import annotations

import openpi.models.pi0_config as pi0_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
from openpi.training.config import AssetsConfig, DataConfig, TrainConfig

from rlinf.models.embodiment.openpi.dataconfig.x2robot_lerobot_data_config import (
    RlinfLeRobotX2robotDataConfig,
)

# 默认多库（逗号分隔，见 ``openpi_multirepo_patch``）；与 HF_LEROBOT_HOME 下子目录名一致。
_RESTOCK_GOODS_BEIJING_CHENGDU_SM2SM_REPO_ID = (
    "restock_chips_pys_0430,restock_cola_viee_hzp_0422,restock_cola_xpc_0429,"
    "restock_viee_water_cjx_0427,restock_water_pys_0423,restock_chips_xpc_0428,"
    "restock_cola_viee_water_chips_pys_0430,restock_mix_cjx_0429,restock_viee_water_hzp_0422,"
    "restore_viee_water_pys_0423,restock_chips_xpc_0429,restock_cola_viee_water_chips_xpc_0504,"
    "restock_viee_chips_cjx_0427,restock_viee_xpc_0503,restore_water_chips_pys_0423,"
    "restock_cola_chips_pys_0430,restock_cola_viee_xpc_0504,restock_viee_chips_hzp_0422,"
    "restock_water_chips_cjx_0427,retock_cola_viee_water_chips_cjx_0428,restock_cola_chips_xpc_0504,"
    "restock_cola_water_huzp_0422,restock_viee_hzp_0425,restock_water_chips_cjx_0428,"
    "retock_cola_water_cjx_0427,restock_cola_viee_chips_pys_0430,restock_cola_water_xpc_0504,"
    "restock_viee_hzp_0428,restock_water_cjx_0424,restock_cola_viee_chips_xpc_0503,"
    "restock_cola_xpc_0428,restock_viee_water_chips_xpc_0503,restock_water_hzp_0421"
)

RESTOCK_SM2SM_ASSET_ID = "restock_goods_beijing_chengdu_sm2sm"

REPO_BUNDLES_BY_ASSETS_ID: dict[str, dict] = {
    RESTOCK_SM2SM_ASSET_ID: {
        "repo_id": _RESTOCK_GOODS_BEIJING_CHENGDU_SM2SM_REPO_ID,
        "assets": AssetsConfig(asset_id=RESTOCK_SM2SM_ASSET_ID),
    },
}


def _sm2sm_data_common(
    *,
    assets_dir: str,
    use_quantile_norm: bool = False,
    for_norm: bool = False,
    state_history_size: int = 3,
    state_future_size: int = 2,
    action_dim: int = 28,
    prompt_meta_dropout_p: float = 0.5,
    prompt_meta_dropout_seed: int = 42,
) -> RlinfLeRobotX2robotDataConfig:
    """训练：随机增广（与 openpi sm2sm 脚本一致）；算 norm：全部关闭，统计更稳。"""
    if for_norm:
        random_drop_master = 0.0
        random_drop_history = 0.0
        random_drop_future = 0.0
        random_pos_offset = 0.0
        prompt_meta_dropout_p_eff = 0.0
    else:
        random_drop_master = 0.10
        random_drop_history = 0.50
        random_drop_future = 0.50
        random_pos_offset = 0.020
        prompt_meta_dropout_p_eff = prompt_meta_dropout_p

    return RlinfLeRobotX2robotDataConfig(
        repo_id=_RESTOCK_GOODS_BEIJING_CHENGDU_SM2SM_REPO_ID,
        mode="sm2sm",
        base_config=DataConfig(prompt_from_task=True, use_quantile_norm=use_quantile_norm),
        assets=AssetsConfig(
            assets_dir=assets_dir,
            asset_id=RESTOCK_SM2SM_ASSET_ID,
        ),
        use_delta_actions=False,
        action_dim=action_dim,
        state_history_size=state_history_size,
        state_future_size=state_future_size,
        slave_state_dim=14,
        individual_keys=False,
        random_drop_master=random_drop_master,
        random_drop_history=random_drop_history,
        random_drop_future=random_drop_future,
        random_pos_offset=random_pos_offset,
        prompt_meta_key="meta",
        prompt_meta_dropout_p=prompt_meta_dropout_p_eff,
        prompt_meta_dropout_seed=prompt_meta_dropout_seed,
    )


def x2square_sm2sm_train_configs() -> list[TrainConfig]:
    """返回 sm2sm 相关 ``TrainConfig``（训练、norm、通用别名）。"""
    # 预训练权重与 norm：``get_openpi_config(..., model_path=...)`` 可覆盖为本地 checkpoint。
    pi0_ckpt_root = "/mnt/public/datasets/pretrained-checkpoints/openpi-assets/checkpoints/pi0_base"
    base_assets = f"{pi0_ckpt_root}/assets"
    jax_params = f"{pi0_ckpt_root}/params"
    # Pi0Config：与 OpenPI restock 一致 action_horizon=20、action_dim=32（Pi0 默认）。
    def _pi0():
        return pi0_config.Pi0Config(action_horizon=20, action_dim=32)

    return [
        TrainConfig(
            name="restock_goods_beijing_chengdu_sm2sm",
            model=_pi0(),
            data=_sm2sm_data_common(assets_dir=base_assets, use_quantile_norm=False, for_norm=False),
            weight_loader=weight_loaders.CheckpointWeightLoader(jax_params),
            pytorch_weight_path=pi0_ckpt_root,
            batch_size=128,
            num_workers=20,
            seed=42,
            exp_name="restock_goods_beijing_chengdu_sm2sm_h3f2_a20_dm10dh50df50po20",
        ),
        TrainConfig(
            name="restock_goods_beijing_chengdu_sm2sm_norm",
            model=_pi0(),
            data=_sm2sm_data_common(assets_dir=base_assets, use_quantile_norm=False, for_norm=True),
            weight_loader=weight_loaders.CheckpointWeightLoader(jax_params),
            pytorch_weight_path=pi0_ckpt_root,
            batch_size=128,
            num_workers=20,
            seed=42,
        ),
        TrainConfig(
            name="pi0_restock_sm2sm",
            model=_pi0(),
            data=_sm2sm_data_common(assets_dir=base_assets, use_quantile_norm=False, for_norm=False),
            weight_loader=weight_loaders.CheckpointWeightLoader(jax_params),
            pytorch_weight_path=pi0_ckpt_root,
            batch_size=32,
            num_workers=20,
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            num_train_steps=30_000,
            log_interval=25,
            save_interval=2000,
            seed=42,
        ),
    ]
