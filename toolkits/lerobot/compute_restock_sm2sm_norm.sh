#!/usr/bin/env bash
# 计算 sm2sm 多库 norm，并写入 Pi0 权重目录，供 RLinf SFT 使用：
#   <model_path>/assets/restock_goods_beijing_chengdu_sm2sm/
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# 数据根：其下为各 restock_* 子目录
: "${HF_LEROBOT_HOME:?请先 export HF_LEROBOT_HOME=你的 Lerobot 根目录}"

# 与 SFT yaml 中 actor.model.model_path 一致
: "${PI0_MODEL_PATH:=/mnt/public/datasets/pretrained-checkpoints/openpi-assets/checkpoints/pi0_base}"

OPENPI_NORM_NUM_WORKERS="${OPENPI_NORM_NUM_WORKERS:-20}"
OPENPI_NORM_BATCH_SIZE="${OPENPI_NORM_BATCH_SIZE:-128}"

cd "${REPO_ROOT}"
python toolkits/lerobot/calculate_norm_stats.py \
  --config-name restock_goods_beijing_chengdu_sm2sm_norm \
  --model-path "${PI0_MODEL_PATH}" \
  --num-workers "${OPENPI_NORM_NUM_WORKERS}" \
  --batch-size "${OPENPI_NORM_BATCH_SIZE}"

echo "完成。请确认存在: ${PI0_MODEL_PATH}/assets/restock_goods_beijing_chengdu_sm2sm/"
