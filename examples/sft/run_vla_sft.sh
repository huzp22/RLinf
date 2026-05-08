#! /usr/bin/env bash
# VLA SFT 入口。对 OpenPI sm2sm 配置，可在训练前自动计算 norm（需 HF_LEROBOT_HOME）。
#
# 用法:
#   export HF_LEROBOT_HOME=/path/to/lerobot_root   # 与 data.train_data_paths 一致
#   bash run_vla_sft.sh x2robot_sm2sm_sft_openpi
#
# 跳过 norm（已有 norm_stats.json 时会自动跳过计算）:
#   SKIP_COMPUTE_NORM=1 bash run_vla_sft.sh ...
#
# 可选环境变量:
#   PI0_MODEL_PATH       默认 /mnt/public/datasets/pretrained-checkpoints/openpi-assets/checkpoints/pi0_base
#   OPENPI_NORM_CONFIG_NAME  默认 restock_goods_beijing_chengdu_sm2sm_norm
#   OPENPI_ASSET_ID      默认 restock_goods_beijing_chengdu_sm2sm（用于检测 norm_stats.json 路径）
#   OPENPI_NORM_NUM_WORKERS  norm DataLoader 进程数；默认 20（可用环境变量覆盖）
#   OPENPI_NORM_BATCH_SIZE   norm 扫库 batch；默认 128
#   PI0_PYTORCH_MODEL_PATH   若检测到 JAX Orbax（仅有 params/），自动转换输出目录；默认 ${PI0_MODEL_PATH}_pytorch
#   OPENPI_SKIP_JAX_TO_PT=1  禁用上述自动转换
#   WANDB_API_KEY          若 yaml 使用 wandb 日志，在 shell 中 export（勿写入仓库）
#   OPENPI_JAX_TO_PT_CONFIG_NAME  传给转换脚本的 OpenPI TrainConfig 名；默认 pi0_libero
#   OPENPI_JAX_TO_PT_PRECISION    默认 bfloat16
#   OPENPI_NORM_ASSETS_DIR   若 norm 写在非 model_path/assets 下，设为含 ``<asset_id>/norm_stats.json`` 的 assets 父目录
#   TF_CPP_MIN_LOG_LEVEL   默认 2，压低 Actor 子进程里 TensorFlow C++ 的 INFO（如 oneDNN）
#   TF_ENABLE_ONEDNN_OPTS  默认 0，去掉 oneDNN 提示；若要 Intel oneDNN 优化可设为 1
#   PYTHON               显式指定解释器；未设置时优先 python3，再 python

set -euo pipefail

if [ -n "${PYTHON:-}" ] && command -v "${PYTHON}" >/dev/null 2>&1; then
	:
elif command -v python3 >/dev/null 2>&1; then
	PYTHON=python3
elif command -v python >/dev/null 2>&1; then
	PYTHON=python
else
	echo "[run_vla_sft] 错误: 未找到 python3 或 python。请安装或: export PYTHON=/path/to/python"
	exit 1
fi

export EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH="$(dirname "$(dirname "$EMBODIED_PATH")")"
export SRC_FILE="${EMBODIED_PATH}/train_vla_sft.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_ENABLE_ONEDNN_OPTS="${TF_ENABLE_ONEDNN_OPTS:-0}"

export PYTHONPATH="${REPO_PATH}:${LIBERO_REPO_PATH:-}:${PYTHONPATH:-}"

export DREAMZERO_PATH="${DREAMZERO_PATH:-/path/to/DreamZero}"
export PYTHONPATH="${DREAMZERO_PATH}:${PYTHONPATH}"

if [ -z "${1:-}" ]; then
	CONFIG_NAME="maniskill_ppo_openvlaoft"
else
	CONFIG_NAME="$1"
	shift
fi
# 其余参数交给 Hydra（例如 data.train_data_paths=...）
EXTRA_HYDRA=( "$@" )

PI0_MODEL_PATH="${PI0_MODEL_PATH:-/mnt/public/datasets/pretrained-checkpoints/openpi-assets/checkpoints/pi0_base}"
OPENPI_NORM_CONFIG_NAME="${OPENPI_NORM_CONFIG_NAME:-restock_goods_beijing_chengdu_sm2sm_norm}"
OPENPI_ASSET_ID="${OPENPI_ASSET_ID:-restock_goods_beijing_chengdu_sm2sm}"
NORM_JSON="${PI0_MODEL_PATH}/assets/${OPENPI_ASSET_ID}/norm_stats.json"
CALC_NORM_PY="${REPO_PATH}/toolkits/lerobot/calculate_norm_stats.py"
CONVERT_JAX_PY="${REPO_PATH}/toolkits/openpi/convert_jax_model_to_pytorch.py"
OPENPI_MODEL_PATH_OVERRIDE=""

_needs_openpi_pi0_checkpoint_resolve() {
	case "${CONFIG_NAME}" in
	*openpi* | *sm2sm*) return 0 ;;
	esac
	return 1
}

# 若 model 目录为 JAX Orbax（params/）且无 safetensors，则转换为 PyTorch 并覆盖后续 PI0_MODEL_PATH 与 Hydra actor.model.model_path
_ensure_openpi_pytorch_checkpoint() {
	_needs_openpi_pi0_checkpoint_resolve || return 0
	local src="${PI0_MODEL_PATH}"
	[ -d "${src}/params" ] || return 0
	local has_st=0
	if [ -f "${src}/model.safetensors" ]; then
		has_st=1
	fi
	shopt -s nullglob
	local chunks=( "${src}"/*.safetensors )
	shopt -u nullglob
	if [ "${has_st}" = 1 ] || [ ${#chunks[@]} -gt 0 ]; then
		return 0
	fi
	if [ "${OPENPI_SKIP_JAX_TO_PT:-0}" = "1" ]; then
		echo "[run_vla_sft] 警告: ${src} 为 JAX Orbax 检查点但 OPENPI_SKIP_JAX_TO_PT=1，未自动转换；训练会因缺少 model.safetensors 失败。" >&2
		return 0
	fi
	local out="${PI0_PYTORCH_MODEL_PATH:-${src}_pytorch}"
	if [ -f "${out}/model.safetensors" ]; then
		echo "[run_vla_sft] 使用已有 PyTorch 权重: ${out}"
	else
		echo "[run_vla_sft] 检测到 JAX 检查点（${src} 有 params/ 且无 safetensors），正在转换为 PyTorch -> ${out} ..."
		"${PYTHON}" "${CONVERT_JAX_PY}" --checkpoint-dir "${src}" \
			--output-path "${out}" \
			--config-name "${OPENPI_JAX_TO_PT_CONFIG_NAME:-pi0_libero}" \
			--precision "${OPENPI_JAX_TO_PT_PRECISION:-bfloat16}"
	fi
	PI0_MODEL_PATH="${out}"
	OPENPI_MODEL_PATH_OVERRIDE="${out}"
	export PI0_MODEL_PATH
	NORM_JSON="${PI0_MODEL_PATH}/assets/${OPENPI_ASSET_ID}/norm_stats.json"
}

_ensure_openpi_pytorch_checkpoint

_run_openpi_norm_if_needed() {
	case "${CONFIG_NAME}" in
	*sm2sm* | x2robot_sm2sm_sft_openpi | restock_goods_beijing_chengdu_sm2sm_sft_openpi)
		return 0
		;;
	esac
	if [ "${OPENPI_COMPUTE_NORM_BEFORE_SFT:-0}" = "1" ]; then
		return 0
	fi
	return 1
}

if _run_openpi_norm_if_needed && [ "${SKIP_COMPUTE_NORM:-0}" != "1" ]; then
	if [ -z "${HF_LEROBOT_HOME:-}" ]; then
		echo "[run_vla_sft] 错误: 当前配置需要预先计算 norm，但未设置 HF_LEROBOT_HOME（LeRobot 数据根目录）。"
		echo "  export HF_LEROBOT_HOME=/你的/huggingface/lerobot"
		exit 1
	fi
	if [ -f "${NORM_JSON}" ]; then
		echo "[run_vla_sft] 已存在 ${NORM_JSON}，跳过 norm 计算。"
	else
		OPENPI_NORM_NUM_WORKERS="${OPENPI_NORM_NUM_WORKERS:-20}"
		OPENPI_NORM_BATCH_SIZE="${OPENPI_NORM_BATCH_SIZE:-128}"
		echo "[run_vla_sft] 正在计算 norm -> ${PI0_MODEL_PATH}/assets/${OPENPI_ASSET_ID}/ ..."
		echo "[run_vla_sft] num_workers=${OPENPI_NORM_NUM_WORKERS} batch_size=${OPENPI_NORM_BATCH_SIZE}"
		"${PYTHON}" "${CALC_NORM_PY}" \
			--config-name "${OPENPI_NORM_CONFIG_NAME}" \
			--model-path "${PI0_MODEL_PATH}" \
			--num-workers "${OPENPI_NORM_NUM_WORKERS}" \
			--batch-size "${OPENPI_NORM_BATCH_SIZE}"
		echo "[run_vla_sft] norm 已写入，开始 SFT。"
	fi
fi

echo "Using Python at $(command -v "${PYTHON}")"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"

CMD=("${PYTHON}" "${SRC_FILE}" --config-path "${EMBODIED_PATH}/config/" --config-name "${CONFIG_NAME}"
	runner.logger.log_path="${LOG_DIR}")
if [ -n "${OPENPI_MODEL_PATH_OVERRIDE}" ]; then
	CMD+=(actor.model.model_path="${OPENPI_MODEL_PATH_OVERRIDE}")
fi
if [ "${#EXTRA_HYDRA[@]}" -gt 0 ]; then
	CMD+=("${EXTRA_HYDRA[@]}")
fi
printf '%q ' "${CMD[@]}" >"${MEGA_LOG_FILE}"
echo >>"${MEGA_LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"
