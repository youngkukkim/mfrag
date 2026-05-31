#!/usr/bin/env bash
set -euo pipefail

cd /data2/project/kyk9174844/mfrag

GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-512}"
EPOCHS="${EPOCHS:-50}"
EVAL_EVERY="${EVAL_EVERY:-1}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PLOT_EMBEDDING="${PLOT_EMBEDDING:-1}"
PLOT_EVERY="${PLOT_EVERY:-5}"
OUT_ROOT="${OUT_ROOT:-ckpt/only2}"

LOG_DIR="logs/mfrag_parallel/sitagliptin_shared_noamp_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/sitagliptin_mpo_reg_shared_gpu${GPU}.log"

cmd=(
  env CUDA_VISIBLE_DEVICES="${GPU}"
  conda run -n geam2 python train_mfrag.py
  -g 0
  -t sitagliptin_mpo
  --label_mode reg
  --model_arch shared
  --train_mode joint
  --distance_mode l2
  --ctr_mode reg
  --frag_distance_weight 0.1
  --frag_ctr_weight 0.1
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --eval_every "${EVAL_EVERY}"
  --early_stop_patience "${EARLY_STOP_PATIENCE}"
  --num_workers "${NUM_WORKERS}"
  --out_root "${OUT_ROOT}"
)

if [[ "${PLOT_EMBEDDING}" == "1" ]]; then
  cmd+=(--plot_embedding --plot_every "${PLOT_EVERY}")
fi

echo "[Start] gpu=${GPU} target=sitagliptin_mpo label_mode=reg model_arch=shared out_root=${OUT_ROOT} amp=off" | tee -a "${LOG_PATH}"
"${cmd[@]}" 2>&1 | tee -a "${LOG_PATH}"
echo "[Done] gpu=${GPU} target=sitagliptin_mpo label_mode=reg model_arch=shared out_root=${OUT_ROOT} amp=off" | tee -a "${LOG_PATH}"
