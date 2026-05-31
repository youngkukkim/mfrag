#!/usr/bin/env bash
set -euo pipefail

cd /data2/project/kyk9174844/mfrag

GPU_IDS=(0 1 2 3 4 5 6)
TARGETS=(
  parp1
  fa7
  5ht1b
  braf
  jak2
  qed
  sa
)
JOB_GPUS=(
  0
  1
  2
  3
  4
  5
  6
)

FRAG_DESC_MODE="${FRAG_DESC_MODE:-raw_ecfp}"
MFRAG_ROOT="${MFRAG_ROOT:-ckpt/only2}"
MFRAG_LABEL_MODE="${MFRAG_LABEL_MODE:-reg}"
MFRAG_FINETUNE_TRIGGER="${MFRAG_FINETUNE_TRIGGER:-1000}"
MFRAG_FINETUNE_INTERVAL="${MFRAG_FINETUNE_INTERVAL:-1000}"
MFRAG_FINETUNE_EPOCHS="${MFRAG_FINETUNE_EPOCHS:-3}"
MFRAG_FINETUNE_BATCH_SIZE="${MFRAG_FINETUNE_BATCH_SIZE:-512}"
MFRAG_FINETUNE_LR="${MFRAG_FINETUNE_LR:-1e-4}"
NUM_MOLS="${NUM_MOLS:-6000}"
RUN_LOG_DIR="${RUN_LOG_DIR:-logs/run_mfrag_finetune}"
START_GAP_SECONDS="${START_GAP_SECONDS:-2}"

mkdir -p "${RUN_LOG_DIR}"

resolve_vocab() {
  local target="$1"
  local path="data/my2_${target}.txt"
  if [[ -f "${path}" ]]; then
    echo "${path}"
  else
    echo "data/my2_parp1.txt"
  fi
}

run_one() {
  local gpu="$1"
  local target="$2"
  local vocab_path
  local mfrag_ckpt
  local ts
  local log_path

  vocab_path="$(resolve_vocab "${target}")"
  mfrag_ckpt="${MFRAG_ROOT}/${MFRAG_LABEL_MODE}/${target}/best.pt"
  ts="$(date +%Y%m%d_%H%M%S)"
  log_path="${RUN_LOG_DIR}/${ts}_${target}_gpu${gpu}.log"

  if [[ ! -f "${mfrag_ckpt}" ]]; then
    echo "[SKIP] target=${target} missing_mfrag=${mfrag_ckpt}" | tee -a "${RUN_LOG_DIR}/launcher.log"
    return 0
  fi

  echo "[START] gpu=${gpu} target=${target} vocab=${vocab_path} mfrag=${mfrag_ckpt} log=${log_path}" | tee -a "${RUN_LOG_DIR}/launcher.log"
  conda run -n geam2 python run.py \
    -g "${gpu}" \
    -t "${target}" \
    -v "${vocab_path}" \
    --num_mols "${NUM_MOLS}" \
    --frag_desc_mode "${FRAG_DESC_MODE}" \
    --mfrag_root "${MFRAG_ROOT}" \
    --mfrag_label_mode "${MFRAG_LABEL_MODE}" \
    --enable_mfrag_finetune \
    --mfrag_finetune_trigger "${MFRAG_FINETUNE_TRIGGER}" \
    --mfrag_finetune_interval "${MFRAG_FINETUNE_INTERVAL}" \
    --mfrag_finetune_epochs "${MFRAG_FINETUNE_EPOCHS}" \
    --mfrag_finetune_batch_size "${MFRAG_FINETUNE_BATCH_SIZE}" \
    --mfrag_finetune_lr "${MFRAG_FINETUNE_LR}" \
    > "${log_path}" 2>&1
  echo "[DONE] gpu=${gpu} target=${target} log=${log_path}" | tee -a "${RUN_LOG_DIR}/launcher.log"
}

if [[ "${#TARGETS[@]}" -ne "${#JOB_GPUS[@]}" ]]; then
  echo "[ERROR] TARGETS and JOB_GPUS length mismatch" | tee -a "${RUN_LOG_DIR}/launcher.log"
  exit 1
fi

for idx in "${!TARGETS[@]}"; do
  target="${TARGETS[$idx]}"
  gpu="${JOB_GPUS[$idx]}"
  run_one "${gpu}" "${target}" &
  sleep "${START_GAP_SECONDS}"
done

wait
echo "[ALL DONE] logs=${RUN_LOG_DIR}" | tee -a "${RUN_LOG_DIR}/launcher.log"
