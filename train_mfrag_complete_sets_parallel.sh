#!/usr/bin/env bash
set -euo pipefail

cd /data2/project/kyk9174844/mfrag

GPUS=(0 1 2 3 4 5)

BATCH_SIZE="${BATCH_SIZE:-512}"
EPOCHS="${EPOCHS:-50}"
EVAL_EVERY="${EVAL_EVERY:-1}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PLOT_EMBEDDING="${PLOT_EMBEDDING:-1}"
PLOT_EVERY="${PLOT_EVERY:-5}"
DISTANCE_MODE="${DISTANCE_MODE:-l2}"
FRAG_DISTANCE_WEIGHT="${FRAG_DISTANCE_WEIGHT:-0.1}"
FRAG_CTR_WEIGHT="${FRAG_CTR_WEIGHT:-0.1}"

LOG_DIR="logs/mfrag_parallel/complete_only2_only2_retry_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

JOBS=(
  "sitagliptin_mpo reg shared ckpt/only2 reg"
  "parp1 reg dual ckpt/only2_retry reg"
  "fa7 reg dual ckpt/only2_retry reg"
  "5ht1b reg dual ckpt/only2_retry reg"
  "braf reg dual ckpt/only2_retry reg"
  "jak2 reg dual ckpt/only2_retry reg"
  "amlodipine_mpo reg dual ckpt/only2_retry reg"
  "fexofenadine_mpo reg dual ckpt/only2_retry reg"
  "osimertinib_mpo reg dual ckpt/only2_retry reg"
  "perindopril_mpo reg dual ckpt/only2_retry reg"
  "ranolazine_mpo reg dual ckpt/only2_retry reg"
  "zaleplon_mpo reg dual ckpt/only2_retry reg"
  "qed reg dual ckpt/only2_retry reg"
  "sa reg dual ckpt/only2_retry reg"
  "qed cls dual ckpt/only2_retry cls"
  "sa cls dual ckpt/only2_retry cls"
)

run_job() {
  local gpu="$1"
  local target="$2"
  local label_mode="$3"
  local model_arch="$4"
  local out_root="$5"
  local ctr_mode="$6"
  local log_path="${LOG_DIR}/${target}_${label_mode}_${model_arch}_gpu${gpu}.log"

  local cmd=(
    env CUDA_VISIBLE_DEVICES="${gpu}"
    conda run -n geam2 python train_mfrag.py
    -g 0
    -t "${target}"
    --label_mode "${label_mode}"
    --model_arch "${model_arch}"
    --train_mode joint
    --distance_mode "${DISTANCE_MODE}"
    --ctr_mode "${ctr_mode}"
    --frag_distance_weight "${FRAG_DISTANCE_WEIGHT}"
    --frag_ctr_weight "${FRAG_CTR_WEIGHT}"
    --batch_size "${BATCH_SIZE}"
    --epochs "${EPOCHS}"
    --eval_every "${EVAL_EVERY}"
    --early_stop_patience "${EARLY_STOP_PATIENCE}"
    --amp
    --num_workers "${NUM_WORKERS}"
    --out_root "${out_root}"
  )

  if [[ "${label_mode}" == "cls" && "${ctr_mode}" == "cls" ]]; then
    cmd+=(--ctr_bins 2)
  fi

  if [[ "${PLOT_EMBEDDING}" == "1" ]]; then
    cmd+=(--plot_embedding --plot_every "${PLOT_EVERY}")
  fi

  echo "[Start] gpu=${gpu} target=${target} label_mode=${label_mode} model_arch=${model_arch} out_root=${out_root} ctr_mode=${ctr_mode}" | tee -a "${log_path}"
  "${cmd[@]}" 2>&1 | tee -a "${log_path}"
  echo "[Done] gpu=${gpu} target=${target} label_mode=${label_mode} model_arch=${model_arch} out_root=${out_root} ctr_mode=${ctr_mode}" | tee -a "${log_path}"
}

QUEUE_DIR="$(mktemp -d)"
LOCK_DIR="${QUEUE_DIR}/lock"
NEXT_FILE="${QUEUE_DIR}/next"
printf "0\n" > "${NEXT_FILE}"

next_job() {
  local idx
  while ! mkdir "${LOCK_DIR}" 2>/dev/null; do
    sleep 0.1
  done

  idx="$(cat "${NEXT_FILE}")"
  if (( idx >= ${#JOBS[@]} )); then
    rmdir "${LOCK_DIR}"
    return 1
  fi

  printf "%s\n" "$((idx + 1))" > "${NEXT_FILE}"
  printf "%s\n" "${JOBS[$idx]}"
  rmdir "${LOCK_DIR}"
  return 0
}

for gpu in "${GPUS[@]}"; do
  (
    while job_line="$(next_job)"; do
      read -r target label_mode model_arch out_root ctr_mode <<< "${job_line}"
      run_job "${gpu}" "${target}" "${label_mode}" "${model_arch}" "${out_root}" "${ctr_mode}"
    done
  ) &
done

wait
rm -rf "${QUEUE_DIR}"
echo "[All done] logs=${LOG_DIR}"
