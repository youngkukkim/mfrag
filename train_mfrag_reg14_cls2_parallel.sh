#!/usr/bin/env bash
set -euo pipefail

cd /data2/project/kyk9174844/mfrag

REG_TARGETS=(
  parp1
  fa7
  5ht1b
  braf
  jak2
  amlodipine_mpo
  fexofenadine_mpo
  osimertinib_mpo
  perindopril_mpo
  ranolazine_mpo
  sitagliptin_mpo
  zaleplon_mpo
  qed
  sa
)

CLS_TARGETS=(
  qed
  sa
)

GPUS=(0 1 2 3 4 5)

DISTANCE_MODE="${DISTANCE_MODE:-l2}"
REG_CTR_MODE="${REG_CTR_MODE:-reg}"
CLS_CTR_MODE="${CLS_CTR_MODE:-cls}"
MODEL_ARCH="${MODEL_ARCH:-shared}"
TRAIN_MODE="${TRAIN_MODE:-joint}"
MOL_CKPT_ROOT="${MOL_CKPT_ROOT:-}"
FRAG_DISTANCE_WEIGHT="${FRAG_DISTANCE_WEIGHT:-0.1}"
FRAG_CTR_WEIGHT="${FRAG_CTR_WEIGHT:-0.1}"

BATCH_SIZE="${BATCH_SIZE:-512}"
EPOCHS="${EPOCHS:-50}"
EVAL_EVERY="${EVAL_EVERY:-1}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PLOT_EMBEDDING="${PLOT_EMBEDDING:-1}"
PLOT_EVERY="${PLOT_EVERY:-5}"

OUT_ROOT="${OUT_ROOT:-ckpt/only2}"

JOB_TARGETS=()
JOB_LABELS=()

TARGET_FILTER="${TARGET_FILTER:-}"
SKIP_TARGETS="${SKIP_TARGETS:-}"

should_include_job() {
  local target="$1"
  local label="$2"
  local job_key="${label}/${target}"
  if [[ -n "${TARGET_FILTER}" ]]; then
    case " ${TARGET_FILTER} " in
      *" ${target} "*|*" ${job_key} "*) ;;
      *) return 1 ;;
    esac
  fi
  if [[ -n "${SKIP_TARGETS}" ]]; then
    case " ${SKIP_TARGETS} " in
      *" ${target} "*|*" ${job_key} "*) return 1 ;;
    esac
  fi
  return 0
}

for target in "${REG_TARGETS[@]}"; do
  should_include_job "${target}" reg || continue
  JOB_TARGETS+=("${target}")
  JOB_LABELS+=(reg)
done

for target in "${CLS_TARGETS[@]}"; do
  should_include_job "${target}" cls || continue
  JOB_TARGETS+=("${target}")
  JOB_LABELS+=(cls)
done

LOG_DIR="logs/mfrag_parallel/${MODEL_ARCH}_${OUT_ROOT//\\//_}_reg14_cls2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

run_target() {
  local gpu="$1"
  local target="$2"
  local label_mode="$3"
  local ctr_mode="${REG_CTR_MODE}"
  if [[ "${label_mode}" == "cls" ]]; then
    ctr_mode="${CLS_CTR_MODE}"
  fi
  local log_path="${LOG_DIR}/${target}_${label_mode}_gpu${gpu}.log"

  local cmd=(
    env CUDA_VISIBLE_DEVICES="${gpu}"
    conda run -n geam2 python train_mfrag.py
    -g 0
    -t "${target}"
    --label_mode "${label_mode}"
    --model_arch "${MODEL_ARCH}"
    --train_mode "${TRAIN_MODE}"
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
    --out_root "${OUT_ROOT}"
  )

  if [[ -n "${MOL_CKPT_ROOT}" ]]; then
    cmd+=(--mol_ckpt_root "${MOL_CKPT_ROOT}")
  fi

  if [[ "${FREEZE_MOL_GATHER:-0}" == "1" ]]; then
    cmd+=(--freeze_mol_gather)
  fi

  if [[ "${label_mode}" == "cls" && "${ctr_mode}" == "cls" ]]; then
    cmd+=(--ctr_bins 2)
  fi

  if [[ "${PLOT_EMBEDDING}" == "1" ]]; then
    cmd+=(--plot_embedding --plot_every "${PLOT_EVERY}")
  fi

  echo "[Start] gpu=${gpu} target=${target} label_mode=${label_mode} model_arch=${MODEL_ARCH} train_mode=${TRAIN_MODE} ctr_mode=${ctr_mode}" | tee -a "${log_path}"
  "${cmd[@]}" 2>&1 | tee -a "${log_path}"
  echo "[Done] gpu=${gpu} target=${target} label_mode=${label_mode} model_arch=${MODEL_ARCH} train_mode=${TRAIN_MODE} ctr_mode=${ctr_mode}" | tee -a "${log_path}"
}

QUEUE_DIR="$(mktemp -d)"
QUEUE_FILE="${QUEUE_DIR}/jobs.txt"
LOCK_DIR="${QUEUE_DIR}/lock"
NEXT_FILE="${QUEUE_DIR}/next"
printf "0\n" > "${NEXT_FILE}"

for target_idx in "${!JOB_TARGETS[@]}"; do
  printf "%s\t%s\n" "${JOB_TARGETS[$target_idx]}" "${JOB_LABELS[$target_idx]}" >> "${QUEUE_FILE}"
done

next_job() {
  local idx
  while ! mkdir "${LOCK_DIR}" 2>/dev/null; do
    sleep 0.1
  done

  idx="$(cat "${NEXT_FILE}")"
  if (( idx >= ${#JOB_TARGETS[@]} )); then
    rmdir "${LOCK_DIR}"
    return 1
  fi

  printf "%s\n" "$((idx + 1))" > "${NEXT_FILE}"
  sed -n "$((idx + 1))p" "${QUEUE_FILE}"
  rmdir "${LOCK_DIR}"
  return 0
}

for gpu in "${GPUS[@]}"; do
  (
    while job_line="$(next_job)"; do
      target="${job_line%%$'\t'*}"
      label_mode="${job_line##*$'\t'}"
      run_target "${gpu}" "${target}" "${label_mode}"
    done
  ) &
done

wait
rm -rf "${QUEUE_DIR}"
echo "[All done] logs=${LOG_DIR}"
