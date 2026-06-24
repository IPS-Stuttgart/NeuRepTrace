#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${BUSH_MEG_DATA_DIR:-}" ]]; then
  echo "BUSH_MEG_DATA_DIR must point to the BUSH-MEG data directory." >&2
  exit 2
fi

CONFIG="${CONFIG:-configs/bush_meg/all_protocols.yml}"
OUT_DIR="${OUT_DIR:-results/bush_meg/all_protocols/smoke}"
PARTICIPANTS="${PARTICIPANTS:-}"
SMOKE_PARTICIPANTS="${SMOKE_PARTICIPANTS:-${PARTICIPANTS:-1,2,3}}"
PARTICIPANT_LIMIT="${PARTICIPANT_LIMIT:-}"
FOLD_LIMIT="${FOLD_LIMIT:-2}"
WINDOW_LIMIT="${WINDOW_LIMIT:-1}"
N_JOBS="${N_JOBS:-1}"
METHOD_TIMEOUT_SECONDS="${METHOD_TIMEOUT_SECONDS:-}"
FOLD_TIMEOUT_SECONDS="${FOLD_TIMEOUT_SECONDS:-}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"

PYTHON_BIN="${PYTHON:-}"
python_has_deps() {
  "$1" -c "import matplotlib; import numpy; import pandas" >/dev/null 2>&1
}
if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3 python python.exe py; do
    if command -v "$candidate" >/dev/null 2>&1 && python_has_deps "$candidate"; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi
if [[ -z "$PYTHON_BIN" ]] || ! python_has_deps "$PYTHON_BIN"; then
  echo "Could not find Python with numpy, pandas, and matplotlib. Set PYTHON=/path/to/python." >&2
  exit 127
fi

METHODS=(
  source_loso_logistic
  source_loso_linear_svm
  source_loso_correlation_prototype
  source_loso_decoder_ensemble
  source_loso_response_window_c
  memory_bounded_loso_decode
  covariance_loso
  supervised_lowrank_loso
  source_probability_calibration_class_bias
  source_alignment_procrustes_group_projection
  source_alignment_hyperalignment_group_projection
  source_alignment_mcca_group_projection
  euclidean_alignment
  coral_alignment
  target_baseline_covariance
  subject_sensor_covariance
  dann
  cdan
  reconstruction_source_plus_target
  semi_supervised_lora_few_shot_k1
)

PROTOCOLS="1,2,3"
ORACLE_ARGS=()
if [[ "${SMOKE_INCLUDE_ORACLE:-0}" == "1" ]]; then
  PROTOCOLS="1,2,3,4"
  METHODS+=(
    oracle_target_calibrated_procrustes
    oracle_target_calibrated_hyperalignment
    oracle_target_calibrated_mcca
  )
  ORACLE_ARGS+=(--include-oracle)
fi

ARGS=(
  --config "$CONFIG"
  --data-dir "$BUSH_MEG_DATA_DIR"
  --out-dir "$OUT_DIR"
  --methods "$(IFS=,; echo "${METHODS[*]}")"
  --protocols "$PROTOCOLS"
  --smoke-participants "$SMOKE_PARTICIPANTS"
  --fold-limit "$FOLD_LIMIT"
  --window-limit "$WINDOW_LIMIT"
  --resume
  --n-jobs "$N_JOBS"
)

if [[ -n "$PARTICIPANT_LIMIT" ]]; then
  ARGS+=(--participant-limit "$PARTICIPANT_LIMIT")
fi

if [[ -n "$METHOD_TIMEOUT_SECONDS" ]]; then
  ARGS+=(--method-timeout-seconds "$METHOD_TIMEOUT_SECONDS")
fi

if [[ -n "$FOLD_TIMEOUT_SECONDS" ]]; then
  ARGS+=(--fold-timeout-seconds "$FOLD_TIMEOUT_SECONDS")
fi

"$PYTHON_BIN" -m neureptrace.bushmeg_all_protocols "${ARGS[@]}" "${ORACLE_ARGS[@]}"
