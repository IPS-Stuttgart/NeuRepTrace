#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${BUSH_MEG_DATA_DIR:-}" ]]; then
  echo "BUSH_MEG_DATA_DIR must point to the BUSH-MEG data directory." >&2
  exit 2
fi

CONFIG="${CONFIG:-configs/bush_meg/all_protocols.yml}"
OUT_DIR="${OUT_DIR:-results/bush_meg/all_protocols/full}"
PARTICIPANTS="${PARTICIPANTS:-}"
N_JOBS="${N_JOBS:-1}"
RESUME="${RESUME:-1}"
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
  protocol1_source_only
  protocol2_unlabeled_target_adaptive
  protocol3_few_shot_calibrated
)
PROTOCOLS="1,2,3"
ORACLE_ARGS=()

if [[ "${INCLUDE_ORACLE:-0}" == "1" ]]; then
  METHODS+=(protocol4_oracle_debug)
  PROTOCOLS="1,2,3,4"
  ORACLE_ARGS+=(--include-oracle)
fi

ARGS=(
  --config "$CONFIG"
  --data-dir "$BUSH_MEG_DATA_DIR"
  --out-dir "$OUT_DIR"
  --methods "$(IFS=,; echo "${METHODS[*]}")"
  --protocols "$PROTOCOLS"
  --n-jobs "$N_JOBS"
)

if [[ "$RESUME" == "0" ]]; then
  ARGS+=(--no-resume)
else
  ARGS+=(--resume)
fi

if [[ "${INCLUDE_HEAVY:-0}" == "1" ]]; then
  ARGS+=(--include-heavy)
fi

if [[ -n "$PARTICIPANTS" ]]; then
  ARGS+=(--participants "$PARTICIPANTS")
fi

if [[ -n "$METHOD_TIMEOUT_SECONDS" ]]; then
  ARGS+=(--method-timeout-seconds "$METHOD_TIMEOUT_SECONDS")
fi

if [[ -n "$FOLD_TIMEOUT_SECONDS" ]]; then
  ARGS+=(--fold-timeout-seconds "$FOLD_TIMEOUT_SECONDS")
fi

ARGS+=("$@")

"$PYTHON_BIN" -m neureptrace.bushmeg_all_protocols "${ARGS[@]}" "${ORACLE_ARGS[@]}"
