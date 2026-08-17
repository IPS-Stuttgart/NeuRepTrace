#!/usr/bin/env bash
set -euo pipefail

: "${KATJA_WINDOW_CACHE:?set KATJA_WINDOW_CACHE}"
: "${KATJA_RAW_WINDOW_CACHE:?set KATJA_RAW_WINDOW_CACHE}"
: "${KATJA_BASELINE_RESULTS:?set KATJA_BASELINE_RESULTS}"
: "${KATJA_ADAPTER_SCREEN_ROOT:?set KATJA_ADAPTER_SCREEN_ROOT}"
: "${KATJA_ACCURACY_PUSH_ROOT:?set KATJA_ACCURACY_PUSH_ROOT}"
: "${KATJA_TARGET:?set KATJA_TARGET, for example s05}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
SELECTED_CONFIG="${KATJA_ADAPTER_SCREEN_ROOT}/adapter_screens/target=${KATJA_TARGET}/selected_adapter_config.json"
OUT_DIR="${KATJA_ACCURACY_PUSH_ROOT}/target=${KATJA_TARGET}"

if [[ ! -f "${SELECTED_CONFIG}" ]]; then
  echo "Missing frozen adapter screen: ${SELECTED_CONFIG}" >&2
  exit 2
fi

"${PYTHON_BIN}" -u -m neureptrace.katja_window_accuracy_push \
  --cache "${KATJA_WINDOW_CACHE}" \
  --raw-window-cache "${KATJA_RAW_WINDOW_CACHE}" \
  --baseline-results "${KATJA_BASELINE_RESULTS}" \
  --selected-adapter-config "${SELECTED_CONFIG}" \
  --out-dir "${OUT_DIR}" \
  --targets "${KATJA_TARGET}" \
  --k-values "1,3,5,10,15,20" \
  --split-seeds "0,1,2,3,4" \
  --model-seeds "0,1,2,3,4" \
  --context-modes "offline,causal" \
  --source-epochs 12 \
  --source-validation-patience 4 \
  --adapter-steps 100 \
  --last-block-steps 80 \
  --full-finetune-steps 80 \
  --context-source-epochs 6 \
  --context-adaptation-steps 80 \
  --device "${DEVICE}" \
  --resume
