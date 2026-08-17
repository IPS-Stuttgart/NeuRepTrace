#!/usr/bin/env bash
set -euo pipefail

: "${KATJA_BASELINE_RESULTS:?set KATJA_BASELINE_RESULTS}"
: "${KATJA_ACCURACY_PUSH_SHARDS:?set comma-separated KATJA_ACCURACY_PUSH_SHARDS}"
: "${KATJA_ACCURACY_PUSH_COMBINED:?set KATJA_ACCURACY_PUSH_COMBINED}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m neureptrace.katja_window_accuracy_push \
  --baseline-results "${KATJA_BASELINE_RESULTS}" \
  --aggregate-shards "${KATJA_ACCURACY_PUSH_SHARDS}" \
  --require-full-design \
  --out-dir "${KATJA_ACCURACY_PUSH_COMBINED}"
