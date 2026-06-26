#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${BUSH_MEG_DATA_DIR:-}" ]]; then
  echo "BUSH_MEG_DATA_DIR must point to the BUSH-MEG data directory." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_PROJECT_ROOT="$PROJECT_ROOT"
if command -v cygpath >/dev/null 2>&1; then
  PYTHON_PROJECT_ROOT="$(cygpath -m "$PROJECT_ROOT")"
elif [[ "$PROJECT_ROOT" == /mnt/*/* ]]; then
  drive_and_rest="${PROJECT_ROOT#/mnt/}"
  drive="${drive_and_rest%%/*}"
  rest="${drive_and_rest#*/}"
  drive_upper="$(printf '%s' "$drive" | tr '[:lower:]' '[:upper:]')"
  PYTHON_PROJECT_ROOT="$drive_upper:/$rest"
fi
cd "$PROJECT_ROOT"

CONFIG="${CONFIG:-configs/bush_meg/all_protocols.yml}"
OUT_DIR="${OUT_DIR:-results/bush_meg/all_protocols/protocol3_canary}"
PARTICIPANTS="${PARTICIPANTS:-1,2,3}"
FOLD_LIMIT="${FOLD_LIMIT:-1}"
WINDOW_LIMIT="${WINDOW_LIMIT:-1}"
N_JOBS="${N_JOBS:-1}"
METHOD_TIMEOUT_SECONDS="${METHOD_TIMEOUT_SECONDS:-1800}"
FOLD_TIMEOUT_SECONDS="${FOLD_TIMEOUT_SECONDS:-900}"
METHODS="${METHODS:-source_plus_target_calibration_logistic_k1,few_shot_target_calibrated_decoder_k1,target_calibrated_procrustes_k1}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PYTHON_PROJECT_ROOT/src"

PYTHON_CMD=()
python_has_deps() {
  "$@" -c "import numpy; import pandas; import sklearn" >/dev/null 2>&1
}
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_CMD=("$PYTHON")
elif command -v poetry >/dev/null 2>&1 && python_has_deps poetry run python; then
  PYTHON_CMD=(poetry run python)
else
  for candidate in python3 python python.exe py; do
    if command -v "$candidate" >/dev/null 2>&1 && python_has_deps "$candidate"; then
      PYTHON_CMD=("$candidate")
      break
    fi
  done
fi
if [[ "${#PYTHON_CMD[@]}" -eq 0 ]] || ! python_has_deps "${PYTHON_CMD[@]}"; then
  echo "Could not find Python with numpy, pandas, and sklearn. Set PYTHON=/path/to/python." >&2
  exit 127
fi

"${PYTHON_CMD[@]}" -c 'import sys; sys.path.insert(0, sys.argv[1]); from neureptrace.bushmeg_all_protocols import main; raise SystemExit(main(sys.argv[2:]))' "$PYTHON_PROJECT_ROOT/src" \
  --config "$CONFIG" \
  --data-dir "$BUSH_MEG_DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --participants "$PARTICIPANTS" \
  --participant-limit 3 \
  --fold-limit "$FOLD_LIMIT" \
  --window-limit "$WINDOW_LIMIT" \
  --methods "$METHODS" \
  --protocols 3 \
  --no-resume \
  --n-jobs "$N_JOBS" \
  --method-timeout-seconds "$METHOD_TIMEOUT_SECONDS" \
  --fold-timeout-seconds "$FOLD_TIMEOUT_SECONDS"

SUMMARY_CSV="$OUT_DIR/summary.csv"
PREDICTIONS_CSV="$OUT_DIR/predictions.csv"
METHOD_METADATA_CSV="$OUT_DIR/method_metadata.csv"
AUDIT_MD="$OUT_DIR/audit.md"

if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "Missing summary CSV: $SUMMARY_CSV" >&2
  exit 3
fi
if [[ ! -f "$PREDICTIONS_CSV" ]]; then
  echo "Missing predictions CSV: $PREDICTIONS_CSV" >&2
  exit 3
fi
if [[ ! -f "$METHOD_METADATA_CSV" ]]; then
  echo "Missing method metadata CSV: $METHOD_METADATA_CSV" >&2
  exit 3
fi

"${PYTHON_CMD[@]}" - "$SUMMARY_CSV" "$PREDICTIONS_CSV" "$METHOD_METADATA_CSV" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

summary_path = Path(sys.argv[1])
predictions_path = Path(sys.argv[2])
metadata_path = Path(sys.argv[3])
def read_required_csv(path: Path, label: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise SystemExit(f"{label} is empty: {path}") from exc
    if frame.empty:
        raise SystemExit(f"{label} is empty: {path}")
    return frame

summary = read_required_csv(summary_path, "summary.csv")
predictions = read_required_csv(predictions_path, "predictions.csv")
metadata = read_required_csv(metadata_path, "method_metadata.csv")

protocol3 = summary.loc[pd.to_numeric(summary["protocol_category"], errors="coerce").eq(3)].copy()
if protocol3.empty:
    raise SystemExit("summary.csv contains no Protocol 3 rows")

if not protocol3["calibration_rows_disjoint_from_evaluation"].astype(str).str.lower().eq("true").all():
    bad = protocol3.loc[
        ~protocol3["calibration_rows_disjoint_from_evaluation"].astype(str).str.lower().eq("true"),
        ["method", "outer_test_subject", "calibration_rows_disjoint_from_evaluation"],
    ]
    raise SystemExit("Protocol 3 calibration/evaluation split is not disjoint:\n" + bad.to_string(index=False))

if not protocol3["uses_target_labels_for_fitting"].astype(str).str.lower().eq("true").all():
    bad = protocol3.loc[
        ~protocol3["uses_target_labels_for_fitting"].astype(str).str.lower().eq("true"),
        ["method", "outer_test_subject", "uses_target_labels_for_fitting"],
    ]
    raise SystemExit("Protocol 3 rows do not all declare target labels for fitting:\n" + bad.to_string(index=False))

pred_protocol3 = predictions.loc[pd.to_numeric(predictions["protocol_category"], errors="coerce").eq(3)].copy()
if pred_protocol3.empty:
    raise SystemExit("predictions.csv contains no Protocol 3 prediction rows")

if "is_calibration_row" not in pred_protocol3.columns:
    raise SystemExit("predictions.csv lacks is_calibration_row, cannot verify calibration rows were excluded")
if pred_protocol3["is_calibration_row"].astype(str).str.lower().isin({"true", "1", "yes"}).any():
    bad = pred_protocol3.loc[
        pred_protocol3["is_calibration_row"].astype(str).str.lower().isin({"true", "1", "yes"}),
        [column for column in ("method", "outer_test_subject", "trial_index", "target_row_index", "is_calibration_row") if column in pred_protocol3.columns],
    ]
    raise SystemExit("Prediction rows include calibration rows:\n" + bad.to_string(index=False))

required_methods = {
    "source_plus_target_calibration_logistic_k1",
    "few_shot_target_calibrated_decoder_k1",
    "target_calibrated_procrustes_k1",
}
summary_methods = set(summary["method"].astype(str))
missing_methods = sorted(required_methods - summary_methods)
if missing_methods:
    raise SystemExit(f"Canary summary is missing method rows: {missing_methods}")

metadata_methods = set(metadata["method"].astype(str))
missing_metadata = sorted(required_methods - metadata_methods)
if missing_metadata:
    raise SystemExit(f"Canary method_metadata is missing method rows: {missing_metadata}")

print("Protocol 3 canary checks passed")
print(f"  summary rows: {len(summary)}")
print(f"  prediction rows: {len(predictions)}")
print(f"  methods: {', '.join(sorted(summary_methods & required_methods))}")
PY

"${PYTHON_CMD[@]}" -c 'import sys; sys.path.insert(0, sys.argv[1]); from neureptrace.bushmeg_all_protocols_audit import main; raise SystemExit(main(sys.argv[2:]))' "$PYTHON_PROJECT_ROOT/src" \
  --results-dir "$OUT_DIR" \
  --config "$CONFIG" \
  --out "$AUDIT_MD"

if [[ ! -f "$AUDIT_MD" ]]; then
  echo "Missing audit markdown: $AUDIT_MD" >&2
  exit 3
fi

printf 'Protocol 3 canary complete. Outputs written under %s\n' "$OUT_DIR"
