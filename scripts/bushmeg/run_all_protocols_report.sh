#!/usr/bin/env bash
set -euo pipefail

SUMMARY_DIR="${SUMMARY_DIR:-${1:-results/bush_meg/all_protocols/full}}"
SUMMARY_CSV="${SUMMARY_CSV:-$SUMMARY_DIR/summary.csv}"
METHOD_METADATA_CSV="${METHOD_METADATA_CSV:-$SUMMARY_DIR/method_metadata.csv}"
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

if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "Missing summary CSV: $SUMMARY_CSV" >&2
  exit 2
fi

if [[ ! -f "$METHOD_METADATA_CSV" ]]; then
  echo "Missing method metadata CSV: $METHOD_METADATA_CSV" >&2
  exit 2
fi

"$PYTHON_BIN" -m neureptrace.bushmeg_all_protocols_report \
  --summary-csv "$SUMMARY_CSV" \
  --method-metadata-csv "$METHOD_METADATA_CSV" \
  --out-dir "$SUMMARY_DIR"

"$PYTHON_BIN" - "$SUMMARY_DIR/leaderboard.csv" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

leaderboard_path = Path(sys.argv[1])
leaderboard = pd.read_csv(leaderboard_path)

if leaderboard.empty:
    print("No leaderboard rows found.")
    raise SystemExit(0)

for protocol in (1, 2, 3):
    rows = leaderboard.loc[
        (pd.to_numeric(leaderboard["protocol_category"], errors="coerce") == protocol)
        & (pd.to_numeric(leaderboard["n_rows"], errors="coerce").fillna(0) > 0)
    ].copy()
    print(f"\nProtocol {protocol} top methods")
    if rows.empty:
        print("  No runnable rows.")
        continue
    rows = rows.sort_values("mean_balanced_accuracy", ascending=False).head(10)
    for _, row in rows.iterrows():
        mean_ba = pd.to_numeric(pd.Series([row["mean_balanced_accuracy"]]), errors="coerce").iloc[0]
        sem_ba = pd.to_numeric(pd.Series([row.get("sem_balanced_accuracy")]), errors="coerce").iloc[0]
        sem_text = "NA" if pd.isna(sem_ba) else f"{sem_ba * 100:.2f} pp"
        print(
            f"  {row['method']}: {mean_ba * 100:.2f}% BA "
            f"(SEM {sem_text}, n_subjects={int(row['n_subjects'])}, n_rows={int(row['n_rows'])})"
        )
PY
