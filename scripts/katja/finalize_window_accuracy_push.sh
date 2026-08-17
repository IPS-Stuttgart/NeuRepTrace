#!/usr/bin/env bash
set -euo pipefail

: "${KATJA_BASELINE_RESULTS:?set KATJA_BASELINE_RESULTS}"
: "${KATJA_ACCURACY_PUSH_ROOT:?set KATJA_ACCURACY_PUSH_ROOT}"
: "${KATJA_ACCURACY_PUSH_COMBINED:?set KATJA_ACCURACY_PUSH_COMBINED}"
: "${KATJA_REMOTE_ROOT:?set KATJA_REMOTE_ROOT}"
: "${KATJA_REMOTE_HOST:?set KATJA_REMOTE_HOST}"
: "${KATJA_REMOTE_IDENTITY:?set KATJA_REMOTE_IDENTITY}"

PYTHON_BIN="${PYTHON_BIN:-python}"
POLL_SECONDS="${KATJA_FINALIZE_POLL_SECONDS:-300}"
LOCAL_TARGETS="${KATJA_LOCAL_TARGETS:-s05,s09,s15,s18}"
REMOTE_TARGETS="${KATJA_REMOTE_TARGETS:-s06,s08,s10,s11,s16,s17}"
ALL_TARGETS="${KATJA_ALL_TARGETS:-s05,s06,s08,s09,s10,s11,s15,s16,s17,s18}"
SSH=(ssh -i "${KATJA_REMOTE_IDENTITY}" -o BatchMode=yes "${KATJA_REMOTE_HOST}")

validated_shard() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
try:
    status = json.loads((root / "status.json").read_text())
    validation = json.loads((root / "validation.json").read_text())
except (FileNotFoundError, json.JSONDecodeError):
    raise SystemExit(1)
checks = validation.get("checks", {})
valid = (
    status.get("state") == "complete"
    and validation.get("all_required_checks_pass") is True
    and checks.get("all_expected_results_present") is True
)
raise SystemExit(0 if valid else 1)
PY
}

wait_for_local_target() {
  local target="$1"
  local root="${KATJA_ACCURACY_PUSH_ROOT}/target=${target}"
  until validated_shard "${root}"; do
    sleep "${POLL_SECONDS}"
  done
}

wait_for_remote_target() {
  local target="$1"
  local root="${KATJA_REMOTE_ROOT}/target=${target}"
  until "${SSH[@]}" "test -f '${root}/status.json' && test -f '${root}/validation.json' && grep -q '\"state\": \"complete\"' '${root}/status.json' && grep -q '\"all_expected_results_present\": true' '${root}/validation.json'"; do
    sleep "${POLL_SECONDS}"
  done
  mkdir -p "${KATJA_ACCURACY_PUSH_ROOT}/target=${target}"
  rsync -a --partial \
    -e "ssh -i ${KATJA_REMOTE_IDENTITY} -o BatchMode=yes" \
    "${KATJA_REMOTE_HOST}:${root}/" \
    "${KATJA_ACCURACY_PUSH_ROOT}/target=${target}/"
  validated_shard "${KATJA_ACCURACY_PUSH_ROOT}/target=${target}"
}

IFS=',' read -r -a local_targets <<< "${LOCAL_TARGETS}"
for target in "${local_targets[@]}"; do
  wait_for_local_target "${target//[[:space:]]/}"
done

IFS=',' read -r -a remote_targets <<< "${REMOTE_TARGETS}"
for target in "${remote_targets[@]}"; do
  wait_for_remote_target "${target//[[:space:]]/}"
done

IFS=',' read -r -a all_targets <<< "${ALL_TARGETS}"
shards=()
for target in "${all_targets[@]}"; do
  target="${target//[[:space:]]/}"
  shard="${KATJA_ACCURACY_PUSH_ROOT}/target=${target}"
  validated_shard "${shard}"
  shards+=("${shard}")
done

KATJA_ACCURACY_PUSH_SHARDS="$(IFS=,; echo "${shards[*]}")" \
  scripts/katja/aggregate_window_accuracy_push.sh
