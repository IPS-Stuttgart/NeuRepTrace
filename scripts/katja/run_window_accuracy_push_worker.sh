#!/usr/bin/env bash
set -euo pipefail

: "${KATJA_SCREEN_EXIT_FILE:?set KATJA_SCREEN_EXIT_FILE}"
: "${KATJA_WORKER_TARGETS:?set comma-separated KATJA_WORKER_TARGETS}"
: "${KATJA_WORKER_LOG_DIR:?set KATJA_WORKER_LOG_DIR}"

POLL_SECONDS="${KATJA_SCREEN_POLL_SECONDS:-60}"
mkdir -p "${KATJA_WORKER_LOG_DIR}"

while [[ ! -f "${KATJA_SCREEN_EXIT_FILE}" ]]; do
  sleep "${POLL_SECONDS}"
done

screen_exit="$(tr -d '[:space:]' < "${KATJA_SCREEN_EXIT_FILE}")"
if [[ "${screen_exit}" != "0" ]]; then
  echo "Adapter screen failed with exit ${screen_exit}" >&2
  exit "${screen_exit}"
fi

IFS=',' read -r -a targets <<< "${KATJA_WORKER_TARGETS}"
for target in "${targets[@]}"; do
  target="${target//[[:space:]]/}"
  [[ -n "${target}" ]] || continue
  export KATJA_TARGET="${target}"
  target_log="${KATJA_WORKER_LOG_DIR}/${target}.log"
  target_exit="${KATJA_WORKER_LOG_DIR}/${target}.exit"
  if [[ -f "${target_exit}" ]] && [[ "$(tr -d '[:space:]' < "${target_exit}")" == "0" ]]; then
    continue
  fi
  set +e
  scripts/katja/run_window_accuracy_push_target.sh >"${target_log}" 2>&1
  code=$?
  set -e
  printf '%s\n' "${code}" > "${target_exit}"
  if [[ "${code}" -ne 0 ]]; then
    echo "Target ${target} failed with exit ${code}; rerun the worker to resume." >&2
    exit "${code}"
  fi
done

printf '0\n' > "${KATJA_WORKER_LOG_DIR}/worker.exit"
