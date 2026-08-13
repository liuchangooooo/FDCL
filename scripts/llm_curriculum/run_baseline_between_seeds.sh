#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEFAULT_PYTHON="/home/hnu-w/anaconda3/envs/divo/bin/python"
if [[ -x "${DEFAULT_PYTHON}" ]]; then
    PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
else
    PYTHON_BIN="${PYTHON_BIN:-python}"
fi

SEEDS="${SEEDS:-0 1 2}"
OBSTACLE_NUM="${OBSTACLE_NUM:-2}"
DATE_TAG="${DATE_TAG:-$(date +%Y.%m.%d)}"
LOG_GROUP="${LOG_GROUP:-manual_between}"
LOGGING_MODE="${LOGGING_MODE:-online}"
DRY_RUN="${DRY_RUN:-0}"

EXTRA_ARGS=("$@")

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[error] python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

OBS_DIM=$((4 + 2 * OBSTACLE_NUM))
OBSACTION_DIM=$((OBS_DIM + 6))

run_train() {
    local seed="$1"
    local run_name="manual_between_s${seed}"
    local -a cmd=(
        "${PYTHON_BIN}" train.py
        --config-dir=config/pusht
        --config-name=exp_baseline_random
        "obs_dim=${OBS_DIM}"
        "obsaction_dim=${OBSACTION_DIM}"
        "env.obstacle_num=${OBSTACLE_NUM}"
        "env.obs_dim=[${OBS_DIM}]"
        "training.seed=${seed}"
        "logging.group=${LOG_GROUP}"
        "logging.name=${run_name}"
        "logging.mode=${LOGGING_MODE}"
        "hydra.run.dir=data/outputs/${DATE_TAG}/${run_name}"
    )
    cmd+=("${EXTRA_ARGS[@]}")

    echo
    echo "==> ${run_name}"
    printf '    %q ' "${cmd[@]}"
    echo

    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi

    (
        cd "${REPO_ROOT}"
        "${cmd[@]}"
    )
}

for seed in ${SEEDS}; do
    run_train "${seed}"
done
