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

DEFAULT_G0="${REPO_ROOT}/data/outputs/2026.04.27/20.24.50_td3_pusht_llm_curriculum/initial_generator.py"
G0="${G0:-${DEFAULT_G0}}"
SEEDS="${SEEDS:-0 1 2}"
OBSTACLE_NUM="${OBSTACLE_NUM:-2}"
DATE_TAG="${DATE_TAG:-$(date +%Y.%m.%d)}"
LOG_GROUP="${LOG_GROUP:-main_paired_g0}"
LOGGING_MODE="${LOGGING_MODE:-online}"
DRY_RUN="${DRY_RUN:-0}"

EXTRA_ARGS=("$@")

if [[ ! -f "${G0}" ]]; then
    echo "[error] initial generator not found: ${G0}" >&2
    exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[error] python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" && -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "[warn] OPENAI_API_KEY / DEEPSEEK_API_KEY not found in environment." >&2
    echo "[warn] static runs are fine; evolve runs still need API access unless config supplies api_key." >&2
fi

run_train() {
    local seed="$1"
    local mode="$2"
    local evolve_flag="false"
    local run_name="main_${mode}_s${seed}"

    if [[ "${mode}" == "evolve" ]]; then
        evolve_flag="true"
    fi

    local -a cmd=(
        "${PYTHON_BIN}" train.py
        --config-dir=config/pusht
        --config-name=exp_llm_curriculum
        "obstacle_num=${OBSTACLE_NUM}"
        "training.seed=${seed}"
        "curriculum.generator.init_mode=file"
        "curriculum.generator.init_path=${G0}"
        "curriculum.evolve.enabled=${evolve_flag}"
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
    run_train "${seed}" "static"
    run_train "${seed}" "evolve"
done

