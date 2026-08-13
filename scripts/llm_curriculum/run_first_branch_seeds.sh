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
LOG_GROUP="${LOG_GROUP:-first_branch_g0}"
LOGGING_MODE="${LOGGING_MODE:-online}"
BRANCH_TAG="${BRANCH_TAG:-pre_first_evolve}"
SOURCE_ONLY="${SOURCE_ONLY:-0}"
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
    echo "[warn] source and evolve-branch runs need API access unless config supplies api_key." >&2
fi

run_cmd() {
    local run_name="$1"
    shift
    local -a cmd=("${PYTHON_BIN}" train.py --config-dir=config/pusht --config-name=exp_llm_curriculum)
    cmd+=("$@")
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
    source_run="branch_source_s${seed}"
    static_run="branch_static_s${seed}"
    evolve_run="branch_evolve_s${seed}"
    ckpt_path="${REPO_ROOT}/data/outputs/${DATE_TAG}/${source_run}/checkpoints/${BRANCH_TAG}.ckpt"

    run_cmd "${source_run}" \
        "obstacle_num=${OBSTACLE_NUM}" \
        "training.seed=${seed}" \
        "curriculum.generator.init_mode=file" \
        "curriculum.generator.init_path=${G0}" \
        "curriculum.evolve.enabled=True" \
        "curriculum.branch.save_pre_evolve_checkpoint=True" \
        "curriculum.branch.pre_evolve_checkpoint_tag=${BRANCH_TAG}" \
        "logging.group=${LOG_GROUP}" \
        "logging.name=${source_run}" \
        "logging.mode=${LOGGING_MODE}" \
        "hydra.run.dir=data/outputs/${DATE_TAG}/${source_run}"

    if [[ "${SOURCE_ONLY}" == "1" ]]; then
        if [[ "${DRY_RUN}" == "1" ]]; then
            echo "[info] source-only dry-run: expected checkpoint path ${ckpt_path}"
        else
            if [[ ! -f "${ckpt_path}" ]]; then
                echo "[error] branch checkpoint not found after source run: ${ckpt_path}" >&2
                exit 1
            fi
            echo "[info] source-only mode: checkpoint ready at ${ckpt_path}"
        fi
        continue
    fi

    if [[ "${DRY_RUN}" != "1" && ! -f "${ckpt_path}" ]]; then
        echo "[error] branch checkpoint not found after source run: ${ckpt_path}" >&2
        exit 1
    fi

    run_cmd "${static_run}" \
        "obstacle_num=${OBSTACLE_NUM}" \
        "training.seed=${seed}" \
        "curriculum.generator.init_mode=file" \
        "curriculum.generator.init_path=${G0}" \
        "curriculum.branch.resume_checkpoint=${ckpt_path}" \
        "curriculum.evolve.enabled=False" \
        "logging.group=${LOG_GROUP}" \
        "logging.name=${static_run}" \
        "logging.mode=${LOGGING_MODE}" \
        "hydra.run.dir=data/outputs/${DATE_TAG}/${static_run}"

    run_cmd "${evolve_run}" \
        "obstacle_num=${OBSTACLE_NUM}" \
        "training.seed=${seed}" \
        "curriculum.generator.init_mode=file" \
        "curriculum.generator.init_path=${G0}" \
        "curriculum.branch.resume_checkpoint=${ckpt_path}" \
        "curriculum.evolve.enabled=True" \
        "logging.group=${LOG_GROUP}" \
        "logging.name=${evolve_run}" \
        "logging.mode=${LOGGING_MODE}" \
        "hydra.run.dir=data/outputs/${DATE_TAG}/${evolve_run}"
done
