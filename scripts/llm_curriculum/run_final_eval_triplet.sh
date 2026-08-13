#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SUITE_SCRIPT="${SCRIPT_DIR}/run_final_eval_suite.sh"

MANUAL_CKPT="${MANUAL_CKPT:-}"
STATIC_CKPT="${STATIC_CKPT:-}"
EVOLVE_CKPT="${EVOLVE_CKPT:-}"

DATE_TAG="${DATE_TAG:-$(date +%Y.%m.%d)}"
SUITE_TAG="${SUITE_TAG:-triplet_eval}"
BENCHMARKS="${BENCHMARKS:-B,M,U,D}"
NUM_EPISODES="${NUM_EPISODES:-20}"
NUM_RENDER="${NUM_RENDER:-3}"
MAX_STEPS="${MAX_STEPS:-10}"
SEED="${SEED:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/outputs/${DATE_TAG}/final_eval/${SUITE_TAG}}"
DRY_RUN="${DRY_RUN:-0}"

EXTRA_ARGS=("$@")

if [[ ! -x "${SUITE_SCRIPT}" && ! -f "${SUITE_SCRIPT}" ]]; then
    echo "[error] suite script not found: ${SUITE_SCRIPT}" >&2
    exit 1
fi

for pair in \
    "MANUAL_CKPT:${MANUAL_CKPT}" \
    "STATIC_CKPT:${STATIC_CKPT}" \
    "EVOLVE_CKPT:${EVOLVE_CKPT}"; do
    key="${pair%%:*}"
    value="${pair#*:}"
    if [[ -z "${value}" ]]; then
        echo "[error] ${key} is required" >&2
        exit 1
    fi
    if [[ ! -f "${value}" ]]; then
        echo "[error] ${key} not found: ${value}" >&2
        exit 1
    fi
done

run_one() {
    local label="$1"
    local checkpoint="$2"

    (
        cd "${REPO_ROOT}"
        CHECKPOINT="${checkpoint}" \
        RUN_LABEL="${label}" \
        DATE_TAG="${DATE_TAG}" \
        BENCHMARKS="${BENCHMARKS}" \
        NUM_EPISODES="${NUM_EPISODES}" \
        NUM_RENDER="${NUM_RENDER}" \
        MAX_STEPS="${MAX_STEPS}" \
        SEED="${SEED}" \
        OUTPUT_ROOT="${OUTPUT_ROOT}" \
        DRY_RUN="${DRY_RUN}" \
        bash "${SUITE_SCRIPT}" "${EXTRA_ARGS[@]}"
    )
}

run_one "manual_between" "${MANUAL_CKPT}"
run_one "llm_static" "${STATIC_CKPT}"
run_one "llm_evolve" "${EVOLVE_CKPT}"
