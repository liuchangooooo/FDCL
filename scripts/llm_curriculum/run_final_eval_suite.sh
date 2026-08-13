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

CHECKPOINT="${CHECKPOINT:-}"
DATE_TAG="${DATE_TAG:-$(date +%Y.%m.%d)}"
MODE="${MODE:-policy_only}"
BENCHMARKS="${BENCHMARKS:-B,M,U,D}"
NUM_EPISODES="${NUM_EPISODES:-20}"
NUM_RENDER="${NUM_RENDER:-3}"
MAX_STEPS="${MAX_STEPS:-10}"
SEED="${SEED:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/outputs/${DATE_TAG}/final_eval}"
RUN_LABEL="${RUN_LABEL:-}"
DRY_RUN="${DRY_RUN:-0}"

EXTRA_ARGS=("$@")

if [[ -z "${CHECKPOINT}" ]]; then
    echo "[error] CHECKPOINT is required" >&2
    exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "[error] checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[error] python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ -z "${RUN_LABEL}" ]]; then
    RUN_LABEL="$(basename "$(dirname "$(dirname "${CHECKPOINT}")")")"
fi

BENCHMARK_LIST="[${BENCHMARKS}]"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_LABEL}"

cmd=(
    "${PYTHON_BIN}" evaluation.py
    --config-dir=config/evaluation
    --config-name=eval
    "mode=${MODE}"
    "policy.checkpoint=${CHECKPOINT}"
    "output_dir=${OUTPUT_DIR}"
    "num_episodes=${NUM_EPISODES}"
    "num_render=${NUM_RENDER}"
    "max_steps=${MAX_STEPS}"
    "seed=${SEED}"
    "benchmarks.order=${BENCHMARK_LIST}"
)
cmd+=("${EXTRA_ARGS[@]}")

echo
echo "==> ${RUN_LABEL}"
printf '    %q ' "${cmd[@]}"
echo

if [[ "${DRY_RUN}" == "1" ]]; then
    exit 0
fi

(
    cd "${REPO_ROOT}"
    "${cmd[@]}"
)
