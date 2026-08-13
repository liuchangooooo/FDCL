#!/usr/bin/env bash
# Usage: run_hard_dense_eval.sh <label> <checkpoint>
# Evaluates one checkpoint on the dense-OOD suite (H3..H6), 100 episodes, seeds 0/1/2, on GPU.
set -uo pipefail
source ~/anaconda3/etc/profile.d/conda.sh && conda activate divo

LABEL="$1"
CKPT="$2"
ROOT="data/outputs/2026.06.24/eval_hard_dense/${LABEL}"

for S in 0 1 2; do
  echo "==> ${LABEL} seed=${S}"
  python evaluation.py --config-dir=config/evaluation --config-name=eval_hard_dense \
    policy.checkpoint="${CKPT}" \
    output_dir="${ROOT}/seed${S}" \
    num_episodes=100 device=cuda:0 seed="${S}" 2>&1 | grep -iE "success_rate=|error|Traceback"
done
echo "DONE ${LABEL}"
