#!/bin/bash
# ============================================================
# ACGS 对比实验启动脚本
# Exp A: TD3 + Random Obstacles (Baseline)
# Exp B: TD3 + LLM Curriculum Obstacles (Ours)
# ============================================================

set -e

# DeepSeek API Key (Exp B 需要)
export DEEPSEEK_API_KEY="sk-be61cee13ada4dec87d37b461a1793c0"

SEED=${1:-42}  # 可通过参数指定 seed，默认 42
echo "============================================"
echo " ACGS Comparison Experiment"
echo " Seed: $SEED"
echo "============================================"

# ============================================================
# 配置选择
# ============================================================
# 运行方式:
#   ./run_comparison.sh          → 提示选择实验
#   ./run_comparison.sh 42 A     → 直接跑 Baseline
#   ./run_comparison.sh 42 B     → 直接跑 LLM Curriculum
#   ./run_comparison.sh 42 AB    → 两个都跑(先A后B)

EXP_CHOICE=${2:-""}

run_baseline() {
    echo ""
    echo ">>> [Exp A] TD3 + Random Obstacles (Baseline)"
    echo "    Config: config/pusht/exp_baseline_random.yaml"
    echo "    Seed: $SEED"
    echo "    WandB: ACGS-Comparison / baseline"
    echo ""
    
    conda run -n divo python train.py \
        --config-name pusht/exp_baseline_random \
        training.seed=$SEED \
        exp_name="baseline_random_s${SEED}"
}

run_llm_curriculum() {
    echo ""
    echo ">>> [Exp B] TD3 + LLM Curriculum (Ours)"
    echo "    Config: config/pusht/exp_llm_curriculum.yaml"
    echo "    Seed: $SEED"
    echo "    WandB: ACGS-Comparison / llm_curriculum"
    echo "    API: DeepSeek"
    echo ""
    
    conda run -n divo python train.py \
        --config-name pusht/exp_llm_curriculum \
        training.seed=$SEED \
        exp_name="llm_curriculum_s${SEED}"
}

if [ "$EXP_CHOICE" == "A" ]; then
    run_baseline
elif [ "$EXP_CHOICE" == "B" ]; then
    run_llm_curriculum
elif [ "$EXP_CHOICE" == "AB" ]; then
    run_baseline
    run_llm_curriculum
else
    echo ""
    echo "选择要运行的实验:"
    echo "  A  - Baseline (TD3 + Random Obstacles)"
    echo "  B  - Ours (TD3 + LLM Curriculum)"
    echo "  AB - 两个都跑"
    echo ""
    read -p "输入选择 [A/B/AB]: " choice
    case $choice in
        A|a) run_baseline ;;
        B|b) run_llm_curriculum ;;
        AB|ab) run_baseline; run_llm_curriculum ;;
        *) echo "无效选择"; exit 1 ;;
    esac
fi
