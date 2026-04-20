#!/bin/bash

# Phase 0 对比实验运行脚本
#
# 对比两种障碍物生成方式：
# 1. LLM Stage A 生成（智能布局）
# 2. Baseline 随机生成（环境默认）
#
# 使用方法：
#   bash run_phase0_comparison.sh

set -e

echo "========================================="
echo "Phase 0 对比实验"
echo "========================================="
echo ""
echo "实验设置："
echo "- 训练 epochs: 500"
echo "- 障碍物数量: 2"
echo "- 每 10 episodes 更新一次障碍物配置"
echo ""
echo "对比组："
echo "1. LLM Stage A（智能布局）"
echo "2. Baseline（随机障碍物）"
echo ""
echo "========================================="

# 检查 API key
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "⚠ Warning: DEEPSEEK_API_KEY not set"
    echo "LLM 实验将无法运行，只运行 Baseline"
    echo ""
    
    # 只运行 Baseline
    echo "========================================="
    echo "运行 Baseline 实验（随机障碍物）"
    echo "========================================="
    python train.py task=pusht exp=phase0_baseline
    
    echo ""
    echo "========================================="
    echo "Baseline 实验完成"
    echo "========================================="
    exit 0
fi

# 运行 Baseline 实验
echo "========================================="
echo "[1/2] 运行 Baseline 实验（随机障碍物）"
echo "========================================="
python train.py task=pusht exp=phase0_baseline

echo ""
echo "========================================="
echo "[2/2] 运行 LLM 实验（Stage A 智能布局）"
echo "========================================="
python train.py task=pusht exp=phase0_llm

echo ""
echo "========================================="
echo "对比实验完成！"
echo "========================================="
echo ""
echo "查看结果："
echo "1. WandB: https://wandb.ai/your-project/pusht_phase0_comparison"
echo "2. 本地日志: outputs/"
echo ""
echo "对比指标："
echo "- 成功率（success_rate）"
echo "- Episode 奖励（episode_reward）"
echo "- Episode 长度（episode_length）"
echo "- 训练稳定性（reward 方差）"
echo ""
echo "========================================="
