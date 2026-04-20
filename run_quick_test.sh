#!/bin/bash

# 快速测试脚本：验证 LLM 障碍物生成是否有效
# 预计运行时间：1-2 小时
# API 调用次数：约 50 次

echo "=========================================="
echo "LLM 障碍物生成快速测试"
echo "=========================================="
echo ""
echo "测试配置："
echo "  - 训练步数: 5000 steps"
echo "  - 预计 episodes: ~500"
echo "  - LLM 生成频率: 每 10 episodes"
echo "  - 预计 API 调用: ~50 次"
echo "  - 预计运行时间: 1-2 小时"
echo ""
echo "测试目标："
echo "  1. 验证 LLM 生成的障碍物是否符合物理约束"
echo "  2. 观察训练曲线是否正常"
echo "  3. 检查是否有明显的 bug"
echo ""
echo "=========================================="
echo ""

# 检查 API key
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "❌ 错误: 未设置 DEEPSEEK_API_KEY 环境变量"
    echo ""
    echo "请先设置 API key:"
    echo "  export DEEPSEEK_API_KEY='your-api-key'"
    exit 1
fi

echo "✓ API key 已设置"
echo ""
echo "开始训练..."
echo ""

# 运行训练
MUJOCO_GL=egl python train.py \
    --config-dir=config/pusht \
    --config-name=test_llm_quick

echo ""
echo "=========================================="
echo "测试完成"
echo "=========================================="
