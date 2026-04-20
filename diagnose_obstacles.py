#!/usr/bin/env python3
"""诊断障碍物生成问题"""

import os
import sys
import numpy as np
from DIVO.env.pusht.llm_topology_generator import (
    LLMTopologyGenerator,
    StrategyExecutor
)

def main():
    print("=" * 80)
    print("障碍物生成诊断")
    print("=" * 80)
    
    # 1. 检查 API key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ 未找到 DEEPSEEK_API_KEY")
        sys.exit(1)
    
    print(f"✓ API key: {api_key[:10]}...")
    
    # 2. 初始化生成器
    print("\n初始化 LLM Generator...")
    generator = LLMTopologyGenerator(
        api_type="deepseek",
        model="deepseek-chat",
        api_key=api_key
    )
    
    # 3. 生成拓扑生成器代码
    print("\n生成拓扑生成器代码...")
    tblock_pose = np.array([0.15, 0.15, np.pi/4])
    code = generator.generate_topology_generator(tblock_pose, num_obstacles=2)
    
    if code is None:
        print("❌ 代码生成失败")
        sys.exit(1)
    
    print(f"✓ 代码生成成功，长度: {len(code)} 字符")
    
    # 4. 加载到执行器
    print("\n加载到执行器...")
    executor = StrategyExecutor()
    if not executor.load_topology_generator(code):
        print("❌ 代码加载失败")
        sys.exit(1)
    
    print("✓ 代码加载成功")
    
    # 5. 测试生成多次
    print("\n" + "=" * 80)
    print("测试生成 10 次，检查障碍物数量")
    print("=" * 80)
    
    for i in range(10):
        # 随机 T-block 位置
        tblock_pose = np.array([
            np.random.uniform(-0.18, 0.18),
            np.random.uniform(-0.18, 0.18),
            np.random.uniform(-np.pi, np.pi)
        ])
        
        obstacles = executor.generate(tblock_pose, num_obstacles=2)
        
        print(f"\n测试 {i+1}:")
        print(f"  T-block: ({tblock_pose[0]:.3f}, {tblock_pose[1]:.3f}, {np.degrees(tblock_pose[2]):.0f}°)")
        print(f"  生成障碍物数量: {len(obstacles)}")
        
        if len(obstacles) != 2:
            print(f"  ⚠️ 警告: 期望 2 个，实际 {len(obstacles)} 个")
        
        for j, obs in enumerate(obstacles):
            print(f"    Obs{j+1}: x={obs['x']:.3f}, y={obs['y']:.3f}, purpose={obs.get('purpose', 'N/A')}")
    
    # 6. 统计
    print("\n" + "=" * 80)
    print("统计分析")
    print("=" * 80)
    
    counts = []
    purposes = []
    
    for _ in range(100):
        tblock_pose = np.array([
            np.random.uniform(-0.18, 0.18),
            np.random.uniform(-0.18, 0.18),
            np.random.uniform(-np.pi, np.pi)
        ])
        
        obstacles = executor.generate(tblock_pose, num_obstacles=2)
        counts.append(len(obstacles))
        
        for obs in obstacles:
            purposes.append(obs.get('purpose', 'unknown'))
    
    print(f"100 次生成统计:")
    print(f"  障碍物数量分布:")
    for n in set(counts):
        print(f"    {n} 个: {counts.count(n)} 次 ({counts.count(n)}%)")
    
    print(f"\n  Purpose 分布:")
    for p in set(purposes):
        print(f"    {p}: {purposes.count(p)} 次 ({purposes.count(p)/len(purposes)*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)

if __name__ == "__main__":
    main()
