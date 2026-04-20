"""
测试 LLM 拓扑生成器

简单测试：
1. 生成拓扑生成器代码
2. 加载并执行代码
3. 验证生成的障碍物
"""

import os
import sys
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from DIVO.env.pusht.llm_topology_generator import (
    LLMTopologyGenerator,
    StrategyExecutor
)


def main():
    """运行测试"""
    print("\n" + "="*80)
    print("LLM 拓扑生成器测试")
    print("="*80)
    
    # 检查 API key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("\n⚠ 请设置 DEEPSEEK_API_KEY 环境变量")
        print("export DEEPSEEK_API_KEY='your-api-key'")
        return
    
    # 1. 初始化生成器
    print("\n1. 初始化 LLM 生成器...")
    generator = LLMTopologyGenerator(
        api_type="deepseek",
        api_key=api_key,
        verbose=True
    )
    
    # 2. 生成拓扑生成器代码
    print("\n2. 生成拓扑生成器代码...")
    tblock_pose = np.array([0.15, 0.15, np.pi/4])
    code = generator.generate_topology_generator(
        tblock_pose=tblock_pose,
        num_obstacles=2
    )
    
    if code is None:
        print("❌ 代码生成失败")
        return
    
    print(f"\n生成的完整代码:\n{code}\n")
    
    # 3. 初始化执行器
    print("\n3. 初始化 StrategyExecutor...")
    executor = StrategyExecutor(
        obstacle_size=0.01,
        target_pose=[0, 0, -np.pi/4]
    )
    
    # 4. 加载代码
    print("\n4. 加载拓扑生成器代码...")
    success = executor.load_topology_generator(code)
    
    if not success:
        print("❌ 代码加载失败")
        return
    
    # 5. 测试生成（模拟 10 个 episodes）
    print("\n5. 测试生成（模拟 10 个 episodes）...")
    
    for episode in range(10):
        # 随机生成 tblock_pose
        tblock_pose = np.array([
            np.random.uniform(-0.18, 0.18),
            np.random.uniform(-0.18, 0.18),
            np.random.uniform(0, 2*np.pi)
        ])
        
        print(f"\n--- Episode {episode+1}: T-block at ({tblock_pose[0]:.3f}, {tblock_pose[1]:.3f}, {np.degrees(tblock_pose[2]):.0f}°) ---")
        
        # 生成障碍物
        obstacles = executor.generate(tblock_pose, num_obstacles=2)
        
        if len(obstacles) > 0:
            print(f"生成了 {len(obstacles)} 个障碍物:")
            for j, obs in enumerate(obstacles):
                print(f"  障碍物 {j+1}: ({obs['x']:.3f}, {obs['y']:.3f}) - {obs.get('purpose', '')}")
            
            # 验证
            is_valid, reason = executor.validate_obstacles(obstacles, tblock_pose)
            print(f"验证结果: {'✓' if is_valid else '✗'} {reason}")
        else:
            print("⚠ 未生成任何障碍物")
    
    print("\n" + "="*80)
    print("测试完成")
    print("="*80)


if __name__ == "__main__":
    main()
