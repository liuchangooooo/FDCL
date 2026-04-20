"""
Phase 0 拓扑生成器使用示例

演示如何使用 V4 的新功能：
1. 生成拓扑生成器代码
2. 加载并执行代码
3. 在训练循环中使用
"""

import os
import sys
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from DIVO.env.pusht.llm_obstacle_generator_v4 import (
    LLMObstacleGeneratorV3,
    StrategyExecutor
)


def example_basic_usage():
    """示例 1: 基本用法"""
    print("\n" + "="*80)
    print("示例 1: 基本用法")
    print("="*80)
    
    # 检查 API key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("⚠ 请设置 DEEPSEEK_API_KEY 环境变量")
        print("export DEEPSEEK_API_KEY='your-api-key'")
        return
    
    # 1. 初始化 LLM 生成器
    print("\n1. 初始化 LLM 生成器...")
    generator = LLMObstacleGeneratorV3(
        api_type="deepseek",
        api_key=api_key,
        verbose=True
    )
    
    # 2. 生成拓扑生成器代码（只需调用一次）
    print("\n2. 生成拓扑生成器代码...")
    tblock_pose = np.array([0.15, 0.15, np.pi/4])
    code = generator.generate_phase0_topology_generator(
        tblock_pose=tblock_pose,
        num_obstacles=2
    )
    
    if code is None:
        print("❌ 代码生成失败")
        return
    
    print(f"\n生成的代码:\n{code}")
    
    # 3. 初始化执行器
    print("\n3. 初始化 StrategyExecutor...")
    executor = StrategyExecutor(
        obstacle_size=0.01,
        target_pose=[0, 0, -np.pi/4]
    )
    
    # 4. 加载拓扑生成器
    print("\n4. 加载拓扑生成器...")
    success = executor.load_topology_generator(code)
    
    if not success:
        print("❌ 代码加载失败")
        return
    
    # 5. 使用拓扑生成器生成障碍物（可以调用多次）
    print("\n5. 使用拓扑生成器生成障碍物...")
    
    for i in range(5):
        # 每次随机一个新的 tblock_pose
        tblock_pose = np.array([
            np.random.uniform(-0.18, 0.18),
            np.random.uniform(-0.18, 0.18),
            np.random.uniform(0, 2*np.pi)
        ])
        
        print(f"\n--- 生成 {i+1}: T-block at ({tblock_pose[0]:.3f}, {tblock_pose[1]:.3f}, {np.degrees(tblock_pose[2]):.0f}°) ---")
        
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
    
    print("\n✓ 基本用法示例完成")


def example_training_loop():
    """示例 2: 在训练循环中使用"""
    print("\n" + "="*80)
    print("示例 2: 在训练循环中使用")
    print("="*80)
    
    # 检查 API key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("⚠ 请设置 DEEPSEEK_API_KEY 环境变量")
        return
    
    # Stage 0: 冷启动
    print("\n=== Stage 0: Cold Start ===")
    
    # 初始化
    generator = LLMObstacleGeneratorV3(
        api_type="deepseek",
        api_key=api_key,
        verbose=False  # 训练时关闭详细输出
    )
    
    executor = StrategyExecutor(
        obstacle_size=0.01,
        target_pose=[0, 0, -np.pi/4]
    )
    
    # 生成初始拓扑生成器
    print("\n生成初始拓扑生成器...")
    initial_tblock_pose = np.array([0.15, 0.15, np.pi/4])
    code = generator.generate_phase0_topology_generator(
        tblock_pose=initial_tblock_pose,
        num_obstacles=2
    )
    
    if code is None:
        print("❌ 代码生成失败")
        return
    
    # 加载到执行器
    print("加载拓扑生成器...")
    success = executor.load_topology_generator(code)
    
    if not success:
        print("❌ 代码加载失败")
        return
    
    # 模拟训练 100 episodes
    print("\n开始训练（模拟 100 episodes）...")
    
    success_count = 0
    valid_count = 0
    
    for episode in range(100):
        # 随机生成 tblock_pose
        tblock_pose = np.array([
            np.random.uniform(-0.18, 0.18),
            np.random.uniform(-0.18, 0.18),
            np.random.uniform(0, 2*np.pi)
        ])
        
        # 生成障碍物
        obstacles = executor.generate(tblock_pose, num_obstacles=2)
        
        # 验证
        is_valid, reason = executor.validate_obstacles(obstacles, tblock_pose)
        
        if is_valid:
            valid_count += 1
            
            # 这里应该：
            # env.set_obstacle_config(obstacles)
            # obs = env.reset()
            # ... 训练循环 ...
            
            # 模拟训练结果
            success = np.random.random() > 0.3  # 假设 70% 成功率
            if success:
                success_count += 1
        
        # 每 10 个 episode 打印一次进度
        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode+1}/100: "
                  f"有效率 {valid_count/(episode+1)*100:.1f}%, "
                  f"成功率 {success_count/(episode+1)*100:.1f}%")
    
    print(f"\n训练完成:")
    print(f"  有效配置: {valid_count}/100 ({valid_count}%)")
    print(f"  成功 episodes: {success_count}/100 ({success_count}%)")
    
    print("\n✓ 训练循环示例完成")


def example_compare_v3_v4():
    """示例 3: 对比 V3 和 V4 的成本"""
    print("\n" + "="*80)
    print("示例 3: 对比 V3 和 V4 的成本")
    print("="*80)
    
    print("\n假设训练 100 个 episodes:")
    
    print("\n【V3 方式】生成坐标")
    print("  - 每个 episode 调用一次 LLM")
    print("  - 总 LLM 调用次数: 100 次")
    print("  - 估计成本: 100 × $0.001 = $0.10")
    print("  - 估计时间: 100 × 2秒 = 200秒")
    
    print("\n【V4 方式】生成拓扑生成器")
    print("  - 只在开始时调用一次 LLM")
    print("  - 总 LLM 调用次数: 1 次")
    print("  - 估计成本: 1 × $0.001 = $0.001")
    print("  - 估计时间: 1 × 2秒 = 2秒")
    
    print("\n【节省】")
    print("  - 成本节省: 99%")
    print("  - 时间节省: 99%")
    
    print("\n✓ 成本对比示例完成")


def main():
    """运行所有示例"""
    print("\n" + "="*80)
    print("Phase 0 拓扑生成器使用示例")
    print("="*80)
    
    # 示例 1: 基本用法
    example_basic_usage()
    
    # 示例 2: 训练循环
    example_training_loop()
    
    # 示例 3: 成本对比
    example_compare_v3_v4()
    
    print("\n" + "="*80)
    print("所有示例完成")
    print("="*80)


if __name__ == "__main__":
    main()
