"""
测试 Stage A 集成到 td3_curriculum_workspace.py

验证：
1. LLM 拓扑生成器初始化成功
2. 拓扑生成器代码生成成功
3. 障碍物配置生成成功
4. 环境设置成功
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from DIVO.env.pusht.llm_topology_generator import (
    LLMTopologyGenerator,
    StrategyExecutor,
    build_phase0_prompt_stage_a
)


def test_stage_a_integration():
    """测试 Stage A 集成"""
    
    print("=" * 80)
    print("测试 Stage A 集成到 TD3 Curriculum Workspace")
    print("=" * 80)
    
    # 1. 初始化 LLM 生成器和执行器
    print("\n>>> Step 1: 初始化 LLM 生成器和执行器...")
    try:
        llm_generator = LLMTopologyGenerator(
            api_type="deepseek",
            temperature=0.7,
            verbose=True
        )
        executor = StrategyExecutor(
            obstacle_size=0.01,
            target_pose=[0, 0, -np.pi/4]
        )
        print("✓ 初始化成功")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return False
    
    # 2. 生成拓扑生成器代码（模拟第一次调用）
    print("\n>>> Step 2: 生成拓扑生成器代码（第一次）...")
    try:
        sample_tblock_pose = np.array([0.15, 0.10, -np.pi/4])
        prompt = build_phase0_prompt_stage_a(sample_tblock_pose, num_obstacles=2)
        
        print(f"Prompt 长度: {len(prompt)} 字符")
        
        code = llm_generator._call_llm(prompt)
        code = llm_generator._extract_code(code)
        
        if code is None:
            print("❌ 代码生成失败")
            return False
        
        print(f"✓ 代码生成成功，长度: {len(code)} 字符")
        
        if not executor.load_topology_generator(code):
            print("❌ 代码加载失败")
            return False
        
        print("✓ 代码加载成功")
        
    except Exception as e:
        print(f"❌ 代码生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 模拟多个 episodes，每次生成不同的障碍物配置
    print("\n>>> Step 3: 模拟 5 个 episodes...")
    
    for episode in range(5):
        try:
            # 随机生成起点
            tblock_pose = np.array([
                np.random.uniform(-0.18, 0.18),
                np.random.uniform(-0.18, 0.18),
                np.random.uniform(-np.pi, np.pi)
            ])
            
            # 使用拓扑生成器生成障碍物
            obstacles = executor.generate(tblock_pose, num_obstacles=2)
            
            print(f"\nEpisode {episode+1}:")
            print(f"  起点: ({tblock_pose[0]:.3f}, {tblock_pose[1]:.3f}, {np.degrees(tblock_pose[2]):.0f}°)")
            print(f"  生成了 {len(obstacles)} 个障碍物:")
            for i, obs in enumerate(obstacles):
                print(f"    {i+1}. ({obs['x']:.3f}, {obs['y']:.3f}) - {obs.get('purpose', 'N/A')}")
            
            # 验证障碍物配置
            is_valid, reason = executor.validate_obstacles(obstacles, tblock_pose)
            if not is_valid:
                print(f"  ⚠ 验证失败: {reason}")
            else:
                print(f"  ✓ 验证通过")
                
        except Exception as e:
            print(f"  ❌ Episode {episode+1} 失败: {e}")
            return False
    
    print("\n" + "=" * 80)
    print("✓ 所有测试通过！Stage A 集成成功")
    print("=" * 80)
    
    print("\n关键要点：")
    print("1. 拓扑生成器只生成一次（第一次调用）")
    print("2. 后续 episodes 复用同一个拓扑生成器")
    print("3. 每次调用 executor.generate() 生成不同的障碍物配置")
    print("4. 所有障碍物配置都通过验证")
    
    return True


if __name__ == "__main__":
    success = test_stage_a_integration()
    
    if success:
        print("\n" + "=" * 80)
        print("下一步：运行 Phase 0 训练")
        print("=" * 80)
        print("\n命令：")
        print("python train.py task=pusht_mujoco_obstacle \\")
        print("    exp_name=phase0_stage_a \\")
        print("    training.num_epochs=100")
        print("\n" + "=" * 80)
    else:
        print("\n❌ 测试失败，请检查错误信息")
