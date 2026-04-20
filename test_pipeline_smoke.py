"""
管线冒烟测试：验证 Curriculum Training Pipeline 端到端不 crash
不需要 wandb，不需要长时间训练，只验证：
1. LLM 环境创建成功
2. 课程管理器初始化成功
3. 环境 reset/step 正常
4. 观测空间维度正确
5. set_obstacle_config 工作
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np


def test_env_creation():
    """测试 1: LLM 环境能否正常创建"""
    print("=" * 60)
    print("测试 1: LLM 环境创建")
    print("=" * 60)
    
    from DIVO.env.pusht import get_pusht_env
    
    env_args = {
        '_target_': 'pusht_mujoco_llm',
        'obstacle': True,
        'obstacle_num': 2,
        'obstacle_size': 0.05,
        'obstacle_shape': 'box',
        'obstacle_dist': 'random',
        'action_scale': 4,
        'NUM_SUBSTEPS': 25,
        'action_dim': [6],
        'obs_dim': [8],
        'action_reg': True,
        'reg_coeff': 1.0,
        'dynamics_randomization': True,
    }
    
    env = get_pusht_env(**env_args)
    print(f"  ✓ 环境创建成功: {type(env).__name__}")
    
    action_dim, obs_dim = env.get_info()
    print(f"  ✓ action_dim={action_dim}, obs_dim={obs_dim}")
    
    return env


def test_env_reset_step(env):
    """测试 2: 环境 reset/step 正常"""
    print("\n" + "=" * 60)
    print("测试 2: 环境 reset/step")
    print("=" * 60)
    
    obs = env.reset()
    print(f"  ✓ reset() 成功, obs shape={obs.shape}, obs={obs[0, :4]}")
    
    # 随机 action
    action = np.random.uniform(-1, 1, size=(6,))
    next_obs, reward, done, info = env.step(action)
    print(f"  ✓ step() 成功, reward={reward:.4f}, done={done}")
    
    return obs


def test_set_obstacle_config(env):
    """测试 3: set_obstacle_config 接口"""
    print("\n" + "=" * 60)
    print("测试 3: set_obstacle_config 接口")
    print("=" * 60)
    
    llm_config = [{'x': 0.1, 'y': 0.1}, {'x': -0.1, 'y': -0.05}]
    env.set_obstacle_config(llm_config)
    print(f"  ✓ set_obstacle_config() 成功")
    
    obs = env.reset()
    print(f"  ✓ reset() with LLM config 成功, obs shape={obs.shape}")
    
    # 验证障碍物位置
    positions = env.get_obstacle_positions()
    print(f"  ✓ 障碍物位置: {positions}")
    
    env.clear_obstacle_config()
    print(f"  ✓ clear_obstacle_config() 成功")


def test_curriculum_manager():
    """测试 4: 课程管理器"""
    print("\n" + "=" * 60)
    print("测试 4: 课程管理器（无 LLM）")
    print("=" * 60)
    
    from DIVO.env.pusht.llm_curriculum import (
        LLMCurriculumManager,
        AdaptiveLLMCurriculum,
        EpisodeRecord
    )
    
    # 不使用 LLM，只用 rule-based
    curriculum = AdaptiveLLMCurriculum(llm_generator=None, use_llm=False)
    print(f"  ✓ AdaptiveLLMCurriculum 初始化成功")
    print(f"    Stage: {curriculum.state.current_stage}, Difficulty: {curriculum.state.difficulty}")
    
    # 获取障碍物配置（Stage 1 应返回空）
    tblock_pose = [0.15, 0.12, 2.356]
    config, scenario = curriculum.get_obstacle_config(tblock_pose)
    print(f"  ✓ get_obstacle_config(): config={config}, scenario={scenario}")
    
    # 模拟记录几个 episode
    for i in range(5):
        record = EpisodeRecord(
            tblock_pose=tblock_pose,
            obstacle_config=config,
            success=(i > 2),
            reward=float(i),
            steps=10,
            collision=False,
            difficulty=curriculum.state.difficulty,
            scenario_type=scenario
        )
        curriculum.record_episode(record)
    
    print(f"  ✓ record_episode() x5 成功")
    print(f"    Stage: {curriculum.state.current_stage}, "
          f"Success Rate: {curriculum.state.success_rate:.2f}, "
          f"Episodes: {curriculum.state.episodes_in_stage}")
    
    return curriculum


def test_curriculum_with_llm():
    """测试 5: 课程管理器 + LLM 生成器"""
    print("\n" + "=" * 60)
    print("测试 5: 课程管理器 + LLM 生成器")
    print("=" * 60)
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("  ⚠️  DEEPSEEK_API_KEY 未设置，跳过 LLM 测试")
        return None
    
    from DIVO.env.pusht.llm_obstacle_generator_v3 import LLMObstacleGeneratorV3
    from DIVO.env.pusht.llm_curriculum import AdaptiveLLMCurriculum
    
    gen = LLMObstacleGeneratorV3(
        api_type="deepseek",
        api_key=api_key,
        model="deepseek-chat"
    )
    print(f"  ✓ LLMObstacleGeneratorV3 初始化成功")
    
    curriculum = AdaptiveLLMCurriculum(llm_generator=gen, use_llm=True)
    print(f"  ✓ AdaptiveLLMCurriculum (with LLM) 初始化成功")
    
    return curriculum


def test_mini_training_loop(env):
    """测试 6: 迷你训练循环"""
    print("\n" + "=" * 60)
    print("测试 6: 迷你训练循环 (3 episodes)")
    print("=" * 60)
    
    from DIVO.env.pusht.llm_curriculum import (
        AdaptiveLLMCurriculum,
        EpisodeRecord
    )
    
    curriculum = AdaptiveLLMCurriculum(llm_generator=None, use_llm=False)
    
    for ep in range(3):
        # 获取课程配置
        tblock_pose = [np.random.uniform(-0.18, 0.18), 
                       np.random.uniform(-0.18, 0.18),
                       np.random.uniform(0, 2*np.pi)]
        config, scenario = curriculum.get_obstacle_config(tblock_pose)
        
        # 设置障碍物
        if config:
            env.set_obstacle_config(config)
        else:
            env.clear_obstacle_config()
        
        # Reset & step
        obs = env.reset()
        episode_reward = 0
        
        for step in range(10):  # max_steps = 10
            action = np.random.uniform(-1, 1, size=(6,))
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            obs = next_obs
            if done:
                break
        
        # 记录
        record = EpisodeRecord(
            tblock_pose=tblock_pose,
            obstacle_config=config,
            success=False,
            reward=episode_reward,
            steps=step + 1,
            collision=False,
            difficulty=curriculum.state.difficulty,
            scenario_type=scenario
        )
        curriculum.record_episode(record)
        
        print(f"  ✓ Episode {ep+1}: reward={episode_reward:.2f}, "
              f"steps={step+1}, stage={curriculum.state.current_stage}")
    
    print(f"\n  ✓ 迷你训练循环完成！")


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   ACGS Pipeline Smoke Test                              ║")
    print("╚══════════════════════════════════════════════════════════╝\n")
    
    try:
        # 1. 环境创建
        env = test_env_creation()
        
        # 2. Reset/Step
        test_env_reset_step(env)
        
        # 3. LLM 配置接口
        test_set_obstacle_config(env)
        
        # 4. 课程管理器（rule-based）
        test_curriculum_manager()
        
        # 5. LLM 生成器（如果 API key 存在）
        test_curriculum_with_llm()
        
        # 6. 迷你训练循环
        test_mini_training_loop(env)
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED — Pipeline is ready!")
        print("=" * 60)
        
    except Exception as e:
        import traceback
        print(f"\n❌ TEST FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
