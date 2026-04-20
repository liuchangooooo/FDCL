"""
使用 LLM 课程学习的训练示例

运行方法:
cd ~/DIVO && python -m DIVO.env.pusht.mujoco.train_with_curriculum
"""
import numpy as np
from typing import Dict

# 导入环境和课程管理器
from DIVO.env.pusht.mujoco.pusht_mj_rod_llm import PushT_mj_rod_LLM
from DIVO.env.pusht.llm_curriculum import (
    LLMCurriculumManager, 
    AdaptiveLLMCurriculum,
    EpisodeRecord
)
from DIVO.env.pusht.llm_obstacle_generator import LLMObstacleGenerator


def train_with_curriculum(
    num_episodes: int = 500,
    use_llm: bool = False,
    use_adaptive: bool = True,
    print_interval: int = 50
):
    """
    使用课程学习训练
    
    Args:
        num_episodes: 总训练 episode 数
        use_llm: 是否使用真实 LLM（需要 API Key）
        use_adaptive: 是否使用自适应课程
        print_interval: 打印间隔
    """
    print("=" * 60)
    print("🚀 LLM 驱动的课程学习训练")
    print("=" * 60)
    
    # 1. 创建环境
    env = PushT_mj_rod_LLM(
        obstacle=True,
        obstacle_num=2,
        obstacle_size=0.05,
        action_scale=4
    )
    print(f"✓ 环境创建完成")
    
    # 2. 创建 LLM 生成器（可选）
    llm_generator = None
    if use_llm:
        try:
            import os
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ZHIPU_API_KEY")
            if api_key:
                llm_generator = LLMObstacleGenerator(
                    api_type="openai" if os.getenv("OPENAI_API_KEY") else "zhipu",
                    api_key=api_key
                )
                print(f"✓ LLM 生成器创建完成")
            else:
                print("⚠ 未找到 API Key，使用规则生成器")
        except Exception as e:
            print(f"⚠ LLM 初始化失败: {e}，使用规则生成器")
    
    # 3. 创建课程管理器
    if use_adaptive:
        curriculum = AdaptiveLLMCurriculum(
            llm_generator=llm_generator,
            use_llm=(llm_generator is not None)
        )
        print(f"✓ 自适应课程管理器创建完成")
    else:
        curriculum = LLMCurriculumManager(
            llm_generator=llm_generator,
            use_llm=(llm_generator is not None)
        )
        print(f"✓ 课程管理器创建完成")
    
    # 4. 训练循环
    print(f"\n开始训练 {num_episodes} episodes...\n")
    
    total_rewards = []
    success_count = 0
    
    for episode in range(num_episodes):
        # 4.1 随机 T-block 初始位置
        tblock_pose = _random_tblock_pose()
        
        # 4.2 从课程管理器获取障碍物配置
        obstacle_config, scenario_type = curriculum.get_obstacle_config(tblock_pose)
        
        # 4.3 设置环境
        if len(obstacle_config) > 0:
            env.set_obstacle_config(obstacle_config)
        else:
            env.clear_obstacle_config()
        
        # 4.4 重置环境
        obs = env.reset()
        
        # 4.5 运行 episode
        episode_reward = 0
        done = False
        steps = 0
        collision = False
        max_steps = 10
        
        while not done and steps < max_steps:
            # 这里使用随机策略，实际训练时替换为你的策略
            action = _dummy_policy(obs, env)
            
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if reward == -10:  # 碰撞或掉落
                collision = True
        
        success = info.get('success', False)
        if success:
            success_count += 1
        
        total_rewards.append(episode_reward)
        
        # 4.6 记录到课程管理器
        record = EpisodeRecord(
            tblock_pose=tblock_pose,
            obstacle_config=obstacle_config,
            success=success,
            reward=episode_reward,
            steps=steps,
            collision=collision,
            difficulty=curriculum.state.difficulty,
            scenario_type=scenario_type
        )
        curriculum.record_episode(record)
        
        # 4.7 定期打印状态
        if (episode + 1) % print_interval == 0:
            recent_rewards = total_rewards[-print_interval:]
            recent_success = sum(1 for r in list(curriculum.stage_history)[-print_interval:] if r.success)
            
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  最近 {print_interval} 轮: 平均奖励={np.mean(recent_rewards):.2f}, "
                  f"成功率={recent_success/print_interval:.1%}")
            curriculum.print_status()
            
            if use_adaptive and isinstance(curriculum, AdaptiveLLMCurriculum):
                print(curriculum.get_skill_report())
    
    # 5. 训练总结
    print("\n" + "=" * 60)
    print("📈 训练总结")
    print("=" * 60)
    print(f"总 episodes: {num_episodes}")
    print(f"总成功次数: {success_count} ({success_count/num_episodes:.1%})")
    print(f"平均奖励: {np.mean(total_rewards):.2f}")
    print(f"最终阶段: Stage {curriculum.state.current_stage}")
    
    if use_adaptive and isinstance(curriculum, AdaptiveLLMCurriculum):
        print(curriculum.get_skill_report())
    
    return curriculum, total_rewards


def _random_tblock_pose():
    """随机生成有效的 T-block 位置"""
    while True:
        x = np.random.uniform(-0.18, 0.18)
        y = np.random.uniform(-0.18, 0.18)
        theta = np.random.uniform(0, 2*np.pi)
        if abs(x) > 0.1 or abs(y) > 0.1:
            return [x, y, theta]


def _dummy_policy(obs, env):
    """
    简单的演示策略（实际使用时替换为训练的策略）
    
    这里使用一个简单的启发式：朝目标方向移动
    """
    # 解析观测
    tblock_x = obs[0, 0] * env.task._desk_size
    tblock_y = obs[0, 1] * env.task._desk_size
    
    # 目标在原点
    target_x, target_y = 0, 0
    
    # 计算方向
    dx = target_x - tblock_x
    dy = target_y - tblock_y
    
    # 归一化并添加噪声
    dist = np.sqrt(dx**2 + dy**2) + 1e-6
    action = np.array([dx/dist, dy/dist])
    action += np.random.normal(0, 0.3, 2)
    action = np.clip(action, -1, 1)
    
    return action.reshape(1, 1, 2)


def compare_with_without_curriculum():
    """
    对比实验：有/无课程学习
    """
    print("\n" + "=" * 60)
    print("🔬 对比实验: 课程学习 vs 随机训练")
    print("=" * 60)
    
    num_episodes = 200
    
    # 1. 使用课程学习
    print("\n--- 使用课程学习 ---")
    curriculum_manager, curriculum_rewards = train_with_curriculum(
        num_episodes=num_episodes,
        use_llm=False,
        use_adaptive=True,
        print_interval=100
    )
    
    # 2. 不使用课程学习（直接最高难度）
    print("\n--- 不使用课程学习（随机训练）---")
    
    env = PushT_mj_rod_LLM(obstacle=True, obstacle_num=2, obstacle_size=0.05)
    from DIVO.env.pusht.llm_obstacle_generator import RuleBasedObstacleGenerator
    generator = RuleBasedObstacleGenerator()
    
    random_rewards = []
    random_success = 0
    
    for episode in range(num_episodes):
        tblock_pose = _random_tblock_pose()
        config = generator.generate(tblock_pose, strategy="random", num_obstacles=2)
        env.set_obstacle_config(config)
        
        obs = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 10:
            action = _dummy_policy(obs, env)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
        
        if info.get('success', False):
            random_success += 1
        random_rewards.append(episode_reward)
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: 成功率={random_success/(episode+1):.1%}")
    
    # 3. 对比结果
    print("\n" + "=" * 60)
    print("📊 对比结果")
    print("=" * 60)
    print(f"{'指标':<20} {'课程学习':<15} {'随机训练':<15}")
    print("-" * 50)
    print(f"{'平均奖励':<20} {np.mean(curriculum_rewards):<15.2f} {np.mean(random_rewards):<15.2f}")
    print(f"{'最终成功率':<20} {curriculum_manager.state.success_rate:<15.1%} {random_success/num_episodes:<15.1%}")
    print(f"{'最终阶段':<20} {curriculum_manager.state.current_stage:<15} {'N/A':<15}")


if __name__ == "__main__":
    # 单独训练
    train_with_curriculum(
        num_episodes=300,
        use_llm=False,  # 设为 True 并配置 API Key 以使用真实 LLM
        use_adaptive=True,
        print_interval=50
    )
    
    # 对比实验（可选）
    # compare_with_without_curriculum()
