"""
完整的 LLM 障碍物生成 + 质量评价 + 课程学习训练脚本

核心流程:
1. LLM 生成障碍物配置
2. 质量评价器评估配置质量
3. 如果质量不达标，重新生成
4. 执行训练并收集反馈
5. 根据反馈调整难度（课程学习）
6. 将反馈传递给 LLM 进行进化
"""

import numpy as np
import os
from typing import Dict, List
from DIVO.env.pusht.mujoco.pusht_mj_rod_llm import PushT_mj_rod_LLM
from DIVO.env.pusht.llm_obstacle_generator_v3 import LLMObstacleGeneratorV3
from DIVO.env.pusht.obstacle_quality_evaluator import ObstacleQualityEvaluator
from DIVO.env.pusht.curriculum_manager import CurriculumManager


class LLMTrainingManager:
    """
    LLM 训练管理器
    
    整合了:
    - LLM 障碍物生成器
    - 质量评价器
    - 课程学习管理器
    - 训练反馈循环
    """
    
    def __init__(self,
                 env: PushT_mj_rod_LLM,
                 llm_generator: LLMObstacleGeneratorV3,
                 quality_threshold: float = 0.5,
                 max_regenerate_attempts: int = 3):
        """
        Args:
            env: Push-T 环境
            llm_generator: LLM 障碍物生成器
            quality_threshold: 质量阈值（低于此值会重新生成）
            max_regenerate_attempts: 最大重新生成次数
        """
        self.env = env
        self.llm_generator = llm_generator
        self.quality_evaluator = ObstacleQualityEvaluator()
        self.curriculum_manager = CurriculumManager()
        
        self.quality_threshold = quality_threshold
        self.max_regenerate_attempts = max_regenerate_attempts
        
        # 统计信息
        self.episode_count = 0
        self.generation_count = 0
        self.rejected_count = 0
        
        print("✓ LLMTrainingManager 初始化完成")
    
    def generate_and_validate_config(self, 
                                     tblock_pose: List[float],
                                     use_evolution: bool = False,
                                     previous_config: List[Dict] = None,
                                     policy_stats: Dict = None) -> tuple:
        """
        生成并验证障碍物配置
        
        Returns:
            (config, quality_score, detailed_scores, feedback)
        """
        for attempt in range(self.max_regenerate_attempts):
            # Step 1: LLM 生成配置
            if use_evolution and previous_config is not None and policy_stats is not None:
                print(f"\n🔄 进化生成 (attempt {attempt + 1}/{self.max_regenerate_attempts})")
                config = self.llm_generator.evolve(
                    tblock_pose=tblock_pose,
                    previous_config=previous_config,
                    policy_stats=policy_stats,
                    num_obstacles=self.curriculum_manager.get_obstacle_num()
                )
            else:
                print(f"\n🎲 初始生成 (attempt {attempt + 1}/{self.max_regenerate_attempts})")
                difficulty = self.curriculum_manager.get_difficulty_level()
                config = self.llm_generator.generate(
                    tblock_pose=tblock_pose,
                    scenario_type="auto",
                    num_obstacles=self.curriculum_manager.get_obstacle_num(),
                    difficulty=difficulty
                )
            
            self.generation_count += 1
            
            # Step 2: 质量评价
            quality_score, detailed_scores, feedback = self.quality_evaluator.evaluate_obstacle_quality(
                obstacles=config,
                tblock_pose=tblock_pose
            )
            
            print(f"\n📊 质量评分: {quality_score:.3f}")
            print(f"详细评分:")
            print(f"  - 可解性: {detailed_scores['solvability']:.2f}")
            print(f"  - 难度: {detailed_scores['difficulty']:.2f}")
            print(f"  - 多样性: {detailed_scores['diversity']:.2f}")
            print(f"  - 有效性: {detailed_scores['effectiveness']:.2f}")
            print(f"\n{feedback}")
            
            # Step 3: 检查质量
            if quality_score >= self.quality_threshold:
                print(f"✓ 配置通过质量检查")
                self.quality_evaluator.add_to_history(config)
                return config, quality_score, detailed_scores, feedback
            else:
                print(f"✗ 配置质量不达标 ({quality_score:.3f} < {self.quality_threshold})")
                self.rejected_count += 1
        
        # 达到最大尝试次数，使用最后一次生成的配置
        print(f"⚠ 达到最大重试次数，使用最后生成的配置")
        self.quality_evaluator.add_to_history(config)
        return config, quality_score, detailed_scores, feedback
    
    def run_episode(self, policy, tblock_pose: List[float] = None) -> Dict:
        """
        运行一个 episode
        
        Args:
            policy: 策略函数 (obs) -> action
            tblock_pose: 指定 T-block 初始位置（可选）
        
        Returns:
            episode_info: 包含 success, steps, collision, reward, tblock_pose, config
        """
        # 重置环境
        obs = self.env.reset(tblock_pos=tblock_pose)
        
        # 获取实际的 T-block 位置
        actual_tblock_pose = [
            obs[0, 0] * self.env.task._desk_size,
            obs[0, 1] * self.env.task._desk_size,
            np.arctan2(obs[0, 3], obs[0, 2])
        ]
        
        # 生成并验证障碍物配置
        use_evolution = (self.episode_count > 0 and 
                        self.episode_count % 10 == 0 and
                        hasattr(self, 'last_config'))
        
        if use_evolution:
            config, quality_score, detailed_scores, feedback = self.generate_and_validate_config(
                tblock_pose=actual_tblock_pose,
                use_evolution=True,
                previous_config=self.last_config,
                policy_stats=self.curriculum_manager.get_stats()
            )
        else:
            config, quality_score, detailed_scores, feedback = self.generate_and_validate_config(
                tblock_pose=actual_tblock_pose,
                use_evolution=False
            )
        
        # 应用配置到环境
        self.env.set_obstacle_config(config)
        obs = self.env.reset(tblock_pos=tblock_pose)
        
        # 执行 episode
        done = False
        steps = 0
        total_reward = 0
        collision = False
        
        while not done and steps < 100:
            action = policy(obs)
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            steps += 1
            
            if reward == -10:  # 碰撞或掉落
                collision = True
                break
        
        success = info.get('success', False)
        
        # 记录到课程学习管理器
        self.curriculum_manager.record_episode(
            success=success,
            steps=steps,
            collision=collision,
            reward=total_reward
        )
        
        # 保存配置用于进化
        self.last_config = config
        
        self.episode_count += 1
        
        return {
            'success': success,
            'steps': steps,
            'collision': collision,
            'reward': total_reward,
            'tblock_pose': actual_tblock_pose,
            'config': config,
            'quality_score': quality_score,
            'detailed_scores': detailed_scores
        }
    
    def check_and_update_difficulty(self, check_interval: int = 20):
        """
        检查并更新难度（课程学习）
        
        Args:
            check_interval: 每隔多少个 episode 检查一次
        """
        if self.episode_count % check_interval == 0 and self.episode_count > 0:
            print(f"\n{'='*60}")
            print(f"📈 课程学习检查 (Episode {self.episode_count})")
            print(f"{'='*60}")
            
            stats = self.curriculum_manager.get_stats()
            print(f"最近 {check_interval} 个 episode 统计:")
            print(f"  - 成功率: {stats['success_rate']*100:.1f}%")
            print(f"  - 平均步数: {stats['avg_steps']:.1f}")
            print(f"  - 碰撞率: {stats['collision_rate']*100:.1f}%")
            print(f"  - 平均奖励: {stats['avg_reward']:.2f}")
            
            # 更新难度
            old_difficulty = self.curriculum_manager.get_difficulty_level()
            action = self.curriculum_manager.update_difficulty()
            new_difficulty = self.curriculum_manager.get_difficulty_level()
            
            if action == "upgrade":
                print(f"⬆ 难度升级: {old_difficulty} → {new_difficulty}")
            elif action == "downgrade":
                print(f"⬇ 难度降级: {old_difficulty} → {new_difficulty}")
            else:
                print(f"➡ 难度保持: {new_difficulty}")
            
            print(f"{'='*60}\n")
    
    def get_statistics(self) -> Dict:
        """获取训练统计信息"""
        return {
            'episode_count': self.episode_count,
            'generation_count': self.generation_count,
            'rejected_count': self.rejected_count,
            'rejection_rate': self.rejected_count / max(self.generation_count, 1),
            'curriculum_stats': self.curriculum_manager.get_stats(),
            'current_difficulty': self.curriculum_manager.get_difficulty_level()
        }


# ============================================================================
# 使用示例
# ============================================================================

def example_random_policy(obs):
    """示例：随机策略"""
    return np.random.uniform(-1, 1, size=(2,))


def main():
    """主函数：演示完整流程"""
    
    # 1. 初始化环境
    print("初始化环境...")
    env = PushT_mj_rod_LLM(
        obstacle=True,
        obstacle_num=2,
        obstacle_size=0.01,
        obstacle_shape='box',
        eval=False
    )
    
    # 2. 初始化 LLM 生成器
    print("初始化 LLM 生成器...")
    llm_generator = LLMObstacleGeneratorV3(
        api_type="deepseek",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.7
    )
    
    # 3. 初始化训练管理器
    print("初始化训练管理器...")
    manager = LLMTrainingManager(
        env=env,
        llm_generator=llm_generator,
        quality_threshold=0.5,
        max_regenerate_attempts=3
    )
    
    # 4. 运行训练循环
    print("\n开始训练...\n")
    num_episodes = 100
    
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}")
        
        # 运行一个 episode
        episode_info = manager.run_episode(policy=example_random_policy)
        
        # 打印结果
        print(f"\n结果:")
        print(f"  - 成功: {'✓' if episode_info['success'] else '✗'}")
        print(f"  - 步数: {episode_info['steps']}")
        print(f"  - 碰撞: {'是' if episode_info['collision'] else '否'}")
        print(f"  - 奖励: {episode_info['reward']:.2f}")
        print(f"  - 配置质量: {episode_info['quality_score']:.3f}")
        
        # 检查并更新难度
        manager.check_and_update_difficulty(check_interval=20)
    
    # 5. 打印最终统计
    print(f"\n{'='*60}")
    print("训练完成！最终统计:")
    print(f"{'='*60}")
    stats = manager.get_statistics()
    print(f"总 episode 数: {stats['episode_count']}")
    print(f"总生成次数: {stats['generation_count']}")
    print(f"拒绝次数: {stats['rejected_count']}")
    print(f"拒绝率: {stats['rejection_rate']*100:.1f}%")
    print(f"当前难度: {stats['current_difficulty']}")
    print(f"\n最终性能:")
    print(f"  - 成功率: {stats['curriculum_stats']['success_rate']*100:.1f}%")
    print(f"  - 平均步数: {stats['curriculum_stats']['avg_steps']:.1f}")
    print(f"  - 碰撞率: {stats['curriculum_stats']['collision_rate']*100:.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
