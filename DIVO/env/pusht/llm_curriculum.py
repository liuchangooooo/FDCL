"""
LLM 驱动的课程学习框架

核心思想：
1. LLM 根据策略当前能力动态调整环境难度
2. 分析失败案例，生成针对性训练场景
3. 逐步从简单到复杂，提高训练效率

课程阶段：
- Stage 1: 无障碍物，学习基本推动
- Stage 2: 单障碍物，远离路径
- Stage 3: 单障碍物，在路径上
- Stage 4: 多障碍物，形成约束
- Stage 5: 复杂场景，需要规划
"""
import numpy as np
import json
from typing import List, Dict, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field

from DIVO.env.pusht.llm_obstacle_generator_v3 import LLMObstacleGeneratorV3
from DIVO.env.pusht.llm_obstacle_generator import RuleBasedObstacleGenerator


@dataclass
class EpisodeRecord:
    """单个 episode 的记录"""
    tblock_pose: List[float]        # T-block 初始位置
    obstacle_config: List[Dict]     # 障碍物配置
    success: bool                   # 是否成功
    reward: float                   # 总奖励
    steps: int                      # 步数
    collision: bool                 # 是否碰撞
    difficulty: float               # 难度评分
    scenario_type: str              # 场景类型


@dataclass 
class CurriculumState:
    """课程学习状态"""
    current_stage: int = 1          # 当前阶段 (1-5)
    episodes_in_stage: int = 0      # 当前阶段的 episode 数
    success_rate: float = 0.0       # 当前阶段成功率
    avg_reward: float = 0.0         # 平均奖励
    difficulty: float = 0.0         # 当前难度 (0-1)
    
    # 晋级/降级阈值
    promotion_threshold: float = 0.7    # 成功率超过此值则晋级
    demotion_threshold: float = 0.3     # 成功率低于此值则降级
    min_episodes_per_stage: int = 50    # 每阶段最少 episode 数


class LLMCurriculumManager:
    """
    LLM 驱动的课程学习管理器
    
    功能：
    1. 跟踪训练进度和策略能力
    2. 动态调整环境难度
    3. 分析失败案例，生成针对性场景
    4. 提供课程学习的完整接口
    """
    
    def __init__(self,
                 llm_generator: LLMObstacleGeneratorV3 = None,
                 use_llm: bool = True,
                 history_size: int = 100):
        """
        初始化课程管理器
        
        Args:
            llm_generator: LLM 生成器实例
            use_llm: 是否使用 LLM（False 则使用规则生成器）
            history_size: 历史记录大小
        """
        self.use_llm = use_llm
        
        if use_llm and llm_generator is not None:
            self.generator = llm_generator
        else:
            self.generator = RuleBasedObstacleGenerator()
            self.use_llm = False
        
        # 课程状态
        self.state = CurriculumState()
        
        # 历史记录
        self.history = deque(maxlen=history_size)
        self.stage_history = deque(maxlen=history_size)  # 当前阶段的历史
        
        # 失败案例分析
        self.failure_cases = deque(maxlen=50)
        
        # 阶段配置
        self.stage_configs = self._init_stage_configs()
    
    def _init_stage_configs(self) -> Dict:
        """初始化各阶段配置"""
        return {
            1: {
                "name": "基础推动",
                "description": "无障碍物，学习基本推动技能",
                "num_obstacles": 0,
                "difficulty_range": (0.0, 0.2),
                "scenario_types": ["none"]
            },
            2: {
                "name": "简单避障",
                "description": "单障碍物，远离最优路径",
                "num_obstacles": 1,
                "difficulty_range": (0.2, 0.4),
                "scenario_types": ["random"]
            },
            3: {
                "name": "路径规划",
                "description": "单障碍物，阻挡直线路径",
                "num_obstacles": 1,
                "difficulty_range": (0.4, 0.6),
                "scenario_types": ["path_blocking"]
            },
            4: {
                "name": "约束导航",
                "description": "多障碍物，形成通道约束",
                "num_obstacles": 2,
                "difficulty_range": (0.6, 0.8),
                "scenario_types": ["corridor", "path_blocking"]
            },
            5: {
                "name": "复杂规划",
                "description": "复杂场景，需要多步规划",
                "num_obstacles": 2,
                "difficulty_range": (0.8, 1.0),
                "scenario_types": ["rotation", "two_step", "corridor"]
            }
        }
    
    def get_obstacle_config(self, tblock_pose: List[float] = None) -> Tuple[List[Dict], str]:
        """
        获取当前课程阶段的障碍物配置
        
        Args:
            tblock_pose: T-block 初始位置 [x, y, theta]
        
        Returns:
            (obstacle_config, scenario_type)
        """
        stage_config = self.stage_configs[self.state.current_stage]
        
        # 阶段 1：无障碍物
        if stage_config["num_obstacles"] == 0:
            return [], "none"
        
        # 随机选择场景类型
        scenario_type = np.random.choice(stage_config["scenario_types"])
        
        # 计算当前难度
        diff_low, diff_high = stage_config["difficulty_range"]
        # 根据阶段内进度调整难度
        progress = min(self.state.episodes_in_stage / 100, 1.0)
        current_difficulty = diff_low + (diff_high - diff_low) * progress
        
        # 生成配置
        if self.use_llm:
            config = self._generate_with_llm(
                tblock_pose, 
                scenario_type, 
                stage_config["num_obstacles"],
                current_difficulty
            )
        else:
            config = self._generate_with_rules(
                tblock_pose,
                scenario_type,
                stage_config["num_obstacles"]
            )
        
        self.state.difficulty = current_difficulty
        return config, scenario_type
    
    def _generate_with_llm(self, tblock_pose, scenario_type, num_obstacles, difficulty) -> List[Dict]:
        """使用 LLM V3 生成配置"""
        difficulty_str = "easy" if difficulty < 0.4 else ("medium" if difficulty < 0.7 else "hard")
        
        return self.generator.generate(
            tblock_pose=tblock_pose,
            num_obstacles=num_obstacles,
            difficulty=difficulty_str
        )
    
    def _generate_with_rules(self, tblock_pose, scenario_type, num_obstacles) -> List[Dict]:
        """使用规则生成配置"""
        strategy_map = {
            "none": "random",
            "random": "random",
            "path_blocking": "midpoint",
            "corridor": "perpendicular",
            "rotation": "midpoint",
            "two_step": "midpoint"
        }
        strategy = strategy_map.get(scenario_type, "random")
        
        return self.generator.generate(
            tblock_pose=tblock_pose,
            strategy=strategy,
            num_obstacles=num_obstacles
        )
    
    def record_episode(self, record: EpisodeRecord):
        """
        记录 episode 结果并更新课程状态
        
        Args:
            record: episode 记录
        """
        self.history.append(record)
        self.stage_history.append(record)
        self.state.episodes_in_stage += 1
        
        # 记录失败案例
        if not record.success:
            self.failure_cases.append(record)
        
        # 更新统计
        self._update_statistics()
        
        # 检查是否需要调整阶段
        self._check_stage_transition()
    
    def _update_statistics(self):
        """更新统计数据"""
        if len(self.stage_history) == 0:
            return
        
        recent = list(self.stage_history)[-50:]  # 最近 50 个 episode
        
        self.state.success_rate = sum(1 for r in recent if r.success) / len(recent)
        self.state.avg_reward = sum(r.reward for r in recent) / len(recent)
    
    def _check_stage_transition(self):
        """检查是否需要阶段转换"""
        # 至少完成最小 episode 数
        if self.state.episodes_in_stage < self.state.min_episodes_per_stage:
            return
        
        # 晋级条件
        if self.state.success_rate >= self.state.promotion_threshold:
            if self.state.current_stage < 5:
                self._promote()
        
        # 降级条件
        elif self.state.success_rate <= self.state.demotion_threshold:
            if self.state.current_stage > 1:
                self._demote()
    
    def _promote(self):
        """晋级到下一阶段"""
        old_stage = self.state.current_stage
        self.state.current_stage += 1
        self.state.episodes_in_stage = 0
        self.stage_history.clear()
        
        print(f"\n🎉 课程晋级: Stage {old_stage} → Stage {self.state.current_stage}")
        print(f"   {self.stage_configs[self.state.current_stage]['name']}: "
              f"{self.stage_configs[self.state.current_stage]['description']}")
    
    def _demote(self):
        """降级到上一阶段"""
        old_stage = self.state.current_stage
        self.state.current_stage -= 1
        self.state.episodes_in_stage = 0
        self.stage_history.clear()
        
        print(f"\n⚠️ 课程降级: Stage {old_stage} → Stage {self.state.current_stage}")
        print(f"   需要巩固: {self.stage_configs[self.state.current_stage]['name']}")
    
    def _analyze_failures(self) -> str:
        """分析失败案例，生成分析报告"""
        if len(self.failure_cases) == 0:
            return "暂无失败案例"
        
        recent_failures = list(self.failure_cases)[-10:]
        
        # 统计失败原因
        collision_count = sum(1 for f in recent_failures if f.collision)
        timeout_count = len(recent_failures) - collision_count
        
        # 统计失败场景
        scenario_counts = {}
        for f in recent_failures:
            scenario_counts[f.scenario_type] = scenario_counts.get(f.scenario_type, 0) + 1
        
        analysis = f"""
失败分析 (最近 {len(recent_failures)} 次):
- 碰撞失败: {collision_count} 次
- 超时失败: {timeout_count} 次
- 主要失败场景: {max(scenario_counts, key=scenario_counts.get) if scenario_counts else 'N/A'}
"""
        return analysis
    
    def get_status(self) -> Dict:
        """获取当前课程状态"""
        stage_config = self.stage_configs[self.state.current_stage]
        return {
            "stage": self.state.current_stage,
            "stage_name": stage_config["name"],
            "description": stage_config["description"],
            "episodes_in_stage": self.state.episodes_in_stage,
            "success_rate": f"{self.state.success_rate:.1%}",
            "avg_reward": f"{self.state.avg_reward:.2f}",
            "difficulty": f"{self.state.difficulty:.2f}",
            "total_episodes": len(self.history)
        }
    
    def print_status(self):
        """打印当前状态"""
        status = self.get_status()
        print(f"\n{'='*50}")
        print(f"📚 课程学习状态")
        print(f"{'='*50}")
        print(f"阶段: {status['stage']}/5 - {status['stage_name']}")
        print(f"描述: {status['description']}")
        print(f"阶段进度: {status['episodes_in_stage']} episodes")
        print(f"成功率: {status['success_rate']}")
        print(f"平均奖励: {status['avg_reward']}")
        print(f"当前难度: {status['difficulty']}")
        print(f"总训练量: {status['total_episodes']} episodes")
        print(f"{'='*50}\n")


class AdaptiveLLMCurriculum(LLMCurriculumManager):
    """
    自适应 LLM 课程学习
    
    增强功能：
    1. 根据失败模式动态调整场景生成
    2. 针对性训练薄弱环节
    3. 智能难度调节
    """
    
    def __init__(self, llm_generator: LLMObstacleGeneratorV3 = None, **kwargs):
        super().__init__(llm_generator, **kwargs)
        
        # 能力评估
        self.skill_scores = {
            "basic_push": 0.0,      # 基本推动
            "obstacle_avoidance": 0.0,  # 避障
            "path_planning": 0.0,   # 路径规划
            "precise_control": 0.0, # 精确控制
            "multi_step": 0.0       # 多步规划
        }
        
        # 场景类型到技能的映射
        self.scenario_skill_map = {
            "none": ["basic_push"],
            "random": ["basic_push", "obstacle_avoidance"],
            "path_blocking": ["path_planning", "obstacle_avoidance"],
            "corridor": ["precise_control", "path_planning"],
            "rotation": ["precise_control", "multi_step"],
            "two_step": ["multi_step", "path_planning"]
        }
    
    def record_episode(self, record: EpisodeRecord):
        """记录 episode 并更新技能评估"""
        super().record_episode(record)
        self._update_skill_scores(record)
    
    def _update_skill_scores(self, record: EpisodeRecord):
        """更新技能评分"""
        skills = self.scenario_skill_map.get(record.scenario_type, [])
        
        for skill in skills:
            old_score = self.skill_scores[skill]
            # 指数移动平均更新
            alpha = 0.1
            new_value = 1.0 if record.success else 0.0
            self.skill_scores[skill] = (1 - alpha) * old_score + alpha * new_value
    
    def get_weakness(self) -> str:
        """获取最薄弱的技能"""
        return min(self.skill_scores, key=self.skill_scores.get)
    
    def get_targeted_scenario(self) -> str:
        """获取针对薄弱技能的场景类型"""
        weakness = self.get_weakness()
        
        # 反向映射：技能 -> 场景
        skill_scenario_map = {
            "basic_push": "none",
            "obstacle_avoidance": "random",
            "path_planning": "path_blocking",
            "precise_control": "corridor",
            "multi_step": "two_step"
        }
        
        return skill_scenario_map.get(weakness, "auto")
    
    def get_obstacle_config(self, tblock_pose: List[float] = None) -> Tuple[List[Dict], str]:
        """
        获取障碍物配置（带自适应调整）
        
        有 30% 概率针对薄弱技能训练
        """
        # 30% 概率针对性训练
        if np.random.random() < 0.3 and self.state.current_stage >= 2:
            scenario_type = self.get_targeted_scenario()
            stage_config = self.stage_configs[self.state.current_stage]
            
            if self.use_llm:
                config = self._generate_with_llm(
                    tblock_pose,
                    scenario_type,
                    stage_config["num_obstacles"],
                    self.state.difficulty
                )
            else:
                config = self._generate_with_rules(
                    tblock_pose,
                    scenario_type,
                    stage_config["num_obstacles"]
                )
            
            return config, scenario_type
        
        # 70% 正常课程
        return super().get_obstacle_config(tblock_pose)
    
    def get_skill_report(self) -> str:
        """获取技能评估报告"""
        report = "\n📊 技能评估:\n"
        for skill, score in self.skill_scores.items():
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            report += f"  {skill:20s} [{bar}] {score:.1%}\n"
        report += f"\n  薄弱环节: {self.get_weakness()}\n"
        return report
