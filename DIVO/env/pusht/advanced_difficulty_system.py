"""
高级难度系统设计

核心思想：
1. 多级难度（5-7个级别）
2. 多维度评估（位置、数量、场景、初始状态）
3. 动态难度调整（根据多个指标）
4. 场景组合（混合多种挑战）
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# 难度维度定义
# ============================================================================

class PlacementStrategy(Enum):
    """障碍物放置策略"""
    PERIPHERAL = "peripheral"              # 外围干扰
    PATH_BLOCKING = "path_blocking"        # 路径阻挡
    CORRIDOR = "corridor"                  # 狭窄通道
    SURROUNDING = "surrounding"            # 包围目标
    ROTATION_REQUIRED = "rotation_required"  # 需要旋转
    MULTI_STEP = "multi_step"              # 多步规划
    TRAP = "trap"                          # 陷阱（死路）
    DYNAMIC_CONSTRAINT = "dynamic"         # 动态约束


class SkillType(Enum):
    """测试的技能类型"""
    BASIC_AVOIDANCE = "basic_avoidance"        # 基本避障
    PATH_PLANNING = "path_planning"            # 路径规划
    PRECISE_CONTROL = "precise_control"        # 精确控制
    ROTATION_CONTROL = "rotation_control"      # 旋转控制
    MULTI_STEP_REASONING = "multi_step"        # 多步推理
    SPATIAL_REASONING = "spatial_reasoning"    # 空间推理
    RECOVERY = "recovery"                      # 错误恢复
    EFFICIENCY = "efficiency"                  # 效率优化


class ScenarioComplexity(Enum):
    """场景复杂度"""
    SINGLE_CONSTRAINT = "single"      # 单一约束
    DUAL_CONSTRAINT = "dual"          # 双重约束
    MULTI_CONSTRAINT = "multi"        # 多重约束
    ADAPTIVE = "adaptive"             # 自适应（根据 T-block 位置）


# ============================================================================
# 高级难度配置
# ============================================================================

@dataclass
class AdvancedDifficultyConfig:
    """高级难度配置"""
    level: int  # 1-7
    name: str
    
    # 障碍物配置
    num_obstacles_range: Tuple[int, int]
    
    # 位置策略（可以组合多种）
    placement_strategies: Dict[PlacementStrategy, float]  # 策略 -> 权重
    
    # 空间约束
    spatial_constraints: Dict[str, float]
    
    # 测试技能
    target_skills: List[SkillType]
    
    # 场景复杂度
    complexity: ScenarioComplexity
    
    # 特殊要求
    special_requirements: Dict[str, any]
    
    # 升级/降级阈值
    upgrade_thresholds: Dict[str, float]
    downgrade_thresholds: Dict[str, float]
    
    # 描述
    description: str


# ============================================================================
# 7 级难度系统
# ============================================================================

ADVANCED_DIFFICULTY_LEVELS = {
    # ========== Level 1: 入门 ==========
    1: AdvancedDifficultyConfig(
        level=1,
        name="入门 (Beginner)",
        num_obstacles_range=(1, 1),
        
        placement_strategies={
            PlacementStrategy.PERIPHERAL: 1.0,  # 100% 外围
        },
        
        spatial_constraints={
            "min_distance_to_path": 0.10,      # 距离路径 10cm
            "min_distance_to_start": 0.12,
            "min_distance_to_target": 0.12,
            "min_passage_width": 0.18,         # 超宽通道
            "max_obstacle_density": 0.1,       # 低密度
        },
        
        target_skills=[SkillType.BASIC_AVOIDANCE],
        complexity=ScenarioComplexity.SINGLE_CONSTRAINT,
        
        special_requirements={
            "block_any_path": False,
            "allow_straight_push": True,
            "description": "障碍物在角落，完全不影响推动"
        },
        
        upgrade_thresholds={
            "success_rate": 0.90,
            "collision_rate": 0.05,
            "avg_steps": 120,
            "min_episodes": 30
        },
        downgrade_thresholds={
            "success_rate": 0.0,  # 不降级
            "collision_rate": 1.0
        },
        
        description="入门级：障碍物在外围，不影响任何路径，建立基本避障意识"
    ),
    
    # ========== Level 2: 简单 ==========
    2: AdvancedDifficultyConfig(
        level=2,
        name="简单 (Easy)",
        num_obstacles_range=(1, 2),
        
        placement_strategies={
            PlacementStrategy.PERIPHERAL: 0.7,
            PlacementStrategy.PATH_BLOCKING: 0.3,  # 30% 轻微阻挡
        },
        
        spatial_constraints={
            "min_distance_to_path": 0.08,
            "min_distance_to_start": 0.10,
            "min_distance_to_target": 0.10,
            "min_passage_width": 0.15,
            "max_obstacle_density": 0.2,
        },
        
        target_skills=[
            SkillType.BASIC_AVOIDANCE,
            SkillType.PATH_PLANNING
        ],
        complexity=ScenarioComplexity.SINGLE_CONSTRAINT,
        
        special_requirements={
            "block_direct_path_prob": 0.3,     # 30% 概率阻挡直线
            "alternative_paths": "multiple",    # 多条绕行路径
            "description": "偶尔阻挡直线路径，但有多条明显的绕行方案"
        },
        
        upgrade_thresholds={
            "success_rate": 0.85,
            "collision_rate": 0.08,
            "avg_steps": 140,
            "min_episodes": 50
        },
        downgrade_thresholds={
            "success_rate": 0.15,
            "collision_rate": 0.60
        },
        
        description="简单级：偶尔阻挡路径，测试基本路径规划"
    ),
    
    # ========== Level 3: 中等偏易 ==========
    3: AdvancedDifficultyConfig(
        level=3,
        name="中等偏易 (Medium-Easy)",
        num_obstacles_range=(1, 3),
        
        placement_strategies={
            PlacementStrategy.PATH_BLOCKING: 0.5,
            PlacementStrategy.CORRIDOR: 0.3,
            PlacementStrategy.PERIPHERAL: 0.2,
        },
        
        spatial_constraints={
            "min_distance_to_path": 0.05,
            "min_distance_to_start": 0.08,
            "min_distance_to_target": 0.08,
            "min_passage_width": 0.13,         # 13cm 通道
            "max_obstacle_density": 0.3,
        },
        
        target_skills=[
            SkillType.PATH_PLANNING,
            SkillType.PRECISE_CONTROL,
            SkillType.BASIC_AVOIDANCE
        ],
        complexity=ScenarioComplexity.DUAL_CONSTRAINT,
        
        special_requirements={
            "block_direct_path_prob": 0.7,
            "alternative_paths": "limited",     # 有限的绕行路径
            "corridor_width_range": (0.13, 0.15),
            "description": "经常阻挡路径，需要通过适中宽度的通道"
        },
        
        upgrade_thresholds={
            "success_rate": 0.80,
            "collision_rate": 0.12,
            "avg_steps": 160,
            "min_episodes": 80
        },
        downgrade_thresholds={
            "success_rate": 0.20,
            "collision_rate": 0.50
        },
        
        description="中等偏易：经常阻挡路径，通道宽度适中，测试路径规划和基本精确控制"
    ),
    
    # ========== Level 4: 中等 ==========
    4: AdvancedDifficultyConfig(
        level=4,
        name="中等 (Medium)",
        num_obstacles_range=(2, 3),
        
        placement_strategies={
            PlacementStrategy.PATH_BLOCKING: 0.3,
            PlacementStrategy.CORRIDOR: 0.3,
            PlacementStrategy.ROTATION_REQUIRED: 0.2,
            PlacementStrategy.SURROUNDING: 0.2,
        },
        
        spatial_constraints={
            "min_distance_to_path": 0.03,
            "min_distance_to_start": 0.07,
            "min_distance_to_target": 0.07,
            "min_passage_width": 0.12,         # 12cm 通道
            "max_obstacle_density": 0.4,
        },
        
        target_skills=[
            SkillType.PATH_PLANNING,
            SkillType.PRECISE_CONTROL,
            SkillType.ROTATION_CONTROL,
            SkillType.SPATIAL_REASONING
        ],
        complexity=ScenarioComplexity.DUAL_CONSTRAINT,
        
        special_requirements={
            "block_direct_path_prob": 0.85,
            "block_one_alternative_prob": 0.4,  # 40% 概率也挡一条绕行路径
            "rotation_required_prob": 0.3,      # 30% 需要旋转
            "corridor_width_range": (0.12, 0.13),
            "description": "多重约束，可能需要旋转或精确通过窄通道"
        },
        
        upgrade_thresholds={
            "success_rate": 0.75,
            "collision_rate": 0.15,
            "avg_steps": 180,
            "min_episodes": 100
        },
        downgrade_thresholds={
            "success_rate": 0.25,
            "collision_rate": 0.45
        },
        
        description="中等级：多重约束，需要路径规划、精确控制和旋转能力"
    ),
    
    # ========== Level 5: 中等偏难 ==========
    5: AdvancedDifficultyConfig(
        level=5,
        name="中等偏难 (Medium-Hard)",
        num_obstacles_range=(2, 4),
        
        placement_strategies={
            PlacementStrategy.CORRIDOR: 0.25,
            PlacementStrategy.ROTATION_REQUIRED: 0.25,
            PlacementStrategy.SURROUNDING: 0.25,
            PlacementStrategy.MULTI_STEP: 0.15,
            PlacementStrategy.PATH_BLOCKING: 0.10,
        },
        
        spatial_constraints={
            "min_distance_to_path": 0.02,
            "min_distance_to_start": 0.06,
            "min_distance_to_target": 0.06,
            "min_passage_width": 0.11,         # 11cm 窄通道
            "max_obstacle_density": 0.5,
        },
        
        target_skills=[
            SkillType.PRECISE_CONTROL,
            SkillType.ROTATION_CONTROL,
            SkillType.MULTI_STEP_REASONING,
            SkillType.SPATIAL_REASONING,
            SkillType.PATH_PLANNING
        ],
        complexity=ScenarioComplexity.MULTI_CONSTRAINT,
        
        special_requirements={
            "block_direct_path_prob": 0.95,
            "block_multiple_alternatives_prob": 0.5,
            "rotation_required_prob": 0.5,
            "multi_step_required_prob": 0.3,
            "corridor_width_range": (0.11, 0.12),
            "target_surrounded_prob": 0.3,      # 30% 目标被包围
            "description": "复杂约束，经常需要旋转或多步规划"
        },
        
        upgrade_thresholds={
            "success_rate": 0.70,
            "collision_rate": 0.20,
            "avg_steps": 200,
            "min_episodes": 120
        },
        downgrade_thresholds={
            "success_rate": 0.28,
            "collision_rate": 0.42
        },
        
        description="中等偏难：复杂空间约束，需要旋转、多步规划和精确控制"
    ),
    
    # ========== Level 6: 困难 ==========
    6: AdvancedDifficultyConfig(
        level=6,
        name="困难 (Hard)",
        num_obstacles_range=(3, 5),
        
        placement_strategies={
            PlacementStrategy.MULTI_STEP: 0.3,
            PlacementStrategy.SURROUNDING: 0.25,
            PlacementStrategy.ROTATION_REQUIRED: 0.2,
            PlacementStrategy.CORRIDOR: 0.15,
            PlacementStrategy.TRAP: 0.1,        # 10% 陷阱场景
        },
        
        spatial_constraints={
            "min_distance_to_path": 0.01,
            "min_distance_to_start": 0.05,
            "min_distance_to_target": 0.05,
            "min_passage_width": 0.105,        # 10.5cm 极窄通道
            "max_obstacle_density": 0.6,
        },
        
        target_skills=[
            SkillType.MULTI_STEP_REASONING,
            SkillType.PRECISE_CONTROL,
            SkillType.ROTATION_CONTROL,
            SkillType.SPATIAL_REASONING,
            SkillType.RECOVERY,
            SkillType.EFFICIENCY
        ],
        complexity=ScenarioComplexity.MULTI_CONSTRAINT,
        
        special_requirements={
            "block_direct_path_prob": 1.0,
            "block_multiple_alternatives_prob": 0.7,
            "rotation_required_prob": 0.6,
            "multi_step_required_prob": 0.5,
            "corridor_width_range": (0.105, 0.11),
            "target_surrounded_prob": 0.5,
            "trap_scenarios_prob": 0.1,         # 10% 有死路陷阱
            "description": "高度复杂约束，需要高级规划和精确执行"
        },
        
        upgrade_thresholds={
            "success_rate": 0.65,
            "collision_rate": 0.25,
            "avg_steps": 220,
            "min_episodes": 150
        },
        downgrade_thresholds={
            "success_rate": 0.30,
            "collision_rate": 0.40
        },
        
        description="困难级：高度复杂约束，需要多步规划、精确控制和错误恢复能力"
    ),
    
    # ========== Level 7: 专家 ==========
    7: AdvancedDifficultyConfig(
        level=7,
        name="专家 (Expert)",
        num_obstacles_range=(3, 6),
        
        placement_strategies={
            PlacementStrategy.DYNAMIC_CONSTRAINT: 0.3,  # 动态约束
            PlacementStrategy.MULTI_STEP: 0.25,
            PlacementStrategy.TRAP: 0.2,
            PlacementStrategy.SURROUNDING: 0.15,
            PlacementStrategy.ROTATION_REQUIRED: 0.1,
        },
        
        spatial_constraints={
            "min_distance_to_path": 0.01,
            "min_distance_to_start": 0.04,
            "min_distance_to_target": 0.04,
            "min_passage_width": 0.102,        # 10.2cm 极限通道
            "max_obstacle_density": 0.7,
        },
        
        target_skills=[
            SkillType.MULTI_STEP_REASONING,
            SkillType.PRECISE_CONTROL,
            SkillType.ROTATION_CONTROL,
            SkillType.SPATIAL_REASONING,
            SkillType.RECOVERY,
            SkillType.EFFICIENCY,
            SkillType.PATH_PLANNING
        ],
        complexity=ScenarioComplexity.ADAPTIVE,
        
        special_requirements={
            "block_direct_path_prob": 1.0,
            "block_multiple_alternatives_prob": 0.9,
            "rotation_required_prob": 0.7,
            "multi_step_required_prob": 0.7,
            "corridor_width_range": (0.102, 0.105),
            "target_surrounded_prob": 0.7,
            "trap_scenarios_prob": 0.2,
            "adaptive_difficulty": True,        # 根据 T-block 位置自适应
            "description": "极限挑战，需要完美的规划和执行"
        },
        
        upgrade_thresholds={
            "success_rate": 0.60,  # 不再升级
            "collision_rate": 0.30,
            "avg_steps": 250,
            "min_episodes": 200
        },
        downgrade_thresholds={
            "success_rate": 0.32,
            "collision_rate": 0.38
        },
        
        description="专家级：极限挑战，包含陷阱、动态约束，需要完美的规划和执行"
    ),
}


# ============================================================================
# 高级难度管理器
# ============================================================================

class AdvancedDifficultyManager:
    """
    高级难度管理器
    
    特点：
    1. 7 级难度系统
    2. 多维度评估（成功率、碰撞率、步数、效率）
    3. 渐进式升级（带缓冲）
    4. 自适应调整
    """
    
    def __init__(self,
                 initial_level: int = 1,
                 evaluation_window: int = 50,
                 upgrade_patience: int = 2,
                 allow_downgrade: bool = True,
                 enable_adaptive: bool = True):
        """
        Args:
            initial_level: 初始难度级别 (1-7)
            evaluation_window: 评估窗口
            upgrade_patience: 升级耐心
            allow_downgrade: 是否允许降级
            enable_adaptive: 是否启用自适应调整
        """
        self.current_level = initial_level
        self.current_config = ADVANCED_DIFFICULTY_LEVELS[initial_level]
        
        self.evaluation_window = evaluation_window
        self.upgrade_patience = upgrade_patience
        self.allow_downgrade = allow_downgrade
        self.enable_adaptive = enable_adaptive
        
        # 历史记录
        self.episode_history = []
        self.difficulty_history = [(0, initial_level)]
        self.evaluation_history = []
        
        # 计数器
        self.upgrade_counter = 0
        self.downgrade_counter = 0
        
        # 自适应参数
        self.recent_performance_trend = []  # 最近的性能趋势
        
        print(f"✓ AdvancedDifficultyManager 初始化")
        print(f"  初始难度: Level {initial_level} - {self.current_config.name}")
        print(f"  目标技能: {[s.value for s in self.current_config.target_skills]}")
    
    def get_current_config(self) -> AdvancedDifficultyConfig:
        """获取当前难度配置"""
        return self.current_config
    
    def get_placement_strategy(self) -> PlacementStrategy:
        """根据当前难度的权重随机选择放置策略"""
        strategies = list(self.current_config.placement_strategies.keys())
        weights = list(self.current_config.placement_strategies.values())
        
        # 归一化
        total = sum(weights)
        probs = [w / total for w in weights]
        
        return np.random.choice(strategies, p=probs)
    
    def record_episode(self,
                       success: bool,
                       steps: int,
                       collision: bool,
                       reward: float = 0.0,
                       extra_metrics: Dict = None):
        """记录 episode 结果"""
        episode_data = {
            "episode": len(self.episode_history),
            "success": success,
            "steps": steps,
            "collision": collision,
            "reward": reward,
            "difficulty_level": self.current_level,
            "difficulty_name": self.current_config.name
        }
        
        if extra_metrics:
            episode_data.update(extra_metrics)
        
        self.episode_history.append(episode_data)
    
    def compute_stats(self) -> Dict:
        """计算统计数据"""
        if len(self.episode_history) == 0:
            return None
        
        recent = self.episode_history[-self.evaluation_window:]
        
        stats = {
            "success_rate": np.mean([ep["success"] for ep in recent]),
            "collision_rate": np.mean([ep["collision"] for ep in recent]),
            "avg_steps": np.mean([ep["steps"] for ep in recent]),
            "avg_reward": np.mean([ep["reward"] for ep in recent]),
            "sample_count": len(recent),
            "efficiency": self._compute_efficiency(recent)
        }
        
        return stats
    
    def _compute_efficiency(self, episodes: List[Dict]) -> float:
        """计算效率分数（成功且步数少）"""
        successful = [ep for ep in episodes if ep["success"]]
        if len(successful) == 0:
            return 0.0
        
        avg_steps = np.mean([ep["steps"] for ep in successful])
        # 归一化：步数越少，效率越高
        efficiency = max(0, 1.0 - (avg_steps - 80) / 200)
        return efficiency
    
    def should_evaluate(self) -> bool:
        """检查是否应该评估"""
        return len(self.episode_history) >= self.evaluation_window
    
    def evaluate_and_update(self) -> Tuple[int, Dict]:
        """评估并更新难度"""
        if not self.should_evaluate():
            return self.current_level, {"status": "insufficient_data"}
        
        stats = self.compute_stats()
        old_level = self.current_level
        
        # 检查升降级条件
        should_upgrade = self._check_upgrade_conditions(stats)
        should_downgrade = self._check_downgrade_conditions(stats)
        
        evaluation_result = {
            "episode": len(self.episode_history),
            "old_level": old_level,
            "stats": stats,
            "should_upgrade": should_upgrade,
            "should_downgrade": should_downgrade,
            "upgrade_counter": self.upgrade_counter,
            "downgrade_counter": self.downgrade_counter
        }
        
        # 升级逻辑
        if should_upgrade:
            self.upgrade_counter += 1
            self.downgrade_counter = 0
            
            if self.upgrade_counter >= self.upgrade_patience:
                new_level = min(self.current_level + 1, 7)
                if new_level != old_level:
                    self._apply_level_change(new_level, "upgrade", stats)
                    evaluation_result["action"] = "upgraded"
                    self.upgrade_counter = 0
                else:
                    evaluation_result["action"] = "max_level_reached"
            else:
                evaluation_result["action"] = f"upgrade_pending_{self.upgrade_counter}/{self.upgrade_patience}"
        
        # 降级逻辑
        elif should_downgrade and self.allow_downgrade:
            self.downgrade_counter += 1
            self.upgrade_counter = 0
            
            if self.downgrade_counter >= self.upgrade_patience:
                new_level = max(self.current_level - 1, 1)
                if new_level != old_level:
                    self._apply_level_change(new_level, "downgrade", stats)
                    evaluation_result["action"] = "downgraded"
                    self.downgrade_counter = 0
                else:
                    evaluation_result["action"] = "min_level_reached"
            else:
                evaluation_result["action"] = f"downgrade_pending_{self.downgrade_counter}/{self.upgrade_patience}"
        
        # 保持
        else:
            self.upgrade_counter = 0
            self.downgrade_counter = 0
            evaluation_result["action"] = "maintain"
        
        self.evaluation_history.append(evaluation_result)
        
        return self.current_level, evaluation_result
    
    def _check_upgrade_conditions(self, stats: Dict) -> bool:
        """检查升级条件"""
        thresholds = self.current_config.upgrade_thresholds
        
        if len(self.episode_history) < thresholds["min_episodes"]:
            return False
        
        # 所有条件都要满足
        conditions = [
            stats["success_rate"] >= thresholds["success_rate"],
            stats["collision_rate"] <= thresholds["collision_rate"],
            stats["avg_steps"] <= thresholds["avg_steps"]
        ]
        
        return all(conditions)
    
    def _check_downgrade_conditions(self, stats: Dict) -> bool:
        """检查降级条件"""
        thresholds = self.current_config.downgrade_thresholds
        
        # 任一条件满足即降级
        conditions = [
            stats["success_rate"] < thresholds["success_rate"],
            stats["collision_rate"] > thresholds["collision_rate"]
        ]
        
        return any(conditions)
    
    def _apply_level_change(self, new_level: int, action: str, stats: Dict):
        """应用难度变化"""
        old_level = self.current_level
        old_name = self.current_config.name
        
        self.current_level = new_level
        self.current_config = ADVANCED_DIFFICULTY_LEVELS[new_level]
        
        self.difficulty_history.append((len(self.episode_history), new_level))
        
        # 打印信息
        if action == "upgrade":
            print(f"\n📈 难度升级: Level {old_level} ({old_name}) → Level {new_level} ({self.current_config.name})")
        else:
            print(f"\n📉 难度降级: Level {old_level} ({old_name}) → Level {new_level} ({self.current_config.name})")
        
        print(f"   触发原因:")
        print(f"   - 成功率: {stats['success_rate']*100:.1f}%")
        print(f"   - 碰撞率: {stats['collision_rate']*100:.1f}%")
        print(f"   - 平均步数: {stats['avg_steps']:.1f}")
        print(f"   - 效率: {stats['efficiency']:.2f}")
        print(f"   新目标技能: {[s.value for s in self.current_config.target_skills]}")
    
    def get_summary(self) -> Dict:
        """获取训练摘要"""
        return {
            "total_episodes": len(self.episode_history),
            "current_level": self.current_level,
            "current_name": self.current_config.name,
            "difficulty_history": self.difficulty_history,
            "final_stats": self.compute_stats()
        }


# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """使用示例"""
    manager = AdvancedDifficultyManager(initial_level=1)
    
    # 模拟训练
    for episode in range(500):
        config = manager.get_current_config()
        strategy = manager.get_placement_strategy()
        
        # 模拟结果
        success = np.random.random() < 0.7
        steps = np.random.randint(80, 200)
        collision = np.random.random() < 0.15
        
        manager.record_episode(success, steps, collision)
        
        # 定期评估
        if (episode + 1) % 50 == 0:
            new_level, eval_result = manager.evaluate_and_update()
            print(f"\nEpisode {episode+1}: {eval_result['action']}")
    
    # 打印摘要
    summary = manager.get_summary()
    print(f"\n{'='*60}")
    print(f"训练摘要:")
    print(f"总 episodes: {summary['total_episodes']}")
    print(f"最终难度: Level {summary['current_level']} - {summary['current_name']}")
    print(f"难度变化: {summary['difficulty_history']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    example_usage()
