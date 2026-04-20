"""
Push-T 环境难度升级设计

核心思想：
1. 多维度量化难度
2. 基于多指标综合评估
3. 渐进式升级，避免难度跳跃过大
"""
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class DifficultyConfig:
    """难度配置"""
    level: str  # "easy", "medium", "hard"
    
    # 障碍物数量（次要因素）
    num_obstacles_range: Tuple[int, int]  # (min, max)
    
    # 位置策略（主要因素）
    placement_strategy: Dict[str, any]
    
    # 场景类型权重
    scenario_weights: Dict[str, float]
    
    # 升级阈值
    upgrade_thresholds: Dict[str, float]
    downgrade_thresholds: Dict[str, float]
    
    # 描述
    description: str


# ============================================================================
# 预定义的难度配置（基于位置策略）
# ============================================================================

DIFFICULTY_CONFIGS = {
    "easy": DifficultyConfig(
        level="easy",
        num_obstacles_range=(1, 2),  # 1-2个障碍物
        
        # 位置策略：远离关键路径
        placement_strategy={
            "strategy_type": "off_path",
            "min_distance_to_path": 0.08,      # 距离直线路径至少 8cm
            "min_distance_to_start": 0.10,     # 距离起点至少 10cm
            "min_distance_to_target": 0.10,    # 距离终点至少 10cm
            "min_passage_width": 0.15,         # 留出至少 15cm 通道
            "placement_zone": "peripheral",    # 放在外围区域
            "block_direct_path": False,        # 不阻挡直线路径
            "description": "障碍物放在外围，不阻挡主要路径，只做轻微干扰"
        },
        
        scenario_weights={
            "path_blocking": 0.0,   # 不阻挡路径
            "corridor": 0.0,
            "rotation": 0.0,
            "surrounding": 0.0,
            "auto": 1.0             # LLM 自主决策（遵循 easy 策略）
        },
        
        upgrade_thresholds={
            "success_rate": 0.80,
            "collision_rate": 0.10,
            "avg_steps": 150,
            "min_episodes": 50
        },
        downgrade_thresholds={
            "success_rate": 0.10,
            "collision_rate": 0.60
        },
        description="简单模式：障碍物远离关键路径，测试基本避障意识"
    ),
    
    "medium": DifficultyConfig(
        level="medium",
        num_obstacles_range=(1, 3),  # 1-3个障碍物
        
        # 位置策略：部分阻挡路径，但留有明显绕行空间
        placement_strategy={
            "strategy_type": "partial_blocking",
            "path_blocking_probability": 0.6,  # 60% 概率阻挡直线路径
            "min_distance_to_start": 0.08,
            "min_distance_to_target": 0.08,
            "min_passage_width": 0.12,         # 留出 12cm 通道（略大于 T-block）
            "placement_zone": "mid_path",      # 放在路径中段
            "block_direct_path": True,         # 阻挡直线路径
            "alternative_path_width": 0.13,    # 绕行路径宽度
            "description": "障碍物阻挡直线路径，但有明显的绕行方案"
        },
        
        scenario_weights={
            "path_blocking": 0.5,   # 路径阻挡
            "corridor": 0.3,        # 狭窄通道
            "rotation": 0.2,        # 旋转挑战
            "surrounding": 0.0,
            "auto": 0.0
        },
        
        upgrade_thresholds={
            "success_rate": 0.75,
            "collision_rate": 0.15,
            "avg_steps": 180,
            "min_episodes": 100
        },
        downgrade_thresholds={
            "success_rate": 0.20,
            "collision_rate": 0.50
        },
        description="中等模式：阻挡主要路径，需要规划绕行，测试路径规划能力"
    ),
    
    "hard": DifficultyConfig(
        level="hard",
        num_obstacles_range=(2, 4),  # 2-4个障碍物
        
        # 位置策略：形成复杂约束，需要精确操作或多步规划
        placement_strategy={
            "strategy_type": "complex_constraint",
            "path_blocking_probability": 0.9,  # 90% 概率阻挡
            "min_distance_to_start": 0.06,     # 可以更靠近起点
            "min_distance_to_target": 0.06,    # 可以更靠近终点
            "min_passage_width": 0.11,         # 窄通道（接近 T-block 宽度）
            "placement_zone": "strategic",     # 战略位置（路径中点、目标周围）
            "block_direct_path": True,
            "block_alternative_paths": True,   # 也阻挡部分绕行路径
            "require_rotation": 0.4,           # 40% 概率需要旋转
            "multi_step_required": 0.3,        # 30% 概率需要多步规划
            "description": "障碍物形成复杂约束，需要精确操作、旋转或多步规划"
        },
        
        scenario_weights={
            "path_blocking": 0.2,
            "corridor": 0.2,
            "rotation": 0.3,        # 更多旋转挑战
            "surrounding": 0.2,     # 包围目标
            "auto": 0.1             # LLM 自主设计复杂场景
        },
        
        upgrade_thresholds={
            "success_rate": 0.70,
            "collision_rate": 0.20,
            "avg_steps": 200,
            "min_episodes": 150
        },
        downgrade_thresholds={
            "success_rate": 0.25,
            "collision_rate": 0.45
        },
        description="困难模式：复杂约束，需要高级规划、精确操作和旋转控制"
    )
}


# ============================================================================
# 难度升级管理器
# ============================================================================

class DifficultyManager:
    """
    难度升级管理器
    
    特点：
    1. 多指标综合评估（不仅看成功率）
    2. 渐进式升级（有缓冲期）
    3. 支持降级（避免过难导致学习停滞）
    4. 记录详细历史
    """
    
    def __init__(self, 
                 initial_level: str = "easy",
                 evaluation_window: int = 50,
                 upgrade_patience: int = 2,  # 连续满足条件 N 次才升级
                 allow_downgrade: bool = True):
        """
        Args:
            initial_level: 初始难度
            evaluation_window: 评估窗口大小（最近 N 个 episode）
            upgrade_patience: 升级耐心（连续满足条件几次才升级）
            allow_downgrade: 是否允许降级
        """
        self.current_level = initial_level
        self.current_config = DIFFICULTY_CONFIGS[initial_level]
        
        self.evaluation_window = evaluation_window
        self.upgrade_patience = upgrade_patience
        self.allow_downgrade = allow_downgrade
        
        # 历史记录
        self.episode_history = []  # 每个 episode 的结果
        self.difficulty_history = [(0, initial_level)]  # (episode_num, level)
        self.evaluation_history = []  # 每次评估的结果
        
        # 升级计数器
        self.upgrade_counter = 0  # 连续满足升级条件的次数
        self.downgrade_counter = 0
        
        print(f"✓ DifficultyManager 初始化")
        print(f"  初始难度: {initial_level}")
        print(f"  评估窗口: {evaluation_window} episodes")
        print(f"  升级耐心: {upgrade_patience} 次")
    
    def get_current_config(self) -> DifficultyConfig:
        """获取当前难度配置"""
        return self.current_config
    
    def record_episode(self, 
                       success: bool,
                       steps: int,
                       collision: bool,
                       reward: float = 0.0):
        """记录单个 episode 的结果"""
        self.episode_history.append({
            "episode": len(self.episode_history),
            "success": success,
            "steps": steps,
            "collision": collision,
            "reward": reward,
            "difficulty": self.current_level
        })
    
    def compute_stats(self) -> Dict:
        """计算最近窗口的统计数据"""
        if len(self.episode_history) == 0:
            return None
        
        # 取最近的 N 个 episode
        recent = self.episode_history[-self.evaluation_window:]
        
        stats = {
            "success_rate": np.mean([ep["success"] for ep in recent]),
            "collision_rate": np.mean([ep["collision"] for ep in recent]),
            "avg_steps": np.mean([ep["steps"] for ep in recent]),
            "avg_reward": np.mean([ep["reward"] for ep in recent]),
            "sample_count": len(recent)
        }
        
        return stats
    
    def should_evaluate(self) -> bool:
        """检查是否应该评估（是否有足够的数据）"""
        return len(self.episode_history) >= self.evaluation_window
    
    def evaluate_and_update(self) -> Tuple[str, Dict]:
        """
        评估并更新难度
        
        Returns:
            (new_level, evaluation_result)
        """
        if not self.should_evaluate():
            return self.current_level, {"status": "insufficient_data"}
        
        stats = self.compute_stats()
        old_level = self.current_level
        
        # 检查升级条件
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
                new_level = self._upgrade_level()
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
                new_level = self._downgrade_level()
                if new_level != old_level:
                    self._apply_level_change(new_level, "downgrade", stats)
                    evaluation_result["action"] = "downgraded"
                    self.downgrade_counter = 0
                else:
                    evaluation_result["action"] = "min_level_reached"
            else:
                evaluation_result["action"] = f"downgrade_pending_{self.downgrade_counter}/{self.upgrade_patience}"
        
        # 保持当前难度
        else:
            self.upgrade_counter = 0
            self.downgrade_counter = 0
            evaluation_result["action"] = "maintain"
        
        self.evaluation_history.append(evaluation_result)
        
        return self.current_level, evaluation_result
    
    def _check_upgrade_conditions(self, stats: Dict) -> bool:
        """检查是否满足升级条件（所有条件都要满足）"""
        thresholds = self.current_config.upgrade_thresholds
        
        # 检查最小 episode 数
        if len(self.episode_history) < thresholds["min_episodes"]:
            return False
        
        # 检查成功率
        if stats["success_rate"] < thresholds["success_rate"]:
            return False
        
        # 检查碰撞率
        if stats["collision_rate"] > thresholds["collision_rate"]:
            return False
        
        # 检查平均步数（可选）
        if "avg_steps" in thresholds:
            if stats["avg_steps"] > thresholds["avg_steps"]:
                return False
        
        return True
    
    def _check_downgrade_conditions(self, stats: Dict) -> bool:
        """检查是否满足降级条件（任一条件满足即降级）"""
        thresholds = self.current_config.downgrade_thresholds
        
        # 成功率过低
        if stats["success_rate"] < thresholds["success_rate"]:
            return True
        
        # 碰撞率过高
        if stats["collision_rate"] > thresholds["collision_rate"]:
            return True
        
        return False
    
    def _upgrade_level(self) -> str:
        """升级难度"""
        levels = ["easy", "medium", "hard"]
        current_idx = levels.index(self.current_level)
        if current_idx < len(levels) - 1:
            return levels[current_idx + 1]
        return self.current_level
    
    def _downgrade_level(self) -> str:
        """降级难度"""
        levels = ["easy", "medium", "hard"]
        current_idx = levels.index(self.current_level)
        if current_idx > 0:
            return levels[current_idx - 1]
        return self.current_level
    
    def _apply_level_change(self, new_level: str, action: str, stats: Dict):
        """应用难度变化"""
        old_level = self.current_level
        self.current_level = new_level
        self.current_config = DIFFICULTY_CONFIGS[new_level]
        
        # 记录历史
        self.difficulty_history.append((len(self.episode_history), new_level))
        
        # 打印信息
        if action == "upgrade":
            print(f"\n📈 难度升级: {old_level} → {new_level}")
        else:
            print(f"\n📉 难度降级: {old_level} → {new_level}")
        
        print(f"   触发原因:")
        print(f"   - 成功率: {stats['success_rate']*100:.1f}%")
        print(f"   - 碰撞率: {stats['collision_rate']*100:.1f}%")
        print(f"   - 平均步数: {stats['avg_steps']:.1f}")
        print(f"   新配置: {self.current_config.description}")
    
    def get_scenario_type(self) -> str:
        """根据当前难度的权重随机选择场景类型"""
        weights = self.current_config.scenario_weights
        scenarios = list(weights.keys())
        probs = list(weights.values())
        
        # 归一化概率
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            probs = [1.0 / len(probs)] * len(probs)
        
        return np.random.choice(scenarios, p=probs)
    
    def get_summary(self) -> Dict:
        """获取训练摘要"""
        return {
            "total_episodes": len(self.episode_history),
            "current_level": self.current_level,
            "difficulty_history": self.difficulty_history,
            "evaluation_history": self.evaluation_history,
            "final_stats": self.compute_stats()
        }


# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """使用示例"""
    
    difficulty_mgr = DifficultyManager(
        initial_level="easy",
        evaluation_window=50,
        upgrade_patience=2
    )
    
    # 模拟训练
    for episode in range(300):
        # 获取当前配置
        config = difficulty_mgr.get_current_config()
        scenario = difficulty_mgr.get_scenario_type()
        
        # 模拟 episode 结果（这里用随机数模拟）
        # 实际使用时，这些数据来自真实的训练
        success = np.random.random() < 0.7
        steps = np.random.randint(80, 200)
        collision = np.random.random() < 0.15
        
        # 记录结果
        difficulty_mgr.record_episode(success, steps, collision)
        
        # 定期评估
        if (episode + 1) % 50 == 0:
            new_level, eval_result = difficulty_mgr.evaluate_and_update()
            print(f"\nEpisode {episode+1}: {eval_result['action']}")
    
    # 打印摘要
    summary = difficulty_mgr.get_summary()
    print(f"\n训练摘要:")
    print(f"总 episodes: {summary['total_episodes']}")
    print(f"最终难度: {summary['current_level']}")
    print(f"难度变化: {summary['difficulty_history']}")


if __name__ == "__main__":
    example_usage()
