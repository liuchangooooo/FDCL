"""
课程学习管理器

功能：
1. 跟踪训练统计（成功率、步数、碰撞率等）
2. 动态调整难度级别
3. 管理障碍物数量
"""

import numpy as np
from typing import Dict, List
from collections import deque


class CurriculumManager:
    """
    课程学习管理器
    
    根据策略表现动态调整环境难度
    """
    
    def __init__(self,
                 initial_difficulty: str = "easy",
                 initial_obstacle_num: int = 1,
                 history_size: int = 100):
        """
        Args:
            initial_difficulty: 初始难度 ("easy", "medium", "hard")
            initial_obstacle_num: 初始障碍物数量
            history_size: 历史记录大小
        """
        self.difficulty_levels = ["easy", "medium", "hard"]
        self.current_difficulty_idx = self.difficulty_levels.index(initial_difficulty)
        self.obstacle_num = initial_obstacle_num
        
        # 历史记录
        self.episode_history = deque(maxlen=history_size)
        
        # 统计信息
        self.stats_window = 50  # 用于统计的窗口大小
        
    def record_episode(self,
                      success: bool,
                      steps: int,
                      collision: bool,
                      reward: float):
        """
        记录一个 episode 的结果
        
        Args:
            success: 是否成功
            steps: 步数
            collision: 是否碰撞
            reward: 总奖励
        """
        self.episode_history.append({
            'success': success,
            'steps': steps,
            'collision': collision,
            'reward': reward
        })
    
    def get_stats(self) -> Dict:
        """
        获取最近的统计信息
        
        Returns:
            dict: 包含 success_rate, avg_steps, collision_rate, avg_reward
        """
        if len(self.episode_history) == 0:
            return {
                'success_rate': 0.0,
                'avg_steps': 0.0,
                'collision_rate': 0.0,
                'avg_reward': 0.0
            }
        
        recent = list(self.episode_history)[-self.stats_window:]
        
        success_rate = sum(1 for e in recent if e['success']) / len(recent)
        avg_steps = sum(e['steps'] for e in recent) / len(recent)
        collision_rate = sum(1 for e in recent if e['collision']) / len(recent)
        avg_reward = sum(e['reward'] for e in recent) / len(recent)
        
        return {
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'collision_rate': collision_rate,
            'avg_reward': avg_reward
        }
    
    def update_difficulty(self) -> str:
        """
        根据最近的表现更新难度
        
        Returns:
            str: 更新动作 ("upgrade", "downgrade", "maintain")
        """
        stats = self.get_stats()
        success_rate = stats['success_rate']
        
        old_idx = self.current_difficulty_idx
        
        # 晋级条件：成功率 > 80%
        if success_rate > 0.8 and self.current_difficulty_idx < len(self.difficulty_levels) - 1:
            self.current_difficulty_idx += 1
            # 增加障碍物数量
            if self.obstacle_num < 3:
                self.obstacle_num += 1
            return "upgrade"
        
        # 降级条件：成功率 < 30%
        elif success_rate < 0.3 and self.current_difficulty_idx > 0:
            self.current_difficulty_idx -= 1
            # 减少障碍物数量
            if self.obstacle_num > 1:
                self.obstacle_num -= 1
            return "downgrade"
        
        return "maintain"
    
    def get_difficulty_level(self) -> str:
        """获取当前难度级别"""
        return self.difficulty_levels[self.current_difficulty_idx]
    
    def get_obstacle_num(self) -> int:
        """获取当前障碍物数量"""
        return self.obstacle_num
    
    def set_difficulty(self, difficulty: str):
        """手动设置难度"""
        if difficulty in self.difficulty_levels:
            self.current_difficulty_idx = self.difficulty_levels.index(difficulty)
    
    def set_obstacle_num(self, num: int):
        """手动设置障碍物数量"""
        self.obstacle_num = max(1, min(5, num))  # 限制在 1-5 之间

