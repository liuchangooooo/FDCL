"""
Policy Performance Analyzer — ACGS Module 1

对照论文 Section 3.2 Module 1 实现策略性能分析器。
收集策略的结构化诊断信息，输出自然语言报告供 LLM Curriculum Generator 使用。

定量指标：
- 成功率 η
- 碰撞率及碰撞类型分布
- Q 值方差 Var[Q(s,a)]
- 轨迹熵 H(τ)

定性诊断：
- 失败轨迹聚类（按碰撞位置/类型分组）
- 典型失败模式描述
"""
import numpy as np
from collections import deque, Counter
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class StepRecord:
    """单步记录（用于轨迹级分析）"""
    obs: np.ndarray          # 观测
    action: np.ndarray       # 动作
    reward: float            # 奖励
    collision: bool          # 是否碰撞
    q_value: float = 0.0     # Q 值估计


@dataclass
class EpisodeDiagnostic:
    """单个 episode 的诊断信息"""
    tblock_start: List[float]       # T-block 起始位姿 [x, y, θ]
    tblock_end: List[float]         # T-block 终止位姿
    obstacle_config: List[Dict]     # 障碍物配置
    total_reward: float
    steps: int
    success: bool
    collision_count: int            # 碰撞次数
    collision_positions: List[List[float]]  # 碰撞时 T-block 位置
    q_values: List[float]           # 每步 Q 值
    action_log_probs: List[float]   # 动作 log 概率（用于计算轨迹熵）
    scenario_type: str
    difficulty: float


class PolicyPerformanceAnalyzer:
    """
    策略性能分析器 (ACGS Module 1)
    
    在每个课程阶段结束后，收集并分析策略的表现，
    生成结构化诊断报告供 LLM 生成下一批环境。
    """
    
    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: 滑动窗口大小，用于计算滚动统计
        """
        self.window_size = window_size
        self.episode_history: deque = deque(maxlen=window_size)
        
        # 累积统计
        self.total_episodes = 0
        self.stage_episodes = 0  # 当前阶段的 episode 数
        
    def reset_stage(self):
        """新阶段开始时重置阶段级统计"""
        self.stage_episodes = 0
        self.episode_history.clear()
    
    def record_episode(self, diagnostic: EpisodeDiagnostic):
        """记录一个 episode 的诊断信息"""
        self.episode_history.append(diagnostic)
        self.total_episodes += 1
        self.stage_episodes += 1
    
    # ========== 定量指标 ==========
    
    def get_success_rate(self, last_n: int = None) -> float:
        """成功率 η"""
        episodes = self._get_recent(last_n)
        if not episodes:
            return 0.0
        return sum(1 for e in episodes if e.success) / len(episodes)
    
    def get_collision_rate(self, last_n: int = None) -> float:
        """碰撞率（发生过碰撞的 episode 比例）"""
        episodes = self._get_recent(last_n)
        if not episodes:
            return 0.0
        return sum(1 for e in episodes if e.collision_count > 0) / len(episodes)
    
    def get_collision_type_distribution(self, last_n: int = None) -> Dict[str, float]:
        """
        碰撞类型分布 P(collision_type)
        
        基于碰撞位置聚类，将碰撞分为：
        - "path_early": 路径前段碰撞（起点附近）
        - "path_mid": 路径中段碰撞
        - "path_late": 路径后段碰撞（终点附近）
        - "obstacle_hit": 直接撞障碍物
        """
        episodes = self._get_recent(last_n)
        collision_types = Counter()
        total_collisions = 0
        
        for ep in episodes:
            if ep.collision_count == 0:
                continue
            for pos in ep.collision_positions:
                total_collisions += 1
                # 计算碰撞位置到起点和终点的距离比
                start = np.array(ep.tblock_start[:2])
                target = np.array([0, 0])
                collision_pos = np.array(pos[:2])
                
                dist_to_start = np.linalg.norm(collision_pos - start)
                dist_to_target = np.linalg.norm(collision_pos - target)
                total_dist = np.linalg.norm(start - target)
                
                if total_dist < 0.01:
                    progress = 0.5
                else:
                    progress = dist_to_start / total_dist
                
                if progress < 0.33:
                    collision_types["path_early"] += 1
                elif progress < 0.67:
                    collision_types["path_mid"] += 1
                else:
                    collision_types["path_late"] += 1
        
        if total_collisions == 0:
            return {}
        return {k: v / total_collisions for k, v in collision_types.items()}
    
    def get_q_value_variance(self, last_n: int = None) -> float:
        """Q 值方差 Var[Q(s,a)]"""
        episodes = self._get_recent(last_n)
        all_q = []
        for ep in episodes:
            all_q.extend(ep.q_values)
        if len(all_q) < 2:
            return 0.0
        return float(np.var(all_q))
    
    def get_q_value_mean(self, last_n: int = None) -> float:
        """Q 值均值"""
        episodes = self._get_recent(last_n)
        all_q = []
        for ep in episodes:
            all_q.extend(ep.q_values)
        if not all_q:
            return 0.0
        return float(np.mean(all_q))
    
    def get_trajectory_entropy(self, last_n: int = None) -> float:
        """
        轨迹熵 H(τ) = -E[log π(a|s)]
        
        较大的熵意味着策略行为更分散/探索更多，
        较小的熵意味着策略更确定/收敛。
        
        对于 TD3 这种确定性策略，用动作分布的标准差近似。
        """
        episodes = self._get_recent(last_n)
        if not episodes:
            return 0.0
        
        # 对于 TD3，没有显式 log_prob，
        # 用动作空间的协方差行列式的 log 来近似熵
        if episodes[0].action_log_probs:
            all_log_probs = []
            for ep in episodes:
                all_log_probs.extend(ep.action_log_probs)
            if all_log_probs:
                return float(-np.mean(all_log_probs))
        
        # 降级方案：用奖励方差作为行为一致性的代理指标
        rewards = [ep.total_reward for ep in episodes]
        if len(rewards) < 2:
            return 0.0
        return float(np.std(rewards))
    
    def get_avg_episode_length(self, last_n: int = None) -> float:
        """平均 episode 长度"""
        episodes = self._get_recent(last_n)
        if not episodes:
            return 0.0
        return float(np.mean([e.steps for e in episodes]))
    
    def get_avg_reward(self, last_n: int = None) -> float:
        """平均奖励"""
        episodes = self._get_recent(last_n)
        if not episodes:
            return 0.0
        return float(np.mean([e.total_reward for e in episodes]))
    
    # ========== 定性诊断 ==========
    
    def get_failure_clusters(self, last_n: int = None) -> List[Dict]:
        """
        失败轨迹聚类
        
        将失败 episode 按碰撞位置/场景类型分组，
        识别典型失败模式。
        """
        episodes = self._get_recent(last_n)
        failures = [e for e in episodes if not e.success]
        
        if not failures:
            return []
        
        clusters = []
        
        # 按场景类型分组
        scenario_groups = {}
        for ep in failures:
            key = ep.scenario_type
            if key not in scenario_groups:
                scenario_groups[key] = []
            scenario_groups[key].append(ep)
        
        for scenario_type, eps in scenario_groups.items():
            collision_eps = [e for e in eps if e.collision_count > 0]
            non_collision_eps = [e for e in eps if e.collision_count == 0]
            
            cluster = {
                "scenario_type": scenario_type,
                "count": len(eps),
                "collision_failures": len(collision_eps),
                "timeout_failures": len(non_collision_eps),
                "avg_reward": float(np.mean([e.total_reward for e in eps])),
                "avg_steps": float(np.mean([e.steps for e in eps])),
            }
            
            # 碰撞位置统计
            if collision_eps:
                all_collision_pos = []
                for ep in collision_eps:
                    all_collision_pos.extend(ep.collision_positions)
                if all_collision_pos:
                    positions = np.array(all_collision_pos)
                    cluster["collision_centroid"] = positions.mean(axis=0).tolist()
                    cluster["collision_spread"] = float(positions.std())
            
            clusters.append(cluster)
        
        # 按失败数排序
        clusters.sort(key=lambda c: c["count"], reverse=True)
        return clusters
    
    def get_failure_mode_descriptions(self, last_n: int = None) -> List[str]:
        """
        生成人类可读的失败模式描述
        """
        clusters = self.get_failure_clusters(last_n)
        descriptions = []
        
        for c in clusters:
            desc_parts = [f"场景'{c['scenario_type']}'失败{c['count']}次"]
            
            if c["collision_failures"] > 0:
                desc_parts.append(f"其中{c['collision_failures']}次因碰撞")
                if "collision_centroid" in c:
                    cx, cy = c["collision_centroid"][:2]
                    desc_parts.append(
                        f"碰撞集中在({cx:.3f}, {cy:.3f})附近"
                    )
            
            if c["timeout_failures"] > 0:
                desc_parts.append(f"{c['timeout_failures']}次因超时")
            
            desc_parts.append(f"平均奖励{c['avg_reward']:.1f}")
            descriptions.append("，".join(desc_parts))
        
        return descriptions
    
    # ========== 结构化报告（供 LLM 使用） ==========
    
    def generate_diagnostic_report(self, 
                                    current_stage: int,
                                    current_difficulty: float,
                                    last_n: int = None) -> Dict:
        """
        生成结构化诊断报告
        
        返回字典格式，后续可直接注入到 LLM prompt 中。
        """
        success_rate = self.get_success_rate(last_n)
        collision_rate = self.get_collision_rate(last_n)
        collision_dist = self.get_collision_type_distribution(last_n)
        q_var = self.get_q_value_variance(last_n)
        q_mean = self.get_q_value_mean(last_n)
        traj_entropy = self.get_trajectory_entropy(last_n)
        avg_reward = self.get_avg_reward(last_n)
        avg_length = self.get_avg_episode_length(last_n)
        failure_clusters = self.get_failure_clusters(last_n)
        failure_descriptions = self.get_failure_mode_descriptions(last_n)
        
        report = {
            "stage": current_stage,
            "difficulty": current_difficulty,
            "num_episodes": len(self._get_recent(last_n)),
            "quantitative": {
                "success_rate": round(success_rate, 3),
                "collision_rate": round(collision_rate, 3),
                "collision_type_distribution": {
                    k: round(v, 3) for k, v in collision_dist.items()
                },
                "q_value_mean": round(q_mean, 3),
                "q_value_variance": round(q_var, 3),
                "trajectory_entropy": round(traj_entropy, 3),
                "avg_reward": round(avg_reward, 2),
                "avg_episode_length": round(avg_length, 1),
            },
            "qualitative": {
                "failure_clusters": failure_clusters,
                "failure_descriptions": failure_descriptions,
            }
        }
        return report
    
    def format_report_for_llm(self,
                               current_stage: int,
                               current_difficulty: float,
                               last_n: int = None) -> str:
        """
        将诊断报告格式化为自然语言字符串，直接嵌入 LLM prompt。
        
        对照论文 Section 3.2 Module 1:
        "这些信息被格式化为结构化的自然语言报告，
         作为 LLM 生成下一批环境的 prompt 依据。"
        """
        report = self.generate_diagnostic_report(
            current_stage, current_difficulty, last_n
        )
        q = report["quantitative"]
        qual = report["qualitative"]
        
        lines = [
            "## 策略性能诊断报告",
            "",
            f"**训练阶段**: Stage {report['stage']}",
            f"**当前难度**: {report['difficulty']:.2f}",
            f"**评估样本数**: {report['num_episodes']} episodes",
            "",
            "### 定量指标",
            f"- **成功率 (η)**: {q['success_rate']:.1%}",
            f"- **碰撞率**: {q['collision_rate']:.1%}",
            f"- **平均奖励**: {q['avg_reward']:.2f}",
            f"- **平均步数**: {q['avg_episode_length']:.1f}",
            f"- **Q值均值**: {q['q_value_mean']:.3f}",
            f"- **Q值方差**: {q['q_value_variance']:.3f}",
            f"- **行为熵/一致性**: {q['trajectory_entropy']:.3f}",
        ]
        
        if q["collision_type_distribution"]:
            lines.append("- **碰撞位置分布**:")
            for ctype, prob in q["collision_type_distribution"].items():
                label_map = {
                    "path_early": "路径前段（起点附近）",
                    "path_mid": "路径中段",
                    "path_late": "路径后段（终点附近）",
                }
                lines.append(f"  - {label_map.get(ctype, ctype)}: {prob:.1%}")
        
        lines.append("")
        lines.append("### 失败模式分析")
        
        if qual["failure_descriptions"]:
            for desc in qual["failure_descriptions"]:
                lines.append(f"- {desc}")
        else:
            lines.append("- 当前阶段无显著失败模式")
        
        # 添加决策建议
        lines.append("")
        lines.append("### 课程决策参考")
        
        if q["success_rate"] >= 0.7:
            lines.append("- ✅ 成功率较高，策略已适应当前难度，建议**提升难度**")
            if q["collision_rate"] < 0.1:
                lines.append("- 碰撞率极低，可增加更多路径约束")
        elif q["success_rate"] >= 0.3:
            lines.append("- ⚠️ 成功率中等，策略仍在学习，建议**保持当前难度**继续训练")
            if q["collision_rate"] > 0.5:
                lines.append("- 碰撞率偏高，建议生成更多避障场景针对性训练")
        else:
            lines.append("- 🔴 成功率过低，策略能力不足，建议**降低难度**或增加训练时间")
        
        return "\n".join(lines)
    
    # ========== 工具方法 ==========
    
    def _get_recent(self, last_n: int = None) -> List[EpisodeDiagnostic]:
        """获取最近的 episode 记录"""
        if last_n is None:
            return list(self.episode_history)
        return list(self.episode_history)[-last_n:]
