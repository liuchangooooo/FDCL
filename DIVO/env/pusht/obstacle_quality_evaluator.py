"""
障碍物环境质量评价系统

评价生成的障碍物配置的质量，包括四个维度：
1. 可解性（Solvability）：是否有解
2. 难度（Difficulty）：对策略的挑战程度
3. 多样性（Diversity）：与历史配置的差异
4. 有效性（Effectiveness）：是否有意义
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import cdist
from DIVO.utils.util import analytic_obs_collision_check


class ObstacleQualityEvaluator:
    """障碍物质量评价器"""
    
    def __init__(
        self,
        obstacle_size: float = 0.02,
        workspace_size: float = 0.4,
        tblock_width: float = 0.1,
        tblock_thickness: float = 0.03,
        density_norm_max_obstacles: int = 4,
        target_pose: List[float] = None
    ):
        """
        初始化评价器
        
        Args:
            obstacle_size: 障碍物边长 (m)
            workspace_size: 工作空间大小 (m)
            tblock_width: T-block 宽度 (m)
            tblock_thickness: T-block 厚度 (m)
            target_pose: 目标位姿 [x, y, θ]，默认 [0, 0, -π/4]
        """
        self.obstacle_size = obstacle_size
        self.workspace_size = workspace_size
        self.tblock_width = tblock_width
        self.tblock_thickness = tblock_thickness
        # 用“障碍物数量”归一化密度分数：n=0 -> 0.0, n>=density_norm_max_obstacles -> 1.0
        self.density_norm_max_obstacles = max(int(density_norm_max_obstacles), 1)
        self.target_pose = target_pose if target_pose else [0.0, 0.0, -np.pi/4]
        
        # 历史配置存储
        self.history_configs: List[List[Dict]] = []
        
    def evaluate_obstacle_quality(
        self,
        obstacles: List[Dict],
        tblock_pose: List[float],
        target_pose: Optional[List[float]] = None,
        history_configs: Optional[List[List[Dict]]] = None
    ) -> Tuple[float, Dict, str]:
        """
        综合评价障碍物配置的质量
        
        Args:
            obstacles: 障碍物配置列表，每个元素为 {'x': float, 'y': float, ...}
            tblock_pose: T-block 初始位姿 [x, y, θ]
            target_pose: 目标位姿 [x, y, θ]，默认使用 self.target_pose
            history_configs: 历史配置列表，默认使用 self.history_configs
        
        Returns:
            overall_score: 综合质量评分 (0-1)
            detailed_scores: 详细评分字典
            feedback: 人类可读的反馈信息
        """
        if target_pose is None:
            target_pose = self.target_pose
        
        if history_configs is None:
            history_configs = self.history_configs
        
        # 1. 可解性（必须满足）
        is_solvable, solve_reason = self.evaluate_solvability(
            obstacles, tblock_pose, target_pose
        )
        if not is_solvable:
            return 0.0, {"solvability": 0.0}, f"配置不可解: {solve_reason}"
        
        # 2. 难度
        difficulty_score, difficulty_breakdown = self.evaluate_difficulty(
            obstacles, tblock_pose, target_pose
        )
        
        # 3. 多样性
        diversity_score, similar_configs = self.evaluate_diversity(
            obstacles, history_configs
        )
        
        # 4. 有效性
        effectiveness_score, issues = self.evaluate_effectiveness(
            obstacles, tblock_pose, target_pose, difficulty_score
        )
        
        # 综合评分（加权平均）
        # 默认权重
        weights = {
            'solvability': 0.3,    # 可解性最重要
            'difficulty': 0.25,    # 难度适中
            'diversity': 0.25,     # 多样性
            'effectiveness': 0.20  # 有效性
        }
        # 重要：当没有历史配置时，多样性信息不可靠，暂不计入整体评分
        if len(history_configs) == 0:
            weights['diversity'] = 0.0
            # 将剩余权重重新归一化，保持总和为1
            remaining_keys = ['solvability', 'difficulty', 'effectiveness']
            total = sum(weights[k] for k in remaining_keys)
            if total > 0:
                for k in remaining_keys:
                    weights[k] = weights[k] / total

        overall_score = (
            weights['solvability'] * 1.0 +  # 可解
            weights['difficulty'] * difficulty_score +
            weights['diversity'] * diversity_score +
            weights['effectiveness'] * effectiveness_score
        )
        
        detailed_scores = {
            'solvability': 1.0,
            'difficulty': difficulty_score,
            'difficulty_breakdown': difficulty_breakdown,
            'diversity': diversity_score,
            'effectiveness': effectiveness_score,
            'issues': issues
        }
        
        # 生成反馈
        feedback = self._generate_feedback(detailed_scores)
        
        return overall_score, detailed_scores, feedback
    
    def evaluate_solvability(
        self,
        obstacles: List[Dict],
        tblock_pose: List[float],
        target_pose: List[float]
    ) -> Tuple[bool, str]:
        """
        评价：这个配置是否可解？
        
        Returns:
            is_solvable: bool
            reason: str (如果不可解，原因是什么)
        """
        # 检查1：起点是否被困住
        if self._is_trapped(tblock_pose, obstacles):
            return False, "起点被困住，无法移动"
        
        # 注意：目标位置被障碍物占据的检查在LLM生成时已经考虑，此处不再检查
        
        # 检查2：是否存在路径（简单的连通性检查）
        if not self._has_path(tblock_pose, target_pose, obstacles):
            return False, "不存在从起点到终点的路径"
        
        # 注意：T-block 的旋转是在整个移动过程中进行的，不是只在目标位置旋转
        # 只要路径存在，理论上可以通过路径上的任意位置进行旋转来达到目标朝向
        # 因此不需要单独检查目标位置的旋转空间
        
        return True, "可解"
    
    def evaluate_difficulty(
        self,
        obstacles: List[Dict],
        tblock_pose: List[float],
        target_pose: List[float]
    ) -> Tuple[float, Dict]:
        """
        评价：这个配置的难度如何？
        
        Returns:
            difficulty_score: 0-1 (0=极简单, 1=极困难)
            difficulty_breakdown: dict (各维度的难度)
        """
        scores = {}
        
        # 维度1：路径复杂度
        # 对于Push-T任务，路径复杂度主要考虑位置移动的绕行程度
        # 注意：旋转复杂度在维度3（rotation_difficulty）中单独考虑
        direct_path_length = np.linalg.norm(
            np.array(target_pose[:2]) - np.array(tblock_pose[:2])
        )
        actual_path_length = self._compute_shortest_path_length(
            tblock_pose, target_pose, obstacles
        )
        if direct_path_length > 0:
            detour_ratio = actual_path_length / direct_path_length
            scores['path_complexity'] = np.clip((detour_ratio - 1.0) / 2.0, 0.0, 1.0)  # 0-1
        else:
            scores['path_complexity'] = 0.0
        
        # 维度2：空间约束
        min_clearance = self._compute_min_clearance(
            obstacles, tblock_pose, target_pose
        )
        # T-block 宽度 0.1m，clearance 越小越难
        # 限制在 0-1 范围内：clearance < 0.10 时接近1，clearance > 0.18 时接近0
        scores['space_constraint'] = np.clip(1.0 - (min_clearance - 0.10) / 0.08, 0.0, 1.0)
        
        # 维度3：旋转需求
        rotation_needed = abs(target_pose[2] - tblock_pose[2])
        # 归一化到 [0, π]
        rotation_needed = min(rotation_needed, 2*np.pi - rotation_needed)
        # 考虑路径上的旋转空间，而不仅仅是起点
        rotation_space = self._compute_rotation_space(tblock_pose, target_pose, obstacles)
        scores['rotation_difficulty'] = np.clip((rotation_needed / np.pi) * (1.0 - rotation_space), 0.0, 1.0)  # 0-1
        
        # 维度4：障碍物密度
        # 这里按障碍物数量做分段/归一化，而不是按面积密度（避免在0.4x0.4里几乎总饱和）
        num_obs = len(obstacles)
        scores['density'] = np.clip(num_obs / self.density_norm_max_obstacles, 0.0, 1.0)  # 0-1
        
        # 综合难度
        difficulty_score = np.mean(list(scores.values()))
        
        return difficulty_score, scores
    
    def evaluate_diversity(
        self,
        new_config: List[Dict],
        history_configs: List[List[Dict]]
    ) -> Tuple[float, List]:
        """
        评价：这个配置与历史配置的差异度
        
        Returns:
            diversity_score: 0-1 (0=完全相同, 1=完全不同)
            similar_configs: list (最相似的配置)
        """
        if len(history_configs) == 0:
            return 1.0, []
        
        # 提取新配置的特征
        new_features = self._extract_features(new_config)
        
        # 计算与历史配置的相似度
        similarities = []
        for hist_config in history_configs[-100:]:  # 只看最近100个
            hist_features = self._extract_features(hist_config)
            similarity = self._compute_similarity(new_features, hist_features)
            similarities.append(similarity)
        
        # 多样性 = 1 - 最大相似度
        max_similarity = max(similarities) if similarities else 0.0
        diversity_score = 1.0 - max_similarity
        
        # 找出最相似的配置
        if similarities:
            similar_indices = np.argsort(similarities)[-3:]
            similar_configs = [history_configs[i] for i in similar_indices if i < len(history_configs)]
        else:
            similar_configs = []
        
        return diversity_score, similar_configs
    
    def evaluate_effectiveness(
        self,
        obstacles: List[Dict],
        tblock_pose: List[float],
        target_pose: List[float],
        difficulty_score: float
    ) -> Tuple[float, List[str]]:
        """
        评价：这个配置是否有意义？
        
        Returns:
            effectiveness_score: 0-1
            issues: list (发现的问题)
        """
        issues = []
        score = 1.0
        
        # 检查1：障碍物是否太远（无意义）
        for i, obs in enumerate(obstacles):
            dist_to_path = self._distance_to_line(
                np.array([obs['x'], obs['y']]),
                np.array(tblock_pose[:2]),
                np.array(target_pose[:2])
            )
            if dist_to_path > 0.15:  # 距离路径太远
                issues.append(f"障碍物 {i} ({obs['x']:.3f}, {obs['y']:.3f}) 距离路径太远，无实际影响")
                score -= 0.2
        
        # 检查2：障碍物是否重叠
        for i, obs1 in enumerate(obstacles):
            for j, obs2 in enumerate(obstacles[i+1:], start=i+1):
                dist = np.linalg.norm(
                    np.array([obs1['x'], obs1['y']]) - 
                    np.array([obs2['x'], obs2['y']])
                )
                if dist < 0.03:  # 太近
                    issues.append(f"障碍物 {i} 和 {j} 距离过近 ({dist:.3f}m)")
                    score -= 0.2
        
        # 检查3：是否过于简单（可以直线推过去） 
        if self._can_push_straight(tblock_pose, target_pose, obstacles):
            issues.append("可以直线推动，过于简单")
            score -= 0.2

        # 根据整体难度做轻微调整（用连续量反映“变化强度”，但不与难度混为一谈）
        # - 如果整体难度极低（几乎与无障碍基线一样），有效性略降一点
        # - 如果整体难度很高（明显有强约束），在不违背其它检查的前提下略微上调
        if difficulty_score < 0.2:
            score -= 0.1
        elif difficulty_score > 0.8:
            score = min(score + 0.05, 1.0)
        
        score = max(0.0, score)
        
        return score, issues
    
    # ========== 辅助方法 ==========
    
    def _is_trapped(self, tblock_pose: List[float], obstacles: List[Dict]) -> bool:
        """检查起点是否被困住"""
        # 先检查初始位置是否与障碍物碰撞（借鉴V2的改进）
        if self._check_tblock_obstacle_collision(tblock_pose, obstacles):
            return True
        
        # 检查 T-block 周围是否有足够的移动空间
        # 简单检查：周围8个方向是否都被阻挡
        directions = [
            (0.05, 0), (-0.05, 0), (0, 0.05), (0, -0.05),
            (0.035, 0.035), (-0.035, 0.035), (0.035, -0.035), (-0.035, -0.035)
        ]
        
        blocked_directions = 0
        for dx, dy in directions:
            test_pose = [tblock_pose[0] + dx, tblock_pose[1] + dy, tblock_pose[2]]
            if self._check_tblock_obstacle_collision(test_pose, obstacles):
                blocked_directions += 1
        
        # 如果超过6个方向被阻挡，认为被困住
        return blocked_directions >= 6
    
    def _has_path(self, start: List[float], target: List[float], obstacles: List[Dict]) -> bool:
        """
        简单的路径存在性检查（基于采样和碰撞检测）
        """
        # 在起点和终点之间采样多个中间点
        num_samples = 10
        for i in range(num_samples + 1):
            t = i / num_samples
            x = start[0] * (1 - t) + target[0] * t
            y = start[1] * (1 - t) + target[1] * t
            theta = start[2] * (1 - t) + target[2] * t
            
            # 检查这个位置是否与障碍物碰撞
            test_pose = [x, y, theta]
            if self._check_tblock_obstacle_collision(test_pose, obstacles):
                # 如果碰撞，尝试稍微偏移
                for offset in [0.05, 0.08, 0.10]:
                    for dx, dy in [(offset, 0), (-offset, 0), (0, offset), (0, -offset)]:
                        test_pose_offset = [x + dx, y + dy, theta]
                        if not self._check_tblock_obstacle_collision(test_pose_offset, obstacles):
                            break
                    else:
                        continue
                    break
                else:
                    # 所有偏移都失败，路径可能不存在
                    return False
        
        return True
    
    def _compute_shortest_path_length(
        self,
        start: List[float],
        target: List[float],
        obstacles: List[Dict]
    ) -> float:
        """
        计算最短路径长度（简化版本）
        使用 A* 的简化版本：考虑绕行障碍物
        """
        direct_length = np.linalg.norm(
            np.array(target[:2]) - np.array(start[:2])
        )
        
        # 检查直线路径是否被阻挡
        if not self._is_path_blocked(np.array(start[:2]), np.array(target[:2]), obstacles):
            return direct_length
        
        # 如果被阻挡，估算绕行路径长度
        # 简单估算：找到最近的障碍物，计算绕行距离
        min_detour = float('inf')
        for obs in obstacles:
            obs_pos = np.array([obs['x'], obs['y']])
            # 计算绕过这个障碍物的路径长度
            dist_to_obs = np.linalg.norm(obs_pos - np.array(start[:2]))
            dist_obs_to_target = np.linalg.norm(obs_pos - np.array(target[:2]))
            detour = dist_to_obs + dist_obs_to_target
            min_detour = min(min_detour, detour)
        
        return min_detour if min_detour < float('inf') else direct_length * 1.5
    
    def _compute_min_clearance(
        self,
        obstacles: List[Dict],
        tblock_pose: List[float],
        target_pose: List[float]
    ) -> float:
        """计算路径上的最小间隙"""
        # 沿着路径采样点
        num_points = 20
        path_points = []
        for i in range(num_points + 1):
            t = i / num_points
            x = tblock_pose[0] * (1 - t) + target_pose[0] * t
            y = tblock_pose[1] * (1 - t) + target_pose[1] * t
            path_points.append(np.array([x, y]))
        
        min_clearance = float('inf')
        for point in path_points:
            for obs in obstacles:
                dist = np.linalg.norm(point - np.array([obs['x'], obs['y']]))
                # 考虑 T-block 和障碍物的尺寸
                clearance = dist - self.obstacle_size / 2 - self.tblock_width / 2
                min_clearance = min(min_clearance, clearance)
        
        return min_clearance if min_clearance < float('inf') else 0.2
    
    def _compute_rotation_space(
        self, 
        tblock_pose: List[float], 
        target_pose: List[float],
        obstacles: List[Dict]
    ) -> float:
        """
        计算路径上的旋转空间（0-1，1表示空间充足）
        考虑路径上的多个点，而不仅仅是起点
        使用实际路径（考虑障碍物绕行），而不是直线路径
        """
        # 获取实际路径上的采样点（考虑障碍物）
        path_points = self._sample_path_points(tblock_pose, target_pose, obstacles, num_points=10)
        
        # 对每个路径点计算旋转空间
        rotation_spaces = []
        test_angles = np.linspace(0, 2*np.pi, 16)  # 测试16个角度
        
        for point in path_points:
            collision_count = 0
            for angle in test_angles:
                test_pose = [point[0], point[1], angle]
                if self._check_tblock_obstacle_collision(test_pose, obstacles):
                    collision_count += 1
            
            # 该点的旋转空间 = 1 - 碰撞比例
            point_rotation_space = 1.0 - (collision_count / len(test_angles))
            rotation_spaces.append(point_rotation_space)
        
        # 返回所有路径点的最小旋转空间（最保守，反映瓶颈）
        # 也可以使用平均值：return np.mean(rotation_spaces)
        return min(rotation_spaces) if rotation_spaces else 1.0
    
    def _sample_path_points(
        self,
        start: List[float],
        target: List[float],
        obstacles: List[Dict],
        num_points: int = 10
    ) -> List[np.ndarray]:
        """
        在路径上采样点（考虑障碍物绕行）
        
        注意：这是几何估算路径，不是真实的执行路径
        - 质量评估阶段还没有实际运行环境，无法获得真实路径
        - 使用简化的几何估算：直线路径或简化的绕行路径
        - 这足以进行快速的质量筛选
        
        如果直线路径被阻挡，使用简化的绕行路径（起点 -> 绕行点 -> 终点）
        """
        path_points = []
        
        # 检查直线路径是否被阻挡
        if not self._is_path_blocked(
            np.array(start[:2]),
            np.array(target[:2]),
            obstacles
        ):
            # 直线路径畅通，使用直线路径
            for i in range(num_points + 1):
                t = i / num_points
                x = start[0] * (1 - t) + target[0] * t
                y = start[1] * (1 - t) + target[1] * t
                path_points.append(np.array([x, y]))
        else:
            # 直线路径被阻挡，使用简化的绕行路径
            # 找到阻挡直线路径的障碍物，计算绕行点
            blocking_obs = []
            for obs in obstacles:
                obs_pos = np.array([obs['x'], obs['y']])
                # 检查障碍物是否在直线路径附近
                dist_to_path = self._distance_to_line(
                    obs_pos,
                    np.array(start[:2]),
                    np.array(target[:2])
                )
                if dist_to_path < (self.obstacle_size + self.tblock_width) / 2 + 0.02:
                    blocking_obs.append(obs)
            
            if blocking_obs:
                # 找到最近的阻挡障碍物
                closest_obs = min(
                    blocking_obs,
                    key=lambda obs: np.linalg.norm(
                        np.array([obs['x'], obs['y']]) - np.array(start[:2])
                    )
                )
                obs_pos = np.array([closest_obs['x'], closest_obs['y']])
                
                # 计算绕行方向（垂直于路径方向）
                path_vec = np.array(target[:2]) - np.array(start[:2])
                path_length = np.linalg.norm(path_vec)
                if path_length > 1e-6:
                    path_dir = path_vec / path_length
                    perpendicular = np.array([-path_dir[1], path_dir[0]])
                    
                    # 选择绕行方向（选择距离起点更近的一侧）
                    side1 = obs_pos + perpendicular * (self.obstacle_size + self.tblock_width) / 2
                    side2 = obs_pos - perpendicular * (self.obstacle_size + self.tblock_width) / 2
                    
                    if np.linalg.norm(side1 - np.array(start[:2])) < np.linalg.norm(side2 - np.array(start[:2])):
                        waypoint = side1
                    else:
                        waypoint = side2
                    
                    # 生成绕行路径：起点 -> 绕行点 -> 终点
                    for i in range(num_points + 1):
                        t = i / num_points
                        if t < 0.5:
                            # 前半段：起点到绕行点
                            t_seg = t * 2
                            x = start[0] * (1 - t_seg) + waypoint[0] * t_seg
                            y = start[1] * (1 - t_seg) + waypoint[1] * t_seg
                        else:
                            # 后半段：绕行点到终点
                            t_seg = (t - 0.5) * 2
                            x = waypoint[0] * (1 - t_seg) + target[0] * t_seg
                            y = waypoint[1] * (1 - t_seg) + target[1] * t_seg
                        path_points.append(np.array([x, y]))
                else:
                    # 路径长度为0，使用直线路径
                    for i in range(num_points + 1):
                        t = i / num_points
                        x = start[0] * (1 - t) + target[0] * t
                        y = start[1] * (1 - t) + target[1] * t
                        path_points.append(np.array([x, y]))
            else:
                # 没有找到阻挡障碍物，使用直线路径
                for i in range(num_points + 1):
                    t = i / num_points
                    x = start[0] * (1 - t) + target[0] * t
                    y = start[1] * (1 - t) + target[1] * t
                    path_points.append(np.array([x, y]))
        
        return path_points
    
    def _extract_features(self, config: List[Dict]) -> Dict:
        """提取配置的特征向量"""
        if len(config) == 0:
            return {
                'num_obstacles': 0,
                'centroid': np.array([0, 0]),
                'spread': np.array([0, 0]),
                'symmetry': 0.0,
                'clustering': 0.0
            }
        
        positions = np.array([[obs['x'], obs['y']] for obs in config])
        
        features = {
            'num_obstacles': len(config),
            'centroid': np.mean(positions, axis=0),
            'spread': np.std(positions, axis=0),
            'symmetry': self._compute_symmetry(positions),
            'clustering': self._compute_clustering(positions)
        }
        
        return features
    
    def _compute_similarity(self, features1: Dict, features2: Dict) -> float:
        """计算两个配置的相似度"""
        # 空间分布相似度：质心越近越相似
        centroid_dist = np.linalg.norm(features1['centroid'] - features2['centroid'])
        spatial_sim = np.exp(-centroid_dist / 0.1)
        
        # 结构相似度：对称性越接近越相似
        struct_sim = 1.0 - abs(features1['symmetry'] - features2['symmetry'])
        struct_sim = max(min(struct_sim, 1.0), 0.0)
        
        # 扩展1：spread 相似度（分布范围/离散程度）
        # spread 是每个维度的标准差，这里用 L2 距离衡量差异，再用 exp 衰减
        spread_dist = np.linalg.norm(features1['spread'] - features2['spread'])
        spread_sim = np.exp(-spread_dist / 0.1)
        
        # 扩展2：clustering 相似度（聚集度：0-1）
        clustering_diff = abs(features1['clustering'] - features2['clustering'])
        clustering_sim = 1.0 - min(clustering_diff, 1.0)
        
        # 综合相似度
        # 不再考虑数量相似度，用：质心 + 对称性 + spread + clustering 四个维度平均
        similarity = (spatial_sim + struct_sim + spread_sim + clustering_sim) / 4.0
        
        return similarity
    
    def _compute_symmetry(self, positions: np.ndarray) -> float:
        """计算对称性（关于质心对称，借鉴V2的改进）"""
        if len(positions) < 2:
            return 0.0
        
        centroid = np.mean(positions, axis=0)
        # 检查关于质心的对称性（而不是原点，更合理）
        symmetric_count = 0
        for pos in positions:
            # 镜像点：关于质心对称
            mirrored = 2 * centroid - pos
            # 检查是否存在对称点
            dists = np.linalg.norm(positions - mirrored, axis=1)
            if np.min(dists) < 0.05:
                symmetric_count += 1
        
        return symmetric_count / len(positions)
    
    def _compute_clustering(self, positions: np.ndarray) -> float:
        """计算聚集度（0-1，1表示高度聚集）"""
        if len(positions) < 2:
            return 0.0
        
        # 计算平均最近邻距离
        distances = cdist(positions, positions)
        np.fill_diagonal(distances, np.inf)
        min_dists = np.min(distances, axis=1)
        avg_min_dist = np.mean(min_dists)
        
        # 归一化到 0-1（距离越小，聚集度越高）
        clustering = 1.0 - min(avg_min_dist / 0.1, 1.0)
        
        return clustering
    
    def _distance_to_line(self, point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
        """计算点到线段的距离（借鉴V2的改进：考虑线段范围，而非无限直线）"""
        line_vec = line_end - line_start
        line_len = np.linalg.norm(line_vec)
        
        if line_len < 1e-6:
            return np.linalg.norm(point - line_start)
        
        point_vec = point - line_start
        line_dir = line_vec / line_len
        projection = np.dot(point_vec, line_dir)
        
        # 投影在线段外：返回到端点的距离
        if projection < 0:
            return np.linalg.norm(point - line_start)
        elif projection > line_len:
            return np.linalg.norm(point - line_end)
        else:
            # 投影在线段上：返回垂直距离
            perpendicular = point_vec - projection * line_dir
            return np.linalg.norm(perpendicular)
    
    def _check_path_blocking(
        self,
        obstacles: List[Dict],
        tblock_pose: List[float],
        target_pose: List[float]
    ) -> bool:
        """检查是否阻挡路径"""
        return self._is_path_blocked(
            np.array(tblock_pose[:2]),
            np.array(target_pose[:2]),
            obstacles
        )
    
    def _is_path_blocked(
        self,
        start: np.ndarray,
        target: np.ndarray,
        obstacles: List[Dict]
    ) -> bool:
        """检查直线路径是否被阻挡"""
        # 确保输入是 numpy 数组
        start = np.array(start)
        target = np.array(target)
        
        # 在路径上采样点
        num_samples = 20
        for i in range(num_samples + 1):
            t = i / num_samples
            point = start * (1 - t) + target * t
            
            # 检查是否与任何障碍物太近
            for obs in obstacles:
                dist = np.linalg.norm(point - np.array([obs['x'], obs['y']]))
                if dist < (self.obstacle_size / 2 + self.tblock_width / 2 + 0.01):
                    return True
        
        return False
    
    def _check_corridor_formation(
        self,
        obstacles: List[Dict],
        tblock_pose: List[float],
        target_pose: List[float]
    ) -> bool:
        """检查是否形成狭窄通道"""
        if len(obstacles) < 2:
            return False
        
        # 检查路径两侧是否有障碍物形成通道
        path_vec = np.array(target_pose[:2]) - np.array(tblock_pose[:2])
        path_length = np.linalg.norm(path_vec)
        if path_length < 1e-6:
            return False
        
        path_dir = path_vec / path_length
        perpendicular = np.array([-path_dir[1], path_dir[0]])
        
        # 检查路径两侧的障碍物
        left_obs = []
        right_obs = []
        for obs in obstacles:
            obs_pos = np.array([obs['x'], obs['y']])
            rel_pos = obs_pos - np.array(tblock_pose[:2])
            side = np.dot(rel_pos, perpendicular)
            
            # 检查是否在路径范围内
            along_path = np.dot(rel_pos, path_dir)
            if 0 < along_path < path_length:
                if side > 0:
                    right_obs.append(obs)
                else:
                    left_obs.append(obs)
        
        # 如果两侧都有障碍物，可能形成通道
        return len(left_obs) > 0 and len(right_obs) > 0
    
    def _can_push_straight(
        self,
        tblock_pose: List[float],
        target_pose: List[float],
        obstacles: List[Dict]
    ) -> bool:
        """检查是否可以直线推动"""
        # 检查直线路径是否畅通
        return not self._is_path_blocked(
            np.array(tblock_pose[:2]),
            np.array(target_pose[:2]),
            obstacles
        )
    
    def _check_tblock_obstacle_collision(
        self,
        tblock_pose: List[float],
        obstacles: List[Dict]
    ) -> bool:
        """检查 T-block 是否与障碍物碰撞"""
        for obs in obstacles:
            if analytic_obs_collision_check(
                Tblock_angle=tblock_pose[2],
                obs_center=np.array([obs['x'], obs['y']]) - np.array(tblock_pose[:2]),
                obs_size=self.obstacle_size * 2,
                threshold=0.01
            ):
                return True
        return False
    
    def _generate_feedback(self, scores: Dict) -> str:
        """生成人类可读的反馈"""
        feedback = []
        
        # 难度反馈
        diff = scores['difficulty']
        if diff < 0.3:
            feedback.append("✓ 难度：简单")
        elif diff < 0.5:
            feedback.append("✓ 难度：中等")
        elif diff < 0.7:
            feedback.append("✓ 难度：困难")
        else:
            feedback.append("⚠ 难度：极难（可能过难）")
        
        # 多样性反馈
        div = scores['diversity']
        if div > 0.6:
            feedback.append("✓ 多样性：高（与历史配置差异大）")
        elif div > 0.3:
            feedback.append("○ 多样性：中等")
        else:
            feedback.append("✗ 多样性：低（与历史配置相似）")
        
        # 有效性反馈
        eff = scores['effectiveness']
        if eff > 0.7:
            feedback.append("✓ 有效性：高（有明确约束）")
        elif eff > 0.4:
            feedback.append("○ 有效性：中等")
        else:
            feedback.append("✗ 有效性：低")
            if scores.get('issues'):
                issues = scores['issues'][:2]  # 只显示前2个问题
                feedback.append(f"   问题: {', '.join(issues)}")
        
        return "\n".join(feedback)
    
    def add_to_history(self, config: List[Dict]):
        """将配置添加到历史"""
        self.history_configs.append(config.copy())
    
    def clear_history(self):
        """清空历史"""
        self.history_configs = []
