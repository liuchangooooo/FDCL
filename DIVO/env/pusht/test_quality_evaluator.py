"""
测试质量评价器

测试不同的障碍物配置，验证评价系统是否合理
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
from DIVO.env.pusht.obstacle_quality_evaluator import ObstacleQualityEvaluator


def draw_tblock(ax, x, y, angle, color='blue', alpha=0.8):
    """绘制 T 形方块（来自 test_llm_v2.py）"""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    bar1_corners = np.array([[-0.05, 0], [0.05, 0], [0.05, 0.03], [-0.05, 0.03]])
    bar1_rot = bar1_corners @ rot.T + np.array([x, y])
    
    bar2_corners = np.array([[-0.015, -0.07], [0.015, -0.07], [0.015, 0], [-0.015, 0]])
    bar2_rot = bar2_corners @ rot.T + np.array([x, y])
    
    ax.add_patch(patches.Polygon(bar1_rot, closed=True, facecolor=color, edgecolor='black', alpha=alpha, linewidth=1.5))
    ax.add_patch(patches.Polygon(bar2_rot, closed=True, facecolor=color, edgecolor='black', alpha=alpha, linewidth=1.5))


def draw_obstacle(ax, x, y, size=0.01, color='red', alpha=0.7):
    """绘制障碍物（来自 test_llm_v2.py）"""
    rect = FancyBboxPatch(
        (x - size, y - size), size * 2, size * 2,
        boxstyle="round,pad=0.002",
        facecolor=color, edgecolor='darkred',
        alpha=alpha, linewidth=2
    )
    ax.add_patch(rect)


def visualize_scenario(
    tblock_pose, 
    target_pose, 
    obstacles, 
    evaluator,
    quality_score,
    detailed_scores,
    output_path
):
    """
    可视化一个测试场景
    
    Args:
        tblock_pose: T-block 初始位姿 [x, y, θ]
        target_pose: 目标位姿 [x, y, θ]
        obstacles: 障碍物列表
        evaluator: ObstacleQualityEvaluator 实例
        quality_score: 综合质量分数
        detailed_scores: 详细评分
        output_path: 输出图像路径
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # 工作空间大小
    ws_half = evaluator.workspace_size / 2
    
    # 设置坐标轴范围
    ax.set_xlim(-ws_half - 0.05, ws_half + 0.05)
    ax.set_ylim(-ws_half - 0.05, ws_half + 0.05)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    
    # 绘制工作空间边界
    boundary = patches.Rectangle(
        (-ws_half, -ws_half), 
        evaluator.workspace_size, 
        evaluator.workspace_size,
        linewidth=2, 
        edgecolor='red', 
        facecolor='none', 
        linestyle='--'
    )
    ax.add_patch(boundary)
    
    # 绘制起点到终点的连线
    ax.plot(
        [tblock_pose[0], target_pose[0]], 
        [tblock_pose[1], target_pose[1]], 
        'g--', 
        linewidth=1.5, 
        alpha=0.6,
        label='Direct Path'
    )
    
    # 绘制起始 T-block（蓝色，alpha=0.8）
    draw_tblock(ax, tblock_pose[0], tblock_pose[1], tblock_pose[2], color='blue', alpha=0.8)
    
    # 绘制目标 T-block（绿色，alpha=0.4）
    draw_tblock(ax, target_pose[0], target_pose[1], target_pose[2], color='green', alpha=0.4)
    
    # 绘制障碍物
    for i, obs in enumerate(obstacles):
        obs_x = obs['x']
        obs_y = obs['y']
        obs_size = evaluator.obstacle_size
        
        # 使用新的 draw_obstacle 函数
        draw_obstacle(ax, obs_x, obs_y, size=obs_size/2, color='red', alpha=0.7)
        
        # 添加编号
        ax.annotate(f"{i+1}", (obs_x, obs_y), 
                   fontsize=10, ha='center', va='center', 
                   color='white', fontweight='bold')
    
    # 添加标题和分数信息
    title = f"Quality Score: {quality_score:.3f}"
    if detailed_scores.get('solvability', 0.0) == 0.0:
        title += " (Unsolvable)"
    else:
        title += f" | D: {detailed_scores.get('difficulty', 0.0):.2f} | "
        title += f"Div: {detailed_scores.get('diversity', 0.0):.2f} | "
        title += f"Eff: {detailed_scores.get('effectiveness', 0.0):.2f}"
    
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  [已保存可视化图像: {output_path}]")


def test_quality_evaluator():
    """测试质量评价器"""
    
    evaluator = ObstacleQualityEvaluator()
    
    # 测试场景
    test_cases = [
        {
            "name": "简单场景 - 单个障碍物在路径中点",
            "tblock_pose": [0.15, 0.15, np.pi/4],
            "obstacles": [
                {"x": 0.075, "y": 0.075}
            ]
        },
        {
            "name": "中等场景 - 两个障碍物形成通道",
            "tblock_pose": [0.15, 0.15, np.pi/4],
            "obstacles": [
                {"x": 0.07, "y": 0.07},
                {"x": 0.04, "y": 0.10}
            ]
        },
        {
            "name": "困难场景 - 多个障碍物复杂布局",
            "tblock_pose": [0.15, 0.15, np.pi/4],
            "obstacles": [
                {"x": 0.08, "y": 0.08},
                {"x": 0.05, "y": 0.12},
                {"x": 0.12, "y": 0.05}
            ]
        },
        {
            "name": "不可解场景 - 起点被包围",
            "tblock_pose": [0.15, 0.15, np.pi/4],
            "obstacles": [
                {"x": 0.15, "y": 0.22},  # 上
                {"x": 0.15, "y": 0.08},  # 下
                {"x": 0.22, "y": 0.15},  # 右
                {"x": 0.08, "y": 0.15},  # 左
                {"x": 0.20, "y": 0.20},  # 右上
                {"x": 0.10, "y": 0.20}   # 左上
            ]
        },
        {
            "name": "无效场景 - 障碍物太远",
            "tblock_pose": [0.15, 0.15, np.pi/4],
            "obstacles": [
                {"x": -0.18, "y": -0.18}  # 在第三象限，远离路径
            ]
        },
        {
            "name": "碰撞场景 - 障碍物与起点重叠",
            "tblock_pose": [0.15, 0.15, np.pi/4],
            "obstacles": [
                {"x": 0.15, "y": 0.15}  # 与起点重叠
            ]
        },
        {
            "name": "多样性测试 - 第一个配置",
            "tblock_pose": [0.10, 0.10, 0],
            "obstacles": [
                {"x": 0.05, "y": 0.05}
            ],
            "add_to_history": True  # 标记：需要添加到历史
        },
        {
            "name": "多样性测试 - 相似配置",
            "tblock_pose": [0.10, 0.10, 0],
            "obstacles": [
                {"x": 0.06, "y": 0.06}  # 与上一个非常相似
            ]
        },
        {
            "name": "多样性测试 - 不同配置",
            "tblock_pose": [0.10, 0.10, 0],
            "obstacles": [
                {"x": -0.05, "y": 0.10},  # 完全不同的位置
                {"x": 0.10, "y": -0.05}
            ]
        }
    ]
    
    print("="*80)
    print("质量评价器测试")
    print("="*80)
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*80}")
        print(f"测试 {i+1}: {test_case['name']}")
        print(f"{'='*80}")
        
        tblock_pose = test_case['tblock_pose']
        obstacles = test_case['obstacles']
        
        print(f"\nT-block 初始位置: ({tblock_pose[0]:.3f}, {tblock_pose[1]:.3f}, {np.degrees(tblock_pose[2]):.1f}°)")
        print(f"障碍物配置:")
        for j, obs in enumerate(obstacles):
            print(f"  {j+1}. ({obs['x']:.3f}, {obs['y']:.3f})")
        
        # 评价
        quality_score, detailed_scores, feedback = evaluator.evaluate_obstacle_quality(
            obstacles=obstacles,
            tblock_pose=tblock_pose
        )
        
        # 获取目标位姿
        target_pose = evaluator.target_pose
        
        # 可视化场景
        output_path = f"test_scene_{i+1:02d}.png"
        visualize_scenario(
            tblock_pose=tblock_pose,
            target_pose=target_pose,
            obstacles=obstacles,
            evaluator=evaluator,
            quality_score=quality_score,
            detailed_scores=detailed_scores,
            output_path=output_path
        )
        
        # 打印结果
        print(f"\n{'─'*80}")
        print(f"评价结果:")
        print(f"{'─'*80}")
        print(f"综合质量分数: {quality_score:.3f}")
        print(f"\n详细评分:")
        print(f"  1. 可解性 (Solvability):   {detailed_scores.get('solvability', 0.0):.2f}")
        
        # 如果不可解，只显示可解性，其他维度不显示
        if detailed_scores.get('solvability', 0.0) == 0.0:
            print(f"  [配置不可解，其他维度未评估]")
        else:
            print(f"  2. 难度 (Difficulty):       {detailed_scores.get('difficulty', 0.0):.2f}")
            if 'difficulty_breakdown' in detailed_scores:
                breakdown = detailed_scores['difficulty_breakdown']
                print(f"     - 路径复杂度:           {breakdown.get('path_complexity', 0.0):.2f}")
                print(f"     - 空间约束:             {breakdown.get('space_constraint', 0.0):.2f}")
                print(f"     - 旋转难度:             {breakdown.get('rotation_difficulty', 0.0):.2f}")
                print(f"     - 障碍物密度:           {breakdown.get('density', 0.0):.2f}")
            print(f"  3. 多样性 (Diversity):      {detailed_scores.get('diversity', 0.0):.2f}")
            print(f"  4. 有效性 (Effectiveness):  {detailed_scores.get('effectiveness', 0.0):.2f}")
        
        if detailed_scores.get('issues'):
            print(f"\n问题:")
            for issue in detailed_scores['issues']:
                print(f"  - {issue}")
        
        print(f"\n反馈:")
        for line in feedback.split('\n'):
            print(f"  {line}")
        
        # 判断
        if quality_score >= 0.5:
            print(f"\n✓ 配置通过质量检查")
        else:
            print(f"\n✗ 配置未通过质量检查（需要重新生成）")
        
        # 如果需要，添加到历史（用于多样性测试）
        if test_case.get("add_to_history", False):
            evaluator.add_to_history(obstacles)
            print(f"\n[已添加到历史配置]")
    
    print(f"\n{'='*80}")
    print("测试完成")
    print(f"{'='*80}")


def test_diversity_tracking():
    """测试多样性追踪"""
    
    print("\n" + "="*80)
    print("多样性追踪测试")
    print("="*80)
    
    evaluator = ObstacleQualityEvaluator()
    
    # 生成一系列配置
    configs = [
        [{"x": 0.05, "y": 0.05}],
        [{"x": 0.06, "y": 0.06}],  # 相似
        [{"x": -0.10, "y": 0.10}],  # 不同
        [{"x": 0.05, "y": 0.05}],  # 重复第一个
        [{"x": 0.10, "y": -0.10}, {"x": -0.10, "y": -0.10}],  # 完全不同
    ]
    
    tblock_pose = [0.15, 0.15, 0]
    
    target_pose = evaluator.target_pose
    
    for i, config in enumerate(configs):
        print(f"\n配置 {i+1}: {config}")
        
        quality_score, detailed_scores, feedback = evaluator.evaluate_obstacle_quality(
            obstacles=config,
            tblock_pose=tblock_pose
        )
        
        # 添加到历史
        evaluator.add_to_history(config)
        
        print(f"  多样性分数: {detailed_scores['diversity']:.3f}")
        print(f"  综合质量: {quality_score:.3f}")
        
        # 可视化场景
        output_path = f"diversity_test_{i+1:02d}.png"
        visualize_scenario(
            tblock_pose=tblock_pose,
            target_pose=target_pose,
            obstacles=config,
            evaluator=evaluator,
            quality_score=quality_score,
            detailed_scores=detailed_scores,
            output_path=output_path
        )
    
    print(f"\n历史配置数量: {len(evaluator.history_configs)}")


if __name__ == "__main__":
    # 测试质量评价器
    test_quality_evaluator()
    
    # 测试多样性追踪
    test_diversity_tracking()