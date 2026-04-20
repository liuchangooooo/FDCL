"""
检查 LLM 生成的障碍物配置

这个脚本会：
1. 加载训练好的模型
2. 生成几个示例障碍物配置
3. 可视化障碍物布局
4. 分析障碍物难度
"""
import sys
import os
import pathlib
ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
import torch
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 加载配置
config_path = "data/outputs/2026.04.01/11.18.00_phase0_llm/.hydra/config.yaml"
cfg = OmegaConf.load(config_path)

# 初始化 LLM 生成器
from DIVO.env.pusht.llm_topology_generator import (
    LLMTopologyGenerator,
    StrategyExecutor,
    build_phase0_prompt_stage_a
)

print("=" * 80)
print("检查 LLM 生成的障碍物配置")
print("=" * 80)

# 检查是否有 API key
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    print("\n⚠️ 未找到 DEEPSEEK_API_KEY 环境变量")
    print("将使用训练时保存的拓扑生成器代码（如果存在）")
    print("\n如果需要重新生成，请设置环境变量：")
    print("export DEEPSEEK_API_KEY='your_api_key'")
    use_saved_code = True
else:
    print(f"\n✓ 找到 API key: {api_key[:10]}...")
    use_saved_code = False

# 初始化执行器
executor = StrategyExecutor(
    obstacle_size=0.01,
    target_pose=[0, 0, -np.pi/4]
)

# 尝试从 checkpoint 加载拓扑生成器代码
# （实际训练中代码存储在 workspace 对象中，这里我们需要重新生成）

if use_saved_code:
    print("\n⚠️ 无法从 checkpoint 加载拓扑生成器代码")
    print("需要 API key 重新生成")
    sys.exit(1)

# 初始化 LLM 生成器
llm_generator = LLMTopologyGenerator(
    api_type="deepseek",
    api_key=api_key,
    model="deepseek-chat",
    temperature=0.7,
    verbose=True
)

# 生成拓扑生成器代码
print("\n" + "=" * 80)
print("生成拓扑生成器代码")
print("=" * 80)

sample_tblock_pose = np.array([0.15, 0.10, np.pi/4])
prompt = build_phase0_prompt_stage_a(sample_tblock_pose, num_obstacles=2)

print(f"\n示例 T-block 位置: ({sample_tblock_pose[0]:.3f}, {sample_tblock_pose[1]:.3f}, {np.degrees(sample_tblock_pose[2]):.0f}°)")
print(f"Prompt 长度: {len(prompt)} 字符")

code = llm_generator._call_llm(prompt)
code = llm_generator._extract_code(code)

if code is None:
    print("\n❌ 代码生成失败")
    sys.exit(1)

print(f"\n✓ 代码生成成功，长度: {len(code)} 字符")
print("\n生成的代码:")
print("-" * 80)
print(code)
print("-" * 80)

# 加载到执行器
if not executor.load_topology_generator(code):
    print("\n❌ 代码加载失败")
    sys.exit(1)

# 测试生成障碍物
print("\n" + "=" * 80)
print("测试障碍物生成")
print("=" * 80)

test_cases = [
    np.array([0.15, 0.10, np.pi/4]),
    np.array([-0.12, 0.15, -np.pi/3]),
    np.array([0.10, -0.12, np.pi/2]),
    np.array([-0.15, -0.10, 0]),
]

all_obstacles = []
for i, tblock_pose in enumerate(test_cases):
    print(f"\n测试 {i+1}: T-block 位置 = ({tblock_pose[0]:.3f}, {tblock_pose[1]:.3f}, {np.degrees(tblock_pose[2]):.0f}°)")
    
    obstacles = executor.generate(tblock_pose, num_obstacles=2)
    all_obstacles.append((tblock_pose, obstacles))
    
    if obstacles:
        print(f"  生成了 {len(obstacles)} 个障碍物:")
        for j, obs in enumerate(obstacles):
            print(f"    {j+1}. ({obs['x']:.3f}, {obs['y']:.3f}) - {obs.get('purpose', 'N/A')}")
        
        # 计算障碍物到起点的距离
        for j, obs in enumerate(obstacles):
            dist_to_start = np.sqrt((obs['x'] - tblock_pose[0])**2 + (obs['y'] - tblock_pose[1])**2)
            dist_to_goal = np.sqrt(obs['x']**2 + obs['y']**2)
            print(f"    {j+1}. 距离起点: {dist_to_start:.3f}m, 距离终点: {dist_to_goal:.3f}m")
    else:
        print("  ❌ 未生成任何障碍物")

# 可视化
print("\n" + "=" * 80)
print("可视化障碍物布局")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for idx, (tblock_pose, obstacles) in enumerate(all_obstacles):
    ax = axes[idx]
    
    # 设置坐标范围
    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.25, 0.25)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Test {idx+1}: T-block at ({tblock_pose[0]:.2f}, {tblock_pose[1]:.2f}, {np.degrees(tblock_pose[2]):.0f}°)')
    
    # 画工作空间边界
    workspace = patches.Rectangle((-0.2, -0.2), 0.4, 0.4, 
                                  linewidth=2, edgecolor='black', 
                                  facecolor='none', linestyle='--')
    ax.add_patch(workspace)
    
    # 画起点 T-block (简化为矩形)
    tblock_width, tblock_height = 0.10, 0.12
    start_rect = patches.Rectangle(
        (tblock_pose[0] - tblock_width/2, tblock_pose[1] - tblock_height/2),
        tblock_width, tblock_height,
        angle=np.degrees(tblock_pose[2]),
        linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.5
    )
    ax.add_patch(start_rect)
    ax.plot(tblock_pose[0], tblock_pose[1], 'bo', markersize=10, label='Start')
    
    # 画终点 T-block
    goal_pose = [0, 0, -np.pi/4]
    goal_rect = patches.Rectangle(
        (-tblock_width/2, -tblock_height/2),
        tblock_width, tblock_height,
        angle=np.degrees(goal_pose[2]),
        linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.5
    )
    ax.add_patch(goal_rect)
    ax.plot(0, 0, 'go', markersize=10, label='Goal')
    
    # 画起点到终点的直线路径
    ax.plot([tblock_pose[0], 0], [tblock_pose[1], 0], 'k--', alpha=0.3, label='Direct Path')
    
    # 画障碍物
    if obstacles:
        for i, obs in enumerate(obstacles):
            obs_rect = patches.Rectangle(
                (obs['x'] - 0.01, obs['y'] - 0.01),
                0.02, 0.02,
                linewidth=1, edgecolor='red', facecolor='red', alpha=0.7
            )
            ax.add_patch(obs_rect)
            ax.text(obs['x'], obs['y'] + 0.025, f"O{i+1}", 
                   ha='center', va='bottom', fontsize=8)
    
    ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
output_path = "llm_obstacles_visualization.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ 可视化保存到: {output_path}")

# 分析难度
print("\n" + "=" * 80)
print("障碍物难度分析")
print("=" * 80)

total_obstacles = sum(len(obs) for _, obs in all_obstacles)
if total_obstacles == 0:
    print("\n❌ 没有生成任何障碍物，无法分析")
else:
    # 统计障碍物到起点和终点的平均距离
    dist_to_starts = []
    dist_to_goals = []
    dist_to_paths = []
    
    for tblock_pose, obstacles in all_obstacles:
        for obs in obstacles:
            # 距离起点
            dist_start = np.sqrt((obs['x'] - tblock_pose[0])**2 + (obs['y'] - tblock_pose[1])**2)
            dist_to_starts.append(dist_start)
            
            # 距离终点
            dist_goal = np.sqrt(obs['x']**2 + obs['y']**2)
            dist_to_goals.append(dist_goal)
            
            # 距离直线路径（点到线段的距离）
            # 线段: 起点 -> 终点
            p1 = np.array([tblock_pose[0], tblock_pose[1]])
            p2 = np.array([0, 0])
            p = np.array([obs['x'], obs['y']])
            
            # 计算点到线段的距离
            line_vec = p2 - p1
            line_len = np.linalg.norm(line_vec)
            if line_len > 0:
                line_unitvec = line_vec / line_len
                point_vec = p - p1
                proj_length = np.dot(point_vec, line_unitvec)
                proj_length = np.clip(proj_length, 0, line_len)
                proj_point = p1 + proj_length * line_unitvec
                dist_path = np.linalg.norm(p - proj_point)
            else:
                dist_path = np.linalg.norm(p - p1)
            
            dist_to_paths.append(dist_path)
    
    print(f"\n障碍物统计 (共 {total_obstacles} 个):")
    print(f"  平均距离起点: {np.mean(dist_to_starts):.3f}m (std: {np.std(dist_to_starts):.3f}m)")
    print(f"  平均距离终点: {np.mean(dist_to_goals):.3f}m (std: {np.std(dist_to_goals):.3f}m)")
    print(f"  平均距离直线路径: {np.mean(dist_to_paths):.3f}m (std: {np.std(dist_to_paths):.3f}m)")
    
    print(f"\n难度评估:")
    avg_dist_path = np.mean(dist_to_paths)
    if avg_dist_path < 0.05:
        print(f"  ⚠️ 非常困难: 障碍物距离路径平均 {avg_dist_path:.3f}m < 0.05m")
        print(f"     障碍物几乎在路径上，严重阻挡推动")
    elif avg_dist_path < 0.08:
        print(f"  ⚠️ 困难: 障碍物距离路径平均 {avg_dist_path:.3f}m < 0.08m")
        print(f"     障碍物较接近路径，需要精细避障")
    elif avg_dist_path < 0.12:
        print(f"  ✓ 中等: 障碍物距离路径平均 {avg_dist_path:.3f}m")
        print(f"     障碍物在路径侧方，提供适度挑战")
    else:
        print(f"  ✓ 简单: 障碍物距离路径平均 {avg_dist_path:.3f}m > 0.12m")
        print(f"     障碍物远离路径，影响较小")
    
    avg_dist_start = np.mean(dist_to_starts)
    if avg_dist_start < 0.08:
        print(f"  ⚠️ 起点附近有障碍物: 平均距离 {avg_dist_start:.3f}m < 0.08m")
        print(f"     可能阻挡初始推动")

print("\n" + "=" * 80)
print("分析完成")
print("=" * 80)
