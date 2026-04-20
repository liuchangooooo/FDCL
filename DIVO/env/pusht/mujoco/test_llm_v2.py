"""
测试 LLM 障碍物生成器 V3（无场景类型版本）

运行方法:
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; python -m DIVO.env.pusht.mujoco.test_llm_v2
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import os
import warnings

# 配置 matplotlib 支持中文
try:
    import matplotlib.font_manager as fm
    # 尝试查找中文字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    chinese_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            chinese_font = font
            break
    
    if chinese_font:
        plt.rcParams['font.sans-serif'] = [chinese_font]
        plt.rcParams['axes.unicode_minus'] = False
    else:
        # 如果没有中文字体，使用英文标题
        print("⚠️ 未找到中文字体，将使用英文标题")
except Exception as e:
    print(f"⚠️ 字体配置失败: {e}")

# 过滤字体警告
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing.*')

from DIVO.env.pusht.llm_obstacle_generator_v3 import LLMObstacleGeneratorV3
from DIVO.env.pusht.obstacle_quality_evaluator import ObstacleQualityEvaluator

# DeepSeek API 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-be61cee13ada4dec87d37b461a1793c0")


def draw_tblock(ax, x, y, angle, color='blue', alpha=0.8):
    """绘制 T 形方块"""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    bar1_corners = np.array([[-0.05, 0], [0.05, 0], [0.05, 0.03], [-0.05, 0.03]])
    bar1_rot = bar1_corners @ rot.T + np.array([x, y])
    
    bar2_corners = np.array([[-0.015, -0.07], [0.015, -0.07], [0.015, 0], [-0.015, 0]])
    bar2_rot = bar2_corners @ rot.T + np.array([x, y])
    
    ax.add_patch(patches.Polygon(bar1_rot, closed=True, facecolor=color, edgecolor='black', alpha=alpha, linewidth=1.5))
    ax.add_patch(patches.Polygon(bar2_rot, closed=True, facecolor=color, edgecolor='black', alpha=alpha, linewidth=1.5))


def draw_obstacle(ax, x, y, size=0.01, color='red', alpha=0.7):
    """绘制障碍物"""
    rect = FancyBboxPatch(
        (x - size, y - size), size * 2, size * 2,
        boxstyle="round,pad=0.002",
        facecolor=color, edgecolor='darkred',
        alpha=alpha, linewidth=2
    )
    ax.add_patch(rect)


def visualize_scene(ax, tblock_pose, target_pose, config, title="", quality_score=None, detailed_scores=None):
    """可视化单个场景 - 改进版"""
    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.25, 0.25)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, color='gray')
    
    # 桌面边界（更清晰的样式）
    ax.add_patch(patches.Rectangle((-0.25, -0.25), 0.5, 0.5, 
                                   fill=False, edgecolor='#8B4513', linewidth=2, linestyle='-'))
    # 有效区域（更柔和的颜色）
    ax.add_patch(patches.Rectangle((-0.2, -0.2), 0.4, 0.4,
                                   fill=True, facecolor='#FFFACD', alpha=0.2, edgecolor='#DAA520', linewidth=1))
    
    # 直线路径（更明显的样式）
    ax.plot([tblock_pose[0], target_pose[0]], [tblock_pose[1], target_pose[1]],
            '--', linewidth=2, alpha=0.6, color='#32CD32', label='Direct Path', zorder=1)
    
    # 目标 T (绿色，更明显)
    draw_tblock(ax, target_pose[0], target_pose[1], target_pose[2], color='#228B22', alpha=0.5)
    # 初始 T (蓝色，更明显)
    draw_tblock(ax, tblock_pose[0], tblock_pose[1], tblock_pose[2], color='#4169E1', alpha=0.9)
    
    # 障碍物（改进样式）
    obstacle_colors = ['#DC143C', '#FF6347', '#FF4500', '#FF1493']
    for i, obs in enumerate(config):
        color = obstacle_colors[i % len(obstacle_colors)]
        draw_obstacle(ax, obs['x'], obs['y'], color=color, alpha=0.8)
        ax.annotate(f"{i+1}", (obs['x'], obs['y']), 
                   fontsize=11, ha='center', va='center', 
                   color='white', fontweight='bold', zorder=10)
    
    # 改进的标题格式
    difficulty_map = {'easy': 'Easy', 'medium': 'Medium', 'hard': 'Hard'}
    difficulty_display = difficulty_map.get(title.split()[0].lower(), title.split()[0])
    num_obs = len(config)
    
    title_text = f"{difficulty_display} ({num_obs} obstacle{'s' if num_obs > 1 else ''})"
    
    # 质量分数显示
    if quality_score is not None:
        # 根据分数选择颜色
        if quality_score >= 0.8:
            score_color = '#228B22'  # 绿色
        elif quality_score >= 0.6:
            score_color = '#FFA500'  # 橙色
        else:
            score_color = '#DC143C'  # 红色
        
        title_text += f"\nQuality: {quality_score:.3f}"
    
    ax.set_title(title_text, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    
    # 图例（更清晰）
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    
    # 添加坐标轴样式
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')


def test_v3_generation_diversity():
    """测试 V3 生成器的多样性：循环生成3次，看场景是否不同"""
    print("=" * 70)
    print("🧪 测试 LLM 障碍物生成器 V3 - 生成多样性测试")
    print("=" * 70)
    
    gen = LLMObstacleGeneratorV3(
        api_type="deepseek",
        api_key=DEEPSEEK_API_KEY if DEEPSEEK_API_KEY else None,
        model="deepseek-chat"
    )
    
    evaluator = ObstacleQualityEvaluator()
    
    tblock_pose = [0.15, 0.12, np.pi * 0.75]
    target_pose = [0, 0, -np.pi/4]
    
    print(f"\n📍 T-block: ({tblock_pose[0]:.2f}, {tblock_pose[1]:.2f}), θ={np.degrees(tblock_pose[2]):.1f}°")
    print(f"🎯 目标: (0, 0), θ=-45°")
    
    # 测试不同难度级别
    difficulties = ["easy", "medium", "hard"]
    num_obstacles_list = [1, 2, 3]
    
    # 布局：每个难度显示3次生成的结果（3行3列）
    fig = plt.figure(figsize=(18, 15))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3, 
                          left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    axes = []
    for row in range(3):
        for col in range(3):
            axes.append(fig.add_subplot(gs[row, col]))
    
    test_results = []
    
    # 对每个难度级别，循环生成3次
    for row_idx, (difficulty, num_obstacles) in enumerate(zip(difficulties, num_obstacles_list)):
        print(f"\n{'═'*70}")
        print(f"🔹 难度: {difficulty}, 障碍物数量: {num_obstacles}")
        print(f"{'═'*70}")
        
        generation_results = []
        
        # 循环生成3次，检查多样性
        max_attempts_per_gen = 3  # 每次生成最多尝试3次，避免过于相似
        for gen_idx in range(3):
            print(f"\n📌 第 {gen_idx + 1} 次生成")
            print(f"{'─'*50}")
            
            config = None
            last_failed_debug = None
            for attempt in range(max_attempts_per_gen):
                try:
                    candidate_config = gen.generate(
                        tblock_pose=tblock_pose,
                        num_obstacles=num_obstacles,
                        difficulty=difficulty,
                        generation_id=gen_idx + 1  # 传递生成序号，用于多样性提示
                    )
                    
                    # 如果生成的有效障碍物数量不足，认为本次尝试失败，直接重试
                    if candidate_config is None or len(candidate_config) < num_obstacles:
                        print(
                            f"  ⚠️  尝试 {attempt + 1}: 有效障碍物数量不足 "
                            f"(期望 {num_obstacles}, 实际 {0 if candidate_config is None else len(candidate_config)})，重新生成..."
                        )
                        # 记录本次失败尝试的调试信息，用于最终失败时可视化“LLM给了什么点”
                        last_failed_debug = getattr(gen, "last_debug", None)
                        continue
                    
                    # 检查与之前生成的结果是否过于相似
                    if gen_idx > 0:
                        too_similar = False
                        for prev_result in generation_results:
                            prev_positions = np.array([(obs['x'], obs['y']) for obs in prev_result['config']])
                            curr_positions = np.array([(obs['x'], obs['y']) for obs in candidate_config])
                            
                            if len(prev_positions) == len(curr_positions) and len(prev_positions) > 0:
                                # 计算平均位置差异
                                distances = np.linalg.norm(prev_positions - curr_positions, axis=1)
                                avg_dist = np.mean(distances)
                                
                                if avg_dist < 0.05:  # 如果平均位置差异 < 5cm，认为过于相似
                                    too_similar = True
                                    print(f"  ⚠️  尝试 {attempt + 1}: 与之前生成过于相似（平均差异 {avg_dist:.4f}m），重新生成...")
                                    break
                        
                        if too_similar and attempt < max_attempts_per_gen - 1:
                            continue  # 重新生成
                    
                    config = candidate_config
                    break  # 生成成功，跳出尝试循环
                    
                except Exception as e:
                    if attempt == max_attempts_per_gen - 1:
                        print(f"❌ 生成失败（尝试 {attempt + 1} 次）: {e}")
                        import traceback
                        traceback.print_exc()
                    else:
                        print(f"  ⚠️  尝试 {attempt + 1} 失败，重试...")
                        continue
            
            # 如果始终无法得到足够的有效障碍物，则将失败情况也可视化出来
            if config is None or len(config) < num_obstacles:
                valid_count = 0 if config is None else len(config)
                ax = axes[row_idx * 3 + gen_idx]
                # 即使失败也画出 T-block 与目标和直线路径，并把LLM输出的障碍物也画出来
                failed_obstacles = []
                if isinstance(last_failed_debug, dict) and last_failed_debug.get("all_obstacles"):
                    failed_obstacles = last_failed_debug["all_obstacles"]
                visualize_scene(
                    ax, tblock_pose, target_pose, failed_obstacles,
                    title=f"{difficulty.capitalize()} - Gen {gen_idx + 1}",
                    quality_score=None,
                    detailed_scores=None
                )
                # 不再在图中间打红字标记，只在标题里标注失败与有效数量
                ax.set_title(
                    f"{difficulty.capitalize()} - Gen {gen_idx + 1} - 失败 ({valid_count}/{num_obstacles})",
                    fontsize=12,
                    fontweight='bold',
                    pad=10
                )
                continue
            
            print(f"生成的配置 ({len(config)} 个障碍物):")
            for i, obs in enumerate(config):
                print(f"  {i+1}. ({obs['x']:.3f}, {obs['y']:.3f}) - {obs.get('purpose', 'N/A')}")
            
            # 评估质量
            quality, detailed, feedback = evaluator.evaluate_obstacle_quality(
                obstacles=config,
                tblock_pose=tblock_pose
            )
            
            evaluator.add_to_history(config)
            
            print(f"\n📊 质量评估:")
            print(f"  ✓ 质量分数: {quality:.3f}")
            print(f"  - 可解性: {detailed.get('solvability', 0):.3f}")
            print(f"  - 难度: {detailed.get('difficulty', 0):.3f}")
            print(f"  - 多样性: {detailed.get('diversity', 0):.3f}")
            print(f"  - 有效性: {detailed.get('effectiveness', 0):.3f}")
            if 'issues' in detailed and detailed['issues']:
                print(f"  ⚠️  问题: {', '.join(detailed['issues'])}")
            
            generation_results.append({
                'generation': gen_idx + 1,
                'config': config,
                'quality': quality,
                'detailed': detailed,
                'feedback': feedback
            })
            
            # 可视化
            ax = axes[row_idx * 3 + gen_idx]
            visualize_scene(
                ax, tblock_pose, target_pose, config,
                title=f"{difficulty.capitalize()} - Gen {gen_idx + 1}",
                quality_score=quality,
                detailed_scores=detailed
            )
        
        # 对比3次生成的结果
        if len(generation_results) >= 2:
            print(f"\n📈 多样性分析（{difficulty} 配置）:")
            print(f"{'─'*50}")
            
            # 计算障碍物位置的差异
            configs = [r['config'] for r in generation_results]
            positions_list = [[(obs['x'], obs['y']) for obs in config] for config in configs]
            
            print(f"  生成次数: {len(generation_results)}")
            print(f"  障碍物位置对比:")
            for i, (gen_result, positions) in enumerate(zip(generation_results, positions_list)):
                print(f"    生成 {i+1}: {positions} (质量: {gen_result['quality']:.3f})")
            
            # 计算位置差异
            if len(configs) >= 2:
                # 将位置转换为numpy数组
                pos_arrays = [np.array(positions) for positions in positions_list]
                
                # 计算配置之间的平均距离
                for i in range(len(pos_arrays)):
                    for j in range(i+1, len(pos_arrays)):
                        # 对齐障碍物（假设数量相同）
                        if len(pos_arrays[i]) == len(pos_arrays[j]) and len(pos_arrays[i]) > 0:
                            distances = np.linalg.norm(pos_arrays[i] - pos_arrays[j], axis=1)
                            avg_dist = np.mean(distances)
                            print(f"    生成 {i+1} vs 生成 {j+1}: 平均位置差异 {avg_dist:.3f}m")
        
        test_results.append({
            'difficulty': difficulty,
            'num_obstacles': num_obstacles,
            'generations': generation_results
        })
    
    # 保存图片
    os.makedirs("data/outputs", exist_ok=True)
    fig.savefig("data/outputs/llm_v3_generation_diversity.png", dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n✓ 结果已保存到: data/outputs/llm_v3_generation_diversity.png")
    
    # 打印测试总结
    print("\n" + "=" * 70)
    print("📊 生成多样性总结")
    print("=" * 70)
    for result in test_results:
        print(f"\n难度: {result['difficulty']}, 障碍物: {result['num_obstacles']}")
        for gen_result in result['generations']:
            print(f"  生成 {gen_result['generation']}: 质量 {gen_result['quality']:.3f}, "
                  f"障碍物位置: {[(obs['x'], obs['y']) for obs in gen_result['config']]}")
    
    plt.show()
    
    return test_results


def test_v3():
    """测试 V3 生成器（无场景类型版本）- 保留原版本"""
    test_v3_generation_diversity()


if __name__ == "__main__":
    test_v3_generation_diversity()
    print("\n✅ 测试完成!")
