"""
测试两次 Reset 修复方案

验证：
1. 第一次 reset 获取真实起点
2. 基于真实起点生成障碍物
3. 第二次 reset 使用相同起点 + 新障碍物
4. 起点一致性验证
5. force=True 后下一次不传参数 reset 不应粘住旧起点
"""
import sys
import os
import pathlib
ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np

# 初始化环境（直接创建，不使用配置文件）
from DIVO.env.pusht.mujoco.pusht_mj_rod_llm import PushT_mj_rod_LLM

DESK_SIZE = 0.25
POS_TOL = 0.01
ANG_TOL_DEG = 2.0
NUM_FORCE_TESTS = 5

env = PushT_mj_rod_LLM(
    obstacle=True,
    obstacle_num=2,
    obstacle_size=0.01,
    obstacle_shape='box',
    obstacle_dist='between',
    record_frame=False,
    action_dim=[6],
    obs_dim=[8],
    action_scale=4,
    NUM_SUBSTEPS=25,
    action_reg=True,
    reg_coeff=1.0,
    generate_dataset=False,
    motion_pred=False,
    eval=False,
    # 为起点一致性测试降噪，避免质量/动力学随机化干扰结果
    dynamics_randomization=False,
)

print("=" * 80)
print("测试两次 Reset 修复方案")
print("=" * 80)

# 初始化 LLM 生成器和执行器
from DIVO.env.pusht.llm_topology_generator import (
    LLMTopologyGenerator,
    StrategyExecutor,
    build_phase0_prompt_stage_a
)

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    print("\n⚠️ 未找到 DEEPSEEK_API_KEY，跳过 LLM 测试")
    print("测试将使用随机障碍物")
    use_llm = False
else:
    print(f"\n✓ 找到 API key: {api_key[:10]}...")
    use_llm = True
    
    llm_generator = LLMTopologyGenerator(
        api_type="deepseek",
        api_key=api_key,
        model="deepseek-chat",
        temperature=0.7,
        verbose=False
    )
    
    executor = StrategyExecutor(
        obstacle_size=0.01,
        target_pose=[0, 0, -np.pi/4]
    )
    
    # 生成拓扑生成器代码
    print("\n生成拓扑生成器代码...")
    sample_tblock_pose = np.array([0.15, 0.10, np.pi/4])
    prompt = build_phase0_prompt_stage_a(sample_tblock_pose, num_obstacles=2)
    code = llm_generator._call_llm(prompt)
    code = llm_generator._extract_code(code)
    
    if code and executor.load_topology_generator(code):
        print("✓ 拓扑生成器加载成功")
    else:
        print("❌ 拓扑生成器加载失败，使用随机障碍物")
        use_llm = False

# 辅助函数
def get_tblock_pose_from_obs(obs):
    """从归一化 obs 解码 T-block 位姿"""
    x = obs[0, 0] * DESK_SIZE
    y = obs[0, 1] * DESK_SIZE
    theta = np.arctan2(obs[0, 3], obs[0, 2])
    return [x, y, theta]


def angle_diff_deg(a, b):
    """返回角度差绝对值（度），自动处理 -pi/pi 环绕"""
    d = np.arctan2(np.sin(a - b), np.cos(a - b))
    return abs(np.degrees(d))


def pose_diff(pose_a, pose_b):
    dx = abs(pose_a[0] - pose_b[0])
    dy = abs(pose_a[1] - pose_b[1])
    dtheta = angle_diff_deg(pose_a[2], pose_b[2])
    return dx, dy, dtheta


def pose_match(pose_a, pose_b):
    dx, dy, dtheta = pose_diff(pose_a, pose_b)
    return (dx <= POS_TOL) and (dy <= POS_TOL) and (dtheta <= ANG_TOL_DEG)

# 测试两次 reset
print("\n" + "=" * 80)
print("测试两次 Reset 流程")
print("=" * 80)

force_pass = 0
for test_id in range(NUM_FORCE_TESTS):
    print(f"\n测试 {test_id + 1}/{NUM_FORCE_TESTS}:")
    
    # Step 1: 第一次 reset，获取真实起点
    obs1 = env.reset()
    tblock_pose_real = get_tblock_pose_from_obs(obs1)
    print(f"  第一次 reset: T-block = ({tblock_pose_real[0]:.3f}, {tblock_pose_real[1]:.3f}, {np.degrees(tblock_pose_real[2]):.0f}°)")
    
    # Step 2: 基于真实起点生成障碍物
    if use_llm:
        obstacles = executor.generate(tblock_pose_real, num_obstacles=2)
        if obstacles:
            env.set_obstacle_config(obstacles)
            print(f"  生成障碍物: {len(obstacles)} 个")
            for i, obs_cfg in enumerate(obstacles):
                print(f"    {i+1}. ({obs_cfg['x']:.3f}, {obs_cfg['y']:.3f}) - {obs_cfg.get('purpose', 'N/A')}")
        else:
            print("  ⚠️ 未生成障碍物")
            env.clear_obstacle_config()
    else:
        print("  跳过障碍物生成（无 LLM）")
        env.clear_obstacle_config()
    
    # Step 3: 第二次 reset，使用相同起点 + 新障碍物
    obs2 = env.reset(tblock_pos=tblock_pose_real, force_tblock_pos=True)
    tblock_pose_verify = get_tblock_pose_from_obs(obs2)
    print(f"  第二次 reset: T-block = ({tblock_pose_verify[0]:.3f}, {tblock_pose_verify[1]:.3f}, {np.degrees(tblock_pose_verify[2]):.0f}°)")
    
    # Step 4: 验证起点一致性
    dx, dy, dtheta = pose_diff(tblock_pose_real, tblock_pose_verify)
    if not pose_match(tblock_pose_real, tblock_pose_verify):
        print(f"  ❌ 起点不一致！差异: ({dx:.3f}, {dy:.3f}, {dtheta:.1f}°)")
    else:
        force_pass += 1
        print(f"  ✓ 起点一致！差异: ({dx:.4f}, {dy:.4f}, {dtheta:.1f}°)")

assert force_pass == NUM_FORCE_TESTS, (
    f"force_tblock_pos=True 一致性失败：{force_pass}/{NUM_FORCE_TESTS} 通过"
)
print(f"\n✓ force_tblock_pos=True 一致性测试通过: {force_pass}/{NUM_FORCE_TESTS}")

# 测试不使用 force_tblock_pos 的情况（应该随机）
print("\n" + "=" * 80)
print("测试不使用 force_tblock_pos（应该随机）")
print("=" * 80)

obs1 = env.reset()
tblock_pose1 = get_tblock_pose_from_obs(obs1)
print(f"第一次 reset: ({tblock_pose1[0]:.3f}, {tblock_pose1[1]:.3f}, {np.degrees(tblock_pose1[2]):.0f}°)")

obs2 = env.reset(tblock_pos=tblock_pose1, force_tblock_pos=False)
tblock_pose2 = get_tblock_pose_from_obs(obs2)
print(f"第二次 reset (force=False): ({tblock_pose2[0]:.3f}, {tblock_pose2[1]:.3f}, {np.degrees(tblock_pose2[2]):.0f}°)")

dx, dy, dtheta = pose_diff(tblock_pose1, tblock_pose2)
if not pose_match(tblock_pose1, tblock_pose2):
    print(f"✓ 起点不同（符合预期）：差异 ({dx:.3f}, {dy:.3f}, {dtheta:.1f}°)")
else:
    print(f"⚠️ 起点相同（可能是巧合）：差异 ({dx:.4f}, {dy:.4f}, {dtheta:.1f}°)")

# 测试 force 使用后是否粘住旧起点（eval_pose 残留污染检查）
print("\n" + "=" * 80)
print("测试 force 后下一次 reset 不应粘住旧起点")
print("=" * 80)

stale_trials = 5
stale_same_count = 0
for i in range(stale_trials):
    obs_seed = env.reset()
    pose_seed = get_tblock_pose_from_obs(obs_seed)

    # 强制一次
    obs_forced = env.reset(tblock_pos=pose_seed, force_tblock_pos=True)
    pose_forced = get_tblock_pose_from_obs(obs_forced)

    # 下一次不传参数，应恢复随机
    obs_unforced = env.reset()
    pose_unforced = get_tblock_pose_from_obs(obs_unforced)

    if pose_match(pose_forced, pose_unforced):
        stale_same_count += 1
        print(f"  trial {i+1}: ⚠️ unforced 与 forced 相同（可能巧合）")
    else:
        print(f"  trial {i+1}: ✓ unforced 与 forced 不同")

assert stale_same_count < stale_trials, (
    "疑似起点粘连：所有 trial 中 unforced reset 都与 forced 相同"
)
print(f"✓ 残留检查通过：{stale_trials - stale_same_count}/{stale_trials} 次 unforced 为新起点")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)
