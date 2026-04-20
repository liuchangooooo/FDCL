"""
ACGS 验证 1：数据采集验证

跑 10 个 episode（随机 action），检查：
1. info['termination'] 是否正确区分 collision/fall/success/timeout
2. info['collision_detail'] 碰撞时是否包含 type 和 obstacle_id
3. 帧缓冲区（deque maxlen=5）内容是否合理
4. Q 值序列是否按预期记录（本验证中用 None 占位）
"""
import sys
import os
import pathlib
import numpy as np
from collections import deque

ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

from DIVO.env.pusht.mujoco.pusht_mj_rod_llm import PushT_mj_rod_LLM

# ============================================================
# 1. 创建环境
# ============================================================
print("=" * 60)
print("ACGS 数据采集验证")
print("=" * 60)

env = PushT_mj_rod_LLM(
    obstacle=True,
    obstacle_num=2,
    obstacle_size=0.01,
    obstacle_shape='box',
    obstacle_dist='random',
    action_scale=4,
    action_dim=(6,),
    obs_dim=(8,),
    action_reg=True,
    reg_coeff=1.0,
    NUM_SUBSTEPS=25,
)
print(f"环境创建成功: obs_space={env.observation_space.shape}, action_space={env.action_space.shape}")

# ============================================================
# 2. 辅助函数：从 obs 解码 T-block 位姿
# ============================================================
def decode_tblock_pose(obs, desk_size=0.25):
    x = obs[0, 0] * desk_size
    y = obs[0, 1] * desk_size
    theta = np.arctan2(obs[0, 3], obs[0, 2])
    return [x, y, theta]

def get_rod_position(env_instance):
    try:
        raw_state = env_instance._observation_updater.get_observation()
        rod_pos = raw_state['unnamed_model/joint_positions'][0]
        return [float(rod_pos[0]), float(rod_pos[1])]
    except Exception as e:
        return [0.0, 0.0]

# ============================================================
# 3. 跑 10 个 episode
# ============================================================
FRAME_BUFFER_SIZE = 5
NUM_EPISODES = 10
MAX_STEPS = 10  # 与 config 中 max_steps 一致
TBLOCK_BOUND = 0.25
TBLOCK_FALL_BOUND = 0.40
ROD_BOUND = 0.35

termination_counts = {'success': 0, 'collision': 0, 'timeout': 0, 'fall': 0}

for ep in range(NUM_EPISODES):
    obs = env.reset()
    frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
    start_pose = decode_tblock_pose(obs)

    episode_reward = 0
    done = False
    step_count = 0
    last_info = None

    while not done:
        # 随机 action
        action = np.random.uniform(-1, 1, size=env.action_dim)

        next_obs, reward, done, info = env.step(action)

        # 记录帧
        tblock_pose = decode_tblock_pose(next_obs)
        rod_pos = get_rod_position(env)
        frame_buffer.append({
            'tblock': tblock_pose,
            'rod': rod_pos,
            'q_value': None,  # 没有 critic，用 None
        })

        episode_reward += reward
        step_count += 1
        last_info = info
        obs = next_obs

        if step_count >= MAX_STEPS - 1:
            done = True

    # ========================================================
    # 检查结果
    # ========================================================
    termination = last_info.get('termination', 'MISSING')
    collision_detail = last_info.get('collision_detail')
    success = last_info.get('success', False)

    # 统计
    if termination in termination_counts:
        termination_counts[termination] += 1

    print(f"\n--- Episode {ep+1}/{NUM_EPISODES} ---")
    print(f"  起点: T-block({start_pose[0]:.3f}, {start_pose[1]:.3f}, {np.degrees(start_pose[2]):.0f}°)")
    print(f"  步数: {step_count}, reward: {episode_reward:.2f}")
    print(f"  termination: {termination}")
    print(f"  success: {success}")
    print(f"  collision_detail: {collision_detail}")

    # 检查 termination 合法性
    assert termination in ('success', 'collision', 'fall', 'timeout'), \
        f"非法 termination 值: {termination}"

    # 碰撞时检查 detail
    if termination == 'collision':
        assert collision_detail is not None, "碰撞但 collision_detail 为 None"
        assert collision_detail['type'] in ('rod', 'tblock'), \
            f"非法碰撞类型: {collision_detail['type']}"
        assert collision_detail['obstacle_id'] in (0, 1), \
            f"非法障碍物 id: {collision_detail['obstacle_id']}"
        print(f"  -> {collision_detail['type']} 撞了障碍物 #{collision_detail['obstacle_id']}")

    # 检查帧缓冲区
    print(f"  帧缓冲区大小: {len(frame_buffer)} (max={FRAME_BUFFER_SIZE})")
    assert len(frame_buffer) <= FRAME_BUFFER_SIZE, "帧缓冲区超出限制"

    q_sequence = []

    for i, frame in enumerate(frame_buffer):
        t = frame['tblock']
        r = frame['rod']
        q = frame['q_value']

        # 坐标合理性检查
        # 注意：fall 终止时，T-block 可能滑出桌面边界；
        # 对非 fall 仍严格要求在 [-0.25, 0.25]，对 fall 放宽到物理可接受范围。
        if termination == 'fall':
            assert -TBLOCK_FALL_BOUND <= t[0] <= TBLOCK_FALL_BOUND, f"fall时 T-block x 异常: {t[0]}"
            assert -TBLOCK_FALL_BOUND <= t[1] <= TBLOCK_FALL_BOUND, f"fall时 T-block y 异常: {t[1]}"
            if abs(t[0]) > TBLOCK_BOUND or abs(t[1]) > TBLOCK_BOUND:
                print(f"    [warn] fall 轨迹超桌面边界: T({t[0]:.3f}, {t[1]:.3f})")
        else:
            assert -TBLOCK_BOUND <= t[0] <= TBLOCK_BOUND, f"T-block x 超范围: {t[0]}"
            assert -TBLOCK_BOUND <= t[1] <= TBLOCK_BOUND, f"T-block y 超范围: {t[1]}"

        assert -ROD_BOUND <= r[0] <= ROD_BOUND, f"rod x 超范围: {r[0]}"
        assert -ROD_BOUND <= r[1] <= ROD_BOUND, f"rod y 超范围: {r[1]}"

        step_idx = step_count - len(frame_buffer) + i + 1
        q_str = f"Q={q:.2f}" if q is not None else "Q=None"
        q_sequence.append(q)
        print(f"    step{step_idx}: T({t[0]:.3f},{t[1]:.3f},{np.degrees(t[2]):.0f}°) "
              f"rod({r[0]:.3f},{r[1]:.3f}) {q_str}")

    print(f"  Q 值序列(最后{len(q_sequence)}帧): {q_sequence}")

# ============================================================
# 4. 汇总
# ============================================================
print(f"\n{'=' * 60}")
print(f"汇总 ({NUM_EPISODES} episodes)")
print(f"{'=' * 60}")
print(f"  termination 分布: {termination_counts}")
total = sum(termination_counts.values())
assert total == NUM_EPISODES, f"计数不匹配: {total} != {NUM_EPISODES}"
print(f"  计数总和: {total} == {NUM_EPISODES} ✓")

# 检查至少出现 2 种 termination 类型
types_seen = [k for k, v in termination_counts.items() if v > 0]
print(f"  出现的类型: {types_seen}")
assert len(types_seen) >= 2, f"termination 类型不足 2 种: {types_seen}"

print(f"\n所有检查通过 ✓")
