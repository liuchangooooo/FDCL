"""
测试验证时障碍物配置的清除逻辑
"""
import numpy as np
from DIVO.env.pusht import get_pusht_env

def test_validation_obstacle_clearing():
    """测试验证时清除 LLM 配置的逻辑"""
    
    print("=" * 60)
    print("测试：验证时清除 LLM 配置")
    print("=" * 60)
    
    # 创建 LLM 环境
    env_config = {
        '_target_': 'pusht_mujoco_llm',
        'obstacle': True,
        'obstacle_num': 2,
        'obstacle_size': 0.01,
        'obstacle_shape': 'box',
        'obstacle_dist': 'between',
        'action_scale': 4,
        'NUM_SUBSTEPS': 25,
        'action_dim': [6],
        'obs_dim': [8],
        'action_reg': True,
        'reg_coeff': 1.0,
        'dynamics_randomization': False,
    }
    
    env = get_pusht_env(**env_config)
    
    # 测试 1: 设置 LLM 配置
    print("\n[Test 1] 设置 LLM 障碍物配置")
    llm_config = [
        {'x': 0.1, 'y': 0.1},
        {'x': -0.1, 'y': -0.1}
    ]
    env.set_obstacle_config(llm_config)
    print(f"✓ LLM 配置已设置: {env.task.llm_obstacle_config}")
    
    # 测试 2: Reset 使用 LLM 配置
    print("\n[Test 2] Reset 环境（应使用 LLM 配置）")
    obs1 = env.reset()
    positions1 = env.get_obstacle_positions()
    print(f"✓ 障碍物位置: {positions1}")
    print(f"  观测维度: {obs1.shape}")
    
    # 测试 3: 清除 LLM 配置
    print("\n[Test 3] 清除 LLM 配置")
    env.clear_obstacle_config()
    print(f"✓ LLM 配置已清除: {env.task.llm_obstacle_config}")
    
    # 测试 4: Reset 使用原版 between 策略
    print("\n[Test 4] Reset 环境（应使用原版 between 策略）")
    obs2 = env.reset()
    positions2 = env.get_obstacle_positions()
    print(f"✓ 障碍物位置: {positions2}")
    print(f"  观测维度: {obs2.shape}")
    
    # 测试 5: 验证障碍物位置不同
    print("\n[Test 5] 验证两次 reset 的障碍物位置不同")
    pos1_array = np.array([[p['x'], p['y']] for p in positions1])
    pos2_array = np.array([[p['x'], p['y']] for p in positions2])
    diff = np.linalg.norm(pos1_array - pos2_array)
    print(f"  位置差异: {diff:.4f}")
    if diff > 0.01:
        print("✓ 障碍物位置已改变（使用了随机生成）")
    else:
        print("✗ 警告：障碍物位置相同（可能仍在使用 LLM 配置）")
    
    # 测试 6: 多次 reset 验证随机性
    print("\n[Test 6] 多次 reset 验证随机性")
    positions_list = []
    for i in range(3):
        env.reset()
        positions = env.get_obstacle_positions()
        positions_list.append(positions)
        print(f"  Reset {i+1}: {positions}")
    
    # 检查是否每次都不同
    all_different = True
    for i in range(len(positions_list) - 1):
        pos_i = np.array([[p['x'], p['y']] for p in positions_list[i]])
        pos_j = np.array([[p['x'], p['y']] for p in positions_list[i+1]])
        if np.linalg.norm(pos_i - pos_j) < 0.01:
            all_different = False
            break
    
    if all_different:
        print("✓ 每次 reset 障碍物位置都不同（随机生成正常）")
    else:
        print("✗ 警告：某些 reset 的障碍物位置相同")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    test_validation_obstacle_clearing()
