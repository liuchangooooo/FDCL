"""任务 6 组件 sanity:SkillReplayBuffer + reward 路由。"""
import numpy as np
import torch

from nav import nav_env as NE
from nav.td3 import SkillReplayBuffer


def main():
    dev = torch.device("cpu")
    K = 4
    buf = SkillReplayBuffer(NE.OBS_DIM, 2, size=1000, device=dev)
    beta = 0.1

    # w_0 转移:reward_div 应被忽略
    buf.add(np.zeros(NE.OBS_DIM), np.zeros(2), np.zeros(NE.OBS_DIM), 0.0,
            skill_id=0, reward_task=1.0, reward_div=5.0, beta_div=beta)
    # probe 转移:reward_total = task + beta*div
    for k in range(1, K + 1):
        buf.add(np.ones(NE.OBS_DIM), np.zeros(2), np.ones(NE.OBS_DIM), 1.0,
                skill_id=k, reward_task=2.0, reward_div=3.0, beta_div=beta,
                source=SkillReplayBuffer.SRC_DIVERSITY)

    # 直接查内部列
    assert buf.reward_total[0, 0] == 1.0, "w_0 reward_total 应=reward_task(不加 r_div)"
    for k in range(1, K + 1):
        exp = 2.0 + beta * 3.0
        assert abs(buf.reward_total[k, 0] - exp) < 1e-6, f"probe reward_total 应={exp}"
    print("  1) reward_total 路由:w_0=task、probe=task+β·div OK")

    # reward_for_critic 路由
    rfc = buf._reward_for_critic(np.arange(K + 1))
    assert abs(rfc[0, 0] - 1.0) < 1e-6, "w_0 critic 用 reward_task"
    for k in range(1, K + 1):
        assert abs(rfc[k, 0] - (2.0 + beta * 3.0)) < 1e-6, "probe critic 用 reward_total"
    print("  2) reward_for_critic 路由:w_0→task、probe→total OK")

    # 采样形状 + skill_id -> one-hot(用 codebook)
    b = buf.sample(3)
    assert b["obs"].shape == (3, NE.OBS_DIM)
    assert b["skill_id"].shape == (3,)
    assert b["reward_for_critic"].shape == (3, 1)
    assert b["source"].shape == (3,)
    codebook = torch.eye(K + 1)
    w = codebook[b["skill_id"].long()]     # [3, K+1] one-hot
    assert w.shape == (3, K + 1)
    assert torch.allclose(w.sum(1), torch.ones(3))
    print(f"  3) sample 形状 OK;skill_id->one-hot w={tuple(w.shape)} OK")

    # source 标签
    assert int(buf.source[0]) == SkillReplayBuffer.SRC_NORMAL
    assert int(buf.source[1]) == SkillReplayBuffer.SRC_DIVERSITY
    print("  4) source 标签(normal/diversity)OK")
    print("ALL PASS")


if __name__ == "__main__":
    main()
