"""任务 7 组件 sanity:SkillTD3 update(w-条件 critic + executed-w actor)。

用 bandit 式合成数据(done=1,无 bootstrap => 目标 Q = reward_for_critic):
  skill 0(w_0)的转移 reward_task=1.0;skill 1 的转移 reward_task=0.0。
验证训练后:Q1(obs,w_0,act)->~1.0、Q1(obs,w_1,act)->~0.0
=> w-条件 critic + reward 路由 + critic 学习均正常;executed-w actor 更新可跑、损失有限。
"""
import numpy as np
import torch

from nav import nav_env as NE
from nav.td3 import SkillReplayBuffer
from nav.skill_td3 import SkillTD3


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    K = 4
    agent = SkillTD3(NE.OBS_DIM, 2, K=K, device="cpu", policy_delay=1)
    buf = SkillReplayBuffer(NE.OBS_DIM, 2, size=20000, device=agent.device)

    rng = np.random.default_rng(0)
    fixed_obs = rng.standard_normal(NE.OBS_DIM).astype(np.float32)
    fixed_act = np.zeros(2, np.float32)
    for _ in range(4000):
        # skill 0 高回报,skill 1 零回报(其余技能随机填充)
        buf.add(fixed_obs, fixed_act, fixed_obs, 1.0, skill_id=0, reward_task=1.0)
        buf.add(fixed_obs, fixed_act, fixed_obs, 1.0, skill_id=1, reward_task=0.0)

    c_losses, a_losses = [], []
    for _ in range(600):
        cl, al = agent.train_step(buf.sample(256))
        c_losses.append(cl)
        if al is not None:
            a_losses.append(al)
    assert np.isfinite(c_losses).all() and np.isfinite(a_losses).all()
    print(f"  critic_loss {c_losses[0]:.3f} -> {c_losses[-1]:.4f}(下降)")
    assert c_losses[-1] < c_losses[0]

    # 查 Q 是否学到 reward 路由
    with torch.no_grad():
        o = torch.as_tensor(fixed_obs, device=agent.device).unsqueeze(0)
        a = torch.zeros(1, 2, device=agent.device)
        w0 = agent.actor.codebook[0].unsqueeze(0)
        w1 = agent.actor.codebook[1].unsqueeze(0)
        q_w0 = float(agent.critic.Q1(o, w0, a))
        q_w1 = float(agent.critic.Q1(o, w1, a))
    print(f"  Q1(obs,w_0,a)={q_w0:.3f}(期望~1.0)  Q1(obs,w_1,a)={q_w1:.3f}(期望~0.0)")
    assert abs(q_w0 - 1.0) < 0.2, "w_0 未学到 reward_task=1"
    assert abs(q_w1 - 0.0) < 0.2, "w_1 未学到 reward=0"
    assert q_w0 - q_w1 > 0.5, "critic 未按 w 区分价值"

    # act 接口
    act = agent.act(fixed_obs, skill_id=0, noise=0.0)
    assert act.shape == (2,) and np.all(np.abs(act) <= 1.0)
    print("  act 接口 OK")
    print("ALL PASS")


if __name__ == "__main__":
    main()
