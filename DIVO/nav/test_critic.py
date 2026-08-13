"""任务 5 组件 sanity:技能条件 critic Q(obs,w,a)。

验证:
  1) 形状:obs(44)+w(d_w)+a(2) -> (q1,q2) 各 [B,1]
  2) w 条件性:同 (obs,a) 下不同 w 给出不同 Q(critic 以 w 为条件)
  3) 与 SkillActor 联动:Q1(obs, w, actor(obs,w)) 可反传到 actor(actor loss 通路)
  4) 梯度流正常
"""
import numpy as np
import torch

from nav import nav_env as NE
from nav.policy import SkillActor
from nav.td3 import SkillCritic


def main():
    torch.manual_seed(0)
    K = 4
    d_w = K + 1
    B = 6
    actor = SkillActor(K=K)
    critic = SkillCritic(NE.OBS_DIM, 2, d_w)

    obs = torch.randn(B, NE.OBS_DIM)
    a = torch.randn(B, 2).clamp(-1, 1)

    # 1) 形状
    w0 = critic_w(critic, actor, 0, B)
    q1, q2 = critic(obs, w0, a)
    assert q1.shape == (B, 1) and q2.shape == (B, 1)
    print(f"  1) Q 形状 q1={tuple(q1.shape)} q2={tuple(q2.shape)} OK")

    # 2) w 条件性:同 (obs,a) 不同 w -> 不同 Q
    qs = []
    for k in range(d_w):
        wk = critic_w(critic, actor, k, B)
        qs.append(critic.Q1(obs, wk, a))
    qs = torch.stack(qs, 0)  # [d_w, B, 1]
    diffs = [(qs[i] - qs[j]).abs().mean().item()
             for i in range(d_w) for j in range(i + 1, d_w)]
    print(f"  2) 不同 w 的 Q 平均差 min={min(diffs):.4f} max={max(diffs):.4f}(>0 => w 条件生效)")
    assert min(diffs) > 1e-4

    # 3) actor loss 通路:-Q1(obs, w, actor(obs,w)) 可反传到 actor
    wk = critic_w(critic, actor, 1, B)
    a_pi = actor(obs, skill_id=1)
    actor_loss = -critic.Q1(obs, wk, a_pi).mean()
    actor_loss.backward()
    assert actor.decoder[0].weight.grad is not None
    print("  3) actor loss 通路 -Q1(obs,w,π(obs,w)) 反传到 actor OK")

    # 4) critic 自身梯度
    critic.zero_grad()
    q1, q2 = critic(obs, w0, a)
    (q1.mean() + q2.mean()).backward()
    assert critic.q1[0].weight.grad is not None and critic.q2[0].weight.grad is not None
    print("  4) critic 梯度流 OK")
    print("ALL PASS")


def critic_w(critic, actor, skill_id, batch):
    """取 one-hot w 行(用 actor 的 codebook)。"""
    return actor.codebook[skill_id].unsqueeze(0).expand(batch, -1)


if __name__ == "__main__":
    main()
