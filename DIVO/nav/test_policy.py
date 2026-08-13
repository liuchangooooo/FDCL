"""任务 4 组件 sanity:技能条件确定性策略。

验证:
  1) 形状:obs(44)+w -> action(2),tanh 有界
  2) codebook one-hot、不参与梯度
  3) encoder 吃完整 obs(对齐 Push-T z=encoder(obs)):改 obs 任一段都会改变 z
  4) z 不依赖 w(z=encoder(obs),w 是独立技能选择器)
  5) 同 obs 下不同 w 给出不同动作(技能可分)
  6) predict_action 默认 w_0;确定性(两次一致)
"""
import numpy as np
import torch

from nav import nav_env as NE
from nav.policy import SkillActor


def main():
    torch.manual_seed(0)
    K = 4
    actor = SkillActor(K=K, d_z=8)
    B = 5
    obs = torch.randn(B, NE.OBS_DIM)

    # 1) 形状 + tanh
    a = actor(obs, skill_id=0)
    assert a.shape == (B, 2), a.shape
    assert torch.all(a.abs() <= 1.0 + 1e-6)
    print(f"  1) action shape={tuple(a.shape)} in [-1,1] OK")

    # 2) codebook one-hot + no grad
    cb = actor.codebook
    assert cb.shape == (K + 1, K + 1)
    assert torch.allclose(cb, torch.eye(K + 1))
    assert cb.requires_grad is False
    print(f"  2) codebook one-hot [{K+1},{K+1}], requires_grad=False OK")

    # 3) encoder 吃完整 obs:改 state 段 或 障碍段 都会改变 z(对齐 Push-T)
    z0 = actor.encode_z(obs)
    obs_state_changed = obs.clone(); obs_state_changed[:, NE.STATE_SLICE] += 3.0
    z_after_state = actor.encode_z(obs_state_changed)
    assert not torch.allclose(z0, z_after_state), "encoder 吃全 obs:改 state 段应改变 z"
    obs_z_changed = obs.clone(); obs_z_changed[:, NE.Z_INPUT_SLICE] += 3.0
    z_after_z = actor.encode_z(obs_z_changed)
    assert not torch.allclose(z0, z_after_z), "改 pillars_lidar 应改变 z"
    print("  3) z=encoder(完整 obs):改 state/障碍段均改变 z OK(对齐 Push-T)")

    # 4) z 不依赖 w(z=encoder(obs) 与 skill 无关)
    z_w0 = actor.encode_z(obs)
    assert torch.allclose(z0, z_w0), "z 不应依赖 w"
    print("  4) z 独立于 w OK(w 仅作技能选择器,不承载障碍信息)")

    # 5) 同 obs 不同 w -> 不同动作
    acts = torch.stack([actor(obs, skill_id=k) for k in range(K + 1)], dim=0)  # [K+1,B,2]
    diffs = []
    for i in range(K + 1):
        for j in range(i + 1, K + 1):
            diffs.append((acts[i] - acts[j]).abs().mean().item())
    print(f"  5) 不同 w 动作平均差 min={min(diffs):.4f} max={max(diffs):.4f}(>0 => 技能可分)")
    assert min(diffs) > 1e-4

    # 6) predict_action 默认 w_0 + 确定性
    o1 = np.random.randn(NE.OBS_DIM).astype(np.float32)
    a1 = actor.predict_action(o1)
    a2 = actor.predict_action(o1)
    a1w0 = actor.predict_action(o1, skill_id=0)
    assert a1.shape == (2,)
    assert np.allclose(a1, a2) and np.allclose(a1, a1w0)
    print("  6) predict_action 默认 w_0、确定性 OK")

    # 梯度只经 encoder/decoder,不经 codebook
    a = actor(obs, skill_id=1).sum()
    a.backward()
    assert actor.codebook.grad is None
    assert actor.z_encoder[0].weight.grad is not None
    print("  梯度流:encoder/decoder 有梯度、codebook 无梯度 OK")

    print("ALL PASS")


if __name__ == "__main__":
    main()
