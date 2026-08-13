"""技能库版 TD3 update:w-条件 critic + executed-w-only actor 损失。

要点(Req 2/3):
- critic:target/loss 用**该转移自己的 w 标签**(codebook[skill_id]);目标 Q 用
  reward_for_critic(w_0→reward_task,probe→reward_total)。
- actor(delayed):`L_actor = L_exec (+ λ_z·L_z)`,
  `L_exec = -mean Q1(obs, w_exec, π(obs, w_exec))`,w_exec = 该转移实际执行过的技能。
  => 只在实际执行过的 (obs, w) 上更新,critic 不对未执行组合外推(executed-w-only,
     实现方式是"每条转移只用它自己的 skill_id",天然不出现 all-w 覆盖项)。
- L_z:可选,对 z 做 gaussian 正则(batch 均值/方差拉向 N(0,1));默认关(λ_z=0)。
"""
import numpy as np
import torch
import torch.nn.functional as F

from nav.policy import SkillActor
from nav.td3 import SkillCritic


class SkillTD3:
    def __init__(self, obs_dim, act_dim, K=4, device="cuda",
                 gamma=0.99, tau=0.005, actor_lr=3e-4, critic_lr=3e-4,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2,
                 d_z=8, hidden=(256, 256), lambda_z=1.0,
                 codebook_type="one_hot", d_w=None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.K = int(K)
        self.act_dim = int(act_dim)
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.lambda_z = float(lambda_z)
        self._it = 0

        self.actor = SkillActor(K=K, d_z=d_z, hidden=hidden,
                                codebook_type=codebook_type, d_w=d_w).to(self.device)
        self.actor_t = SkillActor(K=K, d_z=d_z, hidden=hidden,
                                  codebook_type=codebook_type, d_w=d_w).to(self.device)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.d_w = self.actor.d_w              # 实际码宽(random 时可 != K+1)
        self.critic = SkillCritic(obs_dim, act_dim, self.d_w, hidden=hidden).to(self.device)
        self.critic_t = SkillCritic(obs_dim, act_dim, self.d_w, hidden=hidden).to(self.device)
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def _w(self, skill_id):
        """skill_id [B] long -> one-hot w [B, d_w]。"""
        return self.actor.codebook[skill_id.long().view(-1)]

    @torch.no_grad()
    def act(self, obs, skill_id=0, noise=0.0):
        o = torch.as_tensor(np.asarray(obs, np.float32), device=self.device).unsqueeze(0)
        a = self.actor(o, skill_id=int(skill_id)).cpu().numpy()[0]
        if noise > 0:
            a = a + noise * np.random.randn(self.act_dim)
        return np.clip(a, -1.0, 1.0)

    def _gaussian_z_reg(self, z):
        # 对齐 DIVO:F.mse_loss(z_mean,0)+F.mse_loss(z_var,1)(逐维均值),使 lambda_z 与 reg_coeff 同尺度
        mu = z.mean(0)
        var = z.var(0, unbiased=False)
        return mu.pow(2).mean() + (var - 1.0).pow(2).mean()

    def train_step(self, batch):
        self._it += 1
        obs = batch["obs"]
        act = batch["act"]
        next_obs = batch["next_obs"]
        done = batch["done"]
        skill_id = batch["skill_id"]
        r_critic = batch["reward_for_critic"]          # w_0→task, probe→total
        w = self._w(skill_id)                          # 该转移自己的 w

        # ---- critic 更新(带该转移 w 标签)----
        with torch.no_grad():
            noise = (torch.randn_like(act) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_a = (self.actor_t(next_obs, skill_id=skill_id) + noise).clamp(-1.0, 1.0)
            q1_t, q2_t = self.critic_t(next_obs, w, next_a)
            target = r_critic + self.gamma * (1 - done) * torch.min(q1_t, q2_t)
        q1, q2 = self.critic(obs, w, act)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss_val = None
        if self._it % self.policy_delay == 0:
            # ---- executed-w-only actor 更新 ----
            state = obs[..., self.actor.state_slice]
            z = self.actor.encode_z(obs)
            a_pi = self.actor.decode_with_skill(state, z, w)     # π(obs, w_exec)
            l_exec = -self.critic.Q1(obs, w, a_pi).mean()
            actor_loss = l_exec
            if self.lambda_z > 0:
                actor_loss = actor_loss + self.lambda_z * self._gaussian_z_reg(z)
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            actor_loss_val = float(actor_loss.item())
            self._soft_update()
        return float(critic_loss.item()), actor_loss_val

    def _soft_update(self):
        for p, pt in zip(self.actor.parameters(), self.actor_t.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
        for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
            pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict(), "K": self.K}, path)

    def load(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_t.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_t.load_state_dict(ckpt["critic"])
