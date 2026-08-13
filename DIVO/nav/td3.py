"""最简确定性 TD3(单策略)—— 用于导航 motivation 验证。

刻意保持最小:确定性 actor(tanh)+ twin critic + target policy smoothing +
delayed policy update。不含任何技能库/多样性,单确定性策略。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes, act=nn.ReLU, out_act=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1]),
                   act() if i < len(sizes) - 2 else out_act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, act_dim], out_act=nn.Tanh)

    def forward(self, obs):
        return self.net(obs)  # 动作范围 [-1,1]


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim, *hidden, 1])
        self.q2 = mlp([obs_dim + act_dim, *hidden, 1])

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)

    def Q1(self, obs, act):
        return self.q1(torch.cat([obs, act], dim=-1))


class SkillCritic(nn.Module):
    """技能条件 twin critic:Q(obs, w, a),内部 q(cat([obs, w, a]))。

    w 为 one-hot 技能码 [B, d_w];不同技能在同一 (obs,a) 下价值可不同。
    """

    def __init__(self, obs_dim, act_dim, d_w, hidden=(256, 256)):
        super().__init__()
        self.d_w = int(d_w)
        in_dim = obs_dim + self.d_w + act_dim
        self.q1 = mlp([in_dim, *hidden, 1])
        self.q2 = mlp([in_dim, *hidden, 1])

    def forward(self, obs, w, act):
        x = torch.cat([obs, w, act], dim=-1)
        return self.q1(x), self.q2(x)

    def Q1(self, obs, w, act):
        return self.q1(torch.cat([obs, w, act], dim=-1))


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, device):
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.next_obs = np.zeros((size, obs_dim), np.float32)
        self.act = np.zeros((size, act_dim), np.float32)
        self.rew = np.zeros((size, 1), np.float32)
        self.done = np.zeros((size, 1), np.float32)
        self.max_size = size
        self.ptr = 0
        self.n = 0
        self.device = device

    def add(self, o, a, r, no, d):
        i = self.ptr
        self.obs[i] = o
        self.act[i] = a
        self.rew[i] = r
        self.next_obs[i] = no
        self.done[i] = d
        self.ptr = (self.ptr + 1) % self.max_size
        self.n = min(self.n + 1, self.max_size)

    def sample(self, batch):
        idx = np.random.randint(0, self.n, size=batch)
        to = lambda x: torch.as_tensor(x[idx], device=self.device)
        return to(self.obs), to(self.act), to(self.rew), to(self.next_obs), to(self.done)


class SkillReplayBuffer:
    """技能库版 replay buffer:每条转移带 skill_id 与三列奖励。

    列:obs / act / next_obs / done / skill_id / reward_task / reward_div / reward_total / source
    reward_for_critic 路由:w_0(skill_id==0)用 reward_task,W_probe(>=1)用 reward_total
    => Q(obs, w_0, ·) 为纯任务价值(部署不被多样性污染)。
    source: 0=normal, 1=diversity_paired。
    """

    SRC_NORMAL = 0
    SRC_DIVERSITY = 1

    def __init__(self, obs_dim, act_dim, size, device):
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.next_obs = np.zeros((size, obs_dim), np.float32)
        self.act = np.zeros((size, act_dim), np.float32)
        self.done = np.zeros((size, 1), np.float32)
        self.skill_id = np.zeros((size,), np.int64)
        self.reward_task = np.zeros((size, 1), np.float32)
        self.reward_div = np.zeros((size, 1), np.float32)
        self.reward_total = np.zeros((size, 1), np.float32)
        self.source = np.zeros((size,), np.int64)
        self.max_size = size
        self.ptr = 0
        self.n = 0
        self.device = device

    def add(self, o, a, no, d, skill_id, reward_task,
            reward_div=0.0, beta_div=0.0, source=SRC_NORMAL):
        i = self.ptr
        self.obs[i] = o
        self.act[i] = a
        self.next_obs[i] = no
        self.done[i] = d
        sid = int(skill_id)
        self.skill_id[i] = sid
        rt = float(reward_task)
        self.reward_task[i] = rt
        rd = float(reward_div)
        self.reward_div[i] = rd
        # w_0 回合不加 r_div;probe 技能加 beta_div*r_div
        self.reward_total[i] = rt if sid == 0 else rt + float(beta_div) * rd
        self.source[i] = int(source)
        self.ptr = (self.ptr + 1) % self.max_size
        self.n = min(self.n + 1, self.max_size)

    def _reward_for_critic(self, idx):
        """路由:w_0 用 reward_task,W_probe 用 reward_total。"""
        rt = self.reward_task[idx]
        rtot = self.reward_total[idx]
        is_w0 = (self.skill_id[idx] == 0)[:, None]
        return np.where(is_w0, rt, rtot).astype(np.float32)

    def sample(self, batch):
        idx = np.random.randint(0, self.n, size=batch)
        t = lambda x: torch.as_tensor(x[idx], device=self.device)
        return {
            "obs": t(self.obs),
            "act": t(self.act),
            "next_obs": t(self.next_obs),
            "done": t(self.done),
            "skill_id": t(self.skill_id),                 # [B] long
            "reward_task": t(self.reward_task),
            "reward_total": t(self.reward_total),
            "reward_for_critic": torch.as_tensor(self._reward_for_critic(idx), device=self.device),
            "source": t(self.source),
        }


class TD3:
    def __init__(self, obs_dim, act_dim, device="cuda",
                 gamma=0.99, tau=0.005, actor_lr=3e-4, critic_lr=3e-4,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2, hidden=(256, 256)):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.actor = Actor(obs_dim, act_dim, hidden).to(self.device)
        self.actor_t = Actor(obs_dim, act_dim, hidden).to(self.device)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic = Critic(obs_dim, act_dim, hidden).to(self.device)
        self.critic_t = Critic(obs_dim, act_dim, hidden).to(self.device)
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.act_dim = act_dim
        self._it = 0

    @torch.no_grad()
    def act(self, obs, noise=0.0):
        """确定性动作(可加高斯探索噪声)。"""
        o = torch.as_tensor(np.asarray(obs, np.float32), device=self.device).unsqueeze(0)
        a = self.actor(o).cpu().numpy()[0]
        if noise > 0:
            a = a + noise * np.random.randn(self.act_dim)
        return np.clip(a, -1.0, 1.0)

    def train_step(self, buf, batch=256):
        self._it += 1
        obs, act, rew, next_obs, done = buf.sample(batch)

        with torch.no_grad():
            noise = (torch.randn_like(act) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_act = (self.actor_t(next_obs) + noise).clamp(-1.0, 1.0)
            q1_t, q2_t = self.critic_t(next_obs, next_act)
            q_t = torch.min(q1_t, q2_t)
            target = rew + self.gamma * (1 - done) * q_t

        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss_val = None
        if self._it % self.policy_delay == 0:
            actor_loss = -self.critic.Q1(obs, self.actor(obs)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            actor_loss_val = float(actor_loss.item())
            for p, pt in zip(self.actor.parameters(), self.actor_t.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return float(critic_loss.item()), actor_loss_val

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict()}, path)

    def load(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_t.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_t.load_state_dict(ckpt["critic"])
