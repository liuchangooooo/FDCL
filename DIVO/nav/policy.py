"""技能条件确定性策略(导航版,与 Push-T 隐技能库结构一致)。

结构(对齐 Push-T ldpi:encoder 吃完整 obs,state 另走直连):
    state = obs[0:28]                       # 本体 + goal_lidar(decoder 的 state 通道,直连)
    z     = z_encoder(obs[0:44])            # 完整 obs 的低维确定性压缩(对齐 Push-T z=encoder(obs))
    a     = decoder(cat([state, z, w]))     # 2 维 tanh 动作
    w     ∈ codebook[K+1, d_w]              # one-hot,索引 0=w_0(部署), 1..K=W_probe

- codebook 为常量 one-hot,不参与梯度(Req 1.4)。
- z 是确定性编码(非采样),保 motivation(Req 1.2)。
- encoder 吃完整 obs(与 Push-T `z=encoder(obs)` 一致);state 作为已知量直连 decoder,
  z 学的是全 obs 的低维残差码(其中承载障碍上下文);w 是技能选择器不承载障碍信息。
- predict_action(obs) 默认拼 w_0 部署;predict_action_with_skill / decode_with_skill 供训练与 probe。
"""
import numpy as np
import torch
import torch.nn as nn

from nav import nav_env as NE


def _mlp(sizes, act=nn.ReLU, out_act=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1]),
                   act() if i < len(sizes) - 2 else out_act()]
    return nn.Sequential(*layers)


class SkillActor(nn.Module):
    """确定性技能条件策略。obs(44) + w(one-hot d_w) -> action(2), tanh。"""

    def __init__(self, K=4, d_z=8, hidden=(256, 256), enc_hidden=(64,),
                 codebook_type="one_hot", d_w=None, codebook_seed=0):
        super().__init__()
        self.K = int(K)
        self.codebook_type = str(codebook_type)
        self.d_z = int(d_z)
        self.state_slice = NE.STATE_SLICE       # 0:28(decoder 的 state 通道)
        self.obs_dim = NE.OBS_DIM               # 44(encoder 吃完整 obs,对齐 Push-T)
        self.state_dim = self.state_slice.stop - self.state_slice.start   # 28
        self.act_dim = 2

        # 技能码本(常量、不参与梯度):one_hot(d_w=K+1) 或 random(d_w 与 K 解耦,N(0,I) 归一化)
        if self.codebook_type == "one_hot":
            self.d_w = self.K + 1
            codebook = torch.eye(self.d_w, dtype=torch.float32)              # [K+1, d_w]
        elif self.codebook_type == "random":
            self.d_w = int(d_w) if d_w else self.K + 1
            g = torch.Generator().manual_seed(int(codebook_seed))
            cb = torch.randn(self.K + 1, self.d_w, generator=g, dtype=torch.float32)
            codebook = cb / (cb.norm(dim=1, keepdim=True) + 1e-8)            # L2 归一化
        else:
            raise ValueError(f"unknown codebook_type: {self.codebook_type}")
        self.register_buffer("codebook", codebook)

        # z-encoder:完整 obs(44) -> z(d_z),确定性(对齐 Push-T z=encoder(obs))
        self.z_encoder = _mlp([self.obs_dim, *enc_hidden, self.d_z])
        # decoder:[state, z, w] -> action
        dec_in = self.state_dim + self.d_z + self.d_w
        self.decoder = _mlp([dec_in, *hidden, self.act_dim], out_act=nn.Tanh)

    # ---------------- 组合式前向 ----------------
    def encode_z(self, obs):
        return self.z_encoder(obs)              # 完整 obs(对齐 Push-T z=encoder(obs))

    def _w_vec(self, skill_id, batch, device):
        """取 skill_id 对应的 one-hot 行,广播到 batch。skill_id: int 或 [B] tensor。"""
        if isinstance(skill_id, int):
            w = self.codebook[skill_id].to(device)
            return w.unsqueeze(0).expand(batch, -1)
        skill_id = skill_id.to(device).long().view(-1)
        return self.codebook.to(device)[skill_id]   # [B, d_w]

    def decode_with_skill(self, state, z, w):
        return self.decoder(torch.cat([state, z, w], dim=-1))

    def forward(self, obs, skill_id=0):
        """obs: [B,44]。skill_id: int(全 batch 同技能) 或 [B]。返回动作 [B,2]。"""
        state = obs[..., self.state_slice]
        z = self.encode_z(obs)
        w = self._w_vec(skill_id, obs.shape[0], obs.device)
        return self.decode_with_skill(state, z, w)

    # ---------------- 便捷接口 ----------------
    @torch.no_grad()
    def predict_action(self, obs, skill_id=0):
        """默认拼 w_0(skill_id=0)部署;接受 np 或 tensor,返回 np 动作。"""
        single = False
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(np.asarray(obs, np.float32))
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            single = True
        obs = obs.to(self.codebook.device).float()
        a = self.forward(obs, skill_id=skill_id)
        a = a.cpu().numpy()
        return a[0] if single else a

    @torch.no_grad()
    def predict_action_with_skill(self, obs, skill_id):
        return self.predict_action(obs, skill_id=skill_id)
