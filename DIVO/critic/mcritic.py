import torch as th
from typing import List, Tuple, Type, NamedTuple, Union, Dict
import torch.nn as nn


class SingleFCCritic(nn.Module):
    """Q-network, optionally skill-conditioned.

    - ``w is None`` (strict DIVO / mode A): ``Q(obs, a) = q_net([obs, a])``.
    - ``w`` given (mode B):                  ``Q(obs, w, a) = q_net([obs, w, a])``.

    The ``w`` argument is placed between obs and actions so the input layout is
    ``[obs, w, actions]`` (matches design.md 组件 2). Backward compatible: old
    callers using ``critic(obs, actions)`` keep working unchanged.
    """

    def __init__(self,
                 q_network,
                 *args,
                 **kwargs):
        super().__init__()
        self.q_net = q_network

    def forward(self, obs, actions, w=None):
        if w is None:
            return self.q_net(th.cat([obs, actions], dim=1))
        if w.dim() == 1:
            w = w.unsqueeze(0).expand(obs.shape[0], -1)
        return self.q_net(th.cat([obs, w, actions], dim=1))


class MultiCritic(nn.Module):
    def __init__(self,
                 q_networks,
                 *args,
                 **kwargs):
        super().__init__()
        self.q_networks = []
        for idx, q_net in enumerate(q_networks):
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs, actions, w=None) -> Tuple[th.Tensor, ...]:
        return [q_net(obs, actions, w) for q_net in self.q_networks]
