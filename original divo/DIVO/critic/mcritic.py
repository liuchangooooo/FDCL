import torch as th
from typing import List, Tuple, Type, NamedTuple, Union, Dict
import torch.nn as nn

class SingleFCCritic(nn.Module):
    def __init__(self,
                 q_network,
                 *args,
                 **kwargs):
        super().__init__()
        self.q_net = q_network

    def forward(self, obs, actions):
        return self.q_net(th.cat([obs, actions], dim=1))
        
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

    def forward(self, obs, actions) -> Tuple[th.Tensor, ...]:
        return [q_net(obs, actions) for q_net in self.q_networks]

    
