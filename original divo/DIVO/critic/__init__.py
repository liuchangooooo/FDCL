from DIVO.nets import get_net
import torchvision.models as models
from torch.nn import Sequential
import torch.nn as nn
import torch
from DIVO.critic.mcritic import (
    SingleFCCritic, 
    MultiCritic,
)

def get_critic(**critic_cfg):
    target = critic_cfg['_target_']
    if target == 'mcritic':
        q_networks = []
        for i in range(critic_cfg['n_critics']):
            subnet_cfg = critic_cfg[f"net{i}"]
            if 'fc' in subnet_cfg['_target_']:
                if subnet_cfg['_target_'] == 'fc_vec':
                    net = get_net(**critic_cfg[f"net{i}"])
                q_networks.append(
                    SingleFCCritic(net))
        critic = MultiCritic(q_networks)
    
    else:
        raise NotImplementedError(f"Critic type {target} not implemented.")
    return critic

