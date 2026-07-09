from omegaconf import OmegaConf
from DIVO.nets import get_net
import torchvision.models as models
from torch.nn import Sequential
import torch.nn as nn
import torch
from DIVO.critic.mcritic import (
    SingleFCCritic, 
    MultiCritic,
)
from DIVO.policy.ldpi import resolve_d_w


def _to_net_cfg(net_cfg):
    if OmegaConf.is_config(net_cfg):
        return OmegaConf.to_container(net_cfg, resolve=True)
    return dict(net_cfg)


def get_critic(**critic_cfg):
    target = critic_cfg['_target_']
    if target == 'mcritic':
        # When skills are enabled the critic is w-conditioned Q(obs, w, a); widen
        # each subnet input by d_w. Defaults keep strict-DIVO behavior (d_w=0).
        skill_enabled = bool(critic_cfg.get('skill_enabled', False))
        K = int(critic_cfg.get('K', 0))
        codebook_type = critic_cfg.get('codebook_type', 'one_hot')
        d_w = resolve_d_w(skill_enabled, K, codebook_type, critic_cfg.get('d_w'))

        q_networks = []
        for i in range(critic_cfg['n_critics']):
            subnet_cfg = critic_cfg[f"net{i}"]
            if 'fc' in subnet_cfg['_target_']:
                if subnet_cfg['_target_'] == 'fc_vec':
                    net_cfg = _to_net_cfg(subnet_cfg)
                    if skill_enabled:
                        net_cfg['in_chan'] = int(net_cfg['in_chan']) + d_w
                    net = get_net(**net_cfg)
                q_networks.append(
                    SingleFCCritic(net))
        critic = MultiCritic(q_networks)
    
    else:
        raise NotImplementedError(f"Critic type {target} not implemented.")
    return critic

