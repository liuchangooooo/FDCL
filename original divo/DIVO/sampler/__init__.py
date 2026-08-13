from DIVO.sampler.fm import FlowMatching
from DIVO.nets import get_net

def get_sampler(**sampler_cfg):
    target = sampler_cfg["_target_"]
    if target == 'flowmodel':
        sampler = FlowMatching(
            get_net(**sampler_cfg['velocity_field_net']),
            **sampler_cfg
        )
    else:
        raise NotImplementedError(f"Sampler type {target} not implemented.")

    return sampler