from DIVO.nets.modules import FC_vec, vf_FC_vec
from DIVO.nets.resnet import get_resnet

def get_net(**net_cfg):
    target = net_cfg["_target_"]
    if target == 'fc_vec':
        net = FC_vec(**net_cfg)
    elif target == 'vf_fc_vec':
        net = vf_FC_vec(**net_cfg)
    elif target == 'resnet':
        net = get_resnet(**net_cfg)
    return net
    