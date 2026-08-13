from DIVO.policy.ldpi import LatentDetPolicy
from DIVO.nets import get_net

def get_policy(env, **policy_cfg):
    target = policy_cfg["_target_"]
    if target == "ldpi":
        policy = LatentDetPolicy(
            get_net(**policy_cfg['encoder_net']),
            get_net(**policy_cfg['decoder_net']),
            env.obs2state,
            **policy_cfg
        )
    else:
        raise NotImplementedError(f"Policy type {target} not implemented.")
    return policy