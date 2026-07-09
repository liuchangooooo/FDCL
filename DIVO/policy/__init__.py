from omegaconf import OmegaConf

from DIVO.policy.ldpi import LatentDetPolicy, resolve_d_w
from DIVO.nets import get_net


def _to_net_cfg(net_cfg):
    """Return a mutable plain-dict copy of a net config (handles OmegaConf)."""
    if OmegaConf.is_config(net_cfg):
        return OmegaConf.to_container(net_cfg, resolve=True)
    return dict(net_cfg)


def get_policy(env, **policy_cfg):
    target = policy_cfg["_target_"]
    if target == "ldpi":
        skill_enabled = bool(policy_cfg.get("skill_enabled", False))
        K = int(policy_cfg.get("K", 0))
        codebook_type = policy_cfg.get("codebook_type", "one_hot")
        codebook_seed = int(policy_cfg.get("codebook_seed", 0))
        d_w = resolve_d_w(skill_enabled, K, codebook_type, policy_cfg.get("d_w"))

        decoder_cfg = _to_net_cfg(policy_cfg["decoder_net"])
        if skill_enabled:
            # Widen the decoder input by d_w so it can consume [state, z, w].
            decoder_cfg["in_chan"] = int(decoder_cfg["in_chan"]) + d_w

        policy = LatentDetPolicy(
            get_net(**policy_cfg["encoder_net"]),
            get_net(**decoder_cfg),
            env.obs2state,
            skill_enabled=skill_enabled,
            K=K,
            d_w=d_w,
            codebook_type=codebook_type,
            codebook_seed=codebook_seed,
        )
    else:
        raise NotImplementedError(f"Policy type {target} not implemented.")
    return policy
