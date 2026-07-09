import torch
import torch.nn as nn


def build_codebook(K: int, d_w: int, codebook_type: str = "one_hot", seed: int = 0) -> torch.Tensor:
    """Build a fixed (non-trainable) skill codebook of shape [K+1, d_w].

    Row 0 is the deployment skill ``w_0``; rows 1..K are the probe skills
    ``W_probe = {w_1, .., w_K}``. Stage 1 uses a one-hot codebook so the skill
    codes are equidistant and塌缩排查无噪声(see design.md 组件/Data Models).

    - ``one_hot``: identity matrix, ``d_w == K + 1``.
    - ``random``:  fixed N(0, I) rows, L2-normalized, ``d_w`` free (>=1).
    """
    num_skills = int(K) + 1
    codebook_type = str(codebook_type).lower()
    if num_skills <= 0:
        raise ValueError("K must be >= 0")
    if int(d_w) <= 0:
        raise ValueError("d_w must be positive")

    if codebook_type == "one_hot":
        if int(d_w) != num_skills:
            raise ValueError(
                f"one_hot codebook requires d_w == K+1 ({num_skills}), got d_w={d_w}"
            )
        return torch.eye(num_skills, dtype=torch.float32)

    if codebook_type == "random":
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        w = torch.randn(num_skills, int(d_w), generator=gen, dtype=torch.float32)
        return w / (w.norm(dim=1, keepdim=True) + 1e-8)

    raise ValueError(f"Unknown codebook_type: {codebook_type}")


def resolve_d_w(skill_enabled: bool, K: int, codebook_type: str = "one_hot", d_w=None) -> int:
    """Resolve the skill-code width used to augment the decoder input.

    Returns 0 when skills are disabled (strict-DIVO / mode A).
    """
    if not skill_enabled:
        return 0
    codebook_type = str(codebook_type).lower()
    if codebook_type == "one_hot":
        return int(K) + 1
    if d_w is None:
        raise ValueError("d_w must be provided for non one_hot codebook_type")
    return int(d_w)


class LatentDetPolicy(nn.Module):
    """Deterministic latent policy, optionally skill-conditioned.

    Strict-DIVO (mode A, ``skill_enabled=False``):
        ``a = decoder(cat([state, z]))`` where ``z = encoder(obs)``.
    Skill-conditioned (mode B, ``skill_enabled=True``):
        ``a = decoder(cat([state, z, w]))`` where ``w`` is a fixed skill code.
        Deployment uses the fixed deployment skill ``w_0`` (skill_id 0);
        ``z = encoder(obs)`` stays the only obstacle/context channel and does
        NOT carry any skill information.
    """

    def __init__(
        self,
        encoder,
        decoder,
        obs2state,
        skill_enabled: bool = False,
        K: int = 0,
        d_w=None,
        codebook_type: str = "one_hot",
        codebook_seed: int = 0,
        *args,
        **kwargs
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.obs2state = obs2state

        self.skill_enabled = bool(skill_enabled)
        self.K = int(K)
        self.codebook_type = str(codebook_type)

        if self.skill_enabled:
            self.d_w = resolve_d_w(True, self.K, self.codebook_type, d_w)
            # Constant (non-trainable) codebook, persisted with the module so the
            # deployment/probe codes are reproducible across load/save.
            self.register_buffer(
                "codebook",
                build_codebook(self.K, self.d_w, self.codebook_type, seed=int(codebook_seed)),
            )
        else:
            self.d_w = 0

    @property
    def num_skills(self) -> int:
        """Total number of skill slots (w_0 plus K probe skills)."""
        return self.K + 1 if self.skill_enabled else 0

    def skill_code(self, skill_id: int) -> torch.Tensor:
        """Return the fixed skill code row for ``skill_id`` (0 == w_0)."""
        if not self.skill_enabled:
            raise RuntimeError("skill_code requires skill_enabled=True")
        return self.codebook[int(skill_id)]

    def decode_with_skill(self, state, z, w):
        """Decode an action from state, latent z, and skill code w.

        ``w`` may be a single code ``[d_w]`` (broadcast to the batch) or a
        per-sample batch ``[B, d_w]``.
        """
        if w.dim() == 1:
            w = w.unsqueeze(0).expand(state.shape[0], -1)
        return self.decoder(torch.cat([state, z, w], dim=1))

    def predict_action(self, obs, skill_id=None):
        z = self.encoder(obs)
        state = self.obs2state(obs)
        if not self.skill_enabled:
            # Strict DIVO (mode A): decoder input carries no skill code.
            return self.decoder(torch.cat([state, z], dim=1))
        # Mode B: default to the deployment skill w_0 (skill_id 0).
        if skill_id is None:
            skill_id = 0
        w = self.skill_code(skill_id)
        return self.decode_with_skill(state, z, w)

    def predict_action_with_skill(self, obs, w):
        """Predict an action for an explicit skill code ``w`` (used in training/probe)."""
        if not self.skill_enabled:
            raise RuntimeError("predict_action_with_skill requires skill_enabled=True")
        z = self.encoder(obs)
        state = self.obs2state(obs)
        return self.decode_with_skill(state, z, w)

    def reset(self):
        pass
