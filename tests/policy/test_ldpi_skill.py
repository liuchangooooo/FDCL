"""Unit tests for skill-conditioned LatentDetPolicy (Task 1).

Fast CPU-only tests (no RL, no MuJoCo). They pin down:
  - w channel does not affect the z computation (encoder output);
  - strict-DIVO mode (skill_enabled=False) keeps the old decoder input width
    and output shape;
  - predict_action(obs) is equivalent to decoding with the deployment skill w_0;
  - get_policy widens the decoder input by d_w when skills are enabled.
"""

import torch

from DIVO.nets.modules import FC_vec
from DIVO.policy import get_policy
from DIVO.policy.ldpi import LatentDetPolicy, build_codebook, resolve_d_w

OBS_DIM = 8
STATE_DIM = 4
LATENT_DIM = 3
ACTION_SIZE = 6
K = 4


def _obs2state(obs):
    return obs[:, :STATE_DIM]


def _encoder():
    return FC_vec(
        in_chan=OBS_DIM,
        out_chan=LATENT_DIM,
        l_hidden=[16],
        activation=["relu"],
        out_activation="linear",
    )


def _decoder(in_chan):
    return FC_vec(
        in_chan=in_chan,
        out_chan=ACTION_SIZE,
        l_hidden=[16],
        activation=["relu"],
        out_activation="tanh",
    )


def _make_policy(skill_enabled):
    torch.manual_seed(0)
    d_w = resolve_d_w(skill_enabled, K, "one_hot")
    decoder_in = STATE_DIM + LATENT_DIM + d_w
    return LatentDetPolicy(
        _encoder(),
        _decoder(decoder_in),
        _obs2state,
        skill_enabled=skill_enabled,
        K=K,
        codebook_type="one_hot",
    )


class _FakeEnv:
    def obs2state(self, obs):
        return obs[:, :STATE_DIM]


def test_codebook_is_one_hot_and_fixed():
    cb = build_codebook(K, K + 1, "one_hot")
    assert cb.shape == (K + 1, K + 1)
    assert torch.allclose(cb, torch.eye(K + 1))


def test_strict_divo_mode_shape_and_no_codebook():
    policy = _make_policy(skill_enabled=False)
    assert policy.skill_enabled is False
    assert policy.d_w == 0
    assert not hasattr(policy, "codebook")
    # strict-DIVO decoder input width is state + z, unchanged from old DIVO.
    assert policy.decoder.in_chan == STATE_DIM + LATENT_DIM

    obs = torch.randn(5, OBS_DIM)
    action = policy.predict_action(obs)
    assert action.shape == (5, ACTION_SIZE)


def test_skill_mode_decoder_width_and_num_skills():
    policy = _make_policy(skill_enabled=True)
    assert policy.d_w == K + 1
    assert policy.num_skills == K + 1
    assert policy.decoder.in_chan == STATE_DIM + LATENT_DIM + (K + 1)


def test_predict_action_defaults_to_w0():
    policy = _make_policy(skill_enabled=True)
    obs = torch.randn(3, OBS_DIM)

    default_action = policy.predict_action(obs)
    w0 = policy.skill_code(0)
    explicit_action = policy.predict_action_with_skill(obs, w0)

    # predict_action(obs) must equal decoding with the deployment skill w_0.
    assert torch.allclose(default_action, explicit_action, atol=1e-6)
    assert torch.allclose(
        default_action, policy.predict_action(obs, skill_id=0), atol=1e-6
    )


def test_w_channel_does_not_affect_z():
    policy = _make_policy(skill_enabled=True)
    obs = torch.randn(4, OBS_DIM)

    # z is a pure function of obs, independent of any skill code.
    z_a = policy.encoder(obs)
    z_b = policy.encoder(obs)
    assert torch.allclose(z_a, z_b, atol=1e-7)

    state = policy.obs2state(obs)
    # Same z/state, different skill codes -> generally different actions,
    # proving w enters the decoder while z stays the obstacle-only channel.
    a0 = policy.decode_with_skill(state, z_a, policy.skill_code(0))
    a1 = policy.decode_with_skill(state, z_a, policy.skill_code(1))
    assert not torch.allclose(a0, a1, atol=1e-4)


def test_decode_with_skill_broadcasts_single_code():
    policy = _make_policy(skill_enabled=True)
    obs = torch.randn(6, OBS_DIM)
    z = policy.encoder(obs)
    state = policy.obs2state(obs)

    w_single = policy.skill_code(2)  # [d_w]
    w_batch = w_single.unsqueeze(0).expand(6, -1)  # [B, d_w]
    a_single = policy.decode_with_skill(state, z, w_single)
    a_batch = policy.decode_with_skill(state, z, w_batch)
    assert a_single.shape == (6, ACTION_SIZE)
    assert torch.allclose(a_single, a_batch, atol=1e-6)


def test_get_policy_widens_decoder_when_skill_enabled():
    base_cfg = dict(
        _target_="ldpi",
        encoder_net=dict(
            _target_="fc_vec",
            in_chan=OBS_DIM,
            out_chan=LATENT_DIM,
            l_hidden=[16],
            activation=["relu"],
            out_activation="linear",
        ),
        decoder_net=dict(
            _target_="fc_vec",
            in_chan=STATE_DIM + LATENT_DIM,
            out_chan=ACTION_SIZE,
            l_hidden=[16],
            activation=["relu"],
            out_activation="tanh",
        ),
    )
    env = _FakeEnv()

    # Mode A: strict DIVO, decoder width unchanged.
    policy_a = get_policy(env, **base_cfg)
    assert policy_a.skill_enabled is False
    assert policy_a.decoder.in_chan == STATE_DIM + LATENT_DIM

    # Mode B: skills enabled, decoder widened by d_w = K+1.
    skill_cfg = dict(base_cfg)
    skill_cfg.update(skill_enabled=True, K=K, codebook_type="one_hot")
    policy_b = get_policy(env, **skill_cfg)
    assert policy_b.skill_enabled is True
    assert policy_b.d_w == K + 1
    assert policy_b.decoder.in_chan == STATE_DIM + LATENT_DIM + (K + 1)
