"""Stage 1 skill-library engineering smoke (Task 14.4).

Exercises the full Route-B Stage 1 pipeline WITHOUT MuJoCo, using a lightweight
FakeEnv, to prove the pieces wire together:
  - skill-conditioned policy + w-conditioned critic build with widened dims;
  - replay buffer stores skill_id / source / three reward columns;
  - rollout_fixed_skill runs (w fixed, z per-step);
  - paired diversity batch computes r_div and writes diversity_paired transitions;
  - executed-w actor loss + w-routed critic loss run and backprop;
  - K_eff (exp(H)) and the Stage 1 go/no-go aggregation compute;
  - the fixed difficulty ladder is probed.

Run: python scripts/stage1_skill_smoke.py
"""

import sys, types, pathlib
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DIVO.nets.modules import FC_vec
from DIVO.policy import get_policy
from DIVO.critic import get_critic
from DIVO.RL.component import StateDictReplayBuffer, reward_for_critic, SOURCE_DIVERSITY_PAIRED
from DIVO.curriculum.skill_diversity import (
    run_paired_diversity_rollout, compute_diversity_rewards, write_diversity_transitions,
    k_eff_from_rollouts, DiversityConfig, collect_skill_actions, action_variance, saturation_rate)
from DIVO.curriculum.stage1_ladder import build_stage1_ladder, flatten_ladder
from DIVO.curriculum.stage1_eval import run_stage1_probe, stage1_gonogo
from DIVO.workspace.rl_workspace.td3_curriculum_workspace import TD3CurriculumWorkspace as W

OBS, ST, LAT, ACT, K, NOBS = 8, 4, 3, 6, 4, 2


class FakeEnv:
    """Minimal Push-T-like env: obs = [x, y, cos, sin, obs1x, obs1y, obs2x, obs2y]."""
    def __init__(self):
        self._obstacles = []
        self._t = 0
        self._pose = np.zeros(3)

    def sample_valid_tblock_pose(self):
        return np.array([0.15, 0.15, -np.pi / 4]) + np.random.uniform(-0.02, 0.02, 3)

    def is_obstacle_config_valid(self, obstacles, start):
        return True

    def set_obstacle_config(self, obstacles):
        self._obstacles = list(obstacles)

    def get_ncon(self):
        return 0

    def _obs(self):
        o = np.zeros((1, OBS), dtype=np.float32)
        o[0, 0], o[0, 1] = self._pose[0], self._pose[1]
        o[0, 2], o[0, 3] = np.cos(self._pose[2]), np.sin(self._pose[2])
        for i, ob in enumerate(self._obstacles[:NOBS]):
            o[0, 4 + 2 * i] = ob["x"]
            o[0, 5 + 2 * i] = ob["y"]
        return o

    def reset(self, tblock_pos=None, force_tblock_pos=False):
        self._t = 0
        self._pose = np.asarray(tblock_pos if tblock_pos is not None else [0.15, 0.15, 0.0], float).copy()
        return self._obs()

    def obs2state(self, obs_th):
        return obs_th[:, :ST]

    def step(self, action):
        self._t += 1
        # drift toward goal (0,0)
        self._pose[:2] *= 0.6
        dist = float(np.hypot(self._pose[0], self._pose[1]))
        done = self._t >= 3 or dist < 0.05
        success = dist < 0.08
        info = {"success": bool(success), "termination": "success" if success else "timeout"}
        reward = -dist
        return self._obs(), reward, done, info


def main():
    torch.manual_seed(0); np.random.seed(0)
    env = FakeEnv()
    device = torch.device("cpu")

    pol_cfg = dict(
        _target_="ldpi", skill_enabled=True, K=K, codebook_type="one_hot",
        encoder_net=dict(_target_="fc_vec", in_chan=OBS, out_chan=LAT, l_hidden=[32], activation=["relu"], out_activation="linear"),
        decoder_net=dict(_target_="fc_vec", in_chan=ST + LAT, out_chan=ACT, l_hidden=[32], activation=["relu"], out_activation="tanh"),
    )
    crit_cfg = dict(
        _target_="mcritic", n_critics=2, skill_enabled=True, K=K, codebook_type="one_hot",
        net0=dict(_target_="fc_vec", in_chan=OBS + ACT, out_chan=1, l_hidden=[32], activation=["relu"], out_activation="linear"),
        net1=dict(_target_="fc_vec", in_chan=OBS + ACT, out_chan=1, l_hidden=[32], activation=["relu"], out_activation="linear"),
    )
    policy = get_policy(env, **pol_cfg)
    critic = get_critic(**crit_cfg)
    assert policy.decoder.in_chan == ST + LAT + (K + 1)
    assert critic.q_networks[0].q_net.in_chan == OBS + ACT + (K + 1)
    print("[1] skill policy+critic built; decoder in=%d critic in=%d" % (
        policy.decoder.in_chan, critic.q_networks[0].q_net.in_chan))

    buf = StateDictReplayBuffer(10000, obs_dim=(OBS,), action_dim=(ACT,), track_skill=True)

    # normal rollouts (Task 6): one skill per episode
    for ep in range(40):
        skill_id = 0 if np.random.rand() < 0.5 else int(np.random.randint(1, K + 1))
        obs = env.reset(tblock_pos=env.sample_valid_tblock_pose(), force_tblock_pos=True)
        done = False
        while not done:
            obs_th = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                a = policy.predict_action(obs_th, skill_id=skill_id).numpy()
            nobs, r, done, info = env.step(a[0])
            buf.add(obs, nobs, a, r, done, skill_id=skill_id, source="normal", reward_task=r)
            obs = nobs
    print("[2] normal rollouts -> buffer size=%d" % buf.size)

    # paired diversity batch (Task 9)
    layouts = [{"start": env.sample_valid_tblock_pose().tolist(),
                "obstacles": [{"x": 0.05, "y": 0.05, "purpose": "b"}, {"x": -0.15, "y": 0.15, "purpose": "f"}]}
               for _ in range(4)]
    rollouts = run_paired_diversity_rollout(env, policy, layouts, list(range(1, K + 1)), device, max_steps=10)
    res = compute_diversity_rewards(rollouts, DiversityConfig(beta_div=0.01, margin=1.0))
    n_written = write_diversity_transitions(buf, res)
    keff = k_eff_from_rollouts(rollouts, threshold=0.5)
    print("[3] paired diversity: wrote %d transitions, mean_r_div=%.4f, K_eff=%.3f" % (
        n_written, res["metrics"]["mean_r_div"], keff["k_eff"]))
    assert n_written > 0

    # verify buffer has diversity_paired + reward_total routing
    b = buf.sample_stratified(batch_size=64, w0_min_ratio=0.5, paired_sample_ratio=0.3)
    assert "skill_id" in b and "reward_total" in b
    routed = reward_for_critic(b["skill_id"], b["reward_task"], b["reward_total"])
    assert routed.shape == b["reward_task"].shape
    print("[4] stratified batch ratios=%s; reward_for_critic routed ok" % (
        {k: round(v, 3) for k, v in b["batch_ratios"].items()}))

    # executed-w actor loss + w-routed critic loss via bound workspace methods
    s = types.SimpleNamespace(skill_enabled=True, model=policy, critic=critic,
                              critic_target=critic, model_target=policy, device=device,
                              lambda_cov_all=0.0, gamma=0.9, cfg=types.SimpleNamespace(training={}))
    s._skill_w_from_batch = W._skill_w_from_batch.__get__(s)
    ploss, _ = W.compute_policy_loss.__get__(s)(b)
    ploss.backward()
    nq = W.compute_next_q_value.__get__(s)(b)
    assert nq.shape[0] == b["actions"].shape[0]
    print("[5] executed-w policy_loss=%.4f backward ok; next_q shape=%s" % (ploss.item(), tuple(nq.shape)))

    # Form A monitors
    obs_th = torch.tensor(b["observations"], dtype=torch.float32)
    acts = collect_skill_actions(policy, obs_th)
    print("[6] ActionVar=%.4f SatRate=%.3f (monitors)" % (action_variance(acts), saturation_rate(acts)))

    # ladder + go/no-go probe
    ladder = build_stage1_ladder(obstacle_num=NOBS, n_per_category=2, seed=0)
    recs = run_stage1_probe(env, policy, ladder, K=K, device=device, max_steps=10)
    go = stage1_gonogo(recs, K=K, k_eff=keff["k_eff"])
    print("[7] ladder probed: %d scenes; gonogo gates=%s" % (len(recs), go["gates"]))
    assert len(recs) == 6 * 2

    print("\nSMOKE OK: full Stage 1 skill pipeline wired end-to-end.")


if __name__ == "__main__":
    main()
