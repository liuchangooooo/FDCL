"""Task-frame trajectory embedding and Form B trajectory-level diversity.

This module implements the Route-B diversity objective (design.md 组件 4):
  - Task 8: ``task_frame_embed`` -- an SE2 start-relative embedding so the same
    trajectory under a different start pose / orientation maps to (nearly) the
    same vector, making trajectories comparable across scenes.
  - Task 9: same-environment paired diversity with a soft near-success gate and a
    margin-bounded repulsion. The per-episode diversity reward ``r_div`` is
    distributed to transitions and delivered to the critic via
    ``reward_total = reward_task + beta_div * reward_div`` (it is NOT a separate
    differentiable actor term).

The pure functions here (embedding / gate / Form B / distribution) are env-free
and unit-tested; ``run_paired_diversity_rollout`` needs a live Push-T env and is
exercised by the Stage 1 smoke run (Task 14.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


DIV_EPS = 1e-8


@dataclass
class DiversityConfig:
    """Form B hyper-parameters (all tunable; Stage 1 defaults are placeholders)."""
    alpha: float = 1.0      # success weight in q(tau)
    beta: float = 1.0       # progress weight
    gamma: float = 1.0      # collision penalty
    eta: float = 0.5        # timeout penalty
    tau_q: float = 0.0      # soft-gate threshold
    temperature: float = 1.0  # soft-gate temperature T
    margin: float = 1.0     # bounded repulsion margin m
    beta_div: float = 1.0e-2  # reward_total = reward_task + beta_div * reward_div
    rdiv_distribution: str = "uniform"  # uniform | terminal | discount
    max_steps: int = 10

    @classmethod
    def from_cfg(cls, cfg: Optional[Mapping[str, Any]]) -> "DiversityConfig":
        if cfg is None:
            return cls()
        get = (lambda k, d: cfg.get(k, d)) if isinstance(cfg, Mapping) else (lambda k, d: getattr(cfg, k, d))
        base = cls()
        return cls(
            alpha=float(get("alpha", base.alpha)),
            beta=float(get("beta", base.beta)),
            gamma=float(get("gamma", base.gamma)),
            eta=float(get("eta", base.eta)),
            tau_q=float(get("tau_q", base.tau_q)),
            temperature=float(get("temperature", base.temperature)),
            margin=float(get("margin", base.margin)),
            beta_div=float(get("beta_div", base.beta_div)),
            rdiv_distribution=str(get("rdiv_distribution", base.rdiv_distribution)),
            max_steps=int(get("max_steps", base.max_steps)),
        )


# ----------------------------------------------------------------------------
# Task 8: task-frame (SE2 start-relative) trajectory embedding
# ----------------------------------------------------------------------------

def task_frame_embed(states: Sequence[Sequence[float]], max_steps: int = 10) -> np.ndarray:
    """Embed a state trajectory in the start pose's SE2 frame.

    Each state is ``[x, y, cos(theta), sin(theta)]``. Positions are expressed
    relative to the first state and rotated into its heading frame; headings are
    made relative too. This is invariant to a global SE2 transform of the whole
    trajectory, so trajectories from different scenes are comparable. The result
    is padded/truncated to a fixed length and flattened.
    """
    arr = np.asarray(states, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros(int(max_steps) * 4, dtype=np.float64)
    arr = arr[:, :4]

    x0, y0 = float(arr[0, 0]), float(arr[0, 1])
    th0 = float(np.arctan2(arr[0, 3], arr[0, 2]))
    c0, s0 = np.cos(th0), np.sin(th0)

    dx = arr[:, 0] - x0
    dy = arr[:, 1] - y0
    # R(-th0) applied to (dx, dy)
    rx = c0 * dx + s0 * dy
    ry = -s0 * dx + c0 * dy
    th = np.arctan2(arr[:, 3], arr[:, 2]) - th0
    rel = np.stack([rx, ry, np.cos(th), np.sin(th)], axis=1)

    if rel.shape[0] < max_steps:
        pad = np.repeat(rel[-1:], int(max_steps) - rel.shape[0], axis=0)
        rel = np.concatenate([rel, pad], axis=0)
    else:
        rel = rel[: int(max_steps)]
    return rel.reshape(-1)


# ----------------------------------------------------------------------------
# Task 9: soft near-success gate + margin-bounded Form B repulsion
# ----------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def soft_gate(quality: Mapping[str, Any], cfg: DiversityConfig) -> float:
    """Soft near-success gate g(tau) in [0, 1] (Task 9.2).

    ``q(tau) = alpha*success + beta*progress - gamma*collision - eta*timeout``;
    ``g(tau) = sigmoid((q - tau_q) / T)``. Successful trajectories get large
    weight; near-success ones still contribute an early diversity signal (avoids
    the r_div == 0 startup problem of a hard success gate).
    """
    success = 1.0 if quality.get("success") else 0.0
    collision = 1.0 if quality.get("collision") else 0.0
    timeout = 1.0 if quality.get("timeout") else 0.0
    progress = float(quality.get("progress", 0.0))
    q = cfg.alpha * success + cfg.beta * progress - cfg.gamma * collision - cfg.eta * timeout
    T = cfg.temperature if abs(cfg.temperature) > DIV_EPS else DIV_EPS
    return _sigmoid((q - cfg.tau_q) / T)


def form_b_rdiv(embeddings: Sequence[np.ndarray], gates: Sequence[float],
                margin: float) -> np.ndarray:
    """Per-skill same-environment diversity reward (Task 9.2).

    ``r_div(k) = g_k * [ sum_{l!=k} g_l * min(d(phi_k, phi_l), m) ]
                      / [ sum_{l!=k} g_l + eps ]``.

    Distances are only compared WITHIN one environment (caller passes one env's
    skills) so environment variance never leaks into the diversity signal.
    """
    K = len(embeddings)
    r = np.zeros(K, dtype=np.float64)
    if K <= 1:
        return r
    emb = [np.asarray(e, dtype=np.float64) for e in embeddings]
    g = np.asarray(gates, dtype=np.float64)
    for k in range(K):
        num = 0.0
        den = 0.0
        for l in range(K):
            if l == k:
                continue
            d = float(np.linalg.norm(emb[k] - emb[l]))
            d = min(d, float(margin))
            num += g[l] * d
            den += g[l]
        r[k] = g[k] * (num / (den + DIV_EPS))
    return r


def distribute_rdiv(r_div: float, n_steps: int, scheme: str = "uniform") -> np.ndarray:
    """Distribute an episode-level r_div across its transitions (Task 9.3).

    - ``uniform``  (default): r_div / n_steps on every step (sums to r_div).
    - ``terminal``: all of r_div on the last step.
    - ``discount``: geometric weights increasing toward the end (normalized).
    """
    n = int(n_steps)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    scheme = str(scheme).lower()
    if scheme == "terminal":
        w = np.zeros(n, dtype=np.float64)
        w[-1] = 1.0
    elif scheme == "discount":
        gamma = 0.9
        w = gamma ** np.arange(n - 1, -1, -1, dtype=np.float64)
        w = w / (w.sum() + DIV_EPS)
    else:  # uniform
        w = np.full(n, 1.0 / n, dtype=np.float64)
    return float(r_div) * w


# ----------------------------------------------------------------------------
# Orchestration: env-free reward computation + env-based paired rollout
# ----------------------------------------------------------------------------

def compute_diversity_rewards(
    rollouts_per_env: Sequence[Sequence[Mapping[str, Any]]],
    cfg: DiversityConfig,
) -> Dict[str, Any]:
    """Compute Form B rewards from paired rollouts (env-free; unit-tested).

    ``rollouts_per_env[i][j]`` is a ``rollout_fixed_skill`` result for probe
    skill ``j`` in environment ``i`` (all probe skills share the same env seed).
    Returns per-(i,j) ``reward_div`` per transition and ``reward_total``, plus
    aggregate metrics. w_0 is never part of the paired batch (its reward_div=0
    is handled by normal collection).
    """
    per_env_rdiv: List[np.ndarray] = []
    enriched: List[List[Dict[str, Any]]] = []
    for env_rollouts in rollouts_per_env:
        embs = [task_frame_embed(r.get("states", []), cfg.max_steps) for r in env_rollouts]
        gates = [soft_gate(r.get("quality", {}), cfg) for r in env_rollouts]
        rdiv = form_b_rdiv(embs, gates, cfg.margin)
        per_env_rdiv.append(rdiv)

        env_out: List[Dict[str, Any]] = []
        for j, r in enumerate(env_rollouts):
            transitions = list(r.get("transitions", []))
            n = len(transitions)
            div_seq = distribute_rdiv(float(rdiv[j]), n, cfg.rdiv_distribution)
            out_trs = []
            for t, tr in enumerate(transitions):
                r_task = float(tr.get("reward_task", 0.0))
                r_div_t = float(div_seq[t]) if t < len(div_seq) else 0.0
                out_trs.append({
                    **tr,
                    "reward_div": r_div_t,
                    "reward_total": r_task + cfg.beta_div * r_div_t,
                    "source": "diversity_paired",
                })
            env_out.append({
                "skill_id": int(r.get("skill_id", j + 1)),
                "gate": float(gates[j]),
                "r_div": float(rdiv[j]),
                "transitions": out_trs,
            })
        enriched.append(env_out)

    all_rdiv = np.concatenate(per_env_rdiv) if per_env_rdiv else np.zeros(0)
    metrics = {
        "mean_r_div": float(all_rdiv.mean()) if all_rdiv.size else 0.0,
        "max_r_div": float(all_rdiv.max()) if all_rdiv.size else 0.0,
        "n_env": len(rollouts_per_env),
    }
    return {"enriched": enriched, "metrics": metrics}


def run_paired_diversity_rollout(
    env: Any,
    policy: Any,
    layouts: Sequence[Mapping[str, Any]],
    probe_skill_ids: Sequence[int],
    device: Any,
    max_steps: int = 10,
) -> List[List[Dict[str, Any]]]:
    """Roll out every probe skill on the SAME set of env layouts (Task 9.1).

    ``layouts[i] = {"start": ..., "obstacles": ...}``. Returns
    ``rollouts_per_env[i][j]`` for probe skill ``probe_skill_ids[j]`` -- all
    skills on env ``i`` share the same layout, so diversity is measured within
    one environment. Requires a live Push-T env (see smoke run).
    """
    from DIVO.curriculum.skill_signal import rollout_fixed_skill

    rollouts_per_env: List[List[Dict[str, Any]]] = []
    for layout in layouts:
        start = layout["start"]
        obstacles = layout.get("obstacles", [])
        env_rollouts = []
        for skill_id in probe_skill_ids:
            env_rollouts.append(
                rollout_fixed_skill(
                    env=env,
                    policy=policy,
                    start=start,
                    obstacle_cfg=obstacles,
                    skill_id=int(skill_id),
                    device=device,
                    max_steps=int(max_steps),
                )
            )
        rollouts_per_env.append(env_rollouts)
    return rollouts_per_env


# ----------------------------------------------------------------------------
# Task 11: K_eff = exp(H(cluster distribution)) + Form A monitors (log-only)
# ----------------------------------------------------------------------------

def _cluster_labels(embeddings: Sequence[np.ndarray], threshold: float) -> List[int]:
    """Deterministic distance-threshold clustering (connected components).

    Two embeddings are linked when their L2 distance < threshold; clusters are
    the connected components. Deterministic given the input order, so K_eff is
    reproducible (part of the fixed protocol).
    """
    n = len(embeddings)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    emb = [np.asarray(e, dtype=np.float64) for e in embeddings]
    for i in range(n):
        for j in range(i + 1, n):
            if float(np.linalg.norm(emb[i] - emb[j])) < float(threshold):
                union(i, j)
    roots = [find(i) for i in range(n)]
    remap: Dict[int, int] = {}
    labels = []
    for r in roots:
        if r not in remap:
            remap[r] = len(remap)
        labels.append(remap[r])
    return labels


def compute_k_eff(
    embeddings_per_env: Sequence[Sequence[np.ndarray]],
    threshold: float,
) -> Dict[str, Any]:
    """Effective number of skills via cluster-distribution entropy (Task 11.1).

    Per environment, cluster the K skill embeddings, then
    ``K_eff_i = exp(H(size_distribution))``; the reported K_eff is the mean over
    environments. Occupied-cluster count is returned for logging only (it
    over-counts under heavily uneven distributions and is NOT the go/no-go
    metric). ``K_eff`` collapses to 1 when all skills behave identically and
    approaches K when all behaviors are distinct.
    """
    per_env_keff: List[float] = []
    per_env_occupied: List[int] = []
    for embs in embeddings_per_env:
        m = len(embs)
        if m == 0:
            continue
        labels = _cluster_labels(embs, threshold)
        _, counts = np.unique(np.asarray(labels), return_counts=True)
        p = counts.astype(np.float64) / counts.sum()
        H = float(-np.sum(p * np.log(p + DIV_EPS)))
        per_env_keff.append(float(np.exp(H)))
        per_env_occupied.append(int(len(counts)))
    return {
        "k_eff": float(np.mean(per_env_keff)) if per_env_keff else 0.0,
        "occupied_clusters": float(np.mean(per_env_occupied)) if per_env_occupied else 0.0,
        "per_env_k_eff": per_env_keff,
    }


def k_eff_from_rollouts(
    rollouts_per_env: Sequence[Sequence[Mapping[str, Any]]],
    threshold: float,
    max_steps: int = 10,
) -> Dict[str, Any]:
    """Convenience: embed paired rollouts then compute K_eff (fixed protocol)."""
    embeddings_per_env = [
        [task_frame_embed(r.get("states", []), max_steps) for r in env_rollouts]
        for env_rollouts in rollouts_per_env
    ]
    return compute_k_eff(embeddings_per_env, threshold)


def action_variance(actions: np.ndarray) -> float:
    """Form A ActionVar monitor (log-only), matching DIVO Table II口径.

    ``actions`` has shape ``[n_obs, n_skills, action_dim]``: variance is taken
    across skills, summed over action dims, then averaged over observations.
    """
    arr = np.asarray(actions, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] < 2:
        return 0.0
    return float(np.mean(np.var(arr, axis=1).sum(axis=-1)))


def saturation_rate(actions: np.ndarray, tau_sat: float = 0.99) -> float:
    """Form A SatRate monitor (log-only): fraction of near-saturated actions."""
    arr = np.asarray(actions, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.mean(np.abs(arr) > float(tau_sat)))


def collect_skill_actions(policy: Any, obs_th: Any, skill_ids: Optional[Sequence[int]] = None):
    """Decode actions for each skill over a batch of obs (for Form A monitors).

    Returns a numpy array ``[n_obs, n_skills, action_dim]``. z is recomputed per
    obs; only w varies across skills.
    """
    import torch

    if not getattr(policy, "skill_enabled", False):
        raise RuntimeError("collect_skill_actions requires a skill_enabled policy")
    if skill_ids is None:
        skill_ids = list(range(policy.num_skills))
    with torch.no_grad():
        z = policy.encoder(obs_th)
        state = policy.obs2state(obs_th)
        per_skill = []
        for k in skill_ids:
            w = policy.skill_code(int(k))
            a = policy.decode_with_skill(state, z, w)
            per_skill.append(a.detach().cpu().numpy())
    # stack -> [n_skills, n_obs, adim] -> [n_obs, n_skills, adim]
    return np.stack(per_skill, axis=0).transpose(1, 0, 2)


def write_diversity_transitions(replay_buffer: Any, diversity_result: Mapping[str, Any]) -> int:
    """Write enriched paired-diversity transitions into the replay buffer.

    Each transition is tagged ``source='diversity_paired'`` with its skill_id and
    the three reward columns. Returns the number of transitions written.
    """
    written = 0
    for env_out in diversity_result.get("enriched", []):
        for skill_out in env_out:
            skill_id = int(skill_out["skill_id"])
            for tr in skill_out["transitions"]:
                replay_buffer.add(
                    tr["obs"], tr["next_obs"], tr["action"],
                    tr["reward_task"], tr["done"],
                    skill_id=skill_id,
                    source="diversity_paired",
                    reward_task=tr["reward_task"],
                    reward_div=tr["reward_div"],
                    reward_total=tr["reward_total"],
                )
                written += 1
    return written
