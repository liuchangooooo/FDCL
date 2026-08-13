from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DIVO.curriculum.adapters.pusht_adapter import (
    PUSHT_CORRIDOR_WIDTH,
    PUSHT_TARGET_XY,
)
from DIVO.curriculum.candidate_verifier import SandboxPushTExecutor
from DIVO.curriculum.obstacle_geometry import encode_xy_to_z


GeneratorLike = Any


@dataclass
class SceneSamplingStats:
    attempts: int = 0
    accepted: int = 0
    empty_generations: int = 0
    wrong_count_generations: int = 0
    invalid_generations: int = 0
    contact_rejects: int = 0
    exceptions: int = 0

    @property
    def valid_scene_rate(self) -> float:
        if self.attempts <= 0:
            return 0.0
        return float(self.accepted / self.attempts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempts": int(self.attempts),
            "accepted": int(self.accepted),
            "empty_generations": int(self.empty_generations),
            "wrong_count_generations": int(self.wrong_count_generations),
            "invalid_generations": int(self.invalid_generations),
            "contact_rejects": int(self.contact_rejects),
            "exceptions": int(self.exceptions),
            "valid_scene_rate": float(self.valid_scene_rate),
        }


def extract_skill_signals(
    env: Any,
    policy: Any,
    generator: GeneratorLike,
    device: torch.device | str,
    latent_dim: int,
    K: int = 16,
    M: int = 60,
    n: int = 4,
    num_obstacles: int = 2,
    max_steps: int = 10,
    seed: int = 0,
    max_scene_attempts: Optional[int] = None,
    generator_timeout_sec: int = 5,
    route_waypoints: int = 5,
    hist_bins: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    progress_every: int = 0,
    fixed_starts: Optional[Sequence[Sequence[float]]] = None,
    z_bank: Optional[torch.Tensor] = None,
    invalid_scene_policy: str = "resample",
    include_behavior: bool = True,
    probe_source: str = "random_z",
    tau_saturation: float = 0.125,
    target_delta: float = 0.15,
    dup_round: int = 2,
    exclude_invalid_from_boundary: bool = False,
) -> Dict[str, Any]:
    """Probe a Push-T policy against scenes from the current generator.

    A scene is the bound pair ``(start, generator(start))``. For every accepted
    scene the same probe skills are injected, making the skill the only intended
    behavior variable during the deterministic probe.

    ``probe_source`` (Task 16, Requirement 7.1):
      - ``"random_z"`` (default, legacy-safe): inject a random latent bank via
        ``rollout_fixed_z`` (z frozen for the whole episode). Preserves the old
        attribution/evolve curriculum and baselines exactly.
      - ``"w_probe"`` (Stage 2 V0, opt-in): inject the *trained* skill codebook
        ``W_probe = {w_1..w_K}`` via ``rollout_fixed_skill`` (w fixed, z_t =
        encoder(obs_t) per step). ``p_i = (1/K) Σ_k 1[π_{w_k} solves e_i]`` uses
        the K probe skills ONLY (w_0 is diagnostic, never in p_i). Requires a
        ``skill_enabled`` policy; ``K`` is taken from ``policy.K``. The Task 18
        skill-library orchestration passes this explicitly.
    """

    device = torch.device(device)
    probe_source = str(probe_source).lower()
    if probe_source not in ("w_probe", "random_z"):
        raise ValueError("probe_source must be 'w_probe' or 'random_z'")
    K = int(K)
    M = int(M)
    n = int(n)
    max_steps = int(max_steps)
    if M <= 0:
        raise ValueError("M must be positive")
    if n < 0:
        raise ValueError("n must be non-negative")

    invalid_scene_policy = str(invalid_scene_policy).lower()
    if invalid_scene_policy not in ("resample", "zero"):
        raise ValueError("invalid_scene_policy must be 'resample' or 'zero'")

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if hasattr(policy, "eval"):
        policy.eval()

    if probe_source == "w_probe":
        # Probe with the trained skill library (Stage 1-validated path).
        if not getattr(policy, "skill_enabled", False):
            raise RuntimeError(
                "probe_source='w_probe' requires a skill_enabled policy; "
                "use probe_source='random_z' for the legacy latent probe."
            )
        K = int(policy.K)
        if K <= 0:
            raise ValueError("policy.K must be positive for W_probe injection")
        z_bank = None  # unused in W_probe mode
    else:
        if K <= 0:
            raise ValueError("K must be positive")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive for random_z probe")
        if z_bank is None:
            z_bank = build_z_bank(K=K, latent_dim=int(latent_dim), seed=int(seed), device=device)
        else:
            z_bank = z_bank.to(device)
            if z_bank.ndim != 2:
                raise ValueError("z_bank must have shape [K, latent_dim]")
            K = int(z_bank.shape[0])
            if int(z_bank.shape[1]) != int(latent_dim):
                raise ValueError(
                    f"z_bank latent dim {z_bank.shape[1]} does not match latent_dim={latent_dim}"
                )
    stats = SceneSamplingStats()
    per_scene: List[Dict[str, Any]] = []
    if fixed_starts is not None:
        starts = [_to_float_list(start) for start in fixed_starts]
        if len(starts) < M:
            raise ValueError(f"fixed_starts has {len(starts)} starts, but M={M}")
        starts = starts[:M]
    else:
        starts = None
    max_attempts = int(max_scene_attempts or max(M * 20, M))

    while len(per_scene) < M and (
        starts is not None or stats.attempts < max_attempts
    ):
        fixed_start = starts[len(per_scene)] if starts is not None else None
        scene = sample_generator_scene(
            env=env,
            generator=generator,
            num_obstacles=int(num_obstacles),
            timeout_sec=int(generator_timeout_sec),
            stats=stats,
            fixed_start=fixed_start,
        )
        if scene is None:
            if fixed_start is not None and invalid_scene_policy == "zero":
                per_scene.append(
                    make_invalid_scene_record(
                        scene_id=len(per_scene),
                        start=fixed_start,
                        K=K,
                        reason="invalid_generated_scene",
                    )
                )
            elif fixed_start is not None:
                raise RuntimeError(
                    "fixed_starts cannot use invalid_scene_policy='resample'; "
                    "use 'zero' for paired verifier probes"
                )
            continue

        scene_id = len(per_scene)
        start = scene["start"]
        obstacles = scene["obstacles"]
        if probe_source == "w_probe":
            scene_result = rollout_scene_with_skills(
                env=env,
                policy=policy,
                start=start,
                obstacles=obstacles,
                K=K,
                device=device,
                max_steps=max_steps,
            )
        else:
            scene_result = rollout_scene_with_z_bank(
                env=env,
                policy=policy,
                start=start,
                obstacles=obstacles,
                z_bank=z_bank,
                device=device,
                max_steps=max_steps,
            )
        successes = [int(route["success"]) for route in scene_result["routes"]]
        feasible = int(max(successes)) if successes else 0
        realized = float(np.mean(successes)) if successes else 0.0
        deployed = int(scene_result["deployed_success"])
        lv = float(realized * (1.0 - realized))
        per_scene.append(
            {
                "scene_id": int(scene_id),
                "start": _to_float_list(start),
                "obstacles": normalize_obstacles(obstacles),
                "feasible": int(feasible),
                "realized": float(realized),
                "deployed": int(deployed),
                "lv": float(lv),
                "routes": scene_result["routes"],
                "deployed_route": scene_result["deployed_route"],
            }
        )
        if progress_every > 0 and len(per_scene) % int(progress_every) == 0:
            print(
                f"[skill_signal] {len(per_scene)}/{M} "
                f"feasible={feasible} realized={realized:.3f} deployed={deployed}"
            )

    if len(per_scene) < M:
        raise RuntimeError(
            f"Only collected {len(per_scene)}/{M} valid scenes after "
            f"{stats.attempts} attempts; sampling stats={stats.to_dict()}"
        )

    difficulty_profile = compute_difficulty_profile(
        per_scene=per_scene,
        K=K,
        hist_bins=hist_bins,
        sampling_stats=stats,
    )
    design_context = build_design_context(
        per_scene=per_scene,
        n=n,
        env=env,
        route_waypoints=int(route_waypoints),
        include_behavior=bool(include_behavior),
        tau_saturation=float(tau_saturation),
        strict_boundary_bins=(probe_source == "w_probe"),
    )
    boundary_signal = compute_boundary_signal(
        per_scene=per_scene,
        K=K,
        tau=float(tau_saturation),
        target_delta=float(target_delta),
        valid_rate=float(stats.valid_scene_rate),
        dup_round=int(dup_round),
        exclude_invalid_from_boundary=bool(exclude_invalid_from_boundary),
    )
    return {
        "difficulty_profile": difficulty_profile,
        "design_context": design_context,
        "boundary_signal": boundary_signal,
        "per_scene": per_scene,
        "probe_source": probe_source,
    }


def build_z_bank(
    K: int,
    latent_dim: int,
    seed: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Build a shared latent bank on CPU first for deterministic device handling."""

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    z_bank = torch.randn(int(K), int(latent_dim), generator=gen, dtype=torch.float32)
    return z_bank.to(torch.device(device))


def sample_fixed_starts(
    env: Any,
    M: int,
    seed: int,
) -> List[List[float]]:
    """Sample a deterministic list of Push-T starts for paired verifier probes."""

    np.random.seed(int(seed))
    starts = []
    for _ in range(int(M)):
        starts.append(_to_float_list(env.sample_valid_tblock_pose()))
    return starts


def sample_generator_scene(
    env: Any,
    generator: GeneratorLike,
    num_obstacles: int,
    timeout_sec: int,
    stats: SceneSamplingStats,
    fixed_start: Optional[Sequence[float]] = None,
) -> Optional[Dict[str, Any]]:
    """Sample a valid scene using the training-time order: start then generator."""

    stats.attempts += 1
    try:
        if fixed_start is None:
            start = np.asarray(env.sample_valid_tblock_pose(), dtype=np.float64)
        else:
            start = np.asarray(fixed_start, dtype=np.float64)
        obstacles = call_generator(
            generator=generator,
            start=start,
            num_obstacles=int(num_obstacles),
            timeout_sec=int(timeout_sec),
        )
        obstacles = normalize_obstacles(obstacles)
        if not obstacles:
            stats.empty_generations += 1
            return None
        if len(obstacles) != int(num_obstacles):
            stats.wrong_count_generations += 1
            return None
        if hasattr(env, "is_obstacle_config_valid"):
            if not env.is_obstacle_config_valid(obstacles, start):
                stats.invalid_generations += 1
                return None
        if hasattr(env, "set_obstacle_config"):
            env.set_obstacle_config(obstacles)
        if hasattr(env, "reset"):
            env.reset(tblock_pos=start, force_tblock_pos=True)
        if hasattr(env, "get_ncon") and int(env.get_ncon()) != 0:
            stats.contact_rejects += 1
            return None
        stats.accepted += 1
        return {"start": _to_float_list(start), "obstacles": obstacles}
    except Exception:
        stats.exceptions += 1
        return None


def make_invalid_scene_record(
    scene_id: int,
    start: Sequence[float],
    K: int,
    reason: str,
) -> Dict[str, Any]:
    routes = [
        {
            "skill_index": int(skill_idx),
            "success": False,
            "states": [],
        }
        for skill_idx in range(int(K))
    ]
    return {
        "scene_id": int(scene_id),
        "start": _to_float_list(start),
        "obstacles": [],
        "feasible": 0,
        "realized": 0.0,
        "deployed": 0,
        "lv": 0.0,
        "routes": routes,
        "deployed_route": {
            "success": False,
            "states": [],
        },
        "invalid": True,
        "invalid_reason": str(reason),
    }


def call_generator(
    generator: GeneratorLike,
    start: Sequence[float],
    num_obstacles: int,
    timeout_sec: int,
) -> List[Dict[str, Any]]:
    """Call either a sandbox executor or a plain generate_obstacles callable."""

    pose = np.asarray(start, dtype=np.float64)
    if hasattr(generator, "generate") and callable(generator.generate):
        return generator.generate(
            pose,
            int(num_obstacles),
            timeout_sec=int(timeout_sec),
        )
    if callable(generator):
        try:
            return generator(pose, int(num_obstacles))
        except TypeError:
            return generator(pose)
    raise TypeError("generator must expose .generate(...) or be callable")


def normalize_obstacles(obstacles: Any) -> List[Dict[str, Any]]:
    """Normalize generator outputs into JSON-safe Push-T obstacle dicts."""

    if not isinstance(obstacles, list):
        return []
    normalized = []
    for obs in obstacles:
        if hasattr(obs, "__len__") and not isinstance(obs, Mapping):
            try:
                values = list(obs)
                if len(values) >= 2:
                    obs = {
                        "x": values[0],
                        "y": values[1],
                        "purpose": values[2] if len(values) > 2 else "",
                    }
            except Exception:
                continue
        if not isinstance(obs, Mapping) or "x" not in obs or "y" not in obs:
            continue
        try:
            normalized.append(
                {
                    "x": float(obs["x"]),
                    "y": float(obs["y"]),
                    "purpose": str(obs.get("purpose", "")),
                }
            )
        except Exception:
            continue
    return normalized


@torch.no_grad()
def rollout_scene_with_z_bank(
    env: Any,
    policy: Any,
    start: Sequence[float],
    obstacles: Sequence[Mapping[str, Any]],
    z_bank: torch.Tensor,
    device: torch.device,
    max_steps: int,
) -> Dict[str, Any]:
    routes = []
    for skill_idx in range(int(z_bank.shape[0])):
        z = z_bank[skill_idx : skill_idx + 1]
        success, traj = rollout_fixed_z(
            env=env,
            policy=policy,
            start=start,
            obstacle_cfg=obstacles,
            z=z,
            device=device,
            max_steps=max_steps,
        )
        routes.append(
            {
                "skill_index": int(skill_idx),
                "success": bool(success),
                "states": [_to_float_list(state) for state in traj],
            }
        )

    deployed_success, deployed_traj = rollout_fixed_z(
        env=env,
        policy=policy,
        start=start,
        obstacle_cfg=obstacles,
        z=None,
        device=device,
        max_steps=max_steps,
    )
    return {
        "routes": routes,
        "deployed_success": bool(deployed_success),
        "deployed_route": {
            "success": bool(deployed_success),
            "states": [_to_float_list(state) for state in deployed_traj],
        },
    }


@torch.no_grad()
def rollout_scene_with_skills(
    env: Any,
    policy: Any,
    start: Sequence[float],
    obstacles: Sequence[Mapping[str, Any]],
    K: int,
    device: torch.device,
    max_steps: int,
) -> Dict[str, Any]:
    """Probe one scene with the trained skill library (Task 16 / Requirement 7.1).

    Injects the K probe skills ``w_1..w_K`` via ``rollout_fixed_skill`` (w fixed,
    ``z_t = encoder(obs_t)`` recomputed per step). ``w_0`` (skill_id 0) is rolled
    out separately as the deployment diagnostic and is NOT part of ``p_i``.
    """
    routes = []
    for skill_id in range(1, int(K) + 1):
        res = rollout_fixed_skill(
            env=env,
            policy=policy,
            start=start,
            obstacle_cfg=obstacles,
            skill_id=int(skill_id),
            device=device,
            max_steps=max_steps,
        )
        routes.append(
            {
                "skill_index": int(skill_id),
                "success": bool(res["success"]),
                "states": [_to_float_list(state) for state in res["states"]],
            }
        )

    # w_0 deployment diagnostic (not counted in p_i, see M5=a).
    res0 = rollout_fixed_skill(
        env=env,
        policy=policy,
        start=start,
        obstacle_cfg=obstacles,
        skill_id=0,
        device=device,
        max_steps=max_steps,
    )
    return {
        "routes": routes,
        "deployed_success": bool(res0["success"]),
        "deployed_route": {
            "success": bool(res0["success"]),
            "states": [_to_float_list(state) for state in res0["states"]],
        },
    }


@torch.no_grad()
def rollout_fixed_skill(
    env: Any,
    policy: Any,
    start: Sequence[float],
    obstacle_cfg: Sequence[Mapping[str, Any]],
    skill_id: int,
    device: torch.device,
    max_steps: int = 10,
) -> Dict[str, Any]:
    """Run one Push-T episode with a FIXED skill code (Task 5 / Requirement 4.2).

    Unlike ``rollout_fixed_z`` (which pins a latent ``z``), this pins the skill
    code ``w = policy.skill_code(skill_id)`` for the whole episode while the
    obstacle/context latent ``z_t = encoder(obs_t)`` is recomputed from the real
    observation at every step. This is the correct probe for Route B: the skill
    is the fixed behavior code, not a frozen latent.

    Returns a dict with the state trajectory (for task-frame embedding), the
    task-reward sequence, per-step transitions (for replay writeback), and a
    ``quality`` summary (success / collision / timeout / fall / progress).
    """
    if not getattr(policy, "skill_enabled", False):
        raise RuntimeError("rollout_fixed_skill requires a skill_enabled policy")

    device = torch.device(device)
    w = policy.skill_code(int(skill_id)).to(device)

    env.set_obstacle_config(normalize_obstacles(list(obstacle_cfg)))
    obs = env.reset(tblock_pos=np.asarray(start, dtype=np.float64), force_tblock_pos=True)
    done = False
    steps = 0
    info: Dict[str, Any] = {"success": False, "termination": "timeout"}
    states: List[np.ndarray] = []
    rewards: List[float] = []
    transitions: List[Dict[str, Any]] = []

    while not done:
        obs_th = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if obs_th.ndim == 1:
            obs_th = obs_th.unsqueeze(0)
        state = env.obs2state(obs_th)
        states.append(state.detach().cpu().numpy().reshape(-1)[:4].astype(np.float64))
        # z_t is recomputed from the real obs every step; only w is fixed.
        z = policy.encoder(obs_th)
        action = policy.decode_with_skill(state, z, w)
        act_np = action.detach().cpu().numpy()[0]
        next_obs, reward, done, info = env.step(act_np)
        rewards.append(float(reward))
        transitions.append(
            {
                "obs": np.asarray(obs, dtype=np.float32).reshape(-1),
                "action": np.asarray(act_np, dtype=np.float32).reshape(-1),
                "reward_task": float(reward),
                "next_obs": np.asarray(next_obs, dtype=np.float32).reshape(-1),
                "done": bool(done),
                "skill_id": int(skill_id),
            }
        )
        obs = next_obs
        steps += 1
        if steps >= int(max_steps):
            done = True

    termination = str(info.get("termination", "timeout"))
    success = bool(info.get("success", False))
    final_reward = float(rewards[-1]) if rewards else 0.0
    quality = {
        "success": success,
        "collision": termination == "collision",
        "timeout": termination == "timeout",
        "fall": termination == "fall",
        # progress is a raw task-reward proxy (higher = closer to goal);
        # the soft near-success gate q(tau) in Task 9 normalizes it.
        "progress": final_reward,
        "final_reward": final_reward,
        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "n_steps": int(steps),
    }
    return {
        "skill_id": int(skill_id),
        "success": success,
        "states": states,
        "rewards": rewards,
        "transitions": transitions,
        "quality": quality,
    }


@torch.no_grad()
def rollout_fixed_z(
    env: Any,
    policy: Any,
    start: Sequence[float],
    obstacle_cfg: Sequence[Mapping[str, Any]],
    z: Optional[torch.Tensor],
    device: torch.device,
    max_steps: int = 10,
) -> Tuple[bool, List[np.ndarray]]:
    """Run one deterministic Push-T episode and return success plus state route."""

    env.set_obstacle_config(normalize_obstacles(list(obstacle_cfg)))
    obs = env.reset(tblock_pos=np.asarray(start, dtype=np.float64), force_tblock_pos=True)
    done = False
    steps = 0
    info = {"success": False}
    traj: List[np.ndarray] = []
    while not done:
        obs_th = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if obs_th.ndim == 1:
            obs_th = obs_th.unsqueeze(0)
        state = env.obs2state(obs_th)
        traj.append(state.detach().cpu().numpy().reshape(-1)[:4].astype(np.float64))
        z_use = policy.encoder(obs_th) if z is None else z
        action = policy.decoder(torch.cat([state, z_use], dim=-1))
        obs, _, done, info = env.step(action.detach().cpu().numpy()[0])
        steps += 1
        if steps >= int(max_steps):
            done = True
    return bool(info.get("success", False)), traj


# Bin-coverage settings mirror DIVO's 2D diversity metric and
# phase4_monitor.compute_unique_bin_coverage: floor each obstacle (x, y) into a
# grid over [-xy_limit, xy_limit] and count unique occupied cells.
COVERAGE_GRID = 8
COVERAGE_XY_LIMIT = 0.2
REALIZED_EPS = 1e-9


def _coverage_bin_index(value: float, grid: int, xy_limit: float) -> int:
    scaled = (float(value) + xy_limit) / (2.0 * xy_limit)
    idx = int(np.floor(scaled * grid))
    return min(max(idx, 0), grid - 1)


def compute_boundary_coverage(
    per_scene: Sequence[Mapping[str, Any]],
    boundary_flags: Sequence[bool],
    grid: int = COVERAGE_GRID,
    xy_limit: float = COVERAGE_XY_LIMIT,
) -> Dict[str, Any]:
    """Unique 2D obstacle-position cell coverage over learnable-band scenes only.

    Reuses the DIVO cell-counting formula (see
    phase4_monitor.compute_unique_bin_coverage), but (1) restricts to boundary
    scenes so scattering infeasible/trivial layouts cannot inflate diversity, and
    (2) normalizes by the number of boundary scenes so a legitimately harder
    candidate (fewer boundary scenes) is not penalized for a low raw cell count.
    """

    grid = max(1, int(grid))
    occupied = set()
    n_boundary = 0
    n_obstacles = 0
    for row, is_boundary in zip(per_scene, boundary_flags):
        if not bool(is_boundary):
            continue
        n_boundary += 1
        for obs in row.get("obstacles", []) or []:
            try:
                x = float(obs["x"])
                y = float(obs["y"])
            except (KeyError, TypeError, ValueError):
                continue
            occupied.add(
                (
                    _coverage_bin_index(x, grid, xy_limit),
                    _coverage_bin_index(y, grid, xy_limit),
                )
            )
            n_obstacles += 1
    total_bins = grid * grid
    unique_bins = len(occupied)
    return {
        "grid": int(grid),
        "xy_limit": float(xy_limit),
        "n_boundary_scenes": int(n_boundary),
        "n_obstacles": int(n_obstacles),
        "unique_bins": int(unique_bins),
        "total_bins": int(total_bins),
        "coverage_ratio": float(unique_bins / total_bins) if total_bins else 0.0,
        "coverage_per_scene": float(unique_bins / n_boundary) if n_boundary else 0.0,
    }


def compute_boundary_signal(
    per_scene: Sequence[Mapping[str, Any]],
    K: int,
    tau: float = 0.125,
    target_delta: float = 0.15,
    valid_rate: float = 0.0,
    dup_round: int = 2,
    exclude_invalid_from_boundary: bool = False,
) -> Dict[str, Any]:
    """Stage 2 V0 library boundary signal ``b = p(1-p)`` (Task 16 / Req 7.2, 7.5).

    ``p_i = (1/K) Σ_k 1[π_{w_k} solves e_i]`` uses the K probe skills only
    (``w_0`` is diagnostic, never in ``p_i``). Boundary is judged by the τ-band
    ``τ < p_i < 1−τ`` (K=4 -> τ=0.125), kept away from saturation. This is the
    pure SFL-style p(1-p) signal: NO weighted coverage S / frontier bias / cover
    descriptor (V0, Requirement 7.3).

    ``exclude_invalid_from_boundary`` (Edit-1 switch, DEFAULT False = legacy):
    when True, scenes flagged ``invalid=True`` (generator produced no valid
    layout at a fixed start; recorded with realized=0 under
    ``invalid_scene_policy='zero'``) are DROPPED from the boundary statistics
    (r_hard / r_easy / mean_b / boundary_count / duplicate_rate), so that a
    *generator validity* failure is no longer conflated with a *library
    capability* failure. Generator validity is still gated separately via
    ``valid_rate``. When False (default) the legacy behavior is kept: invalid
    scenes count as realized=0 (near-hard).
    """
    n_total = len(per_scene)
    scenes = list(per_scene)
    if exclude_invalid_from_boundary:
        scenes = [r for r in scenes if not r.get("invalid", False)]
    n = len(scenes)
    lo, hi = float(tau), 1.0 - float(tau)
    if n == 0:
        return {
            "K": int(K), "n_scenes": 0, "n_total": int(n_total),
            "tau": float(tau), "target_delta": float(target_delta),
            "boundary_count": 0, "boundary_rate": 0.0,
            "r_hard": 0.0, "r_easy": 0.0, "target_rate": 0.0, "mean_b": 0.0,
            "valid_rate": float(valid_rate), "duplicate_rate": 1.0, "w0_success_rate": 0.0,
            "p_values": [],
        }

    p = np.asarray([float(r["realized"]) for r in scenes], dtype=np.float64)
    b = np.asarray([float(r["lv"]) for r in scenes], dtype=np.float64)  # b_i = p_i(1-p_i)
    deployed = np.asarray([float(r.get("deployed", 0)) for r in scenes], dtype=np.float64)

    boundary_mask = (p > lo) & (p < hi)
    boundary_count = int(np.sum(boundary_mask))

    def _sig(obstacles):
        return tuple(sorted(
            (round(float(o["x"]), int(dup_round)), round(float(o["y"]), int(dup_round)))
            for o in obstacles
        ))

    sigs = [_sig(r.get("obstacles", []) or []) for r in scenes]
    duplicate_rate = float(1.0 - len(set(sigs)) / float(n))

    return {
        "K": int(K),
        "n_scenes": int(n),
        "n_total": int(n_total),
        "tau": float(tau),
        "target_delta": float(target_delta),
        "boundary_count": int(boundary_count),
        "boundary_rate": float(boundary_count / n),
        "r_hard": float(np.mean(p <= lo)),
        "r_easy": float(np.mean(p >= hi)),
        "target_rate": float(np.mean(np.abs(p - 0.5) <= float(target_delta))),
        "mean_b": float(np.mean(b)),
        "valid_rate": float(valid_rate),
        "duplicate_rate": duplicate_rate,
        "w0_success_rate": float(np.mean(deployed)),
        "p_values": [float(x) for x in p.tolist()],
    }


def compute_difficulty_profile(
    per_scene: Sequence[Mapping[str, Any]],
    K: int,
    hist_bins: Sequence[float],
    sampling_stats: SceneSamplingStats,
) -> Dict[str, Any]:
    feasible = np.asarray([float(row["feasible"]) for row in per_scene], dtype=np.float64)
    realized = np.asarray([float(row["realized"]) for row in per_scene], dtype=np.float64)
    deployed = np.asarray([float(row["deployed"]) for row in per_scene], dtype=np.float64)
    counts, bins = np.histogram(realized, bins=np.asarray(hist_bins, dtype=np.float64))

    # K-natural difficulty bands (no hand-tuned thresholds): with K injected
    # skills, realized is a multiple of 1/K. A scene is infeasible when no skill
    # solves it (realized == 0), trivial when every skill solves it
    # (realized == 1), and on the learnable boundary otherwise (some skills solve,
    # some fail). This matches CoEvolve's "mixed outcomes" / SFL's "solved
    # sometimes but not always".
    if realized.size:
        infeasible_mask = feasible <= 0.0
        trivial_mask = realized >= (1.0 - REALIZED_EPS)
        boundary_mask = (~infeasible_mask) & (~trivial_mask) & (realized > REALIZED_EPS)
        frac_infeasible = float(np.mean(infeasible_mask))
        frac_trivial = float(np.mean(trivial_mask))
        frac_boundary = float(np.mean(boundary_mask))
    else:
        boundary_mask = np.zeros(0, dtype=bool)
        frac_infeasible = 0.0
        frac_trivial = 0.0
        frac_boundary = 0.0

    boundary_coverage = compute_boundary_coverage(per_scene, boundary_mask.tolist())

    return {
        "n_scenes": int(len(per_scene)),
        "K": int(K),
        "mean_realized": float(realized.mean()) if realized.size else 0.0,
        "frac_infeasible": frac_infeasible,
        "frac_trivial": frac_trivial,
        "frac_boundary": frac_boundary,
        "mean_feasible": float(feasible.mean()) if feasible.size else 0.0,
        "mean_deployed": float(deployed.mean()) if deployed.size else 0.0,
        "mean_lv": float(np.mean([float(row["lv"]) for row in per_scene])) if per_scene else 0.0,
        "boundary_coverage": boundary_coverage,
        "realized_hist": {
            "bins": [float(v) for v in bins.tolist()],
            "counts": [int(v) for v in counts.tolist()],
        },
        "sampling": sampling_stats.to_dict(),
    }


def build_design_context(
    per_scene: Sequence[Mapping[str, Any]],
    n: int,
    env: Any,
    route_waypoints: int,
    include_behavior: bool = True,
    tau_saturation: float = 0.125,
    strict_boundary_bins: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    scenes = list(per_scene)
    if strict_boundary_bins:
        scenes = [row for row in scenes if not bool(row.get("invalid", False))]
        lo, hi = float(tau_saturation), 1.0 - float(tau_saturation)
        focus = sorted(
            (row for row in scenes if lo < float(row["realized"]) < hi),
            key=lambda row: abs(float(row["realized"]) - 0.5),
        )[:n]
        harden = sorted(
            (row for row in scenes if float(row["realized"]) >= hi),
            key=lambda row: float(row["realized"]),
            reverse=True,
        )[:n]
        avoid = sorted(
            (row for row in scenes if float(row["realized"]) <= lo),
            key=lambda row: float(row["realized"]),
        )[:n]
    else:
        focus = sorted(
            scenes, key=lambda row: abs(float(row["realized"]) - 0.5)
        )[:n]
        harden = sorted(
            scenes, key=lambda row: float(row["realized"]), reverse=True
        )[:n]
        avoid = [row for row in scenes if int(row["feasible"]) == 0][:n]
    return {
        "focus": [
            summarize_scene_for_context(
                row,
                env=env,
                route_waypoints=route_waypoints,
                include_behavior=include_behavior,
            )
            for row in focus
        ],
        "harden": [
            summarize_scene_for_context(
                row,
                env=env,
                route_waypoints=route_waypoints,
                include_behavior=include_behavior,
            )
            for row in harden
        ],
        "avoid": [
            summarize_scene_for_context(
                row,
                env=env,
                route_waypoints=route_waypoints,
                include_behavior=include_behavior,
            )
            for row in avoid
        ],
    }


def summarize_scene_for_context(
    scene: Mapping[str, Any],
    env: Any,
    route_waypoints: int,
    include_behavior: bool = True,
) -> Dict[str, Any]:
    summary = {
        "scene_id": int(scene["scene_id"]),
        "start": _to_float_list(scene["start"]),
        "obstacles": normalize_obstacles(list(scene["obstacles"])),
        "feasible": int(scene["feasible"]),
        "realized": float(scene["realized"]),
        "deployed": int(scene["deployed"]),
        "lv": float(scene["lv"]),
    }
    if include_behavior:
        summary["behavior_summary"] = make_behavior_summary(
            scene=scene,
            env=env,
            max_waypoints=int(route_waypoints),
        )
    return summary


def make_behavior_summary(
    scene: Mapping[str, Any],
    env: Any,
    max_waypoints: int = 5,
) -> Dict[str, Any]:
    routes = list(scene.get("routes", []))
    success_routes = [route for route in routes if bool(route.get("success"))]
    failure_routes = [route for route in routes if not bool(route.get("success"))]
    selected_success = select_diverse_routes(success_routes, max_count=2)
    selected_failure = select_failure_routes(failure_routes, selected_success, max_count=1)
    selected = selected_success + selected_failure
    return {
        "requested": {"success": 2, "failure": 1},
        "actual": {
            "success": int(sum(1 for route in selected if bool(route.get("success")))),
            "failure": int(sum(1 for route in selected if not bool(route.get("success")))),
        },
        "routes": [
            route_to_summary(
                route=route,
                start=scene["start"],
                env=env,
                max_waypoints=max_waypoints,
            )
            for route in selected
        ],
    }


def select_diverse_routes(
    routes: Sequence[Mapping[str, Any]],
    max_count: int,
) -> List[Mapping[str, Any]]:
    routes = list(routes)
    if len(routes) <= int(max_count):
        return routes
    if int(max_count) <= 1:
        return routes[: int(max_count)]

    embeddings = [_embed_route(route.get("states", [])) for route in routes]
    best_pair = (0, 1)
    best_dist = -1.0
    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            dist = float(np.linalg.norm(embeddings[i] - embeddings[j]))
            if dist > best_dist:
                best_dist = dist
                best_pair = (i, j)
    return [routes[best_pair[0]], routes[best_pair[1]]][: int(max_count)]


def select_failure_routes(
    failure_routes: Sequence[Mapping[str, Any]],
    selected_success: Sequence[Mapping[str, Any]],
    max_count: int,
) -> List[Mapping[str, Any]]:
    failures = list(failure_routes)
    if len(failures) <= int(max_count):
        return failures
    if int(max_count) <= 0:
        return []
    if not selected_success:
        return failures[: int(max_count)]

    success_embs = [_embed_route(route.get("states", [])) for route in selected_success]
    success_center = np.mean(np.stack(success_embs, axis=0), axis=0)
    ranked = sorted(
        failures,
        key=lambda route: float(np.linalg.norm(_embed_route(route.get("states", [])) - success_center)),
        reverse=True,
    )
    return ranked[: int(max_count)]


def route_to_summary(
    route: Mapping[str, Any],
    start: Sequence[float],
    env: Any,
    max_waypoints: int,
) -> Dict[str, Any]:
    states = list(route.get("states", []))
    indices = downsample_indices(len(states), max_waypoints=max_waypoints)
    return {
        "skill_index": int(route.get("skill_index", -1)),
        "success": bool(route.get("success", False)),
        "waypoints": [
            state_to_waypoint(
                state=states[idx],
                step_index=idx,
                start=start,
                env=env,
            )
            for idx in indices
        ],
    }


def state_to_waypoint(
    state: Sequence[float],
    step_index: int,
    start: Sequence[float],
    env: Any,
) -> Dict[str, float]:
    arr = np.asarray(state, dtype=np.float64).reshape(-1)
    desk_size = get_desk_size(env)
    world_x = float(arr[0] * desk_size)
    world_y = float(arr[1] * desk_size)
    cos_theta = float(arr[2]) if arr.size > 2 else 1.0
    sin_theta = float(arr[3]) if arr.size > 3 else 0.0
    theta = float(np.arctan2(sin_theta, cos_theta))
    z = encode_xy_to_z(
        world_x,
        world_y,
        effective_radius=0.0,
        start_xy=np.asarray(start, dtype=np.float64).reshape(-1)[:2],
        goal_xy=PUSHT_TARGET_XY,
        corridor_width=PUSHT_CORRIDOR_WIDTH,
    )
    return {
        "step": int(step_index),
        "x": world_x,
        "y": world_y,
        "theta": theta,
        "alpha": float(z.alpha),
        "beta": float(z.beta),
    }


def get_desk_size(env: Any) -> float:
    task = getattr(env, "task", None) or getattr(env, "_task", None)
    if task is not None and hasattr(task, "_desk_size"):
        return float(task._desk_size)
    return 1.0


def downsample_indices(length: int, max_waypoints: int) -> List[int]:
    length = int(length)
    max_waypoints = int(max_waypoints)
    if length <= 0 or max_waypoints <= 0:
        return []
    if length <= max_waypoints:
        return list(range(length))
    return sorted({int(round(v)) for v in np.linspace(0, length - 1, max_waypoints)})


def _embed_route(states: Sequence[Sequence[float]], max_steps: int = 10) -> np.ndarray:
    arr = np.asarray(states, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros(max_steps * 4, dtype=np.float64)
    arr = arr[:, :4]
    if arr.shape[0] < max_steps:
        pad = np.repeat(arr[-1:], max_steps - arr.shape[0], axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    else:
        arr = arr[:max_steps]
    return arr.reshape(-1)


def _to_float_list(value: Any) -> List[float]:
    return [float(v) for v in np.asarray(value, dtype=np.float64).reshape(-1).tolist()]


def load_pusht_generator_executor(
    generator_path: str | pathlib.Path,
    obstacle_size: float,
) -> SandboxPushTExecutor:
    path = pathlib.Path(generator_path).expanduser().resolve()
    code = path.read_text(encoding="utf-8")
    executor = SandboxPushTExecutor(obstacle_size=float(obstacle_size))
    if not executor.load(code):
        raise RuntimeError(f"Failed to load generator code from {path}")
    return executor


def build_pusht_env_for_probe(obstacle_num: int, obstacle_size: float) -> Any:
    from DIVO.env import get_env_class

    return get_env_class(
        _target_="pusht_mujoco_llm",
        obstacle=True,
        obstacle_num=int(obstacle_num),
        obstacle_size=float(obstacle_size),
        obstacle_shape="box",
        obstacle_dist="between",
        action_scale=4,
        NUM_SUBSTEPS=25,
        action_dim=[6],
        obs_dim=[4 + 2 * int(obstacle_num)],
        action_reg=True,
        reg_coeff=1.0,
        dynamics_randomization=False,
    )


def load_policy_for_probe(env: Any, ckpt_path: str, device: torch.device | str) -> Tuple[Any, int]:
    from omegaconf import OmegaConf

    from DIVO.policy import get_policy

    ckpt_path = str(ckpt_path)
    run_dir = ckpt_path.rsplit("/", 2)[0]
    p_cfg = OmegaConf.load(f"{run_dir}/.hydra/config.yaml")
    policy = get_policy(env, **p_cfg.policy).to(torch.device(device))
    policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
    policy.eval()
    latent_dim = int(p_cfg.policy.encoder_net.out_chan)
    return policy, latent_dim


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--obstacle_num", type=int, default=2)
    parser.add_argument("--obstacle_size", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=None)
    parser.add_argument("--progress_every", type=int, default=1)
    args = parser.parse_args(argv)

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    env = build_pusht_env_for_probe(args.obstacle_num, args.obstacle_size)
    policy, latent_dim = load_policy_for_probe(env, args.ckpt, device)
    generator = load_pusht_generator_executor(args.generator, args.obstacle_size)
    result = extract_skill_signals(
        env=env,
        policy=policy,
        generator=generator,
        device=device,
        latent_dim=latent_dim,
        K=args.K,
        M=args.M,
        n=args.n,
        num_obstacles=args.obstacle_num,
        max_steps=args.max_steps,
        seed=args.seed,
        progress_every=args.progress_every,
    )

    print("\n===== Skill Signal Difficulty Profile =====")
    print(json.dumps(result["difficulty_profile"], indent=2, ensure_ascii=False))
    print("\n===== Design Context Preview =====")
    for key, rows in result["design_context"].items():
        print(f"\n[{key}] {len(rows)} scenes")
        for row in rows:
            print(
                f"- scene={row['scene_id']} realized={row['realized']:.3f} "
                f"feasible={row['feasible']} deployed={row['deployed']} "
                f"routes={row['behavior_summary']['actual']}"
            )
            print(f"  start={np.round(np.asarray(row['start']), 3).tolist()}")
            print(f"  obstacles={row['obstacles']}")

    if args.out:
        out_path = pathlib.Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        print(f"\nsaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
