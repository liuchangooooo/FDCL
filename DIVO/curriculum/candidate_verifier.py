from __future__ import annotations

import importlib
import json
import math
import signal
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from DIVO.curriculum.adapters.pusht_adapter import (
    PUSHT_CORRIDOR_WIDTH,
    PUSHT_TARGET_XY,
    pusht_effective_radius,
    pusht_encode_layout,
)
from DIVO.curriculum.attribution import AttributionConfig, cell_key
from DIVO.curriculum.learnable_frontier import (
    evaluate_learnable_frontier_shift,
    select_generator_boundary,
)
from DIVO.curriculum.obstacle_geometry import decode_z_to_xy, encode_xy_to_z


@dataclass(frozen=True)
class CandidateVerifierConfig:
    num_pose_samples: int = 500
    obstacle_num: int = 2
    seed: int = 42
    success_rate: float = 0.0
    success_low: float = 0.20
    success_high: float = 0.80
    min_valid_generation_rate: float = 0.95
    timeout_sec: int = 5
    top_cells_to_report: int = 8
    top_sampled_to_report: int = 8


@dataclass(frozen=True)
class SkillVerifierConfig:
    use_verifier: bool = True
    obstacle_num: int = 2
    obstacle_size: float = 0.01
    probe_M: int = 30
    probe_K: int = 8
    probe_n: int = 0
    max_steps: int = 10
    seed: int = 42
    infeasible_cap: float = 0.30
    min_score_delta: float = 0.0
    min_valid_generation_rate: float = 0.95
    validity_num_pose_samples: int = 100
    timeout_sec: int = 5
    route_waypoints: int = 5
    # --- Stage 2 V0 (Task 17 / Requirement 8): single-scalar boundary verifier ---
    # `criterion` selects the acceptance rule. Defaults are LEGACY-safe so the
    # existing skill_signal/boundary_skill/ours modes are unchanged; the Route B
    # V0 (`feedback_mode='skill_library'`) path sets these explicitly.
    criterion: str = "learnable_frontier_shift"
    probe_source: str = "random_z"      # skill-library probe (Task 16) when 'w_probe'
    tau_saturation: float = 0.125       # boundary band: tau < p_i < 1-tau
    target_delta: float = 0.15          # near-0.5 band (diagnostic)
    valid_rate_min: float = 0.8         # absolute hard gate (>= 0.8, NOT 0.95)
    duplicate_rate_max: float = 0.25    # absolute hard gate
    min_boundary_count_delta: int = 4   # net-improvement margin (count version)
    J_epsilon: float = 0.05             # equivalent float margin on boundary_rate
    r_easy_max: float = 0.8             # symmetric saturation escape (too easy)
    r_hard_max: float = 0.8             # symmetric saturation escape (too hard)
    candidates_per_cycle: int = 3       # R candidates per evolve cycle
    eval_layouts_M: int = 64            # scenes per candidate for boundary eval
    # Edit-1 switch (DEFAULT False = legacy): drop invalid scenes (generator made
    # no valid layout at a fixed start, recorded realized=0 under 'zero') from the
    # boundary stats so generator-invalidity is not conflated with library failure.
    # Validity is still gated by valid_rate_min. Turn on only after the pure-revert
    # baseline is measured.
    exclude_invalid_from_boundary: bool = False
    # PRESERVE_AND_DIVERSIFY swap (points 2/3, DEFAULT OFF = legacy hold behavior):
    # on a would-be HOLD (not saturated, no boundary_count net-improve), rotate to
    # a structurally-fresh candidate at equal difficulty (bc >= cur - tolerance,
    # r_easy not worse) instead of freezing G_t. When diversify_prefer_lower_easy
    # and current r_easy > diversify_r_easy_soft, prefer the lowest-r_easy eligible
    # candidate (gentle continuous re-harden, avoids the r_easy_max dead zone).
    diversify_on_hold: bool = False
    diversify_bc_tolerance: int = 2
    diversify_prefer_lower_easy: bool = False
    diversify_r_easy_soft: float = 0.5
    diversify_easy_eps: float = 0.05


@dataclass
class SkillGeneratorScore:
    score: float
    frac_infeasible: float
    mean_realized: float
    mean_feasible: float
    mean_deployed: float
    valid_generation_rate: float
    frac_trivial: float = 0.0
    frac_boundary: float = 0.0
    coverage_per_scene: float = 0.0
    difficulty_profile: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SkillCandidateVerificationResult:
    accepted: bool
    reason: str
    current: SkillGeneratorScore
    candidate: SkillGeneratorScore
    score_delta: float
    feedback_text: str
    frontier_shift: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CellEvidenceValue:
    cell_id: str
    value: float
    rank: int
    failure_lift: float
    total_count: int
    failure_count: int
    p_cell: float = 0.0
    failure_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GeneratorAuditSummary:
    score: float
    num_pose_samples: int
    accepted_generations: int
    empty_generations: int
    invalid_generations: int
    wrong_count_generations: int
    num_obstacle_samples: int
    occupied_cells: int
    cell_coverage: Dict[str, float] = field(default_factory=dict)
    top_sampled_cells: List[Dict[str, Any]] = field(default_factory=list)
    invalid_reason_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def valid_generation_rate(self) -> float:
        if self.num_pose_samples <= 0:
            return 0.0
        return float(self.accepted_generations / self.num_pose_samples)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateVerificationResult:
    accepted: bool
    direction: str
    reason: str
    current: GeneratorAuditSummary
    candidate: GeneratorAuditSummary
    score_delta: float
    cell_values: List[CellEvidenceValue]
    feedback_text: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["cell_values"] = [cell.to_dict() for cell in self.cell_values]
        return data


def verify_pusht_candidate(
    current_generator_code: str,
    candidate_generator_code: str,
    attribution_result: Mapping[str, Any] | Any,
    config: CandidateVerifierConfig,
) -> CandidateVerificationResult:
    """Verify a Push-T candidate generator against attribution-derived evidence."""

    result_dict = _as_mapping(attribution_result)
    attribution_cfg = AttributionConfig.from_dict(
        result_dict.get("metadata", {}).get("config", {})
    )
    cell_values = build_rank_evidence_values(result_dict, attribution_cfg)
    poses = sample_pusht_poses(config.num_pose_samples, config.seed)

    current = audit_pusht_generator_code(
        current_generator_code,
        poses=poses,
        obstacle_num=config.obstacle_num,
        attribution_config=attribution_cfg,
        evidence_values={cell.cell_id: cell.value for cell in cell_values},
        timeout_sec=config.timeout_sec,
        top_sampled_to_report=config.top_sampled_to_report,
    )
    candidate = audit_pusht_generator_code(
        candidate_generator_code,
        poses=poses,
        obstacle_num=config.obstacle_num,
        attribution_config=attribution_cfg,
        evidence_values={cell.cell_id: cell.value for cell in cell_values},
        timeout_sec=config.timeout_sec,
        top_sampled_to_report=config.top_sampled_to_report,
    )

    direction = evidence_direction(
        success_rate=config.success_rate,
        success_low=config.success_low,
        success_high=config.success_high,
    )
    delta = candidate.score - current.score
    accepted, reason = decide_acceptance(
        direction=direction,
        current_score=current.score,
        candidate_score=candidate.score,
        candidate_valid_rate=candidate.valid_generation_rate,
        min_valid_generation_rate=config.min_valid_generation_rate,
        has_evidence=bool(cell_values),
    )
    feedback_text = build_feedback_text(
        accepted=accepted,
        reason=reason,
        direction=direction,
        current=current,
        candidate=candidate,
        score_delta=delta,
        cell_values=cell_values,
        max_cells=config.top_cells_to_report,
    )
    return CandidateVerificationResult(
        accepted=accepted,
        direction=direction,
        reason=reason,
        current=current,
        candidate=candidate,
        score_delta=delta,
        cell_values=cell_values,
        feedback_text=feedback_text,
    )


class SkillCandidateVerifierSession:
    """Paired skill-signal verifier for one ACGS evolve round.

    The current generator is probed exactly once on a fixed batch of starts and
    latent skills.  Every candidate attempt is scored on the same batch, making
    score_delta a paired comparison against the original generator for this
    evolve round.
    """

    def __init__(
        self,
        env: Any,
        policy: Any,
        device: Any,
        latent_dim: int,
        current_generator_code: str,
        config: SkillVerifierConfig,
        current_signal_result: Optional[Mapping[str, Any]] = None,
    ):
        self.env = env
        self.policy = policy
        self.device = device
        self.latent_dim = int(latent_dim)
        self.current_generator_code = str(current_generator_code or "")
        self.config = config
        self.current_signal_result = current_signal_result
        self.probe_source = str(getattr(config, "probe_source", "random_z")).lower()
        self.criterion = str(getattr(config, "criterion", "learnable_frontier_shift")).lower()

        from DIVO.curriculum.skill_signal import build_z_bank, sample_fixed_starts

        self.fixed_starts = sample_fixed_starts(
            env=self.env,
            M=int(self.config.probe_M),
            seed=int(self.config.seed),
        )
        if self.probe_source == "w_probe":
            # Route B V0: probe with the trained skill library; K = policy.K.
            if not getattr(self.policy, "skill_enabled", False):
                raise RuntimeError(
                    "SkillVerifierConfig.probe_source='w_probe' requires a skill_enabled policy"
                )
            self.z_bank = None
        else:
            self.z_bank = build_z_bank(
                K=int(self.config.probe_K),
                latent_dim=self.latent_dim,
                seed=int(self.config.seed),
                device=self.device,
            )
        self.current_score = self._score_code(
            code=self.current_generator_code,
            label="current",
            cached_signal_result=current_signal_result,
        )

    def verify(self, candidate_code: str, attempt_idx: int = 1) -> SkillCandidateVerificationResult:
        if not self.config.use_verifier:
            candidate = self.current_score
            return SkillCandidateVerificationResult(
                accepted=True,
                reason="verifier_disabled",
                current=self.current_score,
                candidate=candidate,
                score_delta=0.0,
                feedback_text="Skill verifier disabled; candidate accepted without rollout verification.",
            )

        if self.criterion == "boundary_count_argmax_escape":
            return self._verify_boundary(candidate_code, attempt_idx)

        valid_rate, validity_profile = self._audit_valid_generation_rate(candidate_code)
        if valid_rate < float(self.config.min_valid_generation_rate):
            candidate = SkillGeneratorScore(
                score=0.0,
                frac_infeasible=1.0,
                mean_realized=0.0,
                mean_feasible=0.0,
                mean_deployed=0.0,
                valid_generation_rate=float(valid_rate),
                difficulty_profile={
                    "validity_audit": validity_profile,
                    "sampling": {"valid_scene_rate": float(valid_rate)},
                },
            )
            result = SkillCandidateVerificationResult(
                accepted=False,
                reason="candidate_low_valid_generation_rate",
                current=self.current_score,
                candidate=candidate,
                score_delta=float(candidate.score - self.current_score.score),
                feedback_text=build_skill_feedback_text(
                    accepted=False,
                    reason="candidate_low_valid_generation_rate",
                    current=self.current_score,
                    candidate=candidate,
                    score_delta=float(candidate.score - self.current_score.score),
                    config=self.config,
                ),
            )
            return result

        candidate = self._score_code(
            code=candidate_code,
            label=f"candidate_{attempt_idx}",
            cached_signal_result=None,
        )
        delta = float(candidate.score - self.current_score.score)
        accepted, reason, frontier_shift = decide_skill_acceptance(
            current=self.current_score,
            candidate=candidate,
            min_valid_generation_rate=float(self.config.min_valid_generation_rate),
            infeasible_cap=float(self.config.infeasible_cap),
            min_score_delta=float(self.config.min_score_delta),
        )
        feedback_text = build_skill_feedback_text(
            accepted=accepted,
            reason=reason,
            current=self.current_score,
            candidate=candidate,
            score_delta=delta,
            config=self.config,
            frontier_shift=frontier_shift,
        )
        return SkillCandidateVerificationResult(
            accepted=accepted,
            reason=reason,
            current=self.current_score,
            candidate=candidate,
            score_delta=delta,
            feedback_text=feedback_text,
            frontier_shift=frontier_shift,
        )

    def _verify_boundary(self, candidate_code: str, attempt_idx: int) -> SkillCandidateVerificationResult:
        """Route B V0 acceptance (Task 17/18): single-scalar boundary_count argmax
        against the incumbent G_t + symmetric saturation escape. The candidate is
        probed with W_probe on the SAME fixed starts as the current generator
        (paired), then judged by ``decide_boundary_selection`` (absolute hard gate
        on valid_rate/duplicate + boundary_count net-improve or saturation escape).
        """
        candidate = self._score_code(
            code=candidate_code,
            label=f"candidate_{attempt_idx}",
            cached_signal_result=None,
        )
        cur_sig = dict(self.current_score.difficulty_profile.get("boundary_signal", {}) or {})
        cand_sig = dict(candidate.difficulty_profile.get("boundary_signal", {}) or {})
        decision = decide_boundary_selection(
            cur_sig,
            [cand_sig],
            self.config,
            candidate_fresh_flags=[
                candidate_code.strip() != self.current_generator_code.strip()
            ],
        )
        accepted = bool(decision.get("action") == "replace")
        reason = str(decision.get("reason", "boundary_decision"))
        cur_bc = int(cur_sig.get("boundary_count", 0))
        cand_bc = int(cand_sig.get("boundary_count", 0))
        delta = float(cand_bc - cur_bc)
        feedback_text = (
            f"[boundary V0] action={decision.get('action')} reason={reason} "
            f"saturated={decision.get('saturated')} dir={decision.get('saturation_direction')} | "
            f"G_t: bc={cur_bc} r_easy={cur_sig.get('r_easy', 0.0):.2f} r_hard={cur_sig.get('r_hard', 0.0):.2f} "
            f"| cand: bc={cand_bc} valid={cand_sig.get('valid_rate', 0.0):.2f} "
            f"dup={cand_sig.get('duplicate_rate', 1.0):.2f} mean_b={cand_sig.get('mean_b', 0.0):.3f} "
            f"r_easy={cand_sig.get('r_easy', 0.0):.2f} r_hard={cand_sig.get('r_hard', 0.0):.2f} "
            f"(need bc>={cur_bc}+{self.config.min_boundary_count_delta} unless saturated)."
        )
        return SkillCandidateVerificationResult(
            accepted=accepted,
            reason=reason,
            current=self.current_score,
            candidate=candidate,
            score_delta=delta,
            feedback_text=feedback_text,
            frontier_shift=decision,
        )

    def _audit_valid_generation_rate(self, code: str) -> Tuple[float, Dict[str, Any]]:
        executor = SandboxPushTExecutor(obstacle_size=float(self.config.obstacle_size))
        if not executor.load(code):
            return 0.0, {"load_failed": True, "accepted": 0, "attempts": 1}

        starts = self.fixed_starts[: max(1, min(int(self.config.validity_num_pose_samples), len(self.fixed_starts)))]
        accepted = 0
        empty = 0
        wrong_count = 0
        invalid = 0
        contact_rejects = 0
        for start in starts:
            obstacles = executor.generate(
                np.asarray(start, dtype=np.float64),
                int(self.config.obstacle_num),
                timeout_sec=int(self.config.timeout_sec),
            )
            if not obstacles:
                empty += 1
                continue
            if len(obstacles) != int(self.config.obstacle_num):
                wrong_count += 1
                continue
            if hasattr(self.env, "is_obstacle_config_valid") and not self.env.is_obstacle_config_valid(obstacles, start):
                invalid += 1
                continue
            if hasattr(self.env, "set_obstacle_config"):
                self.env.set_obstacle_config(obstacles)
            if hasattr(self.env, "reset"):
                self.env.reset(tblock_pos=np.asarray(start, dtype=np.float64), force_tblock_pos=True)
            if hasattr(self.env, "get_ncon") and int(self.env.get_ncon()) != 0:
                contact_rejects += 1
                continue
            accepted += 1
        attempts = max(len(starts), 1)
        return float(accepted / attempts), {
            "attempts": int(attempts),
            "accepted": int(accepted),
            "empty_generations": int(empty),
            "wrong_count_generations": int(wrong_count),
            "invalid_generations": int(invalid),
            "contact_rejects": int(contact_rejects),
            "valid_generation_rate": float(accepted / attempts),
        }

    def _score_code(
        self,
        code: str,
        label: str,
        cached_signal_result: Optional[Mapping[str, Any]],
    ) -> SkillGeneratorScore:
        if cached_signal_result is not None:
            cached = _as_mapping(cached_signal_result)
            cached_config = cached.get("probe_config", {})
            if self._probe_config_matches(cached_config):
                score = score_from_skill_signal(cached_signal_result)
                # Carry the Stage 2 V0 boundary signal so reused G_t scores expose
                # boundary_count/r_easy/r_hard to decide_boundary_selection.
                score.difficulty_profile["boundary_signal"] = dict(
                    cached.get("boundary_signal", {}) or {}
                )
                return score

        executor = SandboxPushTExecutor(obstacle_size=float(self.config.obstacle_size))
        if not executor.load(code):
            profile = {
                "label": label,
                "load_failed": True,
                "sampling": {"valid_scene_rate": 0.0},
                "mean_lv": 0.0,
                "frac_infeasible": 1.0,
                "mean_realized": 0.0,
                "mean_feasible": 0.0,
                "mean_deployed": 0.0,
            }
            return SkillGeneratorScore(
                score=0.0,
                frac_infeasible=1.0,
                mean_realized=0.0,
                mean_feasible=0.0,
                mean_deployed=0.0,
                valid_generation_rate=0.0,
                difficulty_profile=profile,
            )

        from DIVO.curriculum.skill_signal import extract_skill_signals

        was_training = bool(getattr(self.policy, "training", False))
        try:
            signal = extract_skill_signals(
                env=self.env,
                policy=self.policy,
                generator=executor,
                device=self.device,
                latent_dim=self.latent_dim,
                K=int(self.config.probe_K),
                M=int(self.config.probe_M),
                n=int(self.config.probe_n),
                num_obstacles=int(self.config.obstacle_num),
                max_steps=int(self.config.max_steps),
                seed=int(self.config.seed),
                generator_timeout_sec=int(self.config.timeout_sec),
                route_waypoints=int(self.config.route_waypoints),
                fixed_starts=self.fixed_starts,
                z_bank=self.z_bank,
                invalid_scene_policy="zero",
                progress_every=0,
                probe_source=self.probe_source,
                tau_saturation=float(getattr(self.config, "tau_saturation", 0.125)),
                target_delta=float(getattr(self.config, "target_delta", 0.15)),
                exclude_invalid_from_boundary=bool(
                    getattr(self.config, "exclude_invalid_from_boundary", False)
                ),
            )
        finally:
            if was_training and hasattr(self.policy, "train"):
                self.policy.train()
        signal["probe_config"] = self.probe_config_dict()
        score = score_from_skill_signal(signal)
        # Carry the Stage 2 V0 boundary signal (Task 16) for boundary selection.
        score.difficulty_profile["boundary_signal"] = dict(signal.get("boundary_signal", {}) or {})
        return score

    def _probe_config_matches(self, cached_config: Mapping[str, Any]) -> bool:
        if not cached_config:
            return False
        expected = self.probe_config_dict()
        for key in (
            "probe_M",
            "probe_K",
            "obstacle_num",
            "obstacle_size",
            "max_steps",
            "seed",
        ):
            if cached_config.get(key) != expected.get(key):
                return False
        return True

    def probe_config_dict(self) -> Dict[str, Any]:
        return {
            "probe_M": int(self.config.probe_M),
            "probe_K": int(self.config.probe_K),
            "obstacle_num": int(self.config.obstacle_num),
            "obstacle_size": float(self.config.obstacle_size),
            "max_steps": int(self.config.max_steps),
            "seed": int(self.config.seed),
        }


def score_from_skill_signal(signal_result: Mapping[str, Any]) -> SkillGeneratorScore:
    result = _as_mapping(signal_result)
    profile = dict(_as_mapping(result.get("difficulty_profile", {})))
    per_scene = list(result.get("per_scene", []) or [])
    score = float(profile.get("mean_lv", 0.0))
    if per_scene and "mean_lv" not in profile:
        score = float(np.mean([float(row.get("lv", 0.0)) for row in per_scene]))
    sampling = _as_mapping(profile.get("sampling", {}))
    return SkillGeneratorScore(
        score=float(score),
        frac_infeasible=float(profile.get("frac_infeasible", 0.0)),
        mean_realized=float(profile.get("mean_realized", 0.0)),
        mean_feasible=float(profile.get("mean_feasible", 0.0)),
        mean_deployed=float(profile.get("mean_deployed", 0.0)),
        valid_generation_rate=float(sampling.get("valid_scene_rate", 0.0)),
        frac_trivial=float(profile.get("frac_trivial", 0.0)),
        frac_boundary=float(profile.get("frac_boundary", 0.0)),
        coverage_per_scene=float(
            _as_mapping(profile.get("boundary_coverage", {})).get("coverage_per_scene", 0.0)
        ),
        difficulty_profile=profile,
    )


def decide_skill_acceptance(
    current: SkillGeneratorScore,
    candidate: SkillGeneratorScore,
    min_valid_generation_rate: float,
    infeasible_cap: float,
    min_score_delta: float,
) -> Tuple[bool, str, Dict[str, Any]]:
    if candidate.valid_generation_rate < float(min_valid_generation_rate):
        return False, "candidate_low_valid_generation_rate", {}
    frontier_shift = evaluate_learnable_frontier_shift(
        _score_profile(current),
        _score_profile(candidate),
        min_lv_delta=float(min_score_delta),
        infeasible_cap=float(infeasible_cap),
    )
    return (
        bool(frontier_shift["accepted"]),
        str(frontier_shift["reason"]),
        frontier_shift,
    )


def _score_profile(score: SkillGeneratorScore) -> Dict[str, float]:
    return {
        "mean_lv": float(score.score),
        "mean_realized": float(score.mean_realized),
        "frac_infeasible": float(score.frac_infeasible),
        "frac_trivial": float(score.frac_trivial),
        "frac_boundary": float(score.frac_boundary),
        "coverage_per_scene": float(score.coverage_per_scene),
    }


def decide_boundary_selection(
    current_signal: Mapping[str, Any],
    candidate_signals: Sequence[Mapping[str, Any]],
    config: SkillVerifierConfig,
    code_pass_flags: Optional[Sequence[bool]] = None,
    candidate_fresh_flags: Optional[Sequence[bool]] = None,
) -> Dict[str, Any]:
    """Stage 2 V0 acceptance (Task 17 / Requirement 8): single-scalar
    boundary_count argmax over {G_t}∪{passing candidates} + symmetric saturation
    escape. ``*_signal`` are the ``boundary_signal`` dicts from
    ``extract_skill_signals(..., probe_source='w_probe')`` (each carries
    boundary_count / mean_b / r_easy / r_hard / valid_rate / duplicate_rate).
    """
    return select_generator_boundary(
        current_signal,
        list(candidate_signals),
        v_min=float(config.valid_rate_min),
        d_max=float(config.duplicate_rate_max),
        min_boundary_count_delta=int(config.min_boundary_count_delta),
        r_easy_max=float(config.r_easy_max),
        r_hard_max=float(config.r_hard_max),
        code_pass_flags=code_pass_flags,
        candidate_fresh_flags=candidate_fresh_flags,
        diversify_on_hold=bool(getattr(config, "diversify_on_hold", False)),
        diversify_bc_tolerance=int(getattr(config, "diversify_bc_tolerance", 2)),
        diversify_prefer_lower_easy=bool(getattr(config, "diversify_prefer_lower_easy", False)),
        diversify_r_easy_soft=float(getattr(config, "diversify_r_easy_soft", 0.5)),
        diversify_easy_eps=float(getattr(config, "diversify_easy_eps", 0.05)),
    )


def build_skill_feedback_text(
    accepted: bool,
    reason: str,
    current: SkillGeneratorScore,
    candidate: SkillGeneratorScore,
    score_delta: float,
    config: SkillVerifierConfig,
    frontier_shift: Optional[Mapping[str, Any]] = None,
) -> str:
    status = "accepted" if accepted else "rejected"
    frontier_shift = dict(frontier_shift or {})
    lines = [
        "Skill-signal candidate verifier feedback:",
        f"- status: {status}",
        f"- reason: {reason}",
        f"- score = mean(realized * (1 - realized)) over paired probe scenes",
        f"- current_score: {current.score:.6f}",
        f"- candidate_score: {candidate.score:.6f}",
        f"- score_delta(candidate-current): {score_delta:+.6f}",
        f"- current_mean_realized: {current.mean_realized:.6f}",
        f"- candidate_mean_realized: {candidate.mean_realized:.6f}",
        f"- current_frac_infeasible: {current.frac_infeasible:.6f}",
        f"- candidate_frac_infeasible: {candidate.frac_infeasible:.6f}",
        f"- candidate_valid_generation_rate: {candidate.valid_generation_rate:.6f}",
        f"- required_valid_generation_rate: {float(config.min_valid_generation_rate):.6f}",
        f"- infeasible_cap: {float(config.infeasible_cap):.6f}",
        f"- current_frontier_distance: {float(frontier_shift.get('current_frontier_distance', abs(current.mean_realized - 0.5))):.6f}",
        f"- candidate_frontier_distance: {float(frontier_shift.get('candidate_frontier_distance', abs(candidate.mean_realized - 0.5))):.6f}",
        f"- frontier_distance_delta(candidate-current): {float(frontier_shift.get('frontier_distance_delta', abs(candidate.mean_realized - 0.5) - abs(current.mean_realized - 0.5))):+.6f}",
        "",
    ]
    if not accepted:
        if reason == "candidate_low_valid_generation_rate":
            lines.append(
                "Please rewrite the generator so it returns exactly the requested "
                "number of valid, safe obstacles for the current obstacle size."
            )
        elif reason == "candidate_infeasible_cap_exceeded":
            lines.append(
                "Please ease the generator: reduce over-blocking and avoid layouts "
                "where no injected skill can solve the scene."
            )
        elif reason == "candidate_moved_away_from_frontier":
            lines.append(
                "The candidate improved learning value but moved the overall "
                "difficulty center farther from realized ~= 0.5. Rewrite it so "
                "the sampled scenes move toward the skill-library frontier."
            )
        else:
            lines.append(
                "Please rewrite the generator so sampled scenes land closer to the "
                "skill-library learning frontier: not already mastered and not "
                "mostly unsolvable."
            )
    return "\n".join(lines)


def build_rank_evidence_values(
    attribution_result: Mapping[str, Any],
    config: Optional[AttributionConfig] = None,
) -> List[CellEvidenceValue]:
    """Assign rank-normalized values to supported positive-lift cells."""

    cfg = config or AttributionConfig.from_dict(
        attribution_result.get("metadata", {}).get("config", {})
    )
    cells = [
        row
        for row in attribution_result.get("cells", [])
        if int(row.get("total_count", 0)) >= cfg.min_support
        and int(row.get("failure_count", 0)) > 0
        and float(row.get("failure_lift", 0.0)) > 1.0
    ]
    cells.sort(
        key=lambda row: (
            float(row.get("failure_lift", 0.0)),
            int(row.get("failure_count", 0)),
            int(row.get("total_count", 0)),
        ),
        reverse=True,
    )
    total = len(cells)
    values: List[CellEvidenceValue] = []
    for idx, row in enumerate(cells, start=1):
        values.append(
            CellEvidenceValue(
                cell_id=str(row.get("cell_id")),
                value=float((total - idx + 1) / total) if total else 0.0,
                rank=idx,
                failure_lift=float(row.get("failure_lift", 0.0)),
                total_count=int(row.get("total_count", 0)),
                failure_count=int(row.get("failure_count", 0)),
                p_cell=float(row.get("p_cell", 0.0)),
                failure_rate=float(row.get("failure_rate", 0.0)),
            )
        )
    return values


def audit_pusht_generator_code(
    code: str,
    poses: Sequence[np.ndarray],
    obstacle_num: int,
    attribution_config: AttributionConfig,
    evidence_values: Mapping[str, float],
    timeout_sec: int = 5,
    top_sampled_to_report: int = 8,
) -> GeneratorAuditSummary:
    executor = SandboxPushTExecutor()
    if not executor.load(code):
        return GeneratorAuditSummary(
            score=0.0,
            num_pose_samples=len(poses),
            accepted_generations=0,
            empty_generations=len(poses),
            invalid_generations=0,
            wrong_count_generations=0,
            num_obstacle_samples=0,
            occupied_cells=0,
            invalid_reason_counts={"load_failed": 1},
        )

    score_sum = 0.0
    obstacle_samples = 0
    accepted_generations = 0
    empty_generations = 0
    invalid_generations = 0
    wrong_count_generations = 0
    invalid_reason_counts: Counter[str] = Counter()
    cell_counts: Counter[str] = Counter()

    for pose in poses:
        obstacles = executor.generate(pose, obstacle_num, timeout_sec=timeout_sec)
        if not obstacles:
            empty_generations += 1
            continue
        if len(obstacles) != obstacle_num:
            wrong_count_generations += 1
            invalid_generations += 1
            invalid_reason_counts["wrong_obstacle_count"] += 1
            continue

        ok, reason = executor.validate(obstacles, pose)
        if not ok:
            invalid_generations += 1
            invalid_reason_counts[str(reason)] += 1
            continue

        accepted_generations += 1
        for z in pusht_encode_layout(obstacles, pose):
            cell_id = format_cell_id(cell_key(z.to_dict(), attribution_config))
            value = float(evidence_values.get(cell_id, 0.0))
            score_sum += value
            obstacle_samples += 1
            cell_counts[cell_id] += 1

    score = score_sum / obstacle_samples if obstacle_samples else 0.0
    coverage = {
        cell_id: count / obstacle_samples
        for cell_id, count in sorted(cell_counts.items())
    } if obstacle_samples else {}
    top_sampled = [
        {
            "cell_id": cell_id,
            "coverage": count / obstacle_samples if obstacle_samples else 0.0,
            "count": count,
            "evidence_value": float(evidence_values.get(cell_id, 0.0)),
        }
        for cell_id, count in cell_counts.most_common(top_sampled_to_report)
    ]
    return GeneratorAuditSummary(
        score=float(score),
        num_pose_samples=len(poses),
        accepted_generations=int(accepted_generations),
        empty_generations=int(empty_generations),
        invalid_generations=int(invalid_generations),
        wrong_count_generations=int(wrong_count_generations),
        num_obstacle_samples=int(obstacle_samples),
        occupied_cells=len(cell_counts),
        cell_coverage=coverage,
        top_sampled_cells=top_sampled,
        invalid_reason_counts=dict(invalid_reason_counts.most_common(10)),
    )


def evidence_direction(success_rate: float, success_low: float, success_high: float) -> str:
    if float(success_rate) > float(success_high):
        return "increase"
    if float(success_rate) < float(success_low):
        return "decrease"
    return "preserve"


def decide_acceptance(
    direction: str,
    current_score: float,
    candidate_score: float,
    candidate_valid_rate: float,
    min_valid_generation_rate: float,
    has_evidence: bool,
) -> Tuple[bool, str]:
    if not has_evidence:
        return False, "no_positive_evidence_cells"
    if candidate_valid_rate < min_valid_generation_rate:
        return False, "candidate_low_valid_generation_rate"
    if direction == "increase":
        accepted = candidate_score > current_score
        return accepted, "candidate_increased_evidence_score" if accepted else "candidate_did_not_increase_evidence_score"
    if direction == "decrease":
        accepted = candidate_score < current_score
        return accepted, "candidate_decreased_evidence_score" if accepted else "candidate_did_not_decrease_evidence_score"
    accepted = candidate_score >= current_score
    return accepted, "candidate_preserved_evidence_score" if accepted else "candidate_reduced_evidence_score"


def build_feedback_text(
    accepted: bool,
    reason: str,
    direction: str,
    current: GeneratorAuditSummary,
    candidate: GeneratorAuditSummary,
    score_delta: float,
    cell_values: Sequence[CellEvidenceValue],
    max_cells: int = 8,
) -> str:
    status = "accepted" if accepted else "rejected"
    lines = [
        "Candidate distribution audit feedback:",
        f"- status: {status}",
        f"- reason: {reason}",
        f"- audit_direction: {direction}",
        f"- current_evidence_score: {current.score:.6f}",
        f"- candidate_evidence_score: {candidate.score:.6f}",
        f"- score_delta(candidate-current): {score_delta:+.6f}",
        f"- current_valid_generations: {current.accepted_generations}/{current.num_pose_samples}",
        f"- candidate_valid_generations: {candidate.accepted_generations}/{candidate.num_pose_samples}",
        "",
        "High-value attribution cells from the current batch:",
    ]
    if not cell_values:
        lines.append("- none; the candidate could not be verified against attribution evidence")
    else:
        for cell in cell_values[:max_cells]:
            lines.append(
                f"- {cell.cell_id}: value={cell.value:.3f}, "
                f"lift={cell.failure_lift:.3f}, support={cell.total_count}, "
                f"failures={cell.failure_count}, "
                f"current_cov={current.cell_coverage.get(cell.cell_id, 0.0):.3f}, "
                f"candidate_cov={candidate.cell_coverage.get(cell.cell_id, 0.0):.3f}"
            )

    lines.extend(["", "Most sampled cells by the candidate:"])
    if not candidate.top_sampled_cells:
        lines.append("- none")
    else:
        for row in candidate.top_sampled_cells[:max_cells]:
            lines.append(
                f"- {row['cell_id']}: coverage={row['coverage']:.3f}, "
                f"evidence_value={row['evidence_value']:.3f}"
            )

    if not accepted:
        if direction == "increase":
            lines.extend(
                [
                    "",
                    "Please rewrite the generator so its actual sampling distribution "
                    "increases the expected attribution evidence score. Do not only "
                    "describe the change; implement it in the sampling probabilities.",
                ]
            )
        elif direction == "decrease":
            lines.extend(
                [
                    "",
                    "Please rewrite the generator so it reduces excessive sampling of "
                    "the highest-failure evidence cells while preserving feasible, "
                    "useful obstacle structure.",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "Please rewrite the generator so it does not reduce attribution "
                    "evidence coverage and remains compact and feasible.",
                ]
            )
    return "\n".join(lines)


def sample_pusht_poses(num_samples: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    poses = []
    for _ in range(max(0, int(num_samples))):
        x = float(rng.uniform(-0.18, 0.18))
        y = float(rng.uniform(-0.18, 0.18))
        while abs(x) < 0.1 and abs(y) < 0.1:
            x = float(rng.uniform(-0.18, 0.18))
            y = float(rng.uniform(-0.18, 0.18))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        poses.append(np.array([x, y, theta], dtype=np.float64))
    return poses


def format_cell_id(key: Sequence[str]) -> str:
    alpha_bin, beta_abs_bin, blockage_bin = key
    return f"alpha={alpha_bin}|beta_abs={beta_abs_bin}|blockage={blockage_bin}"


def save_verification_artifacts(
    output_dir: Path,
    attempt_idx: int,
    candidate_code: str,
    result: CandidateVerificationResult,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"candidate_{attempt_idx:03d}"
    code_path = output_dir / f"{stem}.py"
    audit_path = output_dir / f"{stem}_audit.json"
    feedback_path = output_dir / f"{stem}_feedback.txt"
    code_path.write_text(candidate_code, encoding="utf-8")
    audit_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    feedback_path.write_text(result.feedback_text + "\n", encoding="utf-8")
    return {
        "code_path": str(code_path),
        "audit_path": str(audit_path),
        "feedback_path": str(feedback_path),
    }


class SandboxPushTExecutor:
    """Small Push-T generator sandbox used by offline candidate verification."""

    def __init__(self, obstacle_size: float = 0.01, target_pose: Optional[Sequence[float]] = None):
        self.obstacle_size = float(obstacle_size)
        self.target_pose = list(target_pose or [0.0, 0.0, -np.pi / 4])
        self.generate_obstacles: Optional[Callable[..., Any]] = None
        self.sandbox_globals = self._build_globals()

    def _build_globals(self) -> Dict[str, Any]:
        def is_safe(obs_x: float, obs_y: float, tblock_x: float, tblock_y: float, tblock_theta: float) -> bool:
            obstacle_pos = np.array([obs_x, obs_y])
            start_pos = np.array([tblock_x, tblock_y])
            target_pos = np.array(self.target_pose[:2])
            if analytic_obs_collision_check(
                Tblock_angle=tblock_theta,
                obs_center=obstacle_pos - start_pos,
                obs_size=self.obstacle_size * 2,
                threshold=0.04 * 2,
            ):
                return False
            if analytic_obs_collision_check(
                Tblock_angle=self.target_pose[-1],
                obs_center=obstacle_pos - target_pos,
                obs_size=self.obstacle_size * 2,
                threshold=0.04 * 2,
            ):
                return False
            return True

        def decode_obstacle(alpha: float, beta: float, start_xy, goal_xy=PUSHT_TARGET_XY):
            return decode_z_to_xy(alpha, beta, start_xy=start_xy, goal_xy=goal_xy)

        def encode_obstacle(obs_x: float, obs_y: float, start_xy, goal_xy=PUSHT_TARGET_XY):
            z = encode_xy_to_z(
                obs_x,
                obs_y,
                effective_radius=pusht_effective_radius(self.obstacle_size),
                start_xy=start_xy,
                goal_xy=goal_xy,
                corridor_width=PUSHT_CORRIDOR_WIDTH,
            )
            return z.to_dict()

        def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "numpy":
                return np
            if name.startswith("numpy."):
                return importlib.import_module(name)
            raise ImportError(f"Import blocked in sandbox: {name}")

        return {
            "np": np,
            "numpy": np,
            "is_safe": is_safe,
            "decode_obstacle": decode_obstacle,
            "encode_obstacle": encode_obstacle,
            "__builtins__": {
                "__import__": _safe_import,
                "range": range,
                "len": len,
                "abs": abs,
                "hash": hash,
                "min": min,
                "max": max,
                "float": float,
                "int": int,
                "bool": bool,
                "list": list,
                "dict": dict,
                "isinstance": isinstance,
                "enumerate": enumerate,
                "zip": zip,
                "sum": sum,
                "round": round,
            },
        }

    def load(self, code: str) -> bool:
        try:
            exec(code, self.sandbox_globals)
            fn = self.sandbox_globals.get("generate_obstacles")
            if not callable(fn):
                return False
            self.generate_obstacles = fn
            return True
        except Exception:
            return False

    def generate(self, tblock_pose: np.ndarray, num_obstacles: int, timeout_sec: int = 5) -> List[Dict[str, Any]]:
        if self.generate_obstacles is None:
            return []

        def _timeout_handler(signum, frame):
            raise TimeoutError("generate_obstacles timed out")

        try:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(max(1, int(timeout_sec)))
            try:
                obstacles = self.generate_obstacles(tblock_pose, num_obstacles)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        except Exception:
            return []

        if not isinstance(obstacles, list):
            return []

        normalized = []
        for obs in obstacles:
            if hasattr(obs, "__len__") and not isinstance(obs, dict):
                try:
                    obs = list(obs)
                    if len(obs) >= 2:
                        obs = {"x": obs[0], "y": obs[1], "purpose": obs[2] if len(obs) > 2 else ""}
                except Exception:
                    continue
            if not isinstance(obs, dict) or "x" not in obs or "y" not in obs:
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

    def validate(self, obstacles: Sequence[Mapping[str, Any]], tblock_pose: np.ndarray) -> Tuple[bool, str]:
        if not obstacles:
            return False, "empty"
        tx, ty, theta = float(tblock_pose[0]), float(tblock_pose[1]), float(tblock_pose[2])
        is_safe = self.sandbox_globals["is_safe"]
        for idx, obs in enumerate(obstacles):
            x = float(obs["x"])
            y = float(obs["y"])
            if not (-0.2 <= x <= 0.2 and -0.2 <= y <= 0.2):
                return False, f"out_of_bounds_{idx}"
            if not is_safe(x, y, tx, ty, theta):
                return False, f"unsafe_{idx}"
        for i in range(len(obstacles)):
            for j in range(i + 1, len(obstacles)):
                dist = math.sqrt(
                    (float(obstacles[i]["x"]) - float(obstacles[j]["x"])) ** 2
                    + (float(obstacles[i]["y"]) - float(obstacles[j]["y"])) ** 2
                )
                if dist < 0.03:
                    return False, "obstacles_too_close"
        return True, "ok"


def analytic_obs_collision_check(Tblock_angle, obs_center, obs_size, threshold=0.01):
    horizontal_length = 0.10
    horizontal_thickness = 0.03
    horizontal_center = (0, 0)
    vertical_length = 0.07
    vertical_thickness = 0.03
    vertical_center = (0, -0.05)

    def rotate_point_around_origin(point, center, angle):
        px, py = point
        ox, oy = center
        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy

    def edge_vectors(points):
        return [points[(i + 1) % len(points)] - points[i] for i in range(len(points))]

    def normalize(v):
        norm = np.linalg.norm(v)
        return v if norm == 0 else v / norm

    def project_polygon(axis, polygon):
        dots = [np.dot(vertex, axis) for vertex in polygon]
        return min(dots), max(dots)

    def overlap(min_a, max_a, min_b, max_b):
        return max_a >= min_b and max_b >= min_a

    def separating_axis_theorem(polygon1, polygon2):
        axes = [normalize(np.array([-edge[1], edge[0]])) for edge in edge_vectors(polygon1) + edge_vectors(polygon2)]
        for axis in axes:
            min_a, max_a = project_polygon(axis, polygon1)
            min_b, max_b = project_polygon(axis, polygon2)
            if not overlap(min_a, max_a, min_b, max_b):
                return False
        return True

    half_side = (obs_size + threshold) / 2
    obstacle_square = np.array(
        [
            (obs_center[0] - half_side, obs_center[1] - half_side),
            (obs_center[0] + half_side, obs_center[1] - half_side),
            (obs_center[0] + half_side, obs_center[1] + half_side),
            (obs_center[0] - half_side, obs_center[1] + half_side),
        ]
    )
    for rect_center, width, height in [
        (horizontal_center, horizontal_length, horizontal_thickness),
        (vertical_center, vertical_thickness, vertical_length),
    ]:
        vertices = [
            (rect_center[0] - width / 2, rect_center[1] - height / 2),
            (rect_center[0] + width / 2, rect_center[1] - height / 2),
            (rect_center[0] + width / 2, rect_center[1] + height / 2),
            (rect_center[0] - width / 2, rect_center[1] + height / 2),
        ]
        rotated = np.array(
            [rotate_point_around_origin(vertex, (0, 0), Tblock_angle) for vertex in vertices]
        )
        if separating_axis_theorem(rotated, obstacle_square):
            return True
    return False


def _as_mapping(value: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError("attribution_result must be a mapping or expose to_dict()")
