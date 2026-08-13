"""
PromptBuilder loads text prompt templates and fills runtime evidence blocks.

The evolve prompt intentionally keeps the program side lightweight:
- coarse mode exposes aggregate policy performance only
- attribution mode adds obstacle z-space attribution, coverage, and history
- cfa mode additionally adds counterfactual evidence
- graph_pattern mode adds failure-conditioned scene-graph pattern evidence

The LLM sees evidence and current generator code, then decides how to revise
the generator. We do not render hand-written revision templates or spatial
failure heuristics here.
"""

import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


def _load_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


_AGGREGATE_KEYS = ("success", "collision", "timeout", "fall")
_SKILL_FEEDBACK_MODES = ("skill_signal", "boundary_skill", "ours", "skill_library")
# Matches the verifier's infeasible cap; used only to steer the prompt direction
# (the verifier remains the hard gate).
_SKILL_INFEASIBLE_CAP = 0.30
_VALID_FEEDBACK_MODES = {
    "static",
    "coarse",
    "attribution",
    "cfa",
    "graph_pattern",
    "skill_signal",
    "boundary_skill",
    "ours",
    "skill_library",
}


class PromptBuilder:
    """
    Load prompt templates and fill placeholders.

    Example:
        builder = PromptBuilder(task_name="PushT", prompt_dir="DIVO/gpt/prompt")
        user = builder.build_evolve_user(
            batch_stats=...,
            fv_result=...,
            current_generator_code=...,
            feedback_mode="attribution",
            attribution_result=...,
        )
    """

    def __init__(self, task_name: str, prompt_dir: str):
        self.task_name = task_name
        self.task_prompt_dir = os.path.join(prompt_dir, task_name)
        if not os.path.isdir(self.task_prompt_dir):
            raise FileNotFoundError(
                f"Prompt directory not found: {self.task_prompt_dir}"
            )

    # ================================================================
    # Template loading
    # ================================================================

    def load_initial_system(self) -> str:
        return _load_template(os.path.join(self.task_prompt_dir, "initial_system.txt"))

    def load_initial_user(self) -> str:
        return _load_template(os.path.join(self.task_prompt_dir, "initial_user.txt"))

    def load_evolve_system(self) -> str:
        return _load_template(os.path.join(self.task_prompt_dir, "evolve_system.txt"))

    def load_evolve_user(self) -> str:
        """Load the raw evolve-user template without rendering evidence."""
        return self._load_evolve_user_template()

    def _load_evolve_user_template(self) -> str:
        return _load_template(os.path.join(self.task_prompt_dir, "evolve_user.txt"))

    # ================================================================
    # Initial generator prompt
    # ================================================================

    def build_initial_user(self, tblock_pose: np.ndarray) -> str:
        template = self.load_initial_user()
        tx, ty, ttheta = tblock_pose
        return template.format(
            tx=tx,
            ty=ty,
            theta_deg=np.degrees(ttheta),
        )

    # ================================================================
    # Common formatting helpers
    # ================================================================

    def _task_kind(self) -> str:
        name = self.task_name.lower()
        if "ant" in name:
            return "antmaze"
        if "nav" in name:
            return "navigate"
        return "pusht"

    def _generator_signature(self) -> str:
        kind = self._task_kind()
        if kind == "antmaze":
            return "def generate_maze_map(seed: int = None) -> list:"
        if kind == "navigate":
            return "def generate_pillars(agent_start: np.ndarray, goal: np.ndarray, num_pillars: int) -> list:"
        return "def generate_obstacles(tblock_pose: np.ndarray, num_obstacles: int) -> list:"

    def _generator_name(self) -> str:
        kind = self._task_kind()
        if kind == "antmaze":
            return "generate_maze_map"
        if kind == "navigate":
            return "generate_pillars"
        return "generate_obstacles"

    def build_initial_user_nav(self, agent_start: np.ndarray, goal: np.ndarray) -> str:
        """导航版 initial_user:填充 {sx}{sy}{gx}{gy}。"""
        template = self.load_initial_user()
        sx, sy = float(agent_start[0]), float(agent_start[1])
        gx, gy = float(goal[0]), float(goal[1])
        return template.format(sx=sx, sy=sy, gx=gx, gy=gy)

    def _as_dict(self, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if is_dataclass(value):
            return asdict(value)
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return dict(value.to_dict())
        if isinstance(value, Mapping):
            return dict(value)
        return {}

    def _as_list(self, value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return []

    def _safe_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _fmt_float(self, value: Any, digits: int = 3) -> str:
        return f"{self._safe_float(value):.{digits}f}"

    def _normalize_feedback_mode(self, feedback_mode: str) -> str:
        mode = str(feedback_mode or "coarse").strip().lower()
        if mode not in _VALID_FEEDBACK_MODES:
            mode = "coarse"
        # Static branches should not call evolve. If they do, use the coarse
        # renderer so the call remains debuggable instead of crashing.
        if mode == "static":
            mode = "coarse"
        return mode

    def _aggregate_counts(
        self,
        batch_stats: Optional[Mapping[str, Any]],
        fv_result: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, int]:
        counts = {key: self._safe_int((batch_stats or {}).get(key, 0)) for key in _AGGREGATE_KEYS}
        if sum(counts.values()) > 0:
            return counts

        detailed = self._as_dict(fv_result).get("counts", {})
        if not isinstance(detailed, Mapping):
            return counts

        for key, value in detailed.items():
            cnt = self._safe_int(value)
            if key == "success":
                counts["success"] += cnt
            elif str(key).startswith("collision"):
                counts["collision"] += cnt
            elif str(key).startswith("timeout"):
                counts["timeout"] += cnt
            elif str(key).startswith("fall"):
                counts["fall"] += cnt
        return counts

    def _dominant_failure_group(self, counts: Mapping[str, int]) -> str:
        failure_counts = {key: int(counts.get(key, 0)) for key in ("collision", "timeout", "fall")}
        if not failure_counts or max(failure_counts.values()) <= 0:
            return "none"
        return max(failure_counts, key=failure_counts.get)

    def _cell_id(self, row: Mapping[str, Any]) -> str:
        if row.get("cell_id"):
            return str(row["cell_id"])
        return (
            f"alpha={row.get('alpha_bin', 'unknown')}|"
            f"beta_abs={row.get('beta_abs_bin', 'unknown')}|"
            f"blockage={row.get('blockage_bin', 'unknown')}"
        )

    def _format_cell_line(self, idx: int, row: Mapping[str, Any]) -> str:
        support = self._safe_int(row.get("total_count", 0))
        failures = self._safe_int(row.get("failure_count", 0))
        return (
            f"{idx}. {self._cell_id(row)} | "
            f"support={support}, failures={failures}, "
            f"failure_rate={self._fmt_float(row.get('failure_rate', 0.0))}, "
            f"lift={self._fmt_float(row.get('failure_lift', 0.0))}"
        )

    def _squash_blank_lines(self, text: str) -> str:
        lines = text.splitlines()
        squashed = []
        blank_count = 0
        for line in lines:
            if line.strip():
                blank_count = 0
                squashed.append(line.rstrip())
            else:
                blank_count += 1
                if blank_count <= 1:
                    squashed.append("")
        return "\n".join(squashed).strip() + "\n"

    # ================================================================
    # Evolve prompt blocks
    # ================================================================

    def _format_task_objective_block(self) -> str:
        kind = self._task_kind()
        if kind == "antmaze":
            lines = [
                "Task objective:",
                "- Robot task: train an Ant policy to navigate from reset/start to goal in generated maze layouts.",
                "- Generator role: define training maze distributions only; do not alter policy, reward, simulator, start, or goal semantics.",
                "- Revision target: executable maze-layout code that remains solvable while exposing useful navigation uncertainty.",
            ]
        elif kind == "navigate":
            lines = [
                "Task objective:",
                "- Robot task: train a Navigation policy to move the Point agent from sampled starts to fixed goal (+0.65, 0.0).",
                "- Generator role: sample pillar layouts for training only; do not alter policy, reward, simulator, goal, pillar count, or pillar size.",
                "- Revision target: executable pillar-generator code that remains solvable while exposing useful pillar uncertainty.",
            ]
        else:
            lines = [
                "Task objective:",
                "- Robot task: train a Push-T policy to push the T-block from sampled start poses to target pose (0, 0, -45deg).",
                "- Generator role: sample obstacle layouts for training only; do not alter policy, reward, simulator, target pose, obstacle count, or obstacle size.",
                "- Revision target: executable obstacle-generator code that remains solvable while exposing useful obstacle uncertainty.",
            ]
        return "\n".join(lines)

    def _format_performance_block(
        self,
        batch_stats: Optional[Mapping[str, Any]],
        fv_result: Optional[Mapping[str, Any]],
        feedback_mode: str,
        reason: str = "",
    ) -> str:
        counts = self._aggregate_counts(batch_stats, fv_result)
        total = max(sum(counts.values()), 1)
        success_rate = counts["success"] / total
        failure_rate = 1.0 - success_rate
        dominant = self._dominant_failure_group(counts)
        labels = {key: key for key in _AGGREGATE_KEYS}
        if self._task_kind() == "navigate":
            labels["fall"] = "out_of_bounds"
            if dominant == "fall":
                dominant = "out_of_bounds"
        difficulty_guidance = self._format_difficulty_guidance(reason, success_rate)

        if feedback_mode == "coarse":
            lines = [
                f"- total_episodes: {total}",
                f"- success_rate: {success_rate:.3f}",
                f"- failure_rate: {failure_rate:.3f}",
                "- termination_distribution: "
                + ", ".join(
                    f"{labels[key]}={counts[key]} ({counts[key] / total:.1%})"
                    for key in _AGGREGATE_KEYS
                ),
                f"- dominant_failure_group: {dominant}",
                difficulty_guidance,
            ]
            return "\n".join(lines)

        lines = [
            (
                f"- success_rate={success_rate:.3f}, failure_rate={failure_rate:.3f}, "
                f"n_episodes={total}, dominant_failure_group={dominant}, "
                f"counts={{success:{counts['success']}, collision:{counts['collision']}, "
                f"timeout:{counts['timeout']}, {labels['fall']}:{counts['fall']}}}"
            ),
        ]
        # In skill-library modes the design direction comes from the skill-library
        # band profile (see the skill-signal block), not from the deployed
        # success rate. Keep the deployed stats above as factual context, but do
        # not emit the success-rate-based harder/easier guidance, which would
        # contradict the band direction.
        if feedback_mode not in _SKILL_FEEDBACK_MODES:
            lines.append(difficulty_guidance)
        return "\n".join(lines)

    def _format_difficulty_guidance(self, reason: str, success_rate: float) -> str:
        reason_text = str(reason or "").strip()
        lower = reason_text.lower()
        if "too_easy" in lower:
            signal = "too_easy"
            action = (
                "create a harder generator while keeping layouts feasible; "
                "increase obstacle complexity or informativeness without changing obstacle count or size"
            )
        elif "too_hard" in lower:
            signal = "too_hard"
            action = (
                "create an easier generator by reducing excessive blocking or simplifying layouts "
                "while preserving useful obstacle structure"
            )
        else:
            signal = "balanced"
            action = "create a variation with similar difficulty while improving diversity and learnability"

        detail = f"; trigger={reason_text}" if reason_text else ""
        return (
            f"- difficulty_signal: {signal} "
            f"(success_rate={success_rate:.3f}{detail}); guidance: {action}."
        )

    def _format_attribution_block(
        self,
        attribution_result: Optional[Any],
        top_k: int = 8,
        low_k: int = 4,
    ) -> str:
        result = self._as_dict(attribution_result)
        if not result:
            return ""

        global_counts = self._as_dict(result.get("global_counts"))
        config = self._as_dict(result.get("config") or result.get("metadata", {}).get("config"))
        top_k = self._safe_int(config.get("top_k", top_k), top_k)
        low_k = self._safe_int(config.get("low_k", low_k), low_k)
        num_obstacle_samples = self._safe_int(global_counts.get("num_obstacle_samples", 0))
        num_failure_samples = self._safe_int(global_counts.get("num_failure_samples", 0))
        failure_rate = self._fmt_float(global_counts.get("failure_rate", 0.0))
        min_support = self._safe_int(global_counts.get("min_support", 0))
        has_any_failure = bool(global_counts.get("has_any_failure", num_failure_samples > 0))
        cells = [self._as_dict(row) for row in self._as_list(result.get("cells"))]
        top_cells = [self._as_dict(row) for row in self._as_list(result.get("top_cells"))]
        if not top_cells:
            supported = [
                row for row in cells
                if self._safe_int(row.get("total_count", 0)) >= self._safe_int(global_counts.get("min_support", 1), 1)
            ]
            top_cells = sorted(
                supported,
                key=lambda row: (
                    self._safe_float(row.get("failure_lift", 0.0)),
                    self._safe_int(row.get("failure_count", 0)),
                    self._safe_int(row.get("total_count", 0)),
                ),
                reverse=True,
            )

        supported_low_candidates = [
            row for row in cells
            if self._safe_int(row.get("total_count", 0)) >= self._safe_int(global_counts.get("min_support", 1), 1)
        ]
        low_preferred = [
            row for row in supported_low_candidates
            if self._safe_float(row.get("failure_lift", 0.0)) < 1.0
        ]
        low_cells = sorted(
            low_preferred or supported_low_candidates,
            key=lambda row: (
                self._safe_float(row.get("failure_lift", 0.0)),
                self._safe_float(row.get("failure_rate", 0.0)),
            ),
        )

        lines = [
            "Obstacle-level attribution evidence:",
            "Lift > 1 means this obstacle cell appears more often in failed rollouts than expected from its sampling frequency.",
            (
                f"- obstacle_samples: {num_obstacle_samples}, "
                f"failure_samples: {num_failure_samples}, "
                f"overall_failure_rate: {failure_rate}, "
                f"min_support: {min_support}"
            ),
        ]

        if num_obstacle_samples == 0:
            lines.append("- no obstacle samples were collected for this evolve batch.")
            return "\n".join(lines)

        if not has_any_failure:
            lines.append(
                "- no failed obstacle samples were collected; obstacle-specific failure attribution is not available for this batch."
            )
            return "\n".join(lines)

        lines.append("- top failure-associated cells:")

        if top_cells:
            for idx, row in enumerate(top_cells[:top_k], start=1):
                lines.append(f"  {self._format_cell_line(idx, row)}")
        else:
            lines.append("  none with sufficient support")

        if low_cells:
            lines.append("- low-attribution / learnable cells:")
            for idx, row in enumerate(low_cells[:low_k], start=1):
                lines.append(f"  {self._format_cell_line(idx, row)}")

        return "\n".join(lines)

    def _format_cfa_block(self, cfa_result: Optional[Any]) -> str:
        result = self._as_dict(cfa_result)
        if not result:
            return ""

        lines = [
            "Counterfactual evidence:",
            "Improvement rate measures whether intervention on an obstacle factor makes failed rollouts recover under the same policy.",
        ]

        factor_rates = self._as_dict(
            result.get("factor_recovery_rates")
            or result.get("recovery_rates")
            or result.get("factors")
        )
        if factor_rates:
            lines.append("- factor_recovery_rates:")
            for factor, value in factor_rates.items():
                lines.append(f"  - {factor}: {self._fmt_float(value)}")

        cause_counts = self._as_dict(result.get("cause_counts") or result.get("cause_distribution"))
        if cause_counts:
            lines.append("- cause_distribution:")
            total = max(sum(self._safe_int(v) for v in cause_counts.values()), 1)
            for cause, value in cause_counts.items():
                cnt = self._safe_int(value)
                lines.append(f"  - {cause}: {cnt} ({cnt / total:.1%})")

        notes = result.get("notes") or result.get("summary")
        if notes:
            lines.append(f"- summary: {notes}")

        return "\n".join(lines)

    def _format_current_generator_block(self, current_generator_code: Optional[str]) -> str:
        lines = ["Current generator code:"]
        if current_generator_code:
            lines.append("```python")
            lines.append(current_generator_code.strip())
            lines.append("```")
        else:
            lines.append("(no current generator available)")
        return "\n".join(lines)

    def _format_coverage_block(
        self,
        coverage_summary: Optional[Any],
        attribution_result: Optional[Any],
        top_k: int = 6,
    ) -> str:
        summary = self._as_dict(coverage_summary)
        result = self._as_dict(attribution_result)
        embedded_coverage = self._as_dict(result.get("coverage"))
        if not summary and embedded_coverage:
            summary = embedded_coverage

        if not summary and not result:
            return ""

        cells = [self._as_dict(row) for row in self._as_list(summary.get("cells"))]
        if not cells:
            cells = [self._as_dict(row) for row in self._as_list(result.get("cells"))]

        global_counts = self._as_dict(summary.get("global_counts"))
        if not global_counts:
            global_counts = self._as_dict(result.get("global_counts"))

        min_support = self._safe_int(global_counts.get("min_support", summary.get("min_support", 0)))
        occupied = self._safe_int(summary.get("occupied_cells", 0))
        if occupied <= 0:
            occupied = len([row for row in cells if self._safe_int(row.get("total_count", 0)) > 0])
        supported = self._safe_int(summary.get("supported_cells", 0))
        if supported <= 0 and min_support > 0:
            supported = len([row for row in cells if self._safe_int(row.get("total_count", 0)) >= min_support])

        top_sampled = self._as_list(summary.get("most_sampled_cells"))
        if not top_sampled:
            top_sampled = sorted(
                cells,
                key=lambda row: self._safe_int(row.get("total_count", 0)),
                reverse=True,
            )[:top_k]
        top_sampled = [self._as_dict(row) for row in top_sampled]

        lines = [
            "Empirical coverage of the current generator:",
            f"- obstacle_samples: {self._safe_int(global_counts.get('num_obstacle_samples', summary.get('num_obstacle_samples', 0)))}",
            f"- occupied_cells: {occupied}",
            f"- supported_cells: {supported}" + (f" (min_support={min_support})" if min_support else ""),
        ]

        alpha_stats = self._as_dict(summary.get("alpha_stats"))
        beta_abs_stats = self._as_dict(summary.get("beta_abs_stats"))
        blockage_stats = self._as_dict(summary.get("blockage_stats"))
        if alpha_stats:
            lines.append(
                "- alpha: "
                f"mean={self._fmt_float(alpha_stats.get('mean', 0.0))}, "
                f"std={self._fmt_float(alpha_stats.get('std', 0.0))}, "
                f"range=[{self._fmt_float(alpha_stats.get('min', 0.0))}, "
                f"{self._fmt_float(alpha_stats.get('max', 0.0))}]"
            )
        if beta_abs_stats:
            lines.append(
                "- abs_beta: "
                f"mean={self._fmt_float(beta_abs_stats.get('mean', 0.0))}, "
                f"std={self._fmt_float(beta_abs_stats.get('std', 0.0))}"
            )
        if blockage_stats:
            lines.append(
                "- blockage: "
                f"mean={self._fmt_float(blockage_stats.get('mean', 0.0))}, "
                f"std={self._fmt_float(blockage_stats.get('std', 0.0))}"
            )

        if top_sampled:
            lines.append("- most_sampled_cells:")
            total_samples = max(
                self._safe_int(global_counts.get("num_obstacle_samples", 0)),
                sum(self._safe_int(row.get("total_count", 0)) for row in top_sampled),
                1,
            )
            for idx, row in enumerate(top_sampled[:top_k], start=1):
                count = self._safe_int(row.get("total_count", 0))
                lines.append(
                    f"  {idx}. {self._cell_id(row)} | count={count}, "
                    f"share={count / total_samples:.1%}"
                )

        return "\n".join(lines)

    def _format_attribution_history_block(
        self,
        attribution_history: Optional[Sequence[Any]],
        current_success_rate: Optional[float] = None,
        max_records: int = 3,
    ) -> str:
        records = [self._as_dict(record) for record in self._as_list(attribution_history)]
        records = [record for record in records if record]
        if not records:
            return ""

        recent = records[-max_records:]
        lines = ["Previous success-rate changes:"]
        rendered = False

        def _label(record: Mapping[str, Any]) -> str:
            evolve_index = record.get("evolve_index") or record.get("round_idx")
            if evolve_index is not None:
                prefix = f"Round {self._safe_int(evolve_index)}"
            else:
                prefix = "previous round"
            episode_idx = record.get("episode_idx") or record.get("total_episode_count")
            if episode_idx is not None:
                prefix += f" (episode={self._safe_int(episode_idx)})"
            return prefix

        def _success_rate(record: Mapping[str, Any]) -> Optional[float]:
            value = record.get("success_rate_at_evolve")
            if value is None:
                value = record.get("success_rate")
            if value is None:
                return None
            return self._safe_float(value)

        for previous, current in zip(recent, recent[1:]):
            previous_rate = _success_rate(previous)
            current_rate = _success_rate(current)
            if previous_rate is None or current_rate is None:
                continue
            delta = current_rate - previous_rate
            lines.append(
                f"- {_label(previous)} -> {_label(current)}: "
                f"success_rate {previous_rate:.3f} -> {current_rate:.3f} ({delta:+.3f})"
            )
            rendered = True

        last_rate = _success_rate(recent[-1])
        if current_success_rate is not None and last_rate is not None:
            current_rate = self._safe_float(current_success_rate)
            delta = current_rate - last_rate
            lines.append(
                f"- after {_label(recent[-1])}: "
                f"success_rate {last_rate:.3f} -> current {current_rate:.3f} ({delta:+.3f})"
            )
            rendered = True

        if not rendered:
            for record in recent:
                rate = _success_rate(record)
                if rate is None:
                    continue
                lines.append(
                    f"- {_label(record)}: success_rate={rate:.3f}"
                )
                rendered = True

        if not rendered:
            return ""

        return "\n".join(lines)

    def _format_revision_request_block(self, feedback_mode: str) -> str:
        is_nav = self._task_kind() == "navigate"
        item = "pillar" if is_nav else "obstacle"
        start = "Point-agent start" if is_nav else "start pose"
        lines = [
            f"Return exactly one complete fenced Python code block defining `{self._generator_name()}`.",
        ]
        if feedback_mode == "coarse":
            lines.append("The available feedback is aggregate performance only; infer revision needs from those statistics and the current code.")
        elif feedback_mode == "attribution":
            lines.append("Use the obstacle attribution and coverage evidence as the main feedback signal for deciding what distributional structure to revise. High-lift cells suggest obstacle regions that may expose policy weaknesses; low-lift or heavily sampled harmless cells may indicate regions that are already learnable or less informative.")
            lines.append("The returned generator may be audited by offline sampling. If the current stage is too easy, the actual sampling distribution should increase the expected attribution evidence score of sampled obstacle cells. If the current stage is too hard, it should reduce excessive sampling of the strongest failure-associated cells while preserving useful structure. Implement this in the sampling probabilities, not only in comments or variable names.")
        elif feedback_mode == "cfa":
            lines.append("Use attribution, coverage, history, and counterfactual evidence as the main feedback signals for deciding what distributional structure to revise.")
        elif feedback_mode == "graph_pattern":
            lines.append("Use the failure-conditioned scene-graph pattern evidence as the main feedback signal. The listed patterns were discovered from raw graph-derived layout features using data-adaptive quantile rules; treat them as statistical evidence about which layout structures expose policy weaknesses, not as fixed hand-written obstacle templates.")
            lines.append("If the current stage is too easy, increase the feasible sampling probability of layouts matching high-support, high-lift graph patterns while preserving diversity. If the current stage is too hard, reduce over-concentration on the strongest high-lift patterns or soften them by relaxing their graph-derived conditions. If the stage is balanced, keep similar difficulty but diversify around the discovered patterns instead of collapsing to one mode. Implement these changes in the actual sampling distribution, not only in comments or variable names.")
        elif feedback_mode == "skill_library":
            lines.append("Use the skill-library boundary probe evidence (W_probe) as the main design signal. p = fraction of the trained library skills w_1..w_K that solve a scene. Follow the stated difficulty direction and its SOURCE/TARGET/GUARD evidence: move source layouts toward the boundary target without crossing into the guarded opposite extreme.")
            lines.append(f"Rewrite the generator from scratch as a stochastic generator that stays diverse across calls and does not collapse to a single narrow layout; using several distinct layout families is one good way to do this. Do not hard-code the exact listed scenes; generalize them into stochastic {item} families conditioned on the {start}.")
            lines.append(f"Aim for learnable-boundary {item} families: sometimes solvable by the library, not trivially mastered by all skills, and not unsolvable by every skill. Keep the sampled families diverse and spread across the layout space. Candidate generators are checked by a paired W_probe verifier that scores boundary_count (how many layouts fall in the tau-band) under an absolute physical-validity gate.")
        elif feedback_mode in ("skill_signal", "boundary_skill", "ours"):
            lines.append("Use the skill-library probe evidence as the main design signal. FOCUS scenes are on the learnable boundary (some injected skills solve them, some fail) and should be generated more; AVOID scenes are infeasible (no injected skill solved them) and should be eased or avoided. Follow the stated difficulty state and design objective.")
            lines.append(f"Rewrite the generator from scratch as a stochastic generator that stays diverse across calls and does not collapse to a single narrow layout; using several distinct layout families is one good way to do this. Do not hard-code the exact listed scenes; generalize them into stochastic {item} families conditioned on the {start}.")
            lines.append(f"Aim for learnable-boundary {item} families: sometimes solvable by the skill library, not trivially mastered by all skills, and not unsolvable by every skill. Keep the sampled families diverse and spread across the layout space. Candidate generators are checked by a paired rollout verifier on feasibility, learnable-band mass, and layout coverage.")
        return "\n".join(lines)

    def _format_graph_pattern_block(
        self,
        graph_pattern_result: Optional[Any],
        top_k: int = 8,
    ) -> str:
        result = self._as_dict(graph_pattern_result)
        if not result:
            return ""

        metadata = self._as_dict(result.get("metadata"))
        stats = self._as_dict(result.get("global_stats"))
        patterns = [self._as_dict(row) for row in self._as_list(result.get("top_patterns"))]
        if not patterns:
            return (
                "Failure-conditioned scene-graph pattern evidence:\n"
                "- No supported high-risk graph patterns were discovered in this batch. "
                "Use aggregate policy statistics and current code structure instead."
            )

        raw_count = self._safe_int(metadata.get("raw_record_count", stats.get("num_episodes", 0)))
        graph_count = self._safe_int(metadata.get("scene_graph_record_count", stats.get("num_episodes", 0)))
        skipped = self._safe_int(metadata.get("skipped_without_scene_graph", 0))
        failure_rate = self._safe_float(stats.get("failure_rate", 0.0))

        lines = [
            "Failure-conditioned scene-graph pattern evidence:",
            (
                "These patterns were discovered automatically from raw scene-graph layout "
                "features using data-adaptive quantile predicates. They are statistical "
                "evidence, not hand-written obstacle templates."
            ),
            (
                f"- episodes_used: {graph_count}/{raw_count}, "
                f"skipped_without_scene_graph: {skipped}, "
                f"global_failure_rate: {failure_rate:.3f}, "
                f"atomic_predicates: {self._safe_int(result.get('atomic_predicate_count', 0))}, "
                f"candidate_patterns: {self._safe_int(result.get('candidate_count', 0))}"
            ),
            "Feature meanings:",
            "- axis.length: distance from start pose to target pose along the task axis.",
            "- start.x/start.y/start.theta: initial T-block pose.",
            "- summary.abs_beta_*: obstacle lateral distance statistics relative to the task axis.",
            "- summary.blockage_*: obstacle blockage statistics relative to the task axis.",
            "- summary.pair_distance_*: obstacle-pair distance statistics.",
            "- summary.pair_delta_alpha_* and summary.pair_delta_beta_*: pair spacing along and across the task axis.",
            "How to use this evidence:",
            "- If the difficulty signal is too_easy, increase feasible coverage of high-support, high-lift graph patterns while preserving stochastic diversity.",
            "- If the difficulty signal is too_hard, relax over-concentrated high-lift pattern conditions instead of deleting all obstacle structure.",
            "- If the difficulty signal is balanced, preserve similar difficulty while diversifying around the discovered patterns; do not collapse to one mode.",
            "Top discovered graph patterns:",
        ]

        for idx, pattern in enumerate(patterns[:top_k], start=1):
            conditions = [str(item) for item in self._as_list(pattern.get("conditions"))]
            condition_text = "; ".join(conditions) if conditions else "<none>"
            lines.append(
                f"{idx}. support={self._safe_int(pattern.get('support', 0))}, "
                f"failure_count={self._safe_int(pattern.get('failure_count', 0))}, "
                f"failure_rate={self._safe_float(pattern.get('failure_rate', 0.0)):.3f}, "
                f"lift={self._safe_float(pattern.get('failure_lift', 0.0)):.3f}, "
                f"lcb_lift={self._safe_float(pattern.get('failure_lift_lcb', 0.0)):.3f}, "
                f"dominant_failure={pattern.get('dominant_failure_key', 'none')} "
                f"({self._safe_int(pattern.get('dominant_failure_count', 0))}); "
                f"conditions: {condition_text}"
            )
        return "\n".join(lines)

    def _format_skill_signal_block(
        self,
        skill_signal_result: Optional[Any],
        include_behavior: bool = True,
    ) -> str:
        result = self._as_dict(skill_signal_result)
        if not result:
            return ""

        # Route B V0 (skill_library): render the W_probe tau-band boundary signal
        # and the single evolution direction instead of the legacy random-z block.
        boundary_signal = self._as_dict(result.get("boundary_signal"))
        if boundary_signal:
            return self._format_boundary_signal_block(
                result, boundary_signal, include_behavior=include_behavior
            )

        profile = self._as_dict(result.get("difficulty_profile"))
        context = self._as_dict(result.get("design_context"))
        hist = self._as_dict(profile.get("realized_hist"))
        bins = hist.get("bins", [])
        counts = hist.get("counts", [])
        coverage = self._as_dict(profile.get("boundary_coverage"))

        state, objective = self._skill_band_direction(profile)

        lines = [
            "Skill-library probe evidence:",
            (
                "The current latent policy was probed by fixing each scene and injecting "
                "the same bank of K latent skills z. realized = fraction of injected "
                "skills that solve a scene."
            ),
            (
                "A scene is on the learnable boundary when some injected skills solve it "
                "and some fail (0 < realized < 1); infeasible when none solve it "
                "(realized = 0); trivial when all solve it (realized = 1). The design "
                "target is to keep most layouts on the learnable boundary, diverse, with "
                "few trivial or infeasible scenes."
            ),
            "",
            f"Difficulty state: {state}",
            f"Design objective: {objective}",
            "",
            "Difficulty snapshot:",
            f"- frac_boundary: {self._fmt_float(profile.get('frac_boundary', 0.0), 3)}",
            f"- frac_trivial: {self._fmt_float(profile.get('frac_trivial', 0.0), 3)}",
            f"- frac_infeasible: {self._fmt_float(profile.get('frac_infeasible', 0.0), 3)}",
            f"- mean_realized: {self._fmt_float(profile.get('mean_realized', 0.0), 3)}",
            f"- boundary_coverage_per_scene: {self._fmt_float(coverage.get('coverage_per_scene', 0.0), 3)}",
        ]
        if bins and counts:
            lines.append(f"- realized_hist bins={bins}, counts={counts}")

        guidance = {
            "focus": (
                "BOUNDARY scenes (some injected skills solve them, some fail). Generate "
                "more layout families like these; generalize them into stochastic "
                "families conditioned on the start pose rather than copying exact coords."
            ),
            "avoid": (
                "INFEASIBLE scenes (no injected skill solved them). Ease or avoid these so "
                "that at least some skill can solve them; do not concentrate mass here."
            ),
        }
        for key in ("focus", "avoid"):
            rows = self._as_list(context.get(key))
            lines.extend(["", f"{key.upper()} - {guidance[key]}"])
            if not rows:
                lines.append("- none")
                continue
            for idx, row_any in enumerate(rows, start=1):
                row = self._as_dict(row_any)
                lines.extend(
                    self._format_skill_scene(
                        row,
                        idx,
                        include_behavior=include_behavior,
                    )
                )
        return "\n".join(lines)

    def _format_boundary_signal_block(
        self,
        result: Mapping[str, Any],
        boundary_signal: Mapping[str, Any],
        include_behavior: bool = True,
    ) -> str:
        """Render the Route B V0 W_probe tau-band boundary evidence (deviation 2B).

        Uses the trained skill library W_probe (not random z), the tau-band
        boundary definition, and the SINGLE evolution direction
        (FIX_VALIDITY / RELAX / HARDEN / PRESERVE_AND_DIVERSIFY) that also drives
        the verifier, so the LLM sees exactly the signal it is being judged on.
        """
        context = self._as_dict(result.get("design_context"))
        K = self._safe_int(boundary_signal.get("K", 0))
        n = self._safe_int(boundary_signal.get("n_scenes", 0))
        tau = self._safe_float(boundary_signal.get("tau", 0.125))
        delta = self._safe_float(boundary_signal.get("target_delta", 0.15))
        direction, objective = self._boundary_direction_objective(boundary_signal)

        lines = [
            "Skill-library boundary probe evidence (W_probe):",
            (
                "The current latent policy was probed by fixing each scene and executing "
                f"the trained skill library W_probe = {{w_1..w_{max(K,1)}}} (the deployment "
                "skill w_0 is diagnostic only and is NOT counted). p = fraction of the K "
                "library skills that solve a scene."
            ),
            (
                f"A scene is on the learnable boundary when tau < p < 1-tau (tau={self._fmt_float(tau, 3)}), "
                "kept away from saturation: p<=tau means the library cannot solve it "
                "(near-infeasible), p>=1-tau means the library solves it with every skill "
                "(near-trivial). The design target keeps most layouts in the tau-band, "
                "diverse, with few saturated scenes."
            ),
            "",
            f"Difficulty direction: {direction}",
            f"Design objective: {objective}",
            "",
            f"Boundary snapshot (K={K}, n={n}):",
            f"- boundary_count: {self._safe_int(boundary_signal.get('boundary_count', 0))} "
            f"(boundary_rate={self._fmt_float(boundary_signal.get('boundary_rate', 0.0), 3)})",
            f"- r_hard (p<=tau, library cannot solve): {self._fmt_float(boundary_signal.get('r_hard', 0.0), 3)}",
            f"- r_easy (p>=1-tau, library solves all): {self._fmt_float(boundary_signal.get('r_easy', 0.0), 3)}",
            f"- target_rate (|p-0.5|<={self._fmt_float(delta, 2)}): {self._fmt_float(boundary_signal.get('target_rate', 0.0), 3)}",
            f"- mean_b (mean p(1-p)): {self._fmt_float(boundary_signal.get('mean_b', 0.0), 3)}",
            f"- valid_rate: {self._fmt_float(boundary_signal.get('valid_rate', 0.0), 3)}, "
            f"duplicate_rate: {self._fmt_float(boundary_signal.get('duplicate_rate', 0.0), 3)}",
            f"- w0_success_rate (deployment diagnostic, excluded from p): "
            f"{self._fmt_float(boundary_signal.get('w0_success_rate', 0.0), 3)}",
        ]

        boundary_text = (
            "BOUNDARY scenes (tau < p < 1-tau) are the target difficulty: some library "
            "skills solve them and some fail. Generalize their structure rather than "
            "copying exact coordinates."
        )
        if direction == "HARDEN":
            sections = (
                ("harden", "SOURCE_EASY", "EASY scenes (p >= 1-tau) should be made harder and moved toward the boundary."),
                ("focus", "TARGET_BOUNDARY", boundary_text),
                ("avoid", "GUARD_HARD", "HARD scenes (p <= tau) mark the over-hardening extreme; do not push the new distribution into this region."),
            )
        elif direction == "RELAX":
            sections = (
                ("avoid", "SOURCE_HARD", "HARD scenes (p <= tau) should be eased and moved toward the boundary, not simply discarded."),
                ("focus", "TARGET_BOUNDARY", boundary_text),
                ("harden", "GUARD_EASY", "EASY scenes (p >= 1-tau) mark the over-relaxation extreme; do not make the new distribution trivial."),
            )
        elif direction == "PRESERVE_AND_DIVERSIFY":
            sections = (("focus", "TARGET_BOUNDARY", boundary_text),)
        else:
            sections = ()
            lines.extend(["", "No skill-scene examples are shown while generator validity is being repaired."])

        for key, title, guidance in sections:
            rows = self._as_list(context.get(key))
            lines.extend(["", f"{title} - {guidance}"])
            if not rows:
                lines.append("- none")
                continue
            for idx, row_any in enumerate(rows, start=1):
                row = self._as_dict(row_any)
                lines.extend(
                    self._format_skill_scene(row, idx, include_behavior=include_behavior)
                )
        return "\n".join(lines)

    def _boundary_direction_objective(self, boundary_signal: Mapping[str, Any]):
        """Single evolution direction for the boundary block. Prefers the direction
        computed by the workspace (``evolution_direction``); otherwise derives it
        from r_hard / r_easy / valid_rate with the same symmetric thresholds."""
        is_nav = self._task_kind() == "navigate"
        item = "pillars" if is_nav else "obstacles"
        initial_entity = "Point agent" if is_nav else "T-block"
        objectives = {
            "FIX_VALIDITY": (
                "Too many generated layouts are physically invalid. Fix validity first: "
                f"keep {item} in-bounds, non-overlapping, and not touching the initial "
                f"{initial_entity}, before changing difficulty."
            ),
            "RELAX": (
                "The skill library cannot solve too many layouts (r_hard high). Ease the "
                "structure (reduce blocking, widen gaps) to move layouts from p~0 toward "
                f"the tau-band (p~0.25-0.5). Do not simply delete {item}."
            ),
            "HARDEN": (
                "The skill library already solves too many layouts (r_easy high). Increase "
                f"structural difficulty using placement and spatial relations only ({item} "
                "count and size are fixed) to move layouts from p~1 toward the tau-band "
                "(p~0.5-0.75)."
            ),
            "PRESERVE_AND_DIVERSIFY": (
                "The distribution already sits on the learnable boundary. Keep a similar "
                "difficulty level but vary the layout distribution and increase diversity "
                "across distinct families; do not push everything harder."
            ),
        }
        direction = str(boundary_signal.get("evolution_direction", "") or "").upper()
        if direction not in objectives:
            valid_rate = self._safe_float(boundary_signal.get("valid_rate", 0.0))
            r_hard = self._safe_float(boundary_signal.get("r_hard", 0.0))
            r_easy = self._safe_float(boundary_signal.get("r_easy", 0.0))
            if valid_rate < 0.8:
                direction = "FIX_VALIDITY"
            elif r_hard >= 0.8:
                direction = "RELAX"
            elif r_easy >= 0.8:
                direction = "HARDEN"
            else:
                direction = "PRESERVE_AND_DIVERSIFY"
        return direction, objectives[direction]

    def _skill_band_direction(self, profile: Mapping[str, Any]):
        """Single, band-anchored design direction from the skill-library profile.

        Derived from the K-natural band fractions (frac_infeasible / frac_trivial /
        frac_boundary), never from the deployed success rate, and always aimed at
        pulling mass into the learnable band -- never "harder for its own sake".
        Returns (state, objective_text).
        """

        frac_inf = self._safe_float(profile.get("frac_infeasible", 0.0))
        frac_triv = self._safe_float(profile.get("frac_trivial", 0.0))
        frac_bound = self._safe_float(profile.get("frac_boundary", 0.0))
        if frac_inf > _SKILL_INFEASIBLE_CAP:
            return (
                "too_hard",
                "Too many layouts are unsolvable by every skill. Ease the structure "
                "(reduce blocking, widen gaps) until those layouts become solvable by at "
                "least some skill, pulling them into the learnable band. Do not simply "
                "delete obstacles.",
            )
        if frac_triv > frac_bound:
            return (
                "too_easy",
                "Most layouts are already solved by the whole skill library. Increase "
                "structural difficulty using placement and spatial relations only "
                "(obstacle count and size are fixed) until more layouts become "
                "sometimes-solvable, pulling them into the learnable band.",
            )
        return (
            "on_boundary",
            "The distribution already sits on the learnable boundary. Keep a similar "
            "difficulty level but vary the layout distribution and increase diversity "
            "across distinct families; do not push everything harder.",
        )

    def _format_skill_scene(
        self,
        scene: Mapping[str, Any],
        idx: int,
        include_behavior: bool = True,
    ) -> List[str]:
        start = scene.get("start", [])
        obstacles = self._as_list(scene.get("obstacles"))
        item = "pillar" if self._task_kind() == "navigate" else "obstacle"
        lines = [
            (
                f"{idx}. scene_id={scene.get('scene_id', 'NA')}, "
                f"realized={self._fmt_float(scene.get('realized', 0.0), 3)}, "
                f"feasible={self._safe_int(scene.get('feasible', 0))}, "
                f"deployed={self._safe_int(scene.get('deployed', 0))}, "
                f"start={self._format_numeric_list(start, digits=3)}"
            )
        ]
        if obstacles:
            lines.append(f"   {item}s:")
            for obs_idx, obs_any in enumerate(obstacles, start=1):
                obs = self._as_dict(obs_any)
                lines.append(
                    "   - "
                    f"{item}{obs_idx}: x={self._fmt_float(obs.get('x', 0.0), 3)}, "
                    f"y={self._fmt_float(obs.get('y', 0.0), 3)}, "
                    f"purpose={obs.get('purpose', '')}"
                )
        else:
            lines.append(f"   {item}s: []")

        if include_behavior and scene.get("behavior_summary"):
            behavior = self._as_dict(scene.get("behavior_summary"))
            routes = self._as_list(behavior.get("routes"))
            if routes:
                lines.append("   behavior_summary:")
                for route_any in routes:
                    route = self._as_dict(route_any)
                    label = "success" if bool(route.get("success", False)) else "failure"
                    waypoint_pairs = []
                    for wp_any in self._as_list(route.get("waypoints")):
                        wp = self._as_dict(wp_any)
                        waypoint_pairs.append(
                            (
                                self._fmt_float(wp.get("alpha", 0.0), 2),
                                self._fmt_float(wp.get("beta", 0.0), 2),
                            )
                        )
                    lines.append(
                        f"   - {label} skill={self._safe_int(route.get('skill_index', -1))}: "
                        f"task_frame_waypoints(alpha,beta)={waypoint_pairs}"
                    )
        return lines

    def _format_numeric_list(self, values: Any, digits: int = 3, max_items: int = 6) -> str:
        items = self._as_list(values)
        if not items:
            return "[]"
        shown = items[:max_items]
        suffix = ", ..." if len(items) > max_items else ""
        return "[" + ", ".join(self._fmt_float(v, digits) for v in shown) + suffix + "]"

    def _format_feedback_evidence_block(
        self,
        feedback_mode: str,
        attribution_block: str = "",
        coverage_block: str = "",
        history_block: str = "",
        cfa_block: str = "",
        graph_pattern_block: str = "",
        skill_signal_block: str = "",
    ) -> str:
        mode = self._normalize_feedback_mode(feedback_mode)
        if mode == "coarse":
            return ""

        if mode in _SKILL_FEEDBACK_MODES:
            if not skill_signal_block:
                return ""
            if mode == "skill_library":
                return (
                    "We also computed trained skill-library boundary evidence for the "
                    "current generator. The same W_probe library evaluates each scene, "
                    "and its cross-skill success distribution determines the evolution "
                    "direction used below:\n\n"
                    + skill_signal_block
                )
            return (
                "We also computed skill-library probe evidence for the current generator. "
                "This evidence uses injected latent skills to summarize which generated scenes "
                "are near the learnable frontier, already mastered, or currently unsolved:\n\n"
                + skill_signal_block
            )

        if mode == "graph_pattern":
            if not graph_pattern_block:
                return ""
            return (
                "We also computed failure-conditioned scene-graph pattern evidence for the current generator. "
                "The evidence summarizes graph-derived layout patterns that were statistically associated "
                "with failed rollouts:\n\n"
                + graph_pattern_block
            )

        sections = [
            (
                "We also computed obstacle-level evidence for the current generator. "
                "The attribution evidence summarizes which obstacle regions were more or less "
                "associated with failed rollouts, and the coverage evidence summarizes where "
                "the current generator actually sampled obstacles:"
            )
        ]
        if attribution_block:
            sections.append("Obstacle-level attribution evidence:\n" + attribution_block)
        if coverage_block:
            sections.append("Generator coverage evidence:\n" + coverage_block)
        if history_block:
            sections.append("Recent evolution history:\n" + history_block)
        if mode == "cfa" and cfa_block:
            sections.append("Counterfactual evidence:\n" + cfa_block)
        return "\n\n".join(sections)

    # ================================================================
    # Evolve prompt assembly
    # ================================================================

    def build_evolve_user(
        self,
        batch_stats: Dict[str, int],
        fv_result: Optional[Dict] = None,
        reason: str = "",
        current_generator_code: Optional[str] = None,
        feedback_mode: str = "coarse",
        attribution_result: Optional[Any] = None,
        coverage_summary: Optional[Any] = None,
        attribution_history: Optional[List[Dict]] = None,
        cfa_result: Optional[Any] = None,
        graph_pattern_result: Optional[Any] = None,
        skill_signal_result: Optional[Any] = None,
        include_behavior: bool = True,
    ) -> str:
        """Load evolve_user.txt and fill the minimal mode-specific blocks.

        ``reason`` is rendered only as a fixed-schedule difficulty signal.
        """
        mode = self._normalize_feedback_mode(feedback_mode)
        template = self._load_evolve_user_template()

        attribution_block = ""
        coverage_block = ""
        history_block = ""
        cfa_block = ""
        graph_pattern_block = ""
        skill_signal_block = ""
        counts = self._aggregate_counts(batch_stats, fv_result)
        current_success_rate = counts["success"] / max(sum(counts.values()), 1)
        if mode in ("attribution", "cfa"):
            attribution_block = self._format_attribution_block(attribution_result)
            coverage_block = self._format_coverage_block(coverage_summary, attribution_result)
            history_block = self._format_attribution_history_block(
                attribution_history,
                current_success_rate=current_success_rate,
            )
        if mode == "cfa":
            cfa_block = self._format_cfa_block(cfa_result)
        if mode == "graph_pattern":
            graph_pattern_block = self._format_graph_pattern_block(graph_pattern_result)
        if mode in ("skill_signal", "boundary_skill", "ours", "skill_library"):
            skill_signal_block = self._format_skill_signal_block(
                skill_signal_result,
                include_behavior=include_behavior,
            )
        request_mode = "coarse"
        if attribution_block:
            request_mode = "attribution"
        if cfa_block:
            request_mode = "cfa"
        if graph_pattern_block:
            request_mode = "graph_pattern"
        if skill_signal_block:
            request_mode = mode

        filled = template.format(
            task_objective_block=self._format_task_objective_block(),
            performance_block=self._format_performance_block(batch_stats, fv_result, mode, reason),
            feedback_evidence=self._format_feedback_evidence_block(
                request_mode,
                attribution_block=attribution_block,
                coverage_block=coverage_block,
                history_block=history_block,
                cfa_block=cfa_block,
                graph_pattern_block=graph_pattern_block,
                skill_signal_block=skill_signal_block,
            ),
            attribution_block=attribution_block,
            cfa_block=cfa_block,
            current_generator_block=self._format_current_generator_block(current_generator_code),
            coverage_block=coverage_block,
            attribution_history_block=history_block,
            graph_pattern_evidence=graph_pattern_block,
            skill_signal_evidence=skill_signal_block,
            revision_request_block=self._format_revision_request_block(request_mode),
            current_generator_context=self._format_current_generator_block(current_generator_code),
            policy_statistics=self._format_performance_block(batch_stats, fv_result, mode, reason),
            attribution_evidence=attribution_block,
            counterfactual_evidence=cfa_block,
            coverage_evidence=coverage_block,
            history_evidence=history_block,
            evolution_request=self._format_revision_request_block(request_mode),
        )
        return self._squash_blank_lines(filled)
