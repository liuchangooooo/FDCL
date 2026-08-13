"""Objective ablation: count-in-band vs. total learnability b = p(1-p).

Question this answers (paper alignment): the manuscript will define
per-environment learnability as b(e) = p(e)(1 - p(e)) and score a generator by
the learnability its induced distribution provides. The implementation currently
ranks generators by the number of in-band probe scenes. Would aligning the code
to the paper definition change any generator-selection decision, i.e. does it
force a re-run?

Design: hold the acceptance rule fixed (same hard gate, same symmetric
saturation escape, same margin semantics, same runtime parameters) and swap only
the maximized objective. The two arms are therefore compared against each other,
not against the historical log: the logs were produced by an earlier revision of
the rule, so log agreement is reported only as a drift diagnostic.

Objective encoding: the rule maximizes the integer field ``boundary_count``, so
the b-arm substitutes ``round(SCALE * sum_i b_i)`` into that field and scales the
acceptance margin identically. Gate inputs (valid_rate, duplicate_rate, r_easy,
r_hard, n_scenes) are untouched, so gates and saturation detection are identical
across arms.
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DIVO.curriculum.candidate_verifier import SkillVerifierConfig
from DIVO.curriculum.learnable_frontier import select_generator_boundary

SCALE = 1000.0

CFG = SkillVerifierConfig()
RULE_KWARGS = dict(
    v_min=float(CFG.valid_rate_min),
    d_max=float(CFG.duplicate_rate_max),
    r_easy_max=float(CFG.r_easy_max),
    r_hard_max=float(CFG.r_hard_max),
    diversify_on_hold=bool(CFG.diversify_on_hold),
)


def total_b(signal):
    """Sum of b_i = p_i (1 - p_i) over the probe scenes of one generator."""
    p_values = signal.get("p_values") or []
    if p_values:
        return sum(float(p) * (1.0 - float(p)) for p in p_values)
    return float(signal.get("mean_b", 0.0)) * int(signal.get("n_scenes", 0))


def as_b_objective(signal):
    """Same signal, with the maximized field carrying scaled total learnability."""
    out = copy.deepcopy(dict(signal))
    out["boundary_count"] = int(round(SCALE * total_b(signal)))
    return out


def flags_from_gate_reports(decision, n_candidates):
    """Recover the per-candidate code_pass / freshness flags actually used."""
    code_pass = [True] * n_candidates
    fresh = [True] * n_candidates
    for report in decision.get("gate_reports", []) or []:
        idx = int(report.get("index", -1))
        if 0 <= idx < n_candidates:
            checks = report.get("checks", {}) or {}
            code_pass[idx] = bool(checks.get("code_pass", True))
            fresh[idx] = bool(report.get("fresh_code_ok", True))
    return code_pass, fresh


def load_rounds(paths):
    rounds = []
    for path in paths:
        for line in pathlib.Path(path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            decision = rec.get("skill_library_decision")
            cur = rec.get("skill_library_current_boundary")
            cands = rec.get("skill_library_candidate_boundaries")
            if not decision or not cur or not cands:
                continue
            code_pass, fresh = flags_from_gate_reports(decision, len(cands))
            rounds.append(
                {
                    "run": pathlib.Path(path).parent.name,
                    "current": cur,
                    "candidates": list(cands),
                    "logged": decision,
                    "code_pass": code_pass,
                    "fresh": fresh,
                    "margin": int(decision.get("min_boundary_count_delta", 4)),
                }
            )
    return rounds


def decide(rd, objective, margin):
    if objective == "count":
        cur, cands = rd["current"], rd["candidates"]
        delta = int(margin)
    else:
        cur = as_b_objective(rd["current"])
        cands = [as_b_objective(c) for c in rd["candidates"]]
        delta = int(round(SCALE * margin))
    return select_generator_boundary(
        cur,
        cands,
        min_boundary_count_delta=delta,
        code_pass_flags=rd["code_pass"],
        candidate_fresh_flags=rd["fresh"],
        **RULE_KWARGS,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", help="acgs_evolve_records.jsonl paths")
    parser.add_argument(
        "--b-margins",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 0.875, 1.0],
        help="acceptance margin in total-learnability units; the count margin of "
        "4 scenes corresponds to 4 * b_band, with b_band in [0.1875, 0.25] at K=4",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    rounds = load_rounds(args.logs)
    print(f"evolution rounds with a recorded selection decision: {len(rounds)}")
    if not rounds:
        return

    # Drift diagnostic only: the logs predate the current rule revision.
    log_match = sum(
        int(
            decide(rd, "count", rd["margin"])["action"] == rd["logged"]["action"]
            and decide(rd, "count", rd["margin"])["chosen_index"]
            == rd["logged"]["chosen_index"]
        )
        for rd in rounds
    )
    print(
        f"[drift check] current count rule reproduces the historical log on "
        f"{log_match}/{len(rounds)} rounds "
        f"(logs were produced by an earlier revision; not used as ground truth)\n"
    )

    count_decisions = [decide(rd, "count", rd["margin"]) for rd in rounds]
    n_replace_count = sum(int(d["action"] == "replace") for d in count_decisions)
    print(
        f"count objective: {n_replace_count} replace / "
        f"{len(rounds) - n_replace_count} hold"
    )

    for b_margin in args.b_margins:
        b_decisions = [decide(rd, "b", b_margin) for rd in rounds]
        n_replace_b = sum(int(d["action"] == "replace") for d in b_decisions)
        same_action = sum(
            int(a["action"] == b["action"])
            for a, b in zip(count_decisions, b_decisions)
        )
        same_full = sum(
            int(a["action"] == b["action"] and a["chosen_index"] == b["chosen_index"])
            for a, b in zip(count_decisions, b_decisions)
        )
        n = len(rounds)
        print(
            f"b objective (margin={b_margin:g}): "
            f"{n_replace_b} replace / {n - n_replace_b} hold | "
            f"same action as count: {same_action}/{n} ({100.0 * same_action / n:.1f}%) | "
            f"same action+choice: {same_full}/{n} ({100.0 * same_full / n:.1f}%)"
        )
        if args.verbose:
            for i, (rd, a, b) in enumerate(zip(rounds, count_decisions, b_decisions)):
                if a["action"] == b["action"] and a["chosen_index"] == b["chosen_index"]:
                    continue
                cur_bc = int(rd["current"].get("boundary_count", 0))
                cur_tb = total_b(rd["current"])
                cand_bc = [int(c.get("boundary_count", 0)) for c in rd["candidates"]]
                cand_tb = [round(total_b(c), 3) for c in rd["candidates"]]
                print(
                    f"    round {i} ({rd['run']}): "
                    f"count -> {a['action']}/{a['chosen_index']} ({a['reason']}); "
                    f"b -> {b['action']}/{b['chosen_index']} ({b['reason']}); "
                    f"G_t bc={cur_bc} sum_b={cur_tb:.3f}; "
                    f"cand bc={cand_bc} sum_b={cand_tb}"
                )


if __name__ == "__main__":
    main()
