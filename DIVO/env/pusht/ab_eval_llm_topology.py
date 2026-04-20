import argparse
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np

from DIVO.env.pusht.llm_topology_generator import (
    LLMTopologyGenerator,
    StrategyExecutor,
    build_phase0_topology_generator_prompt,
    build_phase0_topology_generator_prompt_compact,
)


@dataclass
class TrialResult:
    idx: int
    tblock_pose: List[float]
    load_ok: bool = False
    valid_ok: bool = False
    generated_n: int = 0
    fail_stage: str = ""   # "", "llm", "extract", "load", "validate"
    reason: str = ""


def sample_tblock_poses(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-0.18, 0.18, size=(n, 2))
    theta = rng.uniform(-np.pi, np.pi, size=(n, 1))
    return np.concatenate([xy, theta], axis=1)


def build_prompt(which: str, pose: np.ndarray, k: int) -> str:
    if which == "long":
        return build_phase0_topology_generator_prompt(pose, k)
    if which == "compact":
        return build_phase0_topology_generator_prompt_compact(pose, k)
    raise ValueError(f"Unknown prompt: {which}")


def run_eval(
    prompt_type: str,
    api_type: str,
    api_key: Optional[str],
    model: Optional[str],
    base_url: Optional[str],
    temperature: float,
    num_trials: int,
    num_obstacles: int,
    seed: int,
    out_json: str,
    max_fail_save: int,
) -> Dict[str, Any]:
    poses = sample_tblock_poses(num_trials, seed)

    llm = LLMTopologyGenerator(
        api_type=api_type,
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=temperature,
        verbose=False,
    )
    executor = StrategyExecutor()

    results: List[TrialResult] = []
    fail_examples: List[Dict[str, Any]] = []

    for i in range(num_trials):
        pose = poses[i]
        tr = TrialResult(idx=i, tblock_pose=[float(pose[0]), float(pose[1]), float(pose[2])])

        prompt = build_prompt(prompt_type, pose, num_obstacles)

        response = llm._call_llm(prompt)
        if response is None:
            tr.fail_stage = "llm"
            results.append(tr)
            continue

        code = llm._extract_code(response)
        if code is None:
            tr.fail_stage = "extract"
            if len(fail_examples) < max_fail_save:
                fail_examples.append({"stage": "extract", "pose": tr.tblock_pose, "response": response[:4000]})
            results.append(tr)
            continue

        if not executor.load_topology_generator(code):
            tr.fail_stage = "load"
            if len(fail_examples) < max_fail_save:
                fail_examples.append({"stage": "load", "pose": tr.tblock_pose, "code": code[:4000]})
            results.append(tr)
            continue

        tr.load_ok = True
        obstacles = executor.generate(pose, num_obstacles)
        tr.generated_n = len(obstacles)

        ok, reason = executor.validate_obstacles(obstacles, pose)
        tr.valid_ok = bool(ok)
        if not ok:
            tr.fail_stage = "validate"
            tr.reason = str(reason)
            if len(fail_examples) < max_fail_save:
                fail_examples.append(
                    {"stage": "validate", "pose": tr.tblock_pose, "reason": reason, "obstacles": obstacles}
                )

        results.append(tr)

    n = len(results)
    load_ok = sum(1 for r in results if r.load_ok)
    valid_ok = sum(1 for r in results if r.valid_ok)
    avg_n = float(np.mean([r.generated_n for r in results])) if results else 0.0

    stage_counts: Dict[str, int] = {}
    for r in results:
        stage_counts[r.fail_stage] = stage_counts.get(r.fail_stage, 0) + 1

    summary = {
        "prompt_type": prompt_type,
        "num_trials": n,
        "num_obstacles": num_obstacles,
        "seed": seed,
        "api_type": api_type,
        "model": llm.model,
        "temperature": temperature,
        "load_ok_rate": load_ok / n if n else 0.0,
        "valid_ok_rate": valid_ok / n if n else 0.0,
        "avg_generated_n": avg_n,
        "fail_stage_counts": stage_counts,
    }

    payload = {
        "summary": summary,
        "results": [asdict(r) for r in results],
        "fail_examples": fail_examples,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", choices=["long", "compact"], default="compact")
    ap.add_argument("--api-type", choices=["deepseek", "openai"], default="deepseek")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("-n", "--num-trials", type=int, default=100)
    ap.add_argument("-k", "--num-obstacles", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default="ab_eval_results.json")
    ap.add_argument("--max-fail-save", type=int, default=10)
    args = ap.parse_args()

    payload = run_eval(
        prompt_type=args.prompt,
        api_type=args.api_type,
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        num_trials=args.num_trials,
        num_obstacles=args.num_obstacles,
        seed=args.seed,
        out_json=args.out_json,
        max_fail_save=args.max_fail_save,
    )

    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"saved: {args.out_json}")


if __name__ == "__main__":
    main()