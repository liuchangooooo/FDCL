"""Does the skill-library learnability signal discriminate between generators?

Two offline experiments on the recorded evolution logs, both computed from the
per-scene response rates already stored in ``acgs_evolve_records.jsonl``. Neither
requires enlarging the probe budget.

Experiment 1 -- reliability of the generator ranking.
  Within one round the observed spread of J_L = mean_i r_i(1-r_i) across
  generators mixes real differences with probe sampling noise:

      Var_observed = Var_true + Var_noise,

  so the fraction of the observed spread that is real is
  ``reliability = 1 - Var_noise / Var_observed``, with Var_noise estimated by the
  squared standard error of each generator's own J_L. Reliability near 1 means
  the ranking reflects the generators; near 0 means it reflects which scenes
  happened to be sampled.

  A split-half check is also reported. Note that two complementary halves of a
  fixed scene pool are exactly anti-correlated around the pool mean, so the null
  of that correlation is -1, not 0; the informative quantity is the implied
  variance ratio (1+r)/(1-r), together with the assumption-light rate at which
  the two halves agree on the best generator.

Experiment 2 -- does the signal survive where single-policy feedback does not.
  The spread of the deployed skill's success rate across generators is compared
  with the spread of J_L across the same generators, each against its own
  sampling noise, swept over increasingly strict saturation thresholds. The Q2
  claim requires a regime in which the single-policy spread is no longer
  separable from noise while the learnability spread still is.
"""

from __future__ import annotations

import argparse
import glob
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.replay_b_objective import load_rounds

EPS = 1e-12


def round_matrix(rd):
    """Per-scene learnability for each generator in one round, plus w_0 success."""
    gens = [rd["current"]] + rd["candidates"]
    bs, w0 = [], []
    for g in gens:
        p = np.asarray(g.get("p_values") or [], dtype=np.float64)
        if p.size < 4:
            return None
        bs.append(p * (1.0 - p))
        w0.append(float(g.get("w0_success_rate", np.nan)))
    if len(bs) < 2:
        return None
    return bs, np.asarray(w0, dtype=np.float64)


def variance_reliability(values, se):
    """Fraction of the observed between-generator variance that is not noise."""
    v_obs = float(np.var(values, ddof=1))
    v_noise = float(np.mean(np.asarray(se) ** 2))
    if v_obs <= EPS:
        return 0.0, v_obs, v_noise
    return float(max(0.0, 1.0 - v_noise / v_obs)), v_obs, v_noise


def experiment1(rounds, n_splits, rng):
    rel, vobs, vnoise = [], [], []
    top1_hits = top1_total = 0
    split_r_num, split_r_a, split_r_b = [], [], []
    n_gen = []

    for rd in rounds:
        got = round_matrix(rd)
        if got is None:
            continue
        bs, _ = got
        n_gen.append(len(bs))
        jl = np.array([b.mean() for b in bs])
        se = np.array([b.std(ddof=1) / np.sqrt(len(b)) for b in bs])
        r, vo, vn = variance_reliability(jl, se)
        rel.append(r)
        vobs.append(vo)
        vnoise.append(vn)

        m = min(len(b) for b in bs)
        half = m // 2
        if half < 2:
            continue
        for _ in range(n_splits):
            idx = rng.permutation(m)
            ja = np.array([b[:m][idx[:half]].mean() for b in bs])
            jb = np.array([b[:m][idx[half : 2 * half]].mean() for b in bs])
            top1_hits += int(np.argmax(ja) == np.argmax(jb))
            top1_total += 1
            split_r_a.extend(ja - ja.mean())
            split_r_b.extend(jb - jb.mean())

    a, b = np.asarray(split_r_a), np.asarray(split_r_b)
    r_half = float(np.corrcoef(a, b)[0, 1]) if a.size > 2 else np.nan
    v_ratio = (1.0 + r_half) / (1.0 - r_half) if abs(1.0 - r_half) > EPS else np.nan
    chance = float(np.mean([1.0 / n for n in n_gen])) if n_gen else np.nan

    print("=" * 70)
    print("Experiment 1 -- reliability of the J_L ranking of generators")
    print("=" * 70)
    print(f"  rounds used: {len(rel)}   generators per round: {np.mean(n_gen):.2f}")
    print()
    print("  variance decomposition (per round, then aggregated)")
    print(f"    observed between-generator variance : {np.median(vobs):.3e}")
    print(f"    estimated sampling-noise variance   : {np.median(vnoise):.3e}")
    print(f"    reliability = 1 - noise/observed    : median {np.median(rel):.3f}   "
          f"mean {np.mean(rel):.3f}")
    print(f"    rounds with reliability > 0.5       : "
          f"{100.0 * np.mean(np.asarray(rel) > 0.5):.0f}%")
    print(f"    rounds with reliability = 0         : "
          f"{100.0 * np.mean(np.asarray(rel) <= 0.0):.0f}%")
    print()
    print("  split-half check (null of the correlation is -1, not 0)")
    print(f"    correlation between halves          : {r_half:.3f}")
    print(f"    implied true/noise variance ratio   : {v_ratio:.3f}")
    print(f"    halves agree on the best generator  : "
          f"{100.0 * top1_hits / max(top1_total, 1):.1f}%   "
          f"(chance {100.0 * chance:.1f}%)")
    print()


def experiment2(rounds, thresholds, sep_k):
    rows = []
    for rd in rounds:
        got = round_matrix(rd)
        if got is None:
            continue
        bs, w0 = got
        if np.isnan(w0).any():
            continue
        m = float(np.mean([len(b) for b in bs]))
        jl = np.array([b.mean() for b in bs])
        se_jl = float(np.mean([b.std(ddof=1) / np.sqrt(len(b)) for b in bs]))
        se_w0 = float(np.mean([np.sqrt(max(p * (1.0 - p), 0.0) / m) for p in w0]))
        rows.append(
            {
                "w0_mean": float(w0.mean()),
                "w0_range": float(w0.max() - w0.min()),
                "se_w0": se_w0,
                "jl_range": float(jl.max() - jl.min()),
                "se_jl": se_jl,
            }
        )

    if not rows:
        print("Experiment 2: no usable rounds")
        return

    w0_mean = np.array([r["w0_mean"] for r in rows])
    w0_range = np.array([r["w0_range"] for r in rows])
    se_w0 = np.array([r["se_w0"] for r in rows])
    jl_range = np.array([r["jl_range"] for r in rows])
    se_jl = np.array([r["se_jl"] for r in rows])

    print("=" * 70)
    print("Experiment 2 -- single-policy feedback vs learnability, by saturation")
    print("=" * 70)
    print(f"  'separable' = spread exceeds {sep_k} x its own sampling noise")
    print()
    print(f"  {'w0 success >=':>14} {'n':>4} {'single-policy':>26} {'learnability':>26}")
    print(f"  {'':>14} {'':>4} {'spread  sep%':>26} {'spread  sep%':>26}")
    for t in thresholds:
        mask = w0_mean >= t
        n = int(mask.sum())
        if n < 5:
            print(f"  {t:>14.2f} {n:>4}   (too few rounds)")
            continue
        okw = mask & (se_w0 > 0)
        okj = mask & (se_jl > 0)
        w_sep = 100.0 * np.mean(w0_range[okw] > sep_k * se_w0[okw]) if okw.any() else np.nan
        j_sep = 100.0 * np.mean(jl_range[okj] > sep_k * se_jl[okj]) if okj.any() else np.nan
        print(
            f"  {t:>14.2f} {n:>4}   "
            f"{np.median(w0_range[mask]):>10.4f} {w_sep:>10.0f}%   "
            f"{np.median(jl_range[mask]):>10.4f} {j_sep:>10.0f}%"
        )
    print()
    print("  Q2 holds only where the single-policy column is no longer separable")
    print("  while the learnability column still is.")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="*")
    parser.add_argument("--splits", type=int, default=200)
    parser.add_argument("--sep-k", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths = args.logs or sorted(glob.glob("data/outputs/*/*/acgs_evolve_records.jsonl"))
    rounds = load_rounds(paths)
    print(f"evolution rounds with a recorded selection decision: {len(rounds)}\n")

    rng = np.random.default_rng(args.seed)
    experiment1(rounds, args.splits, rng)
    experiment2(rounds, [0.0, 0.80, 0.90, 0.95, 0.98, 1.0], args.sep_k)


if __name__ == "__main__":
    main()
