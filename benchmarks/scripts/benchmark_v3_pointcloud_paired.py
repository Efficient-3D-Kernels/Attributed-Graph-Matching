"""Paired holdout comparison of v3 point-cloud settings.

This is a synthetic rigid 3D stress test, not a molecular benchmark. It borrows
paired bootstrap intervals and Wilcoxon comparisons from the pasted harness,
while reporting metrics that apply to correspondence recovery.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

from benchmark_v3_pointcloud import make_pair
from gnccp_advanced_v3 import AdvancedGNCCP, robust_kabsch_superposition


CONFIGS = {
    "v3_default": {"rbf_sigma_scale": 1.0, "polish_passes": 2, "fw_max_iter": 3},
    "v3_narrow_kernel": {"rbf_sigma_scale": 0.85, "polish_passes": 2, "fw_max_iter": 3},
    "v3_narrow_kernel_polish4": {"rbf_sigma_scale": 0.85, "polish_passes": 4, "fw_max_iter": 3},
    "v3_fast_fw2_narrow_kernel": {"rbf_sigma_scale": 0.85, "polish_passes": 2, "fw_max_iter": 2},
}


def score(matches, source, target, truth, L, elapsed_ms):
    correct = sum(truth.get(i) == j for i, j in matches)
    pairs = np.asarray(matches, dtype=int)
    if len(pairs) >= 3:
        rmsd = robust_kabsch_superposition(
            source[pairs[:, 0]], target[pairs[:, 1]], trim_ratio=0.0
        )["rmsd"]
    else:
        rmsd = float("nan")
    return {
        "correct_matches": int(correct),
        "pair_accuracy": float(correct / L),
        "all_pair_rmsd": float(rmsd),
        "wall_ms": float(elapsed_ms),
    }


def paired_summary(x, y, rng, better):
    """Difference y-x, paired percentile bootstrap CI, and Wilcoxon p-value."""
    d = np.asarray(y, dtype=float) - np.asarray(x, dtype=float)
    boot = np.mean(rng.choice(d, size=(10000, len(d)), replace=True), axis=1)
    try:
        _, p = stats.wilcoxon(d, alternative="two-sided", method="auto")
        p = float(p)
    except ValueError:
        p = 1.0
    return {
        "mean_difference_test_minus_default": float(d.mean()),
        "paired_bootstrap_95_ci": [float(q) for q in np.quantile(boot, [0.025, 0.975])],
        "wilcoxon_p_unadjusted": p,
        "better_when": better,
    }


def run(n_pairs: int, seed_offset: int, output: Path):
    M, N, L = 16, 22, 12
    samples = {name: [] for name in CONFIGS}
    for i in range(n_pairs):
        seed = seed_offset + i
        source, target, truth = make_pair(seed, M, N, L, noise=0.03, outlier_scale=1.5)
        order = list(CONFIGS)
        np.random.default_rng(seed).shuffle(order)
        for name in order:
            cfg = CONFIGS[name]
            solver = AdvancedGNCCP(
                alpha=0.7, max_iter=8, fw_max_iter=cfg["fw_max_iter"], seed=seed,
                local_refinement=True,
            )
            t0 = time.perf_counter()
            result = solver.match_point_clouds(
                source, target, L=L,
                n_dgm_iters=cfg["polish_passes"],
                rbf_sigma_scale=cfg["rbf_sigma_scale"],
            )
            elapsed = (time.perf_counter() - t0) * 1000
            rows = [i for i, _ in result.matches]
            cols = [j for _, j in result.matches]
            if len(result.matches) != L or len(set(rows)) != L or len(set(cols)) != L:
                raise ValueError(f"{name} returned an invalid exact-{L} partial permutation")
            samples[name].append(score(result.matches, source, target, truth, L, elapsed))

    rng = np.random.default_rng(seed_offset ^ 0x5A17)
    summaries = {}
    for name, rows in samples.items():
        acc = np.asarray([r["pair_accuracy"] for r in rows])
        rmsd = np.asarray([r["all_pair_rmsd"] for r in rows])
        runtime = np.asarray([r["wall_ms"] for r in rows])
        boot = np.mean(rng.choice(acc, size=(10000, len(acc)), replace=True), axis=1)
        summaries[name] = {
            "mean_pair_accuracy": float(acc.mean()),
            "bootstrap_95_ci_pair_accuracy": [float(q) for q in np.quantile(boot, [0.025, 0.975])],
            "median_all_pair_rmsd": float(np.nanmedian(rmsd)),
            "rmsd_below_0_1_units_rate": float(np.mean(rmsd < 0.1)),
            "rmsd_below_1_0_units_rate": float(np.mean(rmsd < 1.0)),
            "median_wall_ms": float(np.median(runtime)),
        }

    comparisons = {}
    base = samples["v3_default"]
    for name in list(CONFIGS)[1:]:
        test = samples[name]
        comparisons[name + "_vs_v3_default"] = {
            "pair_accuracy": paired_summary(
                [r["pair_accuracy"] for r in base],
                [r["pair_accuracy"] for r in test], rng,
                "positive favors candidate",
            ),
            "all_pair_rmsd": paired_summary(
                [r["all_pair_rmsd"] for r in base],
                [r["all_pair_rmsd"] for r in test], rng,
                "negative favors candidate",
            ),
            "wall_ms": paired_summary(
                [r["wall_ms"] for r in base],
                [r["wall_ms"] for r in test], rng,
                "negative favors candidate",
            ),
        }

    payload = {
        "protocol": {
            "type": "paired synthetic rigid 3D point-cloud holdout; not molecular data",
            "pairs": n_pairs,
            "seed_offset": seed_offset,
            "M": M, "N": N, "L": L,
            "noise_sigma": 0.03,
            "outlier_scale": 1.5,
            "coordinate_units": "synthetic units, not angstroms",
            "rmsd": "untrimmed Kabsch RMSD over all predicted L pairs",
            "timing": "match_point_clouds wall time, including feature/graph construction",
            "statistics": "paired percentile bootstrap CIs; two-sided Wilcoxon signed-rank; p-values unadjusted for nine candidate-by-metric comparisons",
        },
        "configs": CONFIGS,
        "summary": summaries,
        "paired_comparisons": comparisons,
        "samples": samples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for name, s in summaries.items():
        lo, hi = s["bootstrap_95_ci_pair_accuracy"]
        print(
            f"{name:29s} acc={s['mean_pair_accuracy']*100:5.1f}% "
            f"CI=[{lo*100:.1f},{hi*100:.1f}] RMSD={s['median_all_pair_rmsd']:.3f} "
            f"units <1={s['rmsd_below_1_0_units_rate']*100:.1f}% "
            f"time={s['median_wall_ms']:.2f}ms"
        )
    for name, comparison in comparisons.items():
        a = comparison["pair_accuracy"]
        r = comparison["all_pair_rmsd"]
        t = comparison["wall_ms"]
        print(
            f"{name}: accuracy delta={a['mean_difference_test_minus_default']*100:+.2f}pp "
            f"CI=[{a['paired_bootstrap_95_ci'][0]*100:+.2f},{a['paired_bootstrap_95_ci'][1]*100:+.2f}] "
            f"p={a['wilcoxon_p_unadjusted']:.4g}; "
            f"RMSD delta={r['mean_difference_test_minus_default']:+.4f} "
            f"p={r['wilcoxon_p_unadjusted']:.4g}; "
            f"time delta={t['mean_difference_test_minus_default']:+.3f}ms "
            f"p={t['wilcoxon_p_unadjusted']:.4g}"
        )
    print(f"Saved {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=300)
    parser.add_argument("--seed-offset", type=int, default=60000)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).with_name("v3_pointcloud_paired_results.json"),
    )
    args = parser.parse_args()
    run(args.pairs, args.seed_offset, args.output)
