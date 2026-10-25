"""Reproducible spatial-plus-structural matching benchmark inspired by the
pasted benchmark_comprehensive.py.

This is a supplemental synthetic benchmark, not the alpha=1 base-paper
reproduction. It uses alpha=0.7 and a coordinate unary cost, matching the
pasted script's intended feature-assisted objective. Generic baselines are
explicitly adapted to produce exact-L partial permutations.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment, quadratic_assignment

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gnccp_advanced_v3 import AdvancedGNCCP
from gnccp_baseline_2018 import OriginalGNCCP_2018
from gnccp_v2_classic import UniversalGNCCP

try:
    import ot
except ImportError:
    ot = None


ALPHA = 0.7


def perturb_edges(A, D, rng, sigma):
    """Independently remove/add the paper-scaled number of undirected edges."""
    out = A.copy()
    n = len(out)
    edges = np.argwhere(np.triu(out > 0, 1))
    k = min(int(round(0.5 * sigma * len(edges))), len(edges))
    if k:
        removed = edges[rng.choice(len(edges), k, replace=False)]
        out[removed[:, 0], removed[:, 1]] = 0
        out[removed[:, 1], removed[:, 0]] = 0
    present = out > 0
    absent = np.argwhere(np.triu(~present, 1))
    if k and len(absent):
        added = absent[rng.choice(len(absent), min(k, len(absent)), replace=False)]
        weights = np.exp(-D[added[:, 0], added[:, 1]])
        out[added[:, 0], added[:, 1]] = weights
        out[added[:, 1], added[:, 0]] = weights
    return out


def make_instance(M, N, L, sigma, density, seed):
    rng = np.random.default_rng(seed)
    G = rng.uniform(0, 1, (M, 2))
    H = rng.uniform(0, 1, (N, 2))
    gi = rng.choice(M, L, replace=False)
    hj = rng.choice(N, L, replace=False)
    rng.shuffle(hj)
    G[gi] = H[hj] + rng.normal(0, sigma, (L, 2))
    gt = list(zip(gi.tolist(), hj.tolist()))

    DG = np.linalg.norm(G[:, None] - G[None, :], axis=-1)
    DH = np.linalg.norm(H[:, None] - H[None, :], axis=-1)

    def adjacency(D):
        upper = D[np.triu_indices(len(D), 1)]
        threshold = np.quantile(upper, density) if upper.size else 0.0
        A = np.where(D < threshold, np.exp(-D), 0.0)
        np.fill_diagonal(A, 0.0)
        return A

    AG = perturb_edges(adjacency(DG), DG, rng, sigma).astype(np.float32)
    AH = perturb_edges(adjacency(DH), DH, rng, sigma).astype(np.float32)
    C = np.linalg.norm(G[:, None] - H[None, :], axis=-1)
    cmax = float(C.max()) if C.size else 0.0
    if cmax > 0:
        C = C / cmax
    return AG, AH, C.astype(np.float32), gt


def wcs_objective(X, AG, AH, C, alpha=ALPHA):
    X = np.asarray(X, dtype=np.float64)
    r = X.sum(axis=1)
    structural = (r[:, None] * np.asarray(AG) * r[None, :]) - X @ np.asarray(AH) @ X.T
    return float(alpha * np.sum(structural * structural) + (1 - alpha) * np.sum(np.asarray(C) * X))


def scores_to_exact_l(scores, L):
    """Maximum-score exact-L injective assignment via a dummy-node augmentation."""
    scores = np.asarray(scores, dtype=np.float64)
    M, N = scores.shape
    size = M + N - L
    cost = np.full((size, size), np.inf, dtype=np.float64)
    cost[:M, :N] = -scores
    cost[:M, N:] = 0.0
    cost[M:, :N] = 0.0
    rows, cols = linear_sum_assignment(cost)
    pairs = [(int(i), int(j)) for i, j in zip(rows, cols) if i < M and j < N]
    if len(pairs) != L:
        raise RuntimeError(f"exact-L projection returned {len(pairs)} matches, expected {L}")
    return pairs


def faq_solve(AG, AH, C, L, seed):
    M, N = C.shape
    if M > N:
        raise ValueError("FAQ adapter currently expects M <= N")
    A_pad = np.zeros((N, N), dtype=np.float64)
    # SciPy FAQ minimizes its QAP trace objective; negate only one graph matrix
    # to make a strong common-edge correspondence favorable.
    A_pad[:M, :M] = -AG
    result = quadratic_assignment(
        A_pad, np.asarray(AH, dtype=np.float64), method="faq",
        options={"maxiter": 30, "rng": int(seed)},
    )
    pairs = [(i, int(result.col_ind[i])) for i in range(M) if result.col_ind[i] < N]
    pairs.sort(key=lambda p: (float(C[p[0], p[1]]), p[0], p[1]))
    return pairs[:L]


def pot_solve(C, L):
    if ot is None:
        raise RuntimeError("POT is not installed")
    M, N = C.shape
    a = np.full(M, 1 / M)
    b = np.full(N, 1 / N)
    mass = float(L / min(M, N))
    T = ot.partial.partial_wasserstein(a, b, C.astype(np.float64), m=mass)
    return scores_to_exact_l(T, L)


def coordinate_only_solve(C, L):
    """Exact-L assignment using only the shared-frame unary coordinate cost."""
    return scores_to_exact_l(-np.asarray(C, dtype=np.float64), L)


def fgw_solve(AG, AH, C, L):
    if ot is None:
        raise RuntimeError("POT is not installed")
    M, N = C.shape
    p = np.full(M, 1 / M)
    q = np.full(N, 1 / N)
    c1 = np.asarray(AG, dtype=np.float64)
    c2 = np.asarray(AH, dtype=np.float64)
    n1, n2 = np.linalg.norm(c1), np.linalg.norm(c2)
    if n1:
        c1 /= n1
    if n2:
        c2 /= n2
    T = ot.gromov.fused_gromov_wasserstein(
        C.astype(np.float64), c1, c2, p, q,
        alpha=0.5, max_iter=100, tol_rel=1e-7, tol_abs=1e-7,
    )
    return scores_to_exact_l(T, L)


def solvers(seed):
    return {
        "Fast-WCS v3": lambda AG, AH, C, L: AdvancedGNCCP(
            alpha=ALPHA, max_iter=8, fw_max_iter=3, tol=1e-3,
            line_search="quartic", normalization="none",
            oracle="exact_augmented", local_refinement=True,
            enable_exchange_moves=True, seed=seed,
        ).solve(AG, AH, C=C, L=L).matches,
        "Fast-WCS v2": lambda AG, AH, C, L: UniversalGNCCP(
            alpha=ALPHA, max_iter=8, fw_max_iter=3, line_search="closed_form",
            normalize_fro=False, oracle="dummy_padded",
            local_refinement=True, seed=seed,
        ).solve(AG, AH, C=C, L=L).matches,
        "Local 2018-style RLX1F": lambda AG, AH, C, L: OriginalGNCCP_2018(
            alpha=ALPHA, max_iter=8, fw_iter=3,
        ).solve(AG, AH, L=L, C=C).matches,
        "SciPy FAQ + unary truncation": lambda AG, AH, C, L: faq_solve(AG, AH, C, L, seed),
        "Coordinate-only Hungarian": lambda AG, AH, C, L: coordinate_only_solve(C, L),
        "POT partial OT + exact-L projection": lambda AG, AH, C, L: pot_solve(C, L),
        "POT FGW + exact-L projection": lambda AG, AH, C, L: fgw_solve(AG, AH, C, L),
    }


def configs():
    return {
        "noise": [(30, 35, 25, s, .5) for s in [0., .02, .04, .06, .08, .10]],
        "size": [(m, m+5, m-5, .03, .5) for m in [10, 15, 20, 25, 30, 40, 50]],
        "scale": [(20, 30, 15, .03, .4), (40, 60, 30, .03, .4), (70, 100, 50, .03, .4)],
        "outliers": [(30, 30+k, 25, .03, .5) for k in [0, 3, 5, 8, 10, 15]],
        "density": [(30, 35, 25, .03, d) for d in [.1, .2, .3, .5, .7, .9]],
    }


def run(runs=10, seed_offset=71000, scenarios="all", output=None):
    all_cfg = configs()
    selected = list(all_cfg) if scenarios == "all" else scenarios.split(",")
    unknown = set(selected) - set(all_cfg)
    if unknown:
        raise ValueError(f"unknown scenarios: {sorted(unknown)}")
    methods = list(solvers(seed_offset))
    payload = {
        "protocol": {
            "description": "supplemental synthetic feature-assisted WCS benchmark inspired by pasted comprehensive script",
            "base_paper_reproduction": False,
            "alpha": ALPHA,
            "features": "normalized Euclidean coordinate cost C; matched coordinates share frame",
            "topology": "distance-quantile sparse graphs with independent per-graph add/remove edge perturbations",
            "edge_perturbation": "round(0.5*sigma*undirected_edge_count) added and removed edges per graph; zero at sigma=0",
            "runs_per_setting": runs,
            "seed_offset": seed_offset,
            "accuracy": "correct planted pairs / L",
            "objective": "common discrete alpha=0.7 graph residual plus normalized coordinate cost; lower is better",
            "timing": "solver plus required baseline projection, per-instance wall clock; randomized method order",
            "adaptations": [
                "FAQ solves padded full QAP and selects L pairs by unary coordinate cost; not an exact-L QAP optimizer.",
                "POT partial OT and FGW return soft couplings, projected to exact-L maximum-score injective matchings.",
                "POT/FGW optimize their native transport criteria; reported common WCS objective is an external evaluation metric.",
                "Local 2018-style RLX1F is an in-workspace reconstruction, not original author code.",
            ],
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "pot": getattr(ot, "__version__", None) if ot is not None else None,
        },
        "method_settings": {
            "Fast-WCS v3": "alpha=.7, max_iter=8, fw=3, quartic, exact augmented oracle, local refinement",
            "Fast-WCS v2": "alpha=.7, max_iter=8, fw=3, closed-form classic line search, dummy-padded oracle",
            "Local 2018-style RLX1F": "local reconstruction, alpha=.7, max_iter=8, fw_iter=3",
            "SciPy FAQ + unary truncation": "padded full QAP then keep L lowest unary-cost pairs",
            "Coordinate-only Hungarian": "exact-L assignment minimizing only normalized unary coordinate cost C",
            "POT partial OT + exact-L projection": "partial Wasserstein on C, then exact-L maximum-score projection",
            "POT FGW + exact-L projection": "FGW alpha=.5, then exact-L maximum-score projection",
        },
        "results": {},
    }
    for scenario in selected:
        payload["results"][scenario] = []
        for ci, (M, N, L, sigma, density) in enumerate(all_cfg[scenario]):
            stats_by = {name: {"accuracy": [], "objective": [], "runtime_ms": [], "errors": []} for name in methods}
            for run_i in range(runs):
                seed = seed_offset + sum(ord(c) for c in scenario) * 100000 + ci * 1000 + run_i
                AG, AH, C, truth = make_instance(M, N, L, sigma, density, seed)
                order = list(methods)
                np.random.default_rng(seed).shuffle(order)
                for name in order:
                    t0 = time.perf_counter()
                    try:
                        matches = solvers(seed)[name](AG, AH, C, L)
                        elapsed = (time.perf_counter() - t0) * 1000
                        rows = [i for i, _ in matches]
                        cols = [j for _, j in matches]
                        if len(matches) != L or len(set(rows)) != L or len(set(cols)) != L:
                            raise ValueError(f"invalid exact-{L} partial permutation ({len(matches)} pairs)")
                        acc = sum((i, j) in set(truth) for i, j in matches) / L
                        obj = wcs_objective(_pairs_matrix(matches, M, N), AG, AH, C)
                        stats_by[name]["accuracy"].append(float(acc))
                        stats_by[name]["objective"].append(obj)
                        stats_by[name]["runtime_ms"].append(float(elapsed))
                    except Exception as e:
                        stats_by[name]["errors"].append(f"seed={seed}: {type(e).__name__}: {e}")
            row = {"M": M, "N": N, "L": L, "sigma": sigma, "density": density, "methods": {}}
            for name, s in stats_by.items():
                def summary(key):
                    a = s[key]
                    return {
                        "mean": float(np.mean(a)) if a else None,
                        "sd": float(np.std(a, ddof=1)) if len(a) > 1 else (0.0 if a else None),
                    }
                row["methods"][name] = {
                    "accuracy": summary("accuracy"),
                    "objective": summary("objective"),
                    "runtime_ms": {**summary("runtime_ms"), "median": float(np.median(s["runtime_ms"])) if s["runtime_ms"] else None},
                    "n_success": len(s["accuracy"]),
                    "errors": s["errors"],
                    "samples": {k: s[k] for k in ("accuracy", "objective", "runtime_ms")},
                }
            payload["results"][scenario].append(row)
            print(f"{scenario} {ci+1}/{len(all_cfg[scenario])} M={M} N={N} L={L} sigma={sigma} density={density}", flush=True)
            if output:
                Path(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Paired, condition-stratified summaries: each benchmark setting is one
    # bootstrap unit, preserving the pairing of seeds within that setting.
    rng = np.random.default_rng(seed_offset ^ 0x44A1)
    payload["paired_summary"] = {}
    for scenario, rows in payload["results"].items():
        comparisons = {}
        candidates = [m for m in methods if m != "Fast-WCS v3"]
        for candidate in candidates:
            comparisons[candidate] = {}
            for metric in ("accuracy", "objective", "runtime_ms"):
                setting_diffs = []
                for row in rows:
                    base = row["methods"]["Fast-WCS v3"]["samples"][metric]
                    test = row["methods"][candidate]["samples"][metric]
                    if len(base) == len(test) and base:
                        setting_diffs.append(float(np.mean(np.asarray(test) - np.asarray(base))))
                if not setting_diffs:
                    comparisons[candidate][metric] = None
                    continue
                d = np.asarray(setting_diffs, dtype=float)
                boot = np.mean(rng.choice(d, size=(10000, len(d)), replace=True), axis=1)
                try:
                    _, pvalue = stats.wilcoxon(d, alternative="two-sided", method="auto")
                    pvalue = float(pvalue)
                except ValueError:
                    pvalue = 1.0
                comparisons[candidate][metric] = {
                    "mean_setting_paired_difference_candidate_minus_v3": float(d.mean()),
                    "bootstrap_95_ci_over_settings": [float(x) for x in np.quantile(boot, [.025, .975])],
                    "wilcoxon_p_unadjusted_over_setting_means": pvalue,
                    "number_of_settings": int(len(d)),
                }
        payload["paired_summary"][scenario] = comparisons
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved {Path(output).resolve()}")
    return payload


def _pairs_matrix(matches, M, N):
    X = np.zeros((M, N), dtype=np.float32)
    for i, j in matches:
        X[i, j] = 1.0
    return X


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=71000)
    parser.add_argument("--scenarios", default="all", help="all or comma list: noise,size,scale,outliers,density")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "comprehensive_adapted_results.json")
    args = parser.parse_args()
    run(args.runs, args.seed_offset, args.scenarios, args.output)
