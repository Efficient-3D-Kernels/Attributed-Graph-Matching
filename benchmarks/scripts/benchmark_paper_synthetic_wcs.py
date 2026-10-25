"""Reproducible reconstruction of Yang et al. (2018) synthetic WCS experiments.

Run the full WCS and PIW synthetic protocols with:
    python benchmarks/benchmark_paper_synthetic_wcs.py --runs 30

This is an independent reimplementation of the paper's generator. The paper does
not fully specify random seeds, rounding of edge perturbations, or solver settings;
those operational choices are recorded in the output JSON. Generic PyGMtools
solvers are adapted by taking their relaxed MxN scores and applying an exact-L
linear assignment projection, so they remain clearly distinct from native
partial-matching implementations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import socket
import statistics
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
try:
    from threadpoolctl import threadpool_limits, threadpool_info
except ImportError:
    threadpool_limits = None
    threadpool_info = lambda: []

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gnccp_advanced_v3 import AdvancedGNCCP
from gnccp_baseline_2018 import OriginalGNCCP_2018
from gnccp_v2_classic import UniversalGNCCP

try:
    import pygmtools as pygm
except ImportError:  # benchmark still runs the in-repository GNCCP solvers
    pygm = None


def _sweep_configs():
    # Values follow the text of Section III-A of the base paper.
    wcs = {
        "noise": [dict(M=30, N=35, L=25, density=.5, sigma=s) for s in np.round(np.arange(0, .1001, .01), 2)],
        "size": [dict(M=m, N=m+5, L=m-5, density=.5, sigma=.05) for m in range(20, 41, 2)],
        "outliers": [dict(M=30, N=35, L=l, density=.5, sigma=.05) for l in range(30, 19, -1)],
        "density": [dict(M=30, N=35, L=25, density=d, sigma=.05) for d in np.round(np.arange(.1, 1.001, .1), 1)],
    }
    # PIW (part-in-whole) companion sweeps in the same base-paper section.
    piw = {
        "piw_noise": [dict(M=45, N=50, L=45, density=.5, sigma=s) for s in np.round(np.arange(0, .1001, .01), 2)],
        "piw_size": [dict(M=n-5, N=n, L=n-5, density=.5, sigma=.05) for n in range(40, 61, 2)],
        "piw_outliers": [dict(M=l, N=50, L=l, density=.5, sigma=.05) for l in range(50, 39, -1)],
        "piw_density": [dict(M=45, N=50, L=45, density=d, sigma=.05) for d in np.round(np.arange(.1, 1.001, .1), 1)],
    }
    return {**wcs, **piw}


def _perturb_edges(A, distances, rng, sigma):
    n = A.shape[0]
    edges = np.transpose(np.triu(A != 0, 1).nonzero())
    k = int(round(.5 * sigma * len(edges)))
    if k == 0:
        return A
    # Select edge removals from the sampled graph and additions from its complement.
    remove = edges[rng.choice(len(edges), size=min(k, len(edges)), replace=False)] if len(edges) else np.empty((0, 2), int)
    present = set(map(tuple, edges.tolist()))
    absent = np.array([(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in present], dtype=int).reshape(-1, 2)
    add = absent[rng.choice(len(absent), size=min(k, len(absent)), replace=False)] if len(absent) else np.empty((0, 2), int)
    out = A.copy()
    if len(remove): out[remove[:, 0], remove[:, 1]] = out[remove[:, 1], remove[:, 0]] = 0
    if len(add):
        # The paper defines edge weights as Euclidean point distances. Apply
        # that rule to noise-added edges too; unit weights were an arbitrary
        # artifact of the first exploratory harness.
        out[add[:, 0], add[:, 1]] = distances[add[:, 0], add[:, 1]]
        out[add[:, 1], add[:, 0]] = distances[add[:, 1], add[:, 0]]
    return out


def make_instance(cfg, seed):
    rng = np.random.default_rng(seed)
    M, N, L = cfg["M"], cfg["N"], cfg["L"]
    H = rng.uniform(0, 1, (N, 2))
    G = rng.uniform(0, 1, (M, 2))
    gi = rng.choice(M, L, replace=False)
    hj = rng.choice(N, L, replace=False)
    rng.shuffle(hj)
    G[gi] = H[hj] + rng.normal(0, cfg["sigma"], (L, 2))
    Xgt = np.zeros((M, N), dtype=np.uint8)
    Xgt[gi, hj] = 1
    DG = np.linalg.norm(G[:, None, :] - G[None, :, :], axis=2)
    DH = np.linalg.norm(H[:, None, :] - H[None, :, :], axis=2)
    # Draw a shared latent topology over planted vertices, then independently
    # sample edges incident to outliers. This is required for the planted
    # correspondence to be observable in graph structure before perturbation.
    UG = np.triu(rng.random((M, M)) < cfg["density"], 1)
    UH = np.triu(rng.random((N, N)) < cfg["density"], 1)
    common = np.triu(rng.random((L, L)) < cfg["density"], 1)
    common = common + common.T
    UG[np.ix_(gi, gi)] = common
    UH[np.ix_(hj, hj)] = common
    AG = ((UG + UG.T) * DG).astype(np.float32)
    AH = ((UH + UH.T) * DH).astype(np.float32)
    # Paper says add and remove 0.5*sigma*#Edge edges from each graph.
    AG = _perturb_edges(AG, DG, rng, cfg["sigma"])
    AH = _perturb_edges(AH, DH, rng, cfg["sigma"])
    return AG.astype(np.float32), AH.astype(np.float32), Xgt


def _wcs_objective(X, AG, AH):
    """Evaluate the paper's alpha=1 WCS objective on a discrete assignment."""
    X = np.asarray(X, dtype=np.float64)
    AG = np.asarray(AG, dtype=np.float64)
    AH = np.asarray(AH, dtype=np.float64)
    selected = X.sum(axis=1)
    residual = (selected[:, None] * selected[None, :]) * AG - X @ AH @ X.T
    return float(np.sum(residual * residual))


def _pairs_to_matrix(result, M, N):
    if hasattr(result, "matches"):
        pairs = result.matches
    elif isinstance(result, tuple):
        pairs = result[0]
    else:
        pairs = result
    X = np.zeros((M, N), dtype=np.uint8)
    for i, j in pairs:
        if 0 <= int(i) < M and 0 <= int(j) < N:
            X[int(i), int(j)] = 1
    return X


def _project_exact_l(scores, L):
    """Maximum-score exact-L partial assignment, without full-then-truncate.

    The dummy-node augmentation forces exactly L real-real assignments. The
    forbidden dummy-dummy block is essential; without it, the optimizer can
    route through dummy-dummy edges and violate cardinality.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite 2D array")
    M, N = scores.shape
    if not 0 <= L <= min(M, N):
        raise ValueError(f"invalid exact cardinality L={L} for shape {(M, N)}")
    size = M + N - L
    cost = np.full((size, size), np.inf, dtype=np.float64)
    cost[:M, :N] = -scores
    cost[:M, N:] = 0.0
    cost[M:, :N] = 0.0
    rows, cols = linear_sum_assignment(cost)
    real = (rows < M) & (cols < N)
    X = np.zeros((M, N), dtype=np.uint8)
    X[rows[real], cols[real]] = 1
    if int(X.sum()) != L:
        raise RuntimeError(f"exact-L projection returned {int(X.sum())}, expected {L}")
    return X


def _validate_assignment(X, L):
    X = np.asarray(X)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("assignment must be a finite 2D matrix")
    if not np.all((X == 0) | (X == 1)):
        raise ValueError("assignment is not binary")
    if int(X.sum()) != L or np.any(X.sum(axis=0) > 1) or np.any(X.sum(axis=1) > 1):
        raise ValueError("solver output is not an exact-L partial permutation")


def _mean_ci95(values, seed, n_boot=2000):
    """Deterministic percentile bootstrap CI for a sample mean."""
    x = np.asarray(values, dtype=np.float64)
    if not len(x):
        return [None, None]
    if len(x) == 1:
        return [float(x[0]), float(x[0])]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return [float(lo), float(hi)]


def _paired_bootstrap_mean_ci(values, seed, n_boot=2000):
    return _mean_ci95(values, seed=seed, n_boot=n_boot)


def _median_ci95(values, seed, n_boot=2000):
    """Deterministic percentile bootstrap CI for a sample median."""
    x = np.asarray(values, dtype=np.float64)
    if not len(x):
        return [None, None]
    if len(x) == 1:
        return [float(x[0]), float(x[0])]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    medians = np.median(x[idx], axis=1)
    lo, hi = np.quantile(medians, [0.025, 0.975])
    return [float(lo), float(hi)]


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _pygm_affinity(AG, AH):
    """Encode the discrete alpha=1 WCS objective as a Lawler score matrix.

    For an exact-L partial permutation, maximizing this matrix is equivalent to
    minimizing the WCS residual sum of squares: each selected pair-of-matches
    contributes -(A_G[i,k]-A_H[j,l])**2, up to a constant offset. The offset
    makes scores nonnegative for generic PyGMtools solvers and is constant over
    all exact-L feasible assignments.
    """
    M, N = AG.shape[0], AH.shape[0]
    source = np.repeat(np.arange(M), N)
    target = np.tile(np.arange(N), M)
    diff = AG[source[:, None], source[None, :]] - AH[target[:, None], target[None, :]]
    cost = diff * diff
    return (float(np.max(cost)) - cost).astype(np.float32)


def _grampa_scores(AG, AH, eta=.10):
    """GRAMPA spectral score (Fan et al., ICML 2020), rectangular extension.

    The published MATLAB code assumes equal graph sizes. For unequal sizes, use
    the same all-eigenpair formula with rectangular all-ones cross terms; final
    exact-L projection is handled by the shared assignment routine.
    """
    lam, U = np.linalg.eigh(AG.astype(np.float64))
    mu, V = np.linalg.eigh(AH.astype(np.float64))
    coeff = 1.0 / ((lam[:, None] - mu[None, :])**2 + eta**2)
    coeff *= U.T @ np.ones((AG.shape[0], AH.shape[0])) @ V
    return U @ coeff @ V.T


class Suite:
    def __init__(self):
        self.solvers = {
            "Fast-WCS v3": lambda seed=None: AdvancedGNCCP(alpha=1.0, max_iter=8, fw_max_iter=3, schedule="cosine", line_search="quartic", oracle="exact_augmented", local_refinement=True, seed=seed),
            "Fast-WCS v2": lambda seed=None: UniversalGNCCP(alpha=1.0, max_iter=8, fw_max_iter=3, schedule="cosine", line_search="closed_form", oracle="dummy_padded", local_refinement=True, seed=seed),
            "Local 2018-style RLX1F": lambda seed=None: OriginalGNCCP_2018(alpha=1.0, max_iter=10, fw_iter=6),
            "GRAMPA (ICML 2020)": lambda seed=None: None,
        }
        if pygm is not None:
            for label, method in [("PyGMtools SM (2024 toolkit)", pygm.sm), ("PyGMtools RRWM (2024 toolkit)", pygm.rrwm), ("PyGMtools IPFP (2024 toolkit)", pygm.ipfp)]:
                self.solvers[label] = (lambda seed=None, method=method: method)

    def solve(self, name, AG, AH, L, seed=None):
        M, N = AG.shape[0], AH.shape[0]
        solver = self.solvers[name](seed=seed)
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if name.startswith("GRAMPA"):
                X = _project_exact_l(_grampa_scores(AG, AH), L)
            elif name.startswith("PyGMtools"):
                K = _pygm_affinity(AG, AH)
                if seed is not None:
                    # PyGMtools' NumPy backend has no public per-call RNG argument.
                    # The runner is single-threaded, so seed its NumPy randomness
                    # immediately before the solver call and record that seed.
                    np.random.seed(int(seed) % (2**32 - 1))
                scores = solver(K, n1=M, n2=N, max_iter=50, backend="numpy")
                scores = np.asarray(scores).reshape(M, N)
                X = _project_exact_l(scores, L)
            else:
                res = solver.solve(AG, AH, L=L)
                X = _pairs_to_matrix(res, M, N)
        elapsed = (time.perf_counter() - t0) * 1000
        warning_records = [{"category": w.category.__name__, "message": str(w.message)} for w in caught]
        return X, elapsed, warning_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30, help="paired random instances per setting")
    ap.add_argument("--timing-runs", type=int, default=50, help="timed instances per setting (paper uses 50)")
    ap.add_argument("--warmups", type=int, default=1, help="untimed warm-up calls per method and setting")
    ap.add_argument("--sweeps", default="all", help="comma-separated noise,size,outliers,density or all")
    ap.add_argument("--methods", default="all", help="all or comma-separated exact names; default all available")
    ap.add_argument("--limit-settings", type=int, default=None, help="debug/smoke-run cap per sweep; omit for full protocol")
    ap.add_argument("--threads", type=int, default=1, help="BLAS/OpenMP threads per process for comparable timings")
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "paper_ready_wcs_results.json"))
    args = ap.parse_args()
    if args.runs < 1 or args.timing_runs < 1 or args.warmups < 0 or args.threads < 1:
        ap.error("runs and timing-runs must be positive; warmups must be nonnegative; threads must be positive")
    if args.limit_settings is not None and args.limit_settings < 1:
        ap.error("limit-settings must be positive")
    configs = _sweep_configs()
    requested = list(configs) if args.sweeps == "all" else args.sweeps.split(",")
    unknown = set(requested)-set(configs)
    if unknown: ap.error(f"unknown sweep(s): {sorted(unknown)}")
    suite = Suite()
    available = list(suite.solvers)
    methods = available if args.methods == "all" else args.methods.split(",")
    missing = set(methods)-set(available)
    if missing: ap.error(f"unavailable method(s): {sorted(missing)}; available: {available}")
    limiter = threadpool_limits(limits=args.threads) if threadpool_limits is not None else None
    if limiter is not None:
        limiter.__enter__()

    outpath = Path(args.out).resolve()
    if outpath.exists():
        ap.error(f"refusing to overwrite existing results: {outpath}; choose a new --out path")

    source_files = [ROOT / "benchmarks" / "benchmark_paper_synthetic_wcs.py",
                    ROOT / "gnccp_advanced_v3.py", ROOT / "gnccp_v2_classic.py",
                    ROOT / "gnccp_baseline_2018.py"]
    payload = {
        "protocol": {
            "run_mode": "smoke" if args.limit_settings is not None else "full_protocol",
            "limited_settings_per_sweep": args.limit_settings,
            "source": "Yang, Qiao & Liu, IEEE TNNLS 2018, Section III-A (synthetic WCS)",
            "generation": "2D coordinates iid U[0,1]; random planted L-partial permutation; matched G coordinates replaced by H plus iid N(0,sigma^2); Euclidean edge weights; a shared Erdős-Rényi topology is assigned to the planted vertex pairs and outlier-incident edges are sampled independently; independently add and remove round(0.5*sigma*edge_count) edges per graph, with every edge weighted by its endpoint Euclidean distance.",
            "accuracy": "correct planted pairs / L, percent; all methods receive identical seeded graph instances",
            "timing": "Separate from accuracy runs. One process; randomized method order per timing repetition; configured warm-ups excluded; wall-clock includes solver construction and adapter projection. BLAS/OpenMP thread count is fixed by --threads. Hardware-dependent; report medians and raw samples, not a universal speed claim.",
            "objective": "F(X)=||(X1)(X1)^T o A_G - X A_H X^T||_F^2, alpha=1; lower is better. Reported on every returned discrete exact-L assignment.",
            "deviations": ["The paper does not fully specify the PRNG, perturbation rounding, topology coupling, or all solver settings; these are fixed here and exposed in this output.", "A shared latent topology over planted pairs is an explicit reconstruction assumption; do not describe this as exact reproduction of the original code without author-code verification.", "The base-paper methods are not all available as original executable implementations. In-repository GNCCP variants and PyGMtools/GRAMPA adapters are labelled accordingly.", "Generic PyGMtools and GRAMPA scores are converted to an exact-L partial permutation using an augmented assignment problem; the adapter is timed and is not native WCS support.", "This is a synthetic WCS/PIW benchmark inspired by ECCV 2022's common-input, version-pinned, objective/runtime reporting. It is not a reproduction of ECCV's computer-vision datasets or containerized solver suite.", "A smoke run using --limit-settings is for code validation only and must not be reported as a full paper benchmark."],
            "accuracy_runs_per_setting": args.runs,
            "timing_runs_per_setting": args.timing_runs,
            "warmups_per_method_per_setting": args.warmups,
            "threads_per_process": args.threads,
            "seed": args.seed,
            "alpha": 1.0,
            "python": platform.python_version(), "platform": platform.platform(),
            "host": socket.gethostname(), "cpu": platform.processor(), "logical_cpu_count": os.cpu_count(),
            "threadpools": threadpool_info(),
            "numpy_version": np.__version__, "scipy_version": scipy.__version__,
            "pygmtools_version": getattr(pygm, "__version__", None),
            "pot_version": _version("POT"),
            "source_sha256": {str(p.relative_to(ROOT)): _sha256(p) for p in source_files if p.exists()},
        },
        "method_mapping": {
            "Fast-WCS v3": "Current proposed implementation, exact-L Burkard oracle, quartic search, exchange/replacement local refinement.",
            "Fast-WCS v2": "In-repository UniversalGNCCP fast classic.",
            "Local 2018-style RLX1F": "In-repository higher-order-gradient GNCCP with grid search and heuristic truncation; local reconstruction, not the original MATLAB code.",
            "PyGMtools SM (2024 toolkit)": "Spectral matching implementation exposed by PyGMtools; exact-L post-projection.",
            "PyGMtools RRWM (2024 toolkit)": "Reweighted random walks matching implementation exposed by PyGMtools; exact-L post-projection.",
            "PyGMtools IPFP (2024 toolkit)": "Integer projected fixed point implementation exposed by PyGMtools; exact-L post-projection.",
            "GRAMPA (ICML 2020)": "Published all-eigenpair spectral method for correlated weighted graph matching, extended to rectangular score matrix and exact-L assignment projection; eta fixed to 0.10.",
        },
        "solver_hyperparameters": {
            "Fast-WCS v3": {"alpha": 1.0, "max_iter": 8, "fw_max_iter": 3, "schedule": "cosine", "line_search": "quartic", "oracle": "exact_augmented", "local_refinement": True, "restarts": 1},
            "Fast-WCS v2": {"alpha": 1.0, "max_iter": 8, "fw_max_iter": 3, "schedule": "cosine", "line_search": "closed_form", "oracle": "dummy_padded", "local_refinement": True, "restarts": 1},
            "Local 2018-style RLX1F": {"alpha": 1.0, "max_iter": 10, "fw_iter": 6, "relaxation": "RLX1", "oracle": "fast_heuristic", "grid_points": 11},
            "GRAMPA (ICML 2020)": {"eta": 0.10, "projection": "augmented exact-L assignment"},
            "PyGMtools SM (2024 toolkit)": {"max_iter": 50, "backend": "numpy", "projection": "augmented exact-L assignment"},
            "PyGMtools RRWM (2024 toolkit)": {"max_iter": 50, "backend": "numpy", "projection": "augmented exact-L assignment"},
            "PyGMtools IPFP (2024 toolkit)": {"max_iter": 50, "backend": "numpy", "projection": "augmented exact-L assignment"},
        },
        "base_paper_method_availability": {
            "SM": "not a separately faithful run: available only as PyGMtools SM implementation",
            "GA": "not implemented in this workspace; must add a verified source or faithful reproduction",
            "PGM": "not implemented in this workspace; must add a verified source or faithful reproduction",
            "EPF": "not implemented in this workspace; must add a verified source or faithful reproduction",
            "RLX1/RLX2": "not present as original paper implementations; the local solver is an RLX1F-style reconstruction",
            "RLX1F/RLX2F": "only local 2018-style RLX1F reconstruction is runnable; RLX2F is unavailable",
        },
        "available_methods": available,
        "selected_methods": methods,
        "omitted_optional_methods": [m for m in ["PyGMtools SM (2024 toolkit)", "PyGMtools RRWM (2024 toolkit)", "PyGMtools IPFP (2024 toolkit)"] if m not in available],
        "results": {},
    }
    for sweep in requested:
        payload["results"][sweep] = []
        sweep_configs = configs[sweep]
        if args.limit_settings is not None:
            sweep_configs = sweep_configs[:args.limit_settings]
        for ci, cfg in enumerate(sweep_configs):
            chance_pct = 100.0 * cfg["L"] / (cfg["M"] * cfg["N"])
            row = {**cfg, "uniform_random_exact_L_chance_accuracy_pct": chance_pct,
                   "methods": {name: {"accuracy_pct": [], "accuracy_seeds": [], "objective": [],
                                      "runtime_ms": [], "timing_seeds": [], "failures": [], "warnings": []} for name in methods}}
            print(f"{sweep} {ci+1}/{len(sweep_configs)} {cfg}", flush=True)
            # Accuracy/objective phase: paired input instances across methods.
            for run in range(args.runs):
                seed = args.seed + sum(ord(c) for c in sweep)*100000 + ci*1000 + run
                AG, AH, Xgt = make_instance(cfg, seed)
                order_rng = np.random.default_rng(seed ^ 0xA5A5A5A5)
                order = list(methods)
                order_rng.shuffle(order)
                for name in order:
                    try:
                        X, _, emitted = suite.solve(name, AG, AH, cfg["L"], seed=seed)
                        _validate_assignment(X, cfg["L"])
                        correct = int(np.sum(X * Xgt))
                        objective = _wcs_objective(X, AG, AH)
                        vals = row["methods"][name]
                        vals["warnings"].extend({"phase": "accuracy", "seed": int(seed), **w} for w in emitted)
                        vals["accuracy_pct"].append(100.0*correct/cfg["L"])
                        vals["accuracy_seeds"].append(int(seed))
                        vals["objective"].append(objective)
                    except Exception as exc:
                        row["methods"][name]["failures"].append({"phase": "accuracy", "seed": int(seed),
                            "error": f"{type(exc).__name__}: {exc}"})

            # Warm-ups use a dedicated deterministic instance and are excluded
            # from timing samples. A failure is recorded, never silently hidden.
            for w in range(args.warmups):
                seed = args.seed + sum(ord(c) for c in sweep)*100000 + ci*1000 + 500000 + w
                AG, AH, _ = make_instance(cfg, seed)
                for name in methods:
                    try:
                        X, _, emitted = suite.solve(name, AG, AH, cfg["L"], seed=seed)
                        _validate_assignment(X, cfg["L"])
                        row["methods"][name]["warnings"].extend({"phase": "warmup", "seed": int(seed), **w} for w in emitted)
                    except Exception as exc:
                        row["methods"][name]["failures"].append({"phase": "warmup", "seed": int(seed),
                            "error": f"{type(exc).__name__}: {exc}"})

            # Independent timing seeds; randomize solver order within each
            # repetition to reduce systematic order/cache bias.
            for run in range(args.timing_runs):
                seed = args.seed + sum(ord(c) for c in sweep)*100000 + ci*1000 + 600000 + run
                AG, AH, _ = make_instance(cfg, seed)
                order_rng = np.random.default_rng(seed ^ 0x5A5A5A5A)
                order = list(methods)
                order_rng.shuffle(order)
                for name in order:
                    try:
                        X, elapsed, emitted = suite.solve(name, AG, AH, cfg["L"], seed=seed)
                        _validate_assignment(X, cfg["L"])
                        vals = row["methods"][name]
                        vals["warnings"].extend({"phase": "timing", "seed": int(seed), **w} for w in emitted)
                        vals["runtime_ms"].append(float(elapsed))
                        vals["timing_seeds"].append(int(seed))
                    except Exception as exc:
                        row["methods"][name]["failures"].append({"phase": "timing", "seed": int(seed),
                            "error": f"{type(exc).__name__}: {exc}"})

            for name, vals in row["methods"].items():
                accuracies, runtimes = vals["accuracy_pct"], vals["runtime_ms"]
                vals["accuracy_mean_pct"] = float(np.mean(accuracies)) if accuracies else None
                vals["accuracy_sd_pct"] = float(np.std(accuracies, ddof=1)) if len(accuracies)>1 else 0.0 if accuracies else None
                vals["accuracy_mean_ci95_pct"] = _mean_ci95(accuracies, args.seed + ci + sum(ord(c) for c in sweep))
                objectives = vals["objective"]
                vals["objective_mean"] = float(np.mean(objectives)) if objectives else None
                vals["objective_sd"] = float(np.std(objectives, ddof=1)) if len(objectives)>1 else 0.0 if objectives else None
                vals["runtime_median_ms"] = float(np.median(runtimes)) if runtimes else None
                vals["runtime_mean_ms"] = float(np.mean(runtimes)) if runtimes else None
                vals["runtime_median_ci95_ms"] = _median_ci95(runtimes, args.seed + ci + sum(ord(c) for c in sweep) + 17)
                vals["n_accuracy_success"] = len(accuracies)
                vals["n_timing_success"] = len(runtimes)
                vals["n_failures"] = len(vals["failures"])
                vals["n_warnings"] = len(vals["warnings"])
                vals["success_rate_pct"] = 100.0 * len(accuracies) / args.runs

            # Paired per-instance accuracy differences against v3 are useful
            # comparisons because every method saw the same generated seeds.
            if "Fast-WCS v3" in row["methods"]:
                v3 = row["methods"]["Fast-WCS v3"]
                v3_by_seed = dict(zip(v3["accuracy_seeds"], v3["accuracy_pct"]))
                row["paired_accuracy_delta_vs_v3_pp"] = {}
                for name, vals in row["methods"].items():
                    common = sorted(set(v3_by_seed).intersection(vals["accuracy_seeds"]))
                    by_seed = dict(zip(vals["accuracy_seeds"], vals["accuracy_pct"]))
                    diffs = [by_seed[s] - v3_by_seed[s] for s in common]
                    row["paired_accuracy_delta_vs_v3_pp"][name] = {
                        "n_paired": len(common), "mean_delta_pp": float(np.mean(diffs)) if diffs else None,
                        "mean_delta_ci95_pp": _paired_bootstrap_mean_ci(diffs, args.seed + ci + sum(ord(c) for c in name)),
                    }
            payload["results"][sweep].append(row)
            outpath.parent.mkdir(parents=True, exist_ok=True)
            outpath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # A compact CSV is convenient for plots and tables.
    csvpath = outpath.with_suffix(".csv")
    with csvpath.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sweep", "M", "N", "L", "sigma", "density", "chance_accuracy_pct", "method", "accuracy_mean_pct", "accuracy_sd_pct", "accuracy_ci95_low_pct", "accuracy_ci95_high_pct", "objective_mean", "objective_sd", "runtime_median_ms", "runtime_ci95_low_ms", "runtime_ci95_high_ms", "n_accuracy_success", "n_timing_success", "n_failures", "n_warnings", "success_rate_pct"])
        for sweep, rows in payload["results"].items():
            for row in rows:
                for name, vals in row["methods"].items():
                    aci = vals.get("accuracy_mean_ci95_pct", [None, None])
                    rci = vals.get("runtime_median_ci95_ms", [None, None])
                    w.writerow([sweep,row['M'],row['N'],row['L'],row['sigma'],row['density'],row['uniform_random_exact_L_chance_accuracy_pct'],name,vals.get('accuracy_mean_pct'),vals.get('accuracy_sd_pct'),aci[0],aci[1],vals.get('objective_mean'),vals.get('objective_sd'),vals.get('runtime_median_ms'),rci[0],rci[1],vals.get('n_accuracy_success'),vals.get('n_timing_success'),vals.get('n_failures'),vals.get('n_warnings'),vals.get('success_rate_pct')])
    print(f"Wrote {outpath}\nWrote {csvpath}")
    if args.limit_settings is not None:
        print("SMOKE RUN: limited settings; do not use these values as paper results.")
    if limiter is not None:
        limiter.__exit__(None, None, None)


if __name__ == "__main__":
    main()
