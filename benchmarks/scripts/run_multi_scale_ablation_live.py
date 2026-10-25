"""Script 4 - live multi-scale component ablation (one contribution switched on per stage).

Every stage runs the SAME homotopy (linear zeta schedule, 10 levels x 6 Frank-Wolfe steps, no early
stopping, float32, structure-only objective alpha = 1) on the SAME instances, so differences are due to
the switched component only:

  S0  Baseline        Yang et al. (2018) RLX1F: Eq. (11) relaxation gradient, U = X 1_{NxN} X^T formed
                      explicitly (Eq. 3), Hungarian + top-L oracle (Sec. II-D), 11-point grid line search
                      with full objective evaluations (Eq. 8), Hungarian + top-L discretisation.
  S1  + Lemma 1       U = r r^T (rank-1, r = X 1_N); everything else as S0.
  S2  + Theorem 1     exact closed-form gradient of the true WCS objective (replaces Eq. 11).
  S3  + Theorem 3     exact quartic line search (Cardano roots, GEMM-free coefficients) replaces the grid.
  S4  + Lemma 2       Burkard (M+N-L)^2 augmented assignment oracle and exact-L discretisation.
  S5  + Prop. 2       vectorised 2-opt refinement (replacement + exchange moves) on the final vertex.
  REF Fast-WCS v3     library defaults (cosine schedule 8 x 3, duality-gap stopping, all components).

S1 is checked against gnccp_baseline_2018.OriginalGNCCP_2018 (RLX1F) for identical matchings.
Regimes: Small N = 30, Medium N = 60, Large N = 100 (M = N, L = 0.8 N, sigma = .03, rho = .5).

Also: (a) per-call latency of every line-search call (grid vs Cardano), (b) objective before/after
2-opt, (c) gradient micro-benchmark (Theorem 1 vs Eq. 11 vs central finite differences) for
N in {10, 20, 30, 50, 100}, with the finite-difference error of Theorem 1.

Outputs: csv/ablation_results.csv, csv/ablation_trials.csv, csv/ablation_linesearch_calls.csv,
         csv/gradient_microbenchmark.csv, json/ablation_results.json, <ts>_ablation/ run dir.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_common as lc  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from gnccp_advanced_v3 import AdvancedGNCCP  # noqa: E402
from gnccp_baseline_2018 import OriginalGNCCP_2018  # noqa: E402

SEED_BASE = 40_261_016
REGIMES = {"Small (N=30)": 30, "Medium (N=60)": 60, "Large (N=100)": 100}
STAGES = [
    ("S0 Baseline (Yang RLX1F)", dict(naive_U=True, grad="rlx1", ls="grid", oracle="topL", twoopt=False)),
    ("S1 + Lemma 1 (rank-1 U)", dict(naive_U=False, grad="rlx1", ls="grid", oracle="topL", twoopt=False)),
    ("S2 + Theorem 1 (exact gradient)", dict(naive_U=False, grad="exact", ls="grid", oracle="topL", twoopt=False)),
    ("S3 + Theorem 3 (Cardano line search)", dict(naive_U=False, grad="exact", ls="cardano", oracle="topL", twoopt=False)),
    ("S4 + Lemma 2 (Burkard oracle)", dict(naive_U=False, grad="exact", ls="cardano", oracle="burkard", twoopt=False)),
    ("S5 + Prop. 2 (2-opt refinement)", dict(naive_U=False, grad="exact", ls="cardano", oracle="burkard", twoopt=True)),
]
REF = "REF Fast-WCS v3 (library defaults)"
LEVELS, FW = 10, 6


class AblationGNCCP:
    def __init__(self, naive_U, grad, ls, oracle, twoopt):
        self.naive_U, self.grad_kind, self.ls, self.oracle, self.twoopt = naive_U, grad, ls, oracle, twoopt
        self.eng = AdvancedGNCCP(alpha=1.0, normalization="none", dtype=np.float32)
        self.ls_ms = []

    # --- objective -------------------------------------------------------------------
    def _U(self, X, ones_NN):
        if self.naive_U:
            return X @ ones_NN @ X.T
        r = X.sum(axis=1)
        return r[:, None] * r[None, :]

    def h0(self, X, AG, AH, ones_NN):
        D = self._U(X, ones_NN) * AG - X @ AH @ X.T
        return float(np.sum(D * D))

    def J(self, X, zeta, AG, AH, ones_NN):
        w = (1.0 - zeta) if zeta >= 0 else (1.0 + zeta)
        return w * self.h0(X, AG, AH, ones_NN) + zeta * float(np.sum(X * X))

    # --- gradient --------------------------------------------------------------------
    def grad(self, X, zeta, AG, AH, ones_NN, C0):
        if self.grad_kind == "exact":
            return self.eng._grad_Jzeta_with_cache(X, AG, AH, C0, zeta)
        # Eq. (11) of Yang et al.; X 1_{NxN} formed explicitly (S0) or as r 1^T (Lemma 1)
        X1 = X @ ones_NN if self.naive_U else np.repeat(X.sum(axis=1)[:, None], X.shape[1], axis=1)
        AG_sq = (AG.T * AG.T) + (AG * AG)
        XtX = X.T @ X
        gH = AG_sq @ X1 - 2.0 * (AG.T @ X @ AH + AG @ X @ AH.T) + 2.0 * (X @ AH @ XtX @ AH.T + X @ AH.T @ XtX @ AH)
        w = (1.0 - zeta) if zeta >= 0 else (1.0 + zeta)
        return w * gH + 2.0 * zeta * X, None

    # --- oracle ------------------------------------------------------------------------
    def lmo(self, g, L, M, N):
        if self.oracle == "burkard":
            return self.eng._augmented_assignment(g, L, M, N)
        ri, ci = linear_sum_assignment(g)
        k = np.argsort(g[ri, ci])[:L]
        return ri[k], ci[k]

    def discretize(self, X, L, M, N):
        if self.oracle == "burkard":
            ri, ci = self.eng._augmented_assignment(-X, L, M, N)
        else:
            ri, ci = linear_sum_assignment(-X)
            k = np.argsort(-X[ri, ci])[:L]
            ri, ci = ri[k], ci[k]
        Xb = np.zeros((M, N), np.float32)
        Xb[ri, ci] = 1.0
        return Xb

    def solve(self, AG, AH, L):
        t0 = time.perf_counter()
        AG = np.asarray(AG, np.float32); AH = np.asarray(AH, np.float32)
        M, N = AG.shape[0], AH.shape[0]
        C0 = np.zeros((M, N), np.float32)
        ones_NN = np.ones((N, N), np.float32)
        X = np.full((M, N), L / (M * N), np.float32)
        for zeta in np.linspace(1.0, -1.0, LEVELS):
            zeta = float(zeta)
            for _ in range(FW):
                g, cache = self.grad(X, zeta, AG, AH, ones_NN, C0)
                ri, ci = self.lmo(g, L, M, N)
                Y = np.zeros((M, N), np.float32); Y[ri, ci] = 1.0
                t1 = time.perf_counter()
                if self.ls == "cardano":
                    lam = self.eng._quartic_step_gemm_free(X, cache, ri, ci, AG, AH, C0, zeta)
                    X = (1.0 - lam) * X + lam * Y
                else:
                    D = Y - X
                    best, best_lam = np.inf, 0.0
                    for lam in np.linspace(0.0, 1.0, 11):
                        c = self.J(np.clip(X + lam * D, 0.0, 1.0), zeta, AG, AH, ones_NN)
                        if c < best:
                            best, best_lam = c, lam
                    X = np.clip(X + best_lam * D, 0.0, 1.0)
                self.ls_ms.append((time.perf_counter() - t1) * 1e3)
        Xb = self.discretize(X, L, M, N)
        F_pre = None
        t_2opt = 0.0
        if self.twoopt:
            F_pre = lc.wcs_objective(Xb, AG, AH)
            t2 = time.perf_counter()
            Xb = self.eng._swap_refinement(Xb, AG, AH, C0, L, max_rounds=self.eng.ls_max_rounds)
            t_2opt = (time.perf_counter() - t2) * 1e3
        return Xb.astype(np.uint8), (time.perf_counter() - t0) * 1e3, F_pre, t_2opt


def trial(task):
    regime, N, t = task
    L = int(round(0.8 * N))
    seed = SEED_BASE + N * 1000 + t
    AG, AH, Xgt = lc.make_synthetic_instance(N, N, L, 0.03, 0.5, seed)
    F_gt = lc.wcs_objective(Xgt, AG, AH)
    rows, calls = [], []
    if not lc._WARMED:
        AblationGNCCP(**STAGES[-1][1]).solve(*lc.make_synthetic_instance(8, 8, 6, .03, .5, 1)[:2], 6)
        lc._WARMED.add("ablation")
    for name, cfg in STAGES:
        eng = AblationGNCCP(**cfg)
        X, ms, F_pre, t2 = eng.solve(AG, AH, L)
        lc.validate_exact_L(X, L)
        F = lc.wcs_objective(X, AG, AH)
        rows.append(dict(regime=regime, N=N, L=L, trial=t, seed=seed, stage=name, ms=ms, F=F, F_gt=F_gt,
                         acc=lc.accuracy_pct(X, Xgt, L), F_pre_2opt=F_pre, ms_2opt=t2 if cfg["twoopt"] else None,
                         ls_calls=len(eng.ls_ms), ls_ms_total=float(np.sum(eng.ls_ms)), ls_ms_median=float(np.median(eng.ls_ms))))
        if cfg is STAGES[1][1] or cfg is STAGES[2][1] or cfg is STAGES[3][1]:
            kind = "11-point grid (full objective evaluations)" if cfg["ls"] == "grid" else "Cardano quartic (Theorem 3)"
            calls += [dict(regime=regime, N=N, trial=t, stage=name, line_search=kind, call=k, ms=v)
                      for k, v in enumerate(eng.ls_ms)]
        if name.startswith(("S0", "S1")):
            if name.startswith("S0"):
                ref = OriginalGNCCP_2018(alpha=1.0, max_iter=LEVELS, fw_iter=FW, relaxation="RLX1", oracle="fast_heuristic").solve(AG, AH, L)
                Xr = np.zeros_like(X); Xr[tuple(np.array(ref.matches).T)] = 1
            rows[-1]["identical_to_OriginalGNCCP_2018"] = int(np.array_equal(Xr, X))
    X, ms, err = lc.timed_call(lc.solve_fastwcs, AG, AH, L)
    F = lc.wcs_objective(X, AG, AH)
    rows.append(dict(regime=regime, N=N, L=L, trial=t, seed=seed, stage=REF, ms=ms, F=F, F_gt=F_gt, acc=lc.accuracy_pct(X, Xgt, L)))
    return rows, calls


def gradient_microbenchmark(sizes=(10, 20, 30, 50, 100), reps=30):
    """Wall-clock of one gradient evaluation: Theorem 1, Eq. (11) (RLX1) and central finite differences (FD)."""
    out = []
    for N in sizes:
        L = int(round(0.8 * N))
        AG, AH, Xgt = lc.make_synthetic_instance(N, N, L, 0.03, 0.5, SEED_BASE + 7 * N)
        rng = np.random.default_rng(N)
        X64 = 0.5 * Xgt + 0.5 * rng.dirichlet(np.ones(N * N)).reshape(N, N) * L
        X64 = np.clip(X64, 0, 1)
        eng64 = AdvancedGNCCP(alpha=1.0, dtype=np.float64)
        C64 = np.zeros((N, N))
        A64, B64 = AG.astype(np.float64), AH.astype(np.float64)
        X32, A32, B32, C32 = (z.astype(np.float32) for z in (X64, A64, B64, C64))
        eng32 = AdvancedGNCCP(alpha=1.0, dtype=np.float32)
        yang = OriginalGNCCP_2018(alpha=1.0)
        ones = np.ones((N, N), np.float32)

        def tmin(f, r):
            best = np.inf
            for _ in range(r):
                t0 = time.perf_counter(); f(); best = min(best, time.perf_counter() - t0)
            return best * 1e3

        t_thm1 = tmin(lambda: eng32._compute_gradient_F(X32, A32, B32, C32), reps)
        t_rlx1 = tmin(lambda: yang._gradient_h1(X32, A32, B32, ones), reps)
        g = eng64._compute_gradient_F(X64, A64, B64, C64)
        eps = 1e-6
        t0 = time.perf_counter()
        fd = np.zeros_like(X64)
        for i in range(N):
            for j in range(N):
                Xp = X64.copy(); Xp[i, j] += eps
                Xm = X64.copy(); Xm[i, j] -= eps
                fd[i, j] = (eng64._compute_F(Xp, A64, B64, C64) - eng64._compute_F(Xm, A64, B64, C64)) / (2 * eps)
        t_fd = (time.perf_counter() - t0) * 1e3
        rel = float(np.max(np.abs(g - fd)) / max(np.max(np.abs(fd)), 1e-12))
        out.append(dict(N=N, M=N, L=L, ms_theorem1=t_thm1, ms_rlx1_eq11=t_rlx1, ms_central_fd=t_fd,
                        fd_objective_evaluations=2 * N * N, speedup_vs_fd=t_fd / t_thm1, speedup_vs_rlx1=t_rlx1 / t_thm1,
                        theorem1_vs_fd_max_rel_err=rel))
        print(f"  grad N={N:3d}: Thm1 {t_thm1:.3f} ms | Eq.11 {t_rlx1:.3f} ms | FD {t_fd:.1f} ms | rel.err {rel:.2e}", flush=True)
    return out


def aggregate(rows):
    out = []
    for regime, N in REGIMES.items():
        rr = [r for r in rows if r["N"] == N]
        base = {r["trial"]: r for r in rr if r["stage"] == STAGES[0][0]}
        prev_med = None
        for name in [s for s, _ in STAGES] + [REF]:
            rs = [r for r in rr if r["stage"] == name]
            ms = np.array([r["ms"] for r in rs])
            sp = np.array([base[r["trial"]]["ms"] / r["ms"] for r in rs])
            red = np.array([100.0 * (base[r["trial"]]["F"] - r["F"]) / max(base[r["trial"]]["F"], 1e-12) for r in rs])
            acc = [r["acc"] for r in rs]
            lo, hi = lc.mean_ci95(acc, seed=N)
            d = dict(regime=regime, N=N, L=rs[0]["L"], stage=name, n_trials=len(rs), ms_median=float(np.median(ms)),
                     ms_q25=float(np.percentile(ms, 25)), ms_q75=float(np.percentile(ms, 75)),
                     cumulative_speedup_median=float(np.median(sp)), stage_speedup_vs_previous=None,
                     objective_reduction_vs_S0_pct_median=float(np.median(red)), objective_reduction_vs_S0_pct_mean=float(np.mean(red)),
                     F_median=float(np.median([r["F"] for r in rs])), F_ratio_to_gt_median=float(np.median([r["F"] / r["F_gt"] for r in rs])),
                     acc_mean=float(np.mean(acc)), acc_ci95_lo=lo, acc_ci95_hi=hi,
                     ls_ms_median_per_call=float(np.median([r["ls_ms_median"] for r in rs])) if "ls_ms_median" in rs[0] else None,
                     ls_share_of_runtime_pct=float(np.median([100 * r["ls_ms_total"] / r["ms"] for r in rs])) if "ls_ms_total" in rs[0] else None)
            if prev_med is not None and name != REF:
                d["stage_speedup_vs_previous"] = prev_med / d["ms_median"]
            if name.startswith(("S0", "S1")):
                d["identical_to_OriginalGNCCP_2018_rate"] = float(np.mean([r["identical_to_OriginalGNCCP_2018"] for r in rs]))
            if name.startswith("S5"):
                pre = np.array([r["F_pre_2opt"] for r in rs]); post = np.array([r["F"] for r in rs])
                d["twoopt_objective_drop_pct_median"] = float(np.median(100 * (pre - post) / np.maximum(pre, 1e-12)))
                d["twoopt_improved_rate"] = float(np.mean(post < pre - 1e-9))
                d["twoopt_ms_median"] = float(np.median([r["ms_2opt"] for r in rs]))
            prev_med = d["ms_median"] if name != REF else prev_med
            out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    name = "ablation" + a.tag
    out_dir = lc.run_dir(name)
    sys.stdout = lc.Tee(out_dir / "run.log")
    print(f"Script 4 | trials={a.trials} workers={a.workers} regimes={REGIMES} levels x FW = {LEVELS} x {FW}", flush=True)
    tasks = [(rg, N, t) for rg, N in REGIMES.items() for t in range(a.trials)]
    tasks.sort(key=lambda z: -z[1])
    res = lc.run_pool(trial, tasks, workers=a.workers, desc=name, log_every=10)
    rows = [r for rs, _ in res for r in rs]
    calls = [c for _, cs in res for c in cs]
    agg = aggregate(rows)
    for d in agg:
        print(f"{d['regime']:14s} {d['stage']:38s} ms {d['ms_median']:9.2f}  x{d['cumulative_speedup_median']:7.2f}  "
              f"dF {d['objective_reduction_vs_S0_pct_median']:+7.2f}%  acc {d['acc_mean']:5.1f}", flush=True)
    print("gradient micro-benchmark (single process, min over repetitions)", flush=True)
    grad = gradient_microbenchmark()
    lc.write_csv(out_dir / "trials.csv", rows)
    lc.write_csv(out_dir / "linesearch_calls.csv", calls)
    for nm, data in (("ablation_results", agg), ("ablation_trials", rows), ("ablation_linesearch_calls", calls),
                     ("gradient_microbenchmark", grad)):
        lc.write_csv(lc.LIVE / "csv" / f"{nm}{a.tag}.csv", data)
    payload = {"script": Path(__file__).name, "args": vars(a), "regimes": REGIMES, "stages": STAGES, "levels": LEVELS,
               "fw_steps": FW, "seed_base": SEED_BASE, "aggregated": agg, "gradient_microbenchmark": grad,
               "provenance": lc.provenance([Path(__file__)])}
    lc.write_json(out_dir / f"{name}_results.json", payload)
    lc.write_json(lc.LIVE / "json" / f"{name}_results.json", payload)
    print("wrote", lc.LIVE / "csv" / f"ablation_results{a.tag}.csv", flush=True)


if __name__ == "__main__":
    main()
