"""
Comprehensive Multi-Topology Benchmark:
Compares Fast-WCS against baseline solvers across three distinct graph topologies:
  1. Erdős–Rényi (shared_er) — baseline classical random graphs.
  2. Random Regular (random_regular) — uniform degree d on every vertex (automorphism stress test).
  3. Barabási–Albert (barabasi_albert) — scale-free networks via preferential attachment (hub networks).

Evaluates across:
  - Noise sweep: sigma in {0.0, 0.02, 0.04, 0.06, 0.08, 0.10} (M=30, N=35, L=25)
  - Outliers sweep: M-L in {0, 2, 4, 6, 8, 10} (M=30, N=35, sigma=0.03)
  - Size sweep: M in {20, 30, 40, 50, 60} (N=M+5, L=M-5, sigma=0.03)

Outputs:
  - benchmarks/live_runs/csv/topologies_trials.csv
  - benchmarks/live_runs/csv/topologies_results.csv
  - benchmarks/live_runs/json/topologies_results.json
"""
from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path
import multiprocessing as mp

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT))

import live_common as lc
from gnccp_advanced_v3 import AdvancedGNCCP
from gnccp_advanced_v4_wloa import AdvancedGNCCP_v4
from run_synthetic_replication_live import solver_table, load_configs

SEED_BASE = 20_261_020
N_TRIALS = 20  # Monte Carlo trials per setting

TOPOLOGIES = ["shared_er", "random_regular", "barabasi_albert"]

SWEEPS = {
    "A_noise": [dict(M=30, N=35, L=25, sigma=s, density=0.5, x=s, x_name="sigma")
                for s in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10)],
    "B_outliers": [dict(M=30, N=35, L=30 - o, sigma=0.03, density=0.5, x=o, x_name="M-L")
                   for o in (0, 2, 4, 6, 8, 10)],
    "C_size": [dict(M=m, N=m + 5, L=m - 5, sigma=0.03, density=0.5, x=m, x_name="M")
               for m in (20, 30, 40, 50, 60)],
}


def eval_one_instance(task):
    topo, scen, si, cfg, t = task
    M, N, L = cfg["M"], cfg["N"], cfg["L"]
    sigma, density = cfg["sigma"], cfg["density"]
    seed = SEED_BASE + (TOPOLOGIES.index(topo) * 100_000) + (list(SWEEPS).index(scen) * 10_000) + (si * 100) + t
    
    AG, AH, Xgt = lc.make_synthetic_instance(M, N, L, sigma, density, seed, topo)
    F_gt = lc.wcs_objective(Xgt, AG, AH)
    
    cal, fcal = load_configs()
    base_table = solver_table(cal, fcal)
    
    # Selected competitors to benchmark
    active_solvers = {
        "Fast-WCS (Multi-Start, 10 paths)": None,  # handled via custom multistart
        "Fast-WCS (Single-Start)": lambda: lc.solve_fastwcs(AG, AH, L),
        "Fast-WCS v4 (Tree-History)": lambda: AdvancedGNCCP_v4(alpha=1.0).solve(AG, AH, L=L).X_binary,
        "RLX1F (Yang 2018)": lambda: lc.solve_yang(AG, AH, L, relaxation="RLX1", oracle="fast_heuristic"),
        "FAQ": lambda: lc.solve_faq(AG, AH, L),
        "SM": lambda: lc.solve_pygm(AG, AH, L, method="sm", affinity=cal["sm"]["affinity"], sigma=cal["sm"]["sigma"]),
        "IPFP": lambda: lc.solve_pygm(AG, AH, L, method="ipfp", affinity=cal["ipfp"]["affinity"], sigma=cal["ipfp"]["sigma"]),
        "RRWM": lambda: lc.solve_pygm(AG, AH, L, method="rrwm", affinity=cal["rrwm"]["affinity"], sigma=cal["rrwm"]["sigma"]),
    }
    
    rows = []
    
    # 1. Evaluate Multi-Start
    t0 = time.perf_counter()
    best_F = float('inf')
    best_X = None
    err = None
    try:
        for k in range(10):
            s_seed = (seed + k * 10000) % (2**31 - 1)
            sk = AdvancedGNCCP(alpha=1.0, seed=s_seed)
            if k == 0:
                X0 = None
            else:
                rng = np.random.default_rng(s_seed)
                noise = rng.uniform(-0.03, 0.03, (M, N)).astype(np.float32)
                X0 = np.clip(np.ones((M, N), dtype=np.float32) * (float(L) / float(M * N)) + noise, 0, 1)
            rk = sk.solve(AG, AH, L=L, X_init=X0)
            Fk = lc.wcs_objective(rk.X_binary, AG, AH)
            if Fk < best_F:
                best_F = Fk
                best_X = rk.X_binary
        ms_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as e:
        ms_ms = (time.perf_counter() - t0) * 1000.0
        err = str(e)
        
    row_ms = dict(topology=topo, scenario=scen, setting=si, x=cfg["x"], x_name=cfg["x_name"],
                  M=M, N=N, L=L, sigma=sigma, density=density, trial=t, seed=seed,
                  method="Fast-WCS (Multi-Start, 10 paths)", ms=ms_ms, F_gt=F_gt, error=err)
    if err is None and best_X is not None:
        try:
            lc.validate_exact_L(best_X, L)
            F = lc.wcs_objective(best_X, AG, AH)
            row_ms.update(acc=lc.accuracy_pct(best_X, Xgt, L), F=F,
                          F_ratio=F / max(F_gt, 1e-12), F_le_gt=int(F <= F_gt + 1e-9))
        except ValueError as e:
            row_ms["error"] = f"invalid output: {e}"
    rows.append(row_ms)
    
    # 2. Evaluate other solvers
    for m, func in active_solvers.items():
        if m == "Fast-WCS (Multi-Start, 10 paths)":
            continue
        X, ms, err = lc.timed_call(func)
        r = dict(topology=topo, scenario=scen, setting=si, x=cfg["x"], x_name=cfg["x_name"],
                 M=M, N=N, L=L, sigma=sigma, density=density, trial=t, seed=seed,
                 method=m, ms=ms, F_gt=F_gt, error=err)
        if err is None and X is not None:
            try:
                lc.validate_exact_L(X, L)
                F = lc.wcs_objective(X, AG, AH)
                r.update(acc=lc.accuracy_pct(X, Xgt, L), F=F,
                         F_ratio=F / max(F_gt, 1e-12), F_le_gt=int(F <= F_gt + 1e-9))
            except ValueError as e:
                r["error"] = f"invalid output: {e}"
        rows.append(r)
        
    return rows


def aggregate_topologies(rows):
    out = []
    keys = sorted({(r["topology"], r["scenario"], r["setting"], r["method"]) for r in rows})
    for topo, scen, si, m in keys:
        rs = [r for r in rows if r["topology"] == topo and r["scenario"] == scen and r["setting"] == si and r["method"] == m]
        ok = [r for r in rs if r.get("error") is None or pd.isna(r.get("error")) or str(r.get("error")).strip() in ("", "nan", "None")]
        acc = [float(r["acc"]) for r in ok if r.get("acc") is not None and not pd.isna(r.get("acc"))]
        lo, hi = lc.mean_ci95(acc, seed=si) if acc else (0.0, 0.0)
        s_acc, s_ms = lc.summarize(acc), lc.summarize([float(r["ms"]) for r in ok])
        r0 = rs[0]
        out.append(dict(topology=topo, scenario=scen, x_name=r0["x_name"], x=r0["x"],
                        M=r0["M"], N=r0["N"], L=r0["L"], sigma=r0["sigma"], density=r0["density"],
                        method=m, n_trials=len(rs), n_ok=len(ok), n_errors=len(rs) - len(ok),
                        acc_mean=s_acc["mean"], acc_sd=s_acc["sd"], acc_ci95_lo=lo, acc_ci95_hi=hi,
                        F_ratio_median=lc.summarize([float(r["F_ratio"]) for r in ok])["median"] if ok else None,
                        frac_F_le_gt=float(np.mean([float(r["F_le_gt"]) for r in ok])) if ok else None,
                        ms_median=s_ms["median"], ms_q25=s_ms["q25"], ms_q75=s_ms["q75"], ms_mean=s_ms["mean"]))
    return out


def main():
    tasks = []
    for topo in TOPOLOGIES:
        for scen, settings in SWEEPS.items():
            for si, cfg in enumerate(settings):
                for t in range(N_TRIALS):
                    tasks.append((topo, scen, si, cfg, t))
                    
    print(f"=== Multi-Topology Synthetic Benchmark ===", flush=True)
    print(f"Topologies: {TOPOLOGIES}", flush=True)
    print(f"Total instances: {len(tasks)} (20 trials/setting across {len(tasks)//N_TRIALS} settings)", flush=True)
    print(f"Launching parallel execution using 8 workers...", flush=True)
    
    t_start = time.perf_counter()
    with mp.Pool(processes=8) as pool:
        results_nested = pool.map(eval_one_instance, tasks)
        
    all_rows = [r for sublist in results_nested for r in sublist]
    elapsed = time.perf_counter() - t_start
    print(f"Execution completed in {elapsed:.2f} s ({elapsed/60.0:.2f} min). Total solver evaluations: {len(all_rows)}", flush=True)
    
    agg = aggregate_topologies(all_rows)
    
    # Save CSVs
    trials_csv = lc.LIVE / "csv" / "topologies_trials.csv"
    results_csv = lc.LIVE / "csv" / "topologies_results.csv"
    results_json = lc.LIVE / "json" / "topologies_results.json"
    
    lc.write_csv(trials_csv, all_rows)
    lc.write_csv(results_csv, agg)
    lc.write_json(results_json, {"topologies": TOPOLOGIES, "sweeps": SWEEPS, "n_trials": N_TRIALS, "aggregated": agg})
    
    print(f"Saved results to:\n  {trials_csv}\n  {results_csv}\n  {results_json}", flush=True)


if __name__ == "__main__":
    main()
