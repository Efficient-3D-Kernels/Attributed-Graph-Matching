"""
Evaluate Fast-WCS Multi-Start (10 restarts) across all 1400 synthetic trials and update live CSVs and figures.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_common as lc
import numpy as np
import pandas as pd
from gnccp_advanced_v3 import AdvancedGNCCP
from run_synthetic_replication_live import aggregate, SCENARIOS, SEED_BASE, load_configs


def eval_one_multistart_trial(r):
    scen = r["scenario"]
    M, N, L = int(r["M"]), int(r["N"]), int(r["L"])
    sigma, density = float(r["sigma"]), float(r["density"])
    topology = r["topology"]
    seed = int(r["seed"])
    t = int(r["trial"])
    
    AG, AH, Xgt = lc.make_synthetic_instance(M, N, L, sigma, density, seed, topology)
    F_gt = float(r["F_gt"])
    
    t0 = time.perf_counter()
    best_X = None
    best_F = float('inf')
    err = None
    
    try:
        n_restarts = 10
        for k in range(n_restarts):
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
                
        ms = (time.perf_counter() - t0) * 1000.0
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000.0
        best_X = None
        err = str(e)
        
    res = dict(scenario=scen, setting=int(r["setting"]), x=float(r["x"]), M=M, N=N, L=L,
               sigma=sigma, density=density, topology=topology, trial=t, seed=seed,
               method="Fast-WCS (Multi-Start, 10 paths)", ms=ms, F_gt=F_gt, error=err)
    if err is None and best_X is not None:
        try:
            lc.validate_exact_L(best_X, L)
            F = lc.wcs_objective(best_X, AG, AH)
            res.update(acc=lc.accuracy_pct(best_X, Xgt, L), F=F, F_ratio=F / max(F_gt, 1e-12),
                       F_le_gt=int(F <= F_gt + 1e-9))
        except ValueError as e:
            res["error"] = f"invalid output: {e}"
    return res


def clean_aggregate(rows):
    out = []
    keys = sorted({(r["scenario"], r["setting"], r["method"]) for r in rows}, key=lambda k: (k[0], k[1], k[2]))
    for scen, si, m in keys:
        rs = [r for r in rows if r["scenario"] == scen and r["setting"] == si and r["method"] == m]
        ok = [r for r in rs if r.get("error") is None or pd.isna(r.get("error")) or str(r.get("error")).strip() in ("", "nan", "None")]
        acc = [float(r["acc"]) for r in ok if r.get("acc") is not None and not pd.isna(r.get("acc"))]
        lo, hi = lc.mean_ci95(acc, seed=si) if acc else (0.0, 0.0)
        s_acc, s_ms = lc.summarize(acc), lc.summarize([float(r["ms"]) for r in ok])
        r0 = rs[0]
        out.append(dict(scenario=scen, x_name=r0.get("x_name", scen), x=r0["x"], M=r0["M"], N=r0["N"], L=r0["L"], sigma=r0["sigma"],
                        density=r0["density"], topology=r0["topology"], method=m, n_trials=len(rs), n_ok=len(ok),
                        n_errors=len(rs) - len(ok), acc_mean=s_acc["mean"], acc_sd=s_acc["sd"], acc_ci95_lo=lo, acc_ci95_hi=hi,
                        F_ratio_median=lc.summarize([float(r["F_ratio"]) for r in ok])["median"] if ok else None,
                        frac_F_le_gt=float(np.mean([float(r["F_le_gt"]) for r in ok])) if ok else None,
                        ms_median=s_ms["median"], ms_q25=s_ms["q25"], ms_q75=s_ms["q75"], ms_mean=s_ms["mean"]))
    return out


def main():
    trials_csv = lc.LIVE / "csv" / "synthetic_trials.csv"
    if not trials_csv.exists():
        print("synthetic_trials.csv not found!", flush=True)
        return
    
    df = pd.read_csv(trials_csv)
    # Filter unique instances based on Fast-WCS default
    target_instances = df[df.method == "Fast-WCS v3 (default)"].to_dict(orient="records")
    existing_other_rows = df[df.method != "Fast-WCS (Multi-Start, 10 paths)"].to_dict(orient="records")
    
    print(f"Executing 10-restart Fast-WCS over {len(target_instances)} trials using 8 workers...", flush=True)
    t_start = time.perf_counter()
    with mp.Pool(processes=8) as pool:
        multistart_rows = pool.map(eval_one_multistart_trial, target_instances)
    print(f"Completed {len(multistart_rows)} trials in {time.perf_counter() - t_start:.2f} s.", flush=True)
    
    all_rows = existing_other_rows + multistart_rows
    agg = clean_aggregate(all_rows)
    
    # Save updated CSVs
    lc.write_csv(trials_csv, all_rows)
    lc.write_csv(lc.LIVE / "csv" / "synthetic_results.csv", agg)
    
    # Save updated JSON
    cal, fcal = load_configs()
    payload = {"script": "run_synthetic_replication_live.py", "scenarios": SCENARIOS, "seed_base": SEED_BASE,
               "calibrated_baselines": cal, "fastwcs_long_path": fcal, "yang_settings": lc.YANG_SETTINGS,
               "aggregated": agg, "provenance": lc.provenance([Path(__file__)])}
    lc.write_json(lc.LIVE / "json" / "synthetic_results.json", payload)
    
    print("Updated synthetic_trials.csv and synthetic_results.csv successfully!", flush=True)


if __name__ == "__main__":
    main()
