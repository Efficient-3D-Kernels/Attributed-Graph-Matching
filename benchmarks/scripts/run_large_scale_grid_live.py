"""Script 3 - live large-scale structural grid: N in {20,40,60,80,100,120,150} x L/N in {.4,.6,.8,1.0}.

M = N, L = round(ratio * N), synthetic protocol of Script 1 (sigma = .03, rho = .5, shared ER topology).
Solvers (identical instances): Fast-WCS v3 (library defaults), Fast-WCS v3 (held-out long path),
Yang et al. (2018) RLX1F (Hungarian + top-L oracle) and RLX1 with the exact LP oracle ("exact GNCCP").
For L = M the Yang solver switches to its Part-In-Whole gradient (Eq. 16), as in the reference code.

Per configuration: runtime median / IQR, accuracy, F / F_gt, and the per-trial speed-up of Fast-WCS
(default) over RLX1F and over RLX1 (exact LP); medians of the per-trial ratios are reported.

Outputs: csv/large_scale_grid.csv (aggregated), csv/large_scale_grid_trials.csv,
         json/large_scale_grid.json, <ts>_large_scale_grid/ run dir.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_common as lc  # noqa: E402
import numpy as np  # noqa: E402

SEED_BASE = 50_261_016
NS = (20, 40, 60, 80, 100, 120, 150)
RATIOS = (0.4, 0.6, 0.8, 1.0)
FAST, LONG, RLX1F, RLX1 = "Fast-WCS v3 (default)", "Fast-WCS v3 (long path)", "RLX1F (Yang 2018)", "RLX1 exact-LP (Yang 2018)"


def solver_table(fcal):
    return {
        FAST: lambda AG, AH, L: lc.solve_fastwcs(AG, AH, L),
        LONG: lambda AG, AH, L: lc.solve_fastwcs(AG, AH, L, **fcal),
        RLX1F: lambda AG, AH, L: lc.solve_yang(AG, AH, L, relaxation="RLX1", oracle="fast_heuristic"),
        RLX1: lambda AG, AH, L: lc.solve_yang(AG, AH, L, relaxation="RLX1", oracle="exact_lp"),
    }


def trial(task):
    N, ratio, t, fcal = task
    L = int(round(ratio * N))
    seed = SEED_BASE + N * 10_000 + int(ratio * 100) * 100 + t
    AG, AH, Xgt = lc.make_synthetic_instance(N, N, L, 0.03, 0.5, seed)
    F_gt = lc.wcs_objective(Xgt, AG, AH)
    table = solver_table(fcal)
    lc.warmup_once(table, "grid")
    order = list(table)
    np.random.default_rng(seed).shuffle(order)
    rows = []
    for m in order:
        X, ms, err = lc.timed_call(table[m], AG, AH, L)
        row = dict(N=N, M=N, L=L, ratio=ratio, trial=t, seed=seed, method=m, ms=ms, F_gt=F_gt, error=err)
        if err is None:
            lc.validate_exact_L(X, L)
            F = lc.wcs_objective(X, AG, AH)
            row.update(acc=lc.accuracy_pct(X, Xgt, L), F=F, F_ratio=F / max(F_gt, 1e-12))
        rows.append(row)
    return rows


def aggregate(rows):
    out = []
    for N in NS:
        for ratio in RATIOS:
            rr = [r for r in rows if r["N"] == N and r["ratio"] == ratio and r.get("error") is None]
            if not rr:
                continue
            by = {m: {r["trial"]: r for r in rr if r["method"] == m} for m in (FAST, LONG, RLX1F, RLX1)}
            d = dict(N=N, M=N, L=rr[0]["L"], ratio=ratio, n_trials=len(by[FAST]))
            for m, key in ((FAST, "fast"), (LONG, "long"), (RLX1F, "rlx1f"), (RLX1, "rlx1")):
                v = list(by[m].values())
                d[f"{key}_ms_median"] = float(np.median([r["ms"] for r in v]))
                d[f"{key}_ms_q25"] = float(np.percentile([r["ms"] for r in v], 25))
                d[f"{key}_ms_q75"] = float(np.percentile([r["ms"] for r in v], 75))
                d[f"{key}_acc_mean"] = float(np.mean([r["acc"] for r in v]))
                d[f"{key}_F_ratio_median"] = float(np.median([r["F_ratio"] for r in v]))
            for base, key in ((RLX1F, "rlx1f"), (RLX1, "rlx1")):
                common = sorted(set(by[FAST]) & set(by[base]))
                sp = [by[base][t]["ms"] / by[FAST][t]["ms"] for t in common]
                d[f"speedup_vs_{key}_median"] = float(np.median(sp))
                d[f"speedup_vs_{key}_min"] = float(np.min(sp))
                d[f"fast_minus_{key}_acc"] = d["fast_acc_mean"] - d[f"{key}_acc_mean"]
                d[f"fast_F_le_{key}_rate"] = float(np.mean([by[FAST][t]["F"] <= by[base][t]["F"] + 1e-9 for t in common]))
            out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ns", default=",".join(map(str, NS)))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    fcal = json.loads((lc.LIVE / "json" / "fastwcs_calibration.json").read_text())["long_path"]["config"]
    name = "large_scale_grid" + a.tag
    out_dir = lc.run_dir(name)
    sys.stdout = lc.Tee(out_dir / "run.log")
    ns = [int(x) for x in a.ns.split(",")]
    print(f"Script 3 | trials={a.trials} workers={a.workers} N={ns} L/N={RATIOS} | long path {fcal}", flush=True)
    tasks = [(N, ratio, t, fcal) for N in ns for ratio in RATIOS for t in range(a.trials)]
    tasks.sort(key=lambda z: -z[0] * z[1])
    rows = [r for rs in lc.run_pool(trial, tasks, workers=a.workers, desc=name, log_every=20) for r in rs]
    agg = aggregate(rows)
    for d in agg:
        print(f"N={d['N']:3d} L/N={d['ratio']:.1f}  Fast {d['fast_ms_median']:8.1f} ms ({d['fast_acc_mean']:5.1f}%) | "
              f"RLX1F {d['rlx1f_ms_median']:8.1f} ms ({d['rlx1f_acc_mean']:5.1f}%) | RLX1 {d['rlx1_ms_median']:9.1f} ms "
              f"({d['rlx1_acc_mean']:5.1f}%) | x{d['speedup_vs_rlx1f_median']:.1f} / x{d['speedup_vs_rlx1_median']:.1f}", flush=True)
    payload = {"script": Path(__file__).name, "args": vars(a), "Ns": ns, "ratios": RATIOS, "seed_base": SEED_BASE,
               "fastwcs_long_path": fcal, "yang_settings": lc.YANG_SETTINGS, "aggregated": agg,
               "provenance": lc.provenance([Path(__file__)])}
    lc.write_csv(out_dir / "trials.csv", rows)
    lc.write_csv(out_dir / "aggregated.csv", agg)
    lc.write_json(out_dir / f"{name}.json", payload)
    lc.write_csv(lc.LIVE / "csv" / f"{name}.csv", agg)
    lc.write_csv(lc.LIVE / "csv" / f"{name}_trials.csv", rows)
    lc.write_json(lc.LIVE / "json" / f"{name}.json", payload)
    print("wrote", lc.LIVE / "csv" / f"{name}.csv", flush=True)


if __name__ == "__main__":
    main()
