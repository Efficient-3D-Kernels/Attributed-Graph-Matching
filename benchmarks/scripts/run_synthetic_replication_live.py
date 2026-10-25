"""Script 1 - live replication of the synthetic WCS protocol of Yang, Qiao & Liu (IEEE TNNLS 2018).

Scenarios (50 Monte Carlo trials per setting, every solver sees the identical instance):
  A  noise     sigma in {0, .02, .04, .06, .08, .10}           M=30, N=35, L=25, rho=.5
  B  size      M in {20, 24, ..., 60}, N=M+5, L=M-5            sigma=.03, rho=.5
  C  outliers  M-L in {0, 2, ..., 10}                          M=30, N=35, sigma=.03, rho=.5
  D  density   rho in {.2, .4, .6, .8, 1.0}                    M=30, N=35, L=25, sigma=.03

Solvers: Fast-WCS v3 (library defaults; and the held-out 'long path' configuration), Yang et al.
RLX1 / RLX2 (exact LP oracle) and RLX1F (fast Hungarian+top-L oracle), SciPy FAQ, PyGMtools SM /
IPFP / RRWM / NGM (pretrained, VOC), POT entropic FGW (eps = 1e-2, 1e-3) and partial GW.
Competitor hyper-parameters come from calibrate_baselines_heldout.py (held-out seeds only).

Outputs
  benchmarks/live_runs/<timestamp>_synthetic/{trials.csv, run.log, synthetic_results.json}
  benchmarks/live_runs/csv/synthetic_results.csv          (aggregated per scenario/setting/method)
  benchmarks/live_runs/csv/synthetic_trials.csv           (one row per trial and method)
  benchmarks/live_runs/json/synthetic_results.json
Timing: every worker process is pinned to one BLAS/OpenMP/torch thread; WORKERS processes run
concurrently, solver order is shuffled per trial; times include output projection to exact-L.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_common as lc  # noqa: E402
import numpy as np  # noqa: E402

SEED_BASE = 20_261_016
SCENARIOS = {
    "A_noise": [dict(M=30, N=35, L=25, sigma=s, density=.5, x=s) for s in (0.0, .02, .04, .06, .08, .10)],
    "B_size": [dict(M=m, N=m + 5, L=m - 5, sigma=.03, density=.5, x=m) for m in range(20, 61, 4)],
    "C_outliers": [dict(M=30, N=35, L=30 - o, sigma=.03, density=.5, x=o) for o in range(0, 11, 2)],
    "D_density": [dict(M=30, N=35, L=25, sigma=.03, density=r, x=r) for r in (.2, .4, .6, .8, 1.0)],
}
X_LABEL = {"A_noise": "sigma", "B_size": "M", "C_outliers": "M-L", "D_density": "rho"}


def load_configs():
    cal = json.loads((lc.LIVE / "json" / "baseline_calibration.json").read_text())["chosen"]
    fcal = json.loads((lc.LIVE / "json" / "fastwcs_calibration.json").read_text())["long_path"]["config"]
    return cal, fcal


def solver_table(cal, fcal):
    def pg(m):
        kw = {k: v for k, v in cal[m].items() if k in ("affinity", "sigma")}
        return lambda AG, AH, L: lc.solve_pygm(AG, AH, L, method=m, **kw)
    return {
        "Fast-WCS v3 (default)": lambda AG, AH, L: lc.solve_fastwcs(AG, AH, L),
        "Fast-WCS v3 (long path)": lambda AG, AH, L: lc.solve_fastwcs(AG, AH, L, **fcal),
        "RLX1 (Yang 2018)": lambda AG, AH, L: lc.solve_yang(AG, AH, L, relaxation="RLX1", oracle="exact_lp"),
        "RLX2 (Yang 2018)": lambda AG, AH, L: lc.solve_yang(AG, AH, L, relaxation="RLX2", oracle="exact_lp"),
        "RLX1F (Yang 2018)": lambda AG, AH, L: lc.solve_yang(AG, AH, L, relaxation="RLX1", oracle="fast_heuristic"),
        "FAQ": lambda AG, AH, L: lc.solve_faq(AG, AH, L),
        "SM": pg("sm"),
        "IPFP": pg("ipfp"),
        "RRWM": pg("rrwm"),
        "NGM (pretrained)": pg("ngm"),
        "FGW (eps=1e-2)": lambda AG, AH, L: lc.solve_fgw(AG, AH, L, epsilon=1e-2),
        "FGW (eps=1e-3)": lambda AG, AH, L: lc.solve_fgw(AG, AH, L, epsilon=1e-3),
        "Partial GW": lambda AG, AH, L: lc.solve_partial_gw(AG, AH, L),
    }


def trial(task):
    scen, si, cfg, t, topology, methods = task
    seed = SEED_BASE + (list(SCENARIOS).index(scen) * 1_000_000) + si * 1000 + t + (500_000 if topology != "shared_er" else 0)
    AG, AH, Xgt = lc.make_synthetic_instance(cfg["M"], cfg["N"], cfg["L"], cfg["sigma"], cfg["density"], seed, topology)
    L = cfg["L"]
    F_gt = lc.wcs_objective(Xgt, AG, AH)
    table = solver_table(*load_configs())
    lc.warmup_once(table, "synthetic")
    order = [m for m in table if m in methods]
    np.random.default_rng(seed).shuffle(order)
    rows = []
    for m in order:
        np.random.seed(seed % (2 ** 32 - 1))
        X, ms, err = lc.timed_call(table[m], AG, AH, L)
        row = dict(scenario=scen, setting=si, x=cfg["x"], M=cfg["M"], N=cfg["N"], L=L, sigma=cfg["sigma"],
                   density=cfg["density"], topology=topology, trial=t, seed=seed, method=m, ms=ms, F_gt=F_gt, error=err)
        if err is None:
            try:
                lc.validate_exact_L(X, L)
                F = lc.wcs_objective(X, AG, AH)
                row.update(acc=lc.accuracy_pct(X, Xgt, L), F=F, F_ratio=F / max(F_gt, 1e-12), F_le_gt=int(F <= F_gt + 1e-9))
            except ValueError as e:
                row["error"] = f"invalid output: {e}"
        rows.append(row)
    return rows


def aggregate(rows):
    out = []
    keys = sorted({(r["scenario"], r["setting"], r["method"]) for r in rows}, key=lambda k: (k[0], k[1], k[2]))
    for scen, si, m in keys:
        rs = [r for r in rows if r["scenario"] == scen and r["setting"] == si and r["method"] == m]
        ok = [r for r in rs if r.get("error") is None]
        acc = [r["acc"] for r in ok]
        lo, hi = lc.mean_ci95(acc, seed=si)
        s_acc, s_ms = lc.summarize(acc), lc.summarize([r["ms"] for r in ok])
        r0 = rs[0]
        out.append(dict(scenario=scen, x_name=X_LABEL[scen], x=r0["x"], M=r0["M"], N=r0["N"], L=r0["L"], sigma=r0["sigma"],
                        density=r0["density"], topology=r0["topology"], method=m, n_trials=len(rs), n_ok=len(ok),
                        n_errors=len(rs) - len(ok), acc_mean=s_acc["mean"], acc_sd=s_acc["sd"], acc_ci95_lo=lo, acc_ci95_hi=hi,
                        F_ratio_median=lc.summarize([r["F_ratio"] for r in ok])["median"],
                        frac_F_le_gt=float(np.mean([r["F_le_gt"] for r in ok])) if ok else None,
                        ms_median=s_ms["median"], ms_q25=s_ms["q25"], ms_q75=s_ms["q75"], ms_mean=s_ms["mean"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--topology", default="shared_er", choices=["shared_er", "threshold"])
    ap.add_argument("--methods", default="all")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cal, fcal = load_configs()
    methods = list(solver_table(cal, fcal)) if a.methods == "all" else a.methods.split(",")
    name = "synthetic" + ("" if a.topology == "shared_er" else "_threshold") + a.tag
    out_dir = lc.run_dir(name)
    sys.stdout = lc.Tee(out_dir / "run.log")
    print(f"Script 1 | trials={a.trials} workers={a.workers} topology={a.topology} methods={methods}\n"
          f"calibrated baselines: {cal}\nFast-WCS long path: {fcal}", flush=True)
    tasks = [(s, si, cfg, t, a.topology, methods) for s in a.scenarios.split(",") for si, cfg in enumerate(SCENARIOS[s])
             for t in range(a.trials)]
    # largest instances first -> better load balance
    tasks.sort(key=lambda z: -(z[2]["M"] * z[2]["N"]))
    res = lc.run_pool(trial, tasks, workers=a.workers, desc=name, log_every=20)
    rows = [r for rs in res for r in rs]
    agg = aggregate(rows)
    fmt = lambda v: "   n/a" if v is None else f"{v:6.1f}"
    for r in agg:
        print(f"{r['scenario']:11s} {r['x_name']}={r['x']:<5} {r['method']:24s} acc {fmt(r['acc_mean'])} "
              f"ms {fmt(r['ms_median'])}  err {r['n_errors']}", flush=True)
    lc.write_csv(out_dir / "trials.csv", rows)
    lc.write_csv(out_dir / "aggregated.csv", agg)
    payload = {"script": Path(__file__).name, "args": vars(a), "scenarios": SCENARIOS, "seed_base": SEED_BASE,
               "calibrated_baselines": cal, "fastwcs_long_path": fcal, "yang_settings": lc.YANG_SETTINGS,
               "aggregated": agg, "provenance": lc.provenance([Path(__file__)])}
    lc.write_json(out_dir / f"{name}_results.json", payload)
    lc.write_csv(lc.LIVE / "csv" / f"{name}_results.csv", agg)
    lc.write_csv(lc.LIVE / "csv" / f"{name}_trials.csv", rows)
    lc.write_json(lc.LIVE / "json" / f"{name}_results.json", payload)
    print("wrote", lc.LIVE / "csv" / f"{name}_results.csv", flush=True)


if __name__ == "__main__":
    main()
