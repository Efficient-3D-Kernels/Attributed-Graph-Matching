"""Held-out configuration study for Fast-WCS itself (same held-out seeds as the baseline calibration).

The engine defaults (8 continuation levels x 3 Frank-Wolfe steps) are speed-oriented. This script
sweeps the engine's own knobs on HELD-OUT instances (seeds 9_000_000+, never used in reported
experiments) and records accuracy, F(X)/F(X_gt) and latency for each configuration, so that the
benchmark scripts can report two pre-registered variants:
  * 'Fast-WCS (default)'  : library defaults, no tuning;
  * 'Fast-WCS (long path)': the most accurate configuration whose held-out median latency stays
    below LATENCY_BUDGET_MS at (M, N, L) = (30, 35, 25).
Output: benchmarks/live_runs/json/fastwcs_calibration.json
"""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_common as lc  # noqa: E402
import numpy as np  # noqa: E402

HELDOUT_SEED = 9_000_000
N_INST = 12
LATENCY_BUDGET_MS = 250.0
GRID = dict(max_iter=[8, 20, 50, 100], fw_max_iter=[3, 10, 20], tol=[1e-3, 1e-6],
            schedule=["cosine", "linear"], normalization=["none", "shared_max"])


def task(kw):
    accs, ratios, ts = [], [], []
    for k in range(N_INST):
        AG, AH, Xgt = lc.make_synthetic_instance(30, 35, 25, .03, .5, seed=HELDOUT_SEED + k)
        t = time.perf_counter()
        X = lc.solve_fastwcs(AG, AH, 25, **kw)
        ts.append(1e3 * (time.perf_counter() - t))
        accs.append(lc.accuracy_pct(X, Xgt, 25))
        ratios.append(lc.wcs_objective(X, AG, AH) / max(lc.wcs_objective(Xgt, AG, AH), 1e-12))
    return {"config": kw, "acc_mean": float(np.mean(accs)), "F_over_Fgt_median": float(np.median(ratios)),
            "ms_median": float(np.median(ts))}


def main():
    out_dir = lc.run_dir("fastwcs_calibration")
    sys.stdout = lc.Tee(out_dir / "run.log")
    keys = list(GRID)
    tasks = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]
    res = lc.run_pool(task, tasks, workers=9, desc="fastwcs-cfg", log_every=16)
    res.sort(key=lambda r: -r["acc_mean"])
    for r in res[:15]:
        print(r, flush=True)
    default = next(r for r in res if r["config"] == dict(max_iter=8, fw_max_iter=3, tol=1e-3, schedule="cosine", normalization="none"))
    feas = [r for r in res if r["ms_median"] <= LATENCY_BUDGET_MS]
    best = max(feas, key=lambda r: (round(r["acc_mean"], 6), -r["ms_median"]))
    print("default:", default, "\nchosen long-path:", best, flush=True)
    payload = {"protocol": {"heldout_seed_base": HELDOUT_SEED, "n_instances": N_INST, "grid": GRID,
                            "latency_budget_ms": LATENCY_BUDGET_MS,
                            "selection": "max held-out accuracy with median latency <= budget; tie -> faster"},
               "default": default, "long_path": best, "all": res, "provenance": lc.provenance([Path(__file__)])}
    lc.write_json(out_dir / "fastwcs_calibration.json", payload)
    lc.write_json(lc.LIVE / "json" / "fastwcs_calibration.json", payload)


if __name__ == "__main__":
    main()
