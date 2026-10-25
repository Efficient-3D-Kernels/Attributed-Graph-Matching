"""Held-out hyper-parameter calibration for the competitor solvers (run BEFORE the benchmarks).

For every PyGMtools solver (SM, RRWM, IPFP, NGM) the affinity construction is chosen from
  * 'wcs'      : Lawler affinity equivalent to the WCS objective (same objective Fast-WCS minimises),
  * 'gauss:s'  : classical edge-only Gaussian affinity exp(-(w_ij - w_ab)^2 / s^2), s in GAUSS_SIGMAS,
by mean matching accuracy on HELD-OUT instances whose seeds (9_000_000+) are never used by any
reported experiment. Fast-WCS is NOT tuned (library defaults). The chosen settings are written to
benchmarks/live_runs/json/baseline_calibration.json and read by the benchmark scripts.

Also recorded (diagnostic): how often each solver's assignment has a LOWER WCS objective than the
planted ground truth -- i.e. whether accuracy loss is an optimisation failure or a property of the
objective -- under both synthetic topologies.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_common as lc  # noqa: E402  (sets single-thread env before numpy)
import numpy as np  # noqa: E402

GAUSS_SIGMAS = (0.05, 0.1, 0.2, 0.4)
HELDOUT_SEED = 9_000_000
N_INST = 12
CFG = dict(M=30, N=35, L=25, sigma=0.03, density=0.5)


def candidates():
    out = {}
    for m in ("sm", "rrwm", "ipfp", "ngm"):
        out[f"{m}|wcs"] = dict(method=m, affinity="wcs")
        for s in GAUSS_SIGMAS:
            out[f"{m}|gauss:{s}"] = dict(method=m, affinity="gauss", sigma=s)
    return out


def task(args):
    topo, k = args
    AG, AH, Xgt = lc.make_synthetic_instance(CFG["M"], CFG["N"], CFG["L"], CFG["sigma"], CFG["density"],
                                             seed=HELDOUT_SEED + 1000 * (topo == "threshold") + k, topology=topo)
    L = CFG["L"]
    f_gt = lc.wcs_objective(Xgt, AG, AH)
    rec = {"topology": topo, "k": k, "F_gt": f_gt, "solvers": {}}
    solvers = {"fastwcs": lambda: lc.solve_fastwcs(AG, AH, L),
               "rlx1f": lambda: lc.solve_yang(AG, AH, L, relaxation="RLX1", oracle="fast_heuristic"),
               "faq": lambda: lc.solve_faq(AG, AH, L),
               "fgw_cg": lambda: lc.solve_fgw(AG, AH, L),
               "pgw": lambda: lc.solve_partial_gw(AG, AH, L)}
    if topo == "shared_er":
        for name, kw in candidates().items():
            solvers[name] = (lambda kw=kw: lc.solve_pygm(AG, AH, L, **kw))
    else:
        for m in ("sm", "rrwm", "ipfp", "ngm"):
            for name, kw in [(f"{m}|wcs", dict(method=m, affinity="wcs")), (f"{m}|gauss:0.1", dict(method=m, affinity="gauss", sigma=.1))]:
                solvers[name] = (lambda kw=kw: lc.solve_pygm(AG, AH, L, **kw))
    for name, fn in solvers.items():
        X, ms, err = lc.timed_call(fn)
        if err:
            rec["solvers"][name] = {"error": err, "ms": ms}
            continue
        lc.validate_exact_L(X, L)
        F = lc.wcs_objective(X, AG, AH)
        rec["solvers"][name] = {"acc": lc.accuracy_pct(X, Xgt, L), "F": F, "F_le_gt": bool(F <= f_gt + 1e-9), "ms": ms}
    return rec


def main():
    out_dir = lc.run_dir("baseline_calibration")
    sys.stdout = lc.Tee(out_dir / "run.log")
    tasks = [(t, k) for t in ("shared_er", "threshold") for k in range(N_INST)]
    recs = lc.run_pool(task, tasks, workers=9, desc="calibration", log_every=4)
    summary = {}
    for topo in ("shared_er", "threshold"):
        rs = [r for r in recs if r["topology"] == topo]
        names = sorted({n for r in rs for n in r["solvers"]})
        summary[topo] = {}
        for n in names:
            v = [r["solvers"][n] for r in rs if n in r["solvers"] and "acc" in r["solvers"][n]]
            summary[topo][n] = {"acc_mean": float(np.mean([x["acc"] for x in v])) if v else None,
                                "ms_median": float(np.median([x["ms"] for x in v])) if v else None,
                                "frac_F_le_gt": float(np.mean([x["F_le_gt"] for x in v])) if v else None,
                                "n_ok": len(v), "n_err": len(rs) - len(v)}
            print(f"{topo:10s} {n:16s} acc {summary[topo][n]['acc_mean']}  ms {summary[topo][n]['ms_median']}  "
                  f"F<=F_gt {summary[topo][n]['frac_F_le_gt']}", flush=True)
    chosen = {}
    for m in ("sm", "rrwm", "ipfp", "ngm"):
        opts = {k: v for k, v in summary["shared_er"].items() if k.startswith(m + "|") and v["acc_mean"] is not None}
        best = max(opts, key=lambda k: (round(opts[k]["acc_mean"], 6), -opts[k]["ms_median"]))
        kw = candidates()[best]
        chosen[m] = {k: v for k, v in kw.items() if k != "method"}
        chosen[m]["heldout_acc_mean"] = opts[best]["acc_mean"]
        print("chosen", m, chosen[m], flush=True)
    payload = {"protocol": {"heldout_seed_base": HELDOUT_SEED, "n_instances": N_INST, "config": CFG,
                            "gauss_sigmas": GAUSS_SIGMAS, "selection": "max held-out mean accuracy, tie -> faster"},
               "chosen": chosen, "summary": summary, "records": recs,
               "provenance": lc.provenance([Path(__file__)])}
    lc.write_json(out_dir / "baseline_calibration.json", payload)
    lc.write_json(lc.LIVE / "json" / "baseline_calibration.json", payload)
    print("wrote", lc.LIVE / "json" / "baseline_calibration.json")


if __name__ == "__main__":
    main()
