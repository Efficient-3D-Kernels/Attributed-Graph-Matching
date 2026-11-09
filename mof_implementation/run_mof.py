"""run_mof.py -- Fast-WCS on real MOF crystal structures (CoRE MOF 2019).

Three sweeps mirroring the paper's synthetic protocol (Sec. VIII-A) but on real atomic graphs:
  noise    : sigma in {0,.05,.1,.2,.3} A,   M=35,N=40,L=30
  outliers : M-L in {0,2,4,6,8,10},        M=35,N=40, sigma=0.1
  size     : M in {20,30,40,60}, N=M+5, L=M-5, sigma=0.1
One instance per (structure, setting); every solver sees identical instances.

Usage:  python run_mof.py --cif_dir data/CoREMOF2019 --n_structures 100 --out results_mof
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
import mof_data as md
import expcommon as ec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif_dir", required=True)
    ap.add_argument("--filter", default=None, help="substring of path, e.g. ASR or FSR")
    ap.add_argument("--n_structures", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--omegas", type=float, nargs="+", default=[50.0],
                    help="element-mismatch penalty(ies); paper Fig.1 uses 50. Add 0 for an ablation")
    ap.add_argument("--sweeps", nargs="+", default=["noise", "outliers", "size"])
    ap.add_argument("--methods", nargs="+", default=["single", "multi", "rlx1f", "faq", "feature"])
    ap.add_argument("--multi_starts", type=int, default=10)
    ap.add_argument("--drop_hydrogen", action="store_true")
    ap.add_argument("--max_atoms", type=int, default=3000, help="skip CIFs with more atoms in the cell")
    ap.add_argument("--out", default="results_mof")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ec.log_environment(args.out, args)

    paths = md.find_cifs(args.cif_dir, args.filter)
    print(f"{len(paths)} CIF files found")
    rng0 = np.random.default_rng(args.seed)
    rng0.shuffle(paths)

    structures = []                                       # first n_structures that load correctly
    for p in paths:
        if len(structures) >= args.n_structures:
            break
        try:
            at = md.load_structure(p)
            if len(at) > args.max_atoms:
                continue
            md.local_cluster(at, 0, 120)                 # check a cluster of 120 atoms can be built
            structures.append((os.path.basename(p), at))
        except Exception:
            continue
    print(f"{len(structures)} usable structures")
    pd.DataFrame({"cif": [s[0] for s in structures]}).to_csv(os.path.join(args.out, "structures_used.csv"),
                                                              index=False)

    settings = []
    if "noise" in args.sweeps:
        settings += [dict(sweep="noise", M=35, N=40, L=30, sigma=s, x=s) for s in (0.0, 0.05, 0.1, 0.2, 0.3)]
    if "outliers" in args.sweeps:
        settings += [dict(sweep="outliers", M=35, N=40, L=35 - k, sigma=0.1, x=k) for k in (0, 2, 4, 6, 8, 10)]
    if "size" in args.sweeps:
        settings += [dict(sweep="size", M=m, N=m + 5, L=m - 5, sigma=0.1, x=m) for m in (20, 30, 40, 60)]

    records = []
    t_start = time.time()
    for omega in args.omegas:
        solvers = ec.make_solvers(args.alpha, args.methods, args.multi_starts, args.seed)
        # warm-up so that imports/allocation are not timed
        warm = None
        for si, (name, at) in enumerate(structures[:1]):
            warm = md.build_mof_instance(at, 20, 25, 15, 0.1, omega, np.random.default_rng(1),
                                         drop_hydrogen=args.drop_hydrogen)
        for f in solvers.values():
            f(warm)
        for st in settings:
            for si, (name, at) in enumerate(structures):
                rng = np.random.default_rng([args.seed, si, int(1000 * st["sigma"]), st["M"], st["L"]])
                try:
                    inst = md.build_mof_instance(at, st["M"], st["N"], st["L"], st["sigma"], omega, rng,
                                                 drop_hydrogen=args.drop_hydrogen)
                except Exception as e:
                    continue
                iid = f"{omega}|{st['sweep']}|{st['x']}|{name}"
                for mname, f in solvers.items():
                    res = f(inst)
                    r = ec.evaluate(inst, res, args.alpha)
                    r.update(method=mname, instance=iid, cif=name, omega=omega, **st)
                    records.append(r)
            print(f"[{time.time() - t_start:7.0f}s] omega={omega} {st['sweep']}={st['x']} done", flush=True)
            pd.DataFrame(records).to_csv(os.path.join(args.out, "per_trial.csv"), index=False)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(args.out, "per_trial.csv"), index=False)
    metrics = ["acc", "F_ratio", "F_le_gt", "elem_consistency", "rmsd_A", "time_ms"]
    summ = ec.summarise(df, ["omega", "sweep", "x", "method"], metrics)
    summ.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    ec.paired_wilcoxon(df).to_csv(os.path.join(args.out, "wilcoxon_vs_single_start.csv"), index=False)
    ec.paired_wilcoxon(df, ref="Fast-WCS (multi-start)").to_csv(
        os.path.join(args.out, "wilcoxon_vs_multi_start.csv"), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for omega in args.omegas:
        for sweep in args.sweeps:
            for metric, lab in (("acc", "Matching accuracy (%)"), ("time_ms", "Median time (ms)")):
                s = summ[(summ.omega == omega) & (summ.sweep == sweep)]
                if s.empty:
                    continue
                plt.figure(figsize=(5, 3.6))
                for m, g in s.groupby("method", sort=False):
                    g = g.sort_values("x")
                    y = g["time_ms_median"] if metric == "time_ms" else g[metric + "_mean"]
                    plt.plot(g["x"], y, marker="o", label=m)
                    if metric == "acc":
                        plt.fill_between(g["x"], g["acc_lo"], g["acc_hi"], alpha=0.15)
                plt.xlabel({"noise": "coordinate noise sigma (A)", "outliers": "outliers M-L",
                            "size": "graph size M"}[sweep])
                plt.ylabel(lab)
                if metric == "time_ms":
                    plt.yscale("log")
                plt.legend(fontsize=6); plt.grid(alpha=.3); plt.tight_layout()
                plt.savefig(os.path.join(args.out, f"mof_{sweep}_{metric}_omega{omega:g}.png"), dpi=200)
                plt.close()
    pd.set_option("display.width", 200)
    print(summ[["omega", "sweep", "x", "method", "n", "acc_mean", "acc_lo", "acc_hi", "F_ratio_mean",
                "rmsd_A_mean", "time_ms_median"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
