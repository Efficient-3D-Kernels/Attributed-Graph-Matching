"""run_conformer.py -- 3D conformer-alignment experiment on real MOF fragments (paper Fig. 7a-d analogue).

Scenarios A/B/C = fragments of M = 16 / 20 / 24 atoms (N = M+4, L = M-2).  G is a randomly rotated and
translated copy of a core of H (SO(3) x R^3 motion), perturbed by N(0, sigma^2) per coordinate (flexibility),
plus 2 outlier atoms; atom order is random.  Methods return a correspondence; Phase 2 superposes the
matched atoms with Kabsch and reports RMSD.  Success = RMSD < 1.0 A (docking threshold) and < 0.1 A.
(The scenario definitions are ours; the paper does not specify them.)
Usage: python run_conformer.py --cif_dir "data/.../CR/ASR" --n_structures 60 --sigma 0.05 --out results_conf
"""
import argparse, os
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist
import mof_data as md, expcommon as ec, fastwcs as fw


def random_rotation(rng):
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif_dir", required=True); ap.add_argument("--filter", default=None)
    ap.add_argument("--n_structures", type=int, default=60)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.7); ap.add_argument("--omega", type=float, default=50.0)
    ap.add_argument("--methods", nargs="+", default=["single", "multi", "rlx1f", "faq", "fgw"])
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="results_conf")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True); ec.log_environment(args.out, args)
    try:
        import ot  # noqa
    except ImportError:
        args.methods = [m for m in args.methods if m != "fgw"]
        print("POT not installed -> FGW baseline skipped (pip install pot)")
    paths = md.find_cifs(args.cif_dir, args.filter)
    np.random.default_rng(args.seed).shuffle(paths)
    structs = []
    for p in paths:
        if len(structs) >= args.n_structures:
            break
        try:
            at = md.load_structure(p); md.local_cluster(at, 0, 80); structs.append((os.path.basename(p), at))
        except Exception:
            pass
    print(len(structs), "structures")
    solvers = ec.make_solvers(args.alpha, args.methods, 10, args.seed)
    for f in solvers.values():                                   # warm-up
        i = md.build_mof_instance(structs[0][1], 16, 20, 14, args.sigma, args.omega, np.random.default_rng(1))
        f(i)
    scen = {"Sc. A (M=16)": 16, "Sc. B (M=20)": 20, "Sc. C (M=24)": 24}
    recs, inv_err = [], 0.0
    for sname, M in scen.items():
        N, L = M + 4, M - 2
        for si, (name, at) in enumerate(structs):
            rng = np.random.default_rng([args.seed, si, M])
            try:
                inst = md.build_mof_instance(at, M, N, L, args.sigma, args.omega, rng)
            except Exception:
                continue
            Rm, t = random_rotation(rng), rng.uniform(-20, 20, 3)
            d_before = cdist(inst["posG"], inst["posG"])
            inst["posG"] = inst["posG"] @ Rm.T + t                    # rigid motion applied to conformer G
            inv_err = max(inv_err, np.abs(cdist(inst["posG"], inst["posG"]) - d_before).max())
            ref = fw.kabsch_rmsd(inst["posG"][inst["gt_rows"]], inst["posH"][inst["gt_cols"]])
            for mname, f in solvers.items():
                r = ec.evaluate(inst, f(inst), args.alpha)
                r.update(method=mname, scenario=sname, M=M, cif=name, instance=f"{sname}|{name}",
                         succ_1A=100.0 * (r["rmsd_A"] < 1.0), succ_0p1A=100.0 * (r["rmsd_A"] < 0.1),
                         rmsd_planted_ref=ref)
                recs.append(r)
        print("done", sname, flush=True)
        pd.DataFrame(recs).to_csv(os.path.join(args.out, "per_trial.csv"), index=False)
    print("max |distance change| under rigid motion (Prop. 4 check): %.2e" % inv_err)
    df = pd.DataFrame(recs)
    summ = ec.summarise(df, ["scenario", "method"], ["acc", "rmsd_A", "succ_1A", "succ_0p1A", "time_ms"])
    med = df.groupby(["scenario", "method"])["rmsd_A"].median().rename("rmsd_A_median").reset_index()
    summ = summ.merge(med, on=["scenario", "method"])
    summ["planted_ref_rmsd_median"] = summ["scenario"].map(df.groupby("scenario")["rmsd_planted_ref"].median())
    summ.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    meths = list(dict.fromkeys(df.method)); sc = list(scen)
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.6)); w = 0.8 / len(meths)
    for k, (col, lab) in enumerate((("acc_mean", "Matching accuracy (%)"), ("rmsd_A_median", "Median RMSD (A)"),
                                    ("succ_1A_mean", "Success RMSD<1.0 A (%)"), ("time_ms_median", "Median time (ms)"))):
        for j, m in enumerate(meths):
            v = [summ[(summ.scenario == s) & (summ.method == m)][col].values[0] for s in sc]
            ax[k].bar(np.arange(3) + j * w, v, w, label=m)
        ax[k].set_xticks(np.arange(3) + 0.4 - w / 2); ax[k].set_xticklabels(sc, fontsize=7); ax[k].set_ylabel(lab)
    ax[1].axhline(1.0, ls="--", c="k"); ax[3].set_yscale("log"); ax[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "conformer_alignment.png"), dpi=200)
    pd.set_option("display.width", 220)
    print(summ[["scenario", "method", "n", "acc_mean", "rmsd_A_median", "succ_1A_mean", "succ_0p1A_mean",
                "time_ms_median", "planted_ref_rmsd_median"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
