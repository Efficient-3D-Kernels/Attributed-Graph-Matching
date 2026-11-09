"""run_synthetic.py -- planted WCS benchmark on Erdos-Renyi, Barabasi-Albert and random-regular
topologies (paper Sec. VIII-A / VIII-E generator), graph sizes up to 1000.

Instance: points uniform in the unit square; L points of H copied into G with N(0,sigma^2) noise;
edge weights = Euclidean distances; the L planted vertices share one topology pattern
(ER(rho) / BA(m=round(rho*(L-1)/2)) / d-regular with d=round(rho*(L-1))); every other vertex pair is an
independent Bernoulli(rho) edge; finally round(sigma*|E|/2) edges are deleted and as many inserted per graph.
Size sweep follows the paper: M=size, N=M+5, L=M-5.  Unattributed (C=0), alpha=1.

Usage: python run_synthetic.py --sizes 50 100 200 500 1000 --rhos 0.1 0.3 0.5 --trials 10 --out results_syn
NOTE: Fast-WCS needs dense N x N matrices and an O(N^3) assignment per step; N=1000 is feasible
(minutes), N=10000 is not on a laptop (see README).
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import cdist
import expcommon as ec


def core_pattern(topology, L, rho, rng):
    seed = int(rng.integers(1 << 31))
    if topology == "ER":
        B = np.triu(rng.random((L, L)) < rho, 1)
        return B | B.T
    if topology == "BA":
        m = int(min(max(1, round(rho * (L - 1) / 2)), L - 1))
        return nx.to_numpy_array(nx.barabasi_albert_graph(L, m, seed=seed)).astype(bool)
    if topology == "RR":
        d = int(min(max(1, round(rho * (L - 1))), L - 1))
        if (L * d) % 2:
            d = d - 1 if d > 1 else d + 1
        return nx.to_numpy_array(nx.random_regular_graph(d, L, seed=seed)).astype(bool)
    raise ValueError(topology)


def _adjacency(n, core_idx, B, rho, sigma, rng):
    adj = np.triu(rng.random((n, n)) < rho, 1)
    adj = adj | adj.T
    adj[np.ix_(core_idx, core_idx)] = B
    np.fill_diagonal(adj, False)
    ti = np.triu_indices(n, 1)
    e = adj[ti]
    k = int(round(sigma * e.sum() / 2))
    if k > 0:
        on = np.flatnonzero(e); off = np.flatnonzero(~e)
        k = min(k, len(on), len(off))
        e[rng.choice(on, k, replace=False)] = False
        e[rng.choice(off, k, replace=False)] = True
    out = np.zeros((n, n), bool)
    out[ti] = e
    return out | out.T


def synthetic_instance(topology, M, N, L, sigma, rho, rng):
    PH = rng.random((N, 2))
    gt_h = rng.choice(N, L, replace=False)
    gt_g = rng.choice(M, L, replace=False)
    PG = rng.random((M, 2))
    PG[gt_g] = PH[gt_h] + sigma * rng.standard_normal((L, 2))
    B = core_pattern(topology, L, rho, rng)
    AG = cdist(PG, PG) * _adjacency(M, gt_g, B, rho, sigma, rng)
    AH = cdist(PH, PH) * _adjacency(N, gt_h, B, rho, sigma, rng)
    return dict(AG=AG, AH=AH, C=np.zeros((M, N)), L=L, gt_rows=gt_g, gt_cols=gt_h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topologies", nargs="+", default=["ER", "BA", "RR"])
    ap.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 200, 500, 1000])
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    ap.add_argument("--sigma", type=float, default=0.03)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--trials_large", type=int, default=3, help="trials for size > 200")
    ap.add_argument("--multi_max_n", type=int, default=200)
    ap.add_argument("--baseline_max_n", type=int, default=300, help="RLX1F / FAQ only up to this size")
    ap.add_argument("--multi_starts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results_syn")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ec.log_environment(args.out, args)
    alpha = 1.0
    recs = []
    t0 = time.time()
    for topo in args.topologies:
        for rho in args.rhos:
            for size in args.sizes:
                M, N, L = size, size + 5, size - 5
                inc = ["single"]
                if size <= args.multi_max_n:
                    inc.append("multi")
                if size <= args.baseline_max_n:
                    inc += ["rlx1f", "faq"]
                solvers = ec.make_solvers(alpha, inc, args.multi_starts, args.seed)
                for t in range(args.trials if size <= 200 else args.trials_large):
                    rng = np.random.default_rng([args.seed, {"ER": 1, "BA": 2, "RR": 3}[topo], int(rho * 100), size, t])
                    inst = synthetic_instance(topo, M, N, L, args.sigma, rho, rng)
                    iid = f"{topo}|{rho}|{size}|{t}"
                    for name, f in solvers.items():
                        r = ec.evaluate(inst, f(inst), alpha)
                        r.update(method=name, instance=iid, topology=topo, rho=rho, size=size, trial=t)
                        recs.append(r)
                print(f"[{time.time() - t0:7.0f}s] {topo} rho={rho} size={size} done", flush=True)
                pd.DataFrame(recs).to_csv(os.path.join(args.out, "per_trial.csv"), index=False)
    df = pd.DataFrame(recs)
    summ = ec.summarise(df, ["topology", "rho", "size", "method"], ["acc", "F_ratio", "F_le_gt", "time_ms"])
    summ.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for topo in args.topologies:
        for rho in args.rhos:
            s = summ[(summ.topology == topo) & (summ.rho == rho)]
            fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
            for m, g in s.groupby("method", sort=False):
                g = g.sort_values("size")
                ax[0].plot(g["size"], g["acc_mean"], marker="o", label=m)
                ax[1].plot(g["size"], g["time_ms_median"], marker="o", label=m)
            ax[0].set_ylabel("Matching accuracy (%)"); ax[1].set_ylabel("Median time (ms)")
            ax[1].set_yscale("log"); ax[1].set_xscale("log")
            for a in ax:
                a.set_xlabel("graph size M (N=M+5, L=M-5)"); a.grid(alpha=.3)
            ax[0].legend(fontsize=6); fig.suptitle(f"{topo}, rho={rho}")
            fig.tight_layout(); fig.savefig(os.path.join(args.out, f"syn_{topo}_rho{rho:g}.png"), dpi=200)
            plt.close(fig)
    pd.set_option("display.width", 200)
    print(summ[["topology", "rho", "size", "method", "n", "acc_mean", "F_ratio_mean", "time_ms_median"]]
          .round(2).to_string(index=False))


if __name__ == "__main__":
    main()
