"""Script 2 - live real-image keypoint matching benchmark (Delaunay graphs + SIFT unary, alpha = 0.5).

Dataset substitution (disclosed in the paper): the 60-landmark Pascal-2007 Motorbike / Pisa annotations
used by Yang et al. (2018) are not distributed with this repository. We use the public Oxford VGG
Affine benchmark (Mikolajczyk et al., IJCV 2005; fetch_oxford_affine.py), whose 8 real sequences
ship with ground-truth homographies. For every image pair (img1 -> img2, img1 -> img3; 16 pairs) we
  1. detect SIFT keypoints (OpenCV, nfeatures = 4000) in both images,
  2. accept a correspondence (u, v) iff the homography maps u within 2 px of v and (u, v) are mutual
     nearest neighbours under the homography (purely geometric ground truth, descriptors unused),
  3. pick 60 well-spread correspondences by farthest-point sampling (deterministic start: the
     candidate nearest to the image-1 centroid of all candidates).
Held-out calibration pairs (img1 -> img4) are never used for testing.

Per trial: L inliers + (M-L) G-only + (N-L) H-only points are sampled disjointly from the 60
correspondences (so outliers are genuine SIFT detections without a partner). Each graph is the
Delaunay triangulation of its own points with Euclidean edge weights rescaled so that the median
edge length is 0.5 (scale invariance across zoom sequences). The unary cost is the L2 distance
between SIFT descriptors divided by its maximum. Objective: 0.5 * WCS + 0.5 * <C, X>.

Settings (Yang et al. 2018, Sec. VI-B): size sweep M in {20,25,30,35,40}, N = M+5, L = M-5;
outlier sweep M = 35, N = 40, M-L in {0,2,...,10}. 16 pairs x 4 samplings = 64 trials per setting.

Outputs: benchmarks/live_runs/<ts>_image/..., csv/image_results.csv, csv/image_trials.csv,
         json/image_results.json, json/image_calibration.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_common as lc  # noqa: E402
import numpy as np  # noqa: E402

DATA = lc.ROOT / "benchmarks" / "data" / "oxford_affine"
SEQUENCES = ["bark", "bikes", "boat", "graf", "leuven", "trees", "ubc", "wall"]
TEST_TARGETS = (2, 3)
CALIB_TARGETS = (4,)
N_GT = 60
ALPHA = 0.5
SAMPLINGS = 4
SEED_BASE = 30_261_016
SETTINGS = {
    "size": [dict(M=m, N=m + 5, L=m - 5, x=m) for m in (20, 25, 30, 35, 40)],
    "outliers": [dict(M=35, N=40, L=35 - o, x=o) for o in range(0, 11, 2)],
}
X_LABEL = {"size": "M", "outliers": "M-L"}


# ----------------------------------------------------------------------------------------- data
def _read_gray(path: Path):
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _image(seq, k):
    d = DATA / seq
    for ext in ("ppm", "pgm"):
        p = d / f"img{k}.{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(d / f"img{k}")


def build_pair(seq: str, k: int, n_gt: int = N_GT, tol_px: float = 2.0):
    """Geometric ground truth for (img1, imgk): returns dict with xy1, xy2, d1, d2 (n_gt rows) or None."""
    import cv2
    from scipy.spatial import cKDTree
    I1, I2 = _read_gray(_image(seq, 1)), _read_gray(_image(seq, k))
    sift = cv2.SIFT_create(nfeatures=4000)
    k1, d1 = sift.detectAndCompute(I1, None)
    k2, d2 = sift.detectAndCompute(I2, None)
    p1 = np.array([kp.pt for kp in k1], np.float64)
    p2 = np.array([kp.pt for kp in k2], np.float64)
    H = np.loadtxt(DATA / seq / f"H1to{k}p")
    q = np.c_[p1, np.ones(len(p1))] @ H.T
    proj = q[:, :2] / q[:, 2:3]
    inside = (proj[:, 0] >= 0) & (proj[:, 1] >= 0) & (proj[:, 0] < I2.shape[1]) & (proj[:, 1] < I2.shape[0])
    t2 = cKDTree(p2)
    dist, j = t2.query(proj)
    ok = inside & (dist < tol_px)
    # mutual nearest neighbour under H
    tp = cKDTree(proj)
    _, back = tp.query(p2[j])
    ok &= back == np.arange(len(p1))
    cand = np.flatnonzero(ok)
    # one correspondence per target keypoint
    _, first = np.unique(j[cand], return_index=True)
    cand = cand[np.sort(first)]
    if len(cand) < n_gt:
        return None
    pts = p1[cand]
    start = int(np.argmin(np.linalg.norm(pts - pts.mean(0), axis=1)))
    chosen = [start]
    dmin = np.linalg.norm(pts - pts[start], axis=1)
    for _ in range(n_gt - 1):
        nxt = int(np.argmax(dmin))
        chosen.append(nxt)
        dmin = np.minimum(dmin, np.linalg.norm(pts - pts[nxt], axis=1))
    sel = cand[np.array(chosen)]
    return dict(seq=seq, k=k, xy1=p1[sel], xy2=p2[j[sel]], d1=d1[sel].astype(np.float64), d2=d2[j[sel]].astype(np.float64),
                reproj_err_px=dist[sel], n_candidates=int(len(cand)))


def load_pairs(targets):
    cache = DATA / f"gt{N_GT}_targets{'-'.join(map(str, targets))}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return list(z["pairs"])
    pairs = []
    for s in SEQUENCES:
        for k in targets:
            p = build_pair(s, k)
            print(f"  GT {s:7s} 1->{k}: " + ("skipped (<60 verified correspondences)" if p is None else
                  f"{p['n_candidates']} verified, mean reprojection error {p['reproj_err_px'].mean():.2f}px"), flush=True)
            if p is not None:
                pairs.append(p)
    np.savez(cache, pairs=np.array(pairs, dtype=object))
    return pairs


def delaunay_graph(xy: np.ndarray) -> np.ndarray:
    from scipy.spatial import Delaunay
    n = len(xy)
    A = np.zeros((n, n))
    tri = Delaunay(xy).simplices
    for a, b in ((0, 1), (1, 2), (0, 2)):
        A[tri[:, a], tri[:, b]] = 1.0
    A = np.maximum(A, A.T)
    D = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    W = A * D
    med = np.median(W[W > 0])
    return W * (0.5 / med)


def make_image_instance(pair: dict, M: int, N: int, L: int, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N_GT)
    inl, g_only, h_only = perm[:L], perm[L:M], perm[M:M + N - L]
    g_idx = np.r_[inl, g_only]
    h_idx = np.r_[inl, h_only]
    pg, ph = rng.permutation(M), rng.permutation(N)          # hide the planted order
    g_idx, h_idx = g_idx[pg], h_idx[ph]
    AG = delaunay_graph(pair["xy1"][g_idx])
    AH = delaunay_graph(pair["xy2"][h_idx])
    C = np.linalg.norm(pair["d1"][g_idx][:, None] - pair["d2"][h_idx][None], axis=-1)
    C /= max(C.max(), 1e-12)
    Xgt = np.zeros((M, N), np.uint8)
    pos_h = {v: a for a, v in enumerate(h_idx)}
    for i, v in enumerate(g_idx):
        if v in inl:
            Xgt[i, pos_h[v]] = 1
    assert Xgt.sum() == L
    return AG, AH, C, Xgt


# -------------------------------------------------------------------------------------- solvers
def solver_table(cal, fcal):
    def pg(m):
        kw = {k: v for k, v in cal[m].items() if k in ("affinity", "sigma")}
        return lambda AG, AH, L, C: lc.solve_pygm(AG, AH, L, C, ALPHA, method=m, **kw)
    return {
        "Fast-WCS v3 (default)": lambda AG, AH, L, C: lc.solve_fastwcs(AG, AH, L, C, ALPHA),
        "Fast-WCS v3 (long path)": lambda AG, AH, L, C: lc.solve_fastwcs(AG, AH, L, C, ALPHA, **fcal),
        "RLX1 (Yang 2018)": lambda AG, AH, L, C: lc.solve_yang(AG, AH, L, C, ALPHA, "RLX1", "exact_lp"),
        "RLX2 (Yang 2018)": lambda AG, AH, L, C: lc.solve_yang(AG, AH, L, C, ALPHA, "RLX2", "exact_lp"),
        "RLX1F (Yang 2018)": lambda AG, AH, L, C: lc.solve_yang(AG, AH, L, C, ALPHA, "RLX1", "fast_heuristic"),
        "FAQ": lambda AG, AH, L, C: lc.solve_faq(AG, AH, L, C, ALPHA),
        "SM": pg("sm"),
        "IPFP": pg("ipfp"),
        "RRWM": pg("rrwm"),
        "NGM (pretrained)": pg("ngm"),
        "FGW (eps=1e-2)": lambda AG, AH, L, C: lc.solve_fgw(AG, AH, L, C, ALPHA, epsilon=1e-2),
        "FGW (eps=1e-3)": lambda AG, AH, L, C: lc.solve_fgw(AG, AH, L, C, ALPHA, epsilon=1e-3),
        "Partial FGW": lambda AG, AH, L, C: lc.solve_partial_gw(AG, AH, L, C, ALPHA),
    }


PYGM_GRID = [dict(affinity="wcs")] + [dict(affinity="gauss", sigma=s) for s in (0.05, 0.1, 0.2, 0.4, 0.8)]


def _calib_task(task):
    pi, j, m, opt = task
    pairs = load_pairs(CALIB_TARGETS)
    AG, AH, C, Xgt = make_image_instance(pairs[pi], 30, 35, 25, 9_100_000 + 100 * pi + j)
    X = lc.solve_pygm(AG, AH, 25, C, ALPHA, method=m, **opt)
    return dict(method=m, **opt, pair=pi, rep=j, acc=lc.accuracy_pct(X, Xgt, 25))


def calibrate(workers):
    """Held-out pairs img1->img4 only: choose each PyGMtools affinity among PYGM_GRID by mean accuracy."""
    pairs = load_pairs(CALIB_TARGETS)
    tasks = [(pi, j, m, opt) for pi in range(len(pairs)) for j in range(2) for m in ("sm", "rrwm", "ipfp", "ngm") for opt in PYGM_GRID]
    rows = [r for r in lc.run_pool(_calib_task, tasks, workers, desc="image-calibration", log_every=100)]
    chosen = {}
    for m in ("sm", "rrwm", "ipfp", "ngm"):
        best = None
        for opt in PYGM_GRID:
            acc = [r["acc"] for r in rows if r["method"] == m and all(r.get(k) == v for k, v in opt.items())
                   and (opt["affinity"] == "gauss" or "sigma" not in r)]
            mu = float(np.mean(acc))
            if best is None or mu > best[1] + 1e-9:
                best = (opt, mu)
        chosen[m] = dict(**best[0], heldout_acc_mean=best[1])
    out = {"heldout_pairs": [f"{p['seq']} 1->{p['k']}" for p in pairs], "grid": PYGM_GRID, "rows": rows, "chosen": chosen,
           "provenance": lc.provenance([Path(__file__)])}
    lc.write_json(lc.LIVE / "json" / "image_calibration.json", out)
    return chosen


# ---------------------------------------------------------------------------------------- trial
def trial(task):
    group, si, cfg, pi, s, methods, cal, fcal = task
    pair = load_pairs(TEST_TARGETS)[pi]
    seed = SEED_BASE + (0 if group == "size" else 1_000_000) + si * 10_000 + pi * 100 + s
    M, N, L = cfg["M"], cfg["N"], cfg["L"]
    AG, AH, C, Xgt = make_image_instance(pair, M, N, L, seed)
    F_gt = lc.wcs_objective(Xgt, AG, AH, C, ALPHA)
    table = solver_table(cal, fcal)
    lc.warmup_once({k: (lambda f: (lambda AG_, AH_, L_: f(AG_, AH_, L_, np.ones((AG_.shape[0], AH_.shape[0])))))(f)
                    for k, f in table.items()}, "image")
    order = [m for m in table if m in methods]
    np.random.default_rng(seed).shuffle(order)
    rows = []
    for m in order:
        np.random.seed(seed % (2 ** 32 - 1))
        X, ms, err = lc.timed_call(table[m], AG, AH, L, C)
        row = dict(group=group, setting=si, x=cfg["x"], M=M, N=N, L=L, pair=f"{pair['seq']} 1->{pair['k']}", sampling=s,
                   seed=seed, method=m, ms=ms, F_gt=F_gt, error=err)
        if err is None:
            try:
                lc.validate_exact_L(X, L)
                F = lc.wcs_objective(X, AG, AH, C, ALPHA)
                row.update(acc=lc.accuracy_pct(X, Xgt, L), F=F, F_ratio=F / max(F_gt, 1e-12), F_le_gt=int(F <= F_gt + 1e-9))
            except ValueError as e:
                row["error"] = f"invalid output: {e}"
        rows.append(row)
    return rows


def aggregate(rows):
    out = []
    keys = sorted({(r["group"], r["setting"], r["method"]) for r in rows})
    for g, si, m in keys:
        rs = [r for r in rows if r["group"] == g and r["setting"] == si and r["method"] == m]
        ok = [r for r in rs if r.get("error") is None]
        acc = [r["acc"] for r in ok]
        lo, hi = lc.mean_ci95(acc, seed=si)
        s_acc, s_ms = lc.summarize(acc), lc.summarize([r["ms"] for r in ok])
        r0 = rs[0]
        out.append(dict(group=g, x_name=X_LABEL[g], x=r0["x"], M=r0["M"], N=r0["N"], L=r0["L"], method=m, n_trials=len(rs),
                        n_ok=len(ok), n_errors=len(rs) - len(ok), acc_mean=s_acc["mean"], acc_sd=s_acc["sd"],
                        acc_ci95_lo=lo, acc_ci95_hi=hi, F_ratio_median=lc.summarize([r["F_ratio"] for r in ok])["median"],
                        frac_F_le_gt=float(np.mean([r["F_le_gt"] for r in ok])) if ok else None,
                        ms_median=s_ms["median"], ms_q25=s_ms["q25"], ms_q75=s_ms["q75"], ms_mean=s_ms["mean"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samplings", type=int, default=SAMPLINGS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--methods", default="all")
    ap.add_argument("--groups", default="size,outliers")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    name = "image" + a.tag
    out_dir = lc.run_dir(name)
    sys.stdout = lc.Tee(out_dir / "run.log")
    print("Script 2 | building geometric ground truth (test pairs img1->img2, img1->img3)", flush=True)
    test_pairs = load_pairs(TEST_TARGETS)
    print("held-out calibration pairs (img1->img4)", flush=True)
    load_pairs(CALIB_TARGETS)
    cal_path = lc.LIVE / "json" / "image_calibration.json"
    cal = json.loads(cal_path.read_text())["chosen"] if cal_path.exists() else calibrate(a.workers)
    fcal = json.loads((lc.LIVE / "json" / "fastwcs_calibration.json").read_text())["long_path"]["config"]
    methods = list(solver_table(cal, fcal)) if a.methods == "all" else a.methods.split(",")
    print(f"test pairs ({len(test_pairs)}): {[p['seq'] + ' 1->' + str(p['k']) for p in test_pairs]}\n"
          f"held-out PyGMtools affinities: {cal}\nFast-WCS long path: {fcal}\nmethods: {methods}", flush=True)
    tasks = [(g, si, cfg, pi, s, methods, cal, fcal) for g in a.groups.split(",") for si, cfg in enumerate(SETTINGS[g])
             for pi in range(len(test_pairs)) for s in range(a.samplings)]
    tasks.sort(key=lambda z: -(z[2]["M"] * z[2]["N"]))
    rows = [r for rs in lc.run_pool(trial, tasks, workers=a.workers, desc=name, log_every=32) for r in rs]
    agg = aggregate(rows)
    fmt = lambda v: "   n/a" if v is None else f"{v:6.1f}"
    for r in agg:
        print(f"{r['group']:8s} {r['x_name']}={r['x']:<3} {r['method']:24s} acc {fmt(r['acc_mean'])} ms {fmt(r['ms_median'])} "
              f"err {r['n_errors']}", flush=True)
    payload = {"script": Path(__file__).name, "args": vars(a), "settings": SETTINGS, "alpha": ALPHA, "seed_base": SEED_BASE,
               "test_pairs": [f"{p['seq']} 1->{p['k']}" for p in test_pairs],
               "gt_mean_reprojection_px": float(np.mean([p["reproj_err_px"].mean() for p in test_pairs])),
               "calibrated_baselines": cal, "fastwcs_long_path": fcal, "yang_settings": lc.YANG_SETTINGS,
               "aggregated": agg, "provenance": lc.provenance([Path(__file__)])}
    lc.write_csv(out_dir / "trials.csv", rows)
    lc.write_csv(out_dir / "aggregated.csv", agg)
    lc.write_json(out_dir / f"{name}_results.json", payload)
    lc.write_csv(lc.LIVE / "csv" / f"{name}_results.csv", agg)
    lc.write_csv(lc.LIVE / "csv" / f"{name}_trials.csv", rows)
    lc.write_json(lc.LIVE / "json" / f"{name}_results.json", payload)
    print("wrote", lc.LIVE / "csv" / f"{name}_results.csv", flush=True)


if __name__ == "__main__":
    main()
