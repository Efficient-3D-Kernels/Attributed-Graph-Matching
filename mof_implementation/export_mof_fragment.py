"""export_mof_fragment.py -- creates mof_fragment.js (one real MOF conformer alignment) for the 3D viewer in mof.html.
Run this inside your mof_implementation folder (it imports your fastwcs.py / mof_data.py):
  python export_mof_fragment.py --cif_dir "data/CoREMOF2019_public_v2_20241118/CR/ASR" --seed 0
Then copy the generated mof_fragment.js next to mof.html.  Use --index to look at a different MOF."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
import fastwcs as fw, mof_data as md


def kabsch_transform(P, Q):
    pm, qm = P.mean(0), Q.mean(0)
    H = (P - pm).T @ (Q - qm)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, pm, qm


def rand_rot(rng):
    q, r = np.linalg.qr(rng.standard_normal((3, 3))); q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0: q[:, 0] *= -1
    return q


ap = argparse.ArgumentParser()
ap.add_argument("--cif_dir", required=True); ap.add_argument("--filter", default=None)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--index", type=int, default=0)
ap.add_argument("--M", type=int, default=24); ap.add_argument("--sigma", type=float, default=0.05)
ap.add_argument("--starts", type=int, default=10); ap.add_argument("--out", default="mof_fragment.js")
a = ap.parse_args()
M, N, L = a.M, a.M + 4, a.M - 2
paths = md.find_cifs(a.cif_dir, a.filter); np.random.default_rng(a.seed).shuffle(paths)
k = -1
for p in paths:
    try:
        at = md.load_structure(p); md.local_cluster(at, 0, 80)
    except Exception:
        continue
    k += 1
    if k < a.index: continue
    rng = np.random.default_rng([a.seed, k])
    inst = md.build_mof_instance(at, M, N, L, a.sigma, 50.0, rng)
    Rm, t = rand_rot(rng), rng.uniform(-20, 20, 3)
    inst["posG"] = inst["posG"] @ Rm.T + t
    res = fw.fastwcs(inst["AG"], inst["AH"], inst["C"], L, alpha=0.7, starts=a.starts, seed=a.seed)
    rows, cols = res["rows"], res["cols"]
    R, pm, qm = kabsch_transform(inst["posG"][rows], inst["posH"][cols])
    G = (R @ (inst["posG"] - pm).T).T + qm                         # query superposed onto target
    c = inst["posH"].mean(0); G -= c; Hh = inst["posH"] - c
    rmsd = float(np.sqrt(np.mean(np.sum((G[rows] - Hh[cols]) ** 2, 1))))
    ref = fw.kabsch_rmsd(inst["posG"][inst["gt_rows"]], inst["posH"][inst["gt_cols"]])
    out = dict(name=os.path.basename(p), M=M, N=N, L=L, sigma=a.sigma, rmsd=rmsd, rmsd_planted=float(ref),
               Zg=inst["ZG"].tolist(), Zh=inst["ZH"].tolist(),
               posG=np.round(G, 3).tolist(), posH=np.round(Hh, 3).tolist(),
               pairs=[[int(r), int(c_)] for r, c_ in zip(rows, cols)],
               gt=[[int(r), int(c_)] for r, c_ in zip(inst["gt_rows"], inst["gt_cols"])])
    open(a.out, "w").write("window.MOF_FRAGMENT = " + json.dumps(out) + ";\n")
    ok = len(set(map(tuple, out["pairs"])) & set(map(tuple, out["gt"])))
    print(f"wrote {a.out}: {out['name']}  correct {ok}/{L}  RMSD {rmsd:.3f} A (planted ref {ref:.3f} A)")
    break
else:
    print("no usable structure found")
