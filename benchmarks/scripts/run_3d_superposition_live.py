"""Script 5 - live 3D structural superposition on untrimmed full-atom Cartesian coordinates.

Instances (per trial, all methods see the identical pair):
  * a drug-like molecule from MOLECULES, all-atom (explicit H) 3D conformer from RDKit ETKDGv3 (seeded);
  * a contiguous patch of K = M + N - L atoms (the K atoms nearest to a random centre atom) is split at
    random into L core atoms (present in both clouds), M - L G-only and N - L H-only outlier atoms;
  * cloud H = R* (x + e) + t*, with R* uniform on SO(3), t* ~ U[-10, 10]^3 Angstrom and isotropic
    Gaussian deformation noise e ~ N(0, sigma^2 I) per atom; both clouds are randomly permuted.
Scenarios (100 trials each):
  S1 near-rigid      sigma = 0.02 A, M = N = 40, L = 36  (10 % outliers per cloud)
  S2 deformation     sigma = 0.25 A, M = N = 40, L = 36
  S3 outlier-heavy   sigma = 0.05 A, M = N = 46, L = 36  (22 % outliers per cloud)

Every method returns a rigid transform (R, t) mapping G into H's frame. Reported RMSD is the UNTRIMMED
RMSD over all L ground-truth core pairs under that transform (independent of which pairs a method
matched); "oracle RMSD" is the Kabsch optimum on the true pairs (the noise floor). Matching methods
also report the fraction of correctly matched core pairs.

Methods
  Fast-WCS v3 hierarchical       AdvancedGNCCP().match_point_clouds (RBF distance-geometry graph match ->
                                 Kabsch -> progressive assignment/Kabsch polish), library defaults
  Fast-WCS v3 (graph stage only) same with n_dgm_iters = 1 (ablation of the spatial polish)
  FAQ / SM / RRWM / IPFP + Kabsch  same exact-L projection as Scripts 1-2 and untrimmed Kabsch on the
                                 matched pairs; FAQ on the RBF adjacency, PyGMtools on the Gaussian
                                 distance affinity with sigma chosen on held-out trials
  FGW (eps=1e-2) / Partial GW + Kabsch   POT on the (max-normalised) intra-cloud distance matrices
  ICP (single start)             Open3D point-to-point ICP from centroid alignment, max corr. 2 A
  ICP (24-start)                 the 24 proper rotations of the cube as initialisations, best fitness
  RANSAC + FPFH + ICP            Open3D global registration (FPFH 3 A, mutual filter) + ICP refinement
  CPD (rigid)                    pycpd rigid CPD (no scaling), centroid initialisation
Metrics: success < 1.0 A, ultra-precision < 0.1 A, median RMSD, runtime (ms), det(R) = +1 rate and the
rate at which the SVD optimum was a reflection that Kabsch corrected.

Outputs: csv/superposition_3d_results.csv, csv/superposition_3d_trials.csv,
         json/superposition_3d_results.json, json/superposition_3d_calibration.json, <ts>_superposition_3d/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_common as lc  # noqa: E402
import numpy as np  # noqa: E402

from gnccp_advanced_v3 import AdvancedGNCCP, robust_kabsch_superposition, _pairwise_euclidean  # noqa: E402

SEED_BASE = 60_261_016
MOLECULES = {
    "imatinib": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "atorvastatin": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O",
    "sildenafil": "CCCc1nn(C)c2c(=O)[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc12",
    "tamoxifen": "CC/C(=C(\\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
    "losartan": "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1",
    "donepezil": "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2",
    "raloxifene": "Oc1ccc(cc1)-c1sc2cc(O)ccc2c1C(=O)c1ccc(OCCN2CCCCC2)cc1",
    "quetiapine": "OCCOCCN1CCN(CC1)C1=Nc2ccccc2Sc2ccccc12",
    "cetirizine": "OC(=O)COCCN1CCN(CC1)C(c1ccccc1)c1ccc(Cl)cc1",
    "verapamil": "COc1ccc(CCN(C)CCCC(C#N)(C(C)C)c2ccc(OC)c(OC)c2)cc1OC",
    "simvastatin": "CCC(C)(C)C(=O)O[C@H]1C[C@@H](C)C=C2C=C[C@H](C)[C@H](CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@@H]21",
    "erlotinib": "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
    "gefitinib": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "haloperidol": "OC1(CCN(CCCC(=O)c2ccc(F)cc2)CC1)c1ccc(Cl)cc1",
    "diltiazem": "COc1ccc(cc1)[C@@H]1Sc2ccccc2N(CCN(C)C)C(=O)[C@@H]1OC(C)=O",
    "celecoxib": "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
    "olanzapine": "CC1=CC2=C(S1)NC1=CC=CC=C1N=C2N1CCN(C)CC1",
    "clopidogrel": "COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1",
    "lisinopril": "NCCCC[C@H](N[C@@H](CCc1ccccc1)C(=O)O)C(=O)N1CCC[C@H]1C(=O)O",
    "ranolazine": "COc1ccccc1OCC(O)CN1CCN(CC(=O)Nc2c(C)cccc2C)CC1",
}
SCENARIOS = {
    "S1_near_rigid": dict(sigma=0.02, M=40, N=40, L=36),
    "S2_deformation": dict(sigma=0.25, M=40, N=40, L=36),
    "S3_outlier_heavy": dict(sigma=0.05, M=46, N=46, L=36),
}
_CONF_CACHE = {}


# ------------------------------------------------------------------------------------- instances
def conformer(name: str, seed: int) -> np.ndarray:
    key = (name, seed)
    if key not in _CONF_CACHE:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.AddHs(Chem.MolFromSmiles(MOLECULES[name]))
        p = AllChem.ETKDGv3(); p.randomSeed = int(seed % (2 ** 31 - 1))
        if AllChem.EmbedMolecule(mol, p) != 0:
            raise RuntimeError(f"ETKDG failed for {name}")
        _CONF_CACHE[key] = mol.GetConformer().GetPositions().astype(np.float64)
    return _CONF_CACHE[key]


def n_atoms(name: str) -> int:
    from rdkit import Chem
    return Chem.AddHs(Chem.MolFromSmiles(MOLECULES[name])).GetNumAtoms()


def random_rotation(rng) -> np.ndarray:
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def make_instance(scen: str, t: int, mol_names, heldout=False):
    cfg = SCENARIOS[scen]
    M, N, L, sigma = cfg["M"], cfg["N"], cfg["L"], cfg["sigma"]
    seed = (9_100_000 if heldout else SEED_BASE) + list(SCENARIOS).index(scen) * 100_000 + t
    rng = np.random.default_rng(seed)
    K = M + N - L
    ok = [m for m in mol_names if n_atoms(m) >= K]
    name = ok[t % len(ok)]
    X = conformer(name, seed)
    centre = X[rng.integers(len(X))]
    patch = np.argsort(np.linalg.norm(X - centre, axis=1))[:K]
    patch = rng.permutation(patch)
    core, g_only, h_only = patch[:L], patch[L:M], patch[M:K]
    g_atoms = np.r_[core, g_only]
    h_atoms = np.r_[core, h_only]
    G = X[g_atoms].copy()
    Rs, ts = random_rotation(rng), rng.uniform(-10, 10, 3)
    H = (X[h_atoms] + rng.normal(0.0, sigma, (N, 3))) @ Rs.T + ts
    pg, ph = rng.permutation(M), rng.permutation(N)
    G, H = G[pg], H[ph]
    inv_g, inv_h = np.argsort(pg), np.argsort(ph)
    gt = np.c_[inv_g[np.arange(L)], inv_h[np.arange(L)]]       # (row in G, row in H) of every core atom
    return dict(G=G, H=H, gt=gt, M=M, N=N, L=L, sigma=sigma, molecule=name, seed=seed, R_true=Rs)


def rmsd_on_gt(R, t, inst):
    P = inst["G"][inst["gt"][:, 0]] @ np.asarray(R).T + np.asarray(t)
    return float(np.sqrt(np.mean(np.sum((P - inst["H"][inst["gt"][:, 1]]) ** 2, axis=1))))


# ---------------------------------------------------------------------------------------- methods
def _rbf(D, scale=0.85):
    s = max(float(np.median(D[D > 0])) * scale, 1e-3)
    A = np.exp(-0.5 * (D / s) ** 2); np.fill_diagonal(A, 0.0)
    return A


def _kabsch_from_X(X, G, H):
    r, c = np.nonzero(X)
    aln = robust_kabsch_superposition(G[r], H[c])
    return dict(R=aln["R"], t=aln["t"], reflection_corrected=aln["reflection_corrected"], pairs=np.c_[r, c])


def m_fastwcs(G, H, L, cal, dgm=4):
    res = AdvancedGNCCP().match_point_clouds(G, H, L=L, n_dgm_iters=dgm)
    a = res.alignment
    return dict(R=a["R"], t=a["t"], reflection_corrected=a["reflection_corrected"], pairs=np.array(res.matches))


def m_faq(G, H, L, cal):
    DG, DH = _pairwise_euclidean(G, G), _pairwise_euclidean(H, H)
    AG, AH = _rbf(DG).astype(np.float32), _rbf(DH).astype(np.float32)
    return _kabsch_from_X(lc.solve_faq(AG, AH, L), G, H)


def m_pygm(method):
    def f(G, H, L, cal):
        DG, DH = _pairwise_euclidean(G, G).astype(np.float32), _pairwise_euclidean(H, H).astype(np.float32)
        X = lc.solve_pygm(DG, DH, L, method=method, affinity="gauss", sigma=cal[method])
        return _kabsch_from_X(X, G, H)
    return f


def m_ot(kind):
    def f(G, H, L, cal):
        DG, DH = _pairwise_euclidean(G, G), _pairwise_euclidean(H, H)
        X = lc.solve_fgw(DG, DH, L, epsilon=1e-2) if kind == "fgw" else lc.solve_partial_gw(DG, DH, L)
        return _kabsch_from_X(X, G, H)
    return f


def _o3d(P):
    import open3d as o3d
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(P, np.float64)))


def _icp(G, H, T0, max_corr=2.0):
    import open3d as o3d
    reg = o3d.pipelines.registration
    return reg.registration_icp(_o3d(G), _o3d(H), max_corr, T0, reg.TransformationEstimationPointToPoint(),
                                reg.ICPConvergenceCriteria(max_iteration=100))


def _T_centroid(G, H, R=np.eye(3)):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = H.mean(0) - G.mean(0) @ R.T
    return T


def m_icp(G, H, L, cal):
    T = np.asarray(_icp(G, H, _T_centroid(G, H)).transformation)
    return dict(R=T[:3, :3], t=T[:3, 3], reflection_corrected=False)


_CUBE = None


def _cube_rotations():
    global _CUBE
    if _CUBE is None:
        import itertools
        rots = []
        for perm in itertools.permutations(range(3)):
            for signs in itertools.product((1, -1), repeat=3):
                R = np.zeros((3, 3))
                for i, j in enumerate(perm):
                    R[i, j] = signs[i]
                if np.linalg.det(R) > 0:
                    rots.append(R)
        _CUBE = rots
    return _CUBE


def m_icp_multi(G, H, L, cal):
    best = None
    for R0 in _cube_rotations():
        r = _icp(G, H, _T_centroid(G, H, R0))
        key = (r.fitness, -r.inlier_rmse)
        if best is None or key > best[0]:
            best = (key, np.asarray(r.transformation))
    T = best[1]
    return dict(R=T[:3, :3], t=T[:3, 3], reflection_corrected=False)


def m_ransac(G, H, L, cal, seed=0):
    import open3d as o3d
    reg = o3d.pipelines.registration
    o3d.utility.random.seed(int(seed % (2 ** 31 - 1)))
    src, tgt = _o3d(G), _o3d(H)
    feats = []
    for pc in (src, tgt):
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=30))
        feats.append(reg.compute_fpfh_feature(pc, o3d.geometry.KDTreeSearchParamHybrid(radius=3.0, max_nn=100)))
    r = reg.registration_ransac_based_on_feature_matching(
        src, tgt, feats[0], feats[1], True, 1.0, reg.TransformationEstimationPointToPoint(False), 3,
        [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(1.0)],
        reg.RANSACConvergenceCriteria(100000, 0.999))
    T = np.asarray(_icp(G, H, np.asarray(r.transformation)).transformation)
    return dict(R=T[:3, :3], t=T[:3, 3], reflection_corrected=False)


def m_cpd(G, H, L, cal):
    from pycpd import RigidRegistration
    c = G.mean(0) - H.mean(0)
    reg = RigidRegistration(X=H, Y=G - c, max_iterations=150, tolerance=1e-8, scale=False)
    reg.register()
    s, Rr, tt = reg.get_registration_parameters()      # TY = s * Y @ Rr + tt
    R = np.asarray(Rr).T
    return dict(R=R, t=np.asarray(tt).ravel() - c @ R.T, reflection_corrected=False)


def method_table():
    return {
        "Fast-WCS v3 hierarchical": m_fastwcs,
        "Fast-WCS v3 (graph stage only)": lambda G, H, L, cal: m_fastwcs(G, H, L, cal, dgm=1),
        "FAQ + Kabsch": m_faq,
        "SM + Kabsch": m_pygm("sm"),
        "RRWM + Kabsch": m_pygm("rrwm"),
        "IPFP + Kabsch": m_pygm("ipfp"),
        "FGW (eps=1e-2) + Kabsch": m_ot("fgw"),
        "Partial GW + Kabsch": m_ot("pgw"),
        "ICP (single start)": m_icp,
        "ICP (24-start)": m_icp_multi,
        "RANSAC + FPFH + ICP": m_ransac,
        "CPD (rigid)": m_cpd,
    }


# ------------------------------------------------------------------------------------------- runs
SIGMA_GRID = (0.1, 0.25, 0.5, 1.0, 2.0)


def _calib_task(task):
    m, s, t = task
    inst = make_instance("S2_deformation", t, list(MOLECULES), heldout=True)
    out = m_pygm(m)(inst["G"], inst["H"], inst["L"], {m: s})
    gt = {tuple(p) for p in inst["gt"].tolist()}
    return dict(method=m, sigma=s, trial=t, acc=100.0 * sum(tuple(p) in gt for p in out["pairs"].tolist()) / inst["L"],
                rmsd=rmsd_on_gt(out["R"], out["t"], inst))


def calibrate(workers, n=10):
    tasks = [(m, s, t) for m in ("sm", "rrwm", "ipfp") for s in SIGMA_GRID for t in range(n)]
    rows = lc.run_pool(_calib_task, tasks, workers, desc="3d-calibration", log_every=50)
    chosen = {}
    for m in ("sm", "rrwm", "ipfp"):
        scores = {s: np.mean([r["acc"] for r in rows if r["method"] == m and r["sigma"] == s]) for s in SIGMA_GRID}
        chosen[m] = float(max(scores, key=lambda s: (scores[s], -s)))
    lc.write_json(lc.LIVE / "json" / "superposition_3d_calibration.json",
                  {"grid_angstrom": SIGMA_GRID, "heldout_trials": n, "scenario": "S2_deformation (held-out seeds)",
                   "rows": rows, "chosen_sigma": chosen, "provenance": lc.provenance([Path(__file__)])})
    return chosen


def trial(task):
    scen, t, methods, cal = task
    inst = make_instance(scen, t, list(MOLECULES))
    G, H, L = inst["G"], inst["H"], inst["L"]
    oracle = robust_kabsch_superposition(G[inst["gt"][:, 0]], H[inst["gt"][:, 1]])
    floor = rmsd_on_gt(oracle["R"], oracle["t"], inst)
    table = method_table()
    if not lc._WARMED:
        w = make_instance("S1_near_rigid", 0, list(MOLECULES), heldout=True)
        for f in table.values():
            try:
                f(w["G"], w["H"], w["L"], cal)
            except Exception:
                pass
        lc._WARMED.add("3d")
    order = [m for m in table if m in methods]
    np.random.default_rng(inst["seed"]).shuffle(order)
    gt = {tuple(p) for p in inst["gt"].tolist()}
    rows = []
    for m in order:
        fn = table[m]
        if m.startswith("RANSAC"):
            out, ms, err = lc.timed_call(fn, G, H, L, cal, seed=inst["seed"])
        else:
            out, ms, err = lc.timed_call(fn, G, H, L, cal)
        row = dict(scenario=scen, trial=t, seed=inst["seed"], molecule=inst["molecule"], M=inst["M"], N=inst["N"], L=L,
                   sigma=inst["sigma"], method=m, ms=ms, oracle_rmsd=floor, error=err)
        if err is None:
            R = np.asarray(out["R"])
            row.update(rmsd=rmsd_on_gt(R, out["t"], inst), det_R=float(np.linalg.det(R)),
                       proper_rotation=int(abs(np.linalg.det(R) - 1.0) < 1e-6 and np.allclose(R.T @ R, np.eye(3), atol=1e-6)),
                       reflection_corrected=int(bool(out.get("reflection_corrected", False))),
                       rot_err_deg=float(np.degrees(np.arccos(np.clip((np.trace(R.T @ inst["R_true"]) - 1) / 2, -1, 1)))))
            if "pairs" in out:
                row["match_acc"] = 100.0 * sum(tuple(p) in gt for p in np.asarray(out["pairs"]).tolist()) / L
        rows.append(row)
    return rows


def aggregate(rows):
    out = []
    for scen in SCENARIOS:
        for m in method_table():
            rs = [r for r in rows if r["scenario"] == scen and r["method"] == m]
            if not rs:
                continue
            ok = [r for r in rs if r.get("error") is None]
            rm = np.array([r["rmsd"] for r in ok]) if ok else np.array([np.nan])
            ms = np.array([r["ms"] for r in ok]) if ok else np.array([np.nan])
            d = dict(scenario=scen, sigma=rs[0]["sigma"], M=rs[0]["M"], N=rs[0]["N"], L=rs[0]["L"], method=m, n_trials=len(rs),
                     n_errors=len(rs) - len(ok),
                     success_lt_1A_pct=100.0 * float(np.sum(rm < 1.0)) / len(rs),
                     success_lt_0p1A_pct=100.0 * float(np.sum(rm < 0.1)) / len(rs),
                     rmsd_median=float(np.median(rm)), rmsd_q25=float(np.percentile(rm, 25)), rmsd_q75=float(np.percentile(rm, 75)),
                     oracle_rmsd_median=float(np.median([r["oracle_rmsd"] for r in rs])),
                     ms_median=float(np.median(ms)), ms_q25=float(np.percentile(ms, 25)), ms_q75=float(np.percentile(ms, 75)),
                     det_plus1_pct=100.0 * float(np.mean([r["proper_rotation"] for r in ok])) if ok else None,
                     reflection_corrected_pct=100.0 * float(np.mean([r["reflection_corrected"] for r in ok])) if ok else None,
                     match_acc_mean=float(np.mean([r["match_acc"] for r in ok])) if ok and "match_acc" in ok[0] else None)
            lo, hi = lc.mean_ci95([float(r["rmsd"] < 1.0) * 100 for r in ok], seed=1) if ok else (None, None)
            d["success_lt_1A_ci95_lo"], d["success_lt_1A_ci95_hi"] = lo, hi
            out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--methods", default="all")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    name = "superposition_3d" + a.tag
    out_dir = lc.run_dir(name)
    sys.stdout = lc.Tee(out_dir / "run.log")
    cal_path = lc.LIVE / "json" / "superposition_3d_calibration.json"
    cal = json.loads(cal_path.read_text())["chosen_sigma"] if cal_path.exists() else calibrate(a.workers)
    methods = list(method_table()) if a.methods == "all" else a.methods.split(",")
    print(f"Script 5 | trials={a.trials} scenarios={a.scenarios} | held-out PyGMtools sigma (A): {cal}\nmethods: {methods}", flush=True)
    tasks = [(s, t, methods, cal) for s in a.scenarios.split(",") for t in range(a.trials)]
    rows = [r for rs in lc.run_pool(trial, tasks, workers=a.workers, desc=name, log_every=25) for r in rs]
    agg = aggregate(rows)
    for d in agg:
        print(f"{d['scenario']:17s} {d['method']:32s} <1A {d['success_lt_1A_pct']:5.1f}%  <0.1A {d['success_lt_0p1A_pct']:5.1f}%  "
              f"med RMSD {d['rmsd_median']:7.3f}  {d['ms_median']:8.1f} ms  det+1 {d['det_plus1_pct']}", flush=True)
    payload = {"script": Path(__file__).name, "args": vars(a), "scenarios": SCENARIOS, "molecules": MOLECULES,
               "seed_base": SEED_BASE, "pygm_sigma": cal, "aggregated": agg, "provenance": lc.provenance([Path(__file__)])}
    lc.write_csv(out_dir / "trials.csv", rows)
    lc.write_csv(out_dir / "aggregated.csv", agg)
    lc.write_json(out_dir / f"{name}_results.json", payload)
    lc.write_csv(lc.LIVE / "csv" / f"{name}_results.csv", agg)
    lc.write_csv(lc.LIVE / "csv" / f"{name}_trials.csv", rows)
    lc.write_json(lc.LIVE / "json" / f"{name}_results.json", payload)
    print("wrote", lc.LIVE / "csv" / f"{name}_results.csv", flush=True)


if __name__ == "__main__":
    main()
