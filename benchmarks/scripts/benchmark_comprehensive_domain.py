"""
benchmark_comprehensive_domain.py
========================================================================================
UNIFIED GRAPH MATCHING BENCHMARK: FAST-WCS V3 VS. BASE PAPER (2018) & POST-2018 SOLVERS
========================================================================================
Evaluates across two standardized domains:
  1. Synthetic Subgraph Matching (Yang et al., IEEE TNNLS 2018 Protocol)
  2. Real 3D Molecular Conformer Superposition (CYP19A1 Active Ligands, Full-Atom Kabsch)

Solvers Benchmarked:
  * Our Flagship Method:
      - Fast-WCS v3 (C:\\Users\\abdus\\Desktop\\Fast-WCS-Research-Deliverables\\gnccp_advanced_v3.py)
  * Base Paper (Yang et al., 2018) Models:
      - GNCCP-2018 Baseline (RLX1F top-L truncation heuristic)
      - Spectral Matching (SM, Cour et al. 2007 via PyGMtools)
      - Integer Projected Fixed Point (IPFP, Leordeanu et al. 2009 via PyGMtools)
      - Reweighted Random Walks Matching (RRWM, Cho et al. 2010 via PyGMtools)
  * Post-2018 Modern Competitors:
      - GRAMPA (Fan et al., ICML 2020)
      - Fused Gromov-Wasserstein (FGW, Titouan et al., ICML 2019 / NeurIPS 2019 via POT)
      - Partial Optimal Transport (POT, Chapel et al., NeurIPS 2020 via POT)
      - SciPy FAQ + Level-3 BLAS 2-Opt Refinement (Vogelstein et al. 2015 + Modern 2-Opt)
      - SciPy FAQ (Standard QAP)
========================================================================================
"""

import os
import sys
import time
import json
import warnings
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment, quadratic_assignment
from scipy.spatial.transform import Rotation
from rdkit import Chem

warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------------------
# Paths and Solver Imports
# --------------------------------------------------------------------------------------
DELIVERABLES_DIR = Path(r"C:\Users\abdus\Desktop\Fast-WCS-Research-Deliverables")
sys.path.insert(0, str(DELIVERABLES_DIR))

from gnccp_advanced_v3 import AdvancedGNCCP, robust_kabsch_superposition, kabsch
from gnccp_baseline_2018 import OriginalGNCCP_2018
from gnccp_v2_classic import UniversalGNCCP

import ot
import pygmtools as pygm


# --------------------------------------------------------------------------------------
# 1. GRAMPA Spectral Solver (Fan et al., ICML 2020)
# --------------------------------------------------------------------------------------
def grampa_match(A_G: np.ndarray, A_H: np.ndarray, L: int, eta: float = 0.10) -> np.ndarray:
    """GRAMPA spectral score with exact-L assignment projection."""
    M, N = A_G.shape[0], A_H.shape[0]
    lam, U = np.linalg.eigh(A_G.astype(np.float64))
    mu, V = np.linalg.eigh(A_H.astype(np.float64))
    denom = (lam[:, None] - mu[None, :]) ** 2 + eta ** 2
    coeff = (1.0 / denom) * (U.T @ np.ones((M, N)) @ V)
    scores = U @ coeff @ V.T
    
    # Exact-L linear assignment
    rows, cols = linear_sum_assignment(-scores)
    order = np.argsort(-scores[rows, cols])[:L]
    return list(zip(rows[order].tolist(), cols[order].tolist()))


# --------------------------------------------------------------------------------------
# 2. PyGMtools Solvers (SM, IPFP, RRWM) with Lawler Affinity & Exact-L
# --------------------------------------------------------------------------------------
def pygm_lawler_match(A_G: np.ndarray, A_H: np.ndarray, L: int, solver_func) -> List[Tuple[int, int]]:
    """Evaluates PyGMtools solvers with Lawler affinity and exact-L projection."""
    M, N = A_G.shape[0], A_H.shape[0]
    source = np.repeat(np.arange(M), N)
    target = np.tile(np.arange(N), M)
    diff = A_G[source[:, None], source[None, :]] - A_H[target[:, None], target[None, :]]
    cost = diff * diff
    K = (float(np.max(cost)) - cost).astype(np.float32)
    
    scores = solver_func(K, n1=M, n2=N, max_iter=40, backend="numpy")
    scores = np.asarray(scores).reshape(M, N)
    
    rows, cols = linear_sum_assignment(-scores)
    order = np.argsort(-scores[rows, cols])[:L]
    return list(zip(rows[order].tolist(), cols[order].tolist()))


# --------------------------------------------------------------------------------------
# 3. FGW & POT Solvers (Optimal Transport)
# --------------------------------------------------------------------------------------
def fgw_match(A_G: np.ndarray, A_H: np.ndarray, C: np.ndarray, L: int, alpha: float = 0.5) -> List[Tuple[int, int]]:
    M, N = A_G.shape[0], A_H.shape[0]
    p = np.ones(M) / M
    q = np.ones(N) / N
    C_norm = C / max(C.max(), 1e-6)
    T = ot.gromov.fused_gromov_wasserstein(C_norm, A_G.astype(np.float64), A_H.astype(np.float64), p, q, alpha=alpha)
    rows, cols = linear_sum_assignment(-T)
    order = np.argsort(-T[rows, cols])[:L]
    return list(zip(rows[order].tolist(), cols[order].tolist()))


def pot_match(C: np.ndarray, L: int) -> List[Tuple[int, int]]:
    M, N = C.shape
    p = np.ones(M) / M
    q = np.ones(N) / N
    C_norm = C / max(C.max(), 1e-6)
    mass = float(L) / min(M, N)
    T = ot.partial.partial_wasserstein(p, q, C_norm, m=mass)
    rows, cols = linear_sum_assignment(-T)
    order = np.argsort(-T[rows, cols])[:L]
    return list(zip(rows[order].tolist(), cols[order].tolist()))


# --------------------------------------------------------------------------------------
# 4. SciPy FAQ Solvers
# --------------------------------------------------------------------------------------
def scipy_faq_match(A_G: np.ndarray, A_H: np.ndarray, L: int, with_2opt: bool = False, seed: int = 42) -> List[Tuple[int, int]]:
    M, N = A_G.shape[0], A_H.shape[0]
    dim = max(M, N)
    A_G_pad = np.zeros((dim, dim), dtype=np.float64)
    A_H_pad = np.zeros((dim, dim), dtype=np.float64)
    A_G_pad[:M, :M] = A_G
    A_H_pad[:N, :N] = A_H
    
    res = quadratic_assignment(A_G_pad, A_H_pad, method='faq', options={'maxiter': 25, 'rng': seed})
    perm = res.col_ind[:M]
    
    if not with_2opt:
        pairs = [(g, int(perm[g])) for g in range(M) if perm[g] < N]
        return pairs[:L]
    
    # Apply Level-3 BLAS 2-opt refinement
    X_bin = np.zeros((M, N), dtype=np.float32)
    for g, h in enumerate(perm):
        if h < N:
            X_bin[g, h] = 1.0
            
    solver_ref = AdvancedGNCCP(seed=seed)
    X_ref = solver_ref._swap_refinement(X_bin, A_G.astype(np.float32), A_H.astype(np.float32), np.zeros((M, N), dtype=np.float32), L=L, max_rounds=15)
    mr, mc = np.where(X_ref > 0.5)
    return list(zip(mr.tolist(), mc.tolist()))[:L]


# ======================================================================================
# DOMAIN 1: SYNTHETIC SUBGRAPH BENCHMARK (Yang et al. 2018 IEEE TNNLS Protocol)
# ======================================================================================
def generate_synthetic_instance(M=30, N=35, L=25, sigma=0.03, density=0.5, seed=42):
    rng = np.random.default_rng(seed)
    H_pts = rng.uniform(0, 1, (N, 2))
    G_pts = rng.uniform(0, 1, (M, 2))
    
    gi = rng.choice(M, L, replace=False)
    hj = rng.choice(N, L, replace=False)
    rng.shuffle(hj)
    G_pts[gi] = H_pts[hj] + rng.normal(0, sigma, (L, 2))
    
    gt_pairs = set(zip(gi.tolist(), hj.tolist()))
    
    DG = np.linalg.norm(G_pts[:, None, :] - G_pts[None, :, :], axis=2)
    DH = np.linalg.norm(H_pts[:, None, :] - H_pts[None, :, :], axis=2)
    
    # Shared latent topology
    UG = np.triu(rng.random((M, M)) < density, 1)
    UH = np.triu(rng.random((N, N)) < density, 1)
    common = np.triu(rng.random((L, L)) < density, 1)
    common = common + common.T
    UG[np.ix_(gi, gi)] = common
    UH[np.ix_(hj, hj)] = common
    
    AG = ((UG + UG.T) * DG).astype(np.float32)
    AH = ((UH + UH.T) * DH).astype(np.float32)
    C = np.linalg.norm(G_pts[:, None, :] - H_pts[None, :, :], axis=-1).astype(np.float32)
    
    return AG, AH, C, gt_pairs


def run_synthetic_benchmark(n_trials=25):
    print("\n" + "=" * 95)
    print("DOMAIN 1: SYNTHETIC SUBGRAPH BENCHMARK (Yang et al. IEEE TNNLS 2018 Protocol)")
    print(f"Settings: M=30, N=35, L=25, Density=0.5, Noise σ=0.03 ({n_trials} trials)")
    print("=" * 95)
    
    models = [
        ("Fast-WCS v3 (Ours Desktop)", "ours_v3", "Our Method"),
        ("GNCCP 2018 Baseline (RLX1F)", "gnccp_2018", "Base Paper (2018)"),
        ("PyGMtools SM", "pygm_sm", "Base Paper Baseline"),
        ("PyGMtools IPFP", "pygm_ipfp", "Base Paper Baseline"),
        ("PyGMtools RRWM", "pygm_rrwm", "Base Paper Baseline"),
        ("GRAMPA (ICML 2020)", "grampa", "Post-2018 Modern"),
        ("Fused Gromov-Wasserstein (FGW)", "fgw", "Post-2018 Modern"),
        ("Partial Optimal Transport (POT)", "pot", "Post-2018 Modern"),
        ("SciPy FAQ + 2-Opt", "faq_2opt", "Post-2018 Modern"),
    ]
    
    results = {m[0]: {"acc": [], "time": []} for m in models}
    
    for trial in range(n_trials):
        seed = 5000 + trial
        AG, AH, C, gt = generate_synthetic_instance(M=30, N=35, L=25, sigma=0.03, density=0.5, seed=seed)
        
        for name, key, category in models:
            t0 = time.perf_counter()
            matches = []
            
            try:
                if key == "ours_v3":
                    solver = AdvancedGNCCP(alpha=0.7, max_iter=8, fw_max_iter=3, ls_max_rounds=25, seed=seed)
                    res = solver.solve(AG, AH, C=C, L=25)
                    matches = res.matches
                elif key == "gnccp_2018":
                    solver = OriginalGNCCP_2018(alpha=0.7, max_iter=10, fw_iter=6)
                    res = solver.solve(AG, AH, L=25)
                    matches = res.matches if hasattr(res, 'matches') else res
                elif key == "pygm_sm":
                    matches = pygm_lawler_match(AG, AH, L=25, solver_func=pygm.sm)
                elif key == "pygm_ipfp":
                    matches = pygm_lawler_match(AG, AH, L=25, solver_func=pygm.ipfp)
                elif key == "pygm_rrwm":
                    matches = pygm_lawler_match(AG, AH, L=25, solver_func=pygm.rrwm)
                elif key == "grampa":
                    matches = grampa_match(AG, AH, L=25)
                elif key == "fgw":
                    matches = fgw_match(AG, AH, C, L=25)
                elif key == "pot":
                    matches = pot_match(C, L=25)
                elif key == "faq_2opt":
                    matches = scipy_faq_match(AG, AH, L=25, with_2opt=True, seed=seed)
            except Exception as e:
                matches = []
                
            rt = (time.perf_counter() - t0) * 1000.0
            acc = (sum(1 for p in matches if p in gt) / 25.0) * 100.0
            results[name]["acc"].append(acc)
            results[name]["time"].append(rt)
            
    print(f"\n{'Algorithm':34s} | {'Category':20s} | {'Accuracy (%)':14s} | {'Time (ms)':10s}")
    print("-" * 88)
    summary_syn = []
    for name, key, cat in models:
        acc_m = np.mean(results[name]["acc"])
        acc_s = np.std(results[name]["acc"])
        t_m = np.mean(results[name]["time"])
        print(f"{name:34s} | {cat:20s} | {acc_m:5.1f}% ± {acc_s:4.1f}% | {t_m:7.2f} ms")
        summary_syn.append({"method": name, "category": cat, "acc_mean": acc_m, "acc_std": acc_s, "time_ms": t_m})
    print("=" * 88)
    return summary_syn


# ======================================================================================
# DOMAIN 2: REAL 3D MOLECULAR CONFORMER BENCHMARK (CYP19A1 Library)
# ======================================================================================
def run_3d_molecular_benchmark(n_trials=35):
    print("\n" + "=" * 105)
    print("DOMAIN 2: REAL 3D MOLECULAR CONFORMER BENCHMARK (CYP19A1 Active Ligands)")
    print(f"Settings: M=18, N=26, L=14, Thermal Gaussian noise σ=0.02 Å ({n_trials} trials)")
    print("=" * 105)
    
    sdf_path = r"c:\Users\abdus\Downloads\FYP\Efficient-3D-Kernels-for-molecular-property-prediction\Data\1843_actives_new.sdf"
    suppl = Chem.SDMolSupplier(sdf_path)
    confs = []
    for m in suppl:
        if m is not None and m.GetNumConformers() > 0 and m.GetNumAtoms() >= 20:
            pos = m.GetConformer().GetPositions().astype(np.float64)
            pos -= np.mean(pos, axis=0)
            confs.append(pos)
            if len(confs) >= 20:
                break
                
    models = [
        ("Fast-WCS v3 (Ours Balanced)", "ours_bal", "Our Flagship Method"),
        ("Fast-WCS v3 (Ours High-Acc)", "ours_high", "Our Flagship Method"),
        ("Fast-WCS Classic (V1 - No 2-Opt)", "classic_v1", "Ablation Baseline"),
        ("GNCCP 2018 Baseline (RLX1F)", "gnccp_2018", "Base Paper (2018)"),
        ("PyGMtools SM", "pygm_sm", "Base Paper Baseline"),
        ("GRAMPA (ICML 2020)", "grampa", "Post-2018 Modern"),
        ("Fused Gromov-Wasserstein (FGW)", "fgw", "Post-2018 Modern"),
        ("Partial Optimal Transport (POT)", "pot", "Post-2018 Modern"),
        ("SciPy FAQ + 2-Opt", "faq_2opt", "Post-2018 Modern"),
        ("SciPy FAQ (Standard QAP)", "scipy_faq", "Post-2018 Modern"),
    ]
    
    results = {m[0]: {"acc": [], "rmsd": [], "subang": [], "ultra": [], "time": []} for m in models}
    
    for trial in range(n_trials):
        seed = 7000 + trial
        rng = np.random.RandomState(seed)
        pos_orig = confs[trial % len(confs)]
        M, N, L = 18, 26, 14
        
        idx_G = rng.choice(len(pos_orig), M, replace=False)
        P = pos_orig[idx_G].copy()
        angles = rng.uniform(0, 360, 3)
        R_gt = Rotation.from_euler('zyx', angles, degrees=True).as_matrix()
        t_gt = rng.uniform(-5.0, 5.0, 3)
        P_trans = P @ R_gt.T + t_gt + rng.normal(0, 0.02, P.shape)
        
        Q = np.zeros((N, 3), dtype=np.float64)
        Q[:L] = P_trans[:L]
        span = np.ptp(P_trans, axis=0)
        Q[L:] = rng.uniform(P_trans.min(0) - 0.2 * span, P_trans.max(0) + 0.2 * span, (N - L, 3))
        gt_pairs = set((i, i) for i in range(L))
        
        dist_P = np.linalg.norm(P[:, None] - P[None, :], axis=-1)
        dist_Q = np.linalg.norm(Q[:, None] - Q[None, :], axis=-1)
        sigma_rbf = max(float(np.median(dist_P[dist_P > 0])), 1e-3)
        AG = np.exp(-0.5 * (dist_P / sigma_rbf) ** 2).astype(np.float32)
        AH = np.exp(-0.5 * (dist_Q / sigma_rbf) ** 2).astype(np.float32)
        np.fill_diagonal(AG, 0); np.fill_diagonal(AH, 0)
        C_feat = np.linalg.norm(P[:, None] - Q[None, :], axis=-1).astype(np.float32)
        
        for name, key, cat in models:
            t0 = time.perf_counter()
            matches = []
            rmsd = 999.0
            
            try:
                if key == "ours_bal":
                    solver = AdvancedGNCCP(alpha=0.7, max_iter=8, fw_max_iter=3, ls_max_rounds=15, seed=seed)
                    res = solver.match_point_clouds(P, Q, L=L, n_dgm_iters=2)
                    matches = res.matches
                    rmsd = res.alignment['rmsd'] if res.alignment else 999.0
                elif key == "ours_high":
                    solver = AdvancedGNCCP(alpha=0.7, max_iter=10, fw_max_iter=4, ls_max_rounds=25, seed=seed)
                    res = solver.match_point_clouds(P, Q, L=L, n_dgm_iters=2)
                    matches = res.matches
                    rmsd = res.alignment['rmsd'] if res.alignment else 999.0
                elif key == "classic_v1":
                    solver = UniversalGNCCP(alpha=0.7, max_iter=8, fw_max_iter=3, local_refinement=False)
                    res = solver.solve(AG, AH, L=L)
                    matches = res.matches
                    p_sub = np.array([P[g] for g, h in matches[:L]])
                    q_sub = np.array([Q[h] for g, h in matches[:L]])
                    rmsd = robust_kabsch_superposition(p_sub, q_sub)['rmsd']
                elif key == "gnccp_2018":
                    solver = OriginalGNCCP_2018(alpha=0.7, max_iter=10, fw_iter=6)
                    res = solver.solve(AG, AH, L=L)
                    matches = res.matches if hasattr(res, 'matches') else res
                    p_sub = np.array([P[g] for g, h in matches[:L]])
                    q_sub = np.array([Q[h] for g, h in matches[:L]])
                    rmsd = robust_kabsch_superposition(p_sub, q_sub)['rmsd']
                elif key == "pygm_sm":
                    matches = pygm_lawler_match(AG, AH, L=L, solver_func=pygm.sm)
                    p_sub = np.array([P[g] for g, h in matches[:L]])
                    q_sub = np.array([Q[h] for g, h in matches[:L]])
                    rmsd = robust_kabsch_superposition(p_sub, q_sub)['rmsd']
                elif key == "grampa":
                    matches = grampa_match(AG, AH, L=L)
                    p_sub = np.array([P[g] for g, h in matches[:L]])
                    q_sub = np.array([Q[h] for g, h in matches[:L]])
                    rmsd = robust_kabsch_superposition(p_sub, q_sub)['rmsd']
                elif key == "fgw":
                    matches = fgw_match(AG, AH, C_feat, L=L)
                    p_sub = np.array([P[g] for g, h in matches[:L]])
                    q_sub = np.array([Q[h] for g, h in matches[:L]])
                    rmsd = robust_kabsch_superposition(p_sub, q_sub)['rmsd']
                elif key == "pot":
                    matches = pot_match(C_feat, L=L)
                    p_sub = np.array([P[g] for g, h in matches[:L]])
                    q_sub = np.array([Q[h] for g, h in matches[:L]])
                    rmsd = robust_kabsch_superposition(p_sub, q_sub)['rmsd']
                elif key == "faq_2opt":
                    matches = scipy_faq_match(AG, AH, L=L, with_2opt=True, seed=seed)
                    p_sub = np.array([P[g] for g, h in matches[:L]])
                    q_sub = np.array([Q[h] for g, h in matches[:L]])
                    rmsd = robust_kabsch_superposition(p_sub, q_sub)['rmsd']
                elif key == "scipy_faq":
                    matches = scipy_faq_match(AG, AH, L=L, with_2opt=False, seed=seed)
                    p_sub = np.array([P[g] for g, h in matches[:L]])
                    q_sub = np.array([Q[h] for g, h in matches[:L]])
                    rmsd = robust_kabsch_superposition(p_sub, q_sub)['rmsd']
            except Exception as e:
                matches = []
                rmsd = 999.0
                
            rt = (time.perf_counter() - t0) * 1000.0
            acc = (sum(1 for p in matches if p in gt_pairs) / float(L)) * 100.0
            results[name]["acc"].append(acc)
            results[name]["rmsd"].append(rmsd)
            results[name]["subang"].append(1.0 if rmsd < 1.0 else 0.0)
            results[name]["ultra"].append(1.0 if rmsd < 0.1 else 0.0)
            results[name]["time"].append(rt)
            
    print("\n" + "=" * 110)
    print(f"{'Method':34s} | {'Category':20s} | {'Time (ms)':9s} | {'Accuracy':9s} | {'Med. RMSD':10s} | {'Sub-Å (<1Å)':11s} | {'Balance':8s}")
    print("-" * 110)
    
    summary_3d = []
    for name, key, cat in models:
        t_m = np.mean(results[name]["time"])
        a_m = np.mean(results[name]["acc"])
        r_med = np.median(results[name]["rmsd"])
        s_m = np.mean(results[name]["subang"]) * 100.0
        u_m = np.mean(results[name]["ultra"]) * 100.0
        bal = a_m / np.log2(1.0 + t_m)
        print(f"{name:34s} | {cat:20s} | {t_m:7.2f} ms | {a_m:7.1f}% | {r_med:8.3f} Å | {s_m:9.1f}% | {bal:7.2f}")
        summary_3d.append({
            "method": name, "category": cat, "time_ms": t_m, "accuracy": a_m,
            "median_rmsd": r_med, "subang_rate": s_m, "ultra_rate": u_m, "balance_score": bal
        })
    print("=" * 110)
    return summary_3d


# ======================================================================================
# MASTER EXECUTION & REPORT GENERATION
# ======================================================================================
if __name__ == "__main__":
    t_master_0 = time.time()
    syn_results = run_synthetic_benchmark(n_trials=25)
    mol_results = run_3d_molecular_benchmark(n_trials=35)
    
    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_sec": time.time() - t_master_0,
        "synthetic_benchmark": syn_results,
        "molecular_3d_benchmark": mol_results
    }
    
    out_json = DELIVERABLES_DIR / "benchmarks" / "master_domain_benchmark_results.json"
    with open(out_json, "w") as f:
        json.dump(output_data, f, indent=2)
        
    print(f"\n[SUCCESS] Master benchmark completed in {output_data['total_elapsed_sec']:.1f}s.")
    print(f"[SUCCESS] JSON written to: {out_json}")
