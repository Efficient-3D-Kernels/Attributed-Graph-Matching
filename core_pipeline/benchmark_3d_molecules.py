"""
benchmark_3d_molecules.py
========================================================================================
RIGOROUS 3D MOLECULAR GRAPH & CONFORMER MATCHING BENCHMARK
========================================================================================
Evaluates 5 Methods on Real 3D Molecular Conformers from Data/1843_actives_new.sdf
under arbitrary 3D rigid rotation (SO(3)), translation, thermal noise, and outliers:
  1. GNCCP Advanced (Ours V2):
     - Analytical quartic line search
     - Burkard augmented (M+N-L) Hungarian oracle
     - O(L) 2-opt vectorized local search (exchange + replacement)
     - Frobenius-balanced scale normalization
     - Robust trimmed Kabsch superposition
  2. GNCCP Classic (V1 baseline)
  3. Partial Optimal Transport (POT)
  4. Fused Gromov-Wasserstein (FGW)
  5. SciPy FAQ (QAP)
========================================================================================
"""

import os
import sys
import time
import warnings
import numpy as np
from scipy.optimize import linear_sum_assignment, quadratic_assignment
from scipy.spatial.transform import Rotation

warnings.filterwarnings('ignore')

# Safe stdout reconfigure
if sys.platform.startswith('win'):
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Solver imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnccp_advanced import AdvancedGNCCP
from gnccp_generalized import UniversalGNCCP

try:
    import ot
    HAS_POT = True
except ImportError:
    HAS_POT = False

from rdkit import Chem


def load_3d_conformers(sdf_path, max_mols=20):
    """Loads molecules with valid 3D conformers from SDF."""
    suppl = Chem.SDMolSupplier(sdf_path)
    confs = []
    for m in suppl:
        if m is not None and m.GetNumConformers() > 0 and m.GetNumAtoms() >= 25:
            pos = m.GetConformer().GetPositions().astype(np.float64)
            # Center molecule at origin
            pos -= np.mean(pos, axis=0)
            confs.append(pos)
            if len(confs) >= max_mols:
                break
    return confs


def kabsch_rmsd(P, Q):
    """Computes exact Kabsch alignment RMSD between paired 3D points P and Q (K x 3)."""
    if len(P) < 3:
        return 999.0
    P_c = P - np.mean(P, axis=0)
    Q_c = Q - np.mean(Q, axis=0)
    H = P_c.T @ Q_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    V = Vt.T
    V[:, -1] *= d
    R = V @ U.T
    P_aligned = P_c @ R
    diff = P_aligned - Q_c
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def generate_3d_instance(pos_orig, M=24, N=36, L=18, noise_sigma=0.03, seed=42):
    """
    Creates a real 3D molecular benchmark instance:
      - Graph G has M atoms
      - Graph H has N atoms (M core atoms + (N - M) 3D outlier atoms)
      - Molecule H is rigidly rotated in 3D by random R in SO(3) and translated by t
      - Common core has L atoms
    """
    rng = np.random.RandomState(seed)
    
    # Subsample M atoms for G
    n_total = len(pos_orig)
    idx_G = rng.choice(n_total, M, replace=False)
    pos_G = pos_orig[idx_G].copy()
    
    # Pick L atoms as common ground truth
    gt_g = rng.choice(M, L, replace=False)
    gt_h_indices = rng.choice(N, L, replace=False)
    
    # Random 3D Rotation in SO(3)
    rot_angles = rng.uniform(0, 360, 3)
    R_3d = Rotation.from_euler('zyx', rot_angles, degrees=True).as_matrix()
    t_3d = rng.uniform(-10.0, 10.0, 3)
    
    # Generate H coordinates
    pos_H = np.zeros((N, 3), dtype=np.float64)
    # Assign core transformed positions
    pos_G_trans = pos_G @ R_3d.T + t_3d + rng.normal(0, noise_sigma, pos_G.shape)
    
    # Fill gt_h_indices with transformed gt_g
    remaining_H = [i for i in range(N) if i not in gt_h_indices]
    remaining_G = [i for i in range(M) if i not in gt_g]
    
    for gi, hi in zip(gt_g, gt_h_indices):
        pos_H[hi] = pos_G_trans[gi]
        
    for gi, hi in zip(remaining_G, remaining_H[:len(remaining_G)]):
        pos_H[hi] = pos_G_trans[gi]
        
    # Extra outliers in H
    bbox_min, bbox_max = np.min(pos_G_trans, axis=0), np.max(pos_G_trans, axis=0)
    for hi in remaining_H[len(remaining_G):]:
        pos_H[hi] = rng.uniform(bbox_min, bbox_max)
        
    # 3D Distance Geometry Adjacency (Rotation Invariant!)
    dist_G = np.linalg.norm(pos_G[:, None] - pos_G[None, :], axis=-1)
    dist_H = np.linalg.norm(pos_H[:, None] - pos_H[None, :], axis=-1)
    
    # RBF 3D interatomic kernel
    A_G = np.exp(-dist_G / 3.0).astype(np.float32)
    A_H = np.exp(-dist_H / 3.0).astype(np.float32)
    np.fill_diagonal(A_G, 0.0)
    np.fill_diagonal(A_H, 0.0)
    
    # Rotation-invariant unary feature cost: based on topological degree & local 3D density
    deg_G = A_G.sum(axis=1)[:, None]
    deg_H = A_H.sum(axis=1)[None, :]
    C_inv = np.abs(deg_G - deg_H).astype(np.float32)
    c_max = C_inv.max()
    if c_max > 0: C_inv /= c_max
    
    # Coordinate distance (NOT rotation invariant - what naive POT would see)
    C_coord = np.linalg.norm(pos_G[:, None] - pos_H[None, :], axis=-1).astype(np.float32)
    C_coord /= C_coord.max()
    
    gt_pairs = set(zip(gt_g, gt_h_indices))
    return pos_G, pos_H, A_G, A_H, C_inv, C_coord, gt_pairs, L


def eval_result(matches, gt_pairs, L, A_G, A_H, C, pos_G, pos_H, alpha=0.7):
    """Computes match accuracy, WCS objective, and post-Kabsch 3D RMSD."""
    correct = sum(1 for g, h in matches if (g, h) in gt_pairs)
    acc = (correct / L * 100.0) if L > 0 else 0.0
    
    M, N = C.shape
    X = np.zeros((M, N), dtype=np.float32)
    for r, c in matches[:L]:
        if r < M and c < N:
            X[r, c] = 1.0
            
    r_vec = X.sum(axis=1)
    diff = (r_vec[:, None] * A_G * r_vec[None, :]) - (X @ A_H @ X.T)
    obj = float(alpha * np.sum(diff ** 2) + (1.0 - alpha) * np.sum(C * X))
    
    # 3D RMSD
    if len(matches) >= 3:
        p_pts = np.array([pos_G[g] for g, h in matches[:L]])
        q_pts = np.array([pos_H[h] for g, h in matches[:L]])
        rmsd = kabsch_rmsd(p_pts, q_pts)
    else:
        rmsd = 999.0
        
    return acc, obj, rmsd


# ======================================================================================
# SOLVERS
# ======================================================================================

def solve_advanced(A_G, A_H, C, L, seed):
    solver = AdvancedGNCCP(alpha=0.7, max_iter=10, fw_max_iter=4, line_search='quartic',
                           normalization='none', local_refinement=True, enable_exchange_moves=True, seed=seed)
    t0 = time.perf_counter()
    res = solver.solve(A_G, A_H, C=C, L=L)
    return res.matches, (time.perf_counter() - t0) * 1000.0

def solve_classic(A_G, A_H, C, L, seed):
    solver = UniversalGNCCP(alpha=0.7, max_iter=15, fw_max_iter=8, line_search='closed_form',
                            oracle='dummy_padded', local_refinement=True)
    t0 = time.perf_counter()
    res = solver.solve(A_G, A_H, C=C, L=L)
    return res.matches, (time.perf_counter() - t0) * 1000.0

def solve_pot_partial(A_G, A_H, C, L, seed):
    if not HAS_POT: return [], 0.0
    t0 = time.perf_counter()
    M, N = C.shape
    a = np.ones(M, dtype=np.float64) / M
    b = np.ones(N, dtype=np.float64) / N
    m_frac = float(L / min(M, N))
    try:
        T = ot.partial.partial_wasserstein(a, b, C.astype(np.float64), m=m_frac)
        row_ind, col_ind = linear_sum_assignment(-T)
        pairs = [(T[r, c], r, c) for r, c in zip(row_ind, col_ind) if r < M and c < N]
        pairs.sort(key=lambda x: x[0], reverse=True)
        matches = [(r, c) for _, r, c in pairs[:L]]
    except Exception:
        matches = []
    return matches, (time.perf_counter() - t0) * 1000.0

def solve_fgw(A_G, A_H, C, L, seed):
    if not HAS_POT: return [], 0.0
    t0 = time.perf_counter()
    M, N = C.shape
    p = np.ones(M, dtype=np.float64) / M
    q = np.ones(N, dtype=np.float64) / N
    try:
        T = ot.gromov.fused_gromov_wasserstein(
            C.astype(np.float64), A_G.astype(np.float64), A_H.astype(np.float64),
            p, q, alpha=0.5, max_iter=50, verbose=False
        )
        row_ind, col_ind = linear_sum_assignment(-T)
        pairs = [(T[r, c], r, c) for r, c in zip(row_ind, col_ind) if r < M and c < N]
        pairs.sort(key=lambda x: x[0], reverse=True)
        matches = [(r, c) for _, r, c in pairs[:L]]
    except Exception:
        matches = []
    return matches, (time.perf_counter() - t0) * 1000.0

def solve_scipy_faq(A_G, A_H, C, L, seed):
    t0 = time.perf_counter()
    M, N = C.shape
    A_G_pad = np.zeros((N, N), dtype=np.float64)
    A_G_pad[:M, :M] = A_G
    A_H_64 = A_H.astype(np.float64)
    try:
        res = quadratic_assignment(A_G_pad, A_H_64, method='faq', options={'maxiter': 30, 'rng': seed})
        pairs = [(C[i, res.col_ind[i]], i, res.col_ind[i]) for i in range(M)]
        pairs.sort(key=lambda x: x[0])
        matches = [(r, c) for _, r, c in pairs[:L]]
    except Exception:
        matches = []
    return matches, (time.perf_counter() - t0) * 1000.0


# ======================================================================================
# BENCHMARK EXECUTION
# ======================================================================================

def run_3d_benchmark():
    sdf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data', '1843_actives_new.sdf')
    print("=" * 105)
    print("3D MOLECULAR CONFORMER MATCHING BENCHMARK (REAL SDF MOLECULES UNDER 3D SO(3) ROTATION)")
    print("=" * 105)
    
    confs = load_3d_conformers(sdf_path, max_mols=10)
    print(f"Loaded {len(confs)} valid 3D conformers from {os.path.basename(sdf_path)}")
    
    n_seeds = 10
    M, N, L = 24, 36, 18
    print(f"Instance configuration: M={M} atoms in G, N={N} atoms in H (with outliers), L={L} common atoms.")
    print(f"Graph H has random 3D rotation in SO(3), translation, and thermal noise sigma=0.03 A.")
    print("=" * 105)
    
    methods = {
        "GNCCP Advanced (Ours V2)": solve_advanced,
        "GNCCP Classic (V1)": solve_classic,
        "Partial Optimal Transport (POT)": solve_pot_partial,
        "Fused Gromov-Wasserstein (FGW)": solve_fgw,
        "SciPy FAQ (QAP)": solve_scipy_faq,
    }
    
    # ----------------------------------------------------------------------------------
    # EXPERIMENT 1: Under 3D Rigid Rotation with Invariant Features C_inv
    # ----------------------------------------------------------------------------------
    print(f"\n--- EXPERIMENT 1: 3D Conformer Matching with Invariant Structural Features ---")
    print(f"{'Method':<34s} | {'Accuracy (%)':>16s} | {'WCS Obj':>12s} | {'3D RMSD (A)':>14s} | {'Runtime (ms)':>12s}")
    print("-" * 105)
    
    for m_name, m_func in methods.items():
        accs, objs, rmsds, rts = [], [], [], []
        for s in range(n_seeds):
            pos_orig = confs[s % len(confs)]
            pos_G, pos_H, A_G, A_H, C_inv, C_coord, gt_pairs, L_act = generate_3d_instance(
                pos_orig, M=M, N=N, L=L, noise_sigma=0.03, seed=s * 100 + 42)
            
            matches, rt = m_func(A_G, A_H, C_inv, L_act, s)
            acc, obj, rmsd = eval_result(matches, gt_pairs, L_act, A_G, A_H, C_inv, pos_G, pos_H)
            
            accs.append(acc)
            objs.append(obj)
            rmsds.append(rmsd)
            rts.append(rt)
            
        print(f"{m_name:<34s} | {np.mean(accs):6.1f}% +/- {np.std(accs):4.1f}% | {np.mean(objs):12.4f} | {np.mean(rmsds):11.3f} A  | {np.mean(rts):10.1f} ms")
        
    # ----------------------------------------------------------------------------------
    # EXPERIMENT 2: Why Coordinate-Only POT Fails Under 3D Rotation (C_coord)
    # ----------------------------------------------------------------------------------
    print(f"\n--- EXPERIMENT 2: What Happens to Coordinate-Only POT when Molecule is Rotated in 3D ---")
    print(f"{'Method':<34s} | {'Accuracy (%)':>16s} | {'WCS Obj':>12s} | {'3D RMSD (A)':>14s} | {'Runtime (ms)':>12s}")
    print("-" * 105)
    
    for m_name in ["GNCCP Advanced (Ours V2)", "Partial Optimal Transport (POT)"]:
        m_func = methods[m_name]
        accs, objs, rmsds, rts = [], [], [], []
        for s in range(n_seeds):
            pos_orig = confs[s % len(confs)]
            pos_G, pos_H, A_G, A_H, C_inv, C_coord, gt_pairs, L_act = generate_3d_instance(
                pos_orig, M=M, N=N, L=L, noise_sigma=0.03, seed=s * 100 + 42)
            
            # Here we pass C_coord (unaligned Euclidean distances in space)
            matches, rt = m_func(A_G, A_H, C_coord, L_act, s)
            acc, obj, rmsd = eval_result(matches, gt_pairs, L_act, A_G, A_H, C_coord, pos_G, pos_H)
            
            accs.append(acc)
            objs.append(obj)
            rmsds.append(rmsd)
            rts.append(rt)
            
        print(f"{m_name:<34s} | {np.mean(accs):6.1f}% +/- {np.std(accs):4.1f}% | {np.mean(objs):12.4f} | {np.mean(rmsds):11.3f} A  | {np.mean(rts):10.1f} ms")
    
    print("\n" + "=" * 105)


if __name__ == '__main__':
    run_3d_benchmark()
