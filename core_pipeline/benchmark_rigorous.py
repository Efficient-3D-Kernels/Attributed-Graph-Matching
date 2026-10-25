"""
benchmark_rigorous.py
========================================================================================
RIGOROUS MULTI-SEED BENCHMARK ON MEDIUM-TO-LARGE SCIENTIFIC GRAPHS
========================================================================================

Compares 5 Graph & Subgraph Matching Methods on N = 30 to 100 node graphs over 10 seeds:
  1. GNCCP Advanced (Ours V2):
     - Analytical quartic line search (exact cubic derivative roots)
     - Pure convex combination FW (zero inner projection overhead)
     - Burkard augmented (M + N - L) exact cardinality-L Hungarian oracle
     - O(L) 2-opt vectorized local search (exchange + replacement moves)
     - Shared scale normalization
  2. GNCCP Classic (V1 baseline):
     - Quadratic line search
     - Padded columns only
     - Replacement moves only
     - Independent Frobenius normalization
  3. SciPy FAQ (Fast Approximate Quadratic Assignment - Vogelstein et al. 2015):
     - scipy.optimize.quadratic_assignment
  4. Partial Optimal Transport (POT):
     - Unbalanced partial Wasserstein transport (mass = L / min(M, N))
  5. Fused Gromov-Wasserstein (FGW):
     - Joint structural and feature transport via POT
========================================================================================
"""

import os
import sys
import time
import json
import warnings
import numpy as np
from scipy.optimize import linear_sum_assignment, quadratic_assignment

warnings.filterwarnings('ignore')

# Safe stdout reconfigure
if sys.platform.startswith('win'):
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Import solvers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnccp_advanced import AdvancedGNCCP
from gnccp_generalized import UniversalGNCCP

# Import POT
try:
    import ot
    HAS_POT = True
except ImportError:
    HAS_POT = False
    print("[WARNING] POT library not installed.")


def generate_benchmark_instance(M, N, L, sigma=0.03, edge_density=0.4, seed=42):
    """
    Generates a synthetic WCS instance with known ground truth.
    M: size of graph G
    N: size of graph H (N >= M)
    L: size of common subgraph (L <= min(M, N))
    """
    rng = np.random.RandomState(seed)
    
    # Generate points in 2D
    coords_G = rng.uniform(0, 1, (M, 2)).astype(np.float64)
    coords_H = rng.uniform(0, 1, (N, 2)).astype(np.float64)
    
    # Ground truth mapping: choose L nodes in G and L nodes in H
    gt_g = rng.choice(M, L, replace=False)
    gt_h = rng.choice(N, L, replace=False)
    
    for gi, hi in zip(gt_g, gt_h):
        coords_H[hi] = coords_G[gi] + rng.normal(0, sigma, 2)
        
    dist_G = np.linalg.norm(coords_G[:, None] - coords_G[None, :], axis=-1)
    dist_H = np.linalg.norm(coords_H[:, None] - coords_H[None, :], axis=-1)
    
    # RBF distance kernel for edge weights
    thresh_G = np.percentile(dist_G[dist_G > 0], edge_density * 100) if np.any(dist_G > 0) else 1.0
    thresh_H = np.percentile(dist_H[dist_H > 0], edge_density * 100) if np.any(dist_H > 0) else 1.0
    
    A_G = np.where(dist_G < thresh_G, np.exp(-dist_G), 0.0).astype(np.float32)
    A_H = np.where(dist_H < thresh_H, np.exp(-dist_H), 0.0).astype(np.float32)
    np.fill_diagonal(A_G, 0.0)
    np.fill_diagonal(A_H, 0.0)
    
    # Unary feature distance matrix C
    C = np.linalg.norm(coords_G[:, None] - coords_H[None, :], axis=-1).astype(np.float32)
    c_max = C.max()
    if c_max > 0: C /= c_max
    
    gt_pairs = set(zip(gt_g, gt_h))
    return A_G, A_H, C, gt_pairs, L


def eval_matches(matches, gt_pairs, L, A_G, A_H, C, alpha=0.7):
    """Computes accuracy and WCS objective."""
    correct = sum(1 for g, h in matches if (g, h) in gt_pairs)
    acc = correct / L if L > 0 else 0.0
    
    M, N = C.shape
    X = np.zeros((M, N), dtype=np.float32)
    for r, c in matches[:L]:
        if r < M and c < N:
            X[r, c] = 1.0
    
    r_vec = X.sum(axis=1)
    diff = (r_vec[:, None] * A_G * r_vec[None, :]) - (X @ A_H @ X.T)
    obj = float(alpha * np.sum(diff ** 2) + (1.0 - alpha) * np.sum(C * X))
    return acc, obj


# ======================================================================================
# SOLVER WRAPPERS
# ======================================================================================

def solve_gnccp_advanced(A_G, A_H, C, L, seed):
    solver = AdvancedGNCCP(alpha=0.7, max_iter=15, fw_max_iter=8, line_search='quartic',
                           normalization='none', local_refinement=True, enable_exchange_moves=True, seed=seed)
    t0 = time.perf_counter()
    res = solver.solve(A_G, A_H, C=C, L=L)
    rt = time.perf_counter() - t0
    return res.matches, rt

def solve_gnccp_classic(A_G, A_H, C, L, seed):
    solver = UniversalGNCCP(alpha=0.7, max_iter=15, fw_max_iter=8, line_search='closed_form',
                            normalize_fro=False, oracle='dummy_padded', local_refinement=True)
    t0 = time.perf_counter()
    res = solver.solve(A_G, A_H, C=C, L=L)
    rt = time.perf_counter() - t0
    return res.matches, rt

def solve_scipy_faq(A_G, A_H, C, L, seed):
    t0 = time.perf_counter()
    M, N = C.shape
    # Pad A_G to N x N
    A_G_pad = np.zeros((N, N), dtype=np.float64)
    A_G_pad[:M, :M] = A_G
    A_H_64 = A_H.astype(np.float64)
    
    try:
        res = quadratic_assignment(A_G_pad, A_H_64, method='faq', options={'maxiter': 30, 'rng': seed})
        # Score each matched pair (i, pi(i)) for i < M
        pairs = [(C[i, res.col_ind[i]], i, res.col_ind[i]) for i in range(M)]
        pairs.sort(key=lambda x: x[0])  # keep lowest unary cost
        matches = [(r, c) for _, r, c in pairs[:L]]
    except Exception:
        matches = []
    rt = time.perf_counter() - t0
    return matches, rt

def solve_pot_partial(A_G, A_H, C, L, seed):
    if not HAS_POT:
        return [], 0.0
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
    rt = time.perf_counter() - t0
    return matches, rt

def solve_pot_fgw(A_G, A_H, C, L, seed):
    if not HAS_POT:
        return [], 0.0
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
    rt = time.perf_counter() - t0
    return matches, rt


# ======================================================================================
# BENCHMARK RUNNER
# ======================================================================================

def run_rigorous_benchmark():
    print("=" * 96)
    print("RIGOROUS MULTI-SEED BENCHMARK (N = 30, 60, 100) — ACCURACY, OBJECTIVE & TIME")
    print("=" * 96)
    
    test_configs = [
        # (label, M, N, L, n_seeds)
        ("Small Scale (M=20, N=30, L=15)", 20, 30, 15, 10),
        ("Medium Scale (M=40, N=60, L=30)", 40, 60, 30, 10),
        ("Large Scale (M=70, N=100, L=50)", 70, 100, 50, 10),
    ]
    
    methods = {
        "GNCCP Advanced (Ours V2)": solve_gnccp_advanced,
        "GNCCP Classic (V1)": solve_gnccp_classic,
        "SciPy FAQ (QAP)": solve_scipy_faq,
        "Partial Optimal Transport (POT)": solve_pot_partial,
        "Fused Gromov-Wasserstein (FGW)": solve_pot_fgw,
    }
    
    overall_summary = {}
    
    for cfg_label, M, N, L, n_seeds in test_configs:
        print(f"\n[BENCHMARK CONFIG: {cfg_label}] — Running {n_seeds} random seeds...")
        print(f"  {'Method':<36s} | {'Accuracy (mean±std)':>20s} | {'Objective (mean)':>16s} | {'Runtime (ms)':>12s}")
        print("  " + "-" * 92)
        
        cfg_stats = {}
        for m_name, m_func in methods.items():
            acc_list = []
            obj_list = []
            rt_list = []
            
            for seed in range(n_seeds):
                A_G, A_H, C, gt_pairs, L_actual = generate_benchmark_instance(
                    M, N, L, sigma=0.03, edge_density=0.35, seed=seed * 100 + M)
                
                matches, rt = m_func(A_G, A_H, C, L_actual, seed)
                acc, obj = eval_matches(matches, gt_pairs, L_actual, A_G, A_H, C, alpha=0.7)
                
                acc_list.append(acc)
                obj_list.append(obj)
                rt_list.append(rt * 1000.0)
                
            acc_m, acc_s = np.mean(acc_list) * 100.0, np.std(acc_list) * 100.0
            obj_m = np.mean(obj_list)
            rt_m = np.mean(rt_list)
            
            cfg_stats[m_name] = {'acc_mean': acc_m, 'acc_std': acc_s, 'obj_mean': obj_m, 'rt_ms': rt_m}
            print(f"  {m_name:<36s} | {acc_m:6.1f}% ± {acc_s:4.1f}%    | {obj_m:16.4f} | {rt_m:10.1f} ms")
            
        overall_summary[cfg_label] = cfg_stats
        
    print("\n" + "=" * 96)
    print("BENCHMARK COMPLETE")
    print("=" * 96)
    return overall_summary


if __name__ == '__main__':
    run_rigorous_benchmark()
