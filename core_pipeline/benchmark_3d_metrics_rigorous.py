"""
benchmark_3d_metrics_rigorous.py
========================================================================================
RIGOROUS 3D PHYSICAL METRICS BENCHMARK: FAST-WCS ADVANCED (OURS V2) VS. ALL COMPETITORS
========================================================================================
Evaluates physical 3D superposition, geometric accuracy, and runtime across 5 models:
  1. Fast-WCS Advanced (Ours V2 - AdvancedGNCCP)
  2. Fast-WCS Classic (V1 - UniversalGNCCP)
  3. Partial Optimal Transport (POT - unbalanced partial Wasserstein)
  4. Fused Gromov-Wasserstein (FGW - joint structural & feature optimal transport)
  5. SciPy FAQ (Fast Approximate Quadratic Assignment - Vogelstein et al. 2015)

Evaluated on real 3D molecular conformers under arbitrary 3D rigid rotation in SO(3),
spatial translation in R^3, thermal coordinate jitter, and 3D geometric outliers.
========================================================================================
"""

import os
import sys
import time
import json
import warnings
import numpy as np

# Safe stdout reconfigure on Windows console
if sys.platform.startswith('win'):
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

warnings.filterwarnings('ignore')

from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment, quadratic_assignment

try:
    import ot
    HAS_POT = True
except ImportError:
    HAS_POT = False

from rdkit import Chem

# Import our solvers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnccp_advanced import AdvancedGNCCP, robust_kabsch_superposition
from gnccp_generalized import UniversalGNCCP, kabsch_superposition


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Exact closed-form Kabsch physical RMSD (in Angstroms)."""
    if len(P) < 3:
        return 999.0
    Pc = P - np.mean(P, axis=0)
    Qc = Q - np.mean(Q, axis=0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    P_rot = Pc @ R.T
    return float(np.sqrt(np.mean(np.sum((P_rot - Qc) ** 2, axis=1))))


def generate_3d_dataset(sdf_path: str, max_mols: int = 15):
    """Loads 3D coordinates of real drug molecules from SDF."""
    suppl = Chem.SDMolSupplier(sdf_path)
    confs = []
    for m in suppl:
        if m is not None and m.GetNumConformers() > 0 and m.GetNumAtoms() >= 20:
            pos = m.GetConformer().GetPositions().astype(np.float64)
            pos -= np.mean(pos, axis=0)  # zero-center
            confs.append(pos)
            if len(confs) >= max_mols:
                break
    return confs


def create_rotated_pair(pos_orig, M=20, N=26, L=14, noise_sigma=0.015, seed=42):
    """
    Simulates physical 3D docking / alignment:
    - G has M atoms
    - Common core has L atoms
    - H is rotated by random R in SO(3), translated by t in R^3, and contaminated by (N-L) outliers.
    """
    rng = np.random.RandomState(seed)
    
    # Subsample M atoms for G
    n_total = len(pos_orig)
    idx_G = rng.choice(n_total, M, replace=False)
    P = pos_orig[idx_G].copy()
    
    # Core atoms to preserve
    core_idx = np.arange(L)
    
    # Random 3D Rotation in SO(3)
    angles = rng.uniform(0, 360, 3)
    R_gt = Rotation.from_euler('zyx', angles, degrees=True).as_matrix()
    t_gt = rng.uniform(-5.0, 5.0, 3)
    
    # Transform core points
    P_trans = P @ R_gt.T + t_gt + rng.normal(0, noise_sigma, P.shape)
    
    # Assemble Q
    Q = np.zeros((N, 3), dtype=np.float64)
    Q[:L] = P_trans[:L]
    
    # Outliers in H
    if N > L:
        bbox_min, bbox_max = np.min(P_trans, axis=0), np.max(P_trans, axis=0)
        span = bbox_max - bbox_min
        Q[L:] = rng.uniform(bbox_min - 0.2 * span, bbox_max + 0.2 * span, (N - L, 3))
        
    gt_pairs = set((i, i) for i in range(L))
    return P, Q, gt_pairs, L


# ======================================================================================
# MODEL RUNNERS
# ======================================================================================

def run_fast_wcs_advanced(P, Q, L, seed):
    """Ours V2: AdvancedGNCCP with DGM Alternating Kabsch & Quartic Line Search."""
    solver = AdvancedGNCCP(
        alpha=0.7, max_iter=8, fw_max_iter=3, tol=1e-3,
        normalization='none', line_search='quartic',
        oracle='exact_augmented', local_refinement=True, enable_exchange_moves=True,
        seed=seed
    )
    t0 = time.perf_counter()
    res = solver.match_point_clouds(P, Q, L=L, n_dgm_iters=2)
    rt = (time.perf_counter() - t0) * 1000.0
    matches = res.matches
    rmsd = res.alignment['rmsd'] if res.alignment else 999.0
    return matches, rmsd, res.objective_score, rt


def run_fast_wcs_classic(P, Q, L, seed):
    """Ours V1: Classic GNCCP without DGM alternating Kabsch or 2-opt exchanges."""
    dist_P = np.linalg.norm(P[:, None] - P[None, :], axis=-1)
    dist_Q = np.linalg.norm(Q[:, None] - Q[None, :], axis=-1)
    A_G = np.exp(-dist_P / 3.0).astype(np.float32)
    A_H = np.exp(-dist_Q / 3.0).astype(np.float32)
    np.fill_diagonal(A_G, 0.0)
    np.fill_diagonal(A_H, 0.0)
    
    solver = UniversalGNCCP(
        alpha=0.7, max_iter=12, fw_max_iter=6, tol=1e-3,
        line_search='closed_form', normalize_fro=False,
        local_refinement=False
    )
    t0 = time.perf_counter()
    res = solver.solve(A_G, A_H, L=L)
    rt = (time.perf_counter() - t0) * 1000.0
    
    matches = res.matches
    p_sub = np.array([P[g] for g, h in matches[:L]])
    q_sub = np.array([Q[h] for g, h in matches[:L]])
    rmsd = kabsch_rmsd(p_sub, q_sub)
    return matches, rmsd, res.objective_score, rt


def run_pot(P, Q, L, seed):
    """Partial Optimal Transport (linear transport on coordinate cost)."""
    if not HAS_POT:
        return [], 999.0, float('inf'), 0.0
    t0 = time.perf_counter()
    M, N = len(P), len(Q)
    C = np.linalg.norm(P[:, None] - Q[None, :], axis=-1)
    c_max = C.max()
    if c_max > 0: C = C / c_max
    a = np.ones(M) / M
    b = np.ones(N) / N
    try:
        T = ot.partial.partial_wasserstein(a, b, C, m=float(L / min(M, N)))
        r_ind, c_ind = linear_sum_assignment(-T)
        pairs = [(T[r, c], r, c) for r, c in zip(r_ind, c_ind) if r < M and c < N]
        pairs.sort(key=lambda x: x[0], reverse=True)
        matches = [(r, c) for _, r, c in pairs[:L]]
    except Exception:
        matches = []
    rt = (time.perf_counter() - t0) * 1000.0
    
    if len(matches) >= 3:
        p_sub = np.array([P[g] for g, h in matches])
        q_sub = np.array([Q[h] for g, h in matches])
        rmsd = kabsch_rmsd(p_sub, q_sub)
    else:
        rmsd = 999.0
    return matches, rmsd, 0.0, rt


def run_fgw(P, Q, L, seed):
    """Fused Gromov-Wasserstein (structural RBF distance + coordinate transport)."""
    if not HAS_POT:
        return [], 999.0, float('inf'), 0.0
    t0 = time.perf_counter()
    M, N = len(P), len(Q)
    dist_P = np.linalg.norm(P[:, None] - P[None, :], axis=-1)
    dist_Q = np.linalg.norm(Q[:, None] - Q[None, :], axis=-1)
    A_G = np.exp(-dist_P / 3.0)
    A_H = np.exp(-dist_Q / 3.0)
    C = np.linalg.norm(P[:, None] - Q[None, :], axis=-1)
    c_max = C.max()
    if c_max > 0: C = C / c_max
    a = np.ones(M) / M
    b = np.ones(N) / N
    try:
        T = ot.gromov.fused_gromov_wasserstein(C, A_G, A_H, a, b, alpha=0.5, max_iter=40, verbose=False)
        r_ind, c_ind = linear_sum_assignment(-T)
        pairs = [(T[r, c], r, c) for r, c in zip(r_ind, c_ind) if r < M and c < N]
        pairs.sort(key=lambda x: x[0], reverse=True)
        matches = [(r, c) for _, r, c in pairs[:L]]
    except Exception:
        matches = []
    rt = (time.perf_counter() - t0) * 1000.0
    
    if len(matches) >= 3:
        p_sub = np.array([P[g] for g, h in matches])
        q_sub = np.array([Q[h] for g, h in matches])
        rmsd = kabsch_rmsd(p_sub, q_sub)
    else:
        rmsd = 999.0
    return matches, rmsd, 0.0, rt


def run_scipy_faq(P, Q, L, seed):
    """SciPy FAQ: Fast Approximate Quadratic Assignment on padded adjacency."""
    t0 = time.perf_counter()
    M, N = len(P), len(Q)
    dist_P = np.linalg.norm(P[:, None] - P[None, :], axis=-1)
    dist_Q = np.linalg.norm(Q[:, None] - Q[None, :], axis=-1)
    A_G = np.exp(-dist_P / 3.0)
    A_H = np.exp(-dist_Q / 3.0)
    A_G_pad = np.zeros((N, N), dtype=np.float64)
    A_G_pad[:M, :M] = A_G
    A_H_64 = A_H.astype(np.float64)
    C = np.linalg.norm(P[:, None] - Q[None, :], axis=-1)
    
    try:
        res = quadratic_assignment(A_G_pad, A_H_64, method='faq', options={'maxiter': 30, 'rng': seed})
        pairs = [(C[i, res.col_ind[i]], i, res.col_ind[i]) for i in range(M)]
        pairs.sort(key=lambda x: x[0])
        matches = [(r, c) for _, r, c in pairs[:L]]
    except Exception:
        matches = []
    rt = (time.perf_counter() - t0) * 1000.0
    
    if len(matches) >= 3:
        p_sub = np.array([P[g] for g, h in matches])
        q_sub = np.array([Q[h] for g, h in matches])
        rmsd = kabsch_rmsd(p_sub, q_sub)
    else:
        rmsd = 999.0
    return matches, rmsd, 0.0, rt


# ======================================================================================
# BENCHMARK SUITE
# ======================================================================================

def execute_rigorous_3d_benchmark():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sdf_candidates = [
        os.path.join(repo_root, "Data", "1843_actives_new.sdf"),
        os.path.join(repo_root, "Data", "Aromatase_actives_new.sdf")
    ]
    sdf_path = next((p for p in sdf_candidates if os.path.exists(p)), None)
    if sdf_path is None:
        print("[ERROR] No SDF dataset found in Data/ directory.")
        return
        
    print("=" * 102)
    print("RIGOROUS PHYSICAL 3D CONFORMER SUPERPOSITION & METRICS BENCHMARK")
    print(f"Data Source: {os.path.basename(sdf_path)} | Protocol: Multi-Seed SO(3) 3D Rotations + Outliers")
    print("=" * 102)

    confs = generate_3d_dataset(sdf_path, max_mols=100)
    print(f"[*] Loaded {len(confs)} real drug molecules with verified 3D conformers.\n")

    test_scenarios = [
        # (name, M, N, L, noise, n_trials)
        ("Scenario A: Moderate Outliers (M=16, N=22, L=12, σ=0.015 Å)", 16, 22, 12, 0.015, 100),
        ("Scenario B: Dense Subgraph in Noise (M=20, N=28, L=15, σ=0.025 Å)", 20, 28, 15, 0.025, 100),
        ("Scenario C: High-Outlier Stress (M=24, N=36, L=16, σ=0.035 Å)", 24, 36, 16, 0.035, 100),
    ]

    def run_scipy_faq_2opt(P, Q, L, seed):
        """SciPy FAQ with the same 2-opt local refinement applied for fair comparison."""
        matches, _, _, rt_faq = run_scipy_faq(P, Q, L, seed)
        t0 = time.perf_counter()
        if len(matches) == L:
            M, N = len(P), len(Q)
            X = np.zeros((M, N), dtype=np.float32)
            for r, c in matches:
                X[r, c] = 1.0
            dist_P = np.linalg.norm(P[:, None] - P[None, :], axis=-1)
            dist_Q = np.linalg.norm(Q[:, None] - Q[None, :], axis=-1)
            A_G = np.exp(-dist_P / 3.0).astype(np.float32)
            A_H = np.exp(-dist_Q / 3.0).astype(np.float32)
            np.fill_diagonal(A_G, 0.0)
            np.fill_diagonal(A_H, 0.0)
            C = np.linalg.norm(P[:, None] - Q[None, :], axis=-1).astype(np.float32)
            c_max = C.max()
            if c_max > 0: C /= c_max
            solver = AdvancedGNCCP(alpha=0.7, ls_max_rounds=200)
            X_refined = solver._swap_refinement(X, A_G, A_H, C, L)
            r_idx, c_idx = np.where(X_refined > 0.5)
            refined_matches = list(zip(r_idx.tolist(), c_idx.tolist()))
            if len(refined_matches) == L:
                matches = refined_matches
        rt_total = rt_faq + (time.perf_counter() - t0) * 1000.0
        if len(matches) >= 3:
            p_sub = np.array([P[g] for g, h in matches])
            q_sub = np.array([Q[h] for g, h in matches])
            rmsd = kabsch_rmsd(p_sub, q_sub)
        else:
            rmsd = 999.0
        return matches, rmsd, 0.0, rt_total

    all_methods = {
        "Fast-WCS Advanced (Ours V2)": run_fast_wcs_advanced,
        "Fast-WCS Classic (V1 - No 2-Opt)": run_fast_wcs_classic,
        "SciPy FAQ + 2-Opt Refinement": run_scipy_faq_2opt,
        "Partial Optimal Transport (POT)": run_pot,
        "Fused Gromov-Wasserstein (FGW)": run_fgw,
        "SciPy FAQ (QAP Baseline)": run_scipy_faq,
    }

    def calc_ci95(data):
        """Calculates 95% bootstrap confidence interval."""
        arr = np.asarray(data, dtype=np.float64)
        if len(arr) == 0: return 0.0, 0.0
        n_boot = 1000
        rng = np.random.RandomState(42)
        idx = rng.randint(0, len(arr), (n_boot, len(arr)))
        means = np.mean(arr[idx], axis=1)
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    full_results = {}

    for sc_name, M, N, L, noise, n_trials in test_scenarios:
        print(f"\n{'='*102}")
        print(f"RUNNING: {sc_name} (N_TRIALS = {n_trials})")
        print(f"{'='*102}")
        print(f"{'Method':<34s} | {'Accuracy (%) [95% CI]':>22s} | {'RMSD (Å)':>10s} | {'< 1.0 Å':>8s} | {'< 0.1 Å':>8s} | {'Time (ms)':>10s}")
        print("-" * 102)

        sc_data = {}
        for m_name in all_methods:
            sc_data[m_name] = {'acc': [], 'rmsd': [], 'time': []}

        for trial_idx in range(n_trials):
            mol_pos = confs[trial_idx % len(confs)]
            P, Q, gt_pairs, L_t = create_rotated_pair(mol_pos, M=M, N=N, L=L, noise_sigma=noise, seed=100 + trial_idx)

            for m_name, fn in all_methods.items():
                matches, rmsd, obj, rt = fn(P, Q, L_t, seed=100 + trial_idx)
                rec = sum(1 for g, h in matches if (g, h) in gt_pairs)
                acc = (rec / L_t) * 100.0 if L_t > 0 else 0.0

                sc_data[m_name]['acc'].append(acc)
                sc_data[m_name]['rmsd'].append(rmsd)
                sc_data[m_name]['time'].append(rt)

        sc_summary = {}
        for m_name in all_methods:
            accs = np.array(sc_data[m_name]['acc'])
            rmsds = np.array(sc_data[m_name]['rmsd'])
            times = np.array(sc_data[m_name]['time'])

            mean_acc = float(np.mean(accs))
            ci_low, ci_high = calc_ci95(accs)
            mean_rmsd, median_rmsd = float(np.mean(rmsds)), float(np.median(rmsds))
            pct_subang = float(np.mean(rmsds < 1.0) * 100.0)
            pct_ultra = float(np.mean(rmsds < 0.1) * 100.0)
            mean_time = float(np.mean(times))

            sc_summary[m_name] = {
                'mean_acc': mean_acc,
                'ci95_acc': [ci_low, ci_high],
                'mean_rmsd': mean_rmsd,
                'median_rmsd': median_rmsd,
                'pct_subang': pct_subang,
                'pct_ultra': pct_ultra,
                'mean_time_ms': mean_time
            }

            ci_str = f"{mean_acc:4.1f}% [{ci_low:4.1f}, {ci_high:4.1f}]"
            print(f"{m_name:<34s} | {ci_str:>22s} | {median_rmsd:8.4f} Å | {pct_subang:7.1f}% | {pct_ultra:7.1f}% | {mean_time:8.1f} ms")

        full_results[sc_name] = sc_summary

    # Save results to JSON
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_3d_metrics_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    print(f"\n[OK] Rigorous 3D Benchmark results successfully written to: {json_path}")


if __name__ == "__main__":
    execute_rigorous_3d_benchmark()
