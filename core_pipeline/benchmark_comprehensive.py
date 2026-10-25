"""
benchmark_comprehensive.py
========================================================================================
COMPREHENSIVE ABLATION & BASELINE COMPARISON BENCHMARK
========================================================================================

Compares our enhanced GNCCP solver against:
  1. Original paper's GNCCP (Yang et al. 2018) — top-K oracle, grid line search, linear schedule
  2. Partial Optimal Transport (POT) — unbalanced Sinkhorn with partial mass
  3. Fused Gromov-Wasserstein (FGW) — joint structural + feature transport

Benchmark scenarios (following IEEE TNNLS 2018 paper Section III-A):
  Scenario 1: Varying noise level σ (graph size fixed)
  Scenario 2: Varying problem size N (noise fixed)
  Scenario 3: Varying outlier count (noise fixed)
  Scenario 4: Varying edge density (noise fixed)

Each scenario runs multiple random seeds and reports mean ± std.
========================================================================================
"""

import os
import sys
import time
import json
import warnings
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Any, Optional

warnings.filterwarnings('ignore')

# Import POT BEFORE any stdout modification (JAX backend checks isatty)
try:
    import ot
    HAS_POT = True
except ImportError:
    HAS_POT = False
    print("[WARNING] POT library not installed. Run: pip install POT")

# Import our solvers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnccp_advanced import AdvancedGNCCP
from gnccp_generalized import UniversalGNCCP, MatchingResult


# ======================================================================================
# SYNTHETIC DATA GENERATOR (Following IEEE TNNLS 2018 Section III-A)
# ======================================================================================

def generate_synthetic_wcs_instance(M, N, L, sigma, edge_density, seed=None):
    """
    Generates a synthetic WCS matching instance following the IEEE paper protocol.
    
    1. Generate M points G and N points H uniformly in [0,1]^2
    2. Create ground-truth partial permutation X_gt with L "1"s
    3. Update L points in G by permuting H with X_gt + Gaussian noise
    4. Build edge weights from Euclidean distances
    5. Build sparse adjacency with given edge density
    6. Perturb graph structure by adding/removing (σ/2 * #edges) edges
    
    Returns:
        A_G, A_H: adjacency matrices
        C: unary cost matrix (Euclidean distances between points)
        X_gt: ground-truth partial permutation matrix
        L: matching size
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate point sets
    G_points = np.random.uniform(0, 1, (M, 2)).astype(np.float64)
    H_points = np.random.uniform(0, 1, (N, 2)).astype(np.float64)
    
    # Ground-truth correspondence: first L rows of G correspond to first L rows of H
    # (random permutation for the L matched nodes)
    gt_g_indices = np.random.choice(M, L, replace=False)
    gt_h_indices = np.random.choice(N, L, replace=False)
    
    X_gt = np.zeros((M, N), dtype=np.float64)
    for gi, hi in zip(gt_g_indices, gt_h_indices):
        X_gt[gi, hi] = 1.0
        # Update G point to be close to corresponding H point + noise
        G_points[gi] = H_points[hi] + np.random.normal(0, sigma, 2)
    
    # Build distance-based edge weights
    dist_G = np.linalg.norm(G_points[:, None] - G_points[None, :], axis=-1)
    dist_H = np.linalg.norm(H_points[:, None] - H_points[None, :], axis=-1)
    
    # Build sparse adjacency with given edge density
    # Edge exists if distance < threshold (set to achieve target density)
    n_possible_G = M * (M - 1) // 2
    n_possible_H = N * (N - 1) // 2
    
    # Use distance percentile to achieve target edge density
    triu_G = dist_G[np.triu_indices(M, k=1)]
    triu_H = dist_H[np.triu_indices(N, k=1)]
    
    threshold_G = np.percentile(triu_G, edge_density * 100) if len(triu_G) > 0 else 1.0
    threshold_H = np.percentile(triu_H, edge_density * 100) if len(triu_H) > 0 else 1.0
    
    A_G = np.where(dist_G < threshold_G, np.exp(-dist_G), 0.0).astype(np.float64)
    A_H = np.where(dist_H < threshold_H, np.exp(-dist_H), 0.0).astype(np.float64)
    np.fill_diagonal(A_G, 0.0)
    np.fill_diagonal(A_H, 0.0)
    
    # Perturb graph structure: add/remove (σ/2 * #edges) edges
    n_edges_G = np.sum(A_G > 0) // 2
    n_perturb = max(1, int(0.5 * sigma * n_edges_G))
    
    # Remove random edges
    edges_G = list(zip(*np.where(np.triu(A_G, k=1) > 0)))
    if len(edges_G) > n_perturb:
        remove_idx = np.random.choice(len(edges_G), min(n_perturb, len(edges_G)), replace=False)
        for idx in remove_idx:
            i, j = edges_G[idx]
            A_G[i, j] = A_G[j, i] = 0.0
    
    # Add random edges
    non_edges = list(zip(*np.where((np.triu(np.ones((M, M)), k=1) > 0) & (A_G == 0))))
    if len(non_edges) > 0:
        add_idx = np.random.choice(len(non_edges), min(n_perturb, len(non_edges)), replace=False)
        for idx in add_idx:
            i, j = non_edges[idx]
            w = np.exp(-dist_G[i, j])
            A_G[i, j] = A_G[j, i] = w
    
    # Unary cost matrix: Euclidean distance between all point pairs
    C = np.linalg.norm(G_points[:, None] - H_points[None, :], axis=-1).astype(np.float64)
    
    return A_G.astype(np.float32), A_H.astype(np.float32), C.astype(np.float32), X_gt, gt_g_indices, gt_h_indices


def compute_accuracy(result_matches, gt_g_indices, gt_h_indices):
    """Computes matching accuracy: fraction of ground-truth pairs correctly recovered."""
    gt_set = set(zip(gt_g_indices, gt_h_indices))
    correct = sum(1 for g, h in result_matches if (g, h) in gt_set)
    return correct / len(gt_g_indices) if len(gt_g_indices) > 0 else 0.0


# ======================================================================================
# BASELINE: PARTIAL OPTIMAL TRANSPORT (POT)
# ======================================================================================

class PartialOTMatcher:
    """
    Partial Optimal Transport baseline for WCS matching.
    
    Uses the POT library's partial_wasserstein to find correspondences
    between graph nodes, transporting only mass L out of total M+N.
    
    The transport plan T is a soft assignment matrix where T[i,j] > 0
    indicates node i in G is matched to node j in H.
    """
    
    def __init__(self, alpha=0.7, reg=0.05):
        self.alpha = alpha
        self.reg = reg  # Sinkhorn regularization
    
    def solve(self, A_G, A_H, C, L, gt_g=None, gt_h=None):
        """
        Solves WCS matching via partial optimal transport.
        
        Returns: matches list, objective value, runtime
        """
        if not HAS_POT:
            return [], float('inf'), 0.0
        
        t_start = time.perf_counter()
        M, N = C.shape
        
        # Uniform marginals
        a = np.ones(M, dtype=np.float64) / M
        b = np.ones(N, dtype=np.float64) / N
        
        # Cost matrix combining unary and structural distance
        C_ot = C.astype(np.float64).copy()
        c_max = C_ot.max()
        if c_max > 0:
            C_ot = C_ot / c_max  # Normalize to [0, 1]
        
        # Mass to transport (fraction of total)
        mass_fraction = L / min(M, N)
        
        try:
            # Partial Wasserstein transport
            T = ot.partial.partial_wasserstein(a, b, C_ot, m=mass_fraction)
        except Exception:
            try:
                # Fallback: entropic partial transport
                T = ot.partial.entropic_partial_wasserstein(
                    a, b, C_ot, reg=self.reg, m=mass_fraction)
            except Exception:
                runtime = time.perf_counter() - t_start
                return [], float('inf'), runtime
        
        # Discretize transport plan to binary matching
        # Scale T so it's comparable, then use Hungarian on -T
        T_scaled = T.copy()
        row_ind, col_ind = linear_sum_assignment(-T_scaled)
        
        # Keep top L pairs by transport mass
        pairs = [(T_scaled[r, c], r, c) for r, c in zip(row_ind, col_ind)]
        pairs.sort(key=lambda x: x[0], reverse=True)
        matches = [(r, c) for _, r, c in pairs[:L]]
        
        # Compute objective using same formula as GNCCP
        X_bin = np.zeros((M, N), dtype=np.float32)
        for r, c in matches:
            X_bin[r, c] = 1.0
        
        r_vec = X_bin.sum(axis=1)
        U = np.outer(r_vec, r_vec)
        diff = (U * A_G) - (X_bin @ A_H @ X_bin.T)
        struct_term = np.sum(diff ** 2)
        app_term = np.sum(C * X_bin)
        F_obj = float(self.alpha * struct_term + (1 - self.alpha) * app_term)
        
        runtime = time.perf_counter() - t_start
        return matches, F_obj, runtime


# ======================================================================================
# BASELINE: FUSED GROMOV-WASSERSTEIN (FGW)
# ======================================================================================

class FusedGromovWassersteinMatcher:
    """
    Fused Gromov-Wasserstein baseline for WCS matching.
    
    Jointly optimizes structural alignment (Gromov-Wasserstein on adjacency)
    and feature alignment (Wasserstein on unary costs).
    
    Uses POT's fused_gromov_wasserstein implementation.
    """
    
    def __init__(self, alpha_fgw=0.5, alpha_obj=0.7):
        self.alpha_fgw = alpha_fgw  # FGW trade-off between structure and features
        self.alpha_obj = alpha_obj  # For computing F objective
    
    def solve(self, A_G, A_H, C, L, gt_g=None, gt_h=None):
        """
        Solves WCS matching via fused Gromov-Wasserstein.
        
        Returns: matches list, objective value, runtime
        """
        if not HAS_POT:
            return [], float('inf'), 0.0
        
        t_start = time.perf_counter()
        M, N = C.shape
        
        # Uniform marginals
        p = np.ones(M, dtype=np.float64) / M
        q = np.ones(N, dtype=np.float64) / N
        
        # Normalize cost and adjacency
        C_fgw = C.astype(np.float64).copy()
        c_max = C_fgw.max()
        if c_max > 0:
            C_fgw = C_fgw / c_max
        
        A_G_d = A_G.astype(np.float64)
        A_H_d = A_H.astype(np.float64)
        
        # Normalize adjacency matrices
        norm_G = np.linalg.norm(A_G_d, 'fro')
        norm_H = np.linalg.norm(A_H_d, 'fro')
        if norm_G > 0: A_G_d = A_G_d / norm_G
        if norm_H > 0: A_H_d = A_H_d / norm_H
        
        try:
            T, log = ot.gromov.fused_gromov_wasserstein(
                C_fgw, A_G_d, A_H_d, p, q,
                alpha=self.alpha_fgw,
                log=True,
                verbose=False,
                max_iter=100
            )
        except Exception:
            try:
                # Fallback: entropic FGW
                T, log = ot.gromov.entropic_fused_gromov_wasserstein(
                    C_fgw, A_G_d, A_H_d, p, q,
                    epsilon=0.01,
                    alpha=self.alpha_fgw,
                    log=True,
                    verbose=False,
                    max_iter=100
                )
            except Exception:
                runtime = time.perf_counter() - t_start
                return [], float('inf'), runtime
        
        # Discretize: Hungarian on -T, keep top L
        row_ind, col_ind = linear_sum_assignment(-T)
        pairs = [(T[r, c], r, c) for r, c in zip(row_ind, col_ind)]
        pairs.sort(key=lambda x: x[0], reverse=True)
        matches = [(r, c) for _, r, c in pairs[:L]]
        
        # Compute objective using GNCCP's formula for fair comparison
        X_bin = np.zeros((M, N), dtype=np.float32)
        for r, c in matches:
            X_bin[r, c] = 1.0
        
        r_vec = X_bin.sum(axis=1)
        U = np.outer(r_vec, r_vec)
        diff = (U * A_G) - (X_bin @ A_H @ X_bin.T)
        struct_term = np.sum(diff ** 2)
        app_term = np.sum(C * X_bin)
        F_obj = float(self.alpha_obj * struct_term + (1 - self.alpha_obj) * app_term)
        
        runtime = time.perf_counter() - t_start
        return matches, F_obj, runtime


# ======================================================================================
# BENCHMARK RUNNER
# ======================================================================================

def run_single_experiment(methods, A_G, A_H, C, L, gt_g, gt_h):
    """Run all methods on one instance, return results dict."""
    results = {}
    for name, method_fn in methods.items():
        matches, obj, rt = method_fn(A_G, A_H, C, L, gt_g, gt_h)
        acc = compute_accuracy(matches, gt_g, gt_h)
        results[name] = {
            'accuracy': acc,
            'objective': obj,
            'runtime_ms': rt * 1000,
            'n_matches': len(matches),
        }
    return results


def run_scenario(scenario_name, param_name, param_values, methods,
                 base_M=30, base_N=35, base_L=25, base_sigma=0.03,
                 base_edge_density=0.5, n_seeds=10, 
                 vary_fn=None):
    """
    Run one benchmark scenario across a parameter sweep.
    """
    print(f"\n{'='*90}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*90}")
    
    all_results = {}
    
    for pval in param_values:
        # Apply parameter variation
        M, N, L, sigma, edge_density = base_M, base_N, base_L, base_sigma, base_edge_density
        if vary_fn:
            M, N, L, sigma, edge_density = vary_fn(pval, M, N, L, sigma, edge_density)
        
        seed_results = {name: {'accuracy': [], 'objective': [], 'runtime_ms': []} 
                       for name in methods}
        
        for seed in range(n_seeds):
            A_G, A_H, C, X_gt, gt_g, gt_h = generate_synthetic_wcs_instance(
                M, N, L, sigma, edge_density, seed=seed * 1000 + hash(str(pval)) % 1000)
            
            result = run_single_experiment(methods, A_G, A_H, C, L, gt_g, gt_h)
            
            for name in methods:
                seed_results[name]['accuracy'].append(result[name]['accuracy'])
                seed_results[name]['objective'].append(result[name]['objective'])
                seed_results[name]['runtime_ms'].append(result[name]['runtime_ms'])
        
        # Print results for this parameter value
        print(f"\n  {param_name} = {pval}")
        print(f"  {'Method':<45s} | {'Acc (mean±std)':>16s} | {'Obj (mean)':>12s} | {'Time (mean)':>12s}")
        print(f"  {'-'*95}")
        
        for name in methods:
            acc_arr = np.array(seed_results[name]['accuracy'])
            obj_arr = np.array(seed_results[name]['objective'])
            rt_arr = np.array(seed_results[name]['runtime_ms'])
            
            print(f"  {name:<45s} | {acc_arr.mean()*100:5.1f}% ± {acc_arr.std()*100:4.1f}% "
                  f"| {obj_arr.mean():10.4f}   | {rt_arr.mean():8.1f} ms")
        
        all_results[str(pval)] = {
            name: {
                'acc_mean': float(np.mean(seed_results[name]['accuracy'])),
                'acc_std': float(np.std(seed_results[name]['accuracy'])),
                'obj_mean': float(np.mean(seed_results[name]['objective'])),
                'obj_std': float(np.std(seed_results[name]['objective'])),
                'rt_mean': float(np.mean(seed_results[name]['runtime_ms'])),
                'rt_std': float(np.std(seed_results[name]['runtime_ms'])),
            }
            for name in methods
        }
    
    return all_results


# ======================================================================================
# MAIN BENCHMARK
# ======================================================================================

def main():
    print("=" * 90)
    print("COMPREHENSIVE ABLATION & BASELINE COMPARISON BENCHMARK")
    print("Following IEEE TNNLS 2018 Section III-A experimental protocol")
    print("=" * 90)
    
    # ---- Build method dictionary ----
    
    # 1. Flagship Engine: AdvancedGNCCP (gnccp_advanced.py)
    solver_advanced = AdvancedGNCCP(
        alpha=0.7, max_iter=8, fw_max_iter=3, tol=1e-3,
        normalization='none', line_search='quartic',
        oracle='exact_augmented', local_refinement=True, enable_exchange_moves=True)

    # 2. Universal GNCCP (gnccp_generalized.py)
    solver_full = UniversalGNCCP(
        alpha=0.7, max_iter=8, fw_max_iter=3, tol=1e-3,
        oracle='dummy_padded', line_search='quartic', schedule='cosine',
        adaptive_schedule=True, local_refinement=True, n_restarts=1)
    
    # 3. Universal GNCCP with 3 restarts
    solver_3restart = UniversalGNCCP(
        alpha=0.7, max_iter=8, fw_max_iter=3, tol=1e-3,
        oracle='dummy_padded', line_search='quartic', schedule='cosine',
        adaptive_schedule=True, local_refinement=True, n_restarts=3)
    
    # 4. Original paper baseline
    solver_original = UniversalGNCCP(
        alpha=0.7, max_iter=15, fw_max_iter=8,
        oracle='top_k_heuristic', line_search='grid', schedule='linear',
        adaptive_schedule=False, local_refinement=False, n_restarts=1)
    
    # POT baseline
    pot_matcher = PartialOTMatcher(alpha=0.7, reg=0.05)
    
    # FGW baseline
    fgw_matcher = FusedGromovWassersteinMatcher(alpha_fgw=0.5, alpha_obj=0.7)
    
    def gnccp_method(solver):
        def fn(A_G, A_H, C, L, gt_g, gt_h):
            res = solver.solve(A_G, A_H, C=C, L=L)
            return res.matches, res.objective_score, res.runtime_sec
        return fn
    
    methods = {
        'Fast-WCS Advanced (gnccp_advanced.py)': gnccp_method(solver_advanced),
        'Universal GNCCP (gnccp_generalized.py)': gnccp_method(solver_full),
        'Universal + 3 restarts': gnccp_method(solver_3restart),
        'ORIGINAL PAPER (topk+grid+linear)': gnccp_method(solver_original),
    }
    
    if HAS_POT:
        methods['Partial Optimal Transport (POT)'] = lambda A_G, A_H, C, L, gt_g, gt_h: pot_matcher.solve(A_G, A_H, C, L, gt_g, gt_h)
        methods['Fused Gromov-Wasserstein (FGW)'] = lambda A_G, A_H, C, L, gt_g, gt_h: fgw_matcher.solve(A_G, A_H, C, L, gt_g, gt_h)
    
    all_scenario_results = {}
    
    # ======================================================================
    # SCENARIO 1: Varying Noise Level σ
    # M=30, N=35, L=25, edge_density=0.5, σ from 0 to 0.1
    # ======================================================================
    def vary_noise(sigma, M, N, L, s, ed):
        return M, N, L, sigma, ed
    
    noise_values = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
    r1 = run_scenario(
        "1: Varying Noise Level σ",
        "σ", noise_values, methods,
        base_M=30, base_N=35, base_L=25, base_sigma=0.03, base_edge_density=0.5,
        n_seeds=10, vary_fn=vary_noise
    )
    all_scenario_results['noise_level'] = r1
    
    # ======================================================================
    # SCENARIO 2: Varying Problem Size N
    # σ=0.03, edge_density=0.5, N from 10 to 50
    # ======================================================================
    def vary_size(n, M, N, L, s, ed):
        return n, n + 5, n - 5, s, ed
    
    size_values = [10, 15, 20, 25, 30, 40, 50]
    r2 = run_scenario(
        "2: Varying Problem Size N",
        "M", size_values, methods,
        base_sigma=0.03, base_edge_density=0.5,
        n_seeds=10, vary_fn=vary_size
    )
    all_scenario_results['problem_size'] = r2
    
    # ======================================================================
    # SCENARIO 3: Varying Outlier Count
    # M=30, σ=0.03, edge_density=0.5, outliers from 0 to 15
    # ======================================================================
    def vary_outliers(n_outliers, M, N, L, s, ed):
        return 30, 30 + n_outliers, 30 - 5, s, ed
    
    outlier_values = [0, 3, 5, 8, 10, 15]
    r3 = run_scenario(
        "3: Varying Outlier Count (N - L extra nodes in H)",
        "Outliers", outlier_values, methods,
        base_sigma=0.03, base_edge_density=0.5,
        n_seeds=10, vary_fn=vary_outliers
    )
    all_scenario_results['outlier_count'] = r3
    
    # ======================================================================
    # SCENARIO 4: Varying Edge Density
    # M=30, N=35, L=25, σ=0.03, density from 0.1 to 0.9
    # ======================================================================
    def vary_density(density, M, N, L, s, ed):
        return M, N, L, s, density
    
    density_values = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    r4 = run_scenario(
        "4: Varying Edge Density",
        "Density", density_values, methods,
        base_M=30, base_N=35, base_L=25, base_sigma=0.03,
        n_seeds=10, vary_fn=vary_density
    )
    all_scenario_results['edge_density'] = r4
    
    # ======================================================================
    # SAVE RESULTS
    # ======================================================================
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'results', 'benchmark_comprehensive.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(all_scenario_results, f, indent=2)
    print(f"\n[SAVED] Full results to: {os.path.abspath(results_path)}")
    
    # ======================================================================
    # SUMMARY TABLE
    # ======================================================================
    print("\n" + "=" * 90)
    print("OVERALL SUMMARY: MEAN ACCURACY ACROSS ALL SCENARIOS")
    print("=" * 90)
    
    method_totals = {name: {'acc': [], 'obj': [], 'rt': []} for name in methods}
    
    for scenario_name, scenario_data in all_scenario_results.items():
        for param_val, method_data in scenario_data.items():
            for method_name, stats in method_data.items():
                method_totals[method_name]['acc'].append(stats['acc_mean'])
                method_totals[method_name]['obj'].append(stats['obj_mean'])
                method_totals[method_name]['rt'].append(stats['rt_mean'])
    
    print(f"\n  {'Method':<50s} | {'Mean Acc':>8s} | {'Mean Obj':>9s} | {'Mean Time':>10s}")
    print(f"  {'-'*85}")
    
    for name in methods:
        accs = method_totals[name]['acc']
        objs = method_totals[name]['obj']
        rts = method_totals[name]['rt']
        if accs:
            print(f"  {name:<50s} | {np.mean(accs)*100:6.1f}%  | {np.mean(objs):9.4f} | {np.mean(rts):8.1f} ms")
    
    # ======================================================================
    # WINNER DETERMINATION
    # ======================================================================
    print("\n" + "=" * 90)
    print("WINNER ANALYSIS")
    print("=" * 90)
    
    best_acc_method = max(methods.keys(), 
                         key=lambda n: np.mean(method_totals[n]['acc']) if method_totals[n]['acc'] else 0)
    best_obj_method = min(methods.keys(),
                         key=lambda n: np.mean(method_totals[n]['obj']) if method_totals[n]['obj'] else float('inf'))
    best_rt_method = min(methods.keys(),
                        key=lambda n: np.mean(method_totals[n]['rt']) if method_totals[n]['rt'] else float('inf'))
    
    print(f"\n  Best Accuracy:  {best_acc_method}")
    print(f"                  → {np.mean(method_totals[best_acc_method]['acc'])*100:.1f}% mean across all scenarios")
    print(f"\n  Best Objective: {best_obj_method}")
    print(f"                  → {np.mean(method_totals[best_obj_method]['obj']):.4f} mean F(X)")
    print(f"\n  Fastest:        {best_rt_method}")
    print(f"                  → {np.mean(method_totals[best_rt_method]['rt']):.1f} ms mean")
    
    print("\n" + "=" * 90)
    print("BENCHMARK COMPLETE")
    print("=" * 90)
    
    return all_scenario_results


if __name__ == '__main__':
    main()
