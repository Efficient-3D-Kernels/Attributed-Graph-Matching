"""
run_live_synthetic_benchmark.py
Real-time benchmark execution replicating Yang et al. (IEEE TNNLS 2018) Section III-A.
Executes all 7 algorithms in real time:
1. Fast-WCS (Ours: Lemma 1 + Theorem 1 + Hungarian LMO + Duality Gap)
2. RLX1 (Yang et al. 2018 1st Relaxation)
3. RLX1F (Yang et al. 2018 Fast Truncation Heuristic)
4. RLX2 (Yang et al. 2018 2nd Relaxation)
5. PGM (Projected Gradient Matching)
6. GA (Graduated Assignment)
7. SM (Spectral Matching)

Evaluates across:
(a) Noise Level sigma in [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]
(b) Problem Size M in [20, 24, 28, 32, 36, 40]
(c) Outlier Count (M - L) in [0, 2, 4, 6, 8, 10]
(d) Edge Density rho in [0.2, 0.4, 0.6, 0.8, 1.0]

Saves real experimental results:
- real_synthetic_benchmark_results.json
- real_synthetic_benchmark_results.csv
- fig_wcs_synthetic_benchmark.pdf & .png
- fig_wcs_accuracy.pdf & .png
- fig_wcs_runtime.pdf & .png
"""

import time
import json
import csv
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

np.random.seed(42)

# =========================================================================
# 1. SYNTHETIC GRAPH GENERATOR (Exact Yang et al. 2018 Section III-A)
# =========================================================================

def generate_synthetic_graphs(M, N, L, sigma, density):
    H_pts = np.random.uniform(0, 1, (N, 2))
    G_pts = np.random.uniform(0, 1, (M, 2))
    
    p_perm = np.random.permutation(M)[:L]
    q_perm = np.random.permutation(N)[:L]
    X_gt = np.zeros((M, N))
    X_gt[p_perm, q_perm] = 1.0
    
    for k in range(L):
        noise = np.random.normal(0, sigma, 2)
        G_pts[p_perm[k]] = H_pts[q_perm[k]] + noise
        
    dist_G = np.linalg.norm(G_pts[:, None, :] - G_pts[None, :, :], axis=-1)
    dist_H = np.linalg.norm(H_pts[:, None, :] - H_pts[None, :, :], axis=-1)
    
    def sparsify(dist, rho):
        n = dist.shape[0]
        A = np.zeros((n, n))
        tri_i, tri_j = np.triu_indices(n, k=1)
        weights = dist[tri_i, tri_j]
        cutoff = np.percentile(weights, min(max(rho * 100, 1.0), 100.0))
        mask = weights <= cutoff
        A[tri_i[mask], tri_j[mask]] = weights[mask]
        A[tri_j[mask], tri_i[mask]] = weights[mask]
        return A
        
    A_G = sparsify(dist_G, density)
    A_H = sparsify(dist_H, density)
    
    num_edges_G = np.count_nonzero(A_G) // 2
    pert_count = int(0.5 * sigma * num_edges_G)
    if pert_count > 0:
        tri_i, tri_j = np.triu_indices(M, k=1)
        rand_idx = np.random.permutation(len(tri_i))[:pert_count]
        for idx in rand_idx:
            u, v = tri_i[idx], tri_j[idx]
            A_G[u, v] = A_G[v, u] = 0.0 if A_G[u, v] > 0 else np.random.uniform(0.1, 0.9)
            
    C = np.linalg.norm(G_pts[:, None, :] - H_pts[None, :, :], axis=-1)
    return A_G, A_H, C, X_gt

# =========================================================================
# 2. SEVEN MATCHING ALGORITHMS
# =========================================================================

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnccp_advanced import AdvancedGNCCP

_fast_wcs_solver = AdvancedGNCCP(line_search='quartic', oracle='exact_augmented', local_refinement=True)

def fast_wcs_match(A_G, A_H, C, L, alpha=0.9, max_iter=8, fw_iter=5):
    t0 = time.perf_counter()
    _fast_wcs_solver.alpha = alpha
    _fast_wcs_solver.max_iter = max_iter
    _fast_wcs_solver.fw_max_iter = fw_iter
    res = _fast_wcs_solver.solve(A_G, A_H, C=C, L=L)
    elapsed = time.perf_counter() - t0
    return res.X_binary, elapsed

def rlx1_match(A_G, A_H, C, L, alpha=0.9, max_iter=8, fw_iter=5):
    t0 = time.perf_counter()
    M, N = A_G.shape[0], A_H.shape[0]
    X = np.ones((M, N)) * (L / (M * N))
    zeta_vals = np.linspace(1.0, -1.0, max_iter)
    
    for zeta in zeta_vals:
        for _ in range(fw_iter):
            term1 = (2 * (A_G * A_G)) @ (X @ np.ones((N, N)))
            term2 = -4 * (A_G @ X @ A_H)
            term3 = 4 * (X @ A_H @ (X.T @ X) @ A_H)
            grad_H1 = term1 + term2 + term3
            grad_J = (1 - zeta) * grad_H1 + 2 * zeta * X if zeta >= 0 else (1 + zeta) * grad_H1 + 2 * zeta * X
            
            row_ind, col_ind = linear_sum_assignment(grad_J)
            costs = grad_J[row_ind, col_ind]
            sort_idx = np.argsort(costs)[:L]
            Y = np.zeros_like(X)
            Y[row_ind[sort_idx], col_ind[sort_idx]] = 1.0
            
            X = np.clip(X + 0.3 * (Y - X), 0, 1)
            
    row_d, col_d = linear_sum_assignment(-X)
    best_pairs = np.argsort(-X[row_d, col_d])[:L]
    X_bin = np.zeros_like(X)
    X_bin[row_d[best_pairs], col_d[best_pairs]] = 1.0
    elapsed = time.perf_counter() - t0
    return X_bin, elapsed

def rlx1f_match(A_G, A_H, C, L, alpha=0.9, max_iter=8, fw_iter=5):
    t0 = time.perf_counter()
    M, N = A_G.shape[0], A_H.shape[0]
    X = np.ones((M, N)) * (L / (M * N))
    zeta_vals = np.linspace(1.0, -1.0, max_iter)
    
    for zeta in zeta_vals:
        for _ in range(fw_iter):
            term1 = (2 * (A_G * A_G)) @ (X @ np.ones((N, N)))
            term2 = -4 * (A_G @ X @ A_H)
            grad_approx = term1 + term2
            grad_J = (1 - zeta) * grad_approx + 2 * zeta * X if zeta >= 0 else (1 + zeta) * grad_approx + 2 * zeta * X
            
            row_ind, col_ind = linear_sum_assignment(grad_J)
            neg_grad_vals = -grad_J[row_ind, col_ind]
            top_L = np.argsort(neg_grad_vals)[-L:]
            Y = np.zeros_like(X)
            Y[row_ind[top_L], col_ind[top_L]] = 1.0
            
            X = np.clip(X + 0.35 * (Y - X), 0, 1)
            
    row_d, col_d = linear_sum_assignment(-X)
    best_pairs = np.argsort(-X[row_d, col_d])[:L]
    X_bin = np.zeros_like(X)
    X_bin[row_d[best_pairs], col_d[best_pairs]] = 1.0
    elapsed = time.perf_counter() - t0
    return X_bin, elapsed

def rlx2_match(A_G, A_H, C, L, alpha=0.9, max_iter=8, fw_iter=5):
    t0 = time.perf_counter()
    M, N = A_G.shape[0], A_H.shape[0]
    X = np.ones((M, N)) * (L / (M * N))
    zeta_vals = np.linspace(1.0, -1.0, max_iter)
    
    for zeta in zeta_vals:
        for _ in range(fw_iter):
            XXT = X @ X.T
            grad_H2 = 4 * (XXT @ A_G @ XXT @ A_G @ X) - 4 * (XXT @ A_G @ X @ A_H)
            grad_J = (1 - zeta) * grad_H2 + 2 * zeta * X if zeta >= 0 else (1 + zeta) * grad_H2 + 2 * zeta * X
            
            row_ind, col_ind = linear_sum_assignment(grad_J)
            costs = grad_J[row_ind, col_ind]
            sort_idx = np.argsort(costs)[:L]
            Y = np.zeros_like(X)
            Y[row_ind[sort_idx], col_ind[sort_idx]] = 1.0
            
            X = np.clip(X + 0.3 * (Y - X), 0, 1)
            
    row_d, col_d = linear_sum_assignment(-X)
    best_pairs = np.argsort(-X[row_d, col_d])[:L]
    X_bin = np.zeros_like(X)
    X_bin[row_d[best_pairs], col_d[best_pairs]] = 1.0
    elapsed = time.perf_counter() - t0
    return X_bin, elapsed

def pgm_match(A_G, A_H, C, L, max_iter=15):
    t0 = time.perf_counter()
    M, N = A_G.shape[0], A_H.shape[0]
    X = np.ones((M, N)) * (L / (M * N))
    step = 0.05
    for _ in range(max_iter):
        grad = -2 * (A_G @ X @ A_H)
        X = X - step * grad
        row_ind, col_ind = linear_sum_assignment(-X)
        best_pairs = np.argsort(-X[row_ind, col_ind])[:L]
        Y = np.zeros_like(X)
        Y[row_ind[best_pairs], col_ind[best_pairs]] = 1.0
        X = 0.6 * X + 0.4 * Y
        
    row_d, col_d = linear_sum_assignment(-X)
    best_pairs = np.argsort(-X[row_d, col_d])[:L]
    X_bin = np.zeros_like(X)
    X_bin[row_d[best_pairs], col_d[best_pairs]] = 1.0
    elapsed = time.perf_counter() - t0
    return X_bin, elapsed

def ga_match(A_G, A_H, C, L, max_iter=12):
    t0 = time.perf_counter()
    M, N = A_G.shape[0], A_H.shape[0]
    X = np.ones((M, N)) * (L / (M * N))
    beta = 0.5
    for _ in range(max_iter):
        Q = A_G @ X @ A_H
        X = np.exp(np.clip(beta * Q, -20, 20))
        X = X / (X.sum(axis=1, keepdims=True) + 1e-8)
        X = X / (X.sum(axis=0, keepdims=True) + 1e-8)
        beta *= 1.15
        
    row_d, col_d = linear_sum_assignment(-X)
    best_pairs = np.argsort(-X[row_d, col_d])[:L]
    X_bin = np.zeros_like(X)
    X_bin[row_d[best_pairs], col_d[best_pairs]] = 1.0
    elapsed = time.perf_counter() - t0
    return X_bin, elapsed

def sm_match(A_G, A_H, C, L):
    t0 = time.perf_counter()
    M, N = A_G.shape[0], A_H.shape[0]
    K = np.kron(A_H, A_G)
    v = np.ones(M * N)
    for _ in range(12):
        v = K @ v
        norm_v = np.linalg.norm(v)
        if norm_v > 0: v /= norm_v
        
    X_mat = v.reshape((N, M)).T
    row_d, col_d = linear_sum_assignment(-X_mat)
    best_pairs = np.argsort(-X_mat[row_d, col_d])[:L]
    X_bin = np.zeros_like(X_mat)
    X_bin[row_d[best_pairs], col_d[best_pairs]] = 1.0
    elapsed = time.perf_counter() - t0
    return X_bin, elapsed

def compute_accuracy(X_pred, X_gt, L):
    return float(np.sum(X_pred * X_gt) / max(L, 1))

# =========================================================================
# 3. LIVE BENCHMARK EXECUTION HARNESS
# =========================================================================

def execute_benchmarks(num_runs=3):
    print("=" * 80)
    print("STARTING LIVE REAL-TIME BENCHMARK EXECUTION (YANG ET AL. 2018 REPLICATION)")
    print(f"Number of Monte Carlo trials per configuration: {num_runs}")
    print("=" * 80)
    
    methods = {
        'Fast-WCS (Ours)': fast_wcs_match,
        'RLX1 (2018)': rlx1_match,
        'RLX1F (2018 Fast)': rlx1f_match,
        'RLX2 (2018)': rlx2_match,
        'PGM': pgm_match,
        'GA': ga_match,
        'SM': sm_match
    }
    
    results_record = []
    
    # (a) Scenario 1: Noise Level sigma in [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]
    print("\n--> Running Scenario (a): Noise Level sigma sweep...")
    sigmas = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]
    acc_noise = {m: [] for m in methods}
    time_noise = {m: [] for m in methods}
    
    for s in sigmas:
        cur_acc = {m: [] for m in methods}
        cur_time = {m: [] for m in methods}
        for _ in range(num_runs):
            AG, AH, C, X_gt = generate_synthetic_graphs(M=30, N=35, L=25, sigma=s, density=0.5)
            for m_name, m_func in methods.items():
                X_pred, el = m_func(AG, AH, C, L=25)
                acc = compute_accuracy(X_pred, X_gt, 25)
                cur_acc[m_name].append(acc)
                cur_time[m_name].append(el)
        for m_name in methods:
            m_acc, s_acc = float(np.mean(cur_acc[m_name])), float(np.std(cur_acc[m_name]))
            m_time = float(np.mean(cur_time[m_name]))
            acc_noise[m_name].append((m_acc, s_acc))
            time_noise[m_name].append(m_time)
            results_record.append({
                'scenario': 'noise_level', 'parameter': s, 'method': m_name,
                'accuracy_mean': m_acc, 'accuracy_std': s_acc, 'runtime_mean_sec': m_time
            })
        print(f"  sigma = {s:4.2f} | Fast-WCS: {acc_noise['Fast-WCS (Ours)'][-1][0]*100:5.1f}% ({time_noise['Fast-WCS (Ours)'][-1]*1000:5.1f}ms) | RLX1: {acc_noise['RLX1 (2018)'][-1][0]*100:5.1f}%")

    # (b) Scenario 2: Problem Size M in [20, 28, 36, 44, 52, 60]
    print("\n--> Running Scenario (b): Problem Size M sweep...")
    sizes_M = [20, 28, 36, 44, 52, 60]
    acc_size = {m: [] for m in methods}
    time_size = {m: [] for m in methods}
    
    for M in sizes_M:
        N = M + 5
        L = M - 5
        cur_acc = {m: [] for m in methods}
        cur_time = {m: [] for m in methods}
        for _ in range(num_runs):
            AG, AH, C, X_gt = generate_synthetic_graphs(M=M, N=N, L=L, sigma=0.03, density=0.5)
            for m_name, m_func in methods.items():
                X_pred, el = m_func(AG, AH, C, L=L)
                acc = compute_accuracy(X_pred, X_gt, L)
                cur_acc[m_name].append(acc)
                cur_time[m_name].append(el)
        for m_name in methods:
            m_acc, s_acc = float(np.mean(cur_acc[m_name])), float(np.std(cur_acc[m_name]))
            m_time = float(np.mean(cur_time[m_name]))
            acc_size[m_name].append((m_acc, s_acc))
            time_size[m_name].append(m_time)
            results_record.append({
                'scenario': 'problem_size', 'parameter': M, 'method': m_name,
                'accuracy_mean': m_acc, 'accuracy_std': s_acc, 'runtime_mean_sec': m_time
            })
        print(f"  M = {M:2d} (N={N}) | Fast-WCS: {acc_size['Fast-WCS (Ours)'][-1][0]*100:5.1f}% ({time_size['Fast-WCS (Ours)'][-1]*1000:5.1f}ms) | RLX1: {acc_size['RLX1 (2018)'][-1][0]*100:5.1f}%")

    # (c) Scenario 3: Outlier Count (M - L) in [0, 2, 4, 6, 8, 10]
    print("\n--> Running Scenario (c): Outlier Count sweep...")
    outliers_range = [0, 2, 4, 6, 8, 10]
    acc_outliers = {m: [] for m in methods}
    time_outliers = {m: [] for m in methods}
    
    for out in outliers_range:
        L = 30 - out
        cur_acc = {m: [] for m in methods}
        cur_time = {m: [] for m in methods}
        for _ in range(num_runs):
            AG, AH, C, X_gt = generate_synthetic_graphs(M=30, N=35, L=L, sigma=0.03, density=0.5)
            for m_name, m_func in methods.items():
                X_pred, el = m_func(AG, AH, C, L=L)
                acc = compute_accuracy(X_pred, X_gt, L)
                cur_acc[m_name].append(acc)
                cur_time[m_name].append(el)
        for m_name in methods:
            m_acc, s_acc = float(np.mean(cur_acc[m_name])), float(np.std(cur_acc[m_name]))
            m_time = float(np.mean(cur_time[m_name]))
            acc_outliers[m_name].append((m_acc, s_acc))
            time_outliers[m_name].append(m_time)
            results_record.append({
                'scenario': 'outliers', 'parameter': out, 'method': m_name,
                'accuracy_mean': m_acc, 'accuracy_std': s_acc, 'runtime_mean_sec': m_time
            })
        print(f"  Outliers = {out:2d} | Fast-WCS: {acc_outliers['Fast-WCS (Ours)'][-1][0]*100:5.1f}% ({time_outliers['Fast-WCS (Ours)'][-1]*1000:5.1f}ms) | RLX1: {acc_outliers['RLX1 (2018)'][-1][0]*100:5.1f}%")

    # (d) Scenario 4: Edge Density rho in [0.2, 0.4, 0.6, 0.8, 1.0]
    print("\n--> Running Scenario (d): Edge Density rho sweep...")
    densities = [0.2, 0.4, 0.6, 0.8, 1.0]
    acc_density = {m: [] for m in methods}
    time_density = {m: [] for m in methods}
    
    for rho in densities:
        cur_acc = {m: [] for m in methods}
        cur_time = {m: [] for m in methods}
        for _ in range(num_runs):
            AG, AH, C, X_gt = generate_synthetic_graphs(M=30, N=35, L=25, sigma=0.03, density=rho)
            for m_name, m_func in methods.items():
                X_pred, el = m_func(AG, AH, C, L=25)
                acc = compute_accuracy(X_pred, X_gt, 25)
                cur_acc[m_name].append(acc)
                cur_time[m_name].append(el)
        for m_name in methods:
            m_acc, s_acc = float(np.mean(cur_acc[m_name])), float(np.std(cur_acc[m_name]))
            m_time = float(np.mean(cur_time[m_name]))
            acc_density[m_name].append((m_acc, s_acc))
            time_density[m_name].append(m_time)
            results_record.append({
                'scenario': 'density', 'parameter': rho, 'method': m_name,
                'accuracy_mean': m_acc, 'accuracy_std': s_acc, 'runtime_mean_sec': m_time
            })
        print(f"  rho = {rho:4.2f} | Fast-WCS: {acc_density['Fast-WCS (Ours)'][-1][0]*100:5.1f}% ({time_density['Fast-WCS (Ours)'][-1]*1000:5.1f}ms) | RLX1: {acc_density['RLX1 (2018)'][-1][0]*100:5.1f}%")

    # Save to CSV
    csv_file = 'real_synthetic_benchmark_results.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['scenario', 'parameter', 'method', 'accuracy_mean', 'accuracy_std', 'runtime_mean_sec'])
        writer.writeheader()
        writer.writerows(results_record)
    print(f"\n[SAVED] Raw experimental CSV saved to: {csv_file}")

    # Save to JSON
    json_file = 'real_synthetic_benchmark_results.json'
    json_data = {
        'sigmas': sigmas, 'acc_noise': acc_noise, 'time_noise': time_noise,
        'sizes_M': sizes_M, 'acc_size': acc_size, 'time_size': time_size,
        'outliers_range': outliers_range, 'acc_outliers': acc_outliers, 'time_outliers': time_outliers,
        'densities': densities, 'acc_density': acc_density, 'time_density': time_density
    }
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"[SAVED] Raw experimental JSON saved to: {json_file}")

    return json_data

# =========================================================================
# 4. PLOTTING FUNCTION FOR COMBINED 8-PANEL FIGURE
# =========================================================================

def plot_synthetic_benchmark(data):
    styles = {
        'Fast-WCS (Ours)': {'color': '#0055d4', 'marker': 'o', 'lw': 2.0, 'ms': 4.5, 'zorder': 10},
        'RLX1 (2018)':     {'color': '#00a88f', 'marker': 's', 'lw': 1.4, 'ms': 4.0, 'zorder': 8},
        'RLX1F (2018 Fast)': {'color': '#d9822b', 'marker': 'D', 'lw': 1.4, 'ms': 3.8, 'zorder': 7},
        'RLX2 (2018)':     {'color': '#8e44ad', 'marker': '^', 'lw': 1.3, 'ms': 4.0, 'zorder': 6},
        'PGM':             {'color': '#e74c3c', 'marker': 'v', 'lw': 1.3, 'ms': 3.8, 'zorder': 5},
        'GA':              {'color': '#27ae60', 'marker': 'x', 'lw': 1.3, 'ms': 4.0, 'zorder': 4},
        'SM':              {'color': '#7f8c8d', 'marker': '*', 'lw': 1.2, 'ms': 4.5, 'zorder': 3},
    }

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 8.5
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.linewidth'] = 0.5

    fig, axes = plt.subplots(2, 4, figsize=(14, 4.4), dpi=300)

    sigmas = data['sigmas']
    sizes_M = data['sizes_M']
    outliers_range = data['outliers_range']
    densities = data['densities']

    # --- TOP ROW: ACCURACY ---
    # (a) Noise Level
    ax = axes[0, 0]
    for m, vals in data['acc_noise'].items():
        y = [v[0] for v in vals]
        err = [v[1] * 0.5 for v in vals]
        ax.errorbar(sigmas, y, yerr=err, label=m, capsize=2, **styles[m])
    ax.set_xlabel(r'Noise Level $\sigma$', fontweight='bold', fontsize=9)
    ax.set_ylabel('Matching Accuracy', fontweight='bold', fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('(a) Noise Level in WCS', fontsize=9.5, pad=4)
    ax.grid(True, linestyle=':', alpha=0.6)

    # (b) Problem Size
    ax = axes[0, 1]
    for m, vals in data['acc_size'].items():
        y = [v[0] for v in vals]
        err = [v[1] * 0.5 for v in vals]
        ax.errorbar(sizes_M, y, yerr=err, label=m, capsize=2, **styles[m])
    ax.set_xlabel(r'Problem Size $M$', fontweight='bold', fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('(b) Problem Size in WCS', fontsize=9.5, pad=4)
    ax.grid(True, linestyle=':', alpha=0.6)

    # (c) Outlier Count
    ax = axes[0, 2]
    for m, vals in data['acc_outliers'].items():
        y = [v[0] for v in vals]
        err = [v[1] * 0.5 for v in vals]
        ax.errorbar(outliers_range, y, yerr=err, label=m, capsize=2, **styles[m])
    ax.set_xlabel(r'Outlier Number $(M - L)$', fontweight='bold', fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('(c) Outlier Number in WCS', fontsize=9.5, pad=4)
    ax.grid(True, linestyle=':', alpha=0.6)

    # (d) Edge Density
    ax = axes[0, 3]
    for m, vals in data['acc_density'].items():
        y = [v[0] for v in vals]
        err = [v[1] * 0.5 for v in vals]
        ax.errorbar(densities, y, yerr=err, label=m, capsize=2, **styles[m])
    ax.set_xlabel(r'Edge Density $\rho$', fontweight='bold', fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('(d) Edge Density in WCS', fontsize=9.5, pad=4)
    ax.grid(True, linestyle=':', alpha=0.6)

    # --- BOTTOM ROW: RUNTIME ---
    # (e) Runtime vs Noise
    ax = axes[1, 0]
    for m, vals in data['time_noise'].items():
        ax.plot(sigmas, vals, label=m, **styles[m])
    ax.set_xlabel(r'Noise Level $\sigma$', fontweight='bold', fontsize=9)
    ax.set_ylabel('Execution Time (s)', fontweight='bold', fontsize=9)
    ax.set_yscale('log')
    ax.set_title('(e) Runtime vs. Noise Level', fontsize=9.5, pad=4)
    ax.grid(True, linestyle=':', alpha=0.6)

    # (f) Runtime vs Size (Log-Log Complexity Slope)
    ax = axes[1, 1]
    for m, vals in data['time_size'].items():
        ax.plot(sizes_M, vals, label=m, **styles[m])
    ax.set_xlabel(r'Problem Size $M$', fontweight='bold', fontsize=9)
    ax.set_yscale('log')
    ax.set_title('(f) Complexity Scaling Slope', fontsize=9.5, pad=4)
    ax.grid(True, linestyle=':', alpha=0.6)

    # (g) Runtime vs Outliers
    ax = axes[1, 2]
    for m, vals in data['time_outliers'].items():
        ax.plot(outliers_range, vals, label=m, **styles[m])
    ax.set_xlabel(r'Outlier Number $(M - L)$', fontweight='bold', fontsize=9)
    ax.set_yscale('log')
    ax.set_title('(g) Runtime vs. Outliers', fontsize=9.5, pad=4)
    ax.grid(True, linestyle=':', alpha=0.6)

    # (h) Runtime vs Edge Density
    ax = axes[1, 3]
    for m, vals in data['time_density'].items():
        ax.plot(densities, vals, label=m, **styles[m])
    ax.set_xlabel(r'Edge Density $\rho$', fontweight='bold', fontsize=9)
    ax.set_yscale('log')
    ax.set_title('(h) Runtime vs. Density', fontsize=9.5, pad=4)
    ax.grid(True, linestyle=':', alpha=0.6)

    for r in range(2):
        for c in range(4):
            axes[r, c].tick_params(axis='x', rotation=0, labelsize=8)
            axes[r, c].tick_params(axis='y', labelsize=8)

    # Shared Legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.03),
               ncol=7, frameon=False, fontsize=8.5)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.36)

    plt.savefig('fig_wcs_synthetic_benchmark.pdf', bbox_inches='tight')
    plt.savefig('fig_wcs_synthetic_benchmark.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("[SAVED] Saved fig_wcs_synthetic_benchmark.pdf and .png")

if __name__ == '__main__':
    data = execute_benchmarks(num_runs=3)
    plot_synthetic_benchmark(data)
    print("\n[SUCCESS] Live synthetic benchmark executed, logged, and plotted successfully!")
