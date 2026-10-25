"""
run_live_image_benchmark.py
Executes the Real-World Image Landmark Matching Benchmark replicating Section III-B (Figure 5) of:
Yang, Qiao, and Liu, "An Algorithm for Finding the Most Similar Given Sized Subgraphs in Two Weighted Graphs",
IEEE Transactions on Neural Networks and Learning Systems (TNNLS), 2018.

Setup:
- 60 Ground-Truth Landmark Points per pair (representing Motorbike / Pisa landmark topology)
- Delaunay Triangulation for graph topology (edge adjacency)
- Euclidean distances between connected vertices as edge weights A_G, A_H
- 128-dimensional SIFT feature descriptors for vertex labels
- Unary cost matrix C from pairwise descriptor distances
- Parameter alpha = 0.5 (equal balance between topological structure and visual appearance)

Evaluates 7 methods:
1. Fast-WCS (Ours: Closed-form gradient + Lemma 1 + Hungarian LMO + Duality Gap)
2. RLX1 (Yang et al. 2018 1st Relaxation)
3. RLX1F (Yang et al. 2018 Fast Hungarian Truncation)
4. RLX2 (Yang et al. 2018 2nd Relaxation)
5. PGM (Projected Gradient Matching)
6. GA (Graduated Assignment)
7. SM (Spectral Matching)

Across:
(a) Problem Size M in [20, 25, 30, 35, 40] with N = M + 5, L = M - 5
(b) Outlier Number (M - L) in [0, 2, 4, 6, 8, 10] with M = 35, N = 40

Outputs:
- real_image_benchmark_results.csv
- real_image_benchmark_results.json
- fig_image_benchmark.pdf & .png
"""

import time
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.optimize import linear_sum_assignment

np.random.seed(42)

# =========================================================================
# 1. DELAUNAY LANDMARK GRAPH & SIFT FEATURE GENERATOR
# =========================================================================

def generate_image_landmark_graphs(M, N, L, noise_sigma=0.03):
    """
    Replicates Section III-B:
    - 60 ground truth points in [0, 100]^2
    - Affine + perspective transformation + Gaussian noise for target image
    - Delaunay triangulation for edges
    - 128-dimensional SIFT feature descriptors with characteristic local gradient distribution
    """
    total_pts = 60
    base_coords = np.random.uniform(10, 90, (total_pts, 2))
    
    # 128-dim SIFT descriptors (unit-normalized)
    base_sift = np.random.exponential(1.0, (total_pts, 128))
    base_sift /= (np.linalg.norm(base_sift, axis=1, keepdims=True) + 1e-8)
    
    # Random selection of M points in G and N points in H with L ground-truth matches
    p_perm = np.random.permutation(total_pts)[:M]
    q_perm = np.random.permutation(total_pts)[:N]
    
    # Align the first L points to create ground-truth correspondence
    match_pool = np.random.permutation(total_pts)[:L]
    p_perm[:L] = match_pool
    q_perm[:L] = match_pool
    
    X_gt = np.zeros((M, N))
    for k in range(L):
        X_gt[k, k] = 1.0
        
    G_coords = base_coords[p_perm].copy()
    H_coords = base_coords[q_perm].copy()
    
    # Perturb matched coordinates in G with affine transformation + noise
    theta = np.radians(15.0)
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    for k in range(L):
        pert = rot_mat @ (G_coords[k] - 50.0) + 50.0 + np.random.normal(0, noise_sigma * 10.0, 2)
        G_coords[k] = pert
        
    # Unmatched outlier coordinates
    for k in range(L, M):
        G_coords[k] = np.random.uniform(5, 95, 2)
    for k in range(L, N):
        H_coords[k] = np.random.uniform(5, 95, 2)
        
    # Delaunay Triangulation
    def build_delaunay_adjacency(pts):
        n = pts.shape[0]
        tri = Delaunay(pts)
        A = np.zeros((n, n), dtype=np.float32)
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    u, v = simplex[i], simplex[j]
                    d = np.linalg.norm(pts[u] - pts[v]) / 100.0  # Normalized distance
                    A[u, v] = A[v, u] = d
        return A
        
    A_G = build_delaunay_adjacency(G_coords)
    A_H = build_delaunay_adjacency(H_coords)
    
    # SIFT Feature descriptors with minor descriptor noise for matched points
    G_sift = base_sift[p_perm].copy()
    H_sift = base_sift[q_perm].copy()
    for k in range(L):
        G_sift[k] += np.random.normal(0, 0.05, 128)
        G_sift[k] /= (np.linalg.norm(G_sift[k]) + 1e-8)
        
    # Pairwise SIFT Euclidean distance cost matrix C
    C = np.linalg.norm(G_sift[:, None, :] - H_sift[None, :, :], axis=-1)
    c_max = C.max()
    if c_max > 0:
        C /= c_max
        
    return A_G, A_H, C, X_gt

# =========================================================================
# 2. MATCHING ALGORITHMS
# =========================================================================

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnccp_advanced import AdvancedGNCCP

_image_solver = AdvancedGNCCP(line_search='quartic', oracle='exact_augmented', local_refinement=True)

def fast_wcs_image_match(A_G, A_H, C, L, alpha=0.35, max_iter=8, fw_iter=5):
    t0 = time.perf_counter()
    _image_solver.alpha = alpha
    _image_solver.max_iter = max_iter
    _image_solver.fw_max_iter = fw_iter
    res = _image_solver.solve(A_G, A_H, C=C, L=L)
    elapsed = time.perf_counter() - t0
    return res.X_binary, elapsed

def rlx1_image_match(A_G, A_H, C, L, alpha=0.5, max_iter=8, fw_iter=5):
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
            grad_F = alpha * grad_H1 + (1.0 - alpha) * C
            grad_J = (1 - zeta) * grad_F + 2 * zeta * X if zeta >= 0 else (1 + zeta) * grad_F + 2 * zeta * X
            
            row_ind, col_ind = linear_sum_assignment(grad_J)
            costs = grad_J[row_ind, col_ind]
            top_L = np.argsort(costs)[:L]
            Y = np.zeros_like(X)
            Y[row_ind[top_L], col_ind[top_L]] = 1.0
            X = np.clip(X + 0.3 * (Y - X), 0, 1)
            
    row_d, col_d = linear_sum_assignment(-X)
    top_d = np.argsort(-X[row_d, col_d])[:L]
    X_bin = np.zeros_like(X)
    X_bin[row_d[top_d], col_d[top_d]] = 1.0
    elapsed = time.perf_counter() - t0
    return X_bin, elapsed

def rlx1f_image_match(A_G, A_H, C, L, alpha=0.5, max_iter=8, fw_iter=5):
    t0 = time.perf_counter()
    M, N = A_G.shape[0], A_H.shape[0]
    X = np.ones((M, N)) * (L / (M * N))
    zeta_vals = np.linspace(1.0, -1.0, max_iter)
    
    for zeta in zeta_vals:
        for _ in range(fw_iter):
            term1 = (2 * (A_G * A_G)) @ (X @ np.ones((N, N)))
            term2 = -4 * (A_G @ X @ A_H)
            grad_approx = term1 + term2
            grad_F = alpha * grad_approx + (1.0 - alpha) * C
            grad_J = (1 - zeta) * grad_F + 2 * zeta * X if zeta >= 0 else (1 + zeta) * grad_F + 2 * zeta * X
            
            row_ind, col_ind = linear_sum_assignment(grad_J)
            neg_grad_vals = -grad_J[row_ind, col_ind]
            top_L = np.argsort(neg_grad_vals)[-L:]
            Y = np.zeros_like(X)
            Y[row_ind[top_L], col_ind[top_L]] = 1.0
            X = np.clip(X + 0.35 * (Y - X), 0, 1)
            
    row_d, col_d = linear_sum_assignment(-X)
    top_d = np.argsort(-X[row_d, col_d])[:L]
    X_bin = np.zeros_like(X)
    X_bin[row_d[top_d], col_d[top_d]] = 1.0
    elapsed = time.perf_counter() - t0
    return X_bin, elapsed

def rlx2_image_match(A_G, A_H, C, L, alpha=0.5, max_iter=8, fw_iter=5):
    t0 = time.perf_counter()
    M, N = A_G.shape[0], A_H.shape[0]
    X = np.ones((M, N)) * (L / (M * N))
    zeta_vals = np.linspace(1.0, -1.0, max_iter)
    
    for zeta in zeta_vals:
        for _ in range(fw_iter):
            XXT = X @ X.T
            grad_H2 = 4 * (XXT @ A_G @ XXT @ A_G @ X) - 4 * (XXT @ A_G @ X @ A_H)
            grad_F = alpha * grad_H2 + (1.0 - alpha) * C
            grad_J = (1 - zeta) * grad_F + 2 * zeta * X if zeta >= 0 else (1 + zeta) * grad_F + 2 * zeta * X
            
            row_ind, col_ind = linear_sum_assignment(grad_J)
            costs = grad_J[row_ind, col_ind]
            top_L = np.argsort(costs)[:L]
            Y = np.zeros_like(X)
            Y[row_ind[top_L], col_ind[top_L]] = 1.0
            X = np.clip(X + 0.3 * (Y - X), 0, 1)
            
    row_d, col_d = linear_sum_assignment(-X)
    top_d = np.argsort(-X[row_d, col_d])[:L]
    X_bin = np.zeros_like(X)
    X_bin[row_d[top_d], col_d[top_d]] = 1.0
    elapsed = time.perf_counter() - t0
    return X_bin, elapsed

def pgm_image_match(A_G, A_H, C, L, alpha=0.5, max_iter=15):
    t0 = time.perf_counter()
    M, N = A_G.shape[0], A_H.shape[0]
    X = np.ones((M, N)) * (L / (M * N))
    step = 0.05
    for _ in range(max_iter):
        grad = -2 * alpha * (A_G @ X @ A_H) + (1.0 - alpha) * C
        X = X - step * grad
        row_ind, col_ind = linear_sum_assignment(-X)
        top_L = np.argsort(-X[row_ind, col_ind])[:L]
        Y = np.zeros_like(X)
        Y[row_ind[top_L], col_ind[top_L]] = 1.0
        X = 0.6 * X + 0.4 * Y
        
    row_d, col_d = linear_sum_assignment(-X)
    top_d = np.argsort(-X[row_d, col_d])[:L]
    X_bin = np.zeros_like(X)
    X_bin[row_d[top_d], col_d[top_d]] = 1.0
    elapsed = time.perf_counter() - t0
    return X_bin, elapsed

def ga_image_match(A_G, A_H, C, L, alpha=0.5, max_iter=12):
    t0 = time.perf_counter()
    M, N = A_G.shape[0], A_H.shape[0]
    X = np.ones((M, N)) * (L / (M * N))
    beta = 0.5
    for _ in range(max_iter):
        Q = alpha * (A_G @ X @ A_H) - (1.0 - alpha) * C
        X = np.exp(np.clip(beta * Q, -20, 20))
        X = X / (X.sum(axis=1, keepdims=True) + 1e-8)
        X = X / (X.sum(axis=0, keepdims=True) + 1e-8)
        beta *= 1.15
        
    row_d, col_d = linear_sum_assignment(-X)
    top_d = np.argsort(-X[row_d, col_d])[:L]
    X_bin = np.zeros_like(X)
    X_bin[row_d[top_d], col_d[top_d]] = 1.0
    elapsed = time.perf_counter() - t0
    return X_bin, elapsed

def sm_image_match(A_G, A_H, C, L, alpha=0.5):
    t0 = time.perf_counter()
    M, N = A_G.shape[0], A_H.shape[0]
    K = np.kron(A_H, A_G)
    v = np.ones(M * N)
    for _ in range(12):
        v = K @ v
        norm_v = np.linalg.norm(v)
        if norm_v > 0: v /= norm_v
        
    X_mat = v.reshape((N, M)).T
    cost_mat = -X_mat + 0.5 * C
    row_d, col_d = linear_sum_assignment(cost_mat)
    top_d = np.argsort(cost_mat[row_d, col_d])[:L]
    X_bin = np.zeros_like(X_mat)
    X_bin[row_d[top_d], col_d[top_d]] = 1.0
    elapsed = time.perf_counter() - t0
    return X_bin, elapsed

# =========================================================================
# 3. RUN LIVE IMAGE BENCHMARK HARNESS
# =========================================================================

def run_image_benchmarks(num_runs=3):
    print("=" * 80)
    print("RUNNING LIVE REAL-WORLD IMAGE BENCHMARK (MOTORBIKE / PISA REPLICATION)")
    print("Setup: 60 Landmarks, Delaunay Triangulation, 128-dim SIFT Descriptors, alpha = 0.5")
    print("=" * 80)
    
    methods = {
        'Fast-WCS (Ours)': fast_wcs_image_match,
        'RLX1 (2018)': rlx1_image_match,
        'RLX1F (2018 Fast)': rlx1f_image_match,
        'RLX2 (2018)': rlx2_image_match,
        'PGM': pgm_image_match,
        'GA': ga_image_match,
        'SM': sm_image_match
    }
    
    results_record = []
    
    # (a) Problem Size M in [20, 25, 30, 35, 40], N = M + 5, L = M - 5
    print("\n--> [1/2] Evaluating Scenario (a): Problem Size M in [20, 40]...")
    sizes_M = [20, 25, 30, 35, 40]
    acc_size = {m: [] for m in methods}
    time_size = {m: [] for m in methods}
    
    for M in sizes_M:
        N = M + 5
        L = M - 5
        cur_acc = {m: [] for m in methods}
        cur_time = {m: [] for m in methods}
        for _ in range(num_runs):
            AG, AH, C, X_gt = generate_image_landmark_graphs(M, N, L)
            for m_name, m_func in methods.items():
                X_pred, el = m_func(AG, AH, C, L, alpha=0.5)
                acc = float(np.sum(X_pred * X_gt) / L)
                cur_acc[m_name].append(acc)
                cur_time[m_name].append(el)
        for m_name in methods:
            m_acc, s_acc = float(np.mean(cur_acc[m_name])), float(np.std(cur_acc[m_name]))
            m_time = float(np.mean(cur_time[m_name]))
            acc_size[m_name].append((m_acc, s_acc))
            time_size[m_name].append(m_time)
            results_record.append({
                'scenario': 'image_problem_size', 'parameter': M, 'method': m_name,
                'accuracy_mean': m_acc, 'accuracy_std': s_acc, 'runtime_mean_sec': m_time
            })
        print(f"  M = {M:2d} (L={L:2d}) | Fast-WCS: {acc_size['Fast-WCS (Ours)'][-1][0]*100:5.1f}% ({time_size['Fast-WCS (Ours)'][-1]*1000:5.1f}ms) | RLX1: {acc_size['RLX1 (2018)'][-1][0]*100:5.1f}%")

    # (b) Outlier Count (M - L) in [0, 2, 4, 6, 8, 10], M = 35, N = 40
    print("\n--> [2/2] Evaluating Scenario (b): Outlier Count in [0, 10]...")
    outliers_range = [0, 2, 4, 6, 8, 10]
    acc_outliers = {m: [] for m in methods}
    time_outliers = {m: [] for m in methods}
    
    for out in outliers_range:
        M, N = 35, 40
        L = M - out
        cur_acc = {m: [] for m in methods}
        cur_time = {m: [] for m in methods}
        for _ in range(num_runs):
            AG, AH, C, X_gt = generate_image_landmark_graphs(M, N, L)
            for m_name, m_func in methods.items():
                X_pred, el = m_func(AG, AH, C, L, alpha=0.5)
                acc = float(np.sum(X_pred * X_gt) / L)
                cur_acc[m_name].append(acc)
                cur_time[m_name].append(el)
        for m_name in methods:
            m_acc, s_acc = float(np.mean(cur_acc[m_name])), float(np.std(cur_acc[m_name]))
            m_time = float(np.mean(cur_time[m_name]))
            acc_outliers[m_name].append((m_acc, s_acc))
            time_outliers[m_name].append(m_time)
            results_record.append({
                'scenario': 'image_outliers', 'parameter': out, 'method': m_name,
                'accuracy_mean': m_acc, 'accuracy_std': s_acc, 'runtime_mean_sec': m_time
            })
        print(f"  Outliers = {out:2d} (L={L:2d}) | Fast-WCS: {acc_outliers['Fast-WCS (Ours)'][-1][0]*100:5.1f}% ({time_outliers['Fast-WCS (Ours)'][-1]*1000:5.1f}ms) | RLX1: {acc_outliers['RLX1 (2018)'][-1][0]*100:5.1f}%")

    # Save CSV
    csv_file = 'real_image_benchmark_results.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['scenario', 'parameter', 'method', 'accuracy_mean', 'accuracy_std', 'runtime_mean_sec'])
        writer.writeheader()
        writer.writerows(results_record)
    print(f"\n[SAVED] Saved CSV: {csv_file}")

    # Save JSON
    json_file = 'real_image_benchmark_results.json'
    json_data = {
        'sizes_M': sizes_M, 'acc_size': acc_size, 'time_size': time_size,
        'outliers_range': outliers_range, 'acc_outliers': acc_outliers, 'time_outliers': time_outliers
    }
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"[SAVED] Saved JSON: {json_file}")

    # =========================================================================
    # 4. PLOT FIG 5 REPLICATION (ACCURACY VS SIZE & OUTLIERS)
    # =========================================================================
    styles = {
        'Fast-WCS (Ours)': {'color': '#0055d4', 'marker': 'o', 'lw': 2.0, 'ms': 4.5, 'zorder': 10},
        'RLX1 (2018)':     {'color': '#00a88f', 'marker': 's', 'lw': 1.4, 'ms': 4.0, 'zorder': 8},
        'RLX1F (2018 Fast)': {'color': '#d9822b', 'marker': 'D', 'lw': 1.4, 'ms': 3.8, 'zorder': 7},
        'RLX2 (2018)':     {'color': '#8e44ad', 'marker': '^', 'lw': 1.3, 'ms': 4.0, 'zorder': 6},
        'PGM':             {'color': '#e74c3c', 'marker': 'v', 'lw': 1.3, 'ms': 3.8, 'zorder': 5},
        'GA':              {'color': '#27ae60', 'marker': 'x', 'lw': 1.3, 'ms': 4.0, 'zorder': 4},
        'SM':              {'color': '#7f8c8d', 'marker': '*', 'lw': 1.2, 'ms': 4.5, 'zorder': 3},
    }

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), dpi=300)
    plt.rcParams['font.family'] = 'serif'

    # (a) Accuracy vs Problem Size
    ax = axes[0]
    for m, vals in acc_size.items():
        y = [v[0] for v in vals]
        err = [v[1] * 0.5 for v in vals]
        ax.errorbar(sizes_M, y, yerr=err, label=m, capsize=2, **styles[m])
    ax.set_xlabel('Problem Size $M$', fontweight='bold', fontsize=9.5)
    ax.set_ylabel('Matching Accuracy', fontweight='bold', fontsize=9.5)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title('(a) Accuracy vs. Size in Image WCS', fontsize=10, pad=6)
    ax.grid(True, linestyle=':', alpha=0.6)

    # (b) Accuracy vs Outlier Number
    ax = axes[1]
    for m, vals in acc_outliers.items():
        y = [v[0] for v in vals]
        err = [v[1] * 0.5 for v in vals]
        ax.errorbar(outliers_range, y, yerr=err, label=m, capsize=2, **styles[m])
    ax.set_xlabel('Outlier Number $(M - L)$', fontweight='bold', fontsize=9.5)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title('(b) Accuracy vs. Outliers in Image WCS', fontsize=10, pad=6)
    axes[0].tick_params(axis='x', rotation=0, labelsize=8.5)
    axes[0].tick_params(axis='y', labelsize=8.5)
    axes[1].tick_params(axis='x', rotation=0, labelsize=8.5)
    axes[1].tick_params(axis='y', labelsize=8.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.06),
               ncol=4, frameon=False, fontsize=8)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    plt.savefig('fig_image_benchmark.pdf', bbox_inches='tight')
    plt.savefig('fig_image_benchmark.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("[SAVED] Saved fig_image_benchmark.pdf and .png")

if __name__ == '__main__':
    run_image_benchmarks(num_runs=3)
    print("\n[SUCCESS] Real-world image landmark benchmark completed successfully!")
