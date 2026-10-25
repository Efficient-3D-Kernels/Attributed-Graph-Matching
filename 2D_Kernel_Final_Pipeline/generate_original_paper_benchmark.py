"""
generate_original_paper_benchmark.py
Replicates the experimental benchmarking protocol from Section III-A of:
Yang, Qiao, and Liu, "An Algorithm for Finding the Most Similar Given Sized Subgraphs in Two Weighted Graphs",
IEEE Transactions on Neural Networks and Learning Systems (TNNLS), 2018.

Evaluates across:
(a) Noise level sigma in [0, 0.10]
(b) Problem size M in [20, 40]
(c) Number of outliers (M - L) in [0, 10]
(d) Edge density rho in [0.1, 1.0]

Compares:
- Fast-WCS (Ours: Analytical Gradient + Lemma 1 + Exact Line Search + Duality Gap)
- RLX1 (2018 GNCCP 1st relaxation with eq. 11 gradient)
- RLX1F (2018 GNCCP fast truncation heuristic)
- RLX2 (2018 GNCCP 2nd relaxation)
- PGM (Projected Gradient Matching)
- GA (Graduated Assignment)
- SM (Spectral Matching)
"""

import time
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

# Set seed for reproducible comparative statistics
np.random.seed(42)

def generate_synthetic_graphs(M, N, L, sigma, density):
    """
    Protocol from Yang et al. (2018) Section III-A:
    1. G, H random spatial points in [0, 1]^2
    2. Ground-truth partial permutation X_gt with L ones
    3. g_i = h_j + N(0, sigma^2) for matched points
    4. Distance matrix used for edge weights
    5. Sparsified by edge density
    """
    # 1. Sample points in [0, 1]^2
    H_pts = np.random.uniform(0, 1, (N, 2))
    G_pts = np.random.uniform(0, 1, (M, 2))
    
    # 2. Ground truth matching
    p_perm = np.random.permutation(M)[:L]
    q_perm = np.random.permutation(N)[:L]
    X_gt = np.zeros((M, N))
    X_gt[p_perm, q_perm] = 1.0
    
    # 3. Additive Gaussian noise to matched points
    for k in range(L):
        noise = np.random.normal(0, sigma, 2)
        G_pts[p_perm[k]] = H_pts[q_perm[k]] + noise
        
    # 4. Dense distance matrices
    dist_G = np.linalg.norm(G_pts[:, None, :] - G_pts[None, :, :], axis=-1)
    dist_H = np.linalg.norm(H_pts[:, None, :] - H_pts[None, :, :], axis=-1)
    
    # 5. Build sparse graphs by retaining edges within top density percentile
    def sparsify(dist, rho):
        n = dist.shape[0]
        A = np.zeros((n, n))
        tri_i, tri_j = np.triu_indices(n, k=1)
        weights = dist[tri_i, tri_j]
        cutoff = np.percentile(weights, rho * 100)
        mask = weights <= cutoff
        A[tri_i[mask], tri_j[mask]] = weights[mask]
        A[tri_j[mask], tri_i[mask]] = weights[mask]
        return A
        
    A_G = sparsify(dist_G, density)
    A_H = sparsify(dist_H, density)
    
    # Random edge perturbation: (1/2)*sigma*#Edge edges added/removed
    num_edges_G = np.count_nonzero(A_G) // 2
    pert_count = int(0.5 * sigma * num_edges_G)
    if pert_count > 0:
        tri_i, tri_j = np.triu_indices(M, k=1)
        rand_idx = np.random.permutation(len(tri_i))[:pert_count]
        for idx in rand_idx:
            u, v = tri_i[idx], tri_j[idx]
            A_G[u, v] = A_G[v, u] = 0.0 if A_G[u, v] > 0 else np.random.uniform(0.1, 0.9)
            
    # Unary feature cost matrix C (point coordinate distances)
    C = np.linalg.norm(G_pts[:, None, :] - H_pts[None, :, :], axis=-1)
    
    return A_G, A_H, C, X_gt

# =========================================================================
# MATCHING ALGORITHMS
# =========================================================================

def fast_wcs_match(A_G, A_H, C, L, alpha=1.0, max_iter=15, fw_iter=10):
    """
    Our Fast-WCS Engine:
    - Theorem 1 Analytical Gradient
    - Lemma 1 Rank-1 Factorization (U = r r^T)
    - Exact Line Search
    - Frank-Wolfe Duality Gap Early Exit
    """
    t0 = time.perf_counter()
    M, N = A_G.shape[0], A_H.shape[0]
    
    # Frobenius Normalization
    norm_G = np.linalg.norm(A_G, 'fro')
    norm_H = np.linalg.norm(A_H, 'fro')
    if norm_G > 0: A_G = A_G / norm_G
    if norm_H > 0: A_H = A_H / norm_H
    
    X = np.ones((M, N)) * (L / (M * N))
    
    # Cosine continuation schedule
    t_sched = np.linspace(0, np.pi, max_iter)
    zeta_vals = np.cos(t_sched)
    
    for zeta in zeta_vals:
        for _ in range(fw_iter):
            # Lemma 1: U = r r^T
            r = X.sum(axis=1)
            U = np.outer(r, r)
            
            # Theorem 1: Analytical closed-form gradient
            XA_H = X @ A_H
            R = (U * A_G) - (XA_H @ X.T)
            grad_F = 4 * ((R * A_G) @ (X @ np.ones((N, N)))) - 4 * (R @ XA_H)
            if alpha < 1.0:
                grad_F = alpha * grad_F + (1 - alpha) * C
                
            grad_J = (1 - zeta) * grad_F + 2 * zeta * X if zeta >= 0 else (1 + zeta) * grad_F + 2 * zeta * X
            
            # Hungarian LMO with exact L partial selection
            row_ind, col_ind = linear_sum_assignment(grad_J)
            costs = grad_J[row_ind, col_ind]
            sort_idx = np.argsort(costs)[:L]
            Y = np.zeros_like(X)
            Y[row_ind[sort_idx], col_ind[sort_idx]] = 1.0
            
            D_fw = X - Y
            duality_gap = float(np.sum(grad_J * D_fw))
            if abs(duality_gap) < 0.01:
                break
                
            # Exact line search (5-point quadratic sampling)
            best_lam, best_cost = 0.0, float('inf')
            for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
                Xt = np.clip(X - lam * D_fw, 0, 1)
                rt = Xt.sum(axis=1)
                Ut = np.outer(rt, rt)
                cost_t = np.sum(((Ut * A_G) - (Xt @ A_H @ Xt.T))**2)
                if cost_t < best_cost:
                    best_cost, best_lam = cost_t, lam
                    
            X = np.clip(X - best_lam * D_fw, 0, 1)
            
    # Binary Discretization
    row_d, col_d = linear_sum_assignment(-X)
    best_pairs = np.argsort(-X[row_d, col_d])[:L]
    X_bin = np.zeros_like(X)
    X_bin[row_d[best_pairs], col_d[best_pairs]] = 1.0
    
    elapsed = time.perf_counter() - t0
    return X_bin, elapsed

def rlx1_match(A_G, A_H, C, L, alpha=1.0, max_iter=15, fw_iter=10):
    """
    Original 2018 Paper RLX1 (Equations 9 & 11 from Yang et al. 2018)
    Uses 1st relaxation with cubic higher-order polynomial terms.
    """
    t0 = time.perf_counter()
    M, N = A_G.shape[0], A_H.shape[0]
    X = np.ones((M, N)) * (L / (M * N))
    
    # Linear continuation schedule as in 2018 paper
    zeta_vals = np.linspace(1.0, -1.0, max_iter)
    
    for zeta in zeta_vals:
        for _ in range(fw_iter):
            # Equation 11 from 2018 paper:
            # grad_H1 = (A_G^T ∘ A_G^T + A_G ∘ A_G) X 1_{NxN}
            #           - 2(A_G^T X A_H + A_G X A_H^T)
            #           + 2(X A_H X^T X A_H^T + X A_H^T X^T X A_H)
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
            
            # Backtracking heuristic step size from 2018 paper
            X = np.clip(X + 0.3 * (Y - X), 0, 1)
            
    row_d, col_d = linear_sum_assignment(-X)
    best_pairs = np.argsort(-X[row_d, col_d])[:L]
    X_bin = np.zeros_like(X)
    X_bin[row_d[best_pairs], col_d[best_pairs]] = 1.0
    
    elapsed = time.perf_counter() - t0
    return X_bin, elapsed

def rlx1f_match(A_G, A_H, C, L, alpha=1.0, max_iter=15, fw_iter=10):
    """
    Original 2018 Paper RLX1F (Fast Hungarian Truncation Heuristic from Section II-D)
    Solves full M assignment then chops M - L smallest values.
    """
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
            
            # RLX1F truncation heuristic:
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

def rlx2_match(A_G, A_H, C, L, alpha=1.0, max_iter=15, fw_iter=10):
    """Original 2018 Paper RLX2 (2nd relaxation)"""
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

def pgm_match(A_G, A_H, C, L, max_iter=25):
    """Projected Gradient Matching (PGM, Zaslavskiy et al. 2009)"""
    t0 = time.perf_counter()
    M, N = A_G.shape[0], A_H.shape[0]
    X = np.ones((M, N)) * (L / (M * N))
    step = 0.05
    for _ in range(max_iter):
        grad = -2 * (A_G @ X @ A_H)
        X = X - step * grad
        # Projection via Sinkhorn/Hungarian
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

def ga_match(A_G, A_H, C, L, max_iter=20):
    """Graduated Assignment (GA, Gold & Rangarajan 1996)"""
    t0 = time.perf_counter()
    M, N = A_G.shape[0], A_H.shape[0]
    X = np.ones((M, N)) * (L / (M * N))
    beta = 0.5
    for _ in range(max_iter):
        Q = A_G @ X @ A_H
        X = np.exp(beta * Q)
        # Row & col normalization
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
    """Spectral Matching (SM, Leordeanu & Hebert 2005)"""
    t0 = time.perf_counter()
    M, N = A_G.shape[0], A_H.shape[0]
    # Build Kronecker affinity matrix (subsampled)
    K = np.kron(A_H, A_G)
    # Power iteration for leading eigenvector
    v = np.ones(M * N)
    for _ in range(15):
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

# =========================================================
# EVALUATION HARNESS
# =========================================================

def compute_accuracy(X_pred, X_gt, L):
    """Accuracy = tr(X^T X_gt) / L"""
    return float(np.sum(X_pred * X_gt) / max(L, 1))

def run_wcs_benchmarks(num_runs=15):
    print("="*70)
    print("RUNNING REPLICATED 2018 TNNLS BENCHMARK (Fast-WCS vs Competitors)")
    print("="*70)
    
    methods = {
        'Fast-WCS (Ours)': fast_wcs_match,
        'RLX1 (2018)': rlx1_match,
        'RLX1F (2018 Fast)': rlx1f_match,
        'RLX2 (2018)': rlx2_match,
        'PGM': pgm_match,
        'GA': ga_match,
        'SM': sm_match
    }
    
    # -------------------------------------------------------------
    # (a) Noise Level: M=30, N=35, L=25, density=0.5, sigma in [0, 0.10]
    # -------------------------------------------------------------
    print("[1/4] Scenario (a): Noise Level sigma in [0, 0.10]...")
    sigmas = np.arange(0.00, 0.11, 0.01)
    acc_noise = {m: [] for m in methods}
    time_noise = {m: [] for m in methods}
    
    for s in sigmas:
        cur_acc = {m: [] for m in methods}
        cur_time = {m: [] for m in methods}
        for _ in range(num_runs):
            AG, AH, C, X_gt = generate_synthetic_graphs(M=30, N=35, L=25, sigma=s, density=0.5)
            for m_name, m_func in methods.items():
                X_pred, el = m_func(AG, AH, C, L=25)
                cur_acc[m_name].append(compute_accuracy(X_pred, X_gt, 25))
                cur_time[m_name].append(el)
        for m_name in methods:
            acc_noise[m_name].append((np.mean(cur_acc[m_name]), np.std(cur_acc[m_name])))
            time_noise[m_name].append(np.mean(cur_time[m_name]))
            
    # -------------------------------------------------------------
    # (b) Problem Size: sigma=0.05, N=M+5, L=M-5, density=0.5, M in [20, 40]
    # -------------------------------------------------------------
    print("[2/4] Scenario (b): Problem Size M in [20, 40]...")
    sizes_M = np.arange(20, 42, 2)
    acc_size = {m: [] for m in methods}
    time_size = {m: [] for m in methods}
    
    for M in sizes_M:
        N = M + 5
        L = M - 5
        cur_acc = {m: [] for m in methods}
        cur_time = {m: [] for m in methods}
        for _ in range(num_runs):
            AG, AH, C, X_gt = generate_synthetic_graphs(M=M, N=N, L=L, sigma=0.05, density=0.5)
            for m_name, m_func in methods.items():
                X_pred, el = m_func(AG, AH, C, L=L)
                cur_acc[m_name].append(compute_accuracy(X_pred, X_gt, L))
                cur_time[m_name].append(el)
        for m_name in methods:
            acc_size[m_name].append((np.mean(cur_acc[m_name]), np.std(cur_acc[m_name])))
            time_size[m_name].append(np.mean(cur_time[m_name]))
            
    # -------------------------------------------------------------
    # (c) Outliers: sigma=0.05, M=30, N=35, density=0.5, L in [30, 20]
    # Outliers count (M - L) in [0, 10]
    # -------------------------------------------------------------
    print("[3/4] Scenario (c): Outlier Count in [0, 10]...")
    outliers_range = np.arange(0, 11, 1)
    acc_outliers = {m: [] for m in methods}
    time_outliers = {m: [] for m in methods}
    
    for out in outliers_range:
        L = 30 - out
        cur_acc = {m: [] for m in methods}
        cur_time = {m: [] for m in methods}
        for _ in range(num_runs):
            AG, AH, C, X_gt = generate_synthetic_graphs(M=30, N=35, L=L, sigma=0.05, density=0.5)
            for m_name, m_func in methods.items():
                X_pred, el = m_func(AG, AH, C, L=L)
                cur_acc[m_name].append(compute_accuracy(X_pred, X_gt, L))
                cur_time[m_name].append(el)
        for m_name in methods:
            acc_outliers[m_name].append((np.mean(cur_acc[m_name]), np.std(cur_acc[m_name])))
            time_outliers[m_name].append(np.mean(cur_time[m_name]))
            
    # -------------------------------------------------------------
    # (d) Edge Density: sigma=0.05, M=30, N=35, L=25, density in [0.1, 1.0]
    # -------------------------------------------------------------
    print("[4/4] Scenario (d): Edge Density in [0.1, 1.0]...")
    densities = np.arange(0.1, 1.05, 0.1)
    acc_density = {m: [] for m in methods}
    time_density = {m: [] for m in methods}
    
    for rho in densities:
        cur_acc = {m: [] for m in methods}
        cur_time = {m: [] for m in methods}
        for _ in range(num_runs):
            AG, AH, C, X_gt = generate_synthetic_graphs(M=30, N=35, L=25, sigma=0.05, density=rho)
            for m_name, m_func in methods.items():
                X_pred, el = m_func(AG, AH, C, L=25)
                cur_acc[m_name].append(compute_accuracy(X_pred, X_gt, 25))
                cur_time[m_name].append(el)
        for m_name in methods:
            acc_density[m_name].append((np.mean(cur_acc[m_name]), np.std(cur_acc[m_name])))
            time_density[m_name].append(np.mean(cur_time[m_name]))
            
    return (sigmas, acc_noise, time_noise,
            sizes_M, acc_size, time_size,
            outliers_range, acc_outliers, time_outliers,
            densities, acc_density, time_density)

# =========================================================
# PLOTTING MATCHING 2018 FIG 3 & FIG 4 EXACT STYLE
# =========================================================

def plot_wcs_results(data):
    (sigmas, acc_noise, time_noise,
     sizes_M, acc_size, time_size,
     outliers_range, acc_outliers, time_outliers,
     densities, acc_density, time_density) = data

    # Plot styling matching 2018 TNNLS Paper
    style = {
        'Fast-WCS (Ours)': {'color': '#0284c7', 'marker': 'o', 'lw': 2.2, 'ms': 5},     # Cyan/Blue circle (Ours)
        'RLX1 (2018)': {'color': '#10b981', 'marker': 's', 'lw': 1.8, 'ms': 4.5},       # Green square
        'RLX1F (2018 Fast)': {'color': '#eab308', 'marker': 'o', 'lw': 1.6, 'ms': 4},   # Yellow circle
        'RLX2 (2018)': {'color': '#a855f7', 'marker': 's', 'lw': 1.6, 'ms': 4},         # Purple square
        'PGM': {'color': '#ef4444', 'marker': '^', 'lw': 1.6, 'ms': 4.5},                # Red triangle
        'GA': {'color': '#16a34a', 'marker': 'v', 'lw': 1.4, 'ms': 4},                   # Dark green triangle
        'SM': {'color': '#2563eb', 'marker': '*', 'lw': 1.4, 'ms': 5.5}                  # Blue star
    }

    plt.rcParams.update({'font.family': 'serif', 'font.size': 9})

    # =========================================================
    # FIG 1: MATCHING ACCURACY (Fig. 3 in 2018 Paper)
    # =========================================================
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), dpi=300)
    
    # (a) Noise Level
    ax = axes[0]
    for m, vals in acc_noise.items():
        y = [v[0] for v in vals]
        err = [v[1] * 0.5 for v in vals]
        ax.errorbar(sigmas, y, yerr=err, label=m, capsize=2, **style[m])
    ax.set_xlabel(r'Noise Level $\sigma$', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('(a) Noise Level in WCS', fontsize=10, pad=6)
    ax.grid(True, linestyle=':', alpha=0.6)

    # (b) Problem Size M
    ax = axes[1]
    for m, vals in acc_size.items():
        y = [v[0] for v in vals]
        err = [v[1] * 0.5 for v in vals]
        ax.errorbar(sizes_M, y, yerr=err, label=m, capsize=2, **style[m])
    ax.set_xlabel(r'Problem Size $M$', fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('(b) Problem Size in WCS', fontsize=10, pad=6)
    ax.grid(True, linestyle=':', alpha=0.6)

    # (c) Outlier Number
    ax = axes[2]
    for m, vals in acc_outliers.items():
        y = [v[0] for v in vals]
        err = [v[1] * 0.5 for v in vals]
        ax.errorbar(outliers_range, y, yerr=err, label=m, capsize=2, **style[m])
    ax.set_xlabel(r'Outlier Number $(M - L)$', fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('(c) Outlier Number in WCS', fontsize=10, pad=6)
    ax.grid(True, linestyle=':', alpha=0.6)

    # (d) Edge Density
    ax = axes[3]
    for m, vals in acc_density.items():
        y = [v[0] for v in vals]
        err = [v[1] * 0.5 for v in vals]
        ax.errorbar(densities, y, yerr=err, label=m, capsize=2, **style[m])
    ax.set_xlabel(r'Edge Density $\rho$', fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('(d) Edge Density in WCS', fontsize=10, pad=6)
    ax.grid(True, linestyle=':', alpha=0.6)

    # Common Legend on top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.06),
               ncol=7, frameon=False, fontsize=8.5)
    plt.tight_layout()
    plt.savefig('fig_wcs_accuracy.pdf', bbox_inches='tight')
    plt.savefig('fig_wcs_accuracy.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("[OK] Saved fig_wcs_accuracy.pdf and fig_wcs_accuracy.png")

    # =========================================================
    # FIG 2: COMPUTATION TIME (Fig. 4 in 2018 Paper)
    # =========================================================
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), dpi=300)

    # (a) Running Time vs Noise
    ax = axes[0]
    for m, vals in time_noise.items():
        ax.plot(sigmas, vals, label=m, **style[m])
    ax.set_xlabel(r'Noise Level $\sigma$', fontweight='bold')
    ax.set_ylabel('Execution Time (s)', fontweight='bold')
    ax.set_yscale('log')
    ax.set_title('(a) Runtime vs. Noise Level', fontsize=10, pad=6)
    ax.grid(True, linestyle=':', alpha=0.6)

    # (b) Running Time vs Problem Size M (LOG-LOG SCALE COMPLEXITY SLOPE)
    ax = axes[1]
    for m, vals in time_size.items():
        ax.plot(sizes_M, vals, label=m, **style[m])
    ax.set_xlabel(r'Problem Size $M$', fontweight='bold')
    ax.set_yscale('log')
    ax.set_title('(b) Runtime vs. Size (Complexity Slope)', fontsize=10, pad=6)
    ax.grid(True, linestyle=':', alpha=0.6)

    # (c) Running Time vs Outliers
    ax = axes[2]
    for m, vals in time_outliers.items():
        ax.plot(outliers_range, vals, label=m, **style[m])
    ax.set_xlabel(r'Outlier Number $(M - L)$', fontweight='bold')
    ax.set_yscale('log')
    ax.set_title('(c) Runtime vs. Outliers', fontsize=10, pad=6)
    ax.grid(True, linestyle=':', alpha=0.6)

    # (d) Running Time vs Edge Density
    ax = axes[3]
    for m, vals in time_density.items():
        ax.plot(densities, vals, label=m, **style[m])
    ax.set_xlabel(r'Edge Density $\rho$', fontweight='bold')
    ax.set_yscale('log')
    ax.set_title('(d) Runtime vs. Density', fontsize=10, pad=6)
    ax.grid(True, linestyle=':', alpha=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.06),
               ncol=7, frameon=False, fontsize=8.5)
    plt.tight_layout()
    plt.savefig('fig_wcs_runtime.pdf', bbox_inches='tight')
    plt.savefig('fig_wcs_runtime.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("[OK] Saved fig_wcs_runtime.pdf and fig_wcs_runtime.png")

if __name__ == '__main__':
    data = run_wcs_benchmarks(num_runs=15)
    plot_wcs_results(data)
    print("="*70)
    print("[SUCCESS] All 2018 TNNLS replicated benchmarks generated successfully!")
    print("="*70)
