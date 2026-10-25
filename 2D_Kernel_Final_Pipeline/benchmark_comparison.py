"""
Benchmark: Optimized WCS-GNCCP vs. Naive (Original Paper) GNCCP
================================================================
Compares our optimized engine against a faithful reimplementation of the
original 2018 paper's approach (no rank-1 factorization, numerical gradient,
fixed step size, no duality gap).

Produces timing curves and an interactive HTML bijective mapping visualization.
"""

import numpy as np
import time
import json
import os
from scipy.optimize import linear_sum_assignment

# =====================================================================
# NAIVE GNCCP (Baseline — Original 2018 Paper Approach)
# =====================================================================
class NaiveGNCCP:
    """Faithful reimplementation of the ORIGINAL paper's approach:
    - No rank-1 factorization: U = X @ 1_{NxN} @ X.T (cubic)
    - Numerical gradient via finite differences (extremely expensive)
    - Fixed step size 2/(t+2)
    - No duality gap early stopping — runs all iterations
    - Full assignment LMO (no L-partial selection)
    """
    
    def __init__(self, alpha=0.7, gamma=0.5, max_iter=8, fw_max_iter=3, tol=1e-3):
        self.alpha = alpha
        self.gamma = gamma
        self.max_iter = max_iter
        self.fw_max_iter = fw_max_iter
        self.tol = tol
    
    def match(self, A_G, A_H, C, L):
        M, N = A_G.shape[0], A_H.shape[0]
        X = np.ones((M, N)) * (L / (M * N))
        zeta_values = np.linspace(1.0, -1.0, self.max_iter)
        
        for zeta in zeta_values:
            for t in range(self.fw_max_iter):
                grad = self._compute_gradient_numerical(X, A_G, A_H, C, zeta)
                Y = self._solve_lp(grad, M, N, L)
                
                # Fixed step size (no line search)
                lambda_opt = 2.0 / (t + 2.0)
                D = X - Y
                X = X - lambda_opt * D
                X = self._project(X, L)
        
        # Discretize
        C_matrix = -X
        row_ind, col_ind = linear_sum_assignment(C_matrix)
        Y_binary = np.zeros((M, N))
        for r, c in zip(row_ind, col_ind):
            Y_binary[r, c] = 1
        count = int(Y_binary.sum())
        if count > L:
            ones = np.where(Y_binary == 1)
            vals = [(X[ones[0][k], ones[1][k]], k) for k in range(len(ones[0]))]
            vals.sort()
            for _, k in vals[:count - L]:
                Y_binary[ones[0][k], ones[1][k]] = 0
        
        return Y_binary, self._compute_score(A_G, A_H, C, Y_binary, L)
    
    def _compute_U_naive(self, X, N):
        """NAIVE: O(MN^2 + M^2N) — the cubic bottleneck the original paper has."""
        ones_NN = np.ones((N, N))
        return X @ ones_NN @ X.T
    
    def _compute_F(self, X, A_G, A_H, C):
        M, N = X.shape
        U = self._compute_U_naive(X, N)
        diff = (U * A_G) - (X @ A_H @ X.T)
        struct = np.sum(diff ** 2)
        app = np.sum(C * X)
        return self.alpha * struct + (1 - self.alpha) * app
    
    def _compute_gradient_numerical(self, X, A_G, A_H, C, zeta):
        """NUMERICAL gradient via finite differences — O(M*N*n^3) per call!"""
        M, N = X.shape
        eps = 1e-5
        grad = np.zeros_like(X)
        F0 = self._compute_Jzeta(X, A_G, A_H, C, zeta)
        for i in range(M):
            for j in range(N):
                X[i, j] += eps
                F1 = self._compute_Jzeta(X, A_G, A_H, C, zeta)
                grad[i, j] = (F1 - F0) / eps
                X[i, j] -= eps
        return grad
    
    def _compute_Jzeta(self, X, A_G, A_H, C, zeta):
        F = self._compute_F(X, A_G, A_H, C)
        x_norm_sq = np.sum(X ** 2)
        if zeta >= 0:
            return (1 - zeta) * F + zeta * x_norm_sq
        else:
            return (1 + zeta) * F + zeta * x_norm_sq
    
    def _solve_lp(self, grad, M, N, L):
        row_ind, col_ind = linear_sum_assignment(grad)
        Y = np.zeros((M, N))
        for r, c in zip(row_ind, col_ind):
            Y[r, c] = 1.0
        # Original paper: full assignment, then project (buggy but faithful)
        return self._project(Y, L)
    
    def _project(self, X, L):
        X = np.maximum(X, 0)
        s = X.sum()
        if s > 0:
            X = X * (L / s)
        for _ in range(20):
            rs = X.sum(axis=1, keepdims=True)
            X = np.where(rs > 1.0, X / rs, X)
            cs = X.sum(axis=0, keepdims=True)
            X = np.where(cs > 1.0, X / cs, X)
            s = X.sum()
            if s > 0:
                X = X * (L / s)
            if np.all(X.sum(axis=1) <= 1.0 + 1e-6) and np.all(X.sum(axis=0) <= 1.0 + 1e-6):
                break
        return np.clip(X, 0, 1)
    
    def _compute_score(self, A_G, A_H, C, X, L):
        r = X.sum(axis=1)
        U = np.outer(r, r)
        diff = U * A_G - X @ A_H @ X.T
        struct = np.sum(diff ** 2)
        app = np.sum(C * X)
        F = self.alpha * struct + (1 - self.alpha) * app
        return np.exp(-self.gamma * F / max(L, 1))


# =====================================================================
# OPTIMIZED GNCCP (Our Engine — no RDKit dependency for benchmarking)
# =====================================================================
class OptimizedGNCCP:
    """Our optimized engine with:
    - Rank-1 factorization: U = r @ r.T (quadratic)
    - Analytical gradient (Theorem 1)
    - Cached X @ A_H
    - Exact line search (5-point sampling)
    - Duality gap early stopping
    - Cosine ζ schedule
    - Proper L-partial LMO
    """
    
    def __init__(self, alpha=0.7, gamma=0.5, max_iter=8, fw_max_iter=3, tol=1e-3):
        self.alpha = alpha
        self.gamma = gamma
        self.max_iter = max_iter
        self.fw_max_iter = fw_max_iter
        self.tol = tol
    
    def match(self, A_G, A_H, C, L):
        M, N = A_G.shape[0], A_H.shape[0]
        X = np.ones((M, N)) * (L / (M * N))
        
        # Cosine ζ schedule
        t_sched = np.linspace(0, np.pi, self.max_iter)
        zeta_values = np.cos(t_sched)
        
        for zeta in zeta_values:
            for t in range(self.fw_max_iter):
                grad = self._compute_gradient_Jzeta(X, A_G, A_H, C, zeta)
                Y = self._solve_lp(grad, M, N, L)
                
                D = X - Y
                gap = np.sum(grad * D)
                if gap < self.tol:
                    break
                
                # Exact line search
                best_lam, best_val = 0.0, float('inf')
                for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    Xt = np.clip(X - lam * D, 0, 1)
                    val = self._compute_Jzeta(Xt, A_G, A_H, C, zeta)
                    if val < best_val:
                        best_val = val
                        best_lam = lam
                
                X = X - best_lam * D
            if float(np.max(X * (1.0 - X))) < 1e-3:
                break
        
        # Discretize directly via exact shifted Burkard LP oracle
        Y_binary = self._solve_lp(-X, M, N, L)
        
        return Y_binary, self._compute_score(A_G, A_H, C, Y_binary, L)
    
    def _compute_gradient_Jzeta(self, X, A_G, A_H, C, zeta):
        grad_F = self._compute_gradient_F(X, A_G, A_H, C)
        if zeta >= 0:
            grad = (1 - zeta) * grad_F + 2 * zeta * X
        else:
            grad = (1 + zeta) * grad_F + 2 * zeta * X
        return np.nan_to_num(grad, nan=0.0, posinf=1e6, neginf=-1e6)
    
    def _compute_gradient_F(self, X, A_G, A_H, C):
        """Analytical gradient with rank-1 U and cached XA_H."""
        r = X.sum(axis=1)
        U = np.outer(r, r)  # Rank-1: O(M^2) instead of O(MN^2)
        
        XA_H = X @ A_H  # Cache this — used twice
        R = (U * A_G) - (XA_H @ X.T)
        RA_X = (R * A_G) @ X
        term1 = RA_X.sum(axis=1, keepdims=True)
        term2 = R @ XA_H  # Reuse cached XA_H
        
        grad_struct = 4.0 * term1 - 4.0 * term2
        return self.alpha * grad_struct + (1 - self.alpha) * C
    
    def _compute_F(self, X, A_G, A_H, C):
        r = X.sum(axis=1)
        U = np.outer(r, r)
        diff = (U * A_G) - (X @ A_H @ X.T)
        struct = np.sum(diff ** 2)
        app = np.sum(C * X)
        return self.alpha * struct + (1 - self.alpha) * app
    
    def _compute_Jzeta(self, X, A_G, A_H, C, zeta):
        F = self._compute_F(X, A_G, A_H, C)
        x_norm_sq = np.sum(X ** 2)
        if zeta >= 0:
            return (1 - zeta) * F + zeta * x_norm_sq
        else:
            return (1 + zeta) * F + zeta * x_norm_sq
    
    def _solve_lp(self, grad, M, N, L):
        if L >= min(M, N):
            row_ind, col_ind = linear_sum_assignment(grad)
            Y = np.zeros((M, N), dtype=np.float32)
            for r, c in zip(row_ind, col_ind):
                if r < M and c < N: Y[r, c] = 1.0
            return Y
        dim = M + N - L
        g_min = float(np.min(grad))
        g_shift = grad - g_min + 1e-9 if g_min < 0 else grad
        aug = np.full((dim, dim), 1e6, dtype=np.float32)
        aug[:M, :N] = g_shift
        aug[:M, N:] = 0.0
        aug[M:, :N] = 0.0
        row_ind, col_ind = linear_sum_assignment(aug)
        Y = np.zeros((M, N), dtype=np.float32)
        for r, c in zip(row_ind, col_ind):
            if r < M and c < N: Y[r, c] = 1.0
        return Y
    
    def _project(self, X, L):
        X = np.maximum(X, 0)
        s = X.sum()
        if s > 0:
            X = X * (L / s)
        for _ in range(20):
            rs = X.sum(axis=1, keepdims=True)
            X = np.where(rs > 1.0, X / rs, X)
            cs = X.sum(axis=0, keepdims=True)
            X = np.where(cs > 1.0, X / cs, X)
            s = X.sum()
            if s > 0:
                X = X * (L / s)
            if np.all(X.sum(axis=1) <= 1.0 + 1e-6) and np.all(X.sum(axis=0) <= 1.0 + 1e-6):
                break
        return np.clip(X, 0, 1)
    
    def _compute_score(self, A_G, A_H, C, X, L):
        r = X.sum(axis=1)
        U = np.outer(r, r)
        diff = U * A_G - X @ A_H @ X.T
        struct = np.sum(diff ** 2)
        app = np.sum(C * X)
        F = self.alpha * struct + (1 - self.alpha) * app
        return np.exp(-self.gamma * F / max(L, 1))


# =====================================================================
# SYNTHETIC GRAPH GENERATOR
# =====================================================================
def generate_random_graph(n, edge_prob=0.3, seed=None):
    """Generate a random weighted graph with molecular-like structure."""
    rng = np.random.RandomState(seed)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < edge_prob:
                w = rng.choice([1.0, 1.5, 2.0, 3.0])  # single/aromatic/double/triple
                A[i, j] = w
                A[j, i] = w
    # Normalize
    norm = np.linalg.norm(A, 'fro')
    if norm > 0:
        A = A / norm
    return A

def generate_cost_matrix(M, N, seed=None):
    """Generate a random cost matrix with element-type penalties."""
    rng = np.random.RandomState(seed)
    C = rng.random((M, N))
    # Add some element mismatch penalties
    for i in range(M):
        for j in range(N):
            if rng.random() < 0.3:  # 30% chance of element mismatch
                C[i, j] = 50.0
    return C


# =====================================================================
# BENCHMARK RUNNER
# =====================================================================
def run_benchmark():
    print("=" * 70)
    print("  WCS-GNCCP Benchmark: Optimized vs. Naive (Original Paper)")
    print("=" * 70)
    
    # Use smaller iteration counts for benchmarking (fair comparison)
    naive_iters = 4
    naive_fw = 4
    opt_iters = 4
    opt_fw = 4
    
    # Graph sizes to benchmark
    # NOTE: Naive uses numerical gradient O(M*N * n^3), so n>15 takes forever
    sizes = [5, 8, 10, 12, 15]
    
    results = {
        'sizes': sizes,
        'naive_times': [],
        'opt_times': [],
        'naive_scores': [],
        'opt_scores': [],
        'speedups': [],
        'mappings': []  # Store the best mapping for visualization
    }
    
    for n in sizes:
        print(f"\n--- Graph Size: {n} x {n+2} ---")
        M, N = n, n + 2
        L = max(min(M, N) - 2, 1)
        
        A_G = generate_random_graph(M, edge_prob=0.4, seed=42)
        A_H = generate_random_graph(N, edge_prob=0.4, seed=123)
        C = generate_cost_matrix(M, N, seed=7)
        
        # --- Naive ---
        naive = NaiveGNCCP(max_iter=naive_iters, fw_max_iter=naive_fw)
        t0 = time.perf_counter()
        X_naive, score_naive = naive.match(A_G, A_H, C, L)
        t_naive = time.perf_counter() - t0
        
        # --- Optimized ---
        opt = OptimizedGNCCP(max_iter=opt_iters, fw_max_iter=opt_fw)
        t0 = time.perf_counter()
        X_opt, score_opt = opt.match(A_G, A_H, C, L)
        t_opt = time.perf_counter() - t0
        
        speedup = t_naive / max(t_opt, 1e-9)
        
        results['naive_times'].append(t_naive)
        results['opt_times'].append(t_opt)
        results['naive_scores'].append(float(score_naive))
        results['opt_scores'].append(float(score_opt))
        results['speedups'].append(speedup)
        
        # Extract mapping for visualization (from optimized)
        matches = list(zip(*np.where(X_opt > 0.5)))
        results['mappings'].append({
            'M': M, 'N': N, 'L': L,
            'matches': [(int(m[0]), int(m[1])) for m in matches],
            'A_G': A_G.tolist(),
            'A_H': A_H.tolist(),
            'score': float(score_opt)
        })
        
        print(f"  Naive:     {t_naive:.4f}s  (score: {score_naive:.4f})")
        print(f"  Optimized: {t_opt:.4f}s  (score: {score_opt:.4f})")
        print(f"  Speedup:   {speedup:.1f}x")
    
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"{'Size':>8} | {'Naive (s)':>12} | {'Ours (s)':>12} | {'Speedup':>10} | {'Naive Score':>12} | {'Our Score':>12}")
    print("-" * 78)
    for i, n in enumerate(sizes):
        print(f"  {n:>5}  | {results['naive_times'][i]:>12.4f} | {results['opt_times'][i]:>12.4f} | "
              f"{results['speedups'][i]:>9.1f}x | {results['naive_scores'][i]:>12.4f} | {results['opt_scores'][i]:>12.4f}")
    
    return results


# =====================================================================
# INTERACTIVE HTML VISUALIZATION
# =====================================================================
def generate_interactive_html(results, output_path):
    """Generate an interactive HTML page showing bijective mappings."""
    
    # Use the largest graph result for the main visualization
    best = results['mappings'][-1]
    M, N, L = best['M'], best['N'], best['L']
    matches = best['matches']
    A_G = best['A_G']
    A_H = best['A_H']
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WCS-GNCCP Bijective Mapping Visualization</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ 
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0a0a1a;
    color: #e0e0e0;
    min-height: 100vh;
  }}
  .header {{
    background: linear-gradient(135deg, #1a0a2e 0%, #16213e 100%);
    padding: 24px 40px;
    border-bottom: 2px solid #a4123f;
  }}
  .header h1 {{ color: #fff; font-size: 1.6em; }}
  .header p {{ color: #aaa; margin-top: 4px; }}
  .container {{ display: flex; flex-wrap: wrap; gap: 20px; padding: 20px; }}
  .panel {{ 
    background: #12122a; border-radius: 12px; padding: 20px;
    border: 1px solid #2a2a4a; flex: 1; min-width: 400px;
  }}
  .panel h2 {{ color: #a4123f; margin-bottom: 12px; font-size: 1.1em; }}
  canvas {{ background: #0d0d20; border-radius: 8px; display: block; margin: auto; }}
  .stats {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px; padding: 20px;
  }}
  .stat-card {{
    background: #12122a; border-radius: 10px; padding: 16px;
    border: 1px solid #2a2a4a; text-align: center;
  }}
  .stat-card .value {{ font-size: 2em; font-weight: 700; color: #a4123f; }}
  .stat-card .label {{ font-size: 0.85em; color: #888; margin-top: 4px; }}
  .controls {{
    padding: 12px 20px; display: flex; gap: 12px; align-items: center;
    flex-wrap: wrap;
  }}
  .btn {{
    background: #a4123f; color: white; border: none; padding: 10px 20px;
    border-radius: 6px; cursor: pointer; font-size: 0.9em; font-weight: 600;
    transition: all 0.2s;
  }}
  .btn:hover {{ background: #c4164f; transform: translateY(-1px); }}
  .btn.secondary {{ background: #2a2a4a; }}
  .btn.secondary:hover {{ background: #3a3a5a; }}
  select {{
    background: #1a1a3a; color: #e0e0e0; border: 1px solid #3a3a5a;
    padding: 10px 16px; border-radius: 6px; font-size: 0.9em;
  }}
  .legend {{ display: flex; gap: 20px; padding: 8px 20px; flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 0.85em; }}
  .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; }}
  .timing-table {{
    width: 100%; border-collapse: collapse; margin-top: 12px;
  }}
  .timing-table th, .timing-table td {{
    padding: 10px 14px; text-align: center; border-bottom: 1px solid #2a2a4a;
  }}
  .timing-table th {{ color: #a4123f; font-size: 0.85em; text-transform: uppercase; }}
  .timing-table td {{ font-family: 'Consolas', monospace; }}
  .speedup {{ color: #4caf50; font-weight: 700; }}
</style>
</head>
<body>

<div class="header">
  <h1>🔬 WCS-GNCCP Bijective Mapping — Interactive Visualization</h1>
  <p>Optimized Graph Matching Engine with Rank-1 Factorization, Analytical Gradient & Duality Gap</p>
</div>

<div class="stats">
  <div class="stat-card">
    <div class="value">{M}</div>
    <div class="label">Graph G Nodes</div>
  </div>
  <div class="stat-card">
    <div class="value">{N}</div>
    <div class="label">Graph H Nodes</div>
  </div>
  <div class="stat-card">
    <div class="value">{L}</div>
    <div class="label">Matched Pairs (L)</div>
  </div>
  <div class="stat-card">
    <div class="value">{best['score']:.4f}</div>
    <div class="label">Similarity Score</div>
  </div>
  <div class="stat-card">
    <div class="value">{results['speedups'][-1]:.1f}x</div>
    <div class="label">Speedup vs Naive</div>
  </div>
</div>

<div class="controls">
  <label>Graph Size:
    <select id="sizeSelect" onchange="changeSize()">
      {"".join(f'<option value="{i}" {"selected" if i == len(results["sizes"])-1 else ""}>{results["sizes"][i]}x{results["sizes"][i]+2}</option>' for i in range(len(results["sizes"])))}
    </select>
  </label>
  <button class="btn" onclick="animateMapping()">▶ Animate Mapping</button>
  <button class="btn secondary" onclick="resetView()">Reset</button>
  <button class="btn secondary" onclick="toggleEdges()">Toggle Edges</button>
</div>

<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background:#4fc3f7"></div> Graph G (Source)</div>
  <div class="legend-item"><div class="legend-dot" style="background:#ff7043"></div> Graph H (Target)</div>
  <div class="legend-item"><div class="legend-dot" style="background:#66bb6a"></div> Matched Pair</div>
  <div class="legend-item"><div class="legend-dot" style="background:#555"></div> Unmatched Node</div>
  <div class="legend-item"><div class="legend-dot" style="background:transparent; border:2px dashed #ffd54f; width:10px; height:10px"></div> Bijective Link</div>
</div>

<div class="container">
  <div class="panel">
    <h2>Bijective Atom Correspondence Map</h2>
    <canvas id="graphCanvas" width="800" height="500"></canvas>
  </div>
  <div class="panel" style="max-width: 500px;">
    <h2>Benchmark: Optimized vs Naive</h2>
    <canvas id="timingCanvas" width="440" height="300"></canvas>
    <table class="timing-table">
      <thead>
        <tr><th>Size</th><th>Naive (s)</th><th>Ours (s)</th><th>Speedup</th></tr>
      </thead>
      <tbody>
        {"".join(f'<tr><td>{results["sizes"][i]}×{results["sizes"][i]+2}</td><td>{results["naive_times"][i]:.4f}</td><td>{results["opt_times"][i]:.4f}</td><td class="speedup">{results["speedups"][i]:.1f}×</td></tr>' for i in range(len(results["sizes"])))}
      </tbody>
    </table>
  </div>
</div>

<script>
const ALL_DATA = {json.dumps(results['mappings'])};
const TIMING = {{
  sizes: {json.dumps(results['sizes'])},
  naive: {json.dumps(results['naive_times'])},
  opt: {json.dumps(results['opt_times'])},
  speedups: {json.dumps(results['speedups'])}
}};

let currentIdx = ALL_DATA.length - 1;
let showEdges = true;
let animPhase = 0;
let animId = null;

function layoutCircle(n, cx, cy, r) {{
  const pts = [];
  for (let i = 0; i < n; i++) {{
    const angle = (2 * Math.PI * i / n) - Math.PI / 2;
    pts.push({{ x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) }});
  }}
  return pts;
}}

function drawGraph() {{
  const canvas = document.getElementById('graphCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  
  const data = ALL_DATA[currentIdx];
  const M = data.M, N = data.N;
  const matches = data.matches;
  const matchedG = new Set(matches.map(m => m[0]));
  const matchedH = new Set(matches.map(m => m[1]));
  
  const gPts = layoutCircle(M, W * 0.25, H * 0.5, Math.min(150, H * 0.35));
  const hPts = layoutCircle(N, W * 0.75, H * 0.5, Math.min(150, H * 0.35));
  
  // Draw edges
  if (showEdges) {{
    ctx.globalAlpha = 0.15;
    ctx.strokeStyle = '#4fc3f7';
    ctx.lineWidth = 1;
    for (let i = 0; i < M; i++) {{
      for (let j = i+1; j < M; j++) {{
        if (data.A_G[i][j] > 0) {{
          ctx.beginPath();
          ctx.moveTo(gPts[i].x, gPts[i].y);
          ctx.lineTo(gPts[j].x, gPts[j].y);
          ctx.stroke();
        }}
      }}
    }}
    ctx.strokeStyle = '#ff7043';
    for (let i = 0; i < N; i++) {{
      for (let j = i+1; j < N; j++) {{
        if (data.A_H[i][j] > 0) {{
          ctx.beginPath();
          ctx.moveTo(hPts[i].x, hPts[i].y);
          ctx.lineTo(hPts[j].x, hPts[j].y);
          ctx.stroke();
        }}
      }}
    }}
    ctx.globalAlpha = 1;
  }}
  
  // Draw bijective mapping links
  const numVisible = animPhase >= 0 ? Math.min(animPhase, matches.length) : matches.length;
  for (let k = 0; k < numVisible; k++) {{
    const [gi, hj] = matches[k];
    const gp = gPts[gi], hp = hPts[hj];
    
    ctx.save();
    ctx.setLineDash([8, 4]);
    ctx.strokeStyle = '#ffd54f';
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.7;
    ctx.beginPath();
    ctx.moveTo(gp.x, gp.y);
    ctx.lineTo(hp.x, hp.y);
    ctx.stroke();
    ctx.restore();
    
    // Label
    const mx = (gp.x + hp.x) / 2, my = (gp.y + hp.y) / 2;
    ctx.fillStyle = '#ffd54f';
    ctx.font = '10px Consolas';
    ctx.textAlign = 'center';
    ctx.fillText(`${{gi}}→${{hj}}`, mx, my - 6);
  }}
  
  // Draw nodes
  for (let i = 0; i < M; i++) {{
    const matched = matchedG.has(i) && numVisible >= matches.findIndex(m => m[0] === i) + 1 && matches.some(m => m[0] === i);
    ctx.beginPath();
    ctx.arc(gPts[i].x, gPts[i].y, matched ? 14 : 10, 0, 2 * Math.PI);
    ctx.fillStyle = matched ? '#66bb6a' : '#4fc3f7';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = '#000';
    ctx.font = 'bold 11px Segoe UI';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(i, gPts[i].x, gPts[i].y);
  }}
  for (let i = 0; i < N; i++) {{
    const matched = matchedH.has(i) && numVisible >= matches.findIndex(m => m[1] === i) + 1 && matches.some(m => m[1] === i);
    ctx.beginPath();
    ctx.arc(hPts[i].x, hPts[i].y, matched ? 14 : 10, 0, 2 * Math.PI);
    ctx.fillStyle = matched ? '#66bb6a' : '#ff7043';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = '#000';
    ctx.font = 'bold 11px Segoe UI';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(i, hPts[i].x, hPts[i].y);
  }}
  
  // Labels
  ctx.fillStyle = '#4fc3f7';
  ctx.font = 'bold 16px Segoe UI';
  ctx.textAlign = 'center';
  ctx.fillText(`Graph G (${{M}} nodes)`, W * 0.25, 28);
  ctx.fillStyle = '#ff7043';
  ctx.fillText(`Graph H (${{N}} nodes)`, W * 0.75, 28);
  
  ctx.fillStyle = '#66bb6a';
  ctx.font = '13px Segoe UI';
  ctx.fillText(`Score: ${{data.score.toFixed(4)}} | L = ${{data.L}} matched pairs`, W * 0.5, H - 16);
}}

function drawTimingChart() {{
  const canvas = document.getElementById('timingCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  
  const pad = {{ l: 60, r: 20, t: 30, b: 40 }};
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  
  const maxT = Math.max(...TIMING.naive) * 1.1;
  const n = TIMING.sizes.length;
  const barW = pw / (n * 3);
  
  // Bars
  for (let i = 0; i < n; i++) {{
    const x = pad.l + i * (pw / n) + barW * 0.5;
    
    // Naive bar
    const h1 = (TIMING.naive[i] / maxT) * ph;
    ctx.fillStyle = '#ef5350';
    ctx.fillRect(x, pad.t + ph - h1, barW, h1);
    
    // Optimized bar
    const h2 = (TIMING.opt[i] / maxT) * ph;
    ctx.fillStyle = '#66bb6a';
    ctx.fillRect(x + barW + 4, pad.t + ph - h2, barW, h2);
    
    // Labels
    ctx.fillStyle = '#888';
    ctx.font = '11px Consolas';
    ctx.textAlign = 'center';
    ctx.fillText(`${{TIMING.sizes[i]}}`, x + barW, H - 10);
    
    // Speedup label
    ctx.fillStyle = '#ffd54f';
    ctx.font = 'bold 11px Consolas';
    const topY = pad.t + ph - Math.max(h1, h2) - 14;
    ctx.fillText(`${{TIMING.speedups[i].toFixed(0)}}×`, x + barW, topY);
  }}
  
  // Y-axis
  ctx.fillStyle = '#888';
  ctx.font = '10px Consolas';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {{
    const v = (maxT * i / 4);
    const y = pad.t + ph - (i / 4) * ph;
    ctx.fillText(v.toFixed(2) + 's', pad.l - 6, y + 4);
    ctx.strokeStyle = '#1a1a3a';
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(W - pad.r, y);
    ctx.stroke();
  }}
  
  // Legend
  ctx.fillStyle = '#ef5350'; ctx.fillRect(pad.l, 6, 14, 14);
  ctx.fillStyle = '#aaa'; ctx.font = '11px Segoe UI'; ctx.textAlign = 'left';
  ctx.fillText('Naive (Numerical Gradient)', pad.l + 20, 17);
  ctx.fillStyle = '#66bb6a'; ctx.fillRect(pad.l + 220, 6, 14, 14);
  ctx.fillStyle = '#aaa';
  ctx.fillText('Ours (Analytical + Rank-1)', pad.l + 240, 17);
}}

function changeSize() {{
  currentIdx = parseInt(document.getElementById('sizeSelect').value);
  animPhase = -1;
  drawGraph();
}}

function animateMapping() {{
  if (animId) cancelAnimationFrame(animId);
  animPhase = 0;
  const total = ALL_DATA[currentIdx].matches.length;
  function step() {{
    drawGraph();
    if (animPhase <= total) {{
      animPhase++;
      animId = setTimeout(step, 600);
    }}
  }}
  step();
}}

function resetView() {{
  if (animId) {{ clearTimeout(animId); animId = null; }}
  animPhase = -1;
  drawGraph();
}}

function toggleEdges() {{
  showEdges = !showEdges;
  drawGraph();
}}

// Initial draw
drawGraph();
drawTimingChart();
</script>
</body>
</html>"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n[OK] Interactive visualization saved to: {output_path}")


# =====================================================================
# MAIN
# =====================================================================
if __name__ == '__main__':
    results = run_benchmark()
    
    # Save interactive HTML
    script_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(script_dir, 'benchmark_visualization.html')
    generate_interactive_html(results, html_path)
    
    print(f"\n[DONE] Open {html_path} in your browser.")
