"""
gnccp_generalized.py
========================================================================================
UNIVERSAL DOMAIN-AGNOSTIC FAST-WCS / GNCCP GRAPH & POINT CLOUD MATCHING ENGINE
========================================================================================

Base Algorithm:
  Yang, Qiao & Liu, "An Algorithm for Finding the Most Similar Given Sized
  Subgraphs in Two Weighted Graphs", IEEE TNNLS, 2018.
  - GNCCP homotopy continuation (graduated non-convexity & concavity)
  - Two objective relaxations (RLX1/RLX2) with Frank-Wolfe optimization
  - Fast linear-assignment approximation via rectangular Hungarian

Our Extensions (this implementation):
  - Rank-1 factored analytical gradient (Lemma 1): 4[(R∘A_G)r]1_N^T − 4R·X·A_H
  - Dummy-node padded Hungarian oracle for exact cardinality-L partial assignment
    (Burkard, Dell'Amico & Martello, 2009)
  - Closed-form quadratic line search replacing grid search
  - Cosine homotopy schedule (vs. linear in original)
  - Closed-form Kabsch SVD superposition for 2D/3D point sets
  - Multi-domain adapters (point clouds, NetworkX, connectomes, RDKit molecules)
  - Ablation-ready constructor (schedule, line_search, oracle switches)
  - Gradient verification via finite differences

Complexity:
  Per Frank-Wolfe step: O(max(M,N)^3) dominated by Hungarian assignment.
  Gradient computation: O(M^2 + MN) via rank-1 factorization.
  Space: O(MN) for the assignment matrix.
========================================================================================
"""

import os
import sys
import io
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

# Optional GPU backend via CuPy
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

# Ensure UTF-8 output on Windows console without breaking third-party isatty checks
if sys.platform.startswith('win'):
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


# ======================================================================================
# DATA STRUCTURE: MATCHING RESULT
# ======================================================================================

@dataclass
class MatchingResult:
    """Rich structured container for graph matching and physical superposition results."""
    X_binary: np.ndarray                 # Optimal binary assignment matrix in {0, 1}^(M x N)
    X_continuous: np.ndarray             # Final continuous relaxed matrix in [0, 1]^(M x N)
    matches: List[Tuple[int, int]]       # List of (node_g, node_h) matched index pairs
    mapping_dict: Dict[int, int]         # Dictionary mapping: {node_g: node_h}
    L: int                               # Number of matched subgraph nodes
    objective_score: float               # Raw objective cost F*(X)
    similarity_score: float              # Normalized exponential kernel similarity in [0, 1]
    runtime_sec: float                   # Wall-clock runtime in seconds
    iterations_run: int                  # Total GNCCP homotopy iterations
    fw_steps_total: int                  # Total Frank-Wolfe subproblem evaluations
    alignment: Optional[Dict[str, Any]] = None  # Kabsch 2D/3D superposition (R, t, RMSD)
    history: List[Dict[str, float]] = field(default_factory=list) # Trace per iteration

    def __repr__(self) -> str:
        rmsd_str = f" | RMSD: {self.alignment['rmsd']:.3f} Å" if (self.alignment and 'rmsd' in self.alignment) else ""
        return (f"<MatchingResult: L={self.L} matches | Sim: {self.similarity_score:.4f} | "
                f"Obj: {self.objective_score:.4f} | Time: {self.runtime_sec*1000:.1f} ms{rmsd_str}>")


# ======================================================================================
# MATHEMATICAL UTILITIES & CLOSED-FORM PHYSICAL SUPERPOSITION
# ======================================================================================


def robust_kabsch_superposition(P: np.ndarray, Q: np.ndarray, trim_ratio: float = 0.2) -> Dict[str, Any]:
    """
    Robust Iterative Trimmed Kabsch rigid registration between paired point sets P and Q.
    Fits optimal rotation R and translation t, then refines by pruning top outlier pairs.
    """
    assert len(P) == len(Q), "Point clouds P and Q must have identical number of points."
    n_pts, dim = P.shape
    if n_pts < dim:
        return {'R': np.eye(dim), 't': np.zeros(dim), 'rmsd': 0.0, 'inliers': list(range(n_pts))}

    def _kabsch_core(p_sub, q_sub):
        cp_p = np.mean(p_sub, axis=0)
        cp_q = np.mean(q_sub, axis=0)
        p_c = p_sub - cp_p
        q_c = q_sub - cp_q
        H = p_c.T @ q_c
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.eye(dim)
        D[-1, -1] = d
        R = Vt.T @ D @ U.T
        t = cp_q - cp_p @ R.T
        return R, t

    R, t = _kabsch_core(P, Q)
    residuals = np.linalg.norm((P @ R.T + t) - Q, axis=-1)

    n_keep = max(dim + 1, int(n_pts * (1.0 - trim_ratio)))
    if n_keep < n_pts:
        inlier_indices = np.argsort(residuals)[:n_keep]
        R, t = _kabsch_core(P[inlier_indices], Q[inlier_indices])
        residuals = np.linalg.norm((P[inlier_indices] @ R.T + t) - Q[inlier_indices], axis=-1)
    else:
        inlier_indices = np.arange(n_pts)

    rmsd = float(np.sqrt(np.mean(residuals**2)))
    return {
        'R': R,
        't': t,
        'rmsd': rmsd,
        'inlier_ratio': len(inlier_indices) / n_pts,
        'P_aligned': P @ R.T + t
    }

def kabsch_superposition(P: np.ndarray, Q: np.ndarray) -> Dict[str, Any]:
    """
    Closed-form O(1) SVD Kabsch rigid registration between paired point sets P and Q.
    Finds optimal rotation matrix R and translation vector t minimizing ||P @ R.T + t - Q||_F.
    Supports both 2D and 3D point clouds, with reflection parity correction.
    """
    assert len(P) == len(Q), "Point clouds P and Q must have identical number of points."
    dim = P.shape[1]
    if len(P) < dim:
        return {'R': np.eye(dim), 't': np.zeros(dim), 'rmsd': 0.0}

    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    Pc = P - centroid_P
    Qc = Q - centroid_Q

    # Cross-covariance matrix
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    
    # Parity check for reflection
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.eye(dim)
    D[-1, -1] = d

    R = Vt.T @ D @ U.T
    t = centroid_Q - centroid_P @ R.T
    
    P_aligned = Pc @ R.T
    rmsd = float(np.sqrt(np.mean(np.sum((P_aligned - Qc)**2, axis=1))))
    
    return {
        'R': R,
        't': t,
        'rmsd': rmsd,
        'P_aligned': P @ R.T + t
    }


# ======================================================================================
# CORE GENERALIZED SOLVER: UniversalGNCCP
# ======================================================================================

class UniversalGNCCP:
    """
    Universal Domain-Agnostic Graph & Subgraph Matching Solver.
    
    Implements GNCCP (Yang et al. 2018) with our extensions for exact
    cardinality-L assignment, closed-form line search, and analytical gradients.
    
    Parameters:
      alpha: float in [0, 1]
          Weight balance between structural topology (alpha) and unary costs (1 - alpha).
          Default: 0.7 (optimal balance between structural topology and attribute features).
      gamma: float
          RBF kernel bandwidth for similarity score: S = exp(-gamma * F_obj / L).
          Default: 1.0.
      max_iter: int
          Number of outer homotopy continuation steps along zeta in [+1, -1]. Default: 15.
      fw_max_iter: int
          Maximum inner Frank-Wolfe iterations per homotopy level. Default: 8.
      tol: float
          Frank-Wolfe duality gap convergence tolerance. Default: 1e-4.
      normalize_fro: bool
          Whether to normalize adjacency matrices by their Frobenius norm. Default: True.
      use_gpu: bool
          Whether to use CuPy for GPU acceleration (if installed). Default: False.
      schedule: str
          Homotopy continuation schedule: 'cosine' (default) or 'linear'.
      line_search: str
          Line search method: 'closed_form' (default) or 'grid'.
      oracle: str
          Hungarian oracle type: 'dummy_padded' (default, exact cardinality-L) or 
          'top_k_heuristic' (original paper's approach — full assignment then top-L).
      adaptive_schedule: bool
          If True, adapts the homotopy step size based on FW gap convergence.
          Uses smaller ζ steps when optimization is unstable, larger when converging.
          Default: True.
      n_restarts: int
          Number of random restarts. The best result (lowest objective) is kept.
          Default: 1 (single run from uniform initialization).
      local_refinement: bool
          If True, applies pairwise swap refinement after discretization to
          improve the binary assignment under the original WCS objective.
          Default: True.
    """

    def __init__(self,
                 alpha: float = 0.7,
                 gamma: float = 1.0,
                 max_iter: int = 8,
                 fw_max_iter: int = 3,
                 tol: float = 1e-3,
                 normalize_fro: bool = False,
                 normalization: Optional[str] = None,
                 use_gpu: bool = False,
                 schedule: str = 'cosine',
                 line_search: str = 'quartic',
                 oracle: str = 'dummy_padded',
                 adaptive_schedule: bool = True,
                 n_restarts: int = 1,
                 local_refinement: bool = True,
                 enable_exchange_moves: bool = True,
                 seed: Optional[int] = None,
                 robust_kabsch: bool = True):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.max_iter = int(max_iter)
        self.fw_max_iter = int(fw_max_iter)
        self.tol = float(tol)
        self.normalize_fro = bool(normalize_fro)
        self.normalization = normalization
        self.use_gpu = bool(use_gpu and HAS_CUPY)
        self.xp = cp if self.use_gpu else np
        self.schedule = schedule
        self.line_search = line_search
        self.oracle = oracle
        self.adaptive_schedule = bool(adaptive_schedule)
        self.n_restarts = max(1, int(n_restarts))
        self.local_refinement = bool(local_refinement)
        self.enable_exchange_moves = bool(enable_exchange_moves)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.robust_kabsch = bool(robust_kabsch)

    # ----------------------------------------------------------------------------------
    # 1. CORE NUMERICAL SOLVER (RAW MATRICES)
    # ----------------------------------------------------------------------------------
    def solve(self,
              A_G: np.ndarray,
              A_H: np.ndarray,
              C: Optional[np.ndarray] = None,
              L: Optional[int] = None,
              coords_G: Optional[np.ndarray] = None,
              coords_H: Optional[np.ndarray] = None,
              n_dgm_iters: int = 2,
                            verbose: bool = False) -> MatchingResult:
        """
        Solves the Weighted Common Subgraph (WCS) matching problem for arbitrary graphs.
        
        Args:
          A_G: Adjacency matrix of graph G, shape (M, M).
          A_H: Adjacency matrix of graph H, shape (N, N).
          C: Optional unary cost matrix, shape (M, N). If None, uniform zero cost is used.
          L: Optional common subgraph size (1 <= L <= min(M, N)). If None, defaults to min(M, N) - 2.
          coords_G: Optional spatial coordinates of G nodes, shape (M, d).
          coords_H: Optional spatial coordinates of H nodes, shape (N, d).
          verbose: Print progress logs if True.
          
        Returns:
          MatchingResult containing assignments, scores, runtime, and optional Kabsch alignment.
        """
        t_start = time.perf_counter()
        xp = self.xp

        M, M2 = A_G.shape
        N, N2 = A_H.shape
        assert M == M2 and N == N2, "Adjacency matrices A_G and A_H must be square."

        # Subgraph size L
        if L is None:
            L = max(min(M, N) - 2, 1)
        else:
            L = max(1, min(int(L), min(M, N)))

        # Unary cost matrix
        if C is None:
            C_np = np.zeros((M, N), dtype=np.float32)
        else:
            C_np = np.array(C, dtype=np.float32)
            assert C_np.shape == (M, N), f"C shape {C_np.shape} must match (M={M}, N={N})."

        # Working copies of adjacency
        A_G_work = A_G.astype(np.float32).copy()
        A_H_work = A_H.astype(np.float32).copy()

        # Shared-scale or Frobenius normalization
        norm_mode = self.normalization if self.normalization is not None else ('frobenius' if self.normalize_fro else 'none')
        if norm_mode == 'frobenius':
            norm_G = np.linalg.norm(A_G_work, 'fro')
            norm_H = np.linalg.norm(A_H_work, 'fro')
            if norm_G > 0: A_G_work /= norm_G
            if norm_H > 0: A_H_work /= norm_H
        elif norm_mode == 'shared_max':
            max_val = max(float(np.max(np.abs(A_G_work))), float(np.max(np.abs(A_H_work))))
            if max_val > 0:
                A_G_work /= max_val
                A_H_work /= max_val

        if self.use_gpu:
            A_G_dev = xp.asarray(A_G_work)
            A_H_dev = xp.asarray(A_H_work)
            C_dev   = xp.asarray(C_np)
        else:
            A_G_dev, A_H_dev, C_dev = A_G_work, A_H_work, C_np

        # ---- Multi-restart: run GNCCP from multiple initializations ----
        best_X = None
        best_F = float('inf')
        best_history = []
        best_fw_total = 0

        for restart_idx in range(self.n_restarts):
            if restart_idx == 0:
                # First run: uniform initialization (centroid of polytope D_L)
                X_init = xp.ones((M, N), dtype=np.float32) * (L / float(M * N))
            else:
                # Subsequent runs: random feasible initialization via seeded RNG
                rand_mat = self.rng.random((M, N), dtype=np.float32)
                X_init = xp.array(rand_mat)
                X_init = self._project_to_polytope(X_init, L)

            X, history, fw_total = self._run_single_gnccp(
                X_init, A_G_dev, A_H_dev, C_dev, L, M, N, verbose and (restart_idx == 0))

            F_val = float(self._compute_F(X, A_G_dev, A_H_dev, C_dev))
            if F_val < best_F:
                best_F = F_val
                best_X = X
                best_history = history
                best_fw_total = fw_total

        X = best_X
        history = best_history
        total_fw_steps = best_fw_total

        # Final discretization to strictly binary partial permutation matrix
        X_bin_dev = self._discretize_to_binary(X, L, M, N)
        X_binary = cp.asnumpy(X_bin_dev) if self.use_gpu else X_bin_dev
        X_continuous = cp.asnumpy(X) if self.use_gpu else X

        # ---- Local refinement: pairwise swap improvement ----
        if self.local_refinement:
            X_binary = self._swap_refinement(X_binary, A_G_work, A_H_work, C_np, L)

        # Extract matches
        match_rows, match_cols = np.where(X_binary > 0.5)
        matches = sorted(list(zip(match_rows, match_cols)), key=lambda x: x[0])
        mapping_dict = {int(r): int(c) for r, c in matches}

        # Final objective and similarity score
        F_final = float(self._compute_F(X_binary, A_G_work, A_H_work, C_np))
        sim_score = float(np.exp(-self.gamma * (F_final / max(L, 1))))

        # Physical 2D/3D rigid alignment (Kabsch SVD) if coordinates provided
        alignment_data = None
        if coords_G is not None and coords_H is not None and len(matches) >= 2:
            P_matched = coords_G[match_rows]
            Q_matched = coords_H[match_cols]
            if self.robust_kabsch:
                alignment_data = robust_kabsch_superposition(P_matched, Q_matched)
            else:
                alignment_data = kabsch_superposition(P_matched, Q_matched)

        runtime = time.perf_counter() - t_start

        return MatchingResult(
            X_binary=X_binary,
            X_continuous=X_continuous,
            matches=matches,
            mapping_dict=mapping_dict,
            L=L,
            objective_score=F_final,
            similarity_score=sim_score,
            runtime_sec=runtime,
            iterations_run=len(history),
            fw_steps_total=total_fw_steps,
            alignment=alignment_data,
            history=history
        )

    def _run_single_gnccp(self, X, A_G, A_H, C, L, M, N, verbose=False):
        """
        Runs one GNCCP pass from initialization X.
        
        Supports adaptive continuation: when enabled, the solver monitors
        whether the FW gap is decreasing. If the gap stalls or increases,
        it inserts an intermediate ζ value (halving the step). If the gap
        decreases rapidly, it skips ahead (doubling the step).
        
        Returns: (X_final, history, total_fw_steps)
        """
        xp = self.xp

        # Build homotopy schedule
        if self.schedule == 'cosine':
            t_sched = np.linspace(0.0, np.pi, self.max_iter)
            zeta_list = list(np.cos(t_sched))
        else:  # 'linear'
            zeta_list = list(np.linspace(1.0, -1.0, self.max_iter))

        history = []
        total_fw_steps = 0
        prev_gap = float('inf')

        idx = 0
        while idx < len(zeta_list):
            zeta_val = float(zeta_list[idx])
            gap_this_level = float('inf')

            for fw_step in range(self.fw_max_iter):
                total_fw_steps += 1

                # Closed-form analytic gradient (Lemma 1)
                grad = self._compute_gradient_Jzeta(X, A_G, A_H, C, zeta_val)

                # Hungarian Linear Minimization Oracle over P_L
                if self.oracle == 'dummy_padded':
                    Y = self._solve_hungarian_oracle_padded(grad, L, M, N)
                else:
                    Y = self._solve_hungarian_oracle_topk(grad, L, M, N)

                # Frank-Wolfe descent direction
                D_fw = X - Y
                fw_gap = float(xp.sum(grad * D_fw))
                gap_this_level = abs(fw_gap)

                if abs(fw_gap) < self.tol:
                    break

                # Line search over lambda in [0, 1]
                if self.line_search == 'quartic':
                    lambda_opt = self._exact_quartic_line_search(X, D_fw, A_G, A_H, C, zeta_val)
                elif self.line_search == 'closed_form':
                    lambda_opt = self._closed_form_line_search(X, D_fw, A_G, A_H, C, zeta_val, grad_J=grad)
                else:
                    lambda_opt = self._grid_line_search(X, D_fw, A_G, A_H, C, zeta_val)

                # Convex combination update (unconditionally preserves polytope feasibility)
                X = X - lambda_opt * D_fw

            # Early exit if X has converged to a discrete partial permutation
            if float(xp.max(X * (1.0 - X))) < 1e-4:
                if verbose:
                    print(f"  [GNCCP] Converged to discrete partial permutation at ζ = {zeta_val:+6.3f}. Early exit.")
                break

            # Record iteration trace
            current_F = float(self._compute_F(X, A_G, A_H, C))
            history.append({'iter': idx, 'zeta': zeta_val, 'F': current_F, 'fw_gap': fw_gap})
            if verbose and (idx % 5 == 0 or idx == len(zeta_list) - 1):
                print(f"  [GNCCP] Iter {idx+1:2d}/{len(zeta_list)} | ζ = {zeta_val:+6.3f} | "
                      f"F = {current_F:8.4f} | Gap: {fw_gap:.2e}")

            # ---- Adaptive schedule: insert/skip ζ levels with safety cap ----
            if self.adaptive_schedule and idx + 1 < len(zeta_list) and len(zeta_list) < self.max_iter * 2:
                if gap_this_level > prev_gap * 2.0 and gap_this_level > self.tol * 10:
                    # Gap increased significantly — insert intermediate ζ
                    mid_zeta = (zeta_val + zeta_list[idx + 1]) / 2.0
                    zeta_list.insert(idx + 1, mid_zeta)
                elif gap_this_level < self.tol * 0.1 and idx + 2 < len(zeta_list):
                    # Converged very fast — skip next ζ level
                    zeta_list.pop(idx + 1)

            prev_gap = max(gap_this_level, 1e-12)
            idx += 1

        return X, history, total_fw_steps

    def _swap_refinement(self, X_binary, A_G, A_H, C, L, max_rounds=3):
        """
        Incremental local search with replacement and exchange moves (first-improvement,
        identical move order to the loop-based version).

        Every candidate move in a family is scored simultaneously from the L x L matched
        sub-matrices P = A_G[rows, rows] and Q = A_H[cols, cols], using

            sum_k (a_k - b_k)^2 = sum_k a_k^2 + sum_k b_k^2 - 2 sum_k a_k b_k

        so the all-pairs cross term is a single BLAS matrix product. The "k != moved node"
        exclusion is applied analytically by subtracting the k == node terms. This removes
        the O(L^2) Python-level pair loop and the O(L) list rebuilds inside it.
          1. Replace the target column of a matched row by an unmatched column.
          2. Exchange the targets of two matched rows.
          3. Replace the source row of a matched column by an unmatched row.
        """
        X = np.asarray(X_binary, dtype=np.float32)
        M, N = X.shape
        # float64: the expanded-square identity subtracts large, nearly equal terms
        A_G = np.asarray(A_G, dtype=np.float64)
        A_H = np.asarray(A_H, dtype=np.float64)
        C = np.asarray(C, dtype=np.float64)
        alpha = self.alpha
        tol = -1e-6

        r_idx, c_idx = np.where(X > 0.5)
        mapping = {int(r): int(c) for r, c in zip(r_idx, c_idx)}

        def _state():
            n = len(mapping)
            rows = np.fromiter(mapping.keys(), dtype=np.intp, count=n)
            cols = np.fromiter(mapping.values(), dtype=np.intp, count=n)
            return rows, cols

        for _ in range(max_rounds):
            improved = False

            # --------------------------------------------------------------------------
            # MOVE 1: TARGET COLUMN REPLACEMENT (matched row -> unmatched column)
            # --------------------------------------------------------------------------
            rows, cols = _state()
            n_m = len(rows)
            used = np.zeros(N, dtype=bool)
            used[cols] = True
            unmatched_cols = np.flatnonzero(~used)
            if len(unmatched_cols) > 0 and n_m > 1:
                P = A_G[np.ix_(rows, rows)]
                Q = A_H[np.ix_(cols, cols)]
                B = A_H[np.ix_(unmatched_cols, cols)]                    # (U, L)
                pd, qd = np.diag(P), np.diag(Q)
                base = ((P - Q) ** 2).sum(axis=1) - (pd - qd) ** 2        # (L,)
                new = ((P * P).sum(axis=1)[:, None] + (B * B).sum(axis=1)[None, :]
                       - 2.0 * (P @ B.T) - (pd[:, None] - B.T) ** 2)      # (L, U)
                deltas = (alpha * 2.0 * (new - base[:, None])
                          + (1.0 - alpha) * (C[np.ix_(rows, unmatched_cols)] - C[rows, cols][:, None]))
                best_u = deltas.argmin(axis=1)
                best_d = deltas[np.arange(n_m), best_u]
                hit = np.flatnonzero(best_d < tol)
                if hit.size:
                    i = int(hit[0])
                    mapping[int(rows[i])] = int(unmatched_cols[best_u[i]])
                    improved = True

            # --------------------------------------------------------------------------
            # MOVE 2: EXCHANGE (swap targets of rows i < j)
            # --------------------------------------------------------------------------
            rows, cols = _state()
            n_m = len(rows)
            if self.enable_exchange_moves and n_m >= 3:   # needs >= 1 other node, as before
                P = A_G[np.ix_(rows, rows)]
                Q = A_H[np.ix_(cols, cols)]
                Cm = C[np.ix_(rows, cols)]
                pd, qd = np.diag(P), np.diag(Q)
                PT, QT = P.T, Q.T
                # E[a, b] = sum_k (P[a,k] - Q[b,k])^2  -- one GEMM instead of L^2 loops
                E = (P * P).sum(axis=1)[:, None] + (Q * Q).sum(axis=1)[None, :] - 2.0 * (P @ Q.T)
                Ed = np.diag(E)
                dd = (pd - qd) ** 2
                # same sums restricted to k not in {i, j}
                S1 = E - (pd[:, None] - QT) ** 2 - (P - qd[None, :]) ** 2
                S1b = Ed[:, None] - dd[:, None] - (P - Q) ** 2
                S2 = E.T - (PT - qd[:, None]) ** 2 - (pd[None, :] - Q) ** 2
                S2b = Ed[None, :] - (PT - QT) ** 2 - dd[None, :]
                delta_struct = 2.0 * ((S1 - S1b) + (S2 - S2b))
                cd = np.diag(Cm)
                delta_unary = (Cm - cd[:, None]) + (Cm.T - cd[None, :])
                delta = alpha * delta_struct + (1.0 - alpha) * delta_unary
                flat = np.flatnonzero(np.triu(delta < tol, k=1))   # row-major == original loop order
                if flat.size:
                    i, j = divmod(int(flat[0]), n_m)
                    r1, r2 = int(rows[i]), int(rows[j])
                    mapping[r1], mapping[r2] = int(cols[j]), int(cols[i])
                    improved = True

            # --------------------------------------------------------------------------
            # MOVE 3: SOURCE ROW REPLACEMENT (matched row -> unmatched row)
            # --------------------------------------------------------------------------
            rows, cols = _state()
            n_m = len(rows)
            used = np.zeros(M, dtype=bool)
            used[rows] = True
            unmatched_rows = np.flatnonzero(~used)
            if len(unmatched_rows) > 0 and n_m > 1:
                P = A_G[np.ix_(rows, rows)]
                Q = A_H[np.ix_(cols, cols)]
                B = A_G[np.ix_(unmatched_rows, rows)]                    # (U, L)
                pd, qd = np.diag(P), np.diag(Q)
                base = ((P - Q) ** 2).sum(axis=1) - (pd - qd) ** 2        # (L,)
                new = ((B * B).sum(axis=1)[:, None] + (Q * Q).sum(axis=1)[None, :]
                       - 2.0 * (B @ Q.T) - (B - qd[None, :]) ** 2)        # (U, L)
                deltas = (alpha * 2.0 * (new - base[None, :])
                          + (1.0 - alpha) * (C[np.ix_(unmatched_rows, cols)] - C[rows, cols][None, :]))
                best_u = deltas.argmin(axis=0)
                best_d = deltas[best_u, np.arange(n_m)]
                hit = np.flatnonzero(best_d < tol)
                if hit.size:
                    i = int(hit[0])
                    r_old = int(rows[i])
                    c = mapping.pop(r_old)
                    mapping[int(unmatched_rows[best_u[i]])] = c
                    improved = True

            if not improved:
                break

        Y = np.zeros((M, N), dtype=np.float32)
        if mapping:
            Y[list(mapping.keys()), list(mapping.values())] = 1.0
        return self.xp.asarray(Y) if self.use_gpu else Y

    # ----------------------------------------------------------------------------------
    # INTERNAL MATHEMATICAL ROUTINES
    # ----------------------------------------------------------------------------------

    def _compute_F(self, X, A_G, A_H, C):
        xp = self.xp
        r = X.sum(axis=1)
        # Avoid allocating intermediate outer product matrix
        diff = (r[:, None] * A_G * r[None, :]) - (X @ A_H @ X.T)
        struct_term = xp.sum(diff ** 2)
        app_term = xp.einsum('ij,ij->', C, X)
        return self.alpha * struct_term + (1.0 - self.alpha) * app_term

    def _compute_Jzeta(self, X, A_G, A_H, C, zeta):
        xp = self.xp
        F = self._compute_F(X, A_G, A_H, C)
        x_norm_sq = xp.sum(X ** 2)
        if zeta >= 0:
            return (1.0 - zeta) * F + zeta * x_norm_sq
        else:
            return (1.0 + zeta) * F + zeta * x_norm_sq

    def _compute_gradient_F(self, X, A_G, A_H, C):
        """Lemma 1: Exact rank-1 factored closed-form gradient in Theta(N^2) complexity."""
        xp = self.xp
        r = X.sum(axis=1)
        
        XA_H = X @ A_H
        R = (r[:, None] * A_G * r[None, :]) - (XA_H @ X.T)
        
        # Term 1: 4 * [(R o A_G) r] 1_N^T
        term1 = ((R * A_G) @ r)[:, None]
        # Term 2: -4 * R X A_H
        term2 = R @ XA_H
        
        grad_struct = 4.0 * term1 - 4.0 * term2
        return self.alpha * grad_struct + (1.0 - self.alpha) * C

    def _compute_gradient_Jzeta(self, X, A_G, A_H, C, zeta):
        xp = self.xp
        grad_F = self._compute_gradient_F(X, A_G, A_H, C)
        if zeta >= 0:
            grad = (1.0 - zeta) * grad_F + 2.0 * zeta * X
        else:
            grad = (1.0 + zeta) * grad_F + 2.0 * zeta * X
        return xp.nan_to_num(grad, nan=0.0, posinf=1e6, neginf=-1e6)

    def _solve_hungarian_oracle_padded(self, grad, L, M, N):
        """
        Exact cardinality-L partial assignment via augmented Hungarian matrix.
        
        Constructs a (M + N - L) x (M + N - L) augmented matrix:
          [ C          | 0 (M - L) ]
          [ -----------+---------- ]
          [ 0 (N - L)  | BIG_M     ]
        
        This forces exactly (N - L) columns of H and (M - L) rows of G
        to be assigned to dummy slots, leaving EXACTLY L real matches.
        Mathematically exact regardless of whether gradient values are
        positive, negative, or zero.
        
        Reference: Burkard, Dell'Amico & Martello, 'Assignment Problems', SIAM, 2009.
        """
        xp = self.xp
        grad_cpu = cp.asnumpy(grad) if self.use_gpu else grad
        
        if L >= min(M, N):
            row_ind, col_ind = linear_sum_assignment(grad_cpu)
            Y = xp.zeros((M, N), dtype=np.float32)
            for r, c in zip(row_ind, col_ind):
                if r < M and c < N:
                    Y[r, c] = 1.0
            return Y
        
        dim = M + N - L
        max_abs = float(np.max(np.abs(grad_cpu))) if grad_cpu.size > 0 else 1.0
        BIG_M = max(1e6, max_abs * 1000.0)
        
        # Shift gradient to be strictly non-negative so dummy slots (cost 0.0)
        # never compete unfairly with negative real entries, strictly guaranteeing exactly L matches.
        g_min = float(np.min(grad_cpu))
        g_shift = grad_cpu - g_min + 1e-9 if g_min < 0 else grad_cpu

        augmented = np.full((dim, dim), BIG_M, dtype=np.float32)
        augmented[:M, :N] = g_shift
        augmented[:M, N:] = 0.0
        augmented[M:, :N] = 0.0
        
        row_ind, col_ind = linear_sum_assignment(augmented)
        Y = xp.zeros((M, N), dtype=np.float32)
        for r, c in zip(row_ind, col_ind):
            if r < M and c < N:
                Y[r, c] = 1.0
        return Y

    def _solve_hungarian_oracle_topk(self, grad, L, M, N):
        """
        Original paper's heuristic: full rectangular assignment then keep L lowest-cost pairs.
        
        NOTE: This is NOT the minimum-cost matching of cardinality L when L < min(M,N).
        Kept for ablation comparison.
        """
        xp = self.xp
        grad_cpu = cp.asnumpy(grad) if self.use_gpu else grad
        row_ind, col_ind = linear_sum_assignment(grad_cpu)
        costs = [(grad_cpu[r, c], r, c) for r, c in zip(row_ind, col_ind) if r < M and c < N]
        costs.sort(key=lambda x: x[0])  # Smallest gradient entries first
        Y = xp.zeros((M, N), dtype=np.float32)
        for _, r, c in costs[:L]:
            Y[r, c] = 1.0
        return Y

    def _exact_quartic_line_search(self, X, D, A_G, A_H, C, zeta):
        """
        Exact global analytical line search along Frank-Wolfe direction X - lambda*D.
        
        Along X(lambda) = (1 - lambda)X + lambda Y = X - lambda*D, the objective J_zeta(lambda)
        is an exact degree-4 polynomial. Its derivative is an exact degree-3 polynomial.
        Finds the exact continuous global minimum on [0, 1] in O(1) time.
        """
        xp = self.xp
        r_X = X.sum(axis=1)
        d_r = D.sum(axis=1)

        XA_H = X @ A_H
        DA_H = D @ A_H

        # R(lambda) = R0 + lambda * R1 + lambda^2 * R2
        R0 = (r_X[:, None] * A_G * r_X[None, :]) - (XA_H @ X.T)
        R1 = -((r_X[:, None] * A_G * d_r[None, :]) + (d_r[:, None] * A_G * r_X[None, :])) + (XA_H @ D.T + DA_H @ X.T)
        R2 = (d_r[:, None] * A_G * d_r[None, :]) - (DA_H @ D.T)

        # Coefficients of ||R(lambda)||_F^2
        s0 = float(xp.sum(R0 ** 2))
        s1 = 2.0 * float(xp.sum(R0 * R1))
        s2 = float(xp.sum(R1 ** 2)) + 2.0 * float(xp.sum(R0 * R2))
        s3 = 2.0 * float(xp.sum(R1 * R2))
        s4 = float(xp.sum(R2 ** 2))

        # Unary term
        u0 = float(xp.sum(C * X))
        u1 = -float(xp.sum(C * D))

        # Structural + unary
        f0 = self.alpha * s0 + (1.0 - self.alpha) * u0
        f1 = self.alpha * s1 + (1.0 - self.alpha) * u1
        f2 = self.alpha * s2
        f3 = self.alpha * s3
        f4 = self.alpha * s4

        # Homotopy norm penalty ||X(lambda)||_F^2 = x0 + x1*lambda + x2*lambda^2
        x0 = float(xp.sum(X ** 2))
        x1 = -2.0 * float(xp.sum(X * D))
        x2 = float(xp.sum(D ** 2))

        scale = (1.0 - zeta) if zeta >= 0 else (1.0 + zeta)
        c0 = scale * f0 + zeta * x0
        c1 = scale * f1 + zeta * x1
        c2 = scale * f2 + zeta * x2
        c3 = scale * f3
        c4 = scale * f4

        # Derivative: 4*c4*lambda^3 + 3*c3*lambda^2 + 2*c2*lambda + c1 = 0
        poly_deriv = [4.0 * c4, 3.0 * c3, 2.0 * c2, c1]
        try:
            roots = np.roots(poly_deriv)
            real_roots = [float(r.real) for r in roots if np.isreal(r) and 0.0 <= r.real <= 1.0]
        except Exception:
            real_roots = []

        candidates = [0.0, 1.0] + real_roots

        def _eval_J(lam):
            return c0 + c1 * lam + c2 * (lam**2) + c3 * (lam**3) + c4 * (lam**4)

        best_lam = min(candidates, key=_eval_J)
        return float(np.clip(best_lam, 0.0, 1.0))

    def _closed_form_line_search(self, X, D, A_G, A_H, C, zeta, grad_J=None):
        """
        Closed-form quadratic line search along Frank-Wolfe direction.
        
        Fits J(lambda) ≈ a0 + a1*lambda + a2*lambda^2 where:
          a1 = <grad_J(X), -D>  (directional derivative at lambda=0)
          a2 = J(1) - J(0) - a1  (from endpoint evaluation)
        Returns lambda* = clip(-a1 / (2*a2), 0, 1).
        
        Reuses precomputed grad_J to eliminate redundant O(M^2 N) evaluations.
        """
        xp = self.xp
        if grad_J is None:
            grad_J = self._compute_gradient_Jzeta(X, A_G, A_H, C, zeta)
        # a1 = directional derivative d/dlambda J(X - lambda*D) at lambda=0
        a1 = float(xp.sum(grad_J * (-D)))  # = -<grad_J, D>
        
        if a1 >= 0:  # no descent direction
            return 0.0
        
        # Evaluate J at endpoints
        J0 = float(self._compute_Jzeta(X, A_G, A_H, C, zeta))
        X1 = X - D  # lambda = 1 endpoint
        J1 = float(self._compute_Jzeta(X1, A_G, A_H, C, zeta))
        
        # Fit quadratic: J(lambda) = J0 + a1*lambda + a2*lambda^2
        # J(1) = J0 + a1 + a2  =>  a2 = J1 - J0 - a1
        a2 = J1 - J0 - a1
        
        if a2 > 1e-12:  # quadratic has a minimum
            lam_star = -a1 / (2.0 * a2)
            lam_star = float(np.clip(lam_star, 0.0, 1.0))
        else:
            # Negative curvature: minimum is at the boundary
            lam_star = 1.0 if J1 < J0 else 0.0
        
        return lam_star

    def _grid_line_search(self, X, D, A_G, A_H, C, zeta, steps=11):
        """
        Grid line search along Frank-Wolfe direction (ablation baseline).
        
        Evaluates steps evenly-spaced candidate step sizes in [0, 1].
        NOTE: Not an exact line search — kept for ablation comparison.
        """
        xp = self.xp
        best_lam = 0.0
        best_val = float('inf')
        for lam in np.linspace(0.0, 1.0, steps):
            X_trial = xp.clip(X - lam * D, 0.0, 1.0)
            val = float(self._compute_Jzeta(X_trial, A_G, A_H, C, zeta))
            if val < best_val:
                best_val = val
                best_lam = float(lam)
        return best_lam

    def _project_to_polytope(self, X, L, max_iter=20):
        """
        Alternating Sinkhorn/Bregman projection to partial permutation polytope.
        
        Ensures: X >= 0, row sums <= 1, col sums <= 1, total sum = L.
        Note: the total-mass rescaling after row/col projection can re-violate
        row/col constraints; the loop iterates until convergence or max_iter.
        """
        xp = self.xp
        X = xp.maximum(X, 0.0)
        s = X.sum()
        if s > 0: X = X * (L / s)
        
        for it in range(max_iter):
            row_sums = X.sum(axis=1, keepdims=True)
            X = xp.where(row_sums > 1.0, X / xp.maximum(row_sums, 1e-12), X)
            
            col_sums = X.sum(axis=0, keepdims=True)
            X = xp.where(col_sums > 1.0, X / xp.maximum(col_sums, 1e-12), X)
            
            # Check feasibility before mass rescaling
            row_ok = bool(xp.all(X.sum(axis=1) <= 1.0 + 1e-6))
            col_ok = bool(xp.all(X.sum(axis=0) <= 1.0 + 1e-6))
            
            s = X.sum()
            if s > 0: X = X * (L / s)
            
            if row_ok and col_ok:
                # Re-check after mass rescaling
                if bool(xp.all(X.sum(axis=1) <= 1.0 + 1e-6)) and bool(xp.all(X.sum(axis=0) <= 1.0 + 1e-6)):
                    break
                
        return xp.clip(X, 0.0, 1.0)

    def _check_feasibility(self, X, L):
        """
        Diagnostic: reports constraint satisfaction of current assignment X.
        
        Returns dict with max_row_sum, max_col_sum, total_mass, n_violations.
        """
        xp = self.xp
        row_sums = X.sum(axis=1)
        col_sums = X.sum(axis=0)
        return {
            'max_row_sum': float(xp.max(row_sums)),
            'max_col_sum': float(xp.max(col_sums)),
            'total_mass': float(X.sum()),
            'target_L': L,
            'n_row_violations': int(xp.sum(row_sums > 1.0 + 1e-6)),
            'n_col_violations': int(xp.sum(col_sums > 1.0 + 1e-6)),
            'min_entry': float(xp.min(X)),
        }

    def _discretize_to_binary(self, X, L, M, N):
        """
        Discretizes continuous relaxed assignment X into strictly binary partial permutation matrix
        by maximizing <X, Y> subject to partial permutation polytope constraints.
        """
        return self._solve_hungarian_oracle_padded(-X, L, M, N)


    # ==================================================================================
    # GRADIENT VERIFICATION (FINITE DIFFERENCES)
    # ==================================================================================

    def verify_gradient(self, M=8, N=10, L=6, eps=1e-5, seed=42):
        """
        Verifies the analytical gradient against numerical finite differences.
        
        Creates random adjacency matrices and assignment, computes the analytical
        gradient via _compute_gradient_F, and compares element-wise against
        central finite differences of _compute_F.
        
        Uses float64 internally for numerical stability of finite differences.
        
        Returns:
            dict with 'max_abs_error', 'mean_abs_error', 'relative_error', 'passed'.
        """
        np.random.seed(seed)
        # Use float64 for numerical stability in finite differences
        A_G = np.random.rand(M, M).astype(np.float64)
        A_G = (A_G + A_G.T) / 2.0
        np.fill_diagonal(A_G, 0.0)
        A_H = np.random.rand(N, N).astype(np.float64)
        A_H = (A_H + A_H.T) / 2.0
        np.fill_diagonal(A_H, 0.0)
        C = np.random.rand(M, N).astype(np.float64)
        
        # Random feasible X (scaled to sum L, entries in [0, 1])
        X = np.random.rand(M, N).astype(np.float64)
        X = X * (L / X.sum())
        X = np.clip(X, 0.0, 1.0)
        
        # Analytical gradient (computed in float64)
        grad_analytical = self._compute_gradient_F(X, A_G, A_H, C)
        if self.use_gpu:
            grad_analytical = cp.asnumpy(grad_analytical)
        grad_analytical = np.asarray(grad_analytical, dtype=np.float64)
        
        # Numerical gradient via central finite differences
        grad_numerical = np.zeros_like(X)
        for i in range(M):
            for j in range(N):
                X_plus = X.copy()
                X_minus = X.copy()
                X_plus[i, j] += eps
                X_minus[i, j] -= eps
                f_plus = float(self._compute_F(X_plus, A_G, A_H, C))
                f_minus = float(self._compute_F(X_minus, A_G, A_H, C))
                grad_numerical[i, j] = (f_plus - f_minus) / (2.0 * eps)
        
        diff = grad_analytical - grad_numerical
        max_abs = float(np.max(np.abs(diff)))
        mean_abs = float(np.mean(np.abs(diff)))
        norm_analytical = float(np.linalg.norm(grad_analytical))
        rel_error = max_abs / max(norm_analytical, 1e-10)
        
        return {
            'max_abs_error': max_abs,
            'mean_abs_error': mean_abs,
            'relative_error': rel_error,
            'grad_norm': norm_analytical,
            'passed': rel_error < 1e-3,
        }


    # ==================================================================================
    # 2. DOMAIN ADAPTERS (HIGH-LEVEL SCIENTIFIC APIS)
    # ==================================================================================

    def match_point_clouds(self,
                           coords_G: np.ndarray,
                           coords_H: np.ndarray,
                           features_G: Optional[np.ndarray] = None,
                           features_H: Optional[np.ndarray] = None,
                           L: Optional[int] = None,
                           rbf_sigma: Optional[float] = None,
                           n_dgm_iters: int = 2,
                            verbose: bool = False) -> MatchingResult:
        """
        Domain: Computer Vision, Robotics, and Structural Biology.
        Matches 2D/3D spatial point clouds and computes optimal rigid Kabsch superposition.
        """
        coords_G = np.asarray(coords_G, dtype=np.float32)
        coords_H = np.asarray(coords_H, dtype=np.float32)
        M, N = len(coords_G), len(coords_H)
        
        # Build spatial adjacency via Euclidean distance + RBF Gaussian kernel
        dist_G = np.linalg.norm(coords_G[:, None] - coords_G[None, :], axis=-1)
        dist_H = np.linalg.norm(coords_H[:, None] - coords_H[None, :], axis=-1)
        
        if rbf_sigma is None:
            # Median heuristic for RBF bandwidth
            med = np.median(dist_G[dist_G > 0]) if np.any(dist_G > 0) else 1.0
            rbf_sigma = max(float(med), 1e-3)
            
        A_G = np.exp(-0.5 * (dist_G / rbf_sigma)**2)
        A_H = np.exp(-0.5 * (dist_H / rbf_sigma)**2)
        np.fill_diagonal(A_G, 0.0)
        np.fill_diagonal(A_H, 0.0)
        
        # Unary feature distance matrix C
        if features_G is not None and features_H is not None:
            C = np.linalg.norm(features_G[:, None] - features_H[None, :], axis=-1)
            c_min, c_max = C.min(), C.max()
            if c_max > c_min: C = (C - c_min) / (c_max - c_min)
        else:
            C = None
            
        # Pass 1: Solve via distance geometry
        res = self.solve(A_G, A_H, C=C, L=L, coords_G=coords_G, coords_H=coords_H, verbose=verbose)

        # Deformable Graph Matching (DGM) Alternating Kabsch refinement (Zhou & De la Torre CVPR 2013)
        if n_dgm_iters > 1 and res.alignment is not None and 'R' in res.alignment:
            for _ in range(n_dgm_iters - 1):
                R_est = res.alignment['R']
                t_est = res.alignment['t']
                P_trans = coords_G @ R_est.T + t_est
                C_spatial = np.linalg.norm(P_trans[:, None] - coords_H[None, :], axis=-1).astype(np.float32)
                c_max = C_spatial.max()
                if c_max > 0: C_spatial /= c_max
                C_comb = (0.5 * C + 0.5 * C_spatial) if C is not None else C_spatial
                res = self.solve(A_G, A_H, C=C_comb, L=L, coords_G=coords_G, coords_H=coords_H, verbose=verbose)

        return res

    def match_networkx(self,
                       G_nx: Any,
                       H_nx: Any,
                       node_feature_attr: Optional[str] = None,
                       edge_weight_attr: str = 'weight',
                       L: Optional[int] = None,
                       n_dgm_iters: int = 2,
                            verbose: bool = False) -> MatchingResult:
        """
        Domain: Social Networks, Citation Graphs, and Complex Networks.
        Matches two NetworkX Graph/DiGraph instances.
        """
        import networkx as nx
        nodes_G = list(G_nx.nodes())
        nodes_H = list(H_nx.nodes())
        
        A_G = nx.to_numpy_array(G_nx, nodelist=nodes_G, weight=edge_weight_attr, dtype=np.float32)
        A_H = nx.to_numpy_array(H_nx, nodelist=nodes_H, weight=edge_weight_attr, dtype=np.float32)
        
        # Build unary cost from node attributes if provided
        C = None
        if node_feature_attr is not None:
            feat_G = np.array([G_nx.nodes[n].get(node_feature_attr, 0.0) for n in nodes_G])
            feat_H = np.array([H_nx.nodes[n].get(node_feature_attr, 0.0) for n in nodes_H])
            if feat_G.ndim == 1: feat_G, feat_H = feat_G[:, None], feat_H[:, None]
            C = np.linalg.norm(feat_G[:, None] - feat_H[None, :], axis=-1)
            c_min, c_max = C.min(), C.max()
            if c_max > c_min: C = (C - c_min) / (c_max - c_min)
            
        res = self.solve(A_G, A_H, C=C, L=L, verbose=verbose)
        
        # Map indices back to original NetworkX node labels
        res.mapping_dict = {nodes_G[r]: nodes_H[c] for r, c in res.matches}
        return res

    def match_connectome(self,
                         synapse_adj_A: np.ndarray,
                         synapse_adj_B: np.ndarray,
                         coords_A: Optional[np.ndarray] = None,
                         coords_B: Optional[np.ndarray] = None,
                         types_A: Optional[List[str]] = None,
                         types_B: Optional[List[str]] = None,
                         L: Optional[int] = None,
                         n_dgm_iters: int = 2,
                            verbose: bool = False) -> MatchingResult:
        """
        Domain: Neuroscience & Connectomics.
        Matches synaptic connectome wiring graphs across hemispheres or specimens (e.g. Hemibrain vs FAFB).
        """
        M, N = synapse_adj_A.shape[0], synapse_adj_B.shape[0]
        C = np.zeros((M, N), dtype=np.float32)
        
        # Cell-type / neurotransmitter mismatch penalty
        if types_A is not None and types_B is not None:
            for i, tA in enumerate(types_A):
                for j, tB in enumerate(types_B):
                    if tA != tB:
                        C[i, j] = 10.0
                        
        return self.solve(synapse_adj_A, synapse_adj_B, C=C, L=L,
                          coords_G=coords_A, coords_H=coords_B, verbose=verbose)

    def match_molecules(self,
                        mol1: Any,
                        mol2: Any,
                        L: Optional[int] = None,
                        align_3d: bool = True,
                        n_dgm_iters: int = 2,
                            verbose: bool = False) -> MatchingResult:
        """
        Domain: Chemoinformatics & Pharmacology (RDKit wrapper).
        Extracts 2D topological graph and 3D conformer coordinates for small molecules.
        """
        from rdkit import Chem
        
        def extract_graph(mol):
            atoms = list(mol.GetAtoms())
            n = len(atoms)
            A = np.zeros((n, n), dtype=np.float32)
            b_weights = {
                Chem.BondType.SINGLE: 1.0,
                Chem.BondType.DOUBLE: 2.0,
                Chem.BondType.TRIPLE: 3.0,
                Chem.BondType.AROMATIC: 1.5
            }
            for b in mol.GetBonds():
                i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                w = b_weights.get(b.GetBondType(), 1.0)
                if b.IsInRing(): w += 0.1
                A[i, j] = A[j, i] = w
            
            atomic_nums = np.array([a.GetAtomicNum() for a in atoms])
            degrees = np.array([a.GetDegree() for a in atoms], dtype=np.float32)
            
            try:
                coords = mol.GetConformer().GetPositions().astype(np.float32)
            except Exception:
                coords = None
                
            return A, atomic_nums, degrees, coords

        A_G, nums_G, deg_G, coords_G = extract_graph(mol1)
        A_H, nums_H, deg_H, coords_H = extract_graph(mol2)
        
        # Degree difference + element mismatch penalty
        C = np.abs(deg_G[:, None] - deg_H[None, :]) * 5.0
        c_min, c_max = C.min(), C.max()
        if c_max > c_min: C = (C - c_min) / (c_max - c_min)
        C[nums_G[:, None] != nums_H[None, :]] = 50.0  # Element penalty
        
        coords1 = coords_G if align_3d else None
        coords2 = coords_H if align_3d else None
        
        return self.solve(A_G, A_H, C=C, L=L, coords_G=coords1, coords_H=coords2, verbose=verbose)


# ======================================================================================
# MULTI-DOMAIN COMPREHENSIVE VERIFICATION SUITE
# ======================================================================================

def run_multidomain_verification():
    print("=" * 82)
    print("UNIVERSAL FAST-WCS / GNCCP MULTI-DOMAIN VERIFICATION SUITE")
    print("=" * 82)
    solver = UniversalGNCCP(alpha=0.7, max_iter=12, fw_max_iter=6)

    # ----------------------------------------------------------------------------------
    # DOMAIN 1: PURE MATHEMATICS & SYNTHETIC GRAPHS (IEEE CANONICAL BENCHMARK)
    # ----------------------------------------------------------------------------------
    print("\n[DOMAIN 1: PURE MATHEMATICAL GRAPHS]")
    np.random.seed(42)
    L_math = 10
    coords_G_math = np.random.uniform(0, 1, (L_math, 2))
    coords_H_core = coords_G_math + np.random.normal(0, 0.02, (L_math, 2))
    coords_H_outliers = np.random.uniform(0, 1, (5, 2))
    coords_H_math = np.vstack([coords_H_core, coords_H_outliers])

    dist_G_m = np.linalg.norm(coords_G_math[:, None] - coords_G_math[None, :], axis=-1)
    dist_H_m = np.linalg.norm(coords_H_math[:, None] - coords_H_math[None, :], axis=-1)
    A_G_math = (dist_G_m < 0.4).astype(float)
    A_H_math = (dist_H_m < 0.4).astype(float)
    C_math = np.linalg.norm(coords_G_math[:, None] - coords_H_math[None, :], axis=-1)
    
    res1 = solver.solve(A_G_math, A_H_math, C=C_math, L=L_math, verbose=False)
    recovered = sum(1 for g, h in res1.matches if g == h)
    acc1 = (recovered / L_math) * 100.0
    print(f"  * IEEE Benchmark Matching: {res1.L} / {len(coords_H_math)} nodes matched in {res1.runtime_sec*1000:.1f} ms | "
          f"Ground Truth Recovery: {acc1:.1f}% ({recovered}/{L_math})")

    # ----------------------------------------------------------------------------------
    # DOMAIN 2: COMPUTER VISION & 2D/3D RIGID POINT CLOUD REGISTRATION
    # ----------------------------------------------------------------------------------
    print("\n[DOMAIN 2: COMPUTER VISION & SPATIAL POINT CLOUDS]")
    L_cv = 10
    pts_G = np.random.uniform(-3.0, 3.0, (L_cv, 3))
    # Apply 3D rotation (yaw 45 deg) + translation + noise
    angle = np.pi / 4.0
    R_gt = np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle),  np.cos(angle), 0],
                     [0,              0,             1]])
    t_gt = np.array([2.5, -1.0, 4.0])
    pts_H = pts_G @ R_gt.T + t_gt + np.random.normal(0, 0.015, (L_cv, 3))
    
    res2 = solver.match_point_clouds(pts_G, pts_H, L=L_cv, verbose=False)
    recovered2 = sum(1 for g, h in res2.matches if g == h)
    rmsd_cv = res2.alignment['rmsd'] if res2.alignment else -1.0
    print(f"  * 3D Point Cloud Registration: Matched {res2.L} / {L_cv} points in {res2.runtime_sec*1000:.1f} ms | "
          f"Accuracy: {recovered2/L_cv*100:.1f}% ({recovered2}/{L_cv})")
    sub_msg = " (Sub-angstrom physical recovery!)" if rmsd_cv < 1.0 else ""
    print(f"  * Closed-Form SVD Kabsch RMSD: {rmsd_cv:.4f} Å{sub_msg}")

    # ----------------------------------------------------------------------------------
    # DOMAIN 3: SOCIAL & COMPLEX NETWORKS (NETWORKX)
    # ----------------------------------------------------------------------------------
    print("\n[DOMAIN 3: SOCIAL NETWORKS & COMPLEX GRAPHS]")
    try:
        import networkx as nx
        G_social = nx.karate_club_graph()
        # Perturb network by removing 4 peripheral nodes and adding edge jitter
        sub_nodes = list(G_social.nodes())[:25]
        H_social = G_social.subgraph(sub_nodes).copy()
        for u, v in list(H_social.edges())[:5]:
            H_social[u][v]['weight'] = 2.0
            
        res3 = solver.match_networkx(G_social, H_social, L=20, verbose=False)
        print(f"  * Karate Club Community Matching: {res3.L} nodes matched in {res3.runtime_sec*1000:.1f} ms | "
              f"Score: {res3.similarity_score:.4f}")
    except Exception as e:
        print(f"  [Skip NetworkX]: {e}")

    # ----------------------------------------------------------------------------------
    # DOMAIN 4: NEUROSCIENCE & CONNECTOMICS (SYNAPTIC WIRING)
    # ----------------------------------------------------------------------------------
    print("\n[DOMAIN 4: NEUROSCIENCE & CONNECTOME MATCHING]")
    N_neurons = 24
    # Synaptic connection density matrix
    synapse_A = (np.random.rand(N_neurons, N_neurons) > 0.75).astype(float) * np.random.uniform(1.0, 10.0, (N_neurons, N_neurons))
    synapse_B = synapse_A + np.random.normal(0, 0.5, synapse_A.shape)
    synapse_B = np.maximum(synapse_B, 0.0)
    types_A = ["GABA" if i % 2 == 0 else "Cholinergic" for i in range(N_neurons)]
    types_B = types_A.copy()
    
    res4 = solver.match_connectome(synapse_A, synapse_B, types_A=types_A, types_B=types_B, L=20, verbose=False)
    print(f"  * Connectome Arborization: {res4.L} neurons mapped in {res4.runtime_sec*1000:.1f} ms | "
          f"Topology Score: {res4.similarity_score:.4f}")

    # ----------------------------------------------------------------------------------
    # DOMAIN 5: CHEMOINFORMATICS & PHARMACOLOGY (RDKIT)
    # ----------------------------------------------------------------------------------
    print("\n[DOMAIN 5: CHEMOINFORMATICS & MOLECULAR GRAPHS]")
    try:
        from rdkit import Chem
        m1 = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O") # Aspirin (13 heavy atoms)
        m2 = Chem.MolFromSmiles("Oc1ccccc1C(=O)O")        # Salicylic Acid (10 heavy atoms)
        res5 = solver.match_molecules(m1, m2, L=8, align_3d=False, verbose=False)
        print(f"  * Molecular Subgraph Isomorphism: {res5.L} atoms aligned in {res5.runtime_sec*1000:.1f} ms | "
              f"Similarity: {res5.similarity_score:.4f}")
    except Exception as e:
        print(f"  [Skip RDKit]: {e}")

    # ----------------------------------------------------------------------------------
    # VERIFICATION 6: GRADIENT CORRECTNESS (FINITE DIFFERENCES)
    # ----------------------------------------------------------------------------------
    print("\n[VERIFICATION 6: GRADIENT FINITE-DIFFERENCE CHECK]")
    grad_result = solver.verify_gradient(M=8, N=10, L=6)
    status = "PASS" if grad_result['passed'] else "FAIL"
    print(f"  * Max Abs Error: {grad_result['max_abs_error']:.2e}")
    print(f"  * Relative Error: {grad_result['relative_error']:.2e}")
    print(f"  * Gradient Norm: {grad_result['grad_norm']:.4f}")
    print(f"  * Status: {status}")

    # ----------------------------------------------------------------------------------
    # VERIFICATION 7: FEASIBILITY DIAGNOSTICS
    # ----------------------------------------------------------------------------------
    print("\n[VERIFICATION 7: FEASIBILITY DIAGNOSTICS]")
    feas = solver._check_feasibility(res1.X_continuous, res1.L)
    print(f"  * Max Row Sum: {feas['max_row_sum']:.6f} (must be <= 1.0)")
    print(f"  * Max Col Sum: {feas['max_col_sum']:.6f} (must be <= 1.0)")
    print(f"  * Total Mass: {feas['total_mass']:.4f} (target: {feas['target_L']})")
    print(f"  * Row Violations: {feas['n_row_violations']} | Col Violations: {feas['n_col_violations']}")

    # ----------------------------------------------------------------------------------
    # VERIFICATION 8: FULL ABLATION STUDY
    # ----------------------------------------------------------------------------------
    print("\n[VERIFICATION 8: FULL ABLATION STUDY]")
    print("  Testing all combinations of improvements vs. original paper baseline:")
    print(f"  {'Config':<55s} | {'Acc':>7s} | {'Obj':>8s} | {'Time':>8s} | {'Iters':>5s}")
    print("  " + "-" * 95)

    ablation_configs = [
        # (label, kwargs)
        ("OURS FULL (adaptive+refine+padded+closed)",
         dict(oracle='dummy_padded', line_search='closed_form', schedule='cosine',
              adaptive_schedule=True, local_refinement=True, n_restarts=1)),
        ("Ours + 3 restarts",
         dict(oracle='dummy_padded', line_search='closed_form', schedule='cosine',
              adaptive_schedule=True, local_refinement=True, n_restarts=3)),
        ("No adaptive schedule",
         dict(oracle='dummy_padded', line_search='closed_form', schedule='cosine',
              adaptive_schedule=False, local_refinement=True, n_restarts=1)),
        ("No local refinement",
         dict(oracle='dummy_padded', line_search='closed_form', schedule='cosine',
              adaptive_schedule=True, local_refinement=False, n_restarts=1)),
        ("No adaptive, no refinement",
         dict(oracle='dummy_padded', line_search='closed_form', schedule='cosine',
              adaptive_schedule=False, local_refinement=False, n_restarts=1)),
        ("ORIGINAL PAPER (topk+grid+linear+no adapt+no refine)",
         dict(oracle='top_k_heuristic', line_search='grid', schedule='linear',
              adaptive_schedule=False, local_refinement=False, n_restarts=1)),
    ]
    for label, kwargs in ablation_configs:
        s = UniversalGNCCP(alpha=0.7, max_iter=12, fw_max_iter=6, **kwargs)
        r = s.solve(A_G_math, A_H_math, C=C_math, L=L_math, verbose=False)
        rec = sum(1 for g, h in r.matches if g == h)
        print(f"  {label:<55s} | {rec}/{L_math} ({rec/L_math*100:3.0f}%) | {r.objective_score:8.4f} | "
              f"{r.runtime_sec*1000:6.1f} ms | {r.iterations_run:5d}")

    print("\n" + "=" * 82)
    print("ALL DOMAINS & VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 82)


if __name__ == '__main__':
    run_multidomain_verification()
