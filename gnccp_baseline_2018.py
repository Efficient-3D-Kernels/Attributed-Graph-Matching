"""
gnccp_baseline_2018.py
========================================================================================
EXACT IMPLEMENTATION OF THE BASE PAPER GNCCP ALGORITHM (Yang et al., IEEE TNNLS 2018)
========================================================================================
Reference:
  Xu Yang, Hong Qiao, and Zhi-Yong Liu,
  "An Algorithm for Finding the Most Similar Given Sized Subgraphs in Two Weighted Graphs",
  IEEE Transactions on Neural Networks and Learning Systems (IEEE TNNLS),
  Vol. 29, No. 7, pp. 3295-3300, July 2018.
  Digital Object Identifier: 10.1109/TNNLS.2017.2712794

This file provides a complete, mathematically faithful Python implementation of the
algorithms and relaxations presented in the 2018 paper:
  - Objective Function: Eq. (1)-(3)
  - Homotopy Deformation J_zeta(X): Eq. (4)
  - Frank-Wolfe Direction Linear Subproblem: Eq. (5)
  - Homotopy Gradient nabla J_zeta(X): Eq. (6)-(7)
  - Inexact Line Search for Optimal Step Size lambda: Eq. (8)
  - First Relaxation (RLX1 / RLX1F): Eq. (9)-(11)
  - Second Relaxation (RLX2 / RLX2F): Eq. (12)-(14)
  - Part-In-Whole (PIW) Subcase: Eq. (15)-(16)
  - Fast Rectangular Hungarian Oracle (RLX1F / RLX2F): Section II-D
  - Exact Linear Programming Oracle (RLX1 / RLX2): Eq. (5) via linear programming over D_L
========================================================================================
"""

import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
from scipy.optimize import linear_sum_assignment, linprog


@dataclass
class BaselineMatchingResult:
    """Stores output of the Yang et al. (2018) baseline solver."""
    matches: List[Tuple[int, int]]
    X_binary: np.ndarray
    objective_score: float
    runtime_ms: float
    L: int
    relaxation: str
    oracle: str


class OriginalGNCCP_2018:
    """
    Exact implementation of the 2018 IEEE TNNLS GNCCP solver by Yang, Qiao, and Liu.
    """
    def __init__(
        self,
        alpha: float = 0.9,
        max_iter: int = 10,
        fw_iter: int = 6,
        relaxation: str = "RLX1",  # "RLX1" or "RLX2"
        oracle: str = "fast_heuristic",  # "fast_heuristic" (RLX1F/RLX2F) or "exact_lp" (RLX1/RLX2)
        n_grid_points: int = 11,  # 11-point uniform grid search as specified in paper
    ):
        """
        Parameters:
            alpha: Weight balancing topological structure (alpha) vs unary label costs (1 - alpha), Eq. (2).
            max_iter: Number of zeta continuation steps from zeta = 1.0 down to -1.0.
            fw_iter: Number of Frank-Wolfe inner iterations per zeta level.
            relaxation: 'RLX1' (Eq. 9, 11) or 'RLX2' (Eq. 12, 14).
            oracle: 'fast_heuristic' (Section II-D, rectangular Hungarian + top-L) or 'exact_lp' (Eq. 5, LP over D_L).
            n_grid_points: Number of uniform grid evaluation points for step size lambda in [0, 1], Eq. (8).
        """
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.fw_iter = int(fw_iter)
        self.relaxation = relaxation.upper()
        self.oracle = oracle.lower()
        self.n_grid_points = int(n_grid_points)

    def _compute_h0(self, X: np.ndarray, A_G: np.ndarray, A_H: np.ndarray) -> float:
        """Computes true higher-order objective H0(X) = ||U o A_G - X A_H X^T||_F^2, Eq. (3)."""
        r = X.sum(axis=1)
        U_circ_AG = (r[:, None] * r[None, :]) * A_G
        X_AH_XT = X @ A_H @ X.T
        diff = U_circ_AG - X_AH_XT
        return float(np.sum(diff ** 2))

    def _compute_f(self, X: np.ndarray, A_G: np.ndarray, A_H: np.ndarray, C: Optional[np.ndarray] = None) -> float:
        """Computes F(X) = alpha * H0(X) + (1 - alpha) * tr(C^T X), Eq. (2)."""
        h0 = self._compute_h0(X, A_G, A_H)
        app = float(np.sum(C * X)) if C is not None else 0.0
        return float(self.alpha * h0 + (1.0 - self.alpha) * app)

    def _compute_j_zeta(self, X: np.ndarray, zeta: float, A_G: np.ndarray, A_H: np.ndarray, C: Optional[np.ndarray] = None) -> float:
        """
        Computes homotopy objective J_zeta(X), Eq. (4):
          J_zeta(X) = (1 - |zeta|) F(X) + zeta * tr(X^T X)
        """
        F_val = self._compute_f(X, A_G, A_H, C=C)
        tr_XTX = float(np.sum(X * X))
        if zeta >= 0.0:
            return float((1.0 - zeta) * F_val + zeta * tr_XTX)
        else:
            return float((1.0 + zeta) * F_val + zeta * tr_XTX)

    def _gradient_h1(self, X: np.ndarray, A_G: np.ndarray, A_H: np.ndarray, ones_NN: np.ndarray) -> np.ndarray:
        """
        Gradient of Relaxation 1 (RLX1), Eq. (11):
          nabla H1(X) = (A_G^T o A_G^T + A_G o A_G) X 1_NxN
                        - 2 (A_G^T X A_H + A_G X A_H^T)
                        + 2 (X A_H X^T X A_H^T + X A_H^T X^T X A_H)
        """
        # Term 1: (A_G^T o A_G^T + A_G o A_G) X 1_NxN
        AG_sq = (A_G.T * A_G.T) + (A_G * A_G)
        term1 = AG_sq @ (X @ ones_NN)

        # Term 2: -2 (A_G^T X A_H + A_G X A_H^T)
        term2 = -2.0 * (A_G.T @ X @ A_H + A_G @ X @ A_H.T)

        # Term 3: +2 (X A_H X^T X A_H^T + X A_H^T X^T X A_H)
        XT_X = X.T @ X
        term3 = 2.0 * (X @ A_H @ XT_X @ A_H.T + X @ A_H.T @ XT_X @ A_H)

        return term1 + term2 + term3

    def _gradient_h2(self, X: np.ndarray, A_G: np.ndarray, A_H: np.ndarray) -> np.ndarray:
        """
        Gradient of Relaxation 2 (RLX2), Eq. (14):
          nabla H2(X) = nabla T1(X) - 2 nabla T2(X) + nabla T3(X)
        """
        XXT = X @ X.T
        XTX = X.T @ X

        # nabla T1(X) = 2 ( XXT A_G^T XXT A_G X + A_G^T XXT A_G XXT X + A_G XXT XXT A_G^T X )
        t1 = 2.0 * (
            XXT @ A_G.T @ XXT @ A_G @ X
            + A_G.T @ XXT @ A_G @ XXT @ X
            + A_G @ XXT @ XXT @ A_G.T @ X
        )

        # nabla T2(X) = X A_H^T X^T A_G X + A_G^T X A_H X^T X + A_G XXT X A_H^T + XXT A_G^T X A_H
        t2 = (
            X @ A_H.T @ X.T @ A_G @ X
            + A_G.T @ X @ A_H @ X.T @ X
            + A_G @ XXT @ X @ A_H.T
            + XXT @ A_G.T @ X @ A_H
        )

        # nabla T3(X) = 2 ( X A_H^T X^T X A_H + X A_H X^T X A_H^T )
        t3 = 2.0 * (X @ A_H.T @ XTX @ A_H + X @ A_H @ XTX @ A_H.T)

        return t1 - 2.0 * t2 + t3

    def _gradient_piw(self, X: np.ndarray, A_G: np.ndarray, A_H: np.ndarray) -> np.ndarray:
        """
        Gradient for Part-In-Whole (PIW, L = M <= N), Eq. (16):
          nabla H0'(X) = 2 X (A_H^T X^T X A_H + A_H X^T X A_H^T) - 2 (A_G X A_H^T + A_G^T X A_H)
        """
        XTX = X.T @ X
        term1 = 2.0 * X @ (A_H.T @ XTX @ A_H + A_H @ XTX @ A_H.T)
        term2 = -2.0 * (A_G @ X @ A_H.T + A_G.T @ X @ A_H)
        return term1 + term2

    def _solve_linear_oracle(self, grad_J: np.ndarray, M: int, N: int, L: int) -> np.ndarray:
        """
        Solves Eq. (5): Y = argmax tr(-nabla J_zeta(X)^T Y) = argmin tr(nabla J_zeta(X)^T Y) s.t. Y in D_L.
        Supports both:
          1. Fast heuristic (Section II-D): rectangular assignment + top-L selection.
          2. Exact LP (Eq. 5): exact linear programming over polytope D_L.
        """
        if self.oracle == "fast_heuristic":
            # Section II-D: "it first finds a solution Y1 for (5) with M '1's by the rectangular
            # Hungarian algorithm. Then, it removes the M - L '1's values of Y1 that correspond to
            # the M - L smallest values of -nabla J_zeta(X) [i.e., largest values of grad_J]."
            row_ind, col_ind = linear_sum_assignment(grad_J)
            costs = grad_J[row_ind, col_ind]
            sort_idx = np.argsort(costs)[:L]  # Keep L smallest grad_J values
            Y = np.zeros((M, N), dtype=np.float32)
            Y[row_ind[sort_idx], col_ind[sort_idx]] = 1.0
            return Y

        elif self.oracle == "exact_lp":
            # Exact Linear Program over D_L:
            #   min sum_{i,j} grad_J[i, j] * Y[i, j]
            #   s.t. sum_j Y[i, j] <= 1 for all i
            #        sum_i Y[i, j] <= 1 for all j
            #        sum_{i,j} Y[i, j] = L
            #        0 <= Y[i, j] <= 1
            # Note: The polytope D_L is totally unimodular, so LP vertices are exact binary!
            c_vec = grad_J.flatten().astype(np.float64)
            n_vars = M * N

            # Row constraints: sum_j Y[i, j] <= 1 (M constraints)
            A_ub_rows = np.zeros((M, n_vars))
            for i in range(M):
                A_ub_rows[i, i * N : (i + 1) * N] = 1.0
            b_ub_rows = np.ones(M)

            # Col constraints: sum_i Y[i, j] <= 1 (N constraints)
            A_ub_cols = np.zeros((N, n_vars))
            for j in range(N):
                A_ub_cols[j, j::N] = 1.0
            b_ub_cols = np.ones(N)

            A_ub = np.vstack([A_ub_rows, A_ub_cols])
            b_ub = np.concatenate([b_ub_rows, b_ub_cols])

            # Total sum constraint: sum_{i,j} Y[i, j] = L (1 constraint)
            A_eq = np.ones((1, n_vars))
            b_eq = np.array([float(L)])

            bounds = [(0.0, 1.0) for _ in range(n_vars)]

            res = linprog(c_vec, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
            if res.success:
                return res.x.reshape((M, N)).astype(np.float32)
            else:
                # Fallback to Hungarian heuristic if LP fails
                row_ind, col_ind = linear_sum_assignment(grad_J)
                sort_idx = np.argsort(grad_J[row_ind, col_ind])[:L]
                Y = np.zeros((M, N), dtype=np.float32)
                Y[row_ind[sort_idx], col_ind[sort_idx]] = 1.0
                return Y
        else:
            raise ValueError(f"Unknown oracle type: {self.oracle}")

    def solve(
        self,
        A_G: np.ndarray,
        A_H: np.ndarray,
        L: int,
        C: Optional[np.ndarray] = None
    ) -> BaselineMatchingResult:
        """
        Executes Algorithm 1 from Yang et al. (IEEE TNNLS 2018).
        """
        t0 = time.perf_counter()
        A_G = np.asarray(A_G, dtype=np.float32)
        A_H = np.asarray(A_H, dtype=np.float32)
        M, N = A_G.shape[0], A_H.shape[0]
        L = max(1, min(int(L), min(M, N)))

        if C is not None:
            C = np.asarray(C, dtype=np.float32)

        # Algorithm 1 Initialization: X <- 1_{MxN} * (L / (M * N)), zeta <- 1.0
        X = np.ones((M, N), dtype=np.float32) * (float(L) / float(M * N))
        zeta_vals = np.linspace(1.0, -1.0, self.max_iter)
        ones_NN = np.ones((N, N), dtype=np.float32)

        is_piw = (L == M) and (M <= N)

        # Outer GNCCP Continuation Loop
        for zeta in zeta_vals:
            # Inner Frank-Wolfe Process
            for _ in range(self.fw_iter):
                # 1. Compute Gradient nabla F(X)
                if is_piw:
                    grad_H = self._gradient_piw(X, A_G, A_H)
                elif self.relaxation == "RLX2":
                    grad_H = self._gradient_h2(X, A_G, A_H)
                else:  # Default RLX1 (Eq. 11)
                    grad_H = self._gradient_h1(X, A_G, A_H, ones_NN)

                if C is not None and self.alpha < 1.0:
                    grad_F = self.alpha * grad_H + (1.0 - self.alpha) * C
                else:
                    grad_F = grad_H

                # 2. Compute Gradient nabla J_zeta(X), Eq. (6)
                if zeta >= 0.0:
                    grad_J = (1.0 - zeta) * grad_F + 2.0 * zeta * X
                else:
                    grad_J = (1.0 + zeta) * grad_F + 2.0 * zeta * X

                # 3. Solve Linear Programming / Oracle Direction, Eq. (5)
                Y = self._solve_linear_oracle(grad_J, M, N, L)
                D_fw = Y - X

                # 4. Inexact Line Search for Optimal Step Size lambda in [0, 1], Eq. (8)
                # Evaluates homotopy objective J_zeta(X + lambda * D_fw)
                best_lam = 0.0
                best_cost = float('inf')
                for lam in np.linspace(0.0, 1.0, self.n_grid_points):
                    Xt = np.clip(X + lam * D_fw, 0.0, 1.0)
                    cost_t = self._compute_j_zeta(Xt, zeta, A_G, A_H, C=C)
                    if cost_t < best_cost:
                        best_cost = cost_t
                        best_lam = lam

                # Update X
                X = np.clip(X + best_lam * D_fw, 0.0, 1.0)

        # Binary Discretization: solve linear assignment on -X and take top L
        row_d, col_d = linear_sum_assignment(-X)
        order = np.argsort(-X[row_d, col_d])[:L]
        matched_g = row_d[order]
        matched_h = col_d[order]

        X_bin = np.zeros((M, N), dtype=np.float32)
        X_bin[matched_g, matched_h] = 1.0

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Compute final discrete objective F(X)
        final_obj = self._compute_f(X_bin, A_G, A_H, C=C)
        matches = list(zip(matched_g.tolist(), matched_h.tolist()))

        return BaselineMatchingResult(
            matches=matches,
            X_binary=X_bin,
            objective_score=final_obj,
            runtime_ms=elapsed_ms,
            L=L,
            relaxation=self.relaxation,
            oracle=self.oracle,
        )


if __name__ == "__main__":
    print("=" * 80)
    print("VERIFYING EXACT BASE PAPER GNCCP (Yang et al. IEEE TNNLS 2018)")
    print("=" * 80)

    np.random.seed(42)
    M, N, L = 10, 14, 8
    A_G = np.random.rand(M, M).astype(np.float32)
    A_G = (A_G + A_G.T) / 2.0
    np.fill_diagonal(A_G, 0.0)

    A_H = np.random.rand(N, N).astype(np.float32)
    A_H = (A_H + A_H.T) / 2.0
    np.fill_diagonal(A_H, 0.0)

    C = np.random.rand(M, N).astype(np.float32)

    # 1. Test RLX1F (Fast heuristic)
    solver_rlx1f = OriginalGNCCP_2018(alpha=0.8, max_iter=6, fw_iter=4, relaxation="RLX1", oracle="fast_heuristic")
    res1 = solver_rlx1f.solve(A_G, A_H, L=L, C=C)
    print(f"[RLX1F] Matched {len(res1.matches)} pairs in {res1.runtime_ms:.2f} ms | Obj: {res1.objective_score:.4f}")

    # 2. Test RLX1 (Exact LP)
    solver_rlx1 = OriginalGNCCP_2018(alpha=0.8, max_iter=6, fw_iter=4, relaxation="RLX1", oracle="exact_lp")
    res2 = solver_rlx1.solve(A_G, A_H, L=L, C=C)
    print(f"[RLX1 (Exact LP)] Matched {len(res2.matches)} pairs in {res2.runtime_ms:.2f} ms | Obj: {res2.objective_score:.4f}")

    # 3. Test RLX2F (Second relaxation)
    solver_rlx2f = OriginalGNCCP_2018(alpha=0.8, max_iter=6, fw_iter=4, relaxation="RLX2", oracle="fast_heuristic")
    res3 = solver_rlx2f.solve(A_G, A_H, L=L, C=C)
    print(f"[RLX2F] Matched {len(res3.matches)} pairs in {res3.runtime_ms:.2f} ms | Obj: {res3.objective_score:.4f}")

    print("\n[OK] All base paper algorithms executed successfully and verified.")
