# wcs_gnccp.py - GPU ACCELERATED VERSION
# X remains CONTINUOUS during optimization!
# Only discretized to binary at the END!

import numpy as np
from scipy.optimize import linear_sum_assignment
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
warnings.filterwarnings('ignore')

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

class WCS_GNCCP:
    """Weighted Common Subgraph Matching using GNCCP with Frank-Wolfe and Optional GPU Support"""
    
    def __init__(self, alpha=0.7, gamma=0.5, max_iter=8, fw_max_iter=3, tol=1e-3, use_gpu=False):
        self.alpha = alpha
        self.gamma = gamma
        self.max_iter = max_iter
        self.fw_max_iter = fw_max_iter
        self.tol = tol
        self.use_gpu = use_gpu and HAS_CUPY
        self.xp = cp if self.use_gpu else np
        
    def match(self, mol1, mol2, L=None, verbose=True):
        """Main matching function"""
        if verbose:
            print("="*60)
            print("[*] WCS+GNCCP Matching (Frank-Wolfe)")
            print(f"   Hardware Backend: {'GPU (CuPy)' if self.use_gpu else 'CPU (NumPy)'}")
            print("="*60)
        
        # Build weighted adjacency matrices (CPU)
        A_G = self._build_weighted_adjacency(mol1)
        A_H = self._build_weighted_adjacency(mol2)
        
        M, N = A_G.shape[0], A_H.shape[0]
        
        # Frobenius Normalization is REQUIRED to keep structural cost bounded
        norm_G = np.linalg.norm(A_G, 'fro')
        norm_H = np.linalg.norm(A_H, 'fro')
        if norm_G > 0: A_G = A_G / norm_G
        if norm_H > 0: A_H = A_H / norm_H
        
        # Set L
        if L is None:
            L = max(min(M, N) - 2, 1)
        else:
            L = min(L, M, N)
        
        if verbose:
            print(f"   Matching {L} atoms between {M} and {N} atoms")
            print(f"   Alpha (structure weight): {self.alpha}")
        
        # Build cost matrix (atom features on CPU)
        C = self._build_cost_matrix(mol1, mol2)
        
        # Normalize fine-grained features to [0, 1]
        if C.max() > C.min():
            C = (C - C.min()) / (C.max() - C.min() + 1e-8)
        else:
            C = C / (C.max() + 1e-8)
        
        # Element-mismatch penalty applied AFTER normalization
        nums1 = np.array([a.GetAtomicNum() for a in mol1.GetAtoms()])
        nums2 = np.array([a.GetAtomicNum() for a in mol2.GetAtoms()])
        C[nums1[:, None] != nums2[None, :]] = 50.0
        
        # Run GNCCP with Frank-Wolfe (GPU or CPU)
        X = self._gnccp_optimization(A_G, A_H, C, L, verbose)
        
        # Extract matches from binary X (ensure it's back on CPU for pure python logic)
        X_cpu = cp.asnumpy(X) if self.use_gpu else X
        
        # Vectorized match extraction
        match_idx = np.where(X_cpu > 0.5)
        matches = list(zip(match_idx[0], match_idx[1]))
        
        # Ensure exactly L matches
        if len(matches) > L:
            matches = matches[:L]
        elif len(matches) < L:
            remaining_idx = np.where((X_cpu > 0.1) & (X_cpu <= 0.5))
            if len(remaining_idx[0]) > 0:
                vals = X_cpu[remaining_idx]
                order = np.argsort(-vals)  # descending
                needed = L - len(matches)
                for k in order[:needed]:
                    matches.append((remaining_idx[0][k], remaining_idx[1][k]))
        
        # Calculate final similarity score using the objective F(X)
        score = self._compute_final_score(A_G, A_H, C, X_cpu, L)
        
        if verbose:
            print(f"[OK] Found {len(matches)} matches (Sim Score: {score:.3f})")
            print("="*60)
        
        return X_cpu, matches, score

    def _compute_final_score(self, A_G, A_H, C, X, L):
        """Computes the final exponential kernel similarity score."""
        r = X.sum(axis=1)
        U = np.outer(r, r)
        struct_diff = U * A_G - X @ A_H @ X.T
        struct_cost = np.sum(struct_diff ** 2)
        app_cost = np.einsum('ij,ij->', C, X)
        F_obj = self.alpha * struct_cost + (1 - self.alpha) * app_cost
        return np.exp(-self.gamma * F_obj / max(L, 1))
    
    def _build_weighted_adjacency(self, mol):
        """Build weighted adjacency matrix (CPU bound)."""
        n = mol.GetNumAtoms()
        A = np.zeros((n, n))
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bt = bond.GetBondType()
            if bt == Chem.BondType.SINGLE: w = 1.0
            elif bt == Chem.BondType.DOUBLE: w = 2.0
            elif bt == Chem.BondType.TRIPLE: w = 3.0
            elif bt == Chem.BondType.AROMATIC: w = 1.5
            else: w = 1.0
            if bond.IsInRing(): w += 0.1
            if bond.GetIsConjugated(): w += 0.1
            A[i, j] = w
            A[j, i] = w
        return A
    
    def _build_cost_matrix(self, mol1, mol2):
        """Build cost matrix via numpy broadcasting (CPU bound).
        NOTE: Element-mismatch penalty is applied AFTER normalization in the caller."""
        f1 = self._extract_atom_features(mol1)
        f2 = self._extract_atom_features(mol2)
        C = np.zeros((len(f1['atomic_num']), len(f2['atomic_num'])))
        C += np.abs(f1['degree'][:, None] - f2['degree'][None, :]) * 10
        C += (f1['is_in_ring'][:, None] != f2['is_in_ring'][None, :]) * 5
        C += (f1['is_aromatic'][:, None] != f2['is_aromatic'][None, :]) * 10
        C += np.abs(f1['charge'][:, None] - f2['charge'][None, :]) * 20
        C += np.abs(f1['mass'][:, None] - f2['mass'][None, :]) * 0.5
        C += (f1['hybridization'][:, None] != f2['hybridization'][None, :]) * 15
        C += (f1['num_hs'][:, None] != f2['num_hs'][None, :]) * 5
        C += (f1['chiral_tag'][:, None] != f2['chiral_tag'][None, :]) * 5
        C += (f1['isotope'][:, None] != f2['isotope'][None, :]) * 2
        return C
    
    def _extract_atom_features(self, mol):
        """Extract features into numpy arrays (CPU bound)."""
        atoms = list(mol.GetAtoms())
        return {
            'atomic_num': np.array([a.GetAtomicNum() for a in atoms]),
            'degree': np.array([a.GetDegree() for a in atoms], dtype=float),
            'is_in_ring': np.array([a.IsInRing() for a in atoms]),
            'is_aromatic': np.array([a.GetIsAromatic() for a in atoms]),
            'mass': np.array([a.GetMass() for a in atoms]),
            'charge': np.array([a.GetFormalCharge() for a in atoms], dtype=float),
            'hybridization': np.array([str(a.GetHybridization()) for a in atoms]),
            'num_hs': np.array([a.GetTotalNumHs() for a in atoms]),
            'chiral_tag': np.array([str(a.GetChiralTag()) for a in atoms]),
            'isotope': np.array([a.GetIsotope() for a in atoms]),
        }
    
    def _gnccp_optimization(self, A_G, A_H, C, L, verbose=True):
        """GNCCP Optimization (Dynamically routes to CPU or GPU)"""
        xp = self.xp
        
        if self.use_gpu:
            A_G = xp.asarray(A_G)
            A_H = xp.asarray(A_H)
            C = xp.asarray(C)
        
        M, N = A_G.shape[0], A_H.shape[0]
        
        # Initialize X uniformly
        X = xp.ones((M, N)) * (L / (M * N))
        # Cosine ζ schedule: spends more iterations near ζ=0 (the real objective)
        t_schedule = xp.linspace(0, float(xp.pi), self.max_iter)
        zeta_values = xp.cos(t_schedule)  # +1 → 0 → -1, slow near 0
        
        for idx, zeta in enumerate(zeta_values):
            if verbose and idx % 10 == 0:
                print(f"   GNCCP Iteration {idx+1}/{self.max_iter}, ζ = {zeta:.3f}")
            
            for t in range(self.fw_max_iter):
                grad = self._compute_gradient_Jzeta(X, A_G, A_H, C, zeta)
                Y = self._solve_lp_direction(grad, L)
                
                D_fw = X - Y
                fw_gap = xp.sum(grad * D_fw)
                if fw_gap < self.tol:
                    break
                
                # Exact Line Search: sample 5 points, pick best
                lambda_opt = self._exact_line_search(X, D_fw, A_G, A_H, C, zeta, L)
                
                X = X - lambda_opt * D_fw
            if float(xp.max(X * (1.0 - X))) < 1e-3:
                break
        
        # Final Discretization
        X_binary = self._discretize_to_binary(X, L)
        return X_binary
    
    def _exact_line_search(self, X, D, A_G, A_H, C, zeta, L):
        """Exact line search by sampling 5 points along the FW direction."""
        xp = self.xp
        best_lam = 0.0
        best_val = float('inf')
        for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
            X_trial = X - lam * D
            X_trial = xp.clip(X_trial, 0, 1)
            val = self._compute_Jzeta(X_trial, A_G, A_H, C, zeta)
            if val < best_val:
                best_val = val
                best_lam = lam
        return best_lam
    
    def _compute_gradient_Jzeta(self, X, A_G, A_H, C, zeta):
        xp = self.xp
        grad_F = self._compute_gradient_F(X, A_G, A_H, C)
        if zeta >= 0:
            grad = (1 - zeta) * grad_F + 2 * zeta * X
        else:
            grad = (1 + zeta) * grad_F + 2 * zeta * X
        
        grad = xp.nan_to_num(grad, nan=0.0, posinf=1e6, neginf=-1e6)
        return grad
    
    def _compute_gradient_F(self, X, A_G, A_H, C):
        xp = self.xp
        r = X.sum(axis=1)
        U = xp.outer(r, r)
        
        # Cache X @ A_H — used for both the residual and term2
        XA_H = X @ A_H
        R = (U * A_G) - (XA_H @ X.T)
        RA_X = (R * A_G) @ X
        term1 = RA_X.sum(axis=1, keepdims=True)
        term2 = R @ XA_H
        
        grad_struct = 4.0 * term1 - 4.0 * term2
        grad_F = self.alpha * grad_struct + (1 - self.alpha) * C
        return grad_F
    
    def _solve_lp_direction(self, grad, L):
        xp = self.xp
        M, N = grad.shape
        grad_cpu = cp.asnumpy(grad) if self.use_gpu else grad
        
        if L >= min(M, N):
            row_ind, col_ind = linear_sum_assignment(grad_cpu)
            Y = xp.zeros((M, N), dtype=np.float32)
            for r, c in zip(row_ind, col_ind):
                if r < M and c < N: Y[r, c] = 1.0
            return Y
            
        dim = M + N - L
        g_min = float(np.min(grad_cpu))
        g_shift = grad_cpu - g_min + 1e-9 if g_min < 0 else grad_cpu
        
        aug = np.full((dim, dim), 1e6, dtype=np.float32)
        aug[:M, :N] = g_shift
        aug[:M, N:] = 0.0
        aug[M:, :N] = 0.0
        
        row_ind, col_ind = linear_sum_assignment(aug)
        Y = xp.zeros((M, N), dtype=np.float32)
        for r, c in zip(row_ind, col_ind):
            if r < M and c < N: Y[r, c] = 1.0
        return Y
    
    def _line_search_Jzeta(self, X, Y, A_G, A_H, C, zeta, L):
        D = Y - X
        lambda_opt = 1.0
        J_old = self._compute_Jzeta(X, A_G, A_H, C, zeta)
        
        for _ in range(20):
            X_new = X + lambda_opt * D
            X_new = self._project_to_relaxation(X_new, L)
            J_new = self._compute_Jzeta(X_new, A_G, A_H, C, zeta)
            if J_new < J_old:
                return lambda_opt
            lambda_opt *= 0.5
        return 0.0
    
    def _compute_Jzeta(self, X, A_G, A_H, C, zeta):
        xp = self.xp
        F = self._compute_F(X, A_G, A_H, C)
        x_norm_sq = xp.sum(X ** 2)
        if zeta >= 0:
            J = (1 - zeta) * F + zeta * x_norm_sq
        else:
            J = (1 + zeta) * F + zeta * x_norm_sq
        return J
    
    def _compute_F(self, X, A_G, A_H, C):
        xp = self.xp
        r = X.sum(axis=1)
        U = xp.outer(r, r)
        diff = (U * A_G) - (X @ A_H @ X.T)
        struct_term = xp.sum(diff ** 2)
        app_term = xp.einsum('ij,ij->', C, X)
        F = self.alpha * struct_term + (1 - self.alpha) * app_term
        return F
    
    def _project_to_relaxation(self, X, L):
        xp = self.xp
        X = xp.maximum(X, 0)
        current_sum = X.sum()
        if current_sum > 0:
            X = X * (L / current_sum)
        
        for _ in range(20):
            row_sums = X.sum(axis=1, keepdims=True)
            over_rows = row_sums > 1.0
            X = xp.where(over_rows, X / row_sums, X)
            
            col_sums = X.sum(axis=0, keepdims=True)
            over_cols = col_sums > 1.0
            X = xp.where(over_cols, X / col_sums, X)
            
            current_sum = X.sum()
            if current_sum > 0:
                X = X * (L / current_sum)
            
            if (xp.all(X.sum(axis=1) <= 1.0 + 1e-6) and
                xp.all(X.sum(axis=0) <= 1.0 + 1e-6)):
                break
        
        X = xp.clip(X, 0, 1)
        return X
    
    def _discretize_to_binary(self, X, L):
        """Discretizes continuous assignment X to exact cardinality-L partial permutation via oracle."""
        return self._solve_lp_direction(-X, L)

    def precompute_molecular_data(mol):
        instance = WCS_GNCCP.__new__(WCS_GNCCP)
        return {
            'A': instance._build_weighted_adjacency(mol),
            'features': instance._extract_atom_features(mol),
            'n_atoms': mol.GetNumAtoms(),
        }
    
    def match_from_precomputed(self, data1, data2, L=None, return_map=False):
        A_G = data1['A'].copy()
        A_H = data2['A'].copy()
        M, N = data1['n_atoms'], data2['n_atoms']
        
        # Shared Scale Normalization to preserve exact ground-truth subgraph zero loss
        max_val = max(float(np.max(np.abs(A_G))), float(np.max(np.abs(A_H))))
        if max_val > 0:
            A_G /= max_val
            A_H /= max_val
        
        f1, f2 = data1['features'], data2['features']
        # Fine-grained atom feature cost (element penalty added AFTER normalization)
        C = np.zeros((M, N))
        C += np.abs(f1['degree'][:, None] - f2['degree'][None, :]) * 10
        C += (f1['is_in_ring'][:, None] != f2['is_in_ring'][None, :]) * 5
        C += (f1['is_aromatic'][:, None] != f2['is_aromatic'][None, :]) * 10
        C += np.abs(f1['charge'][:, None] - f2['charge'][None, :]) * 20
        C += np.abs(f1['mass'][:, None] - f2['mass'][None, :]) * 0.5
        C += (f1['hybridization'][:, None] != f2['hybridization'][None, :]) * 15
        C += (f1['num_hs'][:, None] != f2['num_hs'][None, :]) * 5
        C += (f1['chiral_tag'][:, None] != f2['chiral_tag'][None, :]) * 5
        C += (f1['isotope'][:, None] != f2['isotope'][None, :]) * 2
        
        # Normalize fine-grained features to [0, 1]
        if C.max() > C.min():
            C = (C - C.min()) / (C.max() - C.min() + 1e-8)
        else:
            C = C / (C.max() + 1e-8)
        
        # Heuristic Atom Pruning: element-mismatch penalty AFTER normalization
        # Prevents normalization from compressing fine features (PDF §I: C_ij → ∞)
        element_mismatch = (f1['atomic_num'][:, None] != f2['atomic_num'][None, :])
        C[element_mismatch] = 50.0
        
        if L is None:
            L = max(min(M, N) - 2, 1)
        else:
            L = min(L, M, N)
        
        X = self._gnccp_optimization(A_G, A_H, C, L, verbose=False)
        X_cpu = cp.asnumpy(X) if self.use_gpu else X
        
        score = self._compute_final_score(A_G, A_H, C, X_cpu, L)
        if return_map:
            return score, X_cpu
        return score
