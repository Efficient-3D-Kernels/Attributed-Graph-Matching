import time
import numpy as np
from rdkit import Chem
import os
import torch
from scipy.optimize import linear_sum_assignment

# --- MONKEY PATCH FOR NUMPY 2.0+ COMPATIBILITY WITH GRAKEL ---
if not hasattr(np, 'ComplexWarning'):
    np.ComplexWarning = Warning
# -------------------------------------------------------------

# We will use GraKeL for the competitor kernels
try:
    import grakel
    from grakel.kernels import (
        WeisfeilerLehman, 
        ShortestPath,
        VertexHistogram,
        EdgeHistogram,
        GraphletSampling,
        PyramidMatch
    )
    from grakel import Graph
except ImportError:
    print("GraKeL is not installed. Please run: pip install grakel")
    exit(1)


# PYTORCH IMPLEMENTATION
class WCS_GNCCP_PyTorch:
    def __init__(self, alpha=0.7, gamma=0.5, max_iter=8, fw_max_iter=8, tol=1e-3):
        self.alpha = alpha
        self.gamma = gamma
        self.max_iter = max_iter
        self.fw_max_iter = fw_max_iter
        self.tol = tol
        # Use CPU for small matrices (PCIe bus transfer to GPU is too slow for 10x10 matrices)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def match(self, mol1, mol2):
        L = min(mol1.GetNumAtoms(), mol2.GetNumAtoms())
        
        A_G = torch.tensor(self._build_weighted_adjacency(mol1), dtype=torch.float32, device=self.device)
        A_H = torch.tensor(self._build_weighted_adjacency(mol2), dtype=torch.float32, device=self.device)
        C = torch.tensor(self._build_cost_matrix(mol1, mol2), dtype=torch.float32, device=self.device)
        
        norm_G = torch.norm(A_G, p='fro')
        norm_H = torch.norm(A_H, p='fro')
        if norm_G > 0: A_G /= norm_G
        if norm_H > 0: A_H /= norm_H
        if torch.max(C) > 0: C /= torch.max(C)
        
        M, N = C.shape
        X = torch.full((M, N), L / (M * N), dtype=torch.float32, device=self.device)
        
        zeta_vals = np.linspace(1, -1, self.max_iter)
        for zeta in zeta_vals:
            for t in range(self.fw_max_iter):
                r = X.sum(dim=1)
                U = torch.outer(r, r)
                R = (U * A_G) - (X @ A_H @ X.T)
                
                # Gradient Computation
                RA_X = (R * A_G) @ X
                term1 = RA_X.sum(dim=1, keepdim=True)
                term2 = R @ X @ A_H
                grad_struct = 4.0 * term1 - 4.0 * term2
                grad_F = self.alpha * grad_struct + (1 - self.alpha) * C
                
                if zeta >= 0:
                    grad = (1 - zeta) * grad_F + 2 * zeta * X
                else:
                    grad = (1 + zeta) * grad_F + 2 * zeta * X
                
                # UPGRADE 1: FW Direction (SINKHORN instead of Hungarian for massive speed)
                epsilon = 0.05
                grad_scaled = grad / epsilon
                grad_scaled -= torch.min(grad_scaled)
                Y = torch.exp(-grad_scaled)
                Y = self._project_to_relaxation(Y, L)
                
                # UPGRADE 3: FW Step Size (Exact Decay removes line search loop)
                lam = 2.0 / (t + 2.0)
                X = X + lam * (Y - X)
                X = self._project_to_relaxation(X, L)
        
        # Final Hungarian
        X_cpu = -X.detach().cpu().numpy()
        r_i, c_i = linear_sum_assignment(X_cpu)
        X_bin = torch.zeros((M, N), dtype=torch.float32, device=self.device)
        matched = 0
        for r, c in zip(r_i, c_i):
            if r < M and c < N and matched < L:
                X_bin[r, c] = 1.0
                matched += 1
                
        r = X_bin.sum(dim=1)
        U = torch.outer(r, r)
        obj_F = torch.norm((U * A_G) - (X_bin @ A_H @ X_bin.T), p='fro')**2
        f_val = self.alpha * obj_F + (1 - self.alpha) * torch.sum(C * X_bin)
        
        sim = np.exp(-self.gamma * f_val.item() / L)
        return sim

    def _project_to_relaxation(self, X, L):
        X = torch.clamp(X, min=0.0)
        c_sum = X.sum()
        if c_sum > 0: X = X * (L / c_sum)
        
        row_sums = X.sum(dim=1, keepdim=True)
        mask = row_sums > 1.0
        X = torch.where(mask.expand_as(X), X / (row_sums + 1e-8), X)
        
        col_sums = X.sum(dim=0, keepdim=True)
        mask = col_sums > 1.0
        X = torch.where(mask.expand_as(X), X / (col_sums + 1e-8), X)
        
        c_sum = X.sum()
        if c_sum > 0: X = X * (L / c_sum)
        return torch.clamp(X, 0.0, 1.0)
    
    def _build_weighted_adjacency(self, mol):
        n = mol.GetNumAtoms()
        A = np.zeros((n, n))
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bt = bond.GetBondType()
            w = 1.0
            if bt == Chem.BondType.DOUBLE: w = 2.0
            elif bt == Chem.BondType.TRIPLE: w = 3.0
            elif bt == Chem.BondType.AROMATIC: w = 1.5
            A[i, j] = w 
            A[j, i] = w 
        return A
        
    def _build_cost_matrix(self, mol1, mol2):
        M, N = mol1.GetNumAtoms(), mol2.GetNumAtoms()
        C = np.zeros((M, N))
        f1 = [a.GetAtomicNum() for a in mol1.GetAtoms()]
        f2 = [a.GetAtomicNum() for a in mol2.GetAtoms()]
        for i in range(M):
            for j in range(N):
                if f1[i] != f2[j]: C[i, j] = 1e6
        return C

def rdkit_to_grakel(mol):
    mol = Chem.RemoveHs(mol)
    edges = []
    node_labels = {}
    edge_labels = {}
    for atom in mol.GetAtoms():
        node_labels[atom.GetIdx()] = atom.GetAtomicNum()
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        edges.append((u, v))
        edges.append((v, u))
        b_type = float(bond.GetBondTypeAsDouble())
        edge_labels[(u, v)] = b_type
        edge_labels[(v, u)] = b_type
    return Graph(edges, node_labels=node_labels, edge_labels=edge_labels)

def run_speed_benchmark(sample_size=10):
    print("="*60)
    print("PYTORCH KERNEL SPEED BENCHMARKING")
    print("="*60)
    
    data_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'Aromatase_actives_new.sdf')
    print(f"Loading {sample_size} molecules from: {os.path.basename(data_path)}...")
    mols = []
    suppl = Chem.SDMolSupplier(data_path)
    for m in suppl:
        if m is not None:
            mols.append(Chem.RemoveHs(m))
        if len(mols) >= sample_size:
            break
            
    print(f"Loaded {len(mols)} molecules successfully.\n")
    print("Converting to GraKeL format...")
    grakel_graphs = [rdkit_to_grakel(m) for m in mols]
    
    results = {}
    
    def time_kernel(kernel, name, data):
        start = time.time()
        _ = kernel.fit_transform(data)
        results[name] = time.time() - start

    print("\n--- RUNNING COMPETITORS ---")
    time_kernel(WeisfeilerLehman(n_iter=3, normalize=True), "Weisfeiler-Lehman (WL)", grakel_graphs)
    time_kernel(ShortestPath(normalize=True), "Shortest Path (SP)", grakel_graphs)
    time_kernel(VertexHistogram(normalize=True), "Vertex Histogram (VH)", grakel_graphs)
    
    print("\n--- RUNNING OUR PYTORCH KERNEL ---")
    matcher = WCS_GNCCP_PyTorch()
    start = time.time()
    n = len(mols)
    for i in range(n):
        for j in range(i, n):
            matcher.match(mols[i], mols[j])
    our_time = time.time() - start
    results["WCS-GNCCP (PyTorch)"] = our_time
    
    print("\n" + "="*60)
    print(f"FINAL SPEED RESULTS ({sample_size}x{sample_size} Kernel Matrix)")
    print("="*60)
    print(f"{'Kernel Name':<25} | {'Total Time (s)':<15} | {'Time per pair (ms)':<15}")
    print("-" * 60)
    
    total_pairs = (sample_size * sample_size)
    for name, t in sorted(results.items(), key=lambda x: x[1]):
        print(f"{name:<25} | {t:<15.3f} | {(t / total_pairs) * 1000:<15.3f}")

if __name__ == "__main__":
    run_speed_benchmark(sample_size=10)
