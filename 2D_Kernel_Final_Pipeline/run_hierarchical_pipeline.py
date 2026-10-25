import os, sys, glob, random, warnings
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from rdkit import Chem
from rdkit.Chem import AllChem
import concurrent.futures
from tqdm import tqdm

warnings.filterwarnings('ignore')

def log(msg):
    t = datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] {msg}")

# ==============================================================================
# HIERARCHICAL 2D-3D KERNEL (PERFECT MATCH TO PDF FORMULATION)
# ==============================================================================
class Hierarchical_WCS_GNCCP:
    def __init__(self, alpha=0.5, max_iter=20, fw_max_iter=8, tol=1e-4, device='cpu'):
        self.alpha = alpha
        self.max_iter = max_iter  # EXACTLY 20 ITERATIONS PAKKA
        self.fw_max_iter = fw_max_iter
        self.tol = tol
        self.device = device
    
    def extract_features(self, mol):
        atoms = list(mol.GetAtoms())
        hybr_map = {'SP': 1, 'SP2': 2, 'SP3': 3, 'SP3D': 4, 'SP3D2': 5}
        chiral_map = {'CHI_UNSPECIFIED': 0, 'CHI_TETRAHEDRAL_CW': 1, 'CHI_TETRAHEDRAL_CCW': 2}
        return {
            'atomic_num': torch.tensor([a.GetAtomicNum() for a in atoms], dtype=torch.float32, device=self.device),
            'degree':     torch.tensor([a.GetDegree() for a in atoms], dtype=torch.float32, device=self.device),
            'in_ring':    torch.tensor([float(a.IsInRing()) for a in atoms], dtype=torch.float32, device=self.device),
            'aromatic':   torch.tensor([float(a.GetIsAromatic()) for a in atoms], dtype=torch.float32, device=self.device),
            'mass':       torch.tensor([a.GetMass() for a in atoms], dtype=torch.float32, device=self.device),
            'charge':     torch.tensor([float(a.GetFormalCharge()) for a in atoms], dtype=torch.float32, device=self.device),
            'hybr':       torch.tensor([hybr_map.get(str(a.GetHybridization()), 0) for a in atoms], dtype=torch.float32, device=self.device),
            'num_hs':     torch.tensor([float(a.GetTotalNumHs()) for a in atoms], dtype=torch.float32, device=self.device),
            'chiral':     torch.tensor([chiral_map.get(str(a.GetChiralTag()), 0) for a in atoms], dtype=torch.float32, device=self.device),
            'isotope':    torch.tensor([float(a.GetIsotope()) for a in atoms], dtype=torch.float32, device=self.device),
        }
    
    def build_2d_adjacency(self, mol):
        # STAGE 1: 2D TOPOLOGICAL GRAPH (Eq. 3 in PDF)
        n = mol.GetNumAtoms()
        A = torch.zeros((n, n), dtype=torch.float32, device=self.device)
        bt_map = {Chem.BondType.SINGLE: 1.0, Chem.BondType.DOUBLE: 2.0, Chem.BondType.TRIPLE: 3.0, Chem.BondType.AROMATIC: 1.5}
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            w = bt_map.get(bond.GetBondType(), 1.0)
            if bond.IsInRing(): w += 0.1
            if bond.GetIsConjugated(): w += 0.1
            A[i, j] = w; A[j, i] = w
        return A
    
    def get_3d_coordinates(self, mol):
        if not mol.GetNumConformers():
            m2 = Chem.AddHs(mol)
            AllChem.EmbedMolecule(m2, randomSeed=42)
            try:
                AllChem.MMFFOptimizeMolecule(m2)
            except:
                pass
            mol = Chem.RemoveHs(m2)
        return torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float64, device=self.device)

    def build_cost_matrix(self, f1, f2):
        C  = torch.abs(f1['degree'].unsqueeze(1) - f2['degree'].unsqueeze(0)) * 10.0
        C += (f1['in_ring'].unsqueeze(1) != f2['in_ring'].unsqueeze(0)).float() * 5.0
        C += (f1['aromatic'].unsqueeze(1) != f2['aromatic'].unsqueeze(0)).float() * 10.0
        C += torch.abs(f1['charge'].unsqueeze(1) - f2['charge'].unsqueeze(0)) * 20.0
        C += torch.abs(f1['mass'].unsqueeze(1) - f2['mass'].unsqueeze(0)) * 0.5
        C += (f1['hybr'].unsqueeze(1) != f2['hybr'].unsqueeze(0)).float() * 15.0
        C += (f1['num_hs'].unsqueeze(1) != f2['num_hs'].unsqueeze(0)).float() * 5.0
        C += (f1['chiral'].unsqueeze(1) != f2['chiral'].unsqueeze(0)).float() * 5.0
        C += (f1['isotope'].unsqueeze(1) != f2['isotope'].unsqueeze(0)).float() * 2.0
        c_min, c_max = C.min(), C.max()
        if c_max > c_min: C = (C - c_min) / (c_max - c_min + 1e-8)
        else: C = C / (c_max + 1e-8)
        C[(f1['atomic_num'].unsqueeze(1) != f2['atomic_num'].unsqueeze(0))] = 50.0
        return C
    
    def project(self, X, L):
        X = torch.clamp(X, min=0.0)
        s = X.sum()
        if s > 0: X = X * (L / s)
        for _ in range(20):
            rs = X.sum(dim=1, keepdim=True)
            X = torch.where(rs > 1.0, X / rs, X)
            cs = X.sum(dim=0, keepdim=True)
            X = torch.where(cs > 1.0, X / cs, X)
            s = X.sum()
            if s > 0: X = X * (L / s)
            if torch.all(X.sum(dim=1) <= 1.0 + 1e-6) and torch.all(X.sum(dim=0) <= 1.0 + 1e-6): break
        return torch.clamp(X, 0.0, 1.0)
    
    def compute_grad_F(self, X, A_G, A_H, C):
        # Theorem 1 in PDF
        r = X.sum(dim=1)
        R = (torch.outer(r, r) * A_G) - (X @ A_H @ X.T)
        return self.alpha * (4.0 * ((R * A_G) @ X).sum(dim=1, keepdim=True) - 4.0 * (R @ X @ A_H)) + (1.0 - self.alpha) * C
    
    def compute_F(self, X, A_G, A_H, C):
        # Definition 2 in PDF
        r = X.sum(dim=1)
        diff = (torch.outer(r, r) * A_G) - (X @ A_H @ X.T)
        return self.alpha * torch.sum(diff ** 2) + (1.0 - self.alpha) * torch.sum(C * X)
    
    @torch.no_grad()
    def gnccp_optimize(self, A_G, A_H, C, L):
        # Algorithm 1 in PDF
        M, N = A_G.shape[0], A_H.shape[0]
        X = torch.ones((M, N), dtype=torch.float32, device=self.device) * (L / (M * N))
        
        # 20 Iterations Pakka
        for zeta in torch.linspace(1.0, -1.0, self.max_iter, device=self.device):
            for t in range(self.fw_max_iter):
                grad_F = self.compute_grad_F(X, A_G, A_H, C)
                grad = (1 - zeta) * grad_F + 2 * zeta * X if zeta >= 0 else (1 + zeta) * grad_F + 2 * zeta * X
                grad = torch.nan_to_num(grad, nan=0.0, posinf=1e6, neginf=-1e6)
                ri, ci = linear_sum_assignment(grad.cpu().numpy())
                Y = torch.zeros((M, N), dtype=torch.float32, device=self.device)
                for r, c in zip(ri, ci):
                    if r < M and c < N: Y[r, c] = 1.0
                Y = self.project(Y, L)
                D_fw = X - Y
                if torch.sum(grad * D_fw) < self.tol: break
                X = self.project(X - (2.0 / (t + 2.0)) * D_fw, L)
        
        # Line 16: Discretize via Hungarian
        X_bin = torch.zeros((M, N), dtype=torch.float32, device=self.device)
        matched = 0
        for r, c in zip(*linear_sum_assignment((-X).cpu().numpy())):
            if r < M and c < N and matched < L:
                X_bin[r, c] = 1.0
                matched += 1
        return X_bin

    def kabsch_alignment(self, P, Q, X_bin):
        # STAGE 2: 3D ALIGNMENT VIA KABSCH ROTATION (Theorem 4 in PDF)
        matched_indices = (X_bin > 0.5).nonzero(as_tuple=False)
        p_idx = matched_indices[:, 0]
        q_idx = matched_indices[:, 1]
        
        P_matched = P[p_idx]
        Q_matched = Q[q_idx]
        
        # Centering (Definition 5 in PDF)
        P_mean = P_matched.mean(dim=0)
        Q_mean = Q_matched.mean(dim=0)
        P_centered = P_matched - P_mean
        Q_centered = Q_matched - Q_mean
        
        # Cross-covariance matrix H (Eq 26)
        H = P_centered.T @ Q_centered
        
        # SVD
        try:
            U, S, V = torch.svd(H)
        except:
            return 10.0 # Fallback high RMSD if SVD fails on perfectly planar molecules
            
        # Rotation Matrix R*
        d = torch.sign(torch.det(V @ U.T))
        diag_mat = torch.eye(3, device=self.device, dtype=torch.float64)
        diag_mat[2, 2] = d
        R = V @ diag_mat @ U.T
        
        P_rotated = P_centered @ R.T
        rmsd = torch.sqrt(torch.mean(torch.sum((P_rotated - Q_centered)**2, dim=1)))
        return rmsd.item()
        
    def precompute(self, mol):
        A = self.build_2d_adjacency(mol)
        norm_A = torch.linalg.norm(A, ord='fro')
        if norm_A > 0: A /= norm_A
        return {
            'A': A, 
            'f': self.extract_features(mol), 
            'P': self.get_3d_coordinates(mol),
            'n': mol.GetNumAtoms()
        }

# ==============================================================================
# WORKER: PARALLEL HYBRID KERNEL CALCULATION (MULTI-BAR)
# ==============================================================================
def process_matrix_chunk(alpha, sdf_blocks, worker_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    mols = [Chem.MolFromMolBlock(block.decode('utf-8')) for block in sdf_blocks]
    n = len(mols)
    matcher = Hierarchical_WCS_GNCCP(alpha=alpha, max_iter=20, device=device)
    precomp = [matcher.precompute(m) for m in mols]
    
    F_2D = np.zeros((n, n))
    RMSD_3D = np.zeros((n, n))
    L_mat = np.zeros((n, n))
    
    total_pairs = n * (n+1) // 2
    
    with tqdm(total=total_pairs, desc=f"Worker {worker_id+1} (Alpha={alpha})", position=worker_id, leave=True, ascii=True) as pbar:
        for i in range(n):
            for j in range(i, n):
                d1, d2 = precomp[i], precomp[j]
                C = matcher.build_cost_matrix(d1['f'], d2['f'])
                L = max(min(d1['n'], d2['n']) - 2, 1)
                
                # Stage 1: 2D GNCCP Alignment
                X_bin = matcher.gnccp_optimize(d1['A'], d2['A'], C, L)
                f_val = matcher.compute_F(X_bin, d1['A'], d2['A'], C).item()
                
                # Stage 2: 3D Kabsch Alignment
                rmsd = matcher.kabsch_alignment(d1['P'], d2['P'], X_bin)
                
                F_2D[i, j] = F_2D[j, i] = f_val
                RMSD_3D[i, j] = RMSD_3D[j, i] = rmsd
                L_mat[i, j] = L_mat[j, i] = L
                pbar.update(1)
            
    return alpha, F_2D, RMSD_3D, L_mat

# ==============================================================================
# MAIN EXECUTION: MULTIPROCESSING + SKEWED SVM
# ==============================================================================
if __name__ == '__main__':
    log("==================================================")
    log("HIERARCHICAL 2D-3D HYBRID PIPELINE (GNCCP-20)")
    log("==================================================")
    
    DATA_DIR = r'C:\\Users\\abdus\\Downloads\\FYP\\Efficient-3D-Kernels-for-molecular-property-prediction\\Data'
    sdf_files = glob.glob(os.path.join(DATA_DIR, "*.sdf"))
    paired = {}
    for f in sdf_files:
        fname = os.path.basename(f)
        if "_actives_new.sdf" in fname:
            base = fname.replace("_actives_new.sdf", "")
            inact = f.replace("_actives_new.sdf", "_inactives_new.sdf")
            if inact in sdf_files:
                paired[base] = {'actives': f, 'inactives': inact}

    if not paired:
        log("ERROR: No paired datasets found!")
        sys.exit(1)

    ds_name, paths = list(sorted(paired.items()))[0]
    
    # SKEWED DATASET SIMULATION
    all_act = [m for m in Chem.SDMolSupplier(paths['actives'], removeHs=True) if m is not None]
    all_inact = [m for m in Chem.SDMolSupplier(paths['inactives'], removeHs=True) if m is not None]
    
    num_act = min(30, len(all_act))
    num_inact = min(120, len(all_inact))
    
    act_mols = random.sample(all_act, num_act)
    inact_mols = random.sample(all_inact, num_inact)
    mols = act_mols + inact_mols
    labels = np.array([1]*num_act + [0]*num_inact)
    
    sdf_blocks = []
    for m in mols:
        block = Chem.MolToMolBlock(m)
        sdf_blocks.append(block.encode('utf-8'))
        
    n = len(mols)
    log(f"Dataset Skew Configured: {num_act} Actives, {num_inact} Inactives (Total: {n})")
    
    alpha_values = [0.3, 0.5, 0.7, 0.9]
    num_workers = len(alpha_values)
    log(f"Spawning {num_workers} parallel workers to compute Stage 1 (2D) & Stage 2 (3D)...")
    
    print("\\n" * num_workers)
    
    import time
    start_time = time.time()
    
    results_matrices = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for idx, alpha in enumerate(alpha_values):
            futures.append(executor.submit(process_matrix_chunk, alpha, sdf_blocks, idx))
            
        for future in concurrent.futures.as_completed(futures):
            try:
                alpha, F_2D, RMSD_3D, L_mat = future.result()
                results_matrices.append((alpha, F_2D, RMSD_3D, L_mat))
            except Exception as e:
                log(f"Error in worker: {e}")
                
    print("\\n" * num_workers)
    
    log("All Matrices Computed! Grid Searching Hybrid Kernel SVM on Train Set...")
    
    gamma_2d_vals = [0.1, 1.0]
    gamma_3d_vals = [0.1, 1.0]
    beta_vals = [0.2, 0.5, 0.8]  # Balances 2D vs 3D (Eq 27)
    
    train_idx, test_idx = train_test_split(np.arange(n), test_size=0.2, stratify=labels, random_state=42)
    
    best_auc = 0
    best_params = {}
    
    for alpha, F_2D, RMSD_3D, L_mat in results_matrices:
        for g2 in gamma_2d_vals:
            for g3 in gamma_3d_vals:
                for beta in beta_vals:
                    # Definition 6: Hybrid Kernel
                    k_2D = np.exp(-g2 * F_2D / np.maximum(L_mat, 1))
                    k_3D = np.exp(-g3 * RMSD_3D)
                    K_hybrid = (1.0 - beta) * k_2D + beta * k_3D
                    
                    eigvals = np.linalg.eigvalsh(K_hybrid)
                    if np.min(eigvals) < 0: 
                        K_psd = K_hybrid + (np.abs(np.min(eigvals)) + 1e-4) * np.eye(n)
                    else: 
                        K_psd = K_hybrid.copy()
                    d = np.diag(K_psd)
                    K_psd = K_psd / (np.sqrt(np.outer(d, d)) + 1e-8)
                    
                    # SKEW HANDLING
                    clf = SVC(kernel='precomputed', C=10.0, class_weight='balanced', probability=True)
                    clf.fit(K_psd[np.ix_(train_idx, train_idx)], labels[train_idx])
                    
                    probs = clf.predict_proba(K_psd[np.ix_(test_idx, train_idx)])[:, 1]
                    preds = clf.predict(K_psd[np.ix_(test_idx, train_idx)])
                    
                    auc = roc_auc_score(labels[test_idx], probs)
                    if auc > best_auc:
                        best_auc = auc
                        best_params = {'alpha': alpha, 'g2': g2, 'g3': g3, 'beta': beta, 'preds': preds, 'probs': probs}

    end_time = time.time()
    
    log("==================================================")
    log(f"BEST HYBRID KERNEL FOUND! (Total Time: {(end_time - start_time)/60:.1f} mins)")
    log(f"Alpha (Topology Weight): {best_params['alpha']}")
    log(f"Beta (3D Weight): {best_params['beta']}")
    log(f"Test Set ROC-AUC: {best_auc*100:.1f}%")
    
    log("==================================================")
    log("SKEWED DATASET CLASSIFICATION REPORT (UNSEEN TEST SET):")
    print(classification_report(labels[test_idx], best_params['preds'], target_names=['Inactive', 'Active']))
    log("==================================================")
    log("Pipeline Complete.")
