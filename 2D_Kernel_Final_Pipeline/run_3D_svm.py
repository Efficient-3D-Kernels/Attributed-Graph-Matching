import os, sys, glob, random, warnings
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from rdkit import Chem
from rdkit.Chem import AllChem
import concurrent.futures

warnings.filterwarnings('ignore')

def log(msg):
    t = datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] {msg}")

# ==============================================================================
# 3D KERNEL CLASS DEFINITION 
# ==============================================================================
class WCS_GNCCP_3D_GPU:
    def __init__(self, alpha=0.7, gamma=1.0, max_iter=8, fw_max_iter=8, tol=1e-4, device='cpu'):
        self.alpha, self.gamma, self.max_iter, self.fw_max_iter, self.tol = alpha, gamma, max_iter, fw_max_iter, tol
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
    
    def build_3d_adjacency(self, mol, sigma=2.0):
        """
        Builds a 3D fully-connected spatial adjacency matrix using 
        Euclidean distances between all atoms, wrapped in an RBF kernel.
        """
        n = mol.GetNumAtoms()
        
        # Ensure 3D coordinates exist
        if not mol.GetNumConformers():
            m2 = Chem.AddHs(mol)
            AllChem.EmbedMolecule(m2, randomSeed=42)
            try:
                AllChem.MMFFOptimizeMolecule(m2)
            except:
                pass
            mol = Chem.RemoveHs(m2)
            
        conf = mol.GetConformer()
        coords = torch.tensor(conf.GetPositions(), dtype=torch.float32, device=self.device)
        
        # Calculate pairwise 3D Euclidean distances
        dist_matrix = torch.cdist(coords, coords, p=2)
        
        # Convert distances into spatial adjacency (closer = higher weight)
        A = torch.exp(- (dist_matrix ** 2) / (2 * sigma ** 2))
        return A
    
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
        r = X.sum(dim=1)
        R = (torch.outer(r, r) * A_G) - (X @ A_H @ X.T)
        return self.alpha * (4.0 * ((R * A_G) @ X).sum(dim=1, keepdim=True) - 4.0 * (R @ X @ A_H)) + (1.0 - self.alpha) * C
    
    def compute_F(self, X, A_G, A_H, C):
        r = X.sum(dim=1)
        diff = (torch.outer(r, r) * A_G) - (X @ A_H @ X.T)
        return self.alpha * torch.sum(diff ** 2) + (1.0 - self.alpha) * torch.sum(C * X)
    
    @torch.no_grad()
    def gnccp_optimize(self, A_G, A_H, C, L):
        M, N = A_G.shape[0], A_H.shape[0]
        X = torch.ones((M, N), dtype=torch.float32, device=self.device) * (L / (M * N))
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
        
        X_bin = torch.zeros((M, N), dtype=torch.float32, device=self.device)
        matched = 0
        for r, c in zip(*linear_sum_assignment((-X).cpu().numpy())):
            if r < M and c < N and matched < L:
                X_bin[r, c] = 1.0
                matched += 1
        return X_bin
    
    def precompute(self, mol):
        # USING THE 3D SPATIAL KERNEL INSTEAD OF 2D
        A = self.build_3d_adjacency(mol)
        # Shared scale / unnormalized graph topology
        pass
        return {'A': A, 'f': self.extract_features(mol), 'n': mol.GetNumAtoms()}

# ==============================================================================
# WORKER: 3D KERNEL -> SVM ACCURACY
# ==============================================================================
def process_alpha(alpha, sdf_blocks, labels, gamma_values, worker_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # We pass SDF blocks instead of SMILES so we preserve the 3D coordinates!
    mols = []
    for block in sdf_blocks:
        mols.append(Chem.MolFromMolBlock(block.decode('utf-8')))
        
    n = len(mols)
    
    matcher = WCS_GNCCP_3D_GPU(alpha=alpha, gamma=1.0, max_iter=8, fw_max_iter=8, device=device)
    precomp = [matcher.precompute(m) for m in mols]
    
    F_raw = np.zeros((n, n))
    L_mat = np.zeros((n, n))
    
    total_pairs = n * (n+1) // 2
    count = 0
    
    for i in range(n):
        for j in range(i, n):
            d1, d2 = precomp[i], precomp[j]
            C = matcher.build_cost_matrix(d1['f'], d2['f'])
            L = max(min(d1['n'], d2['n']) - 2, 1)
            X = matcher.gnccp_optimize(d1['A'], d2['A'], C, L)
            
            f_val = matcher.compute_F(X, d1['A'], d2['A'], C).item()
            F_raw[i, j] = F_raw[j, i] = f_val
            L_mat[i, j] = L_mat[j, i] = L
            
            count += 1
            if count % max(1, total_pairs // 4) == 0:
                log(f"[Worker Alpha={alpha}] {count}/{total_pairs} 3D pairs matched...")
                
    # ==========================================================================
    # BRINGING BACK THE SVM 
    # ==========================================================================
    best_acc = 0
    best_gamma = 0
    
    for gamma in gamma_values:
        K = np.exp(-gamma * F_raw / np.maximum(L_mat, 1))
        eigvals = np.linalg.eigvalsh(K)
        if np.min(eigvals) < 0: K_psd = K + (np.abs(np.min(eigvals)) + 1e-4) * np.eye(n)
        else: K_psd = K.copy()
        d = np.diag(K_psd)
        K_psd = K_psd / (np.sqrt(np.outer(d, d)) + 1e-8)
        
        # 5-Fold Cross Validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        clf = SVC(kernel='precomputed', C=10.0, class_weight='balanced')
        accs = []
        for tr, ts in skf.split(K_psd, labels):
            clf.fit(K_psd[np.ix_(tr, tr)], labels[tr])
            accs.append(accuracy_score(labels[ts], clf.predict(K_psd[np.ix_(ts, tr)])))
        
        acc = np.mean(accs) * 100
        if acc > best_acc:
            best_acc = acc
            best_gamma = gamma
            
    return alpha, best_gamma, best_acc

# ==============================================================================
# MAIN EXECUTION THREAD
# ==============================================================================
if __name__ == '__main__':
    log("==================================================")
    log("3D KERNEL SVM PIPELINE INITIALIZING...")
    log("==================================================")
    
    alpha_values = [0.3, 0.5, 0.7, 0.9]
    gamma_values = [0.1, 0.5, 1.0, 2.0]

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
    log(f"TARGET SELECTED: {ds_name}")
    
    # 50 MOLECULES TOTAL AS REQUESTED (25 Actives, 25 Inactives)
    MAX_PER_CLASS = 25
    all_act = [m for m in Chem.SDMolSupplier(paths['actives'], removeHs=True) if m is not None]
    all_inact = [m for m in Chem.SDMolSupplier(paths['inactives'], removeHs=True) if m is not None]
    
    act_mols = random.sample(all_act, min(MAX_PER_CLASS, len(all_act)))
    inact_mols = random.sample(all_inact, min(MAX_PER_CLASS, len(all_inact)))
    mols = act_mols + inact_mols
    labels = np.array([1]*len(act_mols) + [0]*len(inact_mols))
    
    # Save the original SDF blocks to perfectly preserve 3D coordinates across processes
    # (SMILES deletes 3D data, we MUST pass SDF blocks)
    sdf_blocks = []
    for m in mols:
        block = Chem.MolToMolBlock(m)
        sdf_blocks.append(block.encode('utf-8'))
        
    n = len(sdf_blocks)
    log(f"Loaded {n} molecules (with preserved 3D Coordinates!)")
    log(f"Running highly optimized 3D SVM on {n*n} pairs...")
    
    overall_best_acc = 0
    overall_best_alpha = 0
    overall_best_gamma = 0

    import time
    start_time = time.time()
    
    num_workers = len(alpha_values)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for idx, alpha in enumerate(alpha_values):
            futures.append(executor.submit(process_alpha, alpha, sdf_blocks, labels, gamma_values, idx))
            
        for future in concurrent.futures.as_completed(futures):
            try:
                alpha, best_gamma, best_acc = future.result()
                log(f"[Worker Alpha={alpha}] Finished! Best Gamma: {best_gamma} (3D SVM Accuracy: {best_acc:.1f}%)")
                if best_acc > overall_best_acc:
                    overall_best_acc = best_acc
                    overall_best_alpha = alpha
                    overall_best_gamma = best_gamma
            except Exception as e:
                log(f"ERROR in worker: {e}")
                
    end_time = time.time()
    log(f"3D SVM SWEEP COMPLETE! Total Time: {(end_time - start_time)/60:.1f} minutes.")
    log(f"OVERALL BEST 3D MODEL: Alpha={overall_best_alpha}, Gamma={overall_best_gamma} (Accuracy: {overall_best_acc:.1f}%)")
