import os, sys, glob, random, warnings
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem import Draw
import concurrent.futures
import matplotlib.cm as cm
from tqdm import tqdm

warnings.filterwarnings('ignore')

def log(msg):
    t = datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] {msg}")

# ==============================================================================
# CLASS DEFINITION 
# ==============================================================================
class WCS_GNCCP_GPU:
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
    
    def build_adjacency(self, mol):
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
        A = self.build_adjacency(mol)
        # Shared scale / unnormalized graph topology
        pass
        return {'A': A, 'f': self.extract_features(mol), 'n': mol.GetNumAtoms()}

# ==============================================================================
# PURE SCREENING WORKER FUNCTION (NO SVM, NO BS)
# ==============================================================================
def process_alpha(alpha, smiles_list, labels, gamma_values, worker_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    n = len(mols)
    
    matcher = WCS_GNCCP_GPU(alpha=alpha, gamma=1.0, max_iter=8, fw_max_iter=8, device=device)
    precomp = [matcher.precompute(m) for m in mols]
    
    F_raw = np.zeros((n, n))
    L_mat = np.zeros((n, n))
    
    total_pairs = n * (n+1) // 2
    
    with tqdm(total=total_pairs, desc=f"Worker {worker_id+1} (Alpha={alpha})", position=worker_id, leave=True, ascii=True) as pbar:
        for i in range(n):
            for j in range(i, n):
                d1, d2 = precomp[i], precomp[j]
                C = matcher.build_cost_matrix(d1['f'], d2['f'])
                L = max(min(d1['n'], d2['n']) - 2, 1)
                X = matcher.gnccp_optimize(d1['A'], d2['A'], C, L)
                
                f_val = matcher.compute_F(X, d1['A'], d2['A'], C).item()
                F_raw[i, j] = F_raw[j, i] = f_val
                L_mat[i, j] = L_mat[j, i] = L
                pbar.update(1)
                
    # ==========================================================================
    # PURE VIRTUAL SCREENING EVALUATION (NO MACHINE LEARNING)
    # ==========================================================================
    best_auc = 0
    best_gamma = 0
    
    for gamma in gamma_values:
        # 1. Calculate raw similarities
        K = np.exp(-gamma * F_raw / np.maximum(L_mat, 1))
        
        # 2. For every molecule in the dataset, calculate its average similarity 
        # to the KNOWN ACTIVES (excluding itself).
        active_mask = (labels == 1)
        screening_scores = np.zeros(n)
        
        for i in range(n):
            # Create a mask of actives, but remove the current molecule so it doesn't score against itself
            other_actives_mask = active_mask.copy()
            other_actives_mask[i] = False 
            
            # The "Screening Score" is just its average similarity to the actives
            screening_scores[i] = np.mean(K[i, other_actives_mask])
            
        # 3. Calculate how well these raw screening scores rank Actives above Inactives (ROC-AUC)
        auc = roc_auc_score(labels, screening_scores) * 100
        
        if auc > best_auc:
            best_auc = auc
            best_gamma = gamma
            
    return alpha, best_gamma, best_auc, F_raw, L_mat

def visualize_and_save_mapping(smiles_A, smiles_B, alpha, ds_name, score):
    mol_A = Chem.MolFromSmiles(smiles_A)
    mol_B = Chem.MolFromSmiles(smiles_B)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = WCS_GNCCP_GPU(alpha=alpha, gamma=1.0, device=device)
    d1, d2 = m.precompute(mol_A), m.precompute(mol_B)
    C = m.build_cost_matrix(d1['f'], d2['f'])
    L = max(min(d1['n'], d2['n']) - 2, 1)
    X_bin = m.gnccp_optimize(d1['A'], d2['A'], C, L).cpu().numpy()
    
    mapped_A, mapped_B = [], []
    for i in range(X_bin.shape[0]):
        for j in range(X_bin.shape[1]):
            if X_bin[i, j] > 0.5:
                mapped_A.append(i); mapped_B.append(j)
                
    colors = cm.get_cmap('hsv', len(mapped_A))
    dict_A, dict_B = {}, {}
    for idx, (a, b) in enumerate(zip(mapped_A, mapped_B)):
        rgba = colors(idx)
        dict_A[a] = tuple(rgba[:3])
        dict_B[b] = tuple(rgba[:3])
        
    img = Draw.MolsToGridImage(
        [mol_A, mol_B], molsPerRow=2, subImgSize=(500, 500),
        highlightAtomLists=[mapped_A, mapped_B], highlightAtomColors=[dict_A, dict_B],
        legends=["Molecule A", "Molecule B"]
    )
    
    filename = f"{ds_name}_Top_Match.png"
    img.save(filename)
    log(f"📸 Saved Visualization to: {filename}")

# ==============================================================================
# MAIN EXECUTION THREAD
# ==============================================================================
if __name__ == '__main__':
    log("==================================================")
    log("🚀 PARALLEL VIRTUAL SCREENING INITIALIZING...")
    log("==================================================")
    
    SIMILARITY_THRESHOLD = 0.20
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
    
    MAX_PER_CLASS = 50
    all_act = [m for m in Chem.SDMolSupplier(paths['actives'], removeHs=True) if m is not None]
    all_inact = [m for m in Chem.SDMolSupplier(paths['inactives'], removeHs=True) if m is not None]
    
    act_mols = random.sample(all_act, min(MAX_PER_CLASS, len(all_act)))
    inact_mols = random.sample(all_inact, min(MAX_PER_CLASS, len(all_inact)))
    mols = act_mols + inact_mols
    labels = np.array([1]*len(act_mols) + [0]*len(inact_mols))
    
    smiles_list = [Chem.MolToSmiles(m) for m in mols]
    n = len(smiles_list)
    log(f"Loaded {n} molecules ({len(act_mols)} Actives, {len(inact_mols)} Inactives)")
    
    overall_best_auc = 0
    overall_best_alpha = 0
    overall_best_gamma = 0
    winning_F_raw = None
    winning_L_mat = None

    import time
    start_time = time.time()
    
    num_workers = len(alpha_values)
    log(f"Spawning {num_workers} parallel workers...")
    
    print("\\n" * num_workers)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for idx, alpha in enumerate(alpha_values):
            futures.append(executor.submit(process_alpha, alpha, smiles_list, labels, gamma_values, idx))
            
        for future in concurrent.futures.as_completed(futures):
            try:
                alpha, best_gamma, best_auc, F_raw, L_mat = future.result()
                log(f"[Worker Alpha={alpha}] Finished! Best Gamma: {best_gamma} (Screening AUC: {best_auc:.1f}%)")
                if best_auc > overall_best_auc:
                    overall_best_auc = best_auc
                    overall_best_alpha = alpha
                    overall_best_gamma = best_gamma
                    winning_F_raw = F_raw
                    winning_L_mat = L_mat
            except Exception as e:
                log(f"❌ ERROR in worker: {e}")
                
    print("\\n" * num_workers)
    
    end_time = time.time()
    log(f"✅ VIRTUAL SCREENING SWEEP COMPLETE! Total Time: {(end_time - start_time)/60:.1f} minutes.")
    log(f"🏆 WINNING PARAMETERS: Alpha={overall_best_alpha}, Gamma={overall_best_gamma} (Screening AUC: {overall_best_auc:.1f}%)")
    
    log("Generating CSV Export...")
    results = []
    final_sim_matrix = np.exp(-overall_best_gamma * winning_F_raw / np.maximum(winning_L_mat, 1))
    
    for i in range(n):
        for j in range(i+1, n):
            sim = final_sim_matrix[i, j]
            results.append({
                'Molecule_A': i, 'Molecule_B': j,
                'SMILES_A': smiles_list[i], 'SMILES_B': smiles_list[j],
                'Label_A': "Active" if labels[i]==1 else "Inactive", 
                'Label_B': "Active" if labels[j]==1 else "Inactive",
                'Similarity': round(sim, 4), 
                'Passed_Filter': bool(sim >= SIMILARITY_THRESHOLD)
            })
            
    df = pd.DataFrame(results).sort_values('Similarity', ascending=False)
    csv_name = f"{ds_name}_Parallel_Results.csv"
    df.to_csv(csv_name, index=False)
    log(f"💾 Saved {len(df)} pairwise comparisons to {csv_name}")
    
    passed_df = df[df['Passed_Filter'] == True]
    if len(passed_df) > 0:
        top_row = passed_df.iloc[0]
        # Just use the SMILES we safely stored!
        visualize_and_save_mapping(top_row['SMILES_A'], top_row['SMILES_B'], overall_best_alpha, ds_name, top_row['Similarity'])
    else:
        log("⚠️ No matches passed the threshold!")
        
    log("🎉 PIPELINE DONE!")
