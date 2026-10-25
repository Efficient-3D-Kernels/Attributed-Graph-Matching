import os
import nbformat

def create_notebook():
    nb = nbformat.v4.new_notebook()
    cells = []

    # CELL 1: Setup & Install
    cells.append(nbformat.v4.new_code_cell("""\
# ==============================================================================
# 1. KERNEL SETUP & INSTALLATION
# Run this cell first! It will install all required packages into whatever 
# local Python Kernel you are currently using.
# ==============================================================================
!pip install -q rdkit pandas scikit-learn matplotlib tqdm torch scipy

import os, sys, time, glob, random, warnings
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from tqdm.notebook import tqdm
from scipy.optimize import linear_sum_assignment
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.cm as cm
import gc

warnings.filterwarnings('ignore')

def log(msg):
    t = datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] {msg}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f"Pipeline Initialized! Using Device: {device}")
"""))

    # CELL 2: Config
    cells.append(nbformat.v4.new_code_cell("""\
# ==============================================================================
# 2. CONFIGURATION
# ==============================================================================
SIMILARITY_THRESHOLD = 0.20  

# Hyperparameters to sweep
alpha_values = [0.3, 0.5, 0.7, 0.9]
gamma_values = [0.1, 0.5, 1.0, 2.0]
"""))

    # CELL 3: Data Load & ETA 
    cells.append(nbformat.v4.new_code_cell("""\
# ==============================================================================
# 3. LOAD LOCAL DATA
# ==============================================================================
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

if len(paired) == 0:
    log(f"⚠️ ERROR: No paired datasets found in {DATA_DIR}!")
else:
    log(f"Found {len(paired)} dataset targets locally!")
    
    # Pick the first target
    ds_name, paths = list(sorted(paired.items()))[0]
    log(f"\\nTARGET SELECTED: {ds_name}")
    
    # --- BALANCED SAMPLING FOR FAST RESULTS ---
    MAX_PER_CLASS = 100 
    
    all_act = [m for m in Chem.SDMolSupplier(paths['actives'], removeHs=True) if m is not None]
    all_inact = [m for m in Chem.SDMolSupplier(paths['inactives'], removeHs=True) if m is not None]
    
    act_mols = random.sample(all_act, min(MAX_PER_CLASS, len(all_act)))
    inact_mols = random.sample(all_inact, min(MAX_PER_CLASS, len(all_inact)))
    
    mols = act_mols + inact_mols
    labels = np.array([1]*len(act_mols) + [0]*len(inact_mols))
    n = len(mols)
    total_pairs = n * (n + 1) // 2
    
    log(f"Loaded {n} molecules ({len(act_mols)} Actives, {len(inact_mols)} Inactives)")
    log(f"Total pairwise comparisons to compute per Alpha: {total_pairs}")
    
    est_seconds_per_alpha = total_pairs / 25
    total_est_minutes = (est_seconds_per_alpha * len(alpha_values)) / 60
    log(f"⏳ ESTIMATED TIME: ~{total_est_minutes:.1f} minutes for the full sweep.")
"""))

    # CELL 4: Class
    cells.append(nbformat.v4.new_code_cell("""\
class WCS_GNCCP_GPU:
    def __init__(self, alpha=0.7, gamma=1.0, max_iter=8, fw_max_iter=8, tol=1e-4):
        self.alpha, self.gamma, self.max_iter, self.fw_max_iter, self.tol = alpha, gamma, max_iter, fw_max_iter, tol
    
    @staticmethod
    def extract_features(mol):
        atoms = list(mol.GetAtoms())
        hybr_map = {'SP': 1, 'SP2': 2, 'SP3': 3, 'SP3D': 4, 'SP3D2': 5}
        chiral_map = {'CHI_UNSPECIFIED': 0, 'CHI_TETRAHEDRAL_CW': 1, 'CHI_TETRAHEDRAL_CCW': 2}
        return {
            'atomic_num': torch.tensor([a.GetAtomicNum() for a in atoms], dtype=torch.float32, device=device),
            'degree':     torch.tensor([a.GetDegree() for a in atoms], dtype=torch.float32, device=device),
            'in_ring':    torch.tensor([float(a.IsInRing()) for a in atoms], dtype=torch.float32, device=device),
            'aromatic':   torch.tensor([float(a.GetIsAromatic()) for a in atoms], dtype=torch.float32, device=device),
            'mass':       torch.tensor([a.GetMass() for a in atoms], dtype=torch.float32, device=device),
            'charge':     torch.tensor([float(a.GetFormalCharge()) for a in atoms], dtype=torch.float32, device=device),
            'hybr':       torch.tensor([hybr_map.get(str(a.GetHybridization()), 0) for a in atoms], dtype=torch.float32, device=device),
            'num_hs':     torch.tensor([float(a.GetTotalNumHs()) for a in atoms], dtype=torch.float32, device=device),
            'chiral':     torch.tensor([chiral_map.get(str(a.GetChiralTag()), 0) for a in atoms], dtype=torch.float32, device=device),
            'isotope':    torch.tensor([float(a.GetIsotope()) for a in atoms], dtype=torch.float32, device=device),
        }
    
    @staticmethod
    def build_adjacency(mol):
        n = mol.GetNumAtoms()
        A = torch.zeros((n, n), dtype=torch.float32, device=device)
        bt_map = {Chem.BondType.SINGLE: 1.0, Chem.BondType.DOUBLE: 2.0, Chem.BondType.TRIPLE: 3.0, Chem.BondType.AROMATIC: 1.5}
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            w = bt_map.get(bond.GetBondType(), 1.0)
            if bond.IsInRing(): w += 0.1
            if bond.GetIsConjugated(): w += 0.1
            A[i, j] = w; A[j, i] = w
        return A
    
    @staticmethod
    def build_cost_matrix(f1, f2):
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
    
    @staticmethod
    def project(X, L):
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
        X = torch.ones((M, N), dtype=torch.float32, device=device) * (L / (M * N))
        for zeta in torch.linspace(1.0, -1.0, self.max_iter, device=device):
            for t in range(self.fw_max_iter):
                grad_F = self.compute_grad_F(X, A_G, A_H, C)
                grad = (1 - zeta) * grad_F + 2 * zeta * X if zeta >= 0 else (1 + zeta) * grad_F + 2 * zeta * X
                grad = torch.nan_to_num(grad, nan=0.0, posinf=1e6, neginf=-1e6)
                grad_np = grad.cpu().numpy()
                dim = M + N - L
                g_min = float(np.min(grad_np))
                g_shift = grad_np - g_min + 1e-9 if g_min < 0 else grad_np
                if L >= min(M, N):
                    ri, ci = linear_sum_assignment(grad_np)
                else:
                    aug = np.full((dim, dim), 1e6, dtype=np.float32)
                    aug[:M, :N] = g_shift
                    aug[:M, N:] = 0.0
                    aug[M:, :N] = 0.0
                    ri, ci = linear_sum_assignment(aug)
                Y = torch.zeros((M, N), dtype=torch.float32, device=device)
                for r, c in zip(ri, ci):
                    if r < M and c < N: Y[r, c] = 1.0
                D_fw = X - Y
                if torch.sum(grad * D_fw) < self.tol: break
                X = X - (2.0 / (t + 2.0)) * D_fw
        
        X_bin = torch.zeros((M, N), dtype=torch.float32, device=device)
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
"""))

    # CELL 5: Execution
    cells.append(nbformat.v4.new_code_cell("""\
# ==============================================================================
# 4. EXECUTE FULL SWEEP
# ==============================================================================
log("Precomputing GPU Features for all molecules...")
matcher = WCS_GNCCP_GPU()
precomp = [matcher.precompute(m) for m in tqdm(mols)]

best_acc = 0
best_alpha = 0
best_gamma = 0
best_F_raw = None
best_L_mat = None

t_start = time.time()

for alpha in alpha_values:
    log(f"▶️ Starting Sweep for Alpha = {alpha}...")
    matcher = WCS_GNCCP_GPU(alpha=alpha, gamma=1.0, max_iter=8, fw_max_iter=8)
    F_raw = np.zeros((n, n))
    L_mat = np.zeros((n, n))
    
    with tqdm(total=total_pairs, desc=f"Calculating Pairs (Alpha={alpha})") as pbar:
        for i in range(n):
            for j in range(i, n):
                d1, d2 = precomp[i], precomp[j]
                C = matcher.build_cost_matrix(d1['f'], d2['f'])
                L = max(min(d1['n'], d2['n']) - 2, 1)
                X = matcher.gnccp_optimize(d1['A'], d2['A'], C, L)
                
                F_raw[i, j] = F_raw[j, i] = matcher.compute_F(X, d1['A'], d2['A'], C).item()
                L_mat[i, j] = L_mat[j, i] = L
                pbar.update(1)
            
    log(f"Alpha {alpha} matrix calculated. Running SVM Validation across Gammas...")
    for gamma in gamma_values:
        K = np.exp(-gamma * F_raw / np.maximum(L_mat, 1))
        eigvals = np.linalg.eigvalsh(K)
        if np.min(eigvals) < 0: K_psd = K + (np.abs(np.min(eigvals)) + 1e-4) * np.eye(n)
        else: K_psd = K.copy()
        d = np.diag(K_psd)
        K_psd = K_psd / (np.sqrt(np.outer(d, d)) + 1e-8)
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        clf = SVC(kernel='precomputed', C=10.0, class_weight='balanced')
        accs = []
        for tr, ts in skf.split(K_psd, labels):
            clf.fit(K_psd[np.ix_(tr, tr)], labels[tr])
            accs.append(accuracy_score(labels[ts], clf.predict(K_psd[np.ix_(ts, tr)])))
        
        acc = np.mean(accs) * 100
        log(f"   ↳ Gamma = {gamma}: Accuracy = {acc:.1f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_alpha, best_gamma = alpha, gamma
            best_F_raw, best_L_mat = F_raw.copy(), L_mat.copy()

t_end = time.time()
log(f"✅ SWEEP COMPLETE! Total Time: {(t_end - t_start)/60:.1f} minutes.")
log(f"🏆 OVERALL BEST CONFIGURATION: Alpha={best_alpha}, Gamma={best_gamma} (Accuracy: {best_acc:.1f}%)")
"""))

    # CELL 6: Export & Visualize
    cells.append(nbformat.v4.new_code_cell("""\
# ==============================================================================
# 5. EXPORT CSV AND VISUALIZE TOP MATCH
# ==============================================================================
log("Generating Final Similarity Scores CSV...")
results = []
final_sim_matrix = np.exp(-best_gamma * best_F_raw / np.maximum(best_L_mat, 1))

for i in range(n):
    for j in range(i+1, n):
        sim = final_sim_matrix[i, j]
        results.append({
            'Molecule_A': i, 'Molecule_B': j,
            'Label_A': "Active" if labels[i]==1 else "Inactive", 
            'Label_B': "Active" if labels[j]==1 else "Inactive",
            'Similarity': round(sim, 4), 
            'Passed_Filter': bool(sim >= SIMILARITY_THRESHOLD)
        })
        
df = pd.DataFrame(results).sort_values('Similarity', ascending=False)
csv_name = f"{ds_name}_Full_Results.csv"
df.to_csv(csv_name, index=False)
log(f"💾 Saved {len(df)} pairwise comparisons to {csv_name}")

def visualize_mapping(mol_A, mol_B, alpha, title="Bijective Mapping"):
    m = WCS_GNCCP_GPU(alpha=alpha, gamma=1.0)
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
        dict_A[a] = [rgba[:3]]; dict_B[b] = [rgba[:3]]
        
    log(f"Visualizing: {title}")
    img = Draw.MolsToGridImage(
        [mol_A, mol_B], molsPerRow=2, subImgSize=(500, 500),
        highlightAtomLists=[mapped_A, mapped_B], highlightAtomColors=[dict_A, dict_B],
        legends=["Molecule A", "Molecule B"]
    )
    display(img)

passed_df = df[df['Passed_Filter'] == True]
if len(passed_df) > 0:
    top_row = passed_df.iloc[0]
    idx_A, idx_B = int(top_row['Molecule_A']), int(top_row['Molecule_B'])
    visualize_mapping(mols[idx_A], mols[idx_B], best_alpha, 
                      title=f"Top Match: {top_row['Similarity']} ({top_row['Label_A']} vs {top_row['Label_B']})")
else:
    log("⚠️ No matches passed the threshold!")
"""))

    nb['cells'] = cells
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Full_Dataset_Sweep_Local.ipynb')
    with open(out, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Created Notebook: {out}")

if __name__ == "__main__":
    create_notebook()
