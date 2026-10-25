import os
import nbformat

def create_final_notebook():
    nb = nbformat.v4.new_notebook()
    cells = []

    # ═══════════════════════════════════════════════════════════════════
    # CELL 1: Install + Imports
    # ═══════════════════════════════════════════════════════════════════
    cells.append(nbformat.v4.new_code_cell("""\
!pip install -q gdown rdkit pandas matplotlib

import os, sys, time, glob, random, warnings
import numpy as np
import pandas as pd
import torch
import concurrent.futures
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import matplotlib.cm as cm
import matplotlib.colors as mcolors

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
"""))

    # ═══════════════════════════════════════════════════════════════════
    # CELL 2: Configuration (Alpha, Gamma, Threshold)
    # ═══════════════════════════════════════════════════════════════════
    cells.append(nbformat.v4.new_code_cell("""\
# ==============================================================================
# 🛑 CONFIGURATION: UPDATE THESE AFTER YOUR PARAMETER SWEEP FINISHES
# ==============================================================================

# The optimal parameters found from the Kaggle Benchmark Sweep
OPTIMAL_ALPHA = 0.7  
OPTIMAL_GAMMA = 1.0  

# The "Relaxed Boundary" Threshold (e.g. 0.20 means drop the worst 80%)
SIMILARITY_THRESHOLD = 0.20  

# Number of molecules per class to run through the pipeline
N_PER_CLASS = 20  
"""))

    # ═══════════════════════════════════════════════════════════════════
    # CELL 3: Data Download
    # ═══════════════════════════════════════════════════════════════════
    cells.append(nbformat.v4.new_code_cell("""\
import gdown
DATA_DIR = './dataset'
DRIVE_FOLDER_ID = '1F05h8623pwLuN4NF_AMTa1ilzYBzIdaL'

if not os.path.exists(DATA_DIR) or len(glob.glob(os.path.join(DATA_DIR, "*.sdf"))) < 4:
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Downloading SDF files...")
    gdown.download_folder(id=DRIVE_FOLDER_ID, output=DATA_DIR, quiet=False, remaining_ok=True)

# Pair up the datasets
sdf_files = glob.glob(os.path.join(DATA_DIR, "*.sdf"))
paired = {}
for f in sdf_files:
    fname = os.path.basename(f)
    if "_actives_new.sdf" in fname:
        base = fname.replace("_actives_new.sdf", "")
        inact = f.replace("_actives_new.sdf", "_inactives_new.sdf")
        if inact in sdf_files:
            paired[base] = {'actives': f, 'inactives': inact}

print(f"Found {len(paired)} paired datasets!")
# For this pipeline, we will just use the first dataset as a demonstration
TARGET_DATASET = sorted(paired.keys())[0]
paths = paired[TARGET_DATASET]

act_mols = [m for m in Chem.SDMolSupplier(paths['actives'], removeHs=True) if m is not None]
inact_mols = [m for m in Chem.SDMolSupplier(paths['inactives'], removeHs=True) if m is not None]

a = random.sample(act_mols, min(N_PER_CLASS, len(act_mols)))
b = random.sample(inact_mols, min(N_PER_CLASS, len(inact_mols)))
all_mols = a + b
all_labels = ["Active"]*len(a) + ["Inactive"]*len(b)

print(f"Loaded {TARGET_DATASET}: {len(a)} actives + {len(b)} inactives.")
"""))

    # ═══════════════════════════════════════════════════════════════════
    # CELL 4: WCS_GNCCP_GPU Class
    # ═══════════════════════════════════════════════════════════════════
    cells.append(nbformat.v4.new_code_cell("""\
class WCS_GNCCP_GPU:
    def __init__(self, alpha=0.7, gamma=1.0, max_iter=20, fw_max_iter=10, tol=1e-4):
        self.alpha = alpha
        self.gamma = gamma
        self.max_iter = max_iter
        self.fw_max_iter = fw_max_iter
        self.tol = tol
    
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
        bt_map = {Chem.BondType.SINGLE: 1.0, Chem.BondType.DOUBLE: 2.0,
                  Chem.BondType.TRIPLE: 3.0, Chem.BondType.AROMATIC: 1.5}
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            w = bt_map.get(bond.GetBondType(), 1.0)
            if bond.IsInRing(): w += 0.1
            if bond.GetIsConjugated(): w += 0.1
            A[i, j] = w
            A[j, i] = w
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
        
        mismatch = (f1['atomic_num'].unsqueeze(1) != f2['atomic_num'].unsqueeze(0))
        C[mismatch] = 50.0
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
        U = torch.outer(r, r)
        R = (U * A_G) - (X @ A_H @ X.T)
        RA_X = (R * A_G) @ X
        term1 = RA_X.sum(dim=1, keepdim=True)
        term2 = R @ X @ A_H
        grad_struct = 4.0 * term1 - 4.0 * term2
        return self.alpha * grad_struct + (1.0 - self.alpha) * C
    
    def compute_F(self, X, A_G, A_H, C):
        r = X.sum(dim=1)
        U = torch.outer(r, r)
        diff = (U * A_G) - (X @ A_H @ X.T)
        struct = torch.sum(diff ** 2)
        app = torch.sum(C * X)
        return self.alpha * struct + (1.0 - self.alpha) * app
    
    @torch.no_grad()
    def gnccp_optimize(self, A_G, A_H, C, L):
        M, N = A_G.shape[0], A_H.shape[0]
        X = torch.ones((M, N), dtype=torch.float32, device=device) * (L / (M * N))
        zetas = torch.linspace(1.0, -1.0, self.max_iter, device=device)
        for zeta in zetas:
            for t in range(self.fw_max_iter):
                grad_F = self.compute_grad_F(X, A_G, A_H, C)
                if zeta >= 0: grad = (1 - zeta) * grad_F + 2 * zeta * X
                else:         grad = (1 + zeta) * grad_F + 2 * zeta * X
                grad = torch.nan_to_num(grad, nan=0.0, posinf=1e6, neginf=-1e6)
                
                grad_cpu = grad.cpu().numpy()
                dim = M + N - L
                g_min = float(np.min(grad_cpu))
                g_shift = grad_cpu - g_min + 1e-9 if g_min < 0 else grad_cpu
                if L >= min(M, N):
                    ri, ci = linear_sum_assignment(grad_cpu)
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
                fw_gap = torch.sum(grad * D_fw)
                if fw_gap < self.tol: break
                
                lam = 2.0 / (t + 2.0)
                X = X - lam * D_fw
        
        X_cpu = (-X).cpu().numpy()
        ri, ci = linear_sum_assignment(X_cpu)
        X_bin = torch.zeros((M, N), dtype=torch.float32, device=device)
        matched = 0
        for r, c in zip(ri, ci):
            if r < M and c < N and matched < L:
                X_bin[r, c] = 1.0
                matched += 1
        return X_bin
    
    def precompute(self, mol):
        A = self.build_adjacency(mol)
        # Shared scale / unnormalized graph topology
        pass
        return {
            'A': A,
            'f': self.extract_features(mol),
            'n': mol.GetNumAtoms(),
        }
"""))

    # ═══════════════════════════════════════════════════════════════════
    # CELL 5: Matrix calculation & CSV Export
    # ═══════════════════════════════════════════════════════════════════
    cells.append(nbformat.v4.new_code_cell("""\
# ==============================================================================
# PHASE 1: COMPUTE SCORES AND EXPORT CSV
# ==============================================================================
matcher = WCS_GNCCP_GPU(alpha=OPTIMAL_ALPHA, gamma=OPTIMAL_GAMMA)
precomp = [matcher.precompute(m) for m in tqdm(all_mols, desc="Precomputing")]

n = len(all_mols)
pairs_list = [(i, j) for i in range(n) for j in range(i+1, n)]  # Don't compare to self

def process_pair(idx_pair):
    i, j = idx_pair
    d1, d2 = precomp[i], precomp[j]
    C = matcher.build_cost_matrix(d1['f'], d2['f'])
    L = max(min(d1['n'], d2['n']) - 2, 1)
    X = matcher.gnccp_optimize(d1['A'], d2['A'], C, L)
    f_val = matcher.compute_F(X, d1['A'], d2['A'], C).item()
    
    # Calculate similarity score instantly
    sim_score = np.exp(-OPTIMAL_GAMMA * f_val / L)
    return i, j, sim_score

print(f"Calculating {len(pairs_list)} pairwise similarities...")
results = []

workers = min(32, os.cpu_count() * 4)
with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
    futures = [executor.submit(process_pair, p) for p in pairs_list]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(pairs_list), desc="Scoring"):
        i, j, sim_score = future.result()
        
        passed_filter = bool(sim_score >= SIMILARITY_THRESHOLD)
        
        results.append({
            'Molecule_A_Index': i,
            'Molecule_B_Index': j,
            'Label_A': all_labels[i],
            'Label_B': all_labels[j],
            'Similarity_Score': round(sim_score, 4),
            'Passed_2D_Filter': passed_filter
        })

df = pd.DataFrame(results)
# Sort by highest similarity
df = df.sort_values('Similarity_Score', ascending=False)
df.to_csv("Screening_Results.csv", index=False)

print("\\n✅ Saved 'Screening_Results.csv'!")
print(f"Total Pairs Evaluated: {len(df)}")
print(f"Pairs Passed Filter (Sent to 3D): {df['Passed_2D_Filter'].sum()} ({df['Passed_2D_Filter'].mean()*100:.1f}%)")
print(f"Pairs Rejected by 2D Kernel: {len(df) - df['Passed_2D_Filter'].sum()}")
"""))

    # ═══════════════════════════════════════════════════════════════════
    # CELL 6: Bijective Mapping Visualization
    # ═══════════════════════════════════════════════════════════════════
    cells.append(nbformat.v4.new_code_cell("""\
# ==============================================================================
# PHASE 2: BIJECTIVE ATOM MAPPING VISUALIZATION
# ==============================================================================
def visualize_mapping(mol_A, mol_B, alpha, title="Bijective Mapping"):
    # Recompute the exact mapping matrix X
    m = WCS_GNCCP_GPU(alpha=alpha, gamma=1.0)
    d1, d2 = m.precompute(mol_A), m.precompute(mol_B)
    C = m.build_cost_matrix(d1['f'], d2['f'])
    L = max(min(d1['n'], d2['n']) - 2, 1)
    X_bin = m.gnccp_optimize(d1['A'], d2['A'], C, L).cpu().numpy()
    
    # Extract mapped indices
    mapped_A = []
    mapped_B = []
    for i in range(X_bin.shape[0]):
        for j in range(X_bin.shape[1]):
            if X_bin[i, j] > 0.5:
                mapped_A.append(i)
                mapped_B.append(j)
                
    # Generate unique colors for each mapped pair
    colors = cm.get_cmap('hsv', len(mapped_A))
    
    dict_A = {}
    dict_B = {}
    for idx, (a, b) in enumerate(zip(mapped_A, mapped_B)):
        rgba = colors(idx)
        # RDKit expects RGB tuples for colors
        dict_A[a] = [rgba[:3]]
        dict_B[b] = [rgba[:3]]
        
    print(f"\\n{'='*50}")
    print(f"Visualizing mapping for: {title}")
    print(f"Mapped {len(mapped_A)} atoms!")
    print(f"{'='*50}")
    
    img = Draw.MolsToGridImage(
        [mol_A, mol_B], 
        molsPerRow=2, 
        subImgSize=(400, 400),
        highlightAtomLists=[mapped_A, mapped_B],
        highlightAtomColors=[dict_A, dict_B],
        legends=["Molecule A", "Molecule B"]
    )
    display(img)

# Let's randomly pick 3 pairs that PASSED the filter to visualize
passed_df = df[df['Passed_2D_Filter'] == True]
if len(passed_df) > 0:
    samples = passed_df.sample(min(3, len(passed_df)))
    for _, row in samples.iterrows():
        i = row['Molecule_A_Index']
        j = row['Molecule_B_Index']
        visualize_mapping(
            all_mols[i], 
            all_mols[j], 
            OPTIMAL_ALPHA, 
            title=f"Score: {row['Similarity_Score']} ({row['Label_A']} vs {row['Label_B']})"
        )
else:
    print("No pairs passed the threshold to visualize!")
"""))

    # ═══════════════════════════════════════════════════════════════════
    # CELL 7: Phase 3 Placeholder
    # ═══════════════════════════════════════════════════════════════════
    cells.append(nbformat.v4.new_code_cell("""\
# ==============================================================================
# PHASE 3: 3D ALIGNMENT (RMSD) ON FILTERED CANDIDATES
# ==============================================================================
# This is where we will take only the molecules that PASSED the 2D filter
# and run rdMolAlign.GetO3A() or similar to calculate the physical 3D distance.

print("Candidates remaining for 3D simulation:")
print(f"{len(passed_df)} out of {len(df)} total pairs.")
print("The 2D Filter successfully eliminated", len(df) - len(passed_df), "garbage matches from the pipeline!")

# Example skeleton for next steps:
'''
from rdkit.Chem import rdMolAlign

for _, row in passed_df.iterrows():
    mol_A = all_mols[row['Molecule_A_Index']]
    mol_B = all_mols[row['Molecule_B_Index']]
    
    # Generate 3D conformers (if not already 3D)
    # AllChem.EmbedMolecule(mol_A)
    # AllChem.EmbedMolecule(mol_B)
    
    # Calculate 3D RMSD
    # pyO3A = rdMolAlign.GetO3A(mol_A, mol_B)
    # rmsd = pyO3A.Align()
    # print(f"3D RMSD: {rmsd}")
'''
print("\\n[Phase 3 Code to be implemented here]")
"""))

    nb['cells'] = cells

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Final_Screening_Pipeline.ipynb')
    with open(out, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Created Notebook: {out}")

if __name__ == "__main__":
    create_final_notebook()
