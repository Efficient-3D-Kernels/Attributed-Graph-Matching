import nbformat
import os

def create_notebook():
    nb = nbformat.v4.new_notebook()
    nb['metadata'] = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10.0'},
        'accelerator': 'GPU'
    }

    cells = []

    # ═══════════════════════════════════════════════════════════════════
    # CELL 0: Title
    # ═══════════════════════════════════════════════════════════════════
    cells.append(nbformat.v4.new_markdown_cell("""\
# WCS-GNCCP Hybrid 2D+3D Molecular Kernel — Full GPU Benchmark

**Hardware:** Kaggle T4x2 GPU  
**Datasets:** Aromatase, LBD, MMP, p53 (auto-downloaded from Google Drive)  
**Pipeline:** GNCCP Frank-Wolfe optimization → Hungarian discretization → SVM classification  
**Sweep:** Alpha ∈ {0.3, 0.5, 0.7, 0.9}, Gamma ∈ {0.1, 0.5, 1.0, 2.0}
"""))

    # ═══════════════════════════════════════════════════════════════════
    # CELL 1: Install + Imports
    # ═══════════════════════════════════════════════════════════════════
    cells.append(nbformat.v4.new_code_cell("""\
!pip install -q gdown rdkit

import os, sys, time, glob, random, warnings
import numpy as np
import torch
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from rdkit import Chem
from rdkit.Chem import AllChem
import gdown

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

# ── CUDA ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.backends.cudnn.benchmark = True
"""))

    # ═══════════════════════════════════════════════════════════════════
    # CELL 2: Download ALL files from Drive + filter paired datasets
    # ═══════════════════════════════════════════════════════════════════
    cells.append(nbformat.v4.new_code_cell("""\
# ══════════════════════════════════════════════════════════════
# STEP 1: Download ALL files from Google Drive shared folder
# ══════════════════════════════════════════════════════════════
DRIVE_FOLDER_ID = "1F05h8623pwLuN4NF_AMTa1ilzYBzIdaL"
DATA_DIR = "/kaggle/working/data"

if not os.path.exists(DATA_DIR) or len(glob.glob(os.path.join(DATA_DIR, "*.sdf"))) < 4:
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Downloading all SDF files from Google Drive...")
    gdown.download_folder(id=DRIVE_FOLDER_ID, output=DATA_DIR, quiet=False, remaining_ok=True)
    print("Done!")
else:
    print(f"Data already cached at {DATA_DIR}")

# List everything downloaded
all_sdf = sorted(glob.glob(os.path.join(DATA_DIR, "**/*.sdf", ), recursive=True))
# Also check root
all_sdf += sorted(glob.glob(os.path.join(DATA_DIR, "*.sdf")))
all_sdf = sorted(set(all_sdf))

print(f"\\nTotal SDF files found: {len(all_sdf)}")
for f in all_sdf:
    sz = os.path.getsize(f) / 1e6
    print(f"  {os.path.basename(f):>40s}  {sz:6.1f} MB")

# ══════════════════════════════════════════════════════════════
# STEP 2: Filter to PAIRED datasets only (need both actives + inactives)
#         Skip duplicates with (1) or (2) in the name
# ══════════════════════════════════════════════════════════════
paired = {}
for f in all_sdf:
    bn = os.path.basename(f)
    # Skip duplicate downloads
    if "(1)" in bn or "(2)" in bn:
        continue
    if "_actives_new.sdf" in bn:
        prefix = bn.replace("_actives_new.sdf", "")
        # Find matching inactives
        inact_candidates = [x for x in all_sdf if os.path.basename(x) == f"{prefix}_inactives_new.sdf"]
        if inact_candidates:
            paired[prefix] = {
                'actives': f,
                'inactives': inact_candidates[0],
            }

print(f"\\nPaired benchmark datasets: {sorted(paired.keys())}")
for name, paths in sorted(paired.items()):
    print(f"  {name}:")
    print(f"    Actives:   {os.path.basename(paths['actives'])}")
    print(f"    Inactives: {os.path.basename(paths['inactives'])}")
"""))

    # ═══════════════════════════════════════════════════════════════════
    # CELL 3: WCS-GNCCP Kernel (EXACT copy of wcs_gnccp.py logic on GPU)
    # ═══════════════════════════════════════════════════════════════════
    cells.append(nbformat.v4.new_code_cell("""\
# ══════════════════════════════════════════════════════════════
# WCS-GNCCP KERNEL — Exact PyTorch GPU port of wcs_gnccp.py
# ══════════════════════════════════════════════════════════════
class WCS_GNCCP_GPU:
    def __init__(self, alpha=0.7, gamma=0.5, max_iter=8, fw_max_iter=8, tol=1e-4):
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
        if c_max > c_min:
            C = (C - c_min) / (c_max - c_min + 1e-8)
        else:
            C = C / (c_max + 1e-8)
        
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
            if torch.all(X.sum(dim=1) <= 1.0 + 1e-6) and torch.all(X.sum(dim=0) <= 1.0 + 1e-6):
                break
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
    # CELL 4: Master Loop (Memory Efficient: Load -> Precompute -> Sweep -> Flush)
    # ═══════════════════════════════════════════════════════════════════
    cells.append(nbformat.v4.new_code_cell("""\
# ══════════════════════════════════════════════════════════════
# MASTER PIPELINE (Memory-Efficient Dataset Iteration)
# ══════════════════════════════════════════════════════════════
import gc

N_PER_CLASS = 50  # Set higher (e.g. 100, 200) for final benchmarks
alpha_values = [0.3, 0.5, 0.7, 0.9]
gamma_values = [0.1, 0.5, 1.0, 2.0]

all_results = {}

for ds_name, paths in sorted(paired.items()):
    print(f"\\n{'='*75}")
    print(f"🚀 PROCESSING DATASET: {ds_name}")
    print(f"{'='*75}")
    
    # 1. LOAD JUST THIS DATASET
    act_mols = [m for m in Chem.SDMolSupplier(paths['actives'], removeHs=True) if m is not None]
    inact_mols = [m for m in Chem.SDMolSupplier(paths['inactives'], removeHs=True) if m is not None]
    a = random.sample(act_mols, min(N_PER_CLASS, len(act_mols)))
    b = random.sample(inact_mols, min(N_PER_CLASS, len(inact_mols)))
    mols = a + b
    labels = np.array([1]*len(a) + [0]*len(b))
    n = len(mols)
    print(f"Loaded {len(a)} actives + {len(b)} inactives = {n} total")
    
    # 2. PRECOMPUTE ON GPU
    matcher = WCS_GNCCP_GPU(max_iter=8, fw_max_iter=8)  # Dummy init for precompute
    precomp = [matcher.precompute(m) for m in tqdm(mols, desc="Precomputing GPU Tensors")]
    
    # 3. SWEEP ALPHA & GAMMA
    print(f"\\n{'Alpha':>6} {'Gamma':>6} | {'Acc (%)':>8} {'AUC':>8} | {'Time':>8} {'ms/pair':>8}")
    print("-" * 75)
    
    best_acc = 0
    best_cfg = ""
    
    total_pairs = n * (n + 1) // 2
    pairs_list = [(i, j) for i in range(n) for j in range(i, n)]
    
    import concurrent.futures

    for alpha in alpha_values:
        matcher = WCS_GNCCP_GPU(alpha=alpha, gamma=1.0, max_iter=20, fw_max_iter=10)
        F_raw = np.zeros((n, n))
        L_mat = np.zeros((n, n))
        
        # Worker function for parallel execution
        def process_pair(idx_pair):
            i, j = idx_pair
            d1, d2 = precomp[i], precomp[j]
            A_G, A_H = d1['A'], d2['A']
            C = matcher.build_cost_matrix(d1['f'], d2['f'])
            M_i, N_j = d1['n'], d2['n']
            L = max(min(M_i, N_j) - 2, 1)
            
            X = matcher.gnccp_optimize(A_G, A_H, C, L)
            f_val = matcher.compute_F(X, A_G, A_H, C).item()
            return i, j, f_val, L
        
        t0 = time.time()
        
        # Spin up threads to maximize CPU and GPU overlap
        workers = min(32, os.cpu_count() * 4)  # Oversubscribe to keep GPU fed
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_pair, p) for p in pairs_list]
            
            with tqdm(total=total_pairs, desc=f"Alpha={alpha} Kernel", leave=False) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    i, j, f_val, L = future.result()
                    F_raw[i, j] = F_raw[j, i] = f_val
                    L_mat[i, j] = L_mat[j, i] = L
                    pbar.update(1)
        
        kernel_time = time.time() - t0
        total_pairs = n * (n + 1) // 2
        ms = kernel_time / total_pairs * 1000
        
        for gamma in gamma_values:
            K = np.exp(-gamma * F_raw / np.maximum(L_mat, 1))
            
            eigvals = np.linalg.eigvalsh(K)
            min_eig = np.min(eigvals)
            if min_eig < 0: K_psd = K + (np.abs(min_eig) + 1e-4) * np.eye(n)
            else: K_psd = K.copy()
            
            d = np.diag(K_psd)
            K_psd = K_psd / (np.sqrt(np.outer(d, d)) + 1e-8)
            
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            clf = SVC(kernel='precomputed', C=10.0, class_weight='balanced', probability=True)
            accs, aucs = [], []
            for tr, ts in skf.split(K_psd, labels):
                clf.fit(K_psd[np.ix_(tr, tr)], labels[tr])
                preds = clf.predict(K_psd[np.ix_(ts, tr)])
                probs = clf.predict_proba(K_psd[np.ix_(ts, tr)])[:, 1]
                accs.append(accuracy_score(labels[ts], preds))
                try: aucs.append(roc_auc_score(labels[ts], probs))
                except: aucs.append(0.5)
            
            acc = np.mean(accs) * 100
            auc = np.mean(aucs)
            print(f"{alpha:>6.1f} {gamma:>6.1f} | {acc:>7.1f}% {auc:>7.3f} | {kernel_time:>7.1f}s {ms:>7.1f}")
            if acc > best_acc:
                best_acc, best_cfg = acc, f"alpha={alpha}, gamma={gamma}"
                
    all_results[ds_name] = {'acc': best_acc, 'cfg': best_cfg}
    
    # 4. MEMORY FLUSH (Crucial for preventing OOM on Kaggle)
    del act_mols, inact_mols, a, b, mols, labels
    del precomp, F_raw, L_mat, K, K_psd, matcher
    gc.collect()
    torch.cuda.empty_cache()
    print(">> Flushed RAM & VRAM for next dataset.")

print("\\n" + "=" * 70)
print("🏆 FINAL BEST RESULTS PER DATASET")
print("=" * 70)
for ds, r in sorted(all_results.items()):
    print(f"  {ds:<12s}: {r['acc']:5.1f}%  ({r['cfg']})")
print("=" * 70)
"""))

    # ═══════════════════════════════════════════════════════════════════
    # CELL 7: Time Complexity Analysis
    # ═══════════════════════════════════════════════════════════════════
    cells.append(nbformat.v4.new_code_cell("""\
# ══════════════════════════════════════════════════════════════
# STEP 7: Empirical Time Complexity Analysis
# ══════════════════════════════════════════════════════════════
test_ds = sorted(PRECOMPUTED.keys())[0]
test_data = PRECOMPUTED[test_ds]
matcher = WCS_GNCCP_GPU(alpha=0.7, gamma=0.5, max_iter=8, fw_max_iter=8)

print("=" * 60)
print("TIME COMPLEXITY (GPU)")
print("=" * 60)
print(f"{'N':>6} | {'Pairs':>8} | {'Time (s)':>10} | {'ms/pair':>8}")
print("-" * 60)

for N in [10, 25, 50, min(100, len(test_data))]:
    sub = test_data[:N]
    t0 = time.time()
    for i in range(N):
        for j in range(i, N):
            matcher.match(sub[i], sub[j])
    elapsed = time.time() - t0
    pairs = N * (N + 1) // 2
    print(f"{N:>6} | {pairs:>8} | {elapsed:>10.2f} | {elapsed/pairs*1000:>7.2f}")

print("=" * 60)
"""))

    # ═══════════════════════════════════════════════════════════════════
    # Assemble
    # ═══════════════════════════════════════════════════════════════════
    nb['cells'] = cells

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Kaggle_GPU_Benchmark.ipynb')
    with open(out, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Created: {out}")

if __name__ == "__main__":
    create_notebook()