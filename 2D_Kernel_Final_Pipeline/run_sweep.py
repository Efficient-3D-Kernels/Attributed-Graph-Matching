import os, time, sys
import numpy as np
from rdkit import Chem
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
from wcs_gnccp import WCS_GNCCP

def kabsch_rmsd(P, Q):
    if len(P) == 0: return 0.0
    Pc = P - np.mean(P, axis=0)
    Qc = Q - np.mean(Q, axis=0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    P_rot = Pc @ R.T
    return np.sqrt(np.mean(np.sum((P_rot - Qc)**2, axis=1)))

data_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'Data'))
active_path = os.path.join(data_dir, 'Aromatase_actives_new.sdf')
inactive_path = os.path.join(data_dir, 'Aromatase_inactives_new.sdf')

mols, labels = [], []
for m in Chem.SDMolSupplier(active_path, removeHs=True):
    if m: mols.append(m); labels.append(1)
    if len(labels) >= 50: break
inact = 0
for m in Chem.SDMolSupplier(inactive_path, removeHs=True):
    if m: mols.append(m); labels.append(0); inact += 1
    if inact >= 50: break

labels = np.array(labels)
n = len(mols)
our_data = [WCS_GNCCP.precompute_molecular_data(m) for m in mols]
coords = [m.GetConformer().GetPositions() for m in mols]

print("="*65)
print("HYPERPARAMETER SWEEP: ALPHA & GAMMA (100 Mols - Aromatase)")
print("="*65)
print(f"{'Alpha':<8} | {'Gamma':<8} | {'Accuracy (%)':<15} | {'AUC':<10}")
print("-" * 65)

best_acc = 0
best_params = {}

for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
    matcher = WCS_GNCCP(alpha=alpha, gamma=1.0, max_iter=8, fw_max_iter=8)
    
    F_obj_matrix = np.zeros((n, n))
    RMSD_matrix = np.zeros((n, n))
    L_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            score_2d, X_map = matcher.match_from_precomputed(our_data[i], our_data[j], return_map=True)
            
            M, N = our_data[i]['n_atoms'], our_data[j]['n_atoms']
            L = max(min(M, N) - 2, 1)
            
            score_safe = max(score_2d, 1e-12)
            F_obj = -np.log(score_safe)
            
            matched = np.where(X_map == 1)
            if len(matched[0]) > 0:
                rmsd = kabsch_rmsd(coords[i][matched[0]], coords[j][matched[1]])
            else:
                rmsd = 10.0
                
            F_obj_matrix[i, j] = F_obj_matrix[j, i] = F_obj
            RMSD_matrix[i, j] = RMSD_matrix[j, i] = rmsd
            L_matrix[i, j] = L_matrix[j, i] = L

    for gamma in [0.1, 0.5, 1.0, 2.0]:
        K_hybrid = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                # My fix above didn't divide F_obj by L!
                # Wait, earlier I removed the double / L inside the script.
                sim_2d = np.exp(-gamma * F_obj_matrix[i, j])
                sim_3d = np.exp(-gamma * RMSD_matrix[i, j])
                K_hybrid[i, j] = 0.5 * sim_2d + 0.5 * sim_3d
        
        eigvals = np.linalg.eigvalsh(K_hybrid)
        min_eig = np.min(eigvals)
        if min_eig < 0:
            K_hybrid += (np.abs(min_eig) + 1e-4) * np.eye(n)
        
        d = np.diag(K_hybrid)
        K_hybrid = K_hybrid / (np.sqrt(np.outer(d, d)) + 1e-8)
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        clf = SVC(kernel='precomputed', C=10.0, class_weight='balanced', probability=True)
        
        accs, aucs = [], []
        for tr, ts in skf.split(K_hybrid, labels):
            clf.fit(K_hybrid[np.ix_(tr, tr)], labels[tr])
            preds = clf.predict(K_hybrid[np.ix_(ts, tr)])
            probs = clf.predict_proba(K_hybrid[np.ix_(ts, tr)])[:, 1]
            accs.append(accuracy_score(labels[ts], preds))
            try: aucs.append(roc_auc_score(labels[ts], probs))
            except: aucs.append(0.5)
            
        mean_acc = np.mean(accs) * 100
        mean_auc = np.mean(aucs)
        
        print(f"{alpha:<8} | {gamma:<8} | {mean_acc:>5.1f}%          | {mean_auc:.3f}")
        
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_params = {'alpha': alpha, 'gamma': gamma}

print("="*65)
print(f"BEST CONFIGURATION: Alpha = {best_params.get('alpha')}, Gamma = {best_params.get('gamma')} -> Accuracy = {best_acc:.1f}%")
print("="*65)
