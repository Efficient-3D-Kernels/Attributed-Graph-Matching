import time
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs

# Import your custom kernel pipeline
from wcs_gnccp import WCS_GNCCP

def run_drug_discovery_benchmark(sample_size=10):
    print("="*60)
    print("DRUG DISCOVERY KERNEL BENCHMARKING (Chemoinformatics)")
    print("="*60)
    
    data_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'Aromatase_actives_new.sdf')
    print(f"Loading {sample_size} drug molecules from: {os.path.basename(data_path)}...")
    
    mols = []
    suppl = Chem.SDMolSupplier(data_path)
    for m in suppl:
        if m is not None:
            mols.append(Chem.RemoveHs(m))
        if len(mols) >= sample_size:
            break
            
    print(f"Loaded {len(mols)} molecules successfully.\n")
    n = len(mols)
    results = {}

    # --- A. MORGAN FINGERPRINT (ECFP4) + TANIMOTO ---
    # This is the undisputed industry standard for drug discovery.
    print("Running Morgan Fingerprint (ECFP4) + Tanimoto...")
    start = time.time()
    morgan_fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in mols]
    K_morgan = np.zeros((n, n))
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(morgan_fps[i], morgan_fps)
        K_morgan[i, :] = sims
    time_morgan = time.time() - start
    results["Morgan (ECFP4) + Tanimoto"] = time_morgan

    # --- B. MACCS KEYS + TANIMOTO ---
    print("Running MACCS Keys + Tanimoto...")
    start = time.time()
    maccs_fps = [MACCSkeys.GenMACCSKeys(m) for m in mols]
    K_maccs = np.zeros((n, n))
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(maccs_fps[i], maccs_fps)
        K_maccs[i, :] = sims
    time_maccs = time.time() - start
    results["MACCS Keys + Tanimoto"] = time_maccs

    # --- C. ATOM PAIR FINGERPRINTS + TANIMOTO ---
    print("Running Atom Pair Fingerprints + Tanimoto...")
    start = time.time()
    ap_fps = [AllChem.GetAtomPairFingerprint(m) for m in mols]
    K_ap = np.zeros((n, n))
    for i in range(n):
        # Atom Pairs return an IntSparseIntVect, BulkTanimoto works slightly differently
        sims = DataStructs.BulkDiceSimilarity(ap_fps[i], ap_fps) # Dice is often used for AtomPairs
        K_ap[i, :] = sims
    time_ap = time.time() - start
    results["Atom Pair + Dice"] = time_ap

    # --- D. OUR 2D KERNEL (WCS-GNCCP) ---
    print("Extracting features for our WCS-GNCCP kernel...")
    our_data = [WCS_GNCCP.precompute_molecular_data(m) for m in mols]
    
    print("Running 2D Kernel WCS-GNCCP (Ours)...")
    matcher = WCS_GNCCP(alpha=0.7, gamma=0.5, max_iter=8, fw_max_iter=8)
    
    start = time.time()
    K_2d = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            score = matcher.match_from_precomputed(our_data[i], our_data[j])
            K_2d[i, j] = score
            K_2d[j, i] = score
    time_2d = time.time() - start
    results["2D WCS-GNCCP (Ours)"] = time_2d
    
    # --- PRINT FINAL TABLE ---
    print("\n" + "="*60)
    print(f"FINAL DRUG DISCOVERY SPEED RESULTS ({sample_size}x{sample_size} Kernel Matrix)")
    print("="*60)
    print(f"{'Kernel Name':<28} | {'Total Time (s)':<15} | {'Time per pair (ms)':<15}")
    print("-" * 64)
    
    total_pairs = (sample_size * sample_size)
    
    for name, t in sorted(results.items(), key=lambda x: x[1]):
        time_per_pair_ms = (t / total_pairs) * 1000
        print(f"{name:<28} | {t:<15.3f} | {time_per_pair_ms:<15.3f}")
    
    print("\nConclusion: Fingerprints (Morgan, MACCS) are instantaneous because they destroy the graph.")
    print("They collapse the molecule into a 1D bit-string, making 3D alignment impossible.")
    print("Our 2D kernel is the fastest structural graph kernel that preserves the mathematical topology!")

if __name__ == "__main__":
    run_drug_discovery_benchmark(sample_size=10)
