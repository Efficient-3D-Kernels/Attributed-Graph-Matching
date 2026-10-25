import time
import numpy as np
from rdkit import Chem
import os

try:
    from dscribe.descriptors import SOAP, CoulombMatrix
    from ase import Atoms
except ImportError:
    print("dscribe or ase is missing.")
    exit(1)

# Import your custom kernel pipeline
from wcs_gnccp import WCS_GNCCP

def rdkit_to_ase(mol):
    """Convert RDKit molecule to ASE Atoms object for dscribe."""
    # Do NOT remove hydrogens! Quantum 3D kernels MUST have hydrogens to calculate forces.
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    return Atoms(symbols=symbols, positions=positions)

def run_advanced_3d_benchmark(sample_size=10):
    print("="*60)
    print("ADVANCED 3D KERNEL BENCHMARKING (SOAP & Coulomb Matrix)")
    print("="*60)
    
    data_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'Aromatase_actives_new.sdf')
    print(f"Loading {sample_size} molecules with 3D Conformers...")
    
    mols = []
    suppl = Chem.SDMolSupplier(data_path, removeHs=False)
    for m in suppl:
        if m is not None:
            mols.append(m)
        if len(mols) >= sample_size:
            break
            
    print(f"Loaded {len(mols)} 3D molecules successfully.\n")
    
    # 1. Convert to ASE
    print("Converting to ASE physics format...")
    ase_mols = [rdkit_to_ase(m) for m in mols]
    
    # Determine unique species for SOAP
    species = set()
    for m in mols:
        for a in m.GetAtoms():
            species.add(a.GetSymbol())
    species = list(species)
    max_atoms = max([m.GetNumAtoms() for m in mols])
    
    results = {}
    
    # --- A. COULOMB MATRIX ---
    print("Running Coulomb Matrix (dscribe)...")
    cm = CoulombMatrix(n_atoms_max=max_atoms, permutation="sorted_l2")
    start = time.time()
    for ase_m in ase_mols:
        _ = cm.create(ase_m)
    time_cm = time.time() - start
    results["Coulomb Matrix"] = time_cm
    
    # --- B. SOAP (Smooth Overlap of Atomic Positions) ---
    print("Running SOAP Kernel (dscribe)...")
    # Using standard quantum chemistry parameters
    soap = SOAP(species=species, periodic=False, r_cut=6.0, n_max=8, l_max=6)
    start = time.time()
    for ase_m in ase_mols:
        _ = soap.create(ase_m)
    time_soap = time.time() - start
    results["SOAP (Quantum SOTA)"] = time_soap

    # --- C. OUR 2D KERNEL (WCS-GNCCP) ---
    # Strip hydrogens for our kernel because it's a structural topology algorithm!
    print("Running 2D Kernel WCS-GNCCP (Ours)...")
    mols_no_h = [Chem.RemoveHs(m) for m in mols]
    our_data = [WCS_GNCCP.precompute_molecular_data(m) for m in mols_no_h]
    matcher = WCS_GNCCP(alpha=0.7, gamma=0.5, max_iter=8, fw_max_iter=8)
    
    start = time.time()
    n = len(mols_no_h)
    for i in range(n):
        for j in range(i, n):
            matcher.match_from_precomputed(our_data[i], our_data[j])
    time_2d = time.time() - start
    results["2D WCS-GNCCP (Ours)"] = time_2d
    
    # --- PRINT FINAL TABLE ---
    print("\n" + "="*60)
    print(f"FINAL 3D QUANTUM SPEED RESULTS ({sample_size} Molecules)")
    print("="*60)
    print(f"{'Kernel Name':<25} | {'Total Time (s)':<15} | {'Time per molecule/pair (ms)'}")
    print("-" * 60)
    
    total_pairs = (sample_size * sample_size)
    total_mols = sample_size
    
    for name, t in sorted(results.items(), key=lambda x: x[1]):
        # Coulomb and SOAP are descriptors (time per molecule), ours is pairwise (time per pair)
        if "Ours" in name:
            ms = (t / total_pairs) * 1000
        else:
            ms = (t / total_mols) * 1000
        print(f"{name:<25} | {t:<15.3f} | {ms:<15.3f}")

if __name__ == "__main__":
    run_advanced_3d_benchmark(sample_size=10)
