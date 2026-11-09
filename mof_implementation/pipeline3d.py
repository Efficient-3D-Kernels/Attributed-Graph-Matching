"""pipeline3d.py -- Phase 1 (2D Fast-WCS) + Phase 2 (3D Kabsch/RMSD) and the kernels of Fig. 1:
   k_2D = exp(-gamma_2D * F*/L)      (Algorithm 1, line 25)
   k_3D = exp(-gamma_3D * RMSD^2)    (Fig. 1)
   k_hybrid = (1-beta) k_2D + beta k_3D
"""
import numpy as np
from scipy.spatial.distance import cdist
import fastwcs as fw
import mof_data as md


def fragment(atoms, n_atoms):
    """Deterministic local fragment: the n_atoms atoms nearest to the first metal atom."""
    sym = atoms.get_chemical_symbols()
    metals = [i for i, s in enumerate(sym) if s not in md.NONMETALS]
    center = metals[0] if metals else 0
    pos, Z = md.local_cluster(atoms, center, n_atoms)
    return pos, Z


def match_fragments(posG, ZG, posH, ZH, omega=50.0, alpha=0.7, scale="shared_max", starts=1, seed=0):
    """Phase 1 (WCS) + Phase 2 (Kabsch).  L = max(1, min(|V_G|,|V_H|) - 2) as in Remark 3."""
    M, N = len(posG), len(posH)
    L = max(1, min(M, N) - 2)
    AG, AH = fw.scale_adjacency(cdist(posG, posG), cdist(posH, posH), scale)
    C = md.unary_cost(ZG, ZH, omega)
    res = fw.fastwcs(AG, AH, C, L, alpha=alpha, starts=starts, seed=seed)
    rows, cols = res["rows"], res["cols"]
    rmsd = fw.kabsch_rmsd(posG[rows], posH[cols]) if L >= 3 else np.nan
    return dict(F_over_L=res["F"] / L, rmsd=rmsd, L=L, rows=rows, cols=cols)


def kernels(f_over_L, rmsd, g2, g3, beta):
    k2 = np.exp(-g2 * f_over_L)
    k3 = np.exp(-g3 * rmsd ** 2)
    return k2, k3, (1 - beta) * k2 + beta * k3
