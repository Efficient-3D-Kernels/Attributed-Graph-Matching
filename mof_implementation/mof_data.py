"""mof_data.py -- CoRE MOF 2019 (v2.0.0) download, CIF parsing, local cluster extraction and
construction of WCS matching instances with known ground truth (planted protocol).

Graph definition (paper Sec. VI-D, Prop. 4, Fig. 1):
  * vertices = atoms; A_ik = Euclidean (internal) interatomic distance ||p_i - p_k||_2
  * attributes F_i = (covalent radius, atomic mass), scaled to [0,1]-ish
  * unary cost  C_ij = ||F_i - F_j||_2 + Omega * [Z_i != Z_j]     (Omega = 50 in Fig. 1)
"""
import hashlib
import os
import urllib.request
import zipfile
import numpy as np
from scipy.spatial.distance import cdist

ZENODO_URL = "https://zenodo.org/records/3677685/files/CoREMOF2019_public_v2.zip?download=1"

NONMETALS = {"H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar", "As", "Se",
             "Br", "Kr", "Te", "I", "Xe", "At", "Rn"}


def download_coremof(dest="data", url=ZENODO_URL):
    os.makedirs(dest, exist_ok=True)
    zpath = os.path.join(dest, "CoREMOF2019_public_v2.zip")
    if not os.path.exists(zpath):
        print("downloading", url)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(zpath, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    md5 = hashlib.md5(open(zpath, "rb").read()).hexdigest()
    print("md5 of downloaded file:", md5, "(compare with the checksum on the Zenodo record page)")
    out = os.path.join(dest, "CoREMOF2019")
    if not os.path.isdir(out):
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out)
    return out


def find_cifs(root, name_filter=None):
    """All CIFs below root, skipping any path component containing 'NCR' (not computation-ready)."""
    paths = []
    for dp, _, fs in os.walk(root):
        parts = [p.upper() for p in dp.replace("\\", "/").split("/")]
        if any("NCR" in p for p in parts):
            continue
        for f in fs:
            if f.lower().endswith(".cif"):
                p = os.path.join(dp, f)
                if name_filter is None or name_filter.upper() in p.upper():
                    paths.append(p)
    return sorted(paths)


def load_structure(path):
    from ase.io import read
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at = read(path)
    if len(at) == 0 or not at.pbc.any() or at.cell.volume < 1e-3:
        raise ValueError("no periodic cell")
    return at


def local_cluster(atoms, center_idx, n_atoms):
    """The n_atoms atoms nearest to atoms[center_idx] in the periodic crystal
    (periodic images included).  Returns positions (n,3), atomic numbers (n,), sorted by distance."""
    cell = np.array(atoms.cell)
    vol = abs(np.linalg.det(cell))
    widths = np.array([vol / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3]))
                       for i in range(3)])
    rho = len(atoms) / vol
    radius = 1.5 * (3 * n_atoms / (4 * np.pi * rho)) ** (1 / 3) + 3.0
    reps = np.maximum(1, np.ceil(radius / widths).astype(int) * 2 + 1)
    if int(np.prod(reps)) * len(atoms) > 400000:
        raise ValueError("supercell too large")
    big = atoms.repeat(tuple(int(x) for x in reps))
    n0, n1, n2 = (int(x) for x in reps)
    mid = (n0 // 2) * (n1 * n2) + (n1 // 2) * n2 + (n2 // 2)    # ASE repeat ordering
    c = center_idx + mid * len(atoms)
    pos = big.get_positions(); Z = big.get_atomic_numbers()
    d = np.linalg.norm(pos - pos[c], axis=1)
    order = np.argsort(d, kind="stable")[:n_atoms]
    if len(order) < n_atoms:
        raise ValueError("not enough atoms")
    return pos[order], Z[order]


def atom_features(Z):
    from ase.data import covalent_radii, atomic_masses
    Z = np.asarray(Z)
    return np.stack([covalent_radii[Z] / 2.6, atomic_masses[Z] / 250.0], axis=1)


def unary_cost(ZG, ZH, omega):
    FG, FH = atom_features(ZG), atom_features(ZH)
    return cdist(FG, FH) + omega * (ZG[:, None] != ZH[None, :])


def build_mof_instance(atoms, M, N, L, sigma, omega, rng, scale="shared_max", n_outlier_pool=40,
                       drop_hydrogen=False):
    """Planted WCS instance from one crystal.
      H : the N atoms nearest to a randomly chosen metal centre (true coordinates)
      G : the L atoms nearest to that centre (conserved core, coordinates perturbed by
          N(0, sigma^2) in Angstrom) + (M-L) outlier atoms taken from the shell just beyond H
          (real atoms of the same framework without a partner in H); rows/cols randomly permuted.
    Ground truth is known by construction.  Graph edges = full internal distance matrices."""
    from fastwcs import scale_adjacency
    if drop_hydrogen:
        atoms = atoms[[a.number != 1 for a in atoms]]
    sym = atoms.get_chemical_symbols()
    metals = [i for i, s in enumerate(sym) if s not in NONMETALS]
    center = int(rng.choice(metals if metals else np.arange(len(atoms))))
    n_out = M - L
    pos, Z = local_cluster(atoms, center, N + max(n_outlier_pool, 4 * n_out))
    posH, ZH = pos[:N], Z[:N]
    core = np.arange(L)                                      # nearest L atoms inside H
    out_idx = N + rng.choice(len(pos) - N, n_out, replace=False) if n_out else np.array([], int)
    gi = np.concatenate([core, out_idx]).astype(int)
    posG_true, ZG = pos[gi], Z[gi]
    posG = posG_true + sigma * rng.standard_normal(posG_true.shape)
    pg = rng.permutation(M); ph = rng.permutation(N)         # random relabelling
    posG, ZG = posG[pg], ZG[pg]
    posH, ZH = posH[ph], ZH[ph]
    inv_g = np.argsort(pg); inv_h = np.argsort(ph)
    gt_rows = inv_g[core]                                    # position of core atom i in G
    gt_cols = inv_h[core]                                    # ... and in H
    AG = cdist(posG, posG); AH = cdist(posH, posH)
    AG, AH = scale_adjacency(AG, AH, scale)
    C = unary_cost(ZG, ZH, omega)
    return dict(AG=AG, AH=AH, C=C, L=L, gt_rows=gt_rows, gt_cols=gt_cols,
                posG=posG, posH=posH, ZG=ZG, ZH=ZH)
