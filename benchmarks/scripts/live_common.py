"""Shared infrastructure for the live (clean-slate) Fast-WCS benchmark suite.

Everything a benchmark script needs that is not experiment-specific lives here:
  * instance generators (Yang et al. 2018 synthetic protocol),
  * adapters that give every competitor solver identical inputs and project its output to an
    exact-cardinality-L partial permutation with ONE documented rule per output type,
  * statistics (bootstrap CIs, Wilcoxon), provenance capture and CSV/JSON writers,
  * a process-pool runner that pins every worker to one BLAS / OpenMP / torch thread.

No benchmark number is stored or read here; all results are produced by the calling scripts.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import datetime as _dt
import hashlib
import importlib.metadata
import json
import platform
import socket
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy
from scipy import sparse
from scipy.optimize import linear_sum_assignment, linprog, quadratic_assignment

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parents[1]
LIVE = ROOT / "benchmarks" / "live_runs"
FIG_BENCH = ROOT / "benchmarks" / "figures" / "standalone"
FIG_PAPER = ROOT / "paper" / "figures" / "standalone"
sys.path.insert(0, str(ROOT))

from gnccp_advanced_v3 import AdvancedGNCCP, robust_kabsch_superposition  # noqa: E402
from gnccp_baseline_2018 import OriginalGNCCP_2018  # noqa: E402


# ======================================================================================
# provenance / io
# ======================================================================================

def _version(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def provenance(files: Iterable[Path] = ()) -> dict:
    try:
        from threadpoolctl import threadpool_info
        pools = [{k: p.get(k) for k in ("internal_api", "num_threads", "version")} for p in threadpool_info()]
    except Exception:
        pools = []
    files = list(files) + [ROOT / "gnccp_advanced_v3.py", ROOT / "gnccp_baseline_2018.py", SCRIPTS / "live_common.py"]
    return {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(), "platform": platform.platform(),
        "cpu": platform.processor(), "logical_cpus": os.cpu_count(), "host": socket.gethostname(),
        "versions": {p: _version(p) for p in ("numpy", "scipy", "pygmtools", "POT", "torch", "open3d",
                                              "rdkit", "opencv-python", "pycpd", "matplotlib")},
        "threadpools_main_process": pools,
        "sha256": {str(Path(f).resolve().relative_to(ROOT)) if str(f).startswith(str(ROOT)) else str(f): sha256(f)
                   for f in dict.fromkeys(files) if Path(f).exists()},
    }


def write_csv(path: Path, rows: Sequence[dict], fields: Optional[Sequence[str]] = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fields is None:
        fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def write_json(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=_json_default), encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def run_dir(name: str) -> Path:
    d = LIVE / f"{timestamp()}_{name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


class Tee:
    """Mirror stdout into a log file."""
    def __init__(self, path: Path):
        self.f = Path(path).open("a", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s); self.f.write(s); self.f.flush()

    def flush(self):
        self.stdout.flush(); self.f.flush()


# ======================================================================================
# statistics
# ======================================================================================

def mean_ci95(x, seed=0, n_boot=4000) -> Tuple[Optional[float], Optional[float]]:
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=np.float64)
    if len(x) == 0:
        return None, None
    if len(x) == 1 or np.all(x == x[0]):
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    lo, hi = np.quantile(m, [0.025, 0.975])
    return float(lo), float(hi)


def summarize(values) -> dict:
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if len(x) == 0:
        return {"n": 0, "mean": None, "sd": None, "median": None, "q25": None, "q75": None}
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "median": float(np.median(x)), "q25": float(np.quantile(x, .25)), "q75": float(np.quantile(x, .75))}


def wilcoxon_p(a, b) -> Optional[float]:
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 5 or np.allclose(a, b):
        return None
    try:
        return float(wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return None


# ======================================================================================
# synthetic instance generator (Yang, Qiao & Liu, IEEE TNNLS 2018, Sec. III-A)
# ======================================================================================

def _shared_cutoff(DG: np.ndarray, DH: np.ndarray, density: float) -> float:
    """One distance cutoff for both graphs: the density-quantile of their pooled pairwise distances.

    A common cutoff keeps the planted sub-structure consistent across G and H (edges then differ
    only through the coordinate noise and the explicit edge perturbation), while each graph's
    density is ~density.
    """
    if density >= 1.0:
        return np.inf
    d = np.concatenate([DG[np.triu_indices(len(DG), 1)], DH[np.triu_indices(len(DH), 1)]])
    return float(np.quantile(d, density)) if len(d) else np.inf


def _threshold_graph(D: np.ndarray, cutoff: float) -> np.ndarray:
    A = np.where(D <= cutoff, D, 0.0).astype(np.float64)
    np.fill_diagonal(A, 0.0)
    return A


def _perturb_edges(A: np.ndarray, D: np.ndarray, sigma: float, rng) -> np.ndarray:
    """Remove and add round(0.5 * sigma * #Edge) edges (Yang et al. follow [7]); new edges weighted by distance."""
    n = A.shape[0]
    iu = np.triu_indices(n, 1)
    present = A[iu] != 0
    k = int(round(0.5 * sigma * int(present.sum())))
    if k == 0:
        return A
    on, off = np.flatnonzero(present), np.flatnonzero(~present)
    rem = rng.choice(on, size=min(k, len(on)), replace=False) if len(on) else np.empty(0, int)
    add = rng.choice(off, size=min(k, len(off)), replace=False) if len(off) else np.empty(0, int)
    out = A.copy()
    out[iu[0][rem], iu[1][rem]] = 0.0
    out[iu[1][rem], iu[0][rem]] = 0.0
    out[iu[0][add], iu[1][add]] = D[iu[0][add], iu[1][add]]
    out[iu[1][add], iu[0][add]] = D[iu[0][add], iu[1][add]]
    return out


def _er_with_planted(n: int, planted: np.ndarray, common: np.ndarray, density: float, rng) -> np.ndarray:
    """Boolean symmetric adjacency: planted-planted pairs copy `common`, all others iid Bernoulli(density)."""
    U = np.triu(rng.random((n, n)) < density, 1)
    U = U | U.T
    U[np.ix_(planted, planted)] = common
    np.fill_diagonal(U, False)
    return U


def _regular_with_planted(n: int, planted: np.ndarray, common: np.ndarray, d: int, rng) -> np.ndarray:
    """Planted regular graph: planted-planted pairs copy `common`, outliers connected to d nodes."""
    U = np.zeros((n, n), dtype=bool)
    U[np.ix_(planted, planted)] = common
    outliers = [v for v in range(n) if v not in planted]
    if outliers:
        for out_v in outliers:
            candidates = [v for v in range(n) if v != out_v and not U[out_v, v]]
            if candidates:
                targets = rng.choice(candidates, size=min(d, len(candidates)), replace=False)
                for t in targets:
                    U[out_v, t] = True
                    U[t, out_v] = True
    np.fill_diagonal(U, False)
    return U


def _ba_with_planted(n: int, planted: np.ndarray, common: np.ndarray, m: int, rng) -> np.ndarray:
    """Planted Barabasi-Albert graph: planted-planted pairs copy `common`, outliers attached via preferential attachment."""
    import networkx as _nx
    G_nx = _nx.Graph()
    G_nx.add_nodes_from(range(n))
    L = len(planted)
    for idx_i, orig_i in enumerate(planted):
        for idx_j in range(idx_i + 1, L):
            if common[idx_i, idx_j]:
                G_nx.add_edge(orig_i, planted[idx_j])
    outliers = [v for v in range(n) if v not in planted]
    degrees = dict(G_nx.degree())
    for out_v in outliers:
        nodes = list(G_nx.nodes())
        nodes.remove(out_v)
        degs = np.array([degrees[u] + 1.0 for u in nodes], dtype=float)
        probs = degs / degs.sum()
        targets = rng.choice(nodes, size=min(m, len(nodes)), replace=False, p=probs)
        for t in targets:
            G_nx.add_edge(out_v, t)
            degrees[out_v] = degrees.get(out_v, 0) + 1
            degrees[t] = degrees.get(t, 0) + 1
    U = _nx.to_numpy_array(G_nx).astype(bool)
    np.fill_diagonal(U, False)
    return U


def make_synthetic_instance(M: int, N: int, L: int, sigma: float, density: float, seed: int,
                            topology: str = "shared_er"):
    """Returns (A_G, A_H, X_gt) as float32/float32/uint8.

    g_i, h_j ~ U(0,1)^2; a random partial permutation X_gt with L ones; g_i = h_j + N(0, sigma^2 I)
    for (i, j) in X_gt; edge weight = Euclidean distance between the graph's own points.

    topology='shared_er' (default; the construction of [7] that Yang et al. follow): the planted
      vertex pairs share one Erdos-Renyi structure of density rho; every pair involving an outlier is
      an independent Bernoulli(rho) edge.
    topology='threshold' (stress variant): both graphs keep the pairs closer than one shared distance
      cutoff (pooled rho-quantile), so coordinate noise also flips edges near the cutoff.
    topology='random_regular': every vertex in the planted core has exact regular degree d ~ rho*(L-1);
      outliers are wired with regular degree d.
    topology='barabasi_albert': scale-free network generated via preferential attachment, featuring
      prominent high-degree hubs.
    In all cases round(0.5*sigma*#Edge) edges are then removed and as many added in each graph.
    """
    rng = np.random.default_rng(seed)
    H = rng.uniform(0, 1, (N, 2))
    G = rng.uniform(0, 1, (M, 2))
    gi = rng.choice(M, L, replace=False)
    hj = rng.choice(N, L, replace=False)
    G[gi] = H[hj] + rng.normal(0.0, sigma, (L, 2))
    Xgt = np.zeros((M, N), dtype=np.uint8)
    Xgt[gi, hj] = 1
    DG = np.linalg.norm(G[:, None] - G[None], axis=2)
    DH = np.linalg.norm(H[:, None] - H[None], axis=2)
    if topology == "threshold":
        tau = _shared_cutoff(DG, DH, density)
        AG0, AH0 = _threshold_graph(DG, tau), _threshold_graph(DH, tau)
    elif topology == "shared_er":
        common = np.triu(rng.random((L, L)) < density, 1)
        common = common | common.T
        AG0 = _er_with_planted(M, gi, common, density, rng) * DG
        AH0 = _er_with_planted(N, hj, common, density, rng) * DH
    elif topology in ("random_regular", "regular"):
        import networkx as _nx
        d = max(2, int(round(density * (L - 1))))
        if (L * d) % 2 != 0:
            d = d - 1 if d > 2 else d + 1
        common_nx = _nx.random_regular_graph(d, L, seed=int(rng.integers(0, 1e9)))
        common = _nx.to_numpy_array(common_nx).astype(bool)
        AG0 = _regular_with_planted(M, gi, common, d, rng) * DG
        AH0 = _regular_with_planted(N, hj, common, d, rng) * DH
    elif topology in ("barabasi_albert", "ba"):
        import networkx as _nx
        m = max(1, min(L - 1, int(round(density * (L - 1) / 4))))
        if m < 1:
            m = 1
        common_nx = _nx.barabasi_albert_graph(L, m, seed=int(rng.integers(0, 1e9)))
        common = _nx.to_numpy_array(common_nx).astype(bool)
        AG0 = _ba_with_planted(M, gi, common, m, rng) * DG
        AH0 = _ba_with_planted(N, hj, common, m, rng) * DH
    else:
        raise ValueError(topology)
    AG = _perturb_edges(AG0, DG, sigma, rng)
    AH = _perturb_edges(AH0, DH, sigma, rng)
    return AG.astype(np.float32), AH.astype(np.float32), Xgt


# ======================================================================================
# assignment utilities
# ======================================================================================

def project_exact_L(scores: np.ndarray, L: int) -> np.ndarray:
    """Max-score partial permutation with exactly L ones (Burkard augmentation, inf dummy-dummy block)."""
    S = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    M, N = S.shape
    X = np.zeros((M, N), dtype=np.uint8)
    if L >= min(M, N):
        r, c = linear_sum_assignment(-S)
    else:
        size = M + N - L
        cost = np.zeros((size, size))
        cost[:M, :N] = -S
        cost[M:, N:] = np.inf
        r, c = linear_sum_assignment(cost)
        k = (r < M) & (c < N)
        r, c = r[k], c[k]
    X[r, c] = 1
    return X


def truncate_matching_to_L(pairs: np.ndarray, AG: np.ndarray, AH: np.ndarray, L: int, C=None, alpha=1.0) -> np.ndarray:
    """Exact-L projection for solvers that return a FULL discrete matching (FAQ, IPFP).

    Greedily drops the matched pair with the largest contribution to the WCS objective
    (sum_k (A_G[i,k] - A_H[pi(i),pi(k)])^2 over the remaining matched set, plus its unary cost),
    until L pairs remain. Deterministic; no ground-truth information is used.
    """
    pairs = np.asarray(pairs, dtype=np.intp).reshape(-1, 2)
    M, N = AG.shape[0], AH.shape[0]
    rows, cols = pairs[:, 0], pairs[:, 1]
    alive = np.ones(len(rows), bool)
    Dm = (AG[np.ix_(rows, rows)].astype(np.float64) - AH[np.ix_(cols, cols)].astype(np.float64)) ** 2
    contrib = alpha * 2.0 * Dm.sum(axis=1)
    if C is not None and alpha < 1:
        contrib = contrib + (1 - alpha) * np.asarray(C, np.float64)[rows, cols]
    while alive.sum() > L:
        idx = np.flatnonzero(alive)
        worst = idx[int(np.argmax(contrib[idx]))]
        alive[worst] = False
        contrib -= alpha * 2.0 * Dm[:, worst]
    X = np.zeros((M, N), dtype=np.uint8)
    X[rows[alive], cols[alive]] = 1
    return X


def validate_exact_L(X: np.ndarray, L: int):
    X = np.asarray(X)
    if not np.all((X == 0) | (X == 1)) or int(X.sum()) != L or X.sum(0).max(initial=0) > 1 or X.sum(1).max(initial=0) > 1:
        raise ValueError(f"not an exact-L partial permutation (sum={int(X.sum())}, L={L})")


def wcs_objective(X, AG, AH, C=None, alpha=1.0) -> float:
    X = np.asarray(X, np.float64); AG = np.asarray(AG, np.float64); AH = np.asarray(AH, np.float64)
    r = X.sum(1)
    R = (r[:, None] * r[None, :]) * AG - X @ AH @ X.T
    f = float(np.sum(R * R))
    if C is not None and alpha < 1:
        return alpha * f + (1 - alpha) * float(np.sum(np.asarray(C, np.float64) * X))
    return alpha * f


def accuracy_pct(X, Xgt, L) -> float:
    return 100.0 * float(np.sum(np.asarray(X, np.int64) * np.asarray(Xgt, np.int64))) / L


# ======================================================================================
# solver adapters (every adapter: (AG, AH, L, C, alpha) -> exact-L binary X)
# ======================================================================================

def solve_fastwcs(AG, AH, L, C=None, alpha=1.0, **kw):
    s = AdvancedGNCCP(alpha=alpha, **kw)
    r = s.solve(AG, AH, C=C, L=L)
    return r.X_binary.astype(np.uint8)


class YangGNCCP2018(OriginalGNCCP_2018):
    """OriginalGNCCP_2018 with the exact-LP oracle fed a cached sparse constraint matrix.

    The LP (Eq. 5 over D_L) is unchanged; only the per-call dense Python-loop construction of the
    constraint matrix is replaced, so the baseline's wall-clock is not inflated by our harness.
    """
    _lp_cache: Dict[Tuple[int, int, int], tuple] = {}

    def _solve_linear_oracle(self, grad_J, M, N, L):
        if self.oracle != "exact_lp":
            return super()._solve_linear_oracle(grad_J, M, N, L)
        key = (M, N, L)
        if key not in self._lp_cache:
            n = M * N
            idx = np.arange(n)
            A_rows = sparse.csr_matrix((np.ones(n), (idx // N, idx)), shape=(M, n))
            A_cols = sparse.csr_matrix((np.ones(n), (idx % N, idx)), shape=(N, n))
            A_ub = sparse.vstack([A_rows, A_cols]).tocsr()
            b_ub = np.ones(M + N)
            A_eq = sparse.csr_matrix(np.ones((1, n)))
            self._lp_cache[key] = (A_ub, b_ub, A_eq, np.array([float(L)]))
        A_ub, b_ub, A_eq, b_eq = self._lp_cache[key]
        res = linprog(grad_J.ravel().astype(np.float64), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=(0.0, 1.0), method="highs")
        if res.success:
            return res.x.reshape(M, N).astype(np.float32)
        # same fallback as the reference implementation: rectangular Hungarian + top-L
        self.lp_failures = getattr(self, "lp_failures", 0) + 1
        r, c = linear_sum_assignment(grad_J)
        k = np.argsort(grad_J[r, c])[:L]
        Y = np.zeros((M, N), dtype=np.float32)
        Y[r[k], c[k]] = 1.0
        return Y


YANG_SETTINGS = dict(max_iter=10, fw_iter=6, n_grid_points=11)


def solve_yang(AG, AH, L, C=None, alpha=1.0, relaxation="RLX1", oracle="exact_lp"):
    s = YangGNCCP2018(alpha=alpha, relaxation=relaxation, oracle=oracle, **YANG_SETTINGS)
    r = s.solve(AG, AH, L=L, C=C if alpha < 1 else None)
    X = np.zeros((AG.shape[0], AH.shape[0]), np.uint8)
    for i, j in r.matches:
        X[i, j] = 1
    return X


def solve_faq(AG, AH, L, C=None, alpha=1.0):
    """SciPy FAQ (Vogelstein et al. 2015) on zero-padded square graphs, maximising tr(A_G P A_H P^T)."""
    M, N = AG.shape[0], AH.shape[0]
    n = max(M, N)
    A = np.zeros((n, n)); A[:M, :M] = AG
    B = np.zeros((n, n)); B[:N, :N] = AH
    res = quadratic_assignment(A, B, method="faq", options={"maximize": True, "P0": "barycenter"})
    rows = np.arange(n); cols = res.col_ind
    keep = (rows < M) & (cols < N)
    return truncate_matching_to_L(np.c_[rows[keep], cols[keep]], AG, AH, L, C, alpha)


# ---------- PyGMtools (SM / RRWM / IPFP / NGM) ----------------------------------------

def affinity_wcs(AG, AH, C=None, alpha=1.0):
    """Lawler affinity equivalent to the WCS objective on exact-L permutations.

    K[(i,a),(j,b)] = kmax - (A_G[i,j] - A_H[a,b])^2 (i != j, a != b); unary diagonal from C.
    For exact-L assignments sum K = const - F(X), so maximising it == minimising the WCS objective.
    """
    M, N = AG.shape[0], AH.shape[0]
    src, tgt = _pygm_index(M, N)
    D = AG.astype(np.float32)[src[:, None], src[None, :]] - AH.astype(np.float32)[tgt[:, None], tgt[None, :]]
    np.square(D, out=D)
    K = np.float32(alpha) * (np.float32(D.max()) - D)
    del D
    K[(src[:, None] == src[None, :]) | (tgt[:, None] == tgt[None, :])] = 0.0
    if C is not None and alpha < 1:
        c = np.asarray(C, np.float64).ravel(order="F")
        np.fill_diagonal(K, (1 - alpha) * (c.max() - c))
    return K.astype(np.float32)


def _pygm_index(M, N):
    """PyGMtools vectorises X column-wise: entry (i, a) sits at index a*M + i."""
    return np.tile(np.arange(M), N), np.repeat(np.arange(N), M)


def affinity_gaussian(AG, AH, sigma=0.1, C=None, alpha=1.0, node_sigma=None):
    """Classical edge-only Gaussian affinity (Leordeanu & Hebert 2005; Cho et al. 2010).

    K[(i,a),(j,b)] = exp(-(w_ij - w_ab)^2 / sigma^2) when both edges exist, else 0; optional node
    affinity exp(-C_ia / node_sigma) on the diagonal.
    """
    M, N = AG.shape[0], AH.shape[0]
    src, tgt = _pygm_index(M, N)
    WG = AG.astype(np.float32)[src[:, None], src[None, :]]
    WH = AH.astype(np.float32)[tgt[:, None], tgt[None, :]]
    mask = (WG != 0) & (WH != 0)
    WG -= WH
    del WH
    np.square(WG, out=WG)
    WG *= np.float32(-1.0 / sigma ** 2)
    np.exp(WG, out=WG)
    K = WG * mask
    if C is not None and alpha < 1:
        c = np.asarray(C, np.float64).ravel(order="F")
        np.fill_diagonal(K, np.exp(-c / (node_sigma or max(np.median(c), 1e-6))))
    return K.astype(np.float32)


_NGM_NET = None


def solve_pygm(AG, AH, L, C=None, alpha=1.0, method="rrwm", affinity="wcs", sigma=0.1, max_iter=50):
    import pygmtools as pygm
    M, N = AG.shape[0], AH.shape[0]
    K = affinity_wcs(AG, AH, C, alpha) if affinity == "wcs" else affinity_gaussian(AG, AH, sigma, C, alpha)
    if method == "ngm":
        import torch
        global _NGM_NET
        torch.set_num_threads(1)
        with torch.no_grad():
            if _NGM_NET is None:      # load the pretrained weights once per process (not timed per call)
                _, _NGM_NET = pygm.ngm(torch.from_numpy(K), M, N, pretrain="voc", backend="pytorch", return_network=True)
            Xs = pygm.ngm(torch.from_numpy(K), M, N, network=_NGM_NET, backend="pytorch")
        S = Xs.numpy()
    else:
        fn = {"sm": pygm.sm, "rrwm": pygm.rrwm, "ipfp": pygm.ipfp}[method]
        S = np.asarray(fn(K, M, N, max_iter=max_iter, backend="numpy"))
    S = S.reshape(M, N)
    if method == "ipfp":                       # IPFP returns a full discrete matching
        r, c = np.nonzero(S > 0.5)
        return truncate_matching_to_L(np.c_[r, c], AG, AH, L, C, alpha)
    return project_exact_L(S, L)


# ---------- Python Optimal Transport (FGW, entropic FGW, partial GW) ------------------

def _ot_struct(AG, AH):
    s = max(float(np.abs(AG).max(initial=0)), float(np.abs(AH).max(initial=0)), 1e-12)
    return AG.astype(np.float64) / s, AH.astype(np.float64) / s


def solve_fgw(AG, AH, L, C=None, alpha=1.0, epsilon=None, fgw_alpha=None):
    """(Entropic) Fused Gromov-Wasserstein (Vayer et al. 2019/2020) with uniform marginals.

    POT convention: (1 - a) <C, T> + a * GW(T). For the structure-only protocol (alpha=1) a=1 (pure GW).
    epsilon=None -> exact conditional-gradient FGW; otherwise entropic FGW (log-domain Sinkhorn).
    The transport plan is projected to an exact-L assignment by maximum plan mass.
    """
    import ot
    M, N = AG.shape[0], AH.shape[0]
    C1, C2 = _ot_struct(AG, AH)
    p, q = np.full(M, 1.0 / M), np.full(N, 1.0 / N)
    Mf = np.zeros((M, N)) if (C is None or alpha >= 1) else np.asarray(C, np.float64) / max(float(np.max(C)), 1e-12)
    a = 1.0 if (C is None or alpha >= 1) else (alpha if fgw_alpha is None else fgw_alpha)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if epsilon is None:
            T = ot.gromov.fused_gromov_wasserstein(Mf, C1, C2, p, q, loss_fun="square_loss", alpha=a)
        else:
            T = _entropic_fgw_log(Mf, C1, C2, p, q, epsilon, a)
    T = np.asarray(T)
    if not np.isfinite(T).all() or T.sum() <= 0:
        raise FloatingPointError("FGW returned a non-finite / empty plan")
    return project_exact_L(T, L)


def _entropic_fgw_log(Mf, C1, C2, p, q, epsilon, a, max_iter=200, tol=1e-9):
    """POT's entropic FGW 'PGD' iteration (Peyre et al. 2016) with log-domain Sinkhorn.

    ot.gromov.entropic_fused_gromov_wasserstein hard-codes method='sinkhorn', which underflows for
    epsilon = 1e-3 on unit-scale costs. The outer iteration below is POT's (same gradient, same
    KL-projection subproblem, same 1e-9 tolerances); the Sinkhorn subproblem is solved in the log
    domain and warm-started from the previous dual potentials, which changes speed, not the fixed point.
    """
    import ot
    constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, "square_loss")
    T = np.outer(p, q)
    f, g = np.zeros(len(p)), np.zeros(len(q))
    for _ in range(max_iter):
        tens = a * ot.gromov.gwggrad(constC, hC1, hC2, T) + (1 - a) * Mf
        T_new, f, g = _sinkhorn_log_ws(p, q, tens, epsilon, f, g)
        if np.abs(T_new - T).sum() < tol:
            T = T_new
            break
        T = T_new
    return T


def _lse(Z, axis):
    m = Z.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(Z - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def _sinkhorn_log_ws(a, b, Cm, eps, f, g, max_iter=1000, tol=1e-9):
    loga, logb = np.log(a), np.log(b)
    Ce = Cm / eps
    for it in range(max_iter):
        f = eps * (loga - _lse(g[None, :] / eps - Ce, 1))
        g = eps * (logb - _lse(f[:, None] / eps - Ce, 0))
        if it % 10 == 9:
            T = np.exp((f[:, None] + g[None, :]) / eps - Ce)
            if np.abs(T.sum(1) - a).sum() < tol:
                return T, f, g
    return np.exp((f[:, None] + g[None, :]) / eps - Ce), f, g


def solve_partial_gw(AG, AH, L, C=None, alpha=1.0):
    """Partial Gromov-Wasserstein (Chapel et al., NeurIPS 2020): unit node masses, transported mass m = L.

    With a unary cost (alpha < 1) the partial *fused* GW of POT is used: (1 - alpha) <C, T> + alpha GW(T).
    """
    import ot
    M, N = AG.shape[0], AH.shape[0]
    C1, C2 = _ot_struct(AG, AH)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if C is None or alpha >= 1:
            T = ot.partial.partial_gromov_wasserstein(C1, C2, np.ones(M), np.ones(N), m=float(L))
        else:
            Mf = np.asarray(C, np.float64) / max(float(np.max(C)), 1e-12)
            T = ot.gromov.partial_fused_gromov_wasserstein(Mf, C1, C2, np.ones(M), np.ones(N), m=float(L),
                                                           loss_fun="square_loss", alpha=alpha)
    T = np.asarray(T)
    if not np.isfinite(T).all() or T.sum() <= 0:
        raise FloatingPointError("partial GW returned a non-finite / empty plan")
    return project_exact_L(T, L)


# ======================================================================================
# parallel runner
# ======================================================================================

def _worker_init():
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        from threadpoolctl import threadpool_limits
        global _TP_LIMIT
        _TP_LIMIT = threadpool_limits(1)
    except Exception:
        pass
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    warnings.filterwarnings("ignore")


def run_pool(fn: Callable, tasks: List, workers: int, desc: str = "", log_every: int = 25):
    """Run fn(task) over tasks in a spawn pool of single-threaded workers; preserves task order."""
    import multiprocessing as mp
    out = [None] * len(tasks)
    t0 = time.time()
    if workers <= 1:
        _worker_init()
        for i, t in enumerate(tasks):
            out[i] = fn(t)
            if (i + 1) % log_every == 0:
                print(f"[{desc}] {i+1}/{len(tasks)}  {time.time()-t0:.0f}s", flush=True)
        return out
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_worker_init, maxtasksperchild=200) as pool:
        for k, (i, r) in enumerate(pool.imap_unordered(_indexed, [(fn, i, t) for i, t in enumerate(tasks)], chunksize=1)):
            out[i] = r
            if (k + 1) % log_every == 0 or k + 1 == len(tasks):
                el = time.time() - t0
                print(f"[{desc}] {k+1}/{len(tasks)}  elapsed {el:.0f}s  eta {el/(k+1)*(len(tasks)-k-1):.0f}s", flush=True)
    return out


def _indexed(args):
    fn, i, t = args
    return i, fn(t)


_WARMED = set()


def warmup_once(table: Dict[str, Callable], key: str = "default"):
    """Run every solver once per process on a tiny instance (imports, weight loading, JIT caches)
    so that one-off costs never land inside a timed call."""
    if key in _WARMED:
        return
    AG, AH, _ = make_synthetic_instance(8, 10, 6, 0.02, 0.5, seed=1)
    for fn in table.values():
        try:
            fn(AG, AH, 6)
        except Exception:
            pass
    _WARMED.add(key)


def timed_call(fn: Callable, *a, **kw):
    """(result, wall_ms, error_string)."""
    t = time.perf_counter()
    try:
        r = fn(*a, **kw)
        return r, 1e3 * (time.perf_counter() - t), None
    except Exception as e:  # recorded, never hidden
        return None, 1e3 * (time.perf_counter() - t), f"{type(e).__name__}: {str(e)[:200]}"
