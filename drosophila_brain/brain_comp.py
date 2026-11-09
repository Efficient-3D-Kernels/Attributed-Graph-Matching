import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment  # Hungarian algorithm from scipy

_start_time = time.time()
np.set_printoptions(precision=3, suppress=True, linewidth=180)

# 0. FETCH REAL DATA FROM NEUPRINT (both flies, same server)

SERVER = "neuprint-cns.janelia.org"

DATASET_A = "hemibrain:v1.2.2"
ROI_NAMES_A = ["MB(R)"]

DATASET_B = "flywire-fafb:v783b"
ROI_NAMES_B = ["MB_CA_R", "MB_PED_R", "MB_ML_R", "MB_VL_R"]


def _get_token(token=None):  # get the auth token from the neuprint site
    token = token or os.environ.get("NEUPRINT_TOKEN")
    if not token:
        raise RuntimeError(
            "No neuPrint token supplied. Set the NEUPRINT_TOKEN environment "
            "variable or pass token=... explicitly."
        )
    return token


def _check_rois(client, roi_names, dataset):
    """
    Sanity-check the requested ROIs against the server before summing connection weights.
    neuPrint reports synapse-count weights per ROI.  Summing rows across several requested
    ROIs is only valid if those ROIs are disjoint (primary ROIs); mixing an aggregate
    (non-primary) ROI with its own sub-ROIs would double-count synapses.
    """
    all_rois = set(getattr(client, "all_rois", None) or [])
    primary = set(getattr(client, "primary_rois", None) or [])
    if all_rois:
        missing = sorted(set(roi_names) - all_rois)
        if missing:
            raise ValueError(f"[{dataset}] ROI(s) not found on the server: {missing}. "
                             f"Some available ROIs: {sorted(all_rois)[:25]}")
    if not primary:
        print(f"[{dataset}] WARNING: could not determine which ROIs are primary; make sure "
              f"{list(roi_names)} are disjoint or a single aggregate ROI (else weights double-count).")
        return
    nonprimary = sorted(set(roi_names) - primary)
    if nonprimary and len(roi_names) > 1:
        print(f"[{dataset}] WARNING: requested ROIs mix non-primary/aggregate ROI(s) {nonprimary} with "
              f"others; summing their per-ROI weights can DOUBLE-COUNT synapses. Request either one "
              f"aggregate ROI or only disjoint primary ROIs.")
    elif nonprimary:
        print(f"[{dataset}] note: {nonprimary} is a non-primary (aggregate) ROI; as the only requested "
              f"ROI, each synapse is counted once.")
    else:
        print(f"[{dataset}] all requested ROIs are primary (disjoint): per-ROI weights can be summed.")


def fetch_and_save(dataset, roi_names, out_csv, server=SERVER, token=None):
    """
    Fetch ROI-restricted neuron-neuron connections and save one undirected-agnostic
    directed edge per (pre, post) pair with weight = total synapse count inside the
    requested ROI(s).  The ROI semantics are checked by _check_rois, summary statistics
    are printed, and metadata is written next to the CSV.  NOTE: the two datasets use
    different ROI vocabularies and synapse-prediction pipelines, so synapse-count scales
    are not guaranteed comparable; the fixed weight bins are shared by both graphs --
    inspect print_bin_distribution() for both flies.
    """
    from neuprint import Client, fetch_adjacencies

    client = Client(server, dataset=dataset, token=_get_token(token))
    _check_rois(client, roi_names, dataset)
    _, conn_df = fetch_adjacencies(rois=roi_names, include_nonprimary=True, client=client)

    conn_df = conn_df[conn_df["roi"].isin(roi_names)].copy()  # keep only requested ROIs
    if conn_df.empty:
        raise ValueError(f"[{dataset}] no connections returned inside ROIs {list(roi_names)}.")
    conn_df = (
        conn_df.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"]
        .sum()      # sum per-ROI synapse counts (valid for disjoint ROIs) -> one edge per ordered pair
        .rename(columns={"bodyId_pre": "source", "bodyId_post": "target"})
    )
    conn_df["region"] = "mushroom_body"
    conn_df = conn_df[["source", "target", "weight", "region"]]
    conn_df.to_csv(out_csv, index=False)

    w = conn_df["weight"].to_numpy(dtype=float)
    n_neurons = len(set(conn_df["source"]) | set(conn_df["target"]))
    print(f"Saved {out_csv}: {len(conn_df)} directed edges, {n_neurons} neurons, weight "
          f"min/median/max = {w.min():.0f}/{np.median(w):.0f}/{w.max():.0f}")
    try:
        import neuprint
        nv = getattr(neuprint, "__version__", "unknown")
    except Exception:
        nv = "unknown"
    with open(out_csv + ".meta.json", "w") as f:
        json.dump({"server": server, "dataset": dataset, "rois": list(roi_names),
                   "include_nonprimary": True, "n_edges": int(len(conn_df)),
                   "n_neurons": int(n_neurons), "weight_min": float(w.min()),
                   "weight_median": float(np.median(w)), "weight_max": float(w.max()),
                   "neuprint_python_version": nv}, f, indent=2)
    return conn_df


def fetch_soma_coordinates(dataset, roi_names, server=SERVER, token=None):
    """Go to neuPrint and get the soma coordinates of the neurons in the requested ROIs."""
    from neuprint import Client, fetch_neurons, NeuronCriteria as NC

    client = Client(server, dataset=dataset, token=_get_token(token))
    criteria = NC(rois=roi_names, client=client)
    neuron_df, _ = fetch_neurons(criteria, client=client)

    if "somaLocation" in neuron_df.columns:
        coords = neuron_df["somaLocation"].dropna().apply(pd.Series)
        coords.columns = ["x", "y", "z"][: coords.shape[1]]
        coords["bodyId"] = neuron_df.loc[coords.index, "bodyId"].values
    elif {"somaLocation_x", "somaLocation_y", "somaLocation_z"}.issubset(neuron_df.columns):
        coords = neuron_df[["bodyId", "somaLocation_x", "somaLocation_y", "somaLocation_z"]].dropna()
        coords.columns = ["bodyId", "x", "y", "z"]
    else:
        print(f"WARNING: no soma coordinate columns found for {dataset}; Stage 2 will be skipped for this fly.")
        return pd.DataFrame(columns=["bodyId", "x", "y", "z"])

    coords["bodyId"] = coords["bodyId"].astype(str)
    return coords[["bodyId", "x", "y", "z"]]


def fetch_real_datasets(fetch_coords=False, fly_A_file="flyA_hemibrain_MB.csv",
                        fly_B_file="flyB_fafb_MB.csv",
                        coords_A_file="flyA_hemibrain_MB_coords.csv",
                        coords_B_file="flyB_fafb_MB_coords.csv"):
    """Fetch the configured hemibrain/FlyWire ROI data and optionally soma coordinates.

    Authentication is read from the NEUPRINT_TOKEN environment variable. No token is
    stored in this script. The four output paths can be supplied from the CLI.
    """
    fetch_and_save(DATASET_A, ROI_NAMES_A, fly_A_file)
    fetch_and_save(DATASET_B, ROI_NAMES_B, fly_B_file)
    if fetch_coords:
        ca = fetch_soma_coordinates(DATASET_A, ROI_NAMES_A)
        cb = fetch_soma_coordinates(DATASET_B, ROI_NAMES_B)
        if ca.empty or cb.empty:
            raise RuntimeError("Soma coordinates could not be fetched for both datasets. "
                               "Check neuPrint permissions, ROI names, and somaLocation availability.")
        ca.to_csv(coords_A_file, index=False)
        cb.to_csv(coords_B_file, index=False)
        print(f"Saved soma coordinates: {coords_A_file} ({len(ca)} neurons), "
              f"{coords_B_file} ({len(cb)} neurons)")
    print("\nData fetch complete. Verify ROI definitions and coordinate frames before interpreting matches.")


def fetch_coordinates_only(coords_A_file="flyA_hemibrain_MB_coords.csv",
                           coords_B_file="flyB_fafb_MB_coords.csv"):
    """Fetch only the soma coordinates (keeps existing connectivity CSVs untouched)."""
    ca = fetch_soma_coordinates(DATASET_A, ROI_NAMES_A)
    cb = fetch_soma_coordinates(DATASET_B, ROI_NAMES_B)
    if ca.empty or cb.empty:
        raise RuntimeError("Soma coordinates could not be fetched for both datasets.")
    ca.to_csv(coords_A_file, index=False)
    cb.to_csv(coords_B_file, index=False)
    print(f"Saved soma coordinates: {coords_A_file} ({len(ca)} neurons), {coords_B_file} ({len(cb)} neurons)")


# 1. CONFIGURATION  bins are used to represent adjacency matrices and graph representations
N_BINS = 8
BIN_EDGES = np.array([0, 10, 60, 110, 160, 210, 260, 280, np.inf], dtype=float)
BIN_RANGES = ["0-10", "10-60", "60-110", "110-160", "160-210", "210-260", "260-280", ">=280"]


def weight_to_bin(weight):
    weight = float(weight)
    if weight < 0:
        raise ValueError("Connectivity weight cannot be negative.")  # weights are put into bins
    for b in range(N_BINS):
        if weight < BIN_EDGES[b + 1]:  # a connection weight/strength is put into its bin
            return b
    return N_BINS - 1


def weights_to_bins(weights):
    return np.array([weight_to_bin(w) for w in weights], dtype=int)


# 2. LOAD / FILTER

def _clean_columns(df):
    """Strip whitespace and any UTF-8 BOM from header names."""
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def load_connectivity_csv(filename, source_col="source", target_col="target",
                          weight_col="weight", region_col="region"):
    df = pd.read_csv(filename, dtype=str, encoding="utf-8-sig")
    _clean_columns(df)
    if df.empty:
        raise ValueError(f"{filename} contains no rows.")
    required = [source_col, target_col, weight_col, region_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nAvailable: {list(df.columns)}")
    df = df[required].copy()
    df.columns = ["source", "target", "weight", "region"]
    df = df.dropna()
    df["source"] = df["source"].str.strip()
    df["target"] = df["target"].str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="raise").astype(float)
    if not np.isfinite(df["weight"].to_numpy()).all():
        raise ValueError(f"{filename}: non-finite connection weights found.")
    if (df["weight"] < 0).any():
        raise ValueError(f"{filename}: negative connection weights found.")
    df["region"] = df["region"].astype(str).str.strip()
    return df


def select_region(df, region):
    result = df[df["region"] == str(region)].copy()  # only takes the mushroom body region
    if len(result) == 0:
        raise ValueError(f"No edges found for region '{region}'.")
    return result


def count_neurons(df):
    return len(set(df["source"]) | set(df["target"]))


def remove_isolated_neurons(df):
    """
    Drop self-connections.  A neuron is isolated if it has no connection to ANOTHER
    neuron.  Nodes exist here only through edge rows, so after removing self-loops every
    remaining node has degree >= 1.
    """
    out = df[df["source"] != df["target"]].copy()
    dropped = count_neurons(df) - count_neurons(out)
    if dropped > 0:
        print(f"  removed {dropped} neuron(s) that only had self-connections")
    return out


def subsample_by_neuron_count(df, max_neurons, seed=0, strategy="random"):
    if max_neurons is not None and (int(max_neurons) != max_neurons or max_neurons < 1):
        raise ValueError(f"max_neurons must be a positive integer or None, got {max_neurons}")
    neurons = sorted(set(df["source"]) | set(df["target"]))
    if max_neurons is None or len(neurons) <= max_neurons:
        return df
    if strategy == "top_degree":
        # keep the best-connected neurons -> far denser, more informative subgraph than random
        deg = pd.concat([df["source"], df["target"]]).value_counts()
        keep = set(deg.index[:int(max_neurons)])
    elif strategy == "random":
        rng = np.random.RandomState(seed)
        keep = set(rng.choice(neurons, size=int(max_neurons), replace=False))
    else:
        raise ValueError("sample strategy must be 'random' or 'top_degree'")
    result = df[df["source"].isin(keep) & df["target"].isin(keep)].copy()
    print(f"Subsampled from {len(neurons)} to {max_neurons} neurons "
          f"({len(result)} edges remain; neurons left without any edge in the sample drop out "
          f"of the graph, so it will have fewer nodes).")
    return result


def create_neuron_index(df):
    neurons = sorted(set(df["source"]) | set(df["target"]))  # IDs are large, so index them for the matrix
    neuron_to_index = {n: i for i, n in enumerate(neurons)}
    index_to_neuron = {i: n for n, i in neuron_to_index.items()}
    return neuron_to_index, index_to_neuron


def _edge_value(w, weight_mode):
    if weight_mode == "binned":
        return weight_to_bin(w) + 1          # 0 means no edge; 1..N_BINS are bins
    if weight_mode == "raw":
        return float(w)
    if weight_mode == "log1p":
        return float(np.log1p(w))
    raise ValueError("weight_mode must be one of: binned, raw, log1p")


def build_combinatorial_graph(df, symmetrize="sum", directed=False, weight_mode="binned"):
    """Build the adjacency matrix for the WCS solver.

    IMPORTANT: neuPrint synapses are directed. The cited Fast-WCS formulation assumes symmetric
    adjacency, so by default this makes an explicit undirected projection. For reciprocal edges,
    `symmetrize` combines the two directions:  sum = w(i->j)+w(j->i) | mean | max.
    This is an approximation, not direction-preserving matching.  directed=True instead keeps
    A[i,j] = value(w(i->j)) (non-symmetric; use with wcs_gnccp_match(directed=True)).
    weight_mode: 'binned' (default; 1..N_BINS, 0 = no edge), 'raw' (synapse counts) or 'log1p'.
    The input table is not modified.
    """
    if symmetrize not in {"sum", "mean", "max"}:
        raise ValueError("symmetrize must be one of: sum, mean, max")
    if weight_mode not in {"binned", "raw", "log1p"}:
        raise ValueError("weight_mode must be one of: binned, raw, log1p")
    if df.empty:
        raise ValueError("Cannot build a graph from an empty edge table.")

    work = df[["source", "target", "weight"]].copy()
    work["source"] = work["source"].astype(str)
    work["target"] = work["target"].astype(str)
    work["weight"] = pd.to_numeric(work["weight"], errors="raise")
    work = work[work["source"] != work["target"]].copy()  # omit self-connections
    if work.empty:
        raise ValueError("No non-self edges remain in this region.")

    neurons = sorted(set(work["source"]) | set(work["target"]))
    neuron_to_index = {n: i for i, n in enumerate(neurons)}
    index_to_neuron = {i: n for n, i in neuron_to_index.items()}
    A = np.zeros((len(neurons), len(neurons)), dtype=np.float64)

    if directed:
        edges = work.groupby(["source", "target"], as_index=False)["weight"].sum()
        for row in edges.itertuples(index=False):
            A[neuron_to_index[str(row.source)], neuron_to_index[str(row.target)]] = \
                _edge_value(row.weight, weight_mode)
        validate_adjacency(A, "directed connectome adjacency", require_symmetric=False)
        return A, neuron_to_index, index_to_neuron

    work["u"] = work[["source", "target"]].min(axis=1)
    work["v"] = work[["source", "target"]].max(axis=1)
    grouped = work.groupby(["u", "v"], as_index=False)["weight"]
    if symmetrize == "sum":
        undirected = grouped.sum()
    elif symmetrize == "mean":
        undirected = grouped.mean()
    else:
        undirected = grouped.max()
    for row in undirected.itertuples(index=False):
        i, j = neuron_to_index[str(row.u)], neuron_to_index[str(row.v)]
        A[i, j] = A[j, i] = _edge_value(row.weight, weight_mode)
    validate_adjacency(A, "connectome adjacency")
    return A, neuron_to_index, index_to_neuron


def graph_statistics(A, name="Graph"):
    n = A.shape[0]
    edges = np.count_nonzero(np.triu(A, k=1))
    possible = (n * (n - 1)) // 2
    density = edges / possible if possible > 0 else 0  # how big and how connected your graph is
    print(f"\n{name} statistics")
    print("-" * 50)
    print(f"Neurons : {n}")
    print(f"Edges   : {edges}")
    print(f"Density : {density:.6f}")
    return {"neurons": n, "edges": edges, "density": density}


def print_bin_distribution(df, name="Graph"):
    bins = weights_to_bins(df["weight"].values)
    counts = np.bincount(bins, minlength=N_BINS)
    print(f"\n{name} edge-weight distribution")
    print("-" * 60)
    for i in range(N_BINS):
        print(f"Bin {i}: {BIN_RANGES[i]:>10} -> {counts[i]} edges")


# 3. NODE (ATOM/NEURON) COST MATRIX C   (low cost = more similar)
def connectivity_signature(A):
    n = A.shape[0]
    S = np.zeros((n, N_BINS), dtype=np.float64)  # how many connections each neuron has in each bin
    for b in range(N_BINS):
        S[:, b] = np.sum(A == (b + 1), axis=1)
    totals = S.sum(axis=1, keepdims=True)        # normalize so A and B are comparable
    totals[totals == 0] = 1
    return S / totals


def build_node_cost(AG, AH):
    SA = connectivity_signature(AG)
    SB = connectivity_signature(AH)
    norm_A = np.sum(SA ** 2, axis=1)
    norm_B = np.sum(SB ** 2, axis=1)
    C = norm_A[:, None] + norm_B[None, :] - 2.0 * (SA @ SB.T)   # Cij = ||SA(i) - SB(j)||^2
    np.maximum(C, 0, out=C)
    return C
# connectivity_signature() represents each neuron by the proportion of its connections falling into
# each weight bin. build_node_cost() compares every Fly A neuron with every Fly B neuron using the
# squared Euclidean distance between signatures; lower cost = more similar local connectivity.


# 4. FROBENIUS NORMALIZATION (SCALE INVARIANCE)
def frobenius_normalize(A):
    norm = np.linalg.norm(A)
    return A / norm if norm > 0 else A


def minmax_normalize_cost(C):
    cmin, cmax = C.min(), C.max()
    if cmax - cmin < 1e-15:
        return np.zeros_like(C)
    return (C - cmin) / (cmax - cmin)
# Algorithm 1, line 2: C_hat <- (C - Cmin) / (Cmax - Cmin).


# 5. RANK-ONE FACTORIZATION OF U  (Lemma 1)
def rank_one_U(X):
    """Lemma 1: U = X 1 1^T X^T = r r^T with r = X 1_N.  O(MN + M^2)."""
    r = X.sum(axis=1)
    return r, np.outer(r, r)


# 5b. ADJACENCY VALIDATION + OPTIONAL SHARED-SCALE NORMALIZATION (Sec. IV-C)
def validate_adjacency(A, name="A", tol=1e-9, require_symmetric=True):
    """Algorithm 1, line 1: Theorems 1 and 3 require a symmetric, finite, nonnegative adjacency."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {A.shape}.")
    if not np.all(np.isfinite(A)):
        raise ValueError(f"{name} contains NaN or infinite entries.")
    if np.any(A < 0):
        raise ValueError(f"{name} contains negative weights; the model assumes A >= 0.")
    if require_symmetric and not np.allclose(A, A.T, atol=tol):
        raise ValueError(f"{name} must be symmetric (required by Theorems 1 and 3). "
                         f"For genuinely directed graphs use the directed=True generalisation.")
    return A


def validate_parameters(alpha, n_levels, schedule, dzeta, inner_iters_fw, gap_tol,
                        n_starts, perturb_eps, refine_rounds, gamma=0.5):
    def req(cond, msg):
        if not cond:
            raise ValueError(msg)
    req(0.0 <= alpha <= 1.0, f"alpha must be in [0,1], got {alpha}")
    req(gamma >= 0, f"gamma must be >= 0, got {gamma}")
    req(schedule in ("cosine", "linear"), f"schedule must be 'cosine' or 'linear', got {schedule}")
    if schedule == "cosine":
        req(int(n_levels) == n_levels and n_levels >= 2,
            f"n_levels must be an integer >= 2 for the cosine schedule (Eq. 55), got {n_levels}")
    else:
        req(0 < dzeta <= 2.0, f"dzeta must be in (0, 2] for the linear schedule, got {dzeta}")
    req(int(inner_iters_fw) == inner_iters_fw and inner_iters_fw >= 1,
        f"inner_iters_fw must be an integer >= 1, got {inner_iters_fw}")
    req(gap_tol >= 0, f"duality_gap_tol must be >= 0, got {gap_tol}")
    req(int(n_starts) == n_starts and n_starts >= 1, f"n_starts must be an integer >= 1, got {n_starts}")
    req(perturb_eps >= 0, f"perturb_eps must be >= 0, got {perturb_eps}")
    req(int(refine_rounds) == refine_rounds and refine_rounds >= 0,
        f"refine_rounds must be an integer >= 0, got {refine_rounds}")


def normalize_adjacencies(AG, AH, mode="separate"):
    """
    mode = "separate"   : A/||A||_F per graph (Eq. 8; the original behaviour of this script)
           "frobenius"  : common scale sqrt(||AG||_F ||AH||_F)       (Eq. 28)
           "shared_max" : common scale max(||AG||_max, ||AH||_max)   (Eq. 27)
           "none"       : inputs unchanged (the paper's solver default)
    Report the mode with any experiment: it changes the objective scale.
    """
    if mode == "none":
        return AG.copy(), AH.copy()
    if mode == "separate":
        return frobenius_normalize(AG), frobenius_normalize(AH)
    if mode == "shared_max":
        s = max(np.max(np.abs(AG)), np.max(np.abs(AH)))
    elif mode == "frobenius":
        s = float(np.sqrt(np.linalg.norm(AG) * np.linalg.norm(AH)))
    else:
        raise ValueError(f"Unknown norm_mode '{mode}'.")
    if s <= 0:
        return AG.copy(), AH.copy()
    return AG / s, AH / s


# 6. THEOREM 1 -- EXACT CLOSED-FORM STRUCTURAL OBJECTIVE AND GRADIENT
def structural_value_and_grad(X, AG, AH, cache=None, directed=False):
    """
    f(X) = ||R||_F^2,  R = (r r^T) o AG - X AH X^T
    Symmetric adjacency (Theorem 1, Eq. 15):
        grad = 4 [(R o AG) r] 1_N^T - 4 R X AH
    NOTE: the first term uses (R o AG) @ r (not the plain row-sum); they coincide only when r_i = 1.

    directed=True is a GENERALISATION beyond the paper (which assumes symmetric A) for
    non-symmetric AG, AH:
        grad = 2 [(S + S^T) r] 1_N^T - 2 R X AH^T - 2 R^T X AH,   S = R o AG
    which reduces to Eq. 15 when everything is symmetric.  Verified against finite differences
    in engine_audit.  If `cache` is a dict the residual R is stored.
    """
    N = X.shape[1]
    r, U = rank_one_U(X)                        # Lemma 1
    P = X @ AH                                  # GEMM 1
    R = U * AG - P @ X.T                        # GEMM 2
    f_val = float(np.sum(R * R))
    S = R * AG
    if directed:
        v = (S + S.T) @ r
        grad = 2.0 * np.outer(v, np.ones(N)) - 2.0 * (R @ (X @ AH.T)) - 2.0 * (R.T @ P)
    else:
        v = S @ r                               # (M,)  -> rank-one column broadcast
        grad = 4.0 * np.outer(v, np.ones(N)) - 4.0 * (R @ P)   # GEMM 3
    if cache is not None:
        cache["R"] = R
    return f_val, grad


def wcs_objective_and_grad(X, AG, AH, C, alpha, cache=None, directed=False):
    """Definition 2: F(X) = alpha*f(X) + (1-alpha)*trace(C^T X)."""
    f_val, f_grad = structural_value_and_grad(X, AG, AH, cache=cache, directed=directed)
    F_val = alpha * f_val + (1.0 - alpha) * float(np.sum(C * X))
    F_grad = alpha * f_grad + (1.0 - alpha) * C
    return F_val, F_grad


# ============================================================
# 7. DEFINITION 3 / PROPOSITION 2 -- GNCCP OBJECTIVE Jzeta  (Eq. 29)
# ============================================================

def J_zeta_and_grad(X, zeta, AG, AH, C, alpha, cache=None, directed=False):
    F_val, F_grad = wcs_objective_and_grad(X, AG, AH, C, alpha, cache=cache, directed=directed)
    reg_val = float(np.sum(X * X))
    reg_grad = 2.0 * X
    w = (1 - zeta) if zeta >= 0 else (1 + zeta)
    return w * F_val + zeta * reg_val, w * F_grad + zeta * reg_grad


# ============================================================
# 8. DEFINITION 4 -- EXACT BURKARD AUGMENTED HUNGARIAN LMO  (Eq. 57)
# ============================================================

def linear_minimization_oracle(cost, L):
    """
    min_{Y in P_L} <cost, Y> via one (M+N-L) x (M+N-L) assignment.
    Dummy-dummy pairings carry an infinite penalty, so every feasible matching
    has exactly L real-real pairs: no finite big-M, no top-L truncation, and the
    optimum is invariant to a uniform shift of `cost`.
    """
    M, N = cost.shape
    if L > min(M, N):
        raise ValueError(f"L={L} is larger than min(M,N)={min(M, N)}")
    if L == 0:
        return np.zeros_like(cost)

    K = M + N - L
    augmented = np.zeros((K, K), dtype=float)
    augmented[:M, :N] = cost
    augmented[M:, N:] = np.inf                  # (N-L) x (M-L) slack block

    row_ind, col_ind = linear_sum_assignment(augmented)
    Y = np.zeros((M, N), dtype=float)
    keep = (row_ind < M) & (col_ind < N)
    Y[row_ind[keep], col_ind[keep]] = 1.0

    if int(round(Y.sum())) != L:
        raise RuntimeError("Assignment did not produce exactly L matches.")
    return Y


# 9. FEASIBLE STARTING POINTS IN D_L
#    The exact Frank-Wolfe step X + lam (Y - X), lam in [0,1], never leaves D_L, so the only
#    requirement is a feasible start.
def is_in_DL(X, L, tol=1e-9):
    """Membership in D_L (Eq. 6): X >= 0, row sums <= 1, column sums <= 1, total = L."""
    return bool(X.min() >= -tol and X.sum(axis=1).max() <= 1 + tol
                and X.sum(axis=0).max() <= 1 + tol and abs(X.sum() - L) <= tol * max(1.0, L))


def feasible_start(M, N, L, eps=0.0, rng=None):
    """
    Barycenter B = L/(MN) * 1 (row sums L/M <= 1, column sums L/N <= 1, total L) plus an
    optional random perturbation (Eq. 66).  The noise is double-centered, so every row sum,
    column sum and the total are EXACTLY those of the barycenter, and it is scaled so no
    entry goes negative: the result is in D_L by construction.  The effective amplitude is
    min(eps, L/(MN)).
    """
    B = L / (M * N)
    X0 = np.full((M, N), B, dtype=float)
    if eps > 0 and rng is not None:
        E = rng.uniform(-eps, eps, size=(M, N))
        E = E - E.mean(axis=1, keepdims=True) - E.mean(axis=0, keepdims=True) + E.mean()
        amp = np.abs(E).max()
        if amp > 0:
            E *= min(1.0, B / amp)
        X0 = np.maximum(X0 + E, 0.0)
    if not is_in_DL(X0, L):
        raise RuntimeError("Constructed starting point is not in D_L.")
    return X0


# 10. THEOREM 3 -- EXACT QUARTIC LINE SEARCH (replaces the Armijo backtracking)
def quartic_coefficients(X, D, zeta, AG, AH, C, alpha, R0, directed=False):
    """
    Along X(lam) = X - lam*D, Jzeta(lam) is exactly c0 + c1 lam + ... + c4 lam^4.
    R(lam) = R0 + lam R1 + lam^2 R2   (Eqs. 44-47), with
        K = (D AH) X^T,  X AH D^T = K^T  (AH symmetric -> one GEMM saved).
    R0 is the residual already computed during the gradient evaluation.
    Returns [c0, c1, c2, c3, c4].
    """
    rX = X.sum(axis=1)
    dr = D.sum(axis=1)

    T = D @ AH
    K = T @ X.T
    Z = T @ D.T                                              # D AH D^T

    # X AH D^T equals K^T only when AH is symmetric; for directed graphs compute it explicitly.
    cross = (K + (X @ AH) @ D.T) if directed else (K + K.T)
    R1 = -(np.outer(rX, dr) + np.outer(dr, rX)) * AG + cross
    R2 = np.outer(dr, dr) * AG - Z

    s0 = float(np.sum(R0 * R0))
    s1 = 2.0 * float(np.sum(R0 * R1))
    s2 = float(np.sum(R1 * R1)) + 2.0 * float(np.sum(R0 * R2))
    s3 = 2.0 * float(np.sum(R1 * R2))
    s4 = float(np.sum(R2 * R2))

    u0 = float(np.sum(C * X))
    u1 = -float(np.sum(C * D))
    x0 = float(np.sum(X * X))
    x1 = -2.0 * float(np.sum(X * D))
    x2 = float(np.sum(D * D))

    w = (1 - zeta) if zeta >= 0 else (1 + zeta)
    return np.array([
        w * (alpha * s0 + (1 - alpha) * u0) + zeta * x0,
        w * (alpha * s1 + (1 - alpha) * u1) + zeta * x1,
        w * alpha * s2 + zeta * x2,
        w * alpha * s3,
        w * alpha * s4,
    ])


def cubic_real_roots(a, b, c, d, eps=1e-13):
    """Real roots of a x^3 + b x^2 + c x + d = 0 via Cardano (degrades to quadratic/linear)."""
    scale = max(abs(a), abs(b), abs(c), abs(d))
    if scale == 0:
        return []
    if abs(a) <= eps * scale:                                # quadratic / linear
        if abs(b) <= eps * scale:
            return [] if abs(c) <= eps * scale else [-d / c]
        disc = c * c - 4.0 * b * d
        if disc < 0:
            return []
        q = -0.5 * (c + np.copysign(np.sqrt(disc), c))
        return [0.0] if q == 0 else [q / b, d / q]
    b_, c_, d_ = b / a, c / a, d / a
    p = c_ - b_ * b_ / 3.0
    q = 2.0 * b_ ** 3 / 27.0 - b_ * c_ / 3.0 + d_
    shift = -b_ / 3.0
    disc = (q / 2.0) ** 2 + (p / 3.0) ** 3
    if disc > 0:                                             # one real root
        sq = np.sqrt(disc)
        return [float(np.cbrt(-q / 2.0 + sq) + np.cbrt(-q / 2.0 - sq) + shift)]
    if p >= 0:                                               # p = q = 0: triple root
        return [float(shift)]
    m = 2.0 * np.sqrt(-p / 3.0)                              # three real roots (trig form)
    arg = np.clip(3.0 * q / (p * m), -1.0, 1.0)
    theta = np.arccos(arg) / 3.0
    return [float(m * np.cos(theta - 2.0 * np.pi * k / 3.0) + shift) for k in range(3)]


def cardano_line_search(coeffs):
    """Global minimizer of the quartic on [0,1]: compare {0, 1} and real roots of the cubic J'."""
    c0, c1, c2, c3, c4 = coeffs
    cands = [0.0, 1.0] + [t for t in cubic_real_roots(4 * c4, 3 * c3, 2 * c2, c1) if 0.0 < t < 1.0]
    cands = np.array(cands)
    vals = c0 + cands * (c1 + cands * (c2 + cands * (c3 + cands * c4)))
    return float(cands[int(np.argmin(vals))])


# 11. PROPOSITION 2 -- VECTORIZED BLAS-3 2-OPT REFINEMENT
def _two_opt_deltas(rows, cols, AG, AH, C, alpha):
    """
    Exact change in F for every candidate move, scored with matrix products
    (P Q^T, P B^T, B Q^T) instead of Python loops.  Binary X in P_L gives
    f(X) = ||P - Q||_F^2 with P = AG[rows,rows], Q = AH[cols,cols].
    Returns (swap[L,L], col_replace[L,U], row_replace[V,L], unmatched_cols, unmatched_rows).
    """
    M, N = AG.shape[0], AH.shape[0]
    L = len(rows)
    un_c = np.setdiff1d(np.arange(N), cols)
    un_r = np.setdiff1d(np.arange(M), rows)

    P = AG[np.ix_(rows, rows)]
    Q = AH[np.ix_(cols, cols)]
    d, q = np.diag(P), np.diag(Q)
    Cm = C[np.ix_(rows, cols)]
    cd = np.diag(Cm)
    P2, Q2 = (P * P).sum(axis=1), (Q * Q).sum(axis=1)
    E = P2[:, None] + Q2[None, :] - 2.0 * (P @ Q.T)          # E[a,b] = sum_k (P[a,k]-Q[b,k])^2
    Ed = np.diag(E)
    dq2 = (d - q) ** 2

    # Move 2: exchange targets of rows i and j
    swap = None
    if L >= 2:
        N1 = E - (d[:, None] - Q.T) ** 2 - (P - q[None, :]) ** 2
        O = Ed[:, None] - dq2[:, None] - (P - Q) ** 2
        ds = (2.0 * (N1 + N1.T - O - O.T)
              + (d[:, None] - q[None, :]) ** 2 + (d[None, :] - q[:, None]) ** 2
              - dq2[:, None] - dq2[None, :])
        du = Cm + Cm.T - cd[:, None] - cd[None, :]
        swap = alpha * ds + (1 - alpha) * du
        np.fill_diagonal(swap, np.inf)

    old = 2.0 * (Ed - dq2) + dq2                             # structural cost touching index i

    # Move 1: replace the target column of row i by an unmatched column u
    col_rep = None
    if len(un_c) > 0:
        B = AH[np.ix_(un_c, cols)]                           # (U, L)
        B2 = (B * B).sum(axis=1)
        EB = P2[:, None] + B2[None, :] - 2.0 * (P @ B.T)     # (L, U)
        duu = np.diag(AH)[un_c]
        new = 2.0 * (EB - (d[:, None] - B.T) ** 2) + (d[:, None] - duu[None, :]) ** 2
        ds = new - old[:, None]
        du = C[np.ix_(rows, un_c)] - cd[:, None]
        col_rep = alpha * ds + (1 - alpha) * du

    # Move 3: replace matched source row i by an unmatched source row v
    row_rep = None
    if len(un_r) > 0:
        B = AG[np.ix_(un_r, rows)]                           # (V, L)
        B2 = (B * B).sum(axis=1)
        EB = B2[:, None] + Q2[None, :] - 2.0 * (B @ Q.T)     # (V, L)
        dvv = np.diag(AG)[un_r]
        new = 2.0 * (EB - (B - q[None, :]) ** 2) + (dvv[:, None] - q[None, :]) ** 2
        ds = new - old[None, :]
        du = C[np.ix_(un_r, cols)] - cd[None, :]
        row_rep = alpha * ds + (1 - alpha) * du

    return swap, col_rep, row_rep, un_c, un_r


def two_opt_refine(Xb, AG, AH, C, alpha, max_rounds=20, tol=1e-12):
    """
    Best-improvement local search over the three move families.  Never accepts a
    non-improving move, so F is monotone non-increasing.  Hitting max_rounds does
    NOT certify a local optimum.  Returns (X_refined, moves_applied).
    """
    M, N = Xb.shape
    rows = np.where(Xb.sum(axis=1) > 0.5)[0]
    if len(rows) == 0:
        return Xb, 0
    cols = np.argmax(Xb[rows], axis=1)
    applied = 0
    for _ in range(max_rounds):
        swap, col_rep, row_rep, un_c, un_r = _two_opt_deltas(rows, cols, AG, AH, C, alpha)
        best_val, best = -tol, None
        if swap is not None:
            i, j = np.unravel_index(np.argmin(swap), swap.shape)
            if swap[i, j] < best_val:
                best_val, best = swap[i, j], ("swap", i, j)
        if col_rep is not None:
            i, u = np.unravel_index(np.argmin(col_rep), col_rep.shape)
            if col_rep[i, u] < best_val:
                best_val, best = col_rep[i, u], ("col", i, u)
        if row_rep is not None:
            v, i = np.unravel_index(np.argmin(row_rep), row_rep.shape)
            if row_rep[v, i] < best_val:
                best_val, best = row_rep[v, i], ("row", i, v)
        if best is None:
            break
        kind, a, b = best
        if kind == "swap":
            cols[a], cols[b] = cols[b], cols[a]
        elif kind == "col":
            cols[a] = un_c[b]
        else:
            rows[a] = un_r[b]
        applied += 1

    X = np.zeros_like(Xb)
    X[rows, cols] = 1.0
    return X, applied


# 12. ALGORITHM 1, STAGE 1 -- 2D STRUCTURAL WCS-GNCCP MATCHING (Fast-WCS)
def zeta_schedule(n_levels=8, schedule="cosine", dzeta=0.05):
    """Eq. 55 cosine continuation (default) or the original linear 1 -> -1 sweep."""
    if schedule == "cosine":
        if n_levels < 2:
            return np.array([1.0])
        return np.cos(np.pi * np.arange(n_levels) / (n_levels - 1))
    if schedule == "linear":
        return np.linspace(1.0, -1.0, int(round(2.0 / dzeta)) + 1)
    raise ValueError(f"Unknown schedule '{schedule}'.")


def problem_matrices(AG, AH, norm_mode="separate", C_override=None):
    """Everything the objective needs: scaled adjacencies and the min-max node cost C
    (or a caller-supplied, already normalised C)."""
    AGh, AHh = normalize_adjacencies(AG, AH, mode=norm_mode)
    if C_override is not None:
        C = np.asarray(C_override, dtype=float)
        if C.shape != (AG.shape[0], AH.shape[0]) or not np.all(np.isfinite(C)):
            raise ValueError("C_override must be a finite (M, N) matrix.")
    else:
        C = minmax_normalize_cost(build_node_cost(AG, AH))
    return AGh, AHh, C


def _run_homotopy(X, L, zetas, AGh, AHh, C, alpha, inner_iters_fw, gap_tol, directed=False):
    history = []
    for zeta in zetas:
        J_val, gap, lam = 0.0, 0.0, 0.0
        for _ in range(inner_iters_fw):
            cache = {}
            J_val, grad = J_zeta_and_grad(X, float(zeta), AGh, AHh, C, alpha, cache=cache, directed=directed)  # Thm 1
            Y = linear_minimization_oracle(grad, L)                                         # Def. 4
            D = X - Y
            gap = float(np.sum(grad * D))                                                   # Thm 2
            if abs(gap) < gap_tol:        # stopping test, NOT a global certificate at non-convex levels
                break
            lam = cardano_line_search(
                quartic_coefficients(X, D, float(zeta), AGh, AHh, C, alpha, cache["R"], directed=directed))    # Thm 3
            X = X - lam * D                                       # convex step, stays in D_L
        history.append({"zeta": float(zeta), "J": J_val, "gap": gap, "lambda": lam,
                        "max_X": float(X.max())})
    return X, history


def wcs_gnccp_match(AG, AH, L, alpha=0.7, gamma=0.5,
                    n_levels=8, schedule="cosine", dzeta=0.05,
                    inner_iters_fw=3, duality_gap_tol=1e-3,
                    norm_mode="separate", n_starts=1, perturb_eps=0.03,
                    refine=True, refine_rounds=20, seed=0, verbose=False,
                    directed=False, C_override=None, include_barycenter=True):
    """
    Fast-WCS Algorithm 1, Stage 1 (2D structural matching). Returns:
      X_final : discretized partial permutation (M,N) in P_L
      k_score : similarity kernel exp(-gamma F* / L)
      history : per-zeta-step diagnostics (of the best start)
    Options:
      directed=True        general (non-symmetric) gradient/line search; 2-opt refinement is disabled
      C_override           use this (already normalised) M x N node-cost matrix instead of the
                           connectivity-signature cost
      include_barycenter   if False every start is a perturbed start (independent random inits)
    """
    validate_adjacency(AG, "AG", require_symmetric=not directed)
    validate_adjacency(AH, "AH", require_symmetric=not directed)
    validate_parameters(alpha, n_levels, schedule, dzeta, inner_iters_fw, duality_gap_tol,
                        n_starts, perturb_eps, refine_rounds, gamma)
    M, N = AG.shape[0], AH.shape[0]
    if not (1 <= L <= min(M, N)):
        raise ValueError(f"L must be between 1 and min(M,N)={min(M, N)}")
    if directed and refine:
        refine = False   # the vectorized 2-opt deltas are derived for symmetric adjacency
        if verbose:
            print("  directed=True: 2-opt refinement disabled (deltas assume symmetric adjacency).")

    AGh, AHh, C = problem_matrices(AG, AH, norm_mode, C_override)

    zetas = zeta_schedule(n_levels, schedule, dzeta)
    rng = np.random.RandomState(seed)

    best = None
    for k in range(int(n_starts)):
        if k == 0 and include_barycenter:
            X0 = feasible_start(M, N, L)                        # uniform barycenter
        else:
            X0 = feasible_start(M, N, L, perturb_eps, rng)      # Eq. 66, exactly in D_L

        X, history = _run_homotopy(X0, L, zetas, AGh, AHh, C, alpha, inner_iters_fw,
                                   duality_gap_tol, directed=directed)
        Xd = linear_minimization_oracle(-X, L)                  # Algorithm 1, line 21
        moves = 0
        if refine:
            Xd, moves = two_opt_refine(Xd, AGh, AHh, C, alpha, max_rounds=refine_rounds)
        F_star, _ = wcs_objective_and_grad(Xd, AGh, AHh, C, alpha, directed=directed)   # unrelaxed objective

        if verbose:
            print(f"  start {k}: F*={F_star:.6f}  (2-opt moves applied: {moves})")
        if best is None or F_star < best[0]:
            best = (F_star, Xd, history)

    F_star, X_final, history = best
    if verbose:
        for h in history:
            print(f"zeta={h['zeta']:6.2f} | J={h['J']:12.4f} | gap={h['gap']:10.4f} | "
                  f"lam={h['lambda']:.3f} | max(X)={h['max_X']:.4f}")
    k_score = float(np.exp(-gamma * F_star / max(L, 1)))        # Algorithm 1, line 25
    return X_final, k_score, history


# 12b. ENGINE AUDIT (Sec. VII-E) -- run with --self_test
#      Numerical self-check of the solver internals (gradient, line search, LMO, 2-opt).
def engine_audit(seed=0, verbose=True):
    rng = np.random.RandomState(seed)
    M, N, L, alpha = 6, 9, 5, 0.7

    def sym(n):
        A = rng.rand(n, n)
        A = (A + A.T) / 2
        np.fill_diagonal(A, 0)
        return A
    AG, AH, C = sym(M), sym(N), rng.rand(M, N)
    res = {}

    # (1) Theorem 1 gradient vs central finite differences
    X = rng.rand(M, N) * 0.3
    _, g = wcs_objective_and_grad(X, AG, AH, C, alpha)
    fd, h = np.zeros_like(X), 1e-6
    for i in range(M):
        for j in range(N):
            Xp, Xm = X.copy(), X.copy()
            Xp[i, j] += h
            Xm[i, j] -= h
            fd[i, j] = (wcs_objective_and_grad(Xp, AG, AH, C, alpha)[0]
                        - wcs_objective_and_grad(Xm, AG, AH, C, alpha)[0]) / (2 * h)
    res["gradient_rel_err"] = float(np.linalg.norm(g - fd) / np.linalg.norm(fd))

    # (2) quartic coefficients reproduce J_zeta(X - lam D) exactly
    err = 0.0
    for zeta in (0.8, 0.0, -0.6):
        Y = linear_minimization_oracle(rng.randn(M, N), L)
        D = X - Y
        cache = {}
        J_zeta_and_grad(X, zeta, AG, AH, C, alpha, cache=cache)
        c = quartic_coefficients(X, D, zeta, AG, AH, C, alpha, cache["R"])
        for lam in (0.0, 0.3, 0.7, 1.0):
            direct = J_zeta_and_grad(X - lam * D, zeta, AG, AH, C, alpha)[0]
            poly = c[0] + c[1] * lam + c[2] * lam ** 2 + c[3] * lam ** 3 + c[4] * lam ** 4
            err = max(err, abs(direct - poly) / max(1.0, abs(direct)))
    res["quartic_coeff_err"] = float(err)

    # (3) Cardano step never worse than a dense grid, and cubic roots match numpy
    worst, miss = 0.0, 0
    for _ in range(2000):
        c = rng.randn(5) * (10.0 ** rng.randint(-3, 3, size=5))
        lam = cardano_line_search(c)
        grid = np.linspace(0, 1, 2001)
        pv = lambda t: c[0] + t * (c[1] + t * (c[2] + t * (c[3] + t * c[4])))
        worst = max(worst, pv(lam) - pv(grid).min())
        a = rng.randn(4)
        mine = sorted(cubic_real_roots(*a))
        ref = sorted(r.real for r in np.roots(a) if abs(r.imag) < 1e-7)
        if len(mine) < len(ref):
            miss += 1
    res["cardano_worse_than_grid"] = float(worst)
    res["cardano_missed_roots"] = miss

    # (4) augmented LMO: exactly L pairs and optimal vs brute force
    from itertools import permutations
    Mc, Nc, Lc = 3, 4, 2
    cost = rng.randn(Mc, Nc)
    Y = linear_minimization_oracle(cost, Lc)
    best = min(
        sum(cost[i, p[i]] for i in rows_) for rows_ in
        [(a, b) for a in range(Mc) for b in range(a + 1, Mc)]
        for p in permutations(range(Nc), Lc) for p in [dict(zip(rows_, p))]
    )
    res["lmo_exact_L"] = int(Y.sum()) == Lc
    res["lmo_optimal"] = bool(np.isclose(np.sum(cost * Y), best))

    # (5) 2-opt deltas vs brute-force recomputation of F
    Xb = linear_minimization_oracle(rng.randn(M, N), L)
    rows = np.where(Xb.sum(1) > 0.5)[0]
    cols = np.argmax(Xb[rows], axis=1)
    F0 = wcs_objective_and_grad(Xb, AG, AH, C, alpha)[0]
    swap, col_rep, row_rep, un_c, un_r = _two_opt_deltas(rows, cols, AG, AH, C, alpha)

    def F_of(r_, c_):
        Xn = np.zeros((M, N))
        Xn[r_, c_] = 1.0
        return wcs_objective_and_grad(Xn, AG, AH, C, alpha)[0]
    e2 = 0.0
    for i in range(L):
        for j in range(L):
            if i != j:
                c2 = cols.copy()
                c2[i], c2[j] = c2[j], c2[i]
                e2 = max(e2, abs(swap[i, j] - (F_of(rows, c2) - F0)))
    for i in range(L):
        for ui, u in enumerate(un_c):
            c2 = cols.copy()
            c2[i] = u
            e2 = max(e2, abs(col_rep[i, ui] - (F_of(rows, c2) - F0)))
        for vi, v in enumerate(un_r):
            r2 = rows.copy()
            r2[i] = v
            e2 = max(e2, abs(row_rep[vi, i] - (F_of(r2, cols) - F0)))
    res["two_opt_delta_err"] = float(e2)

    # (6) starting points lie exactly in D_L for rectangular / partial cases
    rs = np.random.RandomState(seed + 1)
    res["feasible_starts"] = all(
        is_in_DL(feasible_start(m_, n_, l_, e_, rs), l_)
        for (m_, n_, l_) in [(5, 8, 3), (8, 5, 5), (6, 6, 6), (7, 9, 1)] for e_ in (0.0, 0.03, 0.5))

    # (7) directed generalisation: gradient vs finite differences, quartic vs direct evaluation
    AGd, AHd = rng.rand(M, M), rng.rand(N, N)
    np.fill_diagonal(AGd, 0)
    np.fill_diagonal(AHd, 0)
    Xd_ = rng.rand(M, N) * 0.3
    _, gd = wcs_objective_and_grad(Xd_, AGd, AHd, C, alpha, directed=True)
    fdd = np.zeros_like(Xd_)
    for i in range(M):
        for j in range(N):
            Xp, Xm = Xd_.copy(), Xd_.copy()
            Xp[i, j] += h
            Xm[i, j] -= h
            fdd[i, j] = (wcs_objective_and_grad(Xp, AGd, AHd, C, alpha, directed=True)[0]
                         - wcs_objective_and_grad(Xm, AGd, AHd, C, alpha, directed=True)[0]) / (2 * h)
    res["directed_gradient_rel_err"] = float(np.linalg.norm(gd - fdd) / np.linalg.norm(fdd))
    Dd = Xd_ - linear_minimization_oracle(rng.randn(M, N), L)
    cache = {}
    J_zeta_and_grad(Xd_, 0.4, AGd, AHd, C, alpha, cache=cache, directed=True)
    cd = quartic_coefficients(Xd_, Dd, 0.4, AGd, AHd, C, alpha, cache["R"], directed=True)
    derr = 0.0
    for lam in (0.0, 0.3, 0.7, 1.0):
        direct = J_zeta_and_grad(Xd_ - lam * Dd, 0.4, AGd, AHd, C, alpha, directed=True)[0]
        poly = cd[0] + cd[1] * lam + cd[2] * lam ** 2 + cd[3] * lam ** 3 + cd[4] * lam ** 4
        derr = max(derr, abs(direct - poly) / max(1.0, abs(direct)))
    res["directed_quartic_err"] = float(derr)

    ok = (res["directed_gradient_rel_err"] < 1e-6 and res["directed_quartic_err"] < 1e-9
          and res["feasible_starts"] and res["gradient_rel_err"] < 1e-6 and res["quartic_coeff_err"] < 1e-9
          and res["cardano_worse_than_grid"] < 1e-9 and res["cardano_missed_roots"] == 0
          and res["lmo_exact_L"] and res["lmo_optimal"] and res["two_opt_delta_err"] < 1e-9)
    if verbose:
        print("\nFast-WCS engine audit")
        print("-" * 50)
        for k, v in res.items():
            print(f"{k:28s}: {v}")
        print("PASS" if ok else "FAIL")
    return ok


# 12c. PROPOSITION 6 -- RELABELING INVARIANCE SELF-TEST
def verify_relabeling_invariance(seed=0, M=6, N=8, alpha=0.7, verbose=True):
    rng = np.random.RandomState(seed)
    AG = rng.randint(0, 8, size=(M, M)).astype(float)
    AG = (AG + AG.T) / 2
    np.fill_diagonal(AG, 0)
    AH = rng.randint(0, 8, size=(N, N)).astype(float)
    AH = (AH + AH.T) / 2
    np.fill_diagonal(AH, 0)
    C = rng.rand(M, N)
    X = rng.rand(M, N)

    P = np.eye(M)[rng.permutation(M)]
    Q = np.eye(N)[rng.permutation(N)]
    AGp, AHp, Cp, Xp = P @ AG @ P.T, Q @ AH @ Q.T, P @ C @ Q.T, P @ X @ Q.T

    F1, _ = wcs_objective_and_grad(X, AG, AH, C, alpha)
    F2, _ = wcs_objective_and_grad(Xp, AGp, AHp, Cp, alpha)
    ok = np.isclose(F1, F2)

    if verbose:
        print("\nRelabeling-invariance self-test (Proposition 6)")
        print("------------------------------------------------")
        print(f"F(X; AG, AH, C)          = {F1:.10f}")
        print(f"F(PXQ^T; AG', AH', C')   = {F2:.10f}")
        print("PASS" if ok else "FAIL")
    return ok


# ============================================================
# 13. OPTIONAL POST-HOC COORDINATE DIAGNOSTIC -- NOT A MATCHING REFINEMENT
#     Kabsch evaluates rigid alignment of the already predicted pairs. It does
#     not establish their identity and does not change the Stage-1 assignment.
# ============================================================

def kabsch_rmsd(P_pts, Q_pts):
    """Post-hoc rigid alignment RMSD for an already supplied correspondence.
    This is a diagnostic only; it does not validate neuron identity."""
    Pc = P_pts - P_pts.mean(axis=0)
    Qc = Q_pts - Q_pts.mean(axis=0)
    H = Pc.T @ Qc
    W, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ W.T))
    D = np.diag([1, 1, d])
    R_star = Vt.T @ D @ W.T
    P_aligned = (R_star @ Pc.T).T
    rmsd = float(np.sqrt(np.mean(np.sum((P_aligned - Qc) ** 2, axis=1))))
    return rmsd, R_star


def stage2_kabsch(match_pairs, coords_A, coords_B, A_from_idx, B_from_idx,
                  scale_A=1.0, scale_B=1.0):
    """
    Post-hoc diagnostic only.  Kabsch is a rigid fit (rotation + translation, no scaling),
    so coordinates of the two flies must be in the SAME physical unit.  The hemibrain uses
    8 nm voxels; other datasets use different units -- pass scale_A / scale_B (multipliers to a
    common unit such as nm) or the RMSD is not meaningful.
    """
    coordA_map = coords_A.set_index("bodyId")[["x", "y", "z"]]
    coordB_map = coords_B.set_index("bodyId")[["x", "y", "z"]]

    P_pts, Q_pts = [], []
    for i, j in match_pairs:
        bidA, bidB = A_from_idx[i], B_from_idx[j]
        if bidA in coordA_map.index and bidB in coordB_map.index:
            P_pts.append(coordA_map.loc[bidA].values.astype(float) * scale_A)
            Q_pts.append(coordB_map.loc[bidB].values.astype(float) * scale_B)

    if len(P_pts) < 3:
        print(f"Stage 2 skipped: only {len(P_pts)} matched neurons have known "
              f"soma coordinates on both sides (need >= 3 for Kabsch).")
        return None

    rmsd, R_star = kabsch_rmsd(np.array(P_pts), np.array(Q_pts))
    return {"rmsd": rmsd, "n_points": len(P_pts), "rotation": R_star}


def hybrid_kernel(k2D, k3D, beta=0.5):
    """Definition 6."""
    if k3D is None:
        return k2D
    return (1 - beta) * k2D + beta * k3D


# ------------------------------------------------------------
# 13c. STAGE 3D -- COORDINATE-AWARE MATCHING
#      The soma positions enter the optimisation through the node cost:
#          C = (1 - w3d) * C_2D + w3d * C_3D
#      C_3D = distance between a fly-A soma (after a similarity alignment into fly B's frame)
#      and a fly-B soma.  The alignment is estimated from the current matches themselves
#      (robust, trimmed Umeyama), then the matching is re-solved; repeat align_iters times.
# ------------------------------------------------------------

def _load_coords(path):
    frame = _clean_columns(pd.read_csv(path, dtype={"bodyId": str}, encoding="utf-8-sig"))
    required = {"bodyId", "x", "y", "z"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}; found {list(frame.columns)}")
    frame["bodyId"] = frame["bodyId"].astype(str).str.strip()
    for col in ("x", "y", "z"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["bodyId", "x", "y", "z"]).drop_duplicates("bodyId")
    if frame.empty:
        raise ValueError(f"No usable bodyId/x/y/z rows in {path}.")
    return frame


def coords_to_array(coords_df, id_from_idx, n, scale=1.0):
    """(n,3) array of coordinates in graph-index order; NaN where a neuron has no soma coordinate."""
    d = coords_df.set_index("bodyId")[["x", "y", "z"]].astype(float)
    return d.reindex([id_from_idx[i] for i in range(n)]).to_numpy() * scale


def umeyama_align(Pp, Qp, with_scale=True):
    """Least-squares similarity transform q ~ s R p + t (Umeyama 1991); reflections excluded."""
    mp, mq = Pp.mean(0), Qp.mean(0)
    Pc, Qc = Pp - mp, Qp - mq
    var_p = (Pc ** 2).sum() / len(Pp)
    U, D, Vt = np.linalg.svd(Qc.T @ Pc / len(Pp))
    Sg = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Sg[2, 2] = -1
    R = U @ Sg @ Vt
    s = float((D * np.diag(Sg)).sum() / var_p) if with_scale and var_p > 0 else 1.0
    return s, R, mq - s * R @ mp


def robust_umeyama(Pp, Qp, iters=6, keep=0.6, min_pts=4):
    """Trimmed Umeyama: repeatedly refit on the best `keep` fraction so wrong pairs are down-weighted."""
    n = len(Pp)
    if n < min_pts:
        return None
    idx = np.arange(n)
    for _ in range(iters):
        s, R, t = umeyama_align(Pp[idx], Qp[idx])
        res = np.linalg.norm((s * (R @ Pp.T)).T + t - Qp, axis=1)
        new = np.argsort(res)[:max(min_pts, int(np.ceil(keep * n)))]
        if set(new) == set(idx):
            break
        idx = new
    s, R, t = umeyama_align(Pp[idx], Qp[idx])
    return s, R, t, np.linalg.norm((s * (R @ Pp.T)).T + t - Qp, axis=1)


def _cost3d(cA, cB, tf):
    """Min-max normalised A->B distance matrix after alignment; missing coordinates get the mean cost."""
    from scipy.spatial.distance import cdist
    s, R, t = tf
    TA = (s * (R @ cA.T)).T + t
    va, vb = ~np.isnan(TA).any(1), ~np.isnan(cB).any(1)
    D = np.full((len(cA), len(cB)), np.nan)
    D[np.ix_(va, vb)] = cdist(TA[va], cB[vb])
    D = np.where(np.isnan(D), np.nanmean(D) if np.isfinite(D).any() else 0.0, D)
    return minmax_normalize_cost(D), TA


def _pairs_with_coords(X, cA, cB):
    out = []
    for i in range(X.shape[0]):
        if X[i].max() > 0.5:
            j = int(np.argmax(X[i]))
            if not np.isnan(cA[i]).any() and not np.isnan(cB[j]).any():
                out.append((i, j))
    return out


def _chamfer(cA, cB, tf, trim=0.8):
    """Trimmed mean nearest-neighbour distance from aligned fly-A somata to fly-B somata."""
    from scipy.spatial import cKDTree
    s_, R_, t_ = tf
    A, B = cA[~np.isnan(cA).any(1)], cB[~np.isnan(cB).any(1)]
    d, _ = cKDTree(B).query((s_ * (R_ @ A.T)).T + t_)
    d = np.sort(d)[:max(1, int(np.ceil(trim * len(d))))]
    return float(d.mean())


def icp_alignment(cA, cB, n_restarts=200, iters=15, trim=0.8, seed=0):
    """
    Correspondence-free initial alignment: centroid + RMS-radius scale matching, then trimmed ICP from many
    random rotations (no reflections); the best trimmed chamfer distance wins.  Uses only the two point
    clouds -- the 2D matching is not needed.  Scale is fixed by the RMS-radius ratio (so ICP cannot
    shrink the cloud to cheat).  Returns ((s, R, t), chamfer_score).
    """
    from scipy.spatial import cKDTree
    A, B = cA[~np.isnan(cA).any(1)], cB[~np.isnan(cB).any(1)]
    if len(A) < 4 or len(B) < 4:
        return None, np.nan
    ca, cb = A.mean(0), B.mean(0)
    s0 = np.sqrt(((B - cb) ** 2).sum(1).mean()) / max(np.sqrt(((A - ca) ** 2).sum(1).mean()), 1e-12)
    An = (A - ca) * s0                      # fly A rescaled/centred onto fly B's size, origin at its centroid
    Bn = B - cb
    tree = cKDTree(Bn)
    rng = np.random.RandomState(seed)
    k = max(4, int(np.ceil(trim * len(An))))
    best, best_score = None, np.inf
    for r in range(n_restarts):
        if r == 0:
            R = np.eye(3)
        else:
            R, _ = np.linalg.qr(rng.randn(3, 3))
            if np.linalg.det(R) < 0:
                R[:, 0] *= -1
        t = np.zeros(3)
        for _ in range(iters):
            T = (R @ An.T).T + t
            d, nn = tree.query(T)
            keep = np.argsort(d)[:k]
            _, R, t = umeyama_align(An[keep], Bn[nn[keep]], with_scale=False)
        d, _ = tree.query((R @ An.T).T + t)
        score = float(np.sort(d)[:k].mean())
        if score < best_score:
            best_score, best = score, (R, t)
    R, t = best
    # compose back to  q = s R p + t_total  in the original coordinates:  q = cb + R(s0 (p - ca)) + t
    return (float(s0), R, cb + R @ (-s0 * ca) + t), best_score


def stage_3d_matching(AG, AH, X2, L, cA, cB, C2, w3d, align_iters, solver_kw, align_init="icp", seed=0):
    """
    Align fly A to fly B -> build C_3D -> re-solve with C = (1-w3d) C_2D + w3d C_3D; repeat.
    align_init='icp'     : round 0 uses a correspondence-free ICP alignment of the two soma clouds
                           (does not trust the 2D matches); later rounds refit from the current matches and
                           keep the refit only if it lowers the chamfer distance.
    align_init='matches' : every round fits the alignment from the current (2D) matches.
    Returns (X, k_score or None, history or None, info, residuals).
    """
    from scipy.spatial.distance import cdist
    X, k_new, hist_new = X2, None, None
    info = {"w3d": w3d, "align_init": align_init, "iterations_used": 0,
            "coverage_A": float((~np.isnan(cA).any(1)).mean()),
            "coverage_B": float((~np.isnan(cB).any(1)).mean()), "alignment_quality": []}
    print(f"Soma-coordinate coverage: fly A {info['coverage_A']:.1%}, fly B {info['coverage_B']:.1%}")

    tf_cur = None
    if align_init == "icp":
        tf_cur, score = icp_alignment(cA, cB, seed=seed)
        if tf_cur is None:
            print("  ICP alignment impossible (too few coordinates).")
        else:
            info["icp_chamfer"] = score
            print(f"  ICP initial alignment: scale={tf_cur[0]:.4g}, trimmed chamfer = {score:.4g}")

    for it in range(int(align_iters)):
        if not (it == 0 and tf_cur is not None):
            pr = _pairs_with_coords(X, cA, cB)
            if len(pr) >= 4:
                fit = robust_umeyama(cA[[i for i, _ in pr]], cB[[j for _, j in pr]])
                cand = fit[:3]
                if tf_cur is None or _chamfer(cA, cB, cand) <= _chamfer(cA, cB, tf_cur):
                    tf_cur = cand
            elif tf_cur is None:
                print(f"  3D iter {it}: only {len(pr)} matches have coordinates on both sides (need >= 4) -- stopping.")
                break
        s_, R_, t_ = tf_cur
        C3, TA = _cost3d(cA, cB, tf_cur)
        va, vb = ~np.isnan(TA).any(1), ~np.isnan(cB).any(1)
        rand_med = float(np.median(cdist(TA[va], cB[vb])))        # typical distance of an arbitrary A-B pair
        pr = _pairs_with_coords(X, cA, cB)
        med_res = float(np.median([np.linalg.norm(TA[i] - cB[j]) for i, j in pr])) if pr else float("nan")
        ratio = med_res / rand_med if rand_med > 0 else float("nan")
        info["alignment_quality"].append(ratio)
        C = (1.0 - w3d) * C2 + w3d * C3
        X_prev = X
        X, k_new, hist_new = wcs_gnccp_match(AG, AH, L, C_override=C, **solver_kw)
        same = len(set(zip(*np.nonzero(X))) & set(zip(*np.nonzero(X_prev))))
        info["iterations_used"] = it + 1
        print(f"  3D iter {it}: scale={s_:.4g} | median residual of the PREVIOUS matches / median random-pair "
              f"distance = {ratio:.3f} | {same}/{L} pairs unchanged vs previous round")
        if same == L:
            break

    residuals = {}
    if info["iterations_used"] > 0:
        TA = (tf_cur[0] * (tf_cur[1] @ cA.T)).T + tf_cur[2]
        pr = _pairs_with_coords(X, cA, cB)
        residuals = {(i, j): float(np.linalg.norm(TA[i] - cB[j])) for i, j in pr}
        if residuals:
            info["final_median_residual"] = float(np.median(list(residuals.values())))
        info["final_scale"] = float(tf_cur[0])
        info["final_chamfer"] = _chamfer(cA, cB, tf_cur)
    return X, k_new, hist_new, info, residuals


def save_run_settings(settings, path="run_settings.json"):
    import platform, sys
    import scipy
    settings = dict(settings)
    settings["environment"] = {"python": sys.version.split()[0], "platform": platform.platform(),
                               "numpy": np.__version__, "scipy": scipy.__version__,
                               "pandas": pd.__version__}

    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (set, tuple)):
            return list(o)
        return str(o)
    with open(path, "w") as f:
        json.dump(settings, f, indent=2, default=conv)
    print(f"Saved experimental settings and results summary to {path}")


# ============================================================
# 14. REAL-WORLD EXPERIMENT -- STOPS AT 2D BY DEFAULT
# ============================================================

def run_real_experiment(fly_A_file, fly_B_file, region, alpha=0.7, gamma=0.5,
                        dzeta=0.05, inner_iters_fw=3, max_neurons=None, seed=0,
                        run_stage2=False, coords_A_file=None, coords_B_file=None,
                        gamma_3d=0.1, beta=0.5,
                        n_levels=8, schedule="cosine", gap_tol=1e-3,
                        norm_mode="separate", n_starts=1, refine=True,
                        refine_rounds=20, rmsd_squared=True, self_test=False,
                        symmetrize="sum", match_count=None,
                        coords_scale_A=1.0, coords_scale_B=1.0,
                        settings_out="run_settings.json",
                        sample_strategy="random", match_3d=False, w3d=0.5, align_iters=3, align_init="icp"):
    """
    Stage 1 returns predicted 2D structural correspondences. Optional Kabsch is a post-hoc
    coordinate DIAGNOSTIC on those pairs (not a matching refinement; k3D / k_hybrid are
    diagnostics derived from it and are not used to choose the matching).
    """
    print("=" * 80)
    print("REAL-WORLD -- FAST-WCS MATCHING (Algorithm 1, Stage 1: 2D only by default)")
    print("=" * 80)

    validate_parameters(alpha, n_levels, schedule, dzeta, inner_iters_fw, gap_tol,
                        n_starts, 0.03, refine_rounds, gamma)
    if max_neurons is not None and (int(max_neurons) != max_neurons or max_neurons < 1):
        raise ValueError(f"max_neurons must be a positive integer or None, got {max_neurons}")
    if match_count is not None and (int(match_count) != match_count or match_count < 1):
        raise ValueError(f"match_count must be a positive integer, got {match_count}")

    ok = verify_relabeling_invariance(verbose=False)
    if not ok:
        raise RuntimeError("Relabeling-invariance self-test failed -- stopping before running on real data.")
    print("Relabeling-invariance self-test (Prop. 1): PASS")
    if self_test and not engine_audit():
        raise RuntimeError("Fast-WCS engine audit failed -- stopping before running on real data.")

    A_df = load_connectivity_csv(fly_A_file)
    B_df = load_connectivity_csv(fly_B_file)
    A_region = select_region(A_df, region)
    B_region = select_region(B_df, region)
    print(f"\nLoaded region '{region}': Fly A {count_neurons(A_region)} neurons / {len(A_region)} edges; "
          f"Fly B {count_neurons(B_region)} neurons / {len(B_region)} edges")

    if max_neurons is not None:
        print(f"\nCapping each fly at {max_neurons} neurons for feasibility (seed={seed}, strategy={sample_strategy})...")
        A_region = subsample_by_neuron_count(A_region, max_neurons, seed=seed, strategy=sample_strategy)
        B_region = subsample_by_neuron_count(B_region, max_neurons, seed=seed + 1, strategy=sample_strategy)

    A_region = remove_isolated_neurons(A_region)
    B_region = remove_isolated_neurons(B_region)
    print(f"\nAfter sampling/pruning: Fly A {count_neurons(A_region)} neurons / {len(A_region)} edges | "
          f"Fly B {count_neurons(B_region)} neurons / {len(B_region)} edges")

    print_bin_distribution(A_region, "Fly A")
    print_bin_distribution(B_region, "Fly B")

    AG, A_to_idx, A_from_idx = build_combinatorial_graph(A_region, symmetrize=symmetrize)
    AH, B_to_idx, B_from_idx = build_combinatorial_graph(B_region, symmetrize=symmetrize)
    graph_statistics(AG, "Fly A (symmetrized)")
    graph_statistics(AH, "Fly B (symmetrized)")

    M, N = AG.shape[0], AH.shape[0]
    if match_count is not None:
        L = int(match_count)
    else:
        L = min(M, N)
        print("WARNING: match_count not supplied; all nodes in the smaller graph are forced to match. "
              "Use --match_count for a defensible partial matching.")
    if not (1 <= L <= min(M, N)):
        raise ValueError(f"match_count must be between 1 and min(M,N)={min(M, N)}")
    n_zeta = len(zeta_schedule(n_levels, schedule, dzeta))
    print(f"\nGraph sizes M={M}, N={N}; L = {L}  ({n_zeta} {schedule} zeta levels x <= {inner_iters_fw} FW "
          f"steps x {n_starts} start(s); one O((M+N-L)^3) Hungarian per FW step, closed-form Cardano "
          f"line search, norm_mode='{norm_mode}')")

    print("\nRunning Fast-WCS Stage 1 (2D structural matching)...")
    t0 = time.time()
    X_final, k2D, history = wcs_gnccp_match(
        AG, AH, L=L, alpha=alpha, gamma=gamma,
        n_levels=n_levels, schedule=schedule, dzeta=dzeta,
        inner_iters_fw=inner_iters_fw, duality_gap_tol=gap_tol,
        norm_mode=norm_mode, n_starts=n_starts, refine=refine,
        refine_rounds=refine_rounds, seed=seed, verbose=True,
    )
    solve_time = time.time() - t0
    print(f"\nStage 1 solve time: {solve_time:.2f} s")
    print(f"TOTAL RUNTIME SO FAR: {time.time() - _start_time:.2f} seconds")

    match_pairs = [(i, int(np.argmax(X_final[i]))) for i in range(M) if X_final[i].max() > 0.5]
    print(f"\n2D similarity kernel k2D = {k2D:.6f}  ({len(match_pairs)} matches)")

    # ---------------- 3D stage: coordinate-aware matching and/or Kabsch diagnostic ----------------
    coords_A = coords_B = None
    if run_stage2 or match_3d:
        if coords_A_file and coords_B_file:
            coords_A, coords_B = _load_coords(coords_A_file), _load_coords(coords_B_file)
            if coords_scale_A == 1.0 and coords_scale_B == 1.0:
                print("NOTE: no coordinate scales supplied. The 3D matching stage fits a similarity transform "
                      "(absorbs a uniform unit difference); the Kabsch diagnostic does not.")
        else:
            print("\n3D steps requested but --coords_A_file/--coords_B_file were not both supplied -- skipping.")

    stage3d_info, residuals = None, {}
    if match_3d and coords_A is not None:
        print(f"\nRunning 3D coordinate-aware matching (w3d={w3d}, up to {align_iters} alignment rounds)...")
        cA = coords_to_array(coords_A, A_from_idx, M, coords_scale_A)
        cB = coords_to_array(coords_B, B_from_idx, N, coords_scale_B)
        C2 = minmax_normalize_cost(build_node_cost(AG, AH))
        solver_kw = dict(alpha=alpha, gamma=gamma, n_levels=n_levels, schedule=schedule, dzeta=dzeta,
                         inner_iters_fw=inner_iters_fw, duality_gap_tol=gap_tol, norm_mode=norm_mode,
                         n_starts=n_starts, refine=refine, refine_rounds=refine_rounds, seed=seed)
        X3, k3, hist3, stage3d_info, residuals = stage_3d_matching(
            AG, AH, X_final, L, cA, cB, C2, w3d, align_iters, solver_kw, align_init=align_init, seed=seed)
        if stage3d_info["iterations_used"] > 0:
            n_changed = L - len(set(zip(*np.nonzero(X3))) & set(zip(*np.nonzero(X_final))))
            X_final, k2D, history = X3, k3, hist3
            match_pairs = [(i, int(np.argmax(X_final[i]))) for i in range(M) if X_final[i].max() > 0.5]
            print(f"3D-aware matching done: {n_changed} of {L} pairs differ from the 2D-only matching. "
                  f"Combined-objective kernel = {k2D:.6f}")
        else:
            print("3D matching could not be applied; keeping the 2D matching.")

    k3D = None
    if run_stage2 and coords_A is not None:
        stage2_result = stage2_kabsch(match_pairs, coords_A, coords_B, A_from_idx, B_from_idx,
                                      scale_A=coords_scale_A, scale_B=coords_scale_B)
        if stage2_result is not None:
            rm = stage2_result["rmsd"]
            k3D = float(np.exp(-gamma_3d * (rm ** 2 if rmsd_squared else rm)))
            print(f"3D Kabsch diagnostic: RMSD = {rm:.3f} over {stage2_result['n_points']} points "
                  f"-> k3D = {k3D:.6f}")
    elif not match_3d:
        print("\n3D steps not requested -- stopping at 2D structural matching.")

    k_hybrid = hybrid_kernel(k2D, k3D, beta=beta)
    if k3D is not None:
        print(f"\nDiagnostic hybrid score k_hybrid = {k_hybrid:.6f}  (beta={beta}; not used for matching)")

    used_3d = bool(stage3d_info and stage3d_info["iterations_used"] > 0)
    matches = []
    for i, j in match_pairs:
        row = {"fly_A_neuron": A_from_idx[i], "fly_B_neuron": B_from_idx[j]}
        if used_3d:
            row["aligned_3d_residual"] = residuals.get((i, j), float("nan"))
        matches.append(row)
    matches_df = pd.DataFrame(matches)
    out_name = "predicted_neuron_matches_wcs_3d.csv" if used_3d else "predicted_neuron_matches_wcs_2d.csv"
    matches_df.to_csv(out_name, index=False)
    print(f"\nSaved {out_name} ({len(matches_df)} matches)")
    print(matches_df.head(20))

    export_visualization_data(
        AG=AG, AH=AH, X_final=X_final,
        A_from_idx=A_from_idx, B_from_idx=B_from_idx,
        match_pairs=match_pairs, k2D=k2D, history=history,
        alpha=alpha, gamma=gamma, max_neurons=max_neurons, seed=seed,
    )

    save_run_settings({
        "parameters": {"alpha": alpha, "gamma": gamma, "norm_mode": norm_mode, "symmetrize": symmetrize,
                       "L": L, "seed": seed, "n_starts": n_starts, "schedule": schedule,
                       "n_levels": n_levels, "dzeta": dzeta, "inner_iters_fw": inner_iters_fw,
                       "gap_tol": gap_tol, "refine": refine, "refine_rounds": refine_rounds,
                       "max_neurons": max_neurons, "region": region,
                       "perturb_eps": 0.03, "match_count_arg": match_count,
                       "sample_strategy": sample_strategy, "match_3d": match_3d, "w3d": w3d,
                       "align_iters": align_iters, "align_init": align_init},
        "inputs": {"fly_A_file": fly_A_file, "fly_B_file": fly_B_file},
        "graph_sizes": {"M": M, "N": N, "edges_A": int(np.count_nonzero(np.triu(AG, 1))),
                        "edges_B": int(np.count_nonzero(np.triu(AH, 1)))},
        "stage3d": stage3d_info,
        "results": {"k2D": k2D, "n_matches": len(match_pairs), "solve_time_s": solve_time,
                    "k3D": k3D, "k_hybrid": k_hybrid if k3D is not None else None},
    }, path=settings_out)

    return {
        "AG": AG, "AH": AH, "X_final": X_final,
        "k2D": k2D, "k3D": k3D, "k_hybrid": k_hybrid,
        "matches": matches_df, "history": history,
    }


def export_visualization_data(AG, AH, X_final, A_from_idx, B_from_idx,
                              match_pairs, k2D, history=None,
                              alpha=0.7, gamma=0.5, max_neurons=None, seed=0,
                              out_json="viz_data.json", out_js="viz_data.js"):
    """Export complete graph matrices, X correspondence, signatures, and matches for the UI."""
    M, N = AG.shape[0], AH.shape[0]
    L = len(match_pairs)
    SA = connectivity_signature(AG)
    SB = connectivity_signature(AH)
    C_raw = build_node_cost(AG, AH)
    C_norm = minmax_normalize_cost(C_raw)

    def edges_from_A(A):
        iu = np.triu_indices(A.shape[0], k=1)
        vals = A[iu]
        mask = vals > 0
        src = iu[0][mask].tolist()
        tgt = iu[1][mask].tolist()
        bins = (vals[mask] - 1).astype(int).tolist()
        return [{"s": int(s), "t": int(t), "b": int(b)} for s, t, b in zip(src, tgt, bins)]

    a_to_b = {int(i): int(j) for i, j in match_pairs}
    b_to_a = {int(j): int(i) for i, j in match_pairs}

    hist_clean = []
    if history:
        for h in history:
            hist_clean.append({
                "zeta": float(h.get("zeta", 0)),
                "J": float(h.get("J", 0)),
                "gap": float(h.get("gap", 0)),
                "max_X": float(h.get("max_X", 0))
            })

    viz = {
        "M": int(M), "N": int(N), "L": int(L),
        "A_ids": [str(A_from_idx[i]) for i in range(M)],
        "B_ids": [str(B_from_idx[j]) for j in range(N)],
        "AG_edges": edges_from_A(AG),
        "AH_edges": edges_from_A(AH),
        "AG_matrix": AG.tolist(),
        "AH_matrix": AH.tolist(),
        "AG_deg": np.count_nonzero(AG, axis=1).astype(int).tolist(),
        "AH_deg": np.count_nonzero(AH, axis=1).astype(int).tolist(),
        "SA": SA.tolist(),
        "SB": SB.tolist(),
        "C": C_norm.tolist(),
        "X": X_final.tolist(),
        "matches": [{"i": int(i), "j": int(j)} for i, j in match_pairs],
        "aToB": a_to_b,
        "bToA": b_to_a,
        "k2D": float(k2D),
        "bin_ranges": BIN_RANGES,
        "alpha": float(alpha), "gamma": float(gamma),
        "max_neurons": int(max_neurons) if max_neurons is not None else M,
        "seed": int(seed),
        "datasetA": "hemibrain:v1.2.2 (MB(R))",
        "datasetB": "flywire-fafb:v783b (MB_CA_R, MB_PED_R, MB_ML_R, MB_VL_R)",
        "history": hist_clean,
    }

    with open(out_json, "w") as f:
        json.dump(viz, f)

    with open(out_js, "w") as f:
        f.write("window.CONNECTOME_DATA = " + json.dumps(viz) + ";\n")

    print(f"Exported visualization data to {out_json} and {out_js} (M={M}, N={N}, {len(match_pairs)} matches)")
    return viz


# ============================================================
# 15. MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast-WCS connectome matching on real data (Stage 1: 2D by default)")
    parser.add_argument("--fly_A_file", type=str, default="flyA_hemibrain_MB.csv",
                        help="Real Fly A CSV (source,target,weight,region).")
    parser.add_argument("--fly_B_file", type=str, default="flyB_fafb_MB.csv",
                        help="Real Fly B CSV (source,target,weight,region).")
    parser.add_argument("--region", type=str, default="mushroom_body")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--gamma", type=float, default=0.5, help="2D similarity-kernel bandwidth.")
    parser.add_argument("--schedule", choices=["cosine", "linear"], default="cosine",
                        help="Continuation schedule (Eq. 55). 'linear' reproduces the old 1 -> -1 sweep.")
    parser.add_argument("--levels", type=int, default=8, help="Cosine continuation levels K (paper default 8).")
    parser.add_argument("--dzeta", type=float, default=0.05, help="Step for --schedule linear only.")
    parser.add_argument("--inner_iters_fw", type=int, default=3,
                        help="Max Frank-Wolfe steps per zeta level (paper default 3).")
    parser.add_argument("--gap_tol", type=float, default=1e-3,
                        help="Frank-Wolfe duality-gap stopping tolerance (paper default 1e-3).")
    parser.add_argument("--norm_mode", choices=["separate", "frobenius", "shared_max", "none"],
                        default="separate",
                        help="Adjacency scaling ('separate' = per-graph Frobenius, Eq. 8; "
                             "'frobenius'/'shared_max' = common scale, Eqs. 27-28; 'none' = paper solver "
                             "default). Report it with results.")
    parser.add_argument("--n_starts", type=int, default=1, help="Multi-start restarts (Eq. 66).")
    parser.add_argument("--no_refine", action="store_true", help="Disable 2-opt refinement (Prop. 2).")
    parser.add_argument("--refine_rounds", type=int, default=20,
                        help="Max 2-opt rounds (one best move per round). Hitting the cap is not a local-optimum certificate.")
    parser.add_argument("--max_neurons", type=int, default=300, help="Cap on neurons per fly.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling / multi-start RNG seed.")
    parser.add_argument("--self_test", action="store_true", help="Run the Fast-WCS engine audit first.")
    parser.add_argument("--fetch", action="store_true",
                        help="Fetch fresh CSVs from neuPrint first (needs NEUPRINT_TOKEN).")
    parser.add_argument("--fetch_coords", action="store_true",
                        help="Also fetch soma coordinates; implies --fetch.")
    parser.add_argument("--run_stage2", action="store_true",
                        help="Optional post-hoc Kabsch coordinate diagnostic (not a matching step).")
    parser.add_argument("--coords_A_file", type=str, default=None)
    parser.add_argument("--coords_B_file", type=str, default=None)
    parser.add_argument("--gamma_3d", type=float, default=0.1)
    parser.add_argument("--k3d_linear_rmsd", action="store_true",
                        help="Use k3D = exp(-gamma_3d*RMSD) instead of exp(-gamma_3d*RMSD^2) (Fig. 1).")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--symmetrize", choices=["sum", "mean", "max"], default="sum",
                        help="How reciprocal directed synapse weights are projected to the symmetric graph the WCS formulation requires.")
    parser.add_argument("--match_count", type=int, default=None,
                        help="Number L of pairs to match. Default: min(M,N) (forces full matching).")
    parser.add_argument("--coords_scale_A", type=float, default=1.0,
                        help="Multiplier taking fly A coordinates to a common unit (e.g. nm).")
    parser.add_argument("--coords_scale_B", type=float, default=1.0,
                        help="Multiplier taking fly B coordinates to a common unit (e.g. nm).")
    parser.add_argument("--sample_strategy", choices=["random", "top_degree"], default="random",
                        help="How --max_neurons picks neurons. 'top_degree' keeps the best-connected neurons "
                             "(much denser graphs than random sampling).")
    parser.add_argument("--match_3d", action="store_true",
                        help="Coordinate-aware matching: soma positions enter the node cost "
                             "C = (1-w3d) C_2D + w3d C_3D (needs coordinate CSVs).")
    parser.add_argument("--w3d", type=float, default=0.5, help="Weight of the 3D cost in [0,1] (default 0.5).")
    parser.add_argument("--align_iters", type=int, default=3, help="Alignment/re-solve rounds for --match_3d.")
    parser.add_argument("--align_init", choices=["icp", "matches"], default="icp",
                        help="How the fly-A -> fly-B coordinate alignment is initialised: 'icp' = from the two soma "
                             "clouds alone (recommended); 'matches' = from the 2D matches (can lock in errors).")
    parser.add_argument("--fetch_coords_only", action="store_true",
                        help="Fetch only soma coordinates from neuPrint (keeps existing connectivity CSVs).")
    parser.add_argument("--settings_out", type=str, default="run_settings.json",
                        help="Where to save settings + results summary.")
    args = parser.parse_args()

    if not 0.0 <= args.w3d <= 1.0:
        parser.error("--w3d must be in [0, 1]")
    if args.match_3d or args.run_stage2 or args.fetch_coords_only:
        args.coords_A_file = args.coords_A_file or "flyA_hemibrain_MB_coords.csv"
        args.coords_B_file = args.coords_B_file or "flyB_fafb_MB_coords.csv"
    if args.fetch_coords_only:
        fetch_coordinates_only(args.coords_A_file, args.coords_B_file)
        if not (args.match_3d or args.run_stage2):
            raise SystemExit(0)

    if args.fetch or args.fetch_coords:
        if args.fetch_coords:
            if args.coords_A_file is None:
                args.coords_A_file = "flyA_hemibrain_MB_coords.csv"
            if args.coords_B_file is None:
                args.coords_B_file = "flyB_fafb_MB_coords.csv"
        fetch_real_datasets(fetch_coords=args.fetch_coords,
                            fly_A_file=args.fly_A_file, fly_B_file=args.fly_B_file,
                            coords_A_file=args.coords_A_file or "flyA_hemibrain_MB_coords.csv",
                            coords_B_file=args.coords_B_file or "flyB_fafb_MB_coords.csv")

    run_real_experiment(
        fly_A_file=args.fly_A_file, fly_B_file=args.fly_B_file, region=args.region,
        alpha=args.alpha, gamma=args.gamma, dzeta=args.dzeta,
        inner_iters_fw=args.inner_iters_fw, max_neurons=args.max_neurons, seed=args.seed,
        run_stage2=args.run_stage2, coords_A_file=args.coords_A_file, coords_B_file=args.coords_B_file,
        gamma_3d=args.gamma_3d, beta=args.beta, n_levels=args.levels, schedule=args.schedule,
        gap_tol=args.gap_tol, norm_mode=args.norm_mode, n_starts=args.n_starts,
        refine=not args.no_refine, refine_rounds=args.refine_rounds,
        rmsd_squared=not args.k3d_linear_rmsd, self_test=args.self_test,
        symmetrize=args.symmetrize, match_count=args.match_count,
        coords_scale_A=args.coords_scale_A, coords_scale_B=args.coords_scale_B,
        settings_out=args.settings_out,
        sample_strategy=args.sample_strategy, match_3d=args.match_3d, w3d=args.w3d,
        align_iters=args.align_iters, align_init=args.align_init,
    )
