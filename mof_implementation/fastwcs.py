"""
fastwcs.py -- Fast-WCS: exact analytical Frank-Wolfe / GNCCP homotopy for
Weighted Common Subgraph (WCS) matching.  Implements the paper's

  Lemma 1      rank-one factorisation  U = r r^T,  r = X 1
  Theorem 1    closed-form gradient    4 (R o A_G) r 1^T - 4 R X A_H
  Theorem 2    Frank-Wolfe gap stopping test
  Theorem 3    exact quartic line search (Cardano)
  Sec. V-B     Burkard augmented Hungarian oracle (exact cardinality L)
  Prop. 2      vectorised BLAS-3 2-opt refinement (three move families)
  Eq. (55)     cosine continuation schedule
  Eq. (66)     perturbed multi-start restarts

plus baselines used for comparison: SciPy FAQ, feature-only Hungarian and a
re-implementation of Yang et al.'s RLX1F (Eq. 3 gradient, 11-point line search).

Objective (Eq. 7-8):  F(X) = alpha*||U o A_G - X A_H X^T||_F^2 + (1-alpha)*tr(C^T X)
All adjacency matrices must be symmetric.
"""
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def scale_adjacency(AG, AH, mode="none"):
    """Optional shared-scale normalisation (Sec. IV-C, Eqs. 27-28)."""
    if mode == "none":
        return AG, AH
    if mode == "shared_max":
        s = max(np.abs(AG).max(), np.abs(AH).max())
    elif mode == "frobenius":
        s = np.sqrt(np.linalg.norm(AG) * np.linalg.norm(AH))
    else:
        raise ValueError(mode)
    return (AG / s, AH / s) if s > 0 else (AG, AH)


def pairs_to_X(rows, cols, shape):
    X = np.zeros(shape)
    X[rows, cols] = 1.0
    return X


def wcs_objective_pairs(AG, AH, C, rows, cols, alpha):
    """F(X) for a discrete partial permutation given as index pairs."""
    rows = np.asarray(rows); cols = np.asarray(cols)
    P = AG[np.ix_(rows, rows)]
    Q = AH[np.ix_(cols, cols)]
    f = float(np.sum((P - Q) ** 2))
    u = 0.0 if C is None else float(C[rows, cols].sum())
    return alpha * f + (1 - alpha) * u


def wcs_objective_dense(X, AG, AH, C, alpha):
    r = X.sum(1)
    R = np.outer(r, r) * AG - X @ AH @ X.T
    u = 0.0 if C is None else float(np.sum(C * X))
    return alpha * float(np.sum(R * R)) + (1 - alpha) * u


# ----------------------------------------------------------------------------
# Sec. V-B: exact Burkard augmented Hungarian oracle
# ----------------------------------------------------------------------------

def augmented_assignment(cost, L):
    """argmin_{Y in P_L} <cost, Y> with exactly L pairs, via the augmented
    (M+N-L)x(M+N-L) square assignment of Eq. (57)."""
    M, N = cost.shape
    if not (1 <= L <= min(M, N)):
        raise ValueError("need 1 <= L <= min(M, N)")
    n = M + N - L
    A = np.zeros((n, n))
    A[:M, :N] = cost
    if (N - L) > 0 and (M - L) > 0:
        A[M:, N:] = np.inf            # dummy-dummy pairs forbidden
    r, c = linear_sum_assignment(A)
    keep = (r < M) & (c < N)
    r, c = r[keep], c[keep]
    assert len(r) == L, "augmented oracle must return exactly L pairs"
    return r, c


def topL_assignment(cost, L):
    """RLX1F-style oracle: full rectangular assignment, then keep the L pairs
    with the smallest cost entries (heuristic truncation)."""
    r, c = linear_sum_assignment(cost)
    order = np.argsort(cost[r, c])[:L]
    return r[order], c[order]


# ----------------------------------------------------------------------------
# Theorem 1 / Lemma 1: gradient
# ----------------------------------------------------------------------------

def structural_gradient(X, AG, AH):
    """Exact gradient of f(X)=||U o A_G - X A_H X^T||_F^2 (Theorem 1).
    Returns gradient and cached products."""
    r = X.sum(1)                              # Lemma 1: U = r r^T
    P = X @ AH                                # GEMM 1
    XAX = P @ X.T                             # GEMM 2
    R = np.outer(r, r) * AG - XAX
    v = (R * AG) @ r                          # rank-one broadcast term
    G = 4.0 * v[:, None] - 4.0 * (R @ P)      # GEMM 3
    return G, r, P, XAX, R


# ----------------------------------------------------------------------------
# Theorem 3: exact quartic line search
# ----------------------------------------------------------------------------

def _polish(roots, a, b, c, d, iters=6):
    """Newton steps remove Cardano's cancellation error on badly scaled cubics;
    a step is accepted only if it reduces |f| (guards against divergence near double roots)."""
    out = []
    for x in roots:
        f = ((a * x + b) * x + c) * x + d
        for _ in range(iters):
            df = (3 * a * x + 2 * b) * x + c
            if df == 0.0:
                break
            xn = x - f / df
            fn = ((a * xn + b) * xn + c) * xn + d
            if abs(fn) < abs(f):
                x, f = xn, fn
            else:
                break
        out.append(x)
    return out


def cubic_real_roots(a, b, c, d):
    """Real roots of a x^3 + b x^2 + c x + d: Cardano / trigonometric form with Newton
    polishing.  If the scaled residual of any root is still above 1e-10 (extremely
    ill-conditioned cubic), fall back to the companion-matrix roots (np.roots) so that no
    root is lost; this fallback is an implementation safeguard, not part of the paper."""
    roots = _polish(_cubic_cardano(a, b, c, d), a, b, c, d)
    scale = max(abs(a), abs(b), abs(c), abs(d), 1e-300)
    bad = any(abs(((a * x + b) * x + c) * x + d) > 1e-10 * scale * max(1.0, abs(x)) ** 3
              for x in roots)
    if bad and abs(a) > 0:
        roots = [z.real for z in np.roots([a, b, c, d]) if abs(z.imag) < 1e-9 * max(1.0, abs(z))]
    return roots


def _cubic_cardano(a, b, c, d):
    scale = max(abs(a), abs(b), abs(c), abs(d), 1e-300)
    if abs(a) < 1e-13 * scale:
        if abs(b) < 1e-13 * scale:
            return [] if abs(c) < 1e-13 * scale else [-d / c]
        disc = c * c - 4 * b * d
        if disc < 0:
            return []
        s = np.sqrt(disc)
        return [(-c + s) / (2 * b), (-c - s) / (2 * b)]
    p2, q2, r2 = b / a, c / a, d / a
    p = q2 - p2 * p2 / 3.0
    q = 2.0 * p2 ** 3 / 27.0 - p2 * q2 / 3.0 + r2
    shift = -p2 / 3.0
    disc = (q / 2.0) ** 2 + (p / 3.0) ** 3
    eps = 1e-14 * max(1.0, abs(q), abs(p))
    if disc > eps:                            # one real root
        sd = np.sqrt(disc)
        t = np.cbrt(-q / 2.0 + sd) + np.cbrt(-q / 2.0 - sd)
        return [t + shift]
    if abs(p) < 1e-300:
        return [shift + np.cbrt(-q)]
    if p > 0:                                 # numerical safety (disc ~ 0)
        return [shift + np.cbrt(-q / 2.0) * 2.0]
    m = 2.0 * np.sqrt(-p / 3.0)
    arg = np.clip(3.0 * q / (p * m), -1.0, 1.0)
    th = np.arccos(arg)
    return [shift + m * np.cos((th - 2.0 * np.pi * k) / 3.0) for k in range(3)]


def quartic_coefficients(X, Y, rows, cols, P, XAX, AG, AH, C, alpha, zeta):
    """Coefficients c0..c4 of J_zeta(lambda) along X(lam) = X - lam*D, D = X - Y
    (Theorem 3, Eqs. 43-50).  P = X A_H and XAX = X A_H X^T are reused from the
    gradient evaluation; Y A_H is a row gather because Y is a partial permutation."""
    D = X - Y
    rX = X.sum(1); dr = D.sum(1)
    YA = np.zeros_like(X); YA[rows] = AH[cols]
    DA = P - YA                               # D A_H
    K = DA @ X.T                              # K = D A_H X^T
    DAD = DA @ D.T                            # D A_H D^T
    R0 = np.outer(rX, rX) * AG - XAX
    R1 = -(np.outer(rX, dr) + np.outer(dr, rX)) * AG + (K + K.T)
    R2 = np.outer(dr, dr) * AG - DAD
    s0 = np.sum(R0 * R0); s1 = 2 * np.sum(R0 * R1)
    s2 = np.sum(R1 * R1) + 2 * np.sum(R0 * R2)
    s3 = 2 * np.sum(R1 * R2); s4 = np.sum(R2 * R2)
    if C is None:
        u0 = u1 = 0.0
    else:
        u0 = np.sum(C * X); u1 = -np.sum(C * D)
    x0 = np.sum(X * X); x1 = -2 * np.sum(X * D); x2 = np.sum(D * D)
    w = (1 - zeta) if zeta >= 0 else (1 + zeta)
    c0 = w * (alpha * s0 + (1 - alpha) * u0) + zeta * x0
    c1 = w * (alpha * s1 + (1 - alpha) * u1) + zeta * x1
    c2 = w * alpha * s2 + zeta * x2
    c3 = w * alpha * s3
    c4 = w * alpha * s4
    return np.array([c0, c1, c2, c3, c4])


def minimise_quartic_on_unit_interval(c):
    """argmin_{lam in [0,1]} sum c_k lam^k : endpoints + real roots of the cubic derivative."""
    cand = [0.0, 1.0]
    for rt in cubic_real_roots(4 * c[4], 3 * c[3], 2 * c[2], c[1]):
        if 0.0 < rt < 1.0:
            cand.append(float(rt))
    vals = [np.polyval(c[::-1], t) for t in cand]
    return cand[int(np.argmin(vals))]


# ----------------------------------------------------------------------------
# Prop. 2: vectorised 2-opt refinement (exact deltas, three move families)
# ----------------------------------------------------------------------------

def _move_deltas(AG, AH, C, a, b, UG, UH, alpha):
    """Exact change in F for every candidate move; returns (d1, d2, d3):
    d1: L x |UH| target replacement; d2: L x L pairwise exchange;
    d3: |UG| x L source replacement.  Uses P Q^T, P B^T, B Q^T (BLAS-3)."""
    P = AG[np.ix_(a, a)]; Q = AH[np.ix_(b, b)]
    L = len(a)
    pd = np.diag(P); qd = np.diag(Q)
    P2 = (P * P).sum(1); Q2 = (Q * Q).sum(1)
    diff2 = (P - Q) ** 2
    e0 = diff2.sum(1); dg = (pd - qd) ** 2
    Cm = None if C is None else C[np.ix_(a, b)]
    cd = None if C is None else np.diag(Cm)
    out1 = out3 = None
    # ---- Move 1: target column replacement
    if len(UH) > 0:
        B = AH[np.ix_(UH, b)]                 # U x L
        diagU = AH[UH, UH]
        B2 = (B * B).sum(1)
        S = (P2[:, None] + B2[None, :] - 2 * (P @ B.T)
             - (pd[:, None] ** 2 + B.T ** 2 - 2 * pd[:, None] * B.T))
        old = e0 - dg
        d = 2 * (S - old[:, None]) + (pd[:, None] - diagU[None, :]) ** 2 - dg[:, None]
        du = 0.0 if C is None else (C[np.ix_(a, UH)] - cd[:, None])
        out1 = alpha * d + (1 - alpha) * du
    # ---- Move 2: pairwise target exchange
    T1 = P2[:, None] + Q2[None, :] - 2 * (P @ Q.T)
    new_k = T1 - (pd[:, None] - Q.T) ** 2 - (P - qd[None, :]) ** 2
    new_m = T1.T - (P - qd[:, None]) ** 2 - (pd[None, :] - Q) ** 2
    old_k = e0[:, None] - dg[:, None] - diff2
    old_m = e0[None, :] - dg[None, :] - diff2
    d2 = (2 * (new_k + new_m - old_k - old_m)
          + (pd[:, None] - qd[None, :]) ** 2 + (pd[None, :] - qd[:, None]) ** 2
          - dg[:, None] - dg[None, :])
    du2 = 0.0 if C is None else (Cm + Cm.T - cd[:, None] - cd[None, :])
    out2 = alpha * d2 + (1 - alpha) * du2
    out2 = np.where(np.triu(np.ones((L, L), bool), 1), out2, np.inf)
    # ---- Move 3: source row replacement
    if len(UG) > 0:
        Bg = AG[np.ix_(UG, a)]                # V x L
        diagV = AG[UG, UG]
        Bg2 = (Bg * Bg).sum(1)
        S = (Bg2[:, None] + Q2[None, :] - 2 * (Bg @ Q.T)
             - (Bg ** 2 + qd[None, :] ** 2 - 2 * Bg * qd[None, :]))
        old = e0 - dg
        d = 2 * (S - old[None, :]) + (diagV[:, None] - qd[None, :]) ** 2 - dg[None, :]
        du = 0.0 if C is None else (C[np.ix_(UG, b)] - cd[None, :])
        out3 = alpha * d + (1 - alpha) * du
    return out1, out2, out3


def two_opt_refine(AG, AH, C, rows, cols, alpha, max_rounds=20, tol=1e-12):
    """Best-improvement local search over the three move families."""
    M, N = AG.shape[0], AH.shape[0]
    a = np.array(rows).copy(); b = np.array(cols).copy()
    n_moves = 0
    for _ in range(max_rounds):
        UG = np.setdiff1d(np.arange(M), a); UH = np.setdiff1d(np.arange(N), b)
        d1, d2, d3 = _move_deltas(AG, AH, C, a, b, UG, UH, alpha)
        best = (0.0, None)
        if d1 is not None and d1.size:
            k, u = np.unravel_index(np.argmin(d1), d1.shape)
            if d1[k, u] < best[0]:
                best = (d1[k, u], ("t", k, u))
        k, m = np.unravel_index(np.argmin(d2), d2.shape)
        if d2[k, m] < best[0]:
            best = (d2[k, m], ("x", k, m))
        if d3 is not None and d3.size:
            v, k3 = np.unravel_index(np.argmin(d3), d3.shape)
            if d3[v, k3] < best[0]:
                best = (d3[v, k3], ("s", v, k3))
        if best[1] is None or best[0] > -tol:
            break
        kind, i, j = best[1]
        if kind == "t":
            b[i] = UH[j]
        elif kind == "x":
            b[i], b[j] = b[j], b[i]
        else:
            a[j] = UG[i]
        n_moves += 1
    return a, b, n_moves


# ----------------------------------------------------------------------------
# Multi-start initialisation, Eq. (66)
# ----------------------------------------------------------------------------

def project_DL(X, L, iters=200):
    """Approximate projection onto D_L (alternating clip / row-col scaling /
    total-mass rescaling).  The paper does not specify P_{D_L}; this is our choice."""
    X = np.clip(X, 0.0, 1.0)
    for _ in range(iters):
        X = X / np.maximum(X.sum(1), 1.0)[:, None]
        X = X / np.maximum(X.sum(0), 1.0)[None, :]
        X = X * (L / X.sum())
        if X.sum(1).max() <= 1 + 1e-9 and X.sum(0).max() <= 1 + 1e-9:
            break
    return X


# ----------------------------------------------------------------------------
# Algorithm 1: one homotopy path
# ----------------------------------------------------------------------------

def _homotopy_path(AG, AH, C, L, alpha, K, T_fw, tol, X0, schedule="cosine"):
    M, N = AG.shape[0], AH.shape[0]
    X = X0.copy()
    for k in range(K):
        if K == 1:
            zeta = 1.0
        elif schedule == "cosine":
            zeta = float(np.cos(np.pi * k / (K - 1)))          # Eq. (55)
        else:
            zeta = 1.0 - 2.0 * k / (K - 1)
        w = (1 - zeta) if zeta >= 0 else (1 + zeta)
        for _ in range(T_fw):
            Gf, r, P, XAX, R = structural_gradient(X, AG, AH)
            grad = alpha * Gf + ((1 - alpha) * C if C is not None else 0.0)
            gJ = w * grad + 2.0 * zeta * X
            rows, cols = augmented_assignment(gJ, L)
            Y = pairs_to_X(rows, cols, X.shape)
            gap = float(np.sum(gJ * (X - Y)))                   # Theorem 2
            if abs(gap) < tol:
                break
            c = quartic_coefficients(X, Y, rows, cols, P, XAX, AG, AH, C, alpha, zeta)
            lam = minimise_quartic_on_unit_interval(c)          # Theorem 3
            X = (1 - lam) * X + lam * Y
    return X


def fastwcs(AG, AH, C, L, alpha=0.7, K=8, T_fw=3, tol=1e-3, starts=1, eps=0.03,
            include_uniform=False, refine=True, refine_rounds=20, seed=0,
            schedule="cosine"):
    """Fast-WCS.  starts=1 -> single start from the uniform barycentre.
    starts>1 -> perturbed multi-start (Eq. 66, X0 = P_DL(L/(MN) 1 + E), E~U(-eps,eps));
    the candidate with the smallest discrete objective F* (Eq. 67) is kept, then refined."""
    t0 = time.perf_counter()
    M, N = AG.shape[0], AH.shape[0]
    rng = np.random.default_rng(seed)
    base = np.full((M, N), L / (M * N))
    inits = []
    if starts == 1:
        inits = [base]
    else:
        if include_uniform:
            inits.append(base)
        while len(inits) < starts:
            inits.append(project_DL(base + rng.uniform(-eps, eps, (M, N)), L))
    best = None
    for X0 in inits:
        X = _homotopy_path(AG, AH, C, L, alpha, K, T_fw, tol, X0, schedule)
        rows, cols = augmented_assignment(-X, L)
        F = wcs_objective_pairs(AG, AH, C, rows, cols, alpha)
        if best is None or F < best[0]:
            best = (F, rows, cols)
    F, rows, cols = best
    n_moves = 0
    if refine:
        rows, cols, n_moves = two_opt_refine(AG, AH, C, rows, cols, alpha, refine_rounds)
        F = wcs_objective_pairs(AG, AH, C, rows, cols, alpha)
    return dict(rows=rows, cols=cols, F=F, n_moves=n_moves,
                time=time.perf_counter() - t0)


# ----------------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------------

def reduce_to_L(AG, AH, C, rows, cols, L, alpha):
    """Convert a larger matching into exactly L pairs by repeatedly removing the
    pair that contributes most to F (no ground-truth access)."""
    rows = list(rows); cols = list(cols)
    while len(rows) > L:
        a = np.array(rows); b = np.array(cols)
        P = AG[np.ix_(a, a)]; Q = AH[np.ix_(b, b)]
        D2 = (P - Q) ** 2
        contrib = alpha * (2 * D2.sum(1) - np.diag(D2))
        if C is not None:
            contrib = contrib + (1 - alpha) * C[a, b]
        k = int(np.argmax(contrib))
        rows.pop(k); cols.pop(k)
    return np.array(rows), np.array(cols)


def faq_baseline(AG, AH, C, L, alpha, seed=0):
    """SciPy FAQ on zero-padded graphs (topology only), reduced to exact L."""
    from scipy.optimize import quadratic_assignment
    t0 = time.perf_counter()
    M, N = AG.shape[0], AH.shape[0]
    A = np.zeros((N, N)); A[:M, :M] = AG
    res = quadratic_assignment(A, AH, method="faq",
                               options={"maximize": True, "rng": seed})
    perm = res.col_ind
    rows = np.arange(M); cols = perm[:M]
    rows, cols = reduce_to_L(AG, AH, C, rows, cols, L, alpha)
    return dict(rows=rows, cols=cols, F=wcs_objective_pairs(AG, AH, C, rows, cols, alpha),
                time=time.perf_counter() - t0)


def feature_only_baseline(AG, AH, C, L, alpha):
    """Exact-L Hungarian on the unary cost only (no topology)."""
    t0 = time.perf_counter()
    rows, cols = augmented_assignment(C, L)
    return dict(rows=rows, cols=cols, F=wcs_objective_pairs(AG, AH, C, rows, cols, alpha),
                time=time.perf_counter() - t0)


def _rlx1_value(X, AG, AH, C, alpha):
    r = X.sum(1)
    XAX = X @ AH @ X.T
    h1 = np.sum(AG * AG * np.outer(r, r)) - 2 * np.sum(AG * XAX) + np.sum(XAX * XAX)
    return alpha * h1 + ((1 - alpha) * np.sum(C * X) if C is not None else 0.0)


def rlx1f_baseline(AG, AH, C, L, alpha, levels=10, fw_steps=6, grid=11):
    """Our re-implementation of GNCCP-RLX1F (Yang et al. 2018) as described in the
    paper: polynomial relaxation H1 with gradient Eq. (3), linear zeta schedule 1 -> -1,
    unconstrained assignment + top-L truncation, 11-point grid line search.
    NOTE: not the authors' reference code."""
    t0 = time.perf_counter()
    M, N = AG.shape[0], AH.shape[0]
    X = np.full((M, N), L / (M * N))
    A2 = AG * AG
    for k in range(levels):
        zeta = 1.0 - 2.0 * k / (levels - 1)
        w = (1 - zeta) if zeta >= 0 else (1 + zeta)
        for _ in range(fw_steps):
            P = X @ AH
            XAX = P @ X.T
            g = (2 * np.outer(A2 @ X.sum(1), np.ones(N)) - 4 * AG @ P
                 + 4 * XAX @ P)                                  # Eq. (3)
            gJ = w * (alpha * g + ((1 - alpha) * C if C is not None else 0.0)) + 2 * zeta * X
            rows, cols = topL_assignment(gJ, L)
            Y = pairs_to_X(rows, cols, X.shape)
            if abs(np.sum(gJ * (X - Y))) < 1e-3:
                break
            best_l, best_v = 0.0, None
            for lam in np.linspace(0, 1, grid):
                Xl = (1 - lam) * X + lam * Y
                v = w * _rlx1_value(Xl, AG, AH, C, alpha) + zeta * np.sum(Xl * Xl)
                if best_v is None or v < best_v:
                    best_l, best_v = lam, v
            X = (1 - best_l) * X + best_l * Y
    rows, cols = topL_assignment(-X, L)
    return dict(rows=rows, cols=cols, F=wcs_objective_pairs(AG, AH, C, rows, cols, alpha),
                time=time.perf_counter() - t0)


# ----------------------------------------------------------------------------
# Phase 2 (Fig. 1): Kabsch superposition and RMSD
# ----------------------------------------------------------------------------

def kabsch_rmsd(P, Q):
    """Optimal rigid superposition of P onto Q (rows are matched points);
    returns RMSD after optimal rotation + translation."""
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Dm = np.diag([1.0, 1.0, d])
    Rm = Vt.T @ Dm @ U.T
    diff = (Rm @ Pc.T).T - Qc
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def fgw_baseline(AG, AH, C, L, alpha):
    """Optional POT fused Gromov-Wasserstein baseline (pip install pot); coupling converted
    to an exact-L partial permutation by the augmented assignment (Sec. VII-D protocol)."""
    import ot
    t0 = time.perf_counter()
    M, N = AG.shape[0], AH.shape[0]
    Cm = np.zeros((M, N)) if C is None else C
    T = ot.gromov.fused_gromov_wasserstein(Cm, AG, AH, np.full(M, 1.0 / M), np.full(N, 1.0 / N),
                                           loss_fun="square_loss", alpha=alpha)
    rows, cols = augmented_assignment(-np.asarray(T), L)
    return dict(rows=rows, cols=cols, F=wcs_objective_pairs(AG, AH, C, rows, cols, alpha),
                time=time.perf_counter() - t0)
