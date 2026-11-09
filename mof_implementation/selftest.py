"""selftest.py -- engine audit (analogue of paper Sec. VII-E). Run before any benchmark.
Checks: (1) Theorem 1 gradient vs central finite differences, (2) Theorem 3 quartic
coefficients vs direct evaluation of J_zeta, (3) Cardano cubic solver vs np.roots / grid,
(4) augmented Hungarian oracle vs brute force, (5) 2-opt exact deltas vs recomputed F."""
import itertools
import numpy as np
from fastwcs import *
from fastwcs import _move_deltas

rng = np.random.default_rng(0)


def sym(n):
    A = rng.random((n, n)); A = (A + A.T) / 2; np.fill_diagonal(A, 0); return A


def J_dense(X, AG, AH, C, alpha, zeta):
    w = (1 - zeta) if zeta >= 0 else (1 + zeta)
    return w * wcs_objective_dense(X, AG, AH, C, alpha) + zeta * np.sum(X * X)


# (1) gradient
M, N = 7, 9
AG, AH, C = sym(M), sym(N), rng.random((M, N))
X = rng.random((M, N)) * 0.2
G = structural_gradient(X, AG, AH)[0]
fd = np.zeros_like(X); h = 1e-6
f = lambda Z: np.sum((np.outer(Z.sum(1), Z.sum(1)) * AG - Z @ AH @ Z.T) ** 2)
for i in range(M):
    for j in range(N):
        E = np.zeros_like(X); E[i, j] = h
        fd[i, j] = (f(X + E) - f(X - E)) / (2 * h)
print("gradient max rel err vs finite diff: %.2e" % (np.abs(G - fd).max() / np.abs(fd).max()))

# (2) quartic coefficients
alpha = 0.7
worst = 0
for zeta in (1.0, 0.5, 0.0, -0.4, -1.0):
    rows, cols = augmented_assignment(rng.standard_normal((M, N)), 5)
    Y = pairs_to_X(rows, cols, X.shape)
    _, r, P, XAX, R = structural_gradient(X, AG, AH)
    c = quartic_coefficients(X, Y, rows, cols, P, XAX, AG, AH, C, alpha, zeta)
    for lam in (0.0, 0.3, 0.77, 1.0):
        Xl = X - lam * (X - Y)
        worst = max(worst, abs(np.polyval(c[::-1], lam) - J_dense(Xl, AG, AH, C, alpha, zeta)))
print("quartic polynomial max abs err vs direct J: %.2e" % worst)

# (3) cubic solver
miss = 0; worst_root = 0.0; worse_than_grid = 0
for _ in range(5000):
    cc = rng.standard_normal(5) * 10 ** rng.uniform(-3, 3, 5)
    if rng.random() < 0.2:
        cc[4] = 0.0
    ref = [z.real for z in np.roots([4 * cc[4], 3 * cc[3], 2 * cc[2], cc[1]]) if abs(z.imag) < 1e-9] \
        if abs(cc[4]) > 0 else None
    mine = cubic_real_roots(4 * cc[4], 3 * cc[3], 2 * cc[2], cc[1])
    for z in (ref or []):
        if 0 < z < 1 and min([abs(z - m) for m in mine] + [9]) > 1e-6 * max(1, abs(z)):
            miss += 1
    for m in mine:
        den = np.polyval([4 * cc[4], 3 * cc[3], 2 * cc[2], cc[1]], m)
        worst_root = max(worst_root, abs(den) / (np.abs(cc).max() * max(1, abs(m)) ** 3))
    lam = minimise_quartic_on_unit_interval(cc)
    grid = np.linspace(0, 1, 2001)
    if np.polyval(cc[::-1], lam) > np.polyval(cc[::-1], grid).min() + 1e-9 * np.abs(cc).max():
        worse_than_grid += 1
print("cubic solver: missed roots in [0,1]=%d, worst scaled residual=%.1e, worse-than-grid=%d"
      % (miss, worst_root, worse_than_grid))

# (4) augmented oracle vs brute force
ok = True
for _ in range(30):
    M_, N_, L_ = 4, 5, 3
    cost = rng.standard_normal((M_, N_))
    r, c = augmented_assignment(cost, L_)
    best = min(sum(cost[i, j] for i, j in zip(rs, cs))
               for rs in itertools.combinations(range(M_), L_)
               for cs in itertools.permutations(range(N_), L_))
    ok &= abs(cost[r, c].sum() - best) < 1e-12
print("augmented Hungarian oracle equals brute force:", ok)

# (5) 2-opt deltas
M, N, L = 8, 10, 5
AG, AH, C = sym(M), sym(N), rng.random((M, N))
a = rng.permutation(M)[:L]; b = rng.permutation(N)[:L]
UG = np.setdiff1d(np.arange(M), a); UH = np.setdiff1d(np.arange(N), b)
F0 = wcs_objective_pairs(AG, AH, C, a, b, alpha)
d1, d2, d3 = _move_deltas(AG, AH, C, a, b, UG, UH, alpha)
err = 0.0
for k in range(L):
    for u in range(len(UH)):
        bb = b.copy(); bb[k] = UH[u]
        err = max(err, abs(wcs_objective_pairs(AG, AH, C, a, bb, alpha) - F0 - d1[k, u]))
    for m in range(k + 1, L):
        bb = b.copy(); bb[k], bb[m] = bb[m], bb[k]
        err = max(err, abs(wcs_objective_pairs(AG, AH, C, a, bb, alpha) - F0 - d2[k, m]))
    for v in range(len(UG)):
        aa = a.copy(); aa[k] = UG[v]
        err = max(err, abs(wcs_objective_pairs(AG, AH, C, aa, b, alpha) - F0 - d3[v, k]))
print("2-opt move deltas max abs err vs recomputed F: %.2e" % err)

# end-to-end sanity: planted isomorphic pair, no noise
n = 20
A = sym(n); perm = rng.permutation(n)
B = A[np.ix_(perm, perm)]
res = fastwcs(A, B, np.zeros((n, n)), n, alpha=1.0, starts=5)
print("planted isomorphism, F* = %.2e (0 means exact recovery)" % res["F"])
