"""Numerical audit of gnccp_advanced_v3.py (Step 1 of the live benchmark plan).

Checks, all computed live:
  1. Bitwise parity of the audited engine against the pre-audit copy kept in
     benchmarks/scripts/_reference/gnccp_advanced_v3_pre_audit.py (assignments, objective, FW steps).
  2. Cubic-root solver on adversarial cubics (random, double roots, c4 ~ 0, quadratic, linear):
     every real root in [0, 1] must be recovered, and the quartic argmin must match a 200k-point grid.
  3. Kabsch on degenerate inputs (collinear, coplanar, mirrored clouds): det(R) = +1, optimal RMSD.
  4. Zero-edge / single-node graphs under every normalisation: finite output, exact cardinality.
  5. Wall-clock before/after (single BLAS thread, interleaved repetitions).

Output: benchmarks/live_runs/json/engine_audit.json and benchmarks/live_runs/csv/engine_audit_timing.csv
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from mpmath import mp
from threadpoolctl import threadpool_limits

mp.dps = 50

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gnccp_advanced_v3 as new  # noqa: E402
from live_common import make_synthetic_instance, provenance, write_csv, LIVE  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "pre_audit", Path(__file__).resolve().parent / "_reference" / "gnccp_advanced_v3_pre_audit.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def parity_suite(n_cases=400, seed=7):
    rng = np.random.default_rng(seed)
    mismatches, cases = [], 0
    for k in range(n_cases):
        M = int(rng.integers(6, 61)); N = M + int(rng.integers(0, 11))
        L = int(rng.integers(max(2, M // 2), M + 1))
        alpha = float(rng.choice([1.0, 0.7, 0.5]))
        norm = str(rng.choice(['none', 'shared_max', 'frobenius']))
        sched = str(rng.choice(['cosine', 'linear']))
        AG, AH, _ = make_synthetic_instance(M, N, L, sigma=float(rng.uniform(0, .1)),
                                            density=float(rng.uniform(.2, 1.)), seed=int(rng.integers(1 << 30)))
        C = rng.random((M, N)).astype(np.float32) if alpha < 1 else None
        kw = dict(alpha=alpha, normalization=norm, schedule=sched, max_iter=int(rng.integers(4, 13)),
                  fw_max_iter=int(rng.integers(2, 7)), seed=k, n_restarts=int(rng.choice([1, 1, 2])))
        r_old = old.AdvancedGNCCP(**kw).solve(AG, AH, C=C, L=L)
        r_new = new.AdvancedGNCCP(**kw).solve(AG, AH, C=C, L=L)
        cases += 1
        same = (np.array_equal(r_old.X_binary, r_new.X_binary)
                and r_old.objective_score == r_new.objective_score
                and r_old.fw_steps_total == r_new.fw_steps_total
                and np.array_equal(r_old.X_continuous, r_new.X_continuous))
        if not same:
            mismatches.append({'case': k, 'M': M, 'N': N, 'L': L, 'kw': kw,
                               'obj_old': r_old.objective_score, 'obj_new': r_new.objective_score,
                               'same_binary': bool(np.array_equal(r_old.X_binary, r_new.X_binary))})
    return {'cases': cases, 'bitwise_identical': cases - len(mismatches), 'mismatches': mismatches}


_ENGINE = new.AdvancedGNCCP()


def cubic_suite(n=20000, seed=11):
    rng = np.random.default_rng(seed)
    solver = new.AdvancedGNCCP._real_roots_cubic
    worst_missing, fails, argmin_fail = 0.0, 0, 0
    kinds = {'random': 0, 'double_root': 0, 'tiny_leading': 0, 'quadratic': 0, 'linear': 0}
    grid = np.linspace(0, 1, 200001)
    for i in range(n):
        kind = list(kinds)[i % len(kinds)]
        kinds[kind] += 1
        if kind == 'random':
            a, b, c, d = rng.normal(size=4)
        elif kind == 'double_root':
            r1, r2 = rng.uniform(-.5, 1.5, 2)
            a = rng.choice([-1, 1]) * rng.uniform(.1, 10)
            a, b, c, d = a, -a * (2 * r1 + r2), a * (r1 * r1 + 2 * r1 * r2), -a * r1 * r1 * r2
        elif kind == 'tiny_leading':
            a = rng.normal() * 10 ** rng.uniform(-14, -6); b, c, d = rng.normal(size=3)
        elif kind == 'quadratic':
            a, (b, c, d) = 0.0, rng.normal(size=3)
        else:
            a, b, (c, d) = 0.0, 0.0, rng.normal(size=2)
        got = solver(a, b, c, d)
        # 50-digit reference roots of the float64 coefficients (np.roots itself is not reliable
        # near double roots and is therefore not used as ground truth)
        co = [mp.mpf(float(v)) for v in np.trim_zeros([a, b, c, d], 'f')]
        ref = [] if len(co) < 2 else [float(mp.re(z)) for z in mp.polyroots(co, maxsteps=200, extraprec=200)
                                      if abs(mp.im(z)) < 1e-25 and -1e-9 <= mp.re(z) <= 1 + 1e-9]
        for z in ref:
            miss = min([abs(z - g) for g in got], default=np.inf)
            # double roots are only determined to ~sqrt(eps)
            worst_missing = max(worst_missing, miss if kind != 'double_root' else miss / 1e4)
            if miss > (1e-6 if kind != 'double_root' else 1e-4):
                fails += 1
        # quartic argmin on [0,1]: J' = 4c4 l^3 + 3c3 l^2 + 2c2 l + c1 = cubic(a,b,c,d)
        c4, c3, c2, c1 = a / 4, b / 3, c / 2, d
        lam = _ENGINE._argmin_quartic_unit(0.0, c1, c2, c3, c4)
        J = lambda t: t * (c1 + t * (c2 + t * (c3 + t * c4)))
        jg = J(grid).min()
        if J(lam) > jg + 1e-9 * max(1.0, abs(jg)):
            argmin_fail += 1
    return {'n': n, 'by_kind': kinds, 'missed_roots': fails, 'worst_root_error': float(worst_missing),
            'quartic_argmin_worse_than_200k_grid': argmin_fail}


def kabsch_suite(seed=3):
    rng = np.random.default_rng(seed)
    out = {}
    def rot():
        Qm, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        return Qm * np.sign(np.linalg.det(Qm))
    # collinear
    P = np.outer(rng.normal(size=20), rng.normal(size=3)); R = rot(); Q = P @ R.T + 1.0
    a = new.robust_kabsch_superposition(P, Q)
    out['collinear'] = {'rmsd': a['rmsd'], 'det_R': a['det_R'], 'rank_H': a['rank_H']}
    # coplanar
    P = np.c_[rng.normal(size=(20, 2)), np.zeros(20)]; R = rot(); Q = P @ R.T
    a = new.robust_kabsch_superposition(P, Q)
    out['coplanar'] = {'rmsd': a['rmsd'], 'det_R': a['det_R'], 'rank_H': a['rank_H']}
    # mirrored: best proper rotation of a reflected chiral cloud; must keep det=+1 and fire the guard
    P = rng.normal(size=(30, 3)); Q = P * np.array([1, 1, -1])
    a = new.robust_kabsch_superposition(P, Q)
    out['mirror'] = {'rmsd': a['rmsd'], 'det_R': a['det_R'], 'reflection_corrected': a['reflection_corrected']}
    # exact recovery with noise-free rotation
    P = rng.normal(size=(30, 3)); R = rot(); Q = P @ R.T + rng.normal(size=3)
    a = new.robust_kabsch_superposition(P, Q)
    out['exact'] = {'rmsd': a['rmsd'], 'det_R': a['det_R'], 'rot_err': float(np.abs(a['R'] - R).max())}
    out['passed'] = (out['collinear']['rmsd'] < 1e-8 and out['coplanar']['rmsd'] < 1e-8
                     and abs(out['mirror']['det_R'] - 1) < 1e-9 and out['mirror']['reflection_corrected']
                     and out['exact']['rot_err'] < 1e-9 and abs(out['collinear']['det_R'] - 1) < 1e-9)
    return out


def degenerate_graph_suite():
    out = []
    for norm in ('none', 'shared_max', 'frobenius'):
        for (M, N, L) in ((1, 1, 1), (5, 7, 3), (12, 12, 12)):
            for empty in ('both', 'G'):
                AG = np.zeros((M, M), np.float32)
                AH = np.zeros((N, N), np.float32)
                if empty == 'G' and N > 1:
                    AH[0, 1] = AH[1, 0] = 1.0
                r = new.AdvancedGNCCP(alpha=1.0, normalization=norm).solve(AG, AH, L=L)
                out.append({'norm': norm, 'M': M, 'N': N, 'L': L, 'empty': empty,
                            'finite': bool(np.isfinite(r.objective_score) and np.isfinite(r.X_continuous).all()),
                            'cardinality_ok': int(r.X_binary.sum()) == L})
    return {'cases': out, 'passed': all(c['finite'] and c['cardinality_ok'] for c in out)}


def timing_suite(sizes=(20, 30, 60, 100, 150), reps=15):
    rows = []
    for n in sizes:
        L = int(round(.8 * n))
        insts = [make_synthetic_instance(n, n, L, .03, .5, seed=900 + k)[:2] for k in range(reps)]
        s_old, s_new = old.AdvancedGNCCP(alpha=1.0), new.AdvancedGNCCP(alpha=1.0)
        s_old.solve(*insts[0], L=L); s_new.solve(*insts[0], L=L)
        t_old, t_new = [], []
        for AG, AH in insts:  # interleaved to cancel drift
            t = time.perf_counter(); s_old.solve(AG, AH, L=L); t_old.append(time.perf_counter() - t)
            t = time.perf_counter(); s_new.solve(AG, AH, L=L); t_new.append(time.perf_counter() - t)
        rows.append({'N': n, 'L': L, 'reps': reps,
                     'pre_audit_median_ms': 1e3 * float(np.median(t_old)),
                     'audited_median_ms': 1e3 * float(np.median(t_new)),
                     'speedup': float(np.median(t_old) / np.median(t_new))})
        print(rows[-1], flush=True)
    return rows


def main():
    with threadpool_limits(1):
        report = {'provenance': provenance([Path(__file__), ROOT / 'gnccp_advanced_v3.py',
                                            Path(__file__).resolve().parent / '_reference' / 'gnccp_advanced_v3_pre_audit.py'])}
        report['gradient_check'] = new.AdvancedGNCCP(alpha=0.7).verify_gradient()
        print('gradient', report['gradient_check'], flush=True)
        report['parity'] = parity_suite(); print('parity', {k: v for k, v in report['parity'].items() if k != 'mismatches'}, flush=True)
        report['cubic'] = cubic_suite(); print('cubic', report['cubic'], flush=True)
        report['kabsch'] = kabsch_suite(); print('kabsch', report['kabsch'], flush=True)
        report['degenerate_graphs'] = degenerate_graph_suite(); print('degenerate', report['degenerate_graphs']['passed'], flush=True)
        report['timing'] = timing_suite()
    (LIVE / 'json').mkdir(parents=True, exist_ok=True)
    (LIVE / 'json' / 'engine_audit.json').write_text(json.dumps(report, indent=2, default=float), encoding='utf-8')
    write_csv(LIVE / 'csv' / 'engine_audit_timing.csv', report['timing'])
    print('wrote', LIVE / 'json' / 'engine_audit.json')


if __name__ == '__main__':
    main()
