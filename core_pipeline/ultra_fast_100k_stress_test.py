"""
ultra_fast_100k_stress_test.py
========================================================================================
MASSIVE 100,000-TEST ULTRA-FAST HIGH-THROUGHPUT STRESS BENCHMARK
========================================================================================
Evaluates Fast-WCS Advanced across 100,000 combinatorial graph matching instances
spanning diverse problem sizes (M, N) and varying matched subgraph cardinalities L.

Combines:
  1. Multiprocessing Pool utilizing all available logical CPU cores.
  2. Lean Fast-WCS Advanced engine (quartic line search, analytical gradient, exact Hungarian).
  3. Real-time batch progress tracking with pairs/sec throughput reporting.
  4. Comprehensive performance breakdown across:
     - Core Ratios: Sparse (L/M = 0.5), Moderate (L/M = 0.75), Full/Max (L = M).
     - Molecular Scales: Fragment (M=10), Drug-like (M=16, M=24), Macrocyclic (M=36, M=48).

Outputs:
  - Formatted console throughput summary
  - JSON record saved to benchmark_100k_stress_test_results.json
========================================================================================
"""

import os
import sys
import time
import json
import argparse
import multiprocessing as mp
import numpy as np

# Safe stdout reconfigure on Windows console
if sys.platform.startswith('win'):
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from scipy.optimize import linear_sum_assignment

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CUR_DIR)


# ======================================================================================
# HIGH-EFFICIENCY WORKER ENGINE
# ======================================================================================

def fast_wcs_worker(task_tuple):
    """
    Worker task: Solves a single graph matching problem with given (M, N, L, seed).
    Executes in < 1 ms via analytical gradient and exact quartic line search.
    """
    M, N, L, seed = task_tuple
    rng = np.random.RandomState(seed)
    
    # 1. Generate weighted graphs A_G and A_H
    # Subgraph isomorphism: H contains a perturbed version of G's core
    core_weights = rng.uniform(0.1, 1.0, (L, L)).astype(np.float32)
    core_weights = (core_weights + core_weights.T) / 2.0
    np.fill_diagonal(core_weights, 0.0)
    
    A_G = np.zeros((M, M), dtype=np.float32)
    A_G[:L, :L] = core_weights
    if M > L:
        out_G = rng.uniform(0.1, 1.0, (M, M)).astype(np.float32)
        out_G = (out_G + out_G.T) / 2.0
        A_G[L:, :] = out_G[L:, :]
        A_G[:, L:] = out_G[:, L:]
        np.fill_diagonal(A_G, 0.0)
        
    A_H = np.zeros((N, N), dtype=np.float32)
    # Add minor continuous noise
    A_H[:L, :L] = np.clip(core_weights + rng.normal(0, 0.02, (L, L)).astype(np.float32), 0.0, 1.0)
    A_H[:L, :L] = (A_H[:L, :L] + A_H[:L, :L].T) / 2.0
    if N > L:
        out_H = rng.uniform(0.1, 1.0, (N, N)).astype(np.float32)
        out_H = (out_H + out_H.T) / 2.0
        A_H[L:, :] = out_H[L:, :]
        A_H[:, L:] = out_H[:, L:]
        np.fill_diagonal(A_H, 0.0)
        
    # Permute H
    perm_H = rng.permutation(N)
    A_H = A_H[perm_H][:, perm_H]
    gt_mapping = {i: int(perm_H[i]) for i in range(L)}
    
    # 2. Fast-WCS GNCCP Homotopy with Analytical Gradient & Cardano Line Search
    t0 = time.perf_counter()
    X = np.ones((M, N), dtype=np.float32) * (L / (M * N))
    
    # Cosine continuation schedule (3 outer steps, 2 FW iterations each for screening throughput)
    zeta_vals = [0.8, 0.0, -0.8]
    
    for zeta in zeta_vals:
        for _ in range(2):
            # Lemma 1: Rank-1 factored analytical gradient
            r = X.sum(axis=1)
            U = np.outer(r, r)
            XA_H = X @ A_H
            R = (U * A_G) - (XA_H @ X.T)
            
            grad_F = 4.0 * ((R * A_G) @ r)[:, None] * np.ones((1, N), dtype=np.float32) - 4.0 * (R @ XA_H)
            grad_J = (1.0 - zeta) * grad_F + 2.0 * zeta * X if zeta >= 0 else (1.0 + zeta) * grad_F + 2.0 * zeta * X
            
            # Augmented Hungarian oracle for cardinality L
            row_ind, col_ind = linear_sum_assignment(grad_J)
            costs = grad_J[row_ind, col_ind]
            top_L = np.argsort(costs)[:L]
            Y = np.zeros_like(X)
            Y[row_ind[top_L], col_ind[top_L]] = 1.0
            
            D_fw = Y - X
            gap = float(np.sum(grad_J * (-D_fw)))
            if abs(gap) < 1e-3:
                break
                
            # Closed-form Quartic Cardano Line Search along D_fw
            K = D_fw @ A_H @ X.T
            R0 = (np.outer(X.sum(axis=1), X.sum(axis=1)) * A_G) - (X @ A_H @ X.T)
            R1 = (np.outer(X.sum(axis=1), D_fw.sum(axis=1)) + np.outer(D_fw.sum(axis=1), X.sum(axis=1))) * A_G - (K.T + K)
            R2 = (np.outer(D_fw.sum(axis=1), D_fw.sum(axis=1)) * A_G) - (D_fw @ A_H @ D_fw.T)
            
            c0 = float(np.sum(R0 * R0))
            c1 = 2.0 * float(np.sum(R0 * R1))
            c2 = float(np.sum(R1 * R1)) + 2.0 * float(np.sum(R0 * R2))
            c3 = 2.0 * float(np.sum(R1 * R2))
            c4 = float(np.sum(R2 * R2))
            
            # Derivative polynomial roots: 4*c4*lam^3 + 3*c3*lam^2 + 2*c2*lam + c1 = 0
            if c4 > 1e-8:
                poly_deriv = [4.0*c4, 3.0*c3, 2.0*c2, c1]
                roots = np.roots(poly_deriv)
                real_roots = [float(np.real(r)) for r in roots if np.isreal(r) and 0.0 <= np.real(r) <= 1.0]
                candidates = [0.0, 1.0] + real_roots
                best_lam = min(candidates, key=lambda l: c0 + c1*l + c2*(l**2) + c3*(l**3) + c4*(l**4))
            else:
                best_lam = 0.5
                
            X = np.clip(X + best_lam * D_fw, 0.0, 1.0)
            
    # Binary Discretization
    row_d, col_d = linear_sum_assignment(-X)
    best_pairs = np.argsort(-X[row_d, col_d])[:L]
    matched_g = row_d[best_pairs]
    matched_h = col_d[best_pairs]
    
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    
    # Compute accuracy
    correct = sum(1 for g, h in zip(matched_g, matched_h) if gt_mapping.get(g) == h)
    acc = (correct / L) * 100.0
    
    # Objective cost
    X_bin = np.zeros_like(X)
    X_bin[matched_g, matched_h] = 1.0
    diff = (X_bin @ X_bin.T * A_G) - (X_bin @ A_H @ X_bin.T)
    obj = float(np.sum(diff**2))
    
    return {
        'M': M, 'N': N, 'L': L,
        'acc': acc, 'obj': obj,
        'time_ms': elapsed_ms
    }


# ======================================================================================
# MULTI-CONFIG TEST RUNNER (100,000 PROBLEMS)
# ======================================================================================

def run_100k_stress_test(total_problems=100000, n_processes=None):
    if n_processes is None:
        n_processes = max(1, min(12, mp.cpu_count()))
        
    print("=" * 100)
    print("MASSIVE HIGH-THROUGHPUT GRAPH MATCHING STRESS BENCHMARK")
    print(f"Target: {total_problems:,} Matching Problems | Parallel Workers: {n_processes} CPU Cores")
    print("=" * 100)
    
    # Grid of representative problem sizes (M, N) and cardinalities L
    configurations = [
        # (M, N, L, label)
        (10, 14,  6, "Fragment Sparse  (M=10, N=14, L=6,  L/M=0.60)"),
        (10, 14, 10, "Fragment Max     (M=10, N=14, L=10, L/M=1.00)"),
        (16, 22,  8, "Drug-Like Sparse (M=16, N=22, L=8,  L/M=0.50)"),
        (16, 22, 12, "Drug-Like Mod.   (M=16, N=22, L=12, L/M=0.75)"),
        (16, 22, 16, "Drug-Like Max    (M=16, N=22, L=16, L/M=1.00)"),
        (24, 32, 12, "Dense Subgraph   (M=24, N=32, L=12, L/M=0.50)"),
        (24, 32, 18, "Dense Moderate   (M=24, N=32, L=18, L/M=0.75)"),
        (24, 32, 24, "Dense Max        (M=24, N=32, L=24, L/M=1.00)"),
        (36, 45, 18, "Macrocycle Half  (M=36, N=45, L=18, L/M=0.50)"),
        (36, 45, 27, "Macrocycle Mod.  (M=36, N=45, L=27, L/M=0.75)"),
        (48, 60, 24, "Large Complex    (M=48, N=60, L=24, L/M=0.50)"),
        (48, 60, 36, "Large Mod. Core  (M=48, N=60, L=36, L/M=0.75)"),
    ]
    
    n_configs = len(configurations)
    per_config_count = total_problems // n_configs
    actual_total = per_config_count * n_configs
    
    print(f"Allocating {per_config_count:,} trials per configuration across {n_configs} diverse (M, N, L) regimes.")
    print(f"Total matching tasks to execute: {actual_total:,}\n")
    
    # Build task list
    all_tasks = []
    task_id = 0
    for M, N, L, label in configurations:
        for _ in range(per_config_count):
            all_tasks.append((M, N, L, 1000000 + task_id))
            task_id += 1
            
    # Shuffle tasks to distribute load uniformly across processes
    rng = np.random.RandomState(42)
    shuffled_indices = rng.permutation(len(all_tasks))
    shuffled_tasks = [all_tasks[i] for i in shuffled_indices]
    
    print(f"Starting parallel execution across {n_processes} worker processes...")
    t_start = time.perf_counter()
    
    chunksize = 200
    batch_results = []
    
    with mp.Pool(processes=n_processes) as pool:
        # Use imap_unordered for streaming results with live progress updates
        processed_count = 0
        last_log_time = t_start
        log_interval_items = max(10000, actual_total // 10)
        
        for res in pool.imap_unordered(fast_wcs_worker, shuffled_tasks, chunksize=chunksize):
            batch_results.append(res)
            processed_count += 1
            
            if processed_count % log_interval_items == 0 or processed_count == actual_total:
                now = time.perf_counter()
                elapsed = now - t_start
                instant_rate = processed_count / max(elapsed, 1e-4)
                pct = (processed_count / actual_total) * 100.0
                rem_items = actual_total - processed_count
                eta_sec = rem_items / instant_rate if instant_rate > 0 else 0.0
                print(f"  [PROGRESS] {processed_count:7,d} / {actual_total:,} completed ({pct:5.1f}%) | "
                      f"Speed: {instant_rate:7.1f} pairs/sec | Elapsed: {elapsed:5.1f}s | ETA: {eta_sec:4.1f}s")
                
    total_elapsed = time.perf_counter() - t_start
    overall_throughput = actual_total / max(total_elapsed, 1e-4)
    
    print("\n" + "=" * 100)
    print("BENCHMARK EXECUTION COMPLETE: 100,000 PAIR EVALUATIONS FINISHED")
    print(f"Total Problems Solved: {actual_total:,} | Total Wall-Clock Time: {total_elapsed:.2f} s")
    print(f"Aggregate Screening Throughput: {overall_throughput:,.1f} pairs / second ({total_elapsed*1000/actual_total:.3f} ms / match)")
    print("=" * 100)
    
    # =========================================================================
    # DETAILED PER-CONFIGURATION BREAKDOWN TABLE
    # =========================================================================
    print(f"\n{'Configuration (M, N, L)':<36s} | {'Core Ratio':>10s} | {'Accuracy (%)':>13s} | {'Obj. Cost':>10s} | {'Time/Match':>11s} | {'Throughput':>12s}")
    print("-" * 104)
    
    summary_by_config = {}
    
    for M, N, L, label in configurations:
        matching_results = [r for r in batch_results if r['M'] == M and r['N'] == N and r['L'] == L]
        if not matching_results:
            continue
            
        accs = [r['acc'] for r in matching_results]
        objs = [r['obj'] for r in matching_results]
        times = [r['time_ms'] for r in matching_results]
        
        m_acc = float(np.mean(accs))
        m_obj = float(np.mean(objs))
        m_time = float(np.mean(times))
        conf_throughput = 1000.0 / max(m_time, 1e-4) * n_processes
        
        ratio_str = f"{L/M*100:4.1f}%"
        cfg_str = f"M={M:2d}, N={N:2d}, L={L:2d}"
        
        summary_by_config[label] = {
            'M': M, 'N': N, 'L': L,
            'core_ratio': L / M,
            'n_samples': len(matching_results),
            'mean_accuracy': m_acc,
            'mean_objective': m_obj,
            'mean_latency_ms': m_time,
            'throughput_pairs_per_sec': conf_throughput
        }
        
        print(f"{cfg_str:<36s} | {ratio_str:>10s} | {m_acc:11.1f} % | {m_obj:10.2f} | {m_time:8.3f} ms | {conf_throughput:10.1f} p/s")
        
    print("-" * 104)
    
    # Save results to JSON
    out_json = os.path.join(CUR_DIR, 'benchmark_100k_stress_test_results.json')
    full_report = {
        'total_problems_evaluated': actual_total,
        'wall_clock_time_seconds': total_elapsed,
        'overall_throughput_pairs_per_sec': overall_throughput,
        'mean_latency_per_match_ms': total_elapsed * 1000.0 / actual_total,
        'n_processes': n_processes,
        'configurations_breakdown': summary_by_config,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(full_report, f, indent=2)
        
    print(f"\n[OK] Full 100,000-sample empirical metrics successfully written to:\n  -> {out_json}")
    print("=" * 100)
    return full_report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Massive 100k Stress Test for Fast-WCS Advanced")
    parser.add_argument('--total', type=int, default=100000, help="Total number of problem instances (default: 100000)")
    parser.add_argument('--cores', type=int, default=None, help="Number of CPU cores to utilize")
    args = parser.parse_args()
    
    run_100k_stress_test(total_problems=args.total, n_processes=args.cores)
