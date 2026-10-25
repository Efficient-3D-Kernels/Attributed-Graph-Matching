"""
Unified Social Network Benchmark Suite: Stanford CollegeMsg Dataset
====================================================================
Comprehensive comparison of Fast-WCS v3 (AdvancedGNCCP) against:
  1. Original IEEE TNNLS 2018 Baseline (Yang et al. RLX1F)
  2. Weisfeiler-Lehman Optimal Assignment (WL-OA)
  3. Feature-Only Hungarian Baseline (Multi-scale topological features)
  4. SciPy FAQ (Fast Approximate Quadratic Assignment / Topology-only)
  5. Global Graph Kernels:
     - Weisfeiler-Lehman (WL) Subtree Kernel
     - Shortest-Path Kernel
     - Graphlet Kernel

Experiments:
  Protocol A: Controlled Perturbation Robustness (0%, 5%, 10%, 20%, 30% noise)
              with ground-truth node correspondence on N in {40, 100}.
  Protocol B: Dynamic Temporal Evolution (Early 50% vs Later 50% CollegeMsg interactions).

Metrics:
  - Matching Accuracy (%) [For assignment methods under Protocol A]
  - Normalized WCS Objective F*(X)
  - Graph Similarity Kernel K in [0, 1]
  - Wall-clock runtime (ms)
"""

import os
import sys
import time
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import linear_sum_assignment, quadratic_assignment

# Add root deliverable directory to sys.path to import production solvers
DELIVERABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DELIVERABLE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gnccp_advanced_v3 import AdvancedGNCCP
from gnccp_advanced_v4_wloa import AdvancedGNCCP_v4
from gnccp_baseline_2018 import OriginalGNCCP_2018
from src.data_loader import load_college_msg, split_temporal_data
from src.graph_builder import build_real_social_graphs
from src.enhanced_features import build_enhanced_cost_matrix

from src.kernels.wl_kernel import wl_kernel_similarity
from src.kernels.wl_oa_kernel import _wl_label_history, _node_similarity
from src.kernels.shortest_path_kernel import shortest_path_kernel_similarity
from src.kernels.graphlet_kernel import graphlet_kernel_similarity

RESULTS_DIR = Path(__file__).resolve().parent / "results"
TABLES_DIR = RESULTS_DIR / "tables"
MATCHINGS_DIR = RESULTS_DIR / "matchings"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
MATCHINGS_DIR.mkdir(parents=True, exist_ok=True)


def perturb_graph(G: nx.Graph, noise_pct: float, seed: int = 42) -> nx.Graph:
    """Create a controlled perturbed copy of G with edge modifications and deletions."""
    H = copy.deepcopy(G)
    rng = random.Random(seed)
    edges = list(H.edges())
    num_modify = int(len(edges) * (noise_pct / 100.0))
    if num_modify > 0:
        to_modify = rng.sample(edges, num_modify)
        for u, v in to_modify:
            if rng.random() < 0.5:
                # Perturb weight with random scale factor
                H[u][v]["weight"] = max(0.1, H[u][v].get("weight", 1.0) * rng.uniform(0.5, 1.5))
            else:
                H.remove_edge(u, v)
    return H


def solve_partial_hungarian(cost_matrix: np.ndarray, L: int):
    """Burkard exact partial assignment of size L for an M x N cost matrix."""
    M, N = cost_matrix.shape
    if L >= min(M, N):
        ri, ci = linear_sum_assignment(cost_matrix)
        return ri, ci
    aug = np.zeros((M + N - L, M + N - L), dtype=float)
    aug[:M, :N] = cost_matrix
    aug[M:, N:] = 1e9
    ri, ci = linear_sum_assignment(aug)
    keep = (ri < M) & (ci < N)
    return ri[keep], ci[keep]


def run_comprehensive_benchmark(user_sizes=(40, 100), noise_levels=(0, 5, 10, 20, 30), alpha=0.7, gamma=1.0):
    """Run full benchmark against all graph matching and graph kernel baselines."""
    print("\n" + "=" * 85)
    print("COMPREHENSIVE SOCIAL NETWORK BENCHMARK: FAST-WCS vs KERNELS & BASELINES")
    print("=" * 85)

    data = load_college_msg()
    early_data, later_data = split_temporal_data(data)

    records = []
    kernel_records = []

    for N in user_sizes:
        print(f"\n====================== Graph Size N = {N} users ======================")
        G, _, users = build_real_social_graphs(early_data, later_data, num_users=N)
        L = int(round(0.75 * len(users)))

        for noise in noise_levels:
            print(f"\n--- [N = {N}, Noise = {noise}%] ---")
            H = perturb_graph(G, noise, seed=42)

            # Feature cost matrix C
            C, nodes_G, nodes_H, _, _ = build_enhanced_cost_matrix(G, H)
            A_G = nx.to_numpy_array(G, nodelist=nodes_G, weight="weight", dtype=float)
            A_H = nx.to_numpy_array(H, nodelist=nodes_H, weight="weight", dtype=float)

            # -----------------------------------------------------------------
            # 1. Fast-WCS v3 (Ours, Full Defaults)
            # -----------------------------------------------------------------
            solver_fast = AdvancedGNCCP(alpha=alpha, gamma=gamma, local_refinement=True)
            t0 = time.perf_counter()
            res_fast = solver_fast.solve(A_G, A_H, C=C, L=L)
            t_fast_ms = (time.perf_counter() - t0) * 1000
            correct_fast = sum(1 for r, c in res_fast.matches if nodes_G[r] == nodes_H[c])
            acc_fast = (correct_fast / L) * 100.0

            records.append({
                "N": N, "L": L, "Noise_pct": noise, "Method": "Fast-WCS v3 (Ours)",
                "Accuracy_pct": acc_fast, "Objective": res_fast.objective_score,
                "Similarity_Kernel": res_fast.similarity_score, "Runtime_ms": t_fast_ms, "Type": "Graph Matching"
            })
            print(f"  * Fast-WCS v3 (Ours):     Acc = {acc_fast:5.1f}% | Time = {t_fast_ms:6.1f} ms | Obj = {res_fast.objective_score:7.4f}")

            # -----------------------------------------------------------------
            # 1b. Fast-WCS v4 (WL-OA Augmented)
            # -----------------------------------------------------------------
            solver_v4 = AdvancedGNCCP_v4(alpha=alpha, gamma=gamma, local_refinement=True)
            t0 = time.perf_counter()
            res_v4 = solver_v4.solve(A_G, A_H, C=C, L=L)
            t_v4_ms = (time.perf_counter() - t0) * 1000
            correct_v4 = sum(1 for r, c in res_v4.matches if nodes_G[r] == nodes_H[c])
            acc_v4 = (correct_v4 / L) * 100.0

            records.append({
                "N": N, "L": L, "Noise_pct": noise, "Method": "Fast-WCS v4 (WL-OA Augmented)",
                "Accuracy_pct": acc_v4, "Objective": res_v4.objective_score,
                "Similarity_Kernel": res_v4.similarity_score, "Runtime_ms": t_v4_ms, "Type": "Graph Matching"
            })
            print(f"  * Fast-WCS v4 (WL-OA):    Acc = {acc_v4:5.1f}% | Time = {t_v4_ms:6.1f} ms | Obj = {res_v4.objective_score:7.4f}")

            # -----------------------------------------------------------------
            # 2. Fast-WCS v3 (Ablation: No 2-Opt)
            # -----------------------------------------------------------------
            solver_ablat = AdvancedGNCCP(alpha=alpha, gamma=gamma, local_refinement=False)
            t0 = time.perf_counter()
            res_ablat = solver_ablat.solve(A_G, A_H, C=C, L=L)
            t_ablat_ms = (time.perf_counter() - t0) * 1000
            correct_ablat = sum(1 for r, c in res_ablat.matches if nodes_G[r] == nodes_H[c])
            acc_ablat = (correct_ablat / L) * 100.0

            records.append({
                "N": N, "L": L, "Noise_pct": noise, "Method": "Fast-WCS v3 (No 2-Opt)",
                "Accuracy_pct": acc_ablat, "Objective": res_ablat.objective_score,
                "Similarity_Kernel": res_ablat.similarity_score, "Runtime_ms": t_ablat_ms, "Type": "Graph Matching"
            })
            print(f"  * Fast-WCS v3 (No 2-Opt): Acc = {acc_ablat:5.1f}% | Time = {t_ablat_ms:6.1f} ms | Obj = {res_ablat.objective_score:7.4f}")

            # -----------------------------------------------------------------
            # 3. Yang et al. 2018 (RLX1F Baseline)
            # -----------------------------------------------------------------
            solver_base = OriginalGNCCP_2018(alpha=alpha, relaxation="RLX1", oracle="fast_heuristic")
            t0 = time.perf_counter()
            res_base = solver_base.solve(A_G, A_H, C=C, L=L)
            t_base_ms = (time.perf_counter() - t0) * 1000
            correct_base = sum(1 for r, c in res_base.matches if nodes_G[r] == nodes_H[c])
            acc_base = (correct_base / L) * 100.0

            records.append({
                "N": N, "L": L, "Noise_pct": noise, "Method": "Yang 2018 (RLX1F)",
                "Accuracy_pct": acc_base, "Objective": res_base.objective_score,
                "Similarity_Kernel": float(np.exp(-gamma * (res_base.objective_score / max(L, 1)))),
                "Runtime_ms": t_base_ms, "Type": "Graph Matching"
            })
            print(f"  * Yang 2018 (RLX1F):      Acc = {acc_base:5.1f}% | Time = {t_base_ms:6.1f} ms | Obj = {res_base.objective_score:7.4f}")

            # -----------------------------------------------------------------
            # 4. Weisfeiler-Lehman Optimal Assignment (WL-OA)
            # -----------------------------------------------------------------
            t0 = time.perf_counter()
            hist_G = _wl_label_history(G, h=3)
            hist_H = _wl_label_history(H, h=3)
            sim_mat = np.array([[_node_similarity(hist_G, hist_H, u, v) for v in nodes_H] for u in nodes_G])
            cost_wloa = 1.0 - sim_mat
            ri_wloa, ci_wloa = solve_partial_hungarian(cost_wloa, L)
            t_wloa_ms = (time.perf_counter() - t0) * 1000
            correct_wloa = sum(1 for r, c in zip(ri_wloa, ci_wloa) if nodes_G[r] == nodes_H[c])
            acc_wloa = (correct_wloa / L) * 100.0
            kernel_wloa = float(np.mean([sim_mat[r, c] for r, c in zip(ri_wloa, ci_wloa)]))

            records.append({
                "N": N, "L": L, "Noise_pct": noise, "Method": "WL-OA (Hungarian)",
                "Accuracy_pct": acc_wloa, "Objective": np.nan,
                "Similarity_Kernel": kernel_wloa, "Runtime_ms": t_wloa_ms, "Type": "Optimal Assignment Kernel"
            })
            print(f"  * WL-OA (Hungarian):      Acc = {acc_wloa:5.1f}% | Time = {t_wloa_ms:6.1f} ms | Kernel = {kernel_wloa:6.4f}")

            # -----------------------------------------------------------------
            # 5. Feature-Only Hungarian Baseline (No Graph Topology)
            # -----------------------------------------------------------------
            t0 = time.perf_counter()
            ri_feat, ci_feat = solve_partial_hungarian(C, L)
            t_feat_ms = (time.perf_counter() - t0) * 1000
            correct_feat = sum(1 for r, c in zip(ri_feat, ci_feat) if nodes_G[r] == nodes_H[c])
            acc_feat = (correct_feat / L) * 100.0

            records.append({
                "N": N, "L": L, "Noise_pct": noise, "Method": "Feature-Only Hungarian",
                "Accuracy_pct": acc_feat, "Objective": np.nan,
                "Similarity_Kernel": np.nan, "Runtime_ms": t_feat_ms, "Type": "Ablation Baseline"
            })
            print(f"  * Feature-Only Hungarian: Acc = {acc_feat:5.1f}% | Time = {t_feat_ms:6.1f} ms")

            # -----------------------------------------------------------------
            # 6. SciPy FAQ Baseline (QAP Topology-Only)
            # -----------------------------------------------------------------
            t0 = time.perf_counter()
            res_faq = quadratic_assignment(A_G, A_H, method="faq", options={"maximize": True})
            t_faq_ms = (time.perf_counter() - t0) * 1000
            correct_faq = sum(1 for r, c in enumerate(res_faq.col_ind[:L]) if nodes_G[r] == nodes_H[c])
            acc_faq = (correct_faq / L) * 100.0

            records.append({
                "N": N, "L": L, "Noise_pct": noise, "Method": "SciPy FAQ (Topology Only)",
                "Accuracy_pct": acc_faq, "Objective": np.nan,
                "Similarity_Kernel": np.nan, "Runtime_ms": t_faq_ms, "Type": "Graph Matching"
            })
            print(f"  * SciPy FAQ (QAP Topology):Acc = {acc_faq:5.1f}% | Time = {t_faq_ms:6.1f} ms")

            # -----------------------------------------------------------------
            # 7. Global Graph Kernels (WL, Shortest Path, Graphlet)
            # -----------------------------------------------------------------
            t0 = time.perf_counter()
            k_wl = wl_kernel_similarity(G, H, h=3)
            t_wl_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            k_sp = shortest_path_kernel_similarity(G, H)
            t_sp_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            k_gr = graphlet_kernel_similarity(G, H)
            t_gr_ms = (time.perf_counter() - t0) * 1000

            kernel_records.append({
                "N": N, "Noise_pct": noise,
                "WL_Subtree_Kernel": k_wl, "WL_Time_ms": t_wl_ms,
                "Shortest_Path_Kernel": k_sp, "SP_Time_ms": t_sp_ms,
                "Graphlet_Kernel": k_gr, "Graphlet_Time_ms": t_gr_ms,
            })
            print(f"  [Kernels] WL = {k_wl:.4f} ({t_wl_ms:.1f}ms) | SP = {k_sp:.4f} ({t_sp_ms:.1f}ms) | Graphlet = {k_gr:.4f} ({t_gr_ms:.1f}ms)")

    # Save to CSV
    df_methods = pd.DataFrame(records)
    df_kernels = pd.DataFrame(kernel_records)
    
    methods_path = TABLES_DIR / "social_network_comprehensive_methods.csv"
    kernels_path = TABLES_DIR / "social_network_global_kernels.csv"
    
    df_methods.to_csv(methods_path, index=False)
    df_kernels.to_csv(kernels_path, index=False)
    
    print(f"\n[Done] Methods comparison saved to: {methods_path}")
    print(f"[Done] Global kernels comparison saved to: {kernels_path}")
    return df_methods, df_kernels


if __name__ == "__main__":
    df_m, df_k = run_comprehensive_benchmark(user_sizes=(40, 100), noise_levels=(0, 5, 10, 20, 30))
    print("\nBenchmark completed successfully.")
