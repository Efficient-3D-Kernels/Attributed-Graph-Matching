"""Reproducible exploratory point-cloud benchmark for Fast-WCS v3.

This is a controlled synthetic stress test, not a reproduction of a published
3D registration benchmark. It compares the in-repository GNCCP versions and a
simple exact-cardinality ICP/Hungarian baseline on identical generated pairs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gnccp_advanced_v3 import (
    AdvancedGNCCP,
    kabsch,
    robust_kabsch_superposition,
)
from gnccp_baseline_2018 import OriginalGNCCP_2018
from gnccp_v2_classic import UniversalGNCCP


def make_pair(seed: int, m: int, n: int, l: int, noise: float, outlier_scale: float):
    rng = np.random.default_rng(seed)
    source = rng.normal(size=(m, 3)).astype(np.float32)

    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(-np.pi, np.pi)
    cross = np.array(
        [[0.0, -axis[2], axis[1]],
         [axis[2], 0.0, -axis[0]],
         [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    rotation = np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)
    translation = np.array([1.2, -0.7, 0.4])

    common = source[:l] @ rotation.T + translation
    common += rng.normal(0.0, noise, size=(l, 3))
    distractors = rng.normal(0.0, outlier_scale, size=(n - l, 3))
    distractors += np.array([3.0, 2.0, -1.0])
    target_unpermuted = np.vstack([common, distractors]).astype(np.float32)
    permutation = rng.permutation(n)
    target = target_unpermuted[permutation]
    ground_truth = {i: int(np.flatnonzero(permutation == i)[0]) for i in range(l)}
    return source, target, ground_truth


def rbf_adjacencies(source: np.ndarray, target: np.ndarray):
    d_source = np.linalg.norm(source[:, None, :] - source[None, :, :], axis=-1)
    d_target = np.linalg.norm(target[:, None, :] - target[None, :, :], axis=-1)
    positive = d_source[d_source > 0]
    sigma = max(float(np.median(positive)) if positive.size else 1.0, 1e-3)
    a_source = np.exp(-0.5 * (d_source / sigma) ** 2).astype(np.float32)
    a_target = np.exp(-0.5 * (d_target / sigma) ** 2).astype(np.float32)
    np.fill_diagonal(a_source, 0.0)
    np.fill_diagonal(a_target, 0.0)
    return a_source, a_target


def exact_l_icp(source, target, l, max_iter=20):
    """ICP with v3's exact-cardinality Hungarian assignment at each iteration."""
    m, n = len(source), len(target)
    assignment = AdvancedGNCCP()
    rotation = np.eye(source.shape[1])
    translation = target.mean(axis=0) - source.mean(axis=0)
    previous = np.inf
    best_matches = []

    for _ in range(max_iter):
        aligned = source @ rotation.T + translation
        costs = np.linalg.norm(aligned[:, None, :] - target[None, :, :], axis=-1)
        rows, cols = assignment._augmented_assignment(costs, l, m, n)
        next_rotation, next_translation = kabsch(source[rows], target[cols])
        rmsd = robust_kabsch_superposition(
            source[rows], target[cols], trim_ratio=0.0
        )["rmsd"]
        if rmsd >= previous - 1e-10:
            break
        previous = rmsd
        rotation, translation = next_rotation, next_translation
        best_matches = list(zip(rows.tolist(), cols.tolist()))

    return best_matches


def metric_row(matches, source, target, ground_truth, l, elapsed_ms):
    correct = sum(ground_truth.get(i) == j for i, j in matches)
    pair_accuracy = correct / l
    if len(matches) >= 3:
        pairs = np.asarray(matches, dtype=int)
        rmsd = robust_kabsch_superposition(
            source[pairs[:, 0]], target[pairs[:, 1]], trim_ratio=0.0
        )["rmsd"]
    else:
        rmsd = float("nan")
    return {"pair_accuracy": pair_accuracy, "all_pair_rmsd": rmsd, "wall_ms": elapsed_ms}


def run(n_pairs: int, seed_offset: int, output: Path):
    m, n, l = 16, 22, 12
    settings = {
        "v3_default": {"max_iter": 8, "fw_max_iter": 3, "n_restarts": 1},
        "v3_fw6": {"max_iter": 8, "fw_max_iter": 6, "n_restarts": 1},
        "v3_sigma085": {"max_iter": 8, "fw_max_iter": 3, "n_restarts": 1},
        "v3_sigma085_fw2": {"max_iter": 8, "fw_max_iter": 2, "n_restarts": 1},
        "v3_sigma085_polish3": {"max_iter": 8, "fw_max_iter": 3, "n_restarts": 1},
        "v3_sigma085_polish4": {"max_iter": 8, "fw_max_iter": 3, "n_restarts": 1},
        "v2": {"max_iter": 8, "fw_max_iter": 3, "n_restarts": 1},
        "yang2018_impl": {"max_iter": 8, "fw_iter": 3},
        "icp_hungarian": {},
    }
    sigma_scales = {
        "v3_default": 1.0,
        "v3_fw6": 1.0,
        "v3_sigma085": 0.85,
        "v3_sigma085_fw2": 0.85,
        "v3_sigma085_polish3": 0.85,
        "v3_sigma085_polish4": 0.85,
    }
    dgm_iters = {
        "v3_default": 2,
        "v3_fw6": 2,
        "v3_sigma085": 2,
        "v3_sigma085_fw2": 2,
        "v3_sigma085_polish3": 3,
        "v3_sigma085_polish4": 4,
    }
    samples = {name: [] for name in settings}

    for index in range(n_pairs):
        seed = seed_offset + index
        source, target, ground_truth = make_pair(seed, m, n, l, noise=0.03, outlier_scale=1.5)
        build_start = time.perf_counter()
        a_source, a_target = rbf_adjacencies(source, target)
        graph_build_ms = (time.perf_counter() - build_start) * 1000.0

        solvers = {
            "v3_default": AdvancedGNCCP(alpha=0.7, seed=seed, local_refinement=True, **settings["v3_default"]),
            "v3_fw6": AdvancedGNCCP(alpha=0.7, seed=seed, local_refinement=True, **settings["v3_fw6"]),
            "v3_sigma085": AdvancedGNCCP(alpha=0.7, seed=seed, local_refinement=True, **settings["v3_sigma085"]),
            "v3_sigma085_fw2": AdvancedGNCCP(alpha=0.7, seed=seed, local_refinement=True, **settings["v3_sigma085_fw2"]),
            "v3_sigma085_polish3": AdvancedGNCCP(alpha=0.7, seed=seed, local_refinement=True, **settings["v3_sigma085_polish3"]),
            "v3_sigma085_polish4": AdvancedGNCCP(alpha=0.7, seed=seed, local_refinement=True, **settings["v3_sigma085_polish4"]),
            "v2": UniversalGNCCP(
                alpha=0.7, seed=seed, adaptive_schedule=False,
                local_refinement=True, **settings["v2"]
            ),
            "yang2018_impl": OriginalGNCCP_2018(alpha=0.7, **settings["yang2018_impl"]),
        }

        # Balance CPU/order effects across methods instead of always timing one
        # solver first and another last for every generated pair.
        solver_order = list(solvers)
        np.random.default_rng(seed).shuffle(solver_order)
        for name in solver_order:
            solver = solvers[name]
            start = time.perf_counter()
            if name.startswith("v3"):
                result = solver.match_point_clouds(
                    source, target, L=l, n_dgm_iters=dgm_iters[name],
                    rbf_sigma_scale=sigma_scales[name],
                )
            elif name == "v2":
                result = solver.match_point_clouds(source, target, L=l, n_dgm_iters=2)
            else:
                result = solver.solve(a_source, a_target, L=l)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            # v3/v2 wrappers include graph construction; account for it when
            # timing the 2018 solver, which accepts prebuilt adjacency matrices.
            if name == "yang2018_impl":
                elapsed_ms += graph_build_ms
            samples[name].append(
                metric_row(result.matches, source, target, ground_truth, l, elapsed_ms)
            )

        start = time.perf_counter()
        matches = exact_l_icp(source, target, l)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        samples["icp_hungarian"].append(
            metric_row(matches, source, target, ground_truth, l, elapsed_ms)
        )

    rng = np.random.default_rng(240615)
    summary = {}
    for name, rows in samples.items():
        accuracy = np.asarray([row["pair_accuracy"] for row in rows])
        rmsd = np.asarray([row["all_pair_rmsd"] for row in rows])
        runtime = np.asarray([row["wall_ms"] for row in rows])
        boot = np.mean(
            rng.choice(accuracy, size=(5000, len(accuracy)), replace=True), axis=1
        )
        summary[name] = {
            "mean_pair_accuracy": float(accuracy.mean()),
            "bootstrap_95_ci_pair_accuracy": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
            "median_all_pair_rmsd": float(np.nanmedian(rmsd)),
            "median_wall_ms": float(np.median(runtime)),
        }

    payload = {
        "protocol": {
            "type": "exploratory synthetic rigid 3D correspondence stress test",
            "pairs": n_pairs,
            "seed_offset": seed_offset,
            "M": m,
            "N": n,
            "L": l,
            "matched_coordinate_noise_sigma": 0.03,
            "distractor_coordinate_scale": 1.5,
            "target_pose": "random SO(3) rotation plus [1.2, -0.7, 0.4] translation",
            "target_order": "randomly permuted",
            "metrics": "pair accuracy over L ground-truth pairs; all-pair Kabsch RMSD; wall-clock method call",
            "timing_order": "deterministically randomized per pair to reduce order bias",
            "limitation": "not a molecular dataset or a reproduction of a published point-cloud benchmark",
        },
        "method_settings": settings,
        "kernel_bandwidth_scale": sigma_scales,
        "point_cloud_refinement_passes": dgm_iters,
        "summary": summary,
        "samples": samples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for name, values in summary.items():
        lo, hi = values["bootstrap_95_ci_pair_accuracy"]
        print(
            f"{name:16s} pair_acc={values['mean_pair_accuracy'] * 100:5.1f}% "
            f"boot95=[{lo * 100:.1f},{hi * 100:.1f}] "
            f"RMSDmed={values['median_all_pair_rmsd']:.3f} units "
            f"wallmed={values['median_wall_ms']:.2f}ms"
        )
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--seed-offset", type=int, default=9000)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).with_name("v3_pointcloud_synthetic_results.json"),
    )
    args = parser.parse_args()
    run(args.pairs, args.seed_offset, args.output)
