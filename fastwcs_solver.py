"""
========================================================================================
FAST-WCS UNIFIED DOMAIN DISPATCHER & SOLVER SUITE
========================================================================================
Provides clean, domain-specific entrypoints for the four distinct scientific domains
benchmarked in the Fast-WCS research paper:

1. 'synthetic'  -> FastWCS_Synthetic (Multi-Start Homotopy GNCCP)
   - Solves un-attributed planted subgraph isomorphism (ER, Random Regular, Scale-Free).
   - Breaks flat-barycenter symmetry via exploratory multi-start homotopy paths.
   - Script: fastwcs_synthetic.py

2. 'image'      -> FastWCS_Image (Geometric Feature-Coupled GNCCP)
   - Solves planar image keypoint matching (Delaunay spatial graphs + SIFT unary descriptors).
   - Operates in strictly O(N^2) space in 18 ms with BLAS-3 2-Opt refinement (alpha = 0.5).
   - Script: fastwcs_image_sift.py

3. 'social'     -> FastWCS_Social (Weisfeiler-Lehman Tree-History GNCCP)
   - Solves dynamic scale-free social interaction networks (Stanford CollegeMsg dataset).
   - Combines multi-hop WL tree histograms with a tree-history convex warm start to resolve
     community automorphism traps under severe structural churn.
   - Script: fastwcs_social_network.py

4. 'molecular'  -> FastWCS_Molecular3D (Hierarchical 2D-to-3D Kabsch Superposition)
   - Solves 3D flexible molecular conformer alignment under arbitrary SO(3) rotations and translations.
   - Matches rotation-invariant distance geometry + closed-form reflection-safe Kabsch SVD.
   - Script: fastwcs_molecular_3d.py

Usage:
    from fastwcs_solver import get_solver
    
    # 1. Synthetic relational graphs
    solver = get_solver("synthetic", n_restarts=10)
    result = solver.solve(A_G, A_H, L=25)
    
    # 2. Planar image keypoints
    solver = get_solver("image", alpha=0.5)
    result = solver.solve(A_G, A_H, C=C_sift, L=30)
    
    # 3. Scale-free social networks
    solver = get_solver("social", wl_iterations=3)
    result = solver.solve(A_G, A_H, L=50)
    
    # 4. 3D molecular point clouds
    solver = get_solver("molecular")
    result = solver.superpose(coords_G, coords_H, L=36)
========================================================================================
"""
from __future__ import annotations

from typing import Union, Optional, Any

from fastwcs_synthetic import (
    FastWCS_Synthetic,
    SyntheticMatchingResult,
    evaluate_unrelaxed_energy,
    make_synthetic_benchmark_instance,
    multi_start_homotopy_search,
)
from fastwcs_image_sift import (
    FastWCS_Image,
    ImageKeypointMatchingResult,
    build_delaunay_spatial_graph,
    compute_sift_feature_cost_matrix,
    compute_orb_hamming_cost_matrix,
    detect_keypoints_and_descriptors,
    match_image_keypoints,
    match_images_end_to_end,
)
from fastwcs_social_network import (
    FastWCS_Social,
    SocialNetworkMatchingResult,
    extract_wl_subtree_histories,
    build_wl_topological_cost_matrix,
    solve_wl_oa_hungarian,
    wl_oa_kernel_similarity,
    build_enhanced_social_features,
    match_social_network,
)
from fastwcs_molecular_3d import (
    FastWCS_Molecular3D,
    MolecularSuperpositionResult,
    coords_to_distance_matrix,
    compute_reflection_safe_kabsch,
    superpose_3d_molecules,
)


SOLVER_REGISTRY = {
    "synthetic": FastWCS_Synthetic,
    "relational": FastWCS_Synthetic,
    "image": FastWCS_Image,
    "sift": FastWCS_Image,
    "social": FastWCS_Social,
    "network": FastWCS_Social,
    "molecular": FastWCS_Molecular3D,
    "molecular_3d": FastWCS_Molecular3D,
    "3d": FastWCS_Molecular3D,
}


def get_solver(domain: str, **kwargs) -> Any:
    """Factory function returning the specialized, optimal Fast-WCS solver for a given domain.
    
    Args:
        domain: One of 'synthetic', 'image', 'social', or 'molecular'.
        **kwargs: Domain-specific hyperparameter overrides.
        
    Returns:
        Instantiated domain solver.
    """
    key = domain.strip().lower()
    if key not in SOLVER_REGISTRY:
        valid = sorted(list(set(SOLVER_REGISTRY.keys())))
        raise ValueError(f"Unknown domain '{domain}'. Supported domains: {valid}")
    return SOLVER_REGISTRY[key](**kwargs)


__all__ = [
    # Classes & Factory
    "FastWCS_Synthetic",
    "FastWCS_Image",
    "FastWCS_Social",
    "FastWCS_Molecular3D",
    "get_solver",
    # Dataclasses
    "SyntheticMatchingResult",
    "ImageKeypointMatchingResult",
    "SocialNetworkMatchingResult",
    "MolecularSuperpositionResult",
    # Standalone Modular Functions
    "evaluate_unrelaxed_energy",
    "make_synthetic_benchmark_instance",
    "multi_start_homotopy_search",
    "build_delaunay_spatial_graph",
    "compute_sift_feature_cost_matrix",
    "compute_orb_hamming_cost_matrix",
    "detect_keypoints_and_descriptors",
    "match_image_keypoints",
    "match_images_end_to_end",
    "extract_wl_subtree_histories",
    "build_wl_topological_cost_matrix",
    "solve_wl_oa_hungarian",
    "wl_oa_kernel_similarity",
    "build_enhanced_social_features",
    "match_social_network",
    "coords_to_distance_matrix",
    "compute_reflection_safe_kabsch",
    "superpose_3d_molecules",
]
