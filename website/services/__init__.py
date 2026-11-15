"""
========================================================================================
WEBSITE BACKEND SERVICES PACKAGE
========================================================================================
Modular service layer providing domain-specific algorithms, benchmark generators,
and datasets for 3D Molecular Conformer Superposition, Synthetic Graphs, Social Networks,
Planar Image Keypoints, and Metal-Organic Frameworks.
========================================================================================
"""

from .molecular_service import (
    run_molecular_3d_superposition,
    get_available_molecules,
    PHARMA_MOLECULES
)
from .synthetic_service import (
    run_synthetic_graph_match,
    get_scalability_sweep_data
)
from .social_service import (
    run_social_network_match
)
from .image_service import (
    run_image_keypoint_match,
    get_available_image_presets,
    OXFORD_PRESETS
)
from .mof_service import (
    run_mof_match
)

__all__ = [
    "run_molecular_3d_superposition",
    "get_available_molecules",
    "PHARMA_MOLECULES",
    "run_synthetic_graph_match",
    "get_scalability_sweep_data",
    "run_social_network_match",
    "run_image_keypoint_match",
    "run_mof_match"
]
