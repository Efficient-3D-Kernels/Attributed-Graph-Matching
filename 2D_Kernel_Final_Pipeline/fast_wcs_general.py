"""
fast_wcs_general.py
Convenience wrapper exporting the Flagship Advanced Fast-WCS / GNCCP V2 engine.
"""

from gnccp_advanced import AdvancedGNCCP, MatchingResult, robust_kabsch_superposition
from gnccp_generalized import UniversalGNCCP, kabsch_superposition

# Flagship SOTA Engine
FastWCS = AdvancedGNCCP

__all__ = [
    "FastWCS",
    "AdvancedGNCCP",
    "UniversalGNCCP",
    "MatchingResult",
    "robust_kabsch_superposition",
    "kabsch_superposition"
]

