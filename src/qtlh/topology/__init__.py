"""
Topological Analysis Package
==========================

This package implements advanced topological data analysis techniques for genomic sequences,
providing tools for persistent homology computation and topological feature extraction.

Components
----------
analyzer : Core topological analysis implementation
homology : Persistent homology computation
mapper : Mapper algorithm implementation
utils : Utility functions for topological computations

For detailed documentation, see: https://qtlh-framework.readthedocs.io/topology/
"""

from .analyzer import (
    TopologyConfig,
    TopologicalAnalyzer,
    PersistentHomology,
    MapperAlgorithm
)

__all__ = [
    "TopologyConfig",
    "TopologicalAnalyzer",
    "PersistentHomology",
    "MapperAlgorithm"
]
