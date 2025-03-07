"""
QTL-H: Quantum-enhanced Topological Linguistic Hyperdimensional Framework
=====================================================================

A groundbreaking framework for genomic analysis that integrates quantum computing,
topological analysis, linguistic modeling, and hyperdimensional computing.

Modules
-------
quantum : Quantum processing and feature extraction
hd : Hyperdimensional computing operations
topology : Topological data analysis
language : Genomic language modeling
integration : Feature integration and fusion
validation : Validation and benchmarking

For more information, see: https://qtlh-framework.readthedocs.io/
"""

__version__ = "0.1.0"
__author__ = "QTL-H Development Team"
__email__ = "contact@qtlh-framework.org"

from . import quantum
from . import hd
from . import topology
from . import language
from . import integration
from . import validation

__all__ = [
    "quantum",
    "hd",
    "topology",
    "language",
    "integration",
    "validation"
]
