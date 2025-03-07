"""
Quantum Processing Package
========================

This package implements quantum computing approaches for genomic feature extraction.

Components
----------
processor : Core quantum processing implementation
circuit : Quantum circuit definitions and operations
error : Error mitigation and correction strategies

For detailed documentation, see: https://qtlh-framework.readthedocs.io/quantum/
"""

from .processor import (
    QuantumConfig,
    QuantumProcessor,
    QuantumCircuit,
    ErrorMitigator
)

__all__ = [
    "QuantumConfig",
    "QuantumProcessor",
    "QuantumCircuit",
    "ErrorMitigator"
]
