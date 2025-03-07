"""
Hyperdimensional Computing Package
================================

This package implements hyperdimensional computing operations for genomic analysis,
combining tensor-based and fractal encoding approaches.

Components
----------
encoder : Core hyperdimensional encoding implementation
tensor : Tensor-based sequence encoding
fractal : Fractal pattern encoding and analysis

For detailed documentation, see: https://qtlh-framework.readthedocs.io/hd/
"""

from .encoder import (
    HDConfig,
    HDComputing,
    TensorEncoder,
    FractalEncoder
)

__all__ = [
    "HDConfig",
    "HDComputing",
    "TensorEncoder",
    "FractalEncoder"
]
