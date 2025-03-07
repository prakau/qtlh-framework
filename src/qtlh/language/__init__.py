"""
Genomic Language Modeling Package
===============================

This package implements transformer-based language modeling for genomic sequences,
providing advanced semantic analysis and feature extraction capabilities.

Components
----------
model : Core transformer model implementation
tokenizer : K-mer based sequence tokenization
attention : Specialized attention mechanisms for motif detection
utils : Utility functions for sequence processing

For detailed documentation, see: https://qtlh-framework.readthedocs.io/language/
"""

from .model import (
    LanguageConfig,
    GenomicTransformer,
    GenomicTokenizer,
    MotifAttentionLayer,
    SequenceDataset,
    create_dataloader
)

__all__ = [
    "LanguageConfig",
    "GenomicTransformer",
    "GenomicTokenizer",
    "MotifAttentionLayer",
    "SequenceDataset",
    "create_dataloader"
]
