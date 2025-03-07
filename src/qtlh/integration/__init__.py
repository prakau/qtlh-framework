"""
Feature Integration Package
=========================

This package implements advanced feature integration techniques for combining outputs
from multiple analysis approaches into unified genomic representations.

Components
----------
integrator : Core feature integration implementation
fusion : Feature fusion strategies
attention : Attention-based feature combination
utils : Utility functions for feature manipulation

For detailed documentation, see: https://qtlh-framework.readthedocs.io/integration/
"""

from .integrator import (
    IntegrationConfig,
    FeatureIntegrator,
    IntegratedAnalyzer,
    FusionStrategy,
    WeightedConcatenation,
    AttentionFusion,
    FeatureFusionLayer
)

__all__ = [
    "IntegrationConfig",
    "FeatureIntegrator",
    "IntegratedAnalyzer",
    "FusionStrategy",
    "WeightedConcatenation",
    "AttentionFusion",
    "FeatureFusionLayer"
]
