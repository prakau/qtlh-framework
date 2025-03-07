"""
Validation Package
================

This package implements comprehensive validation, benchmarking, and statistical analysis
capabilities for the QTL-H framework.

Components
----------
validator : Core validation implementation
metrics : Performance metrics computation
cross_validation : Cross-validation procedures
statistics : Statistical analysis tools
visualization : Results visualization

For detailed documentation, see: https://qtlh-framework.readthedocs.io/validation/
"""

from .validator import (
    ValidationConfig,
    Validator,
    PerformanceMetrics,
    CrossValidator,
    StatisticalAnalyzer,
    ResultsVisualizer
)

__all__ = [
    "ValidationConfig",
    "Validator",
    "PerformanceMetrics",
    "CrossValidator",
    "StatisticalAnalyzer",
    "ResultsVisualizer"
]
