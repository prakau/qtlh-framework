"""
Validation Module
===============

This module implements comprehensive validation and benchmarking capabilities for the
QTL-H framework, including performance metrics, cross-validation, and statistical analysis.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Callable
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging

@dataclass
class ValidationConfig:
    """Configuration for validation procedures."""
    n_splits: int = 5
    test_size: float = 0.2
    random_state: int = 42
    n_bootstrap: int = 1000
    confidence_level: float = 0.95
    metrics: List[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                'accuracy', 'precision', 'recall', 'f1',
                'roc_auc', 'pr_auc', 'correlation'
            ]

class PerformanceMetrics:
    """Computes and tracks various performance metrics."""
    
    @staticmethod
    def compute_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute comprehensive set of performance metrics."""
        metrics = {}
        
        # Classification metrics
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)
            
        # Confusion matrix based metrics
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics.update({
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        })
        
        return metrics
    
    @staticmethod
    def compute_confidence_intervals(
        metric_values: np.ndarray,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Compute confidence intervals using bootstrap."""
        lower_percentile = (1 - confidence_level) / 2 * 100
        upper_percentile = (1 + confidence_level) / 2 * 100
        
        return np.percentile(metric_values, [lower_percentile, upper_percentile])

class CrossValidator:
    """Implements cross-validation procedures."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
    def perform_cross_validation(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        stratified: bool = True
    ) -> Dict[str, np.ndarray]:
        """Perform k-fold cross-validation."""
        # Initialize cross-validation splitter
        if stratified and len(np.unique(y)) > 1:
            kf = StratifiedKFold(
                n_splits=self.config.n_splits,
                shuffle=True,
                random_state=self.config.random_state
            )
        else:
            kf = KFold(
                n_splits=self.config.n_splits,
                shuffle=True,
                random_state=self.config.random_state
            )
            
        # Initialize results storage
        results = {metric: [] for metric in self.config.metrics}
        
        # Perform cross-validation
        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Get predictions
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Compute metrics
            fold_metrics = PerformanceMetrics.compute_all_metrics(y_val, y_pred, y_prob)
            
            # Store results
            for metric in self.config.metrics:
                if metric in fold_metrics:
                    results[metric].append(fold_metrics[metric])
                    
        # Convert lists to numpy arrays
        return {k: np.array(v) for k, v in results.items()}

class StatisticalAnalyzer:
    """Performs statistical analysis of results."""
    
    @staticmethod
    def compute_statistical_tests(
        method1_results: np.ndarray,
        method2_results: np.ndarray
    ) -> Dict[str, float]:
        """Compute statistical significance tests between two methods."""
        # Paired t-test
        t_stat, t_pvalue = stats.ttest_rel(method1_results, method2_results)
        
        # Wilcoxon signed-rank test
        w_stat, w_pvalue = stats.wilcoxon(method1_results, method2_results)
        
        # Effect size (Cohen's d)
        d = (np.mean(method1_results) - np.mean(method2_results)) / np.sqrt(
            (np.var(method1_results) + np.var(method2_results)) / 2
        )
        
        return {
            't_statistic': t_stat,
            't_pvalue': t_pvalue,
            'wilcoxon_statistic': w_stat,
            'wilcoxon_pvalue': w_pvalue,
            'cohens_d': d
        }
    
    @staticmethod
    def analyze_feature_importance(
        feature_importance: np.ndarray,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """Analyze and rank feature importance."""
        return pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

class ResultsVisualizer:
    """Visualizes validation results."""
    
    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_metrics_distribution(
        self,
        metrics: Dict[str, np.ndarray],
        title: str = "Performance Metrics Distribution"
    ) -> None:
        """Plot distribution of performance metrics."""
        plt.figure(figsize=(12, 6))
        
        for i, (metric_name, values) in enumerate(metrics.items()):
            plt.subplot(2, (len(metrics) + 1) // 2, i + 1)
            sns.histplot(values, kde=True)
            plt.title(f"{metric_name.replace('_', ' ').title()}")
            plt.xlabel("Value")
            plt.ylabel("Count")
            
        plt.tight_layout()
        
        if self.save_dir:
            plt.savefig(self.save_dir / "metrics_distribution.png")
        plt.close()
    
    def plot_comparison(
        self,
        results1: np.ndarray,
        results2: np.ndarray,
        label1: str,
        label2: str,
        metric: str
    ) -> None:
        """Plot comparison between two methods."""
        plt.figure(figsize=(8, 6))
        
        plt.boxplot([results1, results2], labels=[label1, label2])
        plt.title(f"Comparison of {metric}")
        plt.ylabel(metric)
        
        if self.save_dir:
            plt.savefig(self.save_dir / f"comparison_{metric}.png")
        plt.close()

class Validator:
    """Main validation class integrating all validation components."""
    
    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        save_dir: Optional[str] = None
    ):
        self.config = config or ValidationConfig()
        self.cross_validator = CrossValidator(self.config)
        self.visualizer = ResultsVisualizer(save_dir)
        
    def validate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive model validation."""
        # Cross-validation
        cv_results = self.cross_validator.perform_cross_validation(model, X, y)
        
        # Compute confidence intervals
        confidence_intervals = {}
        for metric, values in cv_results.items():
            ci_low, ci_high = PerformanceMetrics.compute_confidence_intervals(
                values, self.config.confidence_level
            )
            confidence_intervals[metric] = (ci_low, ci_high)
        
        # Feature importance analysis if available
        feature_importance = None
        if hasattr(model, 'feature_importances_') and feature_names:
            feature_importance = StatisticalAnalyzer.analyze_feature_importance(
                model.feature_importances_,
                feature_names
            )
        
        # Visualize results
        self.visualizer.plot_metrics_distribution(cv_results)
        
        return {
            'cv_results': cv_results,
            'confidence_intervals': confidence_intervals,
            'feature_importance': feature_importance,
            'summary_stats': {
                metric: {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
                for metric, values in cv_results.items()
            }
        }
    
    def compare_models(
        self,
        model1: Any,
        model2: Any,
        X: np.ndarray,
        y: np.ndarray,
        model1_name: str = "Model 1",
        model2_name: str = "Model 2"
    ) -> Dict[str, Any]:
        """Compare performance of two models."""
        # Validate both models
        results1 = self.validate_model(model1, X, y)
        results2 = self.validate_model(model2, X, y)
        
        # Statistical comparison
        statistical_tests = {}
        for metric in self.config.metrics:
            if metric in results1['cv_results'] and metric in results2['cv_results']:
                statistical_tests[metric] = StatisticalAnalyzer.compute_statistical_tests(
                    results1['cv_results'][metric],
                    results2['cv_results'][metric]
                )
                
                # Visualize comparison
                self.visualizer.plot_comparison(
                    results1['cv_results'][metric],
                    results2['cv_results'][metric],
                    model1_name,
                    model2_name,
                    metric
                )
        
        return {
            'model1_results': results1,
            'model2_results': results2,
            'statistical_tests': statistical_tests
        }
    
    def save_results(
        self,
        results: Dict[str, Any],
        filename: str
    ) -> None:
        """Save validation results."""
        if self.visualizer.save_dir:
            save_path = self.visualizer.save_dir / filename
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in value.items()
                    }
                elif isinstance(value, np.ndarray):
                    serializable_results[key] = value.tolist()
                else:
                    serializable_results[key] = value
            
            with open(save_path, 'w') as f:
                json.dump(serializable_results, f, indent=4)
                
    def load_results(
        self,
        filename: str
    ) -> Dict[str, Any]:
        """Load validation results."""
        if self.visualizer.save_dir:
            load_path = self.visualizer.save_dir / filename
            
            with open(load_path, 'r') as f:
                results = json.load(f)
                
            # Convert lists back to numpy arrays
            for key, value in results.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, list):
                            results[key][k] = np.array(v)
                elif isinstance(value, list):
                    results[key] = np.array(value)
                    
            return results
