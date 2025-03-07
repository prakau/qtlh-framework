"""
Tests for the validation module.
"""

import numpy as np
import torch
import pytest
from sklearn.linear_model import LogisticRegression
from qtlh.validation import (
    ValidationConfig,
    Validator,
    PerformanceMetrics,
    CrossValidator,
    StatisticalAnalyzer,
    ResultsVisualizer
)

@pytest.fixture
def dummy_model():
    """Create a simple model for testing."""
    return LogisticRegression(random_state=42)

@pytest.fixture
def dummy_data():
    """Create dummy data for testing."""
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, size=100)
    return X, y

@pytest.mark.validation
def test_validation_config():
    """Test validation configuration."""
    config = ValidationConfig(
        n_splits=5,
        test_size=0.2,
        n_bootstrap=1000,
        confidence_level=0.95
    )
    assert config.n_splits == 5
    assert config.test_size == 0.2
    assert config.n_bootstrap == 1000
    assert config.confidence_level == 0.95
    assert isinstance(config.metrics, list)
    assert len(config.metrics) > 0

@pytest.mark.validation
def test_performance_metrics(dummy_model, dummy_data):
    """Test performance metrics computation."""
    X, y = dummy_data
    dummy_model.fit(X[:80], y[:80])
    y_pred = dummy_model.predict(X[80:])
    y_prob = dummy_model.predict_proba(X[80:])[:, 1]
    y_true = y[80:]
    
    metrics = PerformanceMetrics.compute_all_metrics(y_true, y_pred, y_prob)
    
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'roc_auc' in metrics
    assert all(0 <= v <= 1 for v in metrics.values())

@pytest.mark.validation
def test_confidence_intervals():
    """Test confidence interval computation."""
    metric_values = np.random.randn(1000)
    lower, upper = PerformanceMetrics.compute_confidence_intervals(
        metric_values,
        confidence_level=0.95
    )
    assert lower < upper
    assert np.percentile(metric_values, 2.5) == pytest.approx(lower, rel=1e-10)
    assert np.percentile(metric_values, 97.5) == pytest.approx(upper, rel=1e-10)

@pytest.mark.validation
def test_cross_validation(validation_config, dummy_model, dummy_data):
    """Test cross-validation procedures."""
    cv = CrossValidator(validation_config)
    X, y = dummy_data
    
    results = cv.perform_cross_validation(dummy_model, X, y)
    
    assert isinstance(results, dict)
    assert all(metric in results for metric in validation_config.metrics)
    assert all(isinstance(v, np.ndarray) for v in results.values())
    assert all(len(v) == validation_config.n_splits for v in results.values())

@pytest.mark.validation
def test_statistical_analysis():
    """Test statistical analysis tools."""
    method1_results = np.random.randn(100)
    method2_results = np.random.randn(100) + 0.5
    
    stats = StatisticalAnalyzer.compute_statistical_tests(
        method1_results,
        method2_results
    )
    
    assert 't_statistic' in stats
    assert 't_pvalue' in stats
    assert 'wilcoxon_statistic' in stats
    assert 'wilcoxon_pvalue' in stats
    assert 'cohens_d' in stats

@pytest.mark.validation
def test_feature_importance_analysis():
    """Test feature importance analysis."""
    feature_importance = np.random.rand(10)
    feature_names = [f'feature_{i}' for i in range(10)]
    
    results = StatisticalAnalyzer.analyze_feature_importance(
        feature_importance,
        feature_names
    )
    
    assert len(results) == 10
    assert 'feature' in results.columns
    assert 'importance' in results.columns
    assert results['importance'].is_monotonic_decreasing

@pytest.mark.validation
def test_results_visualization(tmp_path):
    """Test results visualization."""
    visualizer = ResultsVisualizer(save_dir=tmp_path)
    
    # Create mock metrics
    metrics = {
        'accuracy': np.random.rand(100),
        'precision': np.random.rand(100),
        'recall': np.random.rand(100)
    }
    
    # Test plot generation
    visualizer.plot_metrics_distribution(metrics)
    assert (tmp_path / "metrics_distribution.png").exists()
    
    # Test comparison plot
    visualizer.plot_comparison(
        np.random.rand(100),
        np.random.rand(100),
        "Method 1",
        "Method 2",
        "Accuracy"
    )
    assert (tmp_path / "comparison_Accuracy.png").exists()

@pytest.mark.validation
def test_validator_comprehensive(validator, dummy_model, dummy_data):
    """Test comprehensive validation workflow."""
    X, y = dummy_data
    
    results = validator.validate_model(
        model=dummy_model,
        X=X,
        y=y,
        feature_names=[f'feature_{i}' for i in range(X.shape[1])]
    )
    
    assert 'cv_results' in results
    assert 'confidence_intervals' in results
    assert 'feature_importance' in results
    assert 'summary_stats' in results

@pytest.mark.validation
def test_model_comparison(validator, dummy_data):
    """Test model comparison functionality."""
    X, y = dummy_data
    
    model1 = LogisticRegression(random_state=42)
    model2 = LogisticRegression(random_state=42, C=0.1)
    
    comparison = validator.compare_models(
        model1=model1,
        model2=model2,
        X=X,
        y=y,
        model1_name="Default LR",
        model2_name="Regularized LR"
    )
    
    assert 'model1_results' in comparison
    assert 'model2_results' in comparison
    assert 'statistical_tests' in comparison

@pytest.mark.validation
def test_results_saving_loading(validator, dummy_model, dummy_data, tmp_path):
    """Test saving and loading validation results."""
    X, y = dummy_data
    
    # Generate results
    results = validator.validate_model(dummy_model, X, y)
    
    # Save results
    validator.save_results(results, "validation_results.json")
    
    # Load results
    loaded_results = validator.load_results("validation_results.json")
    
    # Compare original and loaded results
    assert set(results.keys()) == set(loaded_results.keys())
    for key in results:
        if isinstance(results[key], np.ndarray):
            assert np.allclose(results[key], loaded_results[key])
        elif isinstance(results[key], dict):
            assert results[key] == loaded_results[key]

@pytest.mark.validation
def test_edge_cases(validator):
    """Test handling of edge cases."""
    # Test with empty data
    X_empty = np.array([]).reshape(0, 10)
    y_empty = np.array([])
    
    with pytest.raises(ValueError):
        validator.validate_model(dummy_model(), X_empty, y_empty)
    
    # Test with single class
    X = np.random.randn(100, 10)
    y = np.zeros(100)
    
    with pytest.warns(UserWarning):
        validator.validate_model(dummy_model(), X, y)

@pytest.mark.validation
@pytest.mark.parametrize("metric", [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'roc_auc'
])
def test_individual_metrics(validator, dummy_model, dummy_data, metric):
    """Test individual metric computation."""
    X, y = dummy_data
    results = validator.validate_model(dummy_model, X, y)
    
    assert metric in results['cv_results']
    assert metric in results['confidence_intervals']
    assert metric in results['summary_stats']
