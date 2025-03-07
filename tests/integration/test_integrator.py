"""
Tests for the feature integration module.
"""

import numpy as np
import torch
import pytest
from qtlh.integration import (
    IntegrationConfig,
    FeatureIntegrator,
    IntegratedAnalyzer,
    FusionStrategy,
    WeightedConcatenation,
    AttentionFusion,
    FeatureFusionLayer
)

@pytest.mark.integration
def test_integration_config():
    """Test integration configuration."""
    config = IntegrationConfig(
        feature_dimension=128,
        n_components=32,
        fusion_temperature=0.1,
        attention_heads=4
    )
    assert config.feature_dimension == 128
    assert config.n_components == 32
    assert config.fusion_temperature == 0.1
    assert config.attention_heads == 4

@pytest.mark.integration
def test_feature_fusion_layer(integration_config):
    """Test feature fusion layer."""
    fusion_layer = FeatureFusionLayer(integration_config)
    
    # Create test features
    features = [
        torch.randn(4, 10, integration_config.feature_dimension)
        for _ in range(4)  # 4 different feature types
    ]
    
    # Test fusion
    fused = fusion_layer(features)
    assert isinstance(fused, torch.Tensor)
    assert fused.shape == (4, 10, integration_config.feature_dimension)

@pytest.mark.integration
def test_weighted_concatenation():
    """Test weighted concatenation strategy."""
    weights = {
        'quantum': 1.0,
        'hyperdimensional': 0.8,
        'topological': 0.6,
        'language': 0.7
    }
    strategy = WeightedConcatenation(weights)
    
    # Create test features
    features = {
        'quantum': np.random.randn(10, 5),
        'hyperdimensional': np.random.randn(10, 8),
        'topological': np.random.randn(10, 3),
        'language': np.random.randn(10, 6)
    }
    
    # Test fusion
    fused = strategy.fuse(features)
    assert isinstance(fused, np.ndarray)
    assert fused.shape == (10, sum(f.shape[1] for f in features.values()))

@pytest.mark.integration
def test_attention_fusion(integration_config):
    """Test attention-based fusion strategy."""
    strategy = AttentionFusion(integration_config)
    
    features = {
        'quantum': np.random.randn(10, integration_config.feature_dimension),
        'hyperdimensional': np.random.randn(10, integration_config.feature_dimension),
        'topological': np.random.randn(10, integration_config.feature_dimension),
        'language': np.random.randn(10, integration_config.feature_dimension)
    }
    
    fused = strategy.fuse(features)
    assert isinstance(fused, np.ndarray)
    assert fused.shape == (10, integration_config.feature_dimension)

@pytest.mark.integration
def test_feature_integrator(feature_integrator, mock_features):
    """Test main feature integrator."""
    results = feature_integrator.integrate_features(
        quantum_features=mock_features['quantum'],
        hd_features=mock_features['hd'],
        topological_features=mock_features['topological'],
        language_features=mock_features['language']
    )
    
    assert 'integrated_features' in results
    assert 'reduced_features' in results
    assert 'feature_importance' in results
    assert 'individual_features' in results

@pytest.mark.integration
def test_feature_normalization(feature_integrator):
    """Test feature normalization."""
    # Create features with different scales
    features = {
        'quantum': np.random.randn(10, 5) * 100,  # Large scale
        'hd': np.random.randn(10, 5) * 0.01,      # Small scale
        'topological': np.random.randn(10, 5),     # Normal scale
        'language': np.random.randn(10, 5) * 10    # Medium scale
    }
    
    normalized = {
        name: feature_integrator._normalize_features(f)
        for name, f in features.items()
    }
    
    for feature in normalized.values():
        assert np.allclose(np.linalg.norm(feature, axis=1), 1.0)

@pytest.mark.integration
def test_feature_relationships(feature_integrator, mock_features):
    """Test feature relationship analysis."""
    relationships = feature_integrator.analyze_feature_relationships(mock_features)
    
    # Check correlation matrix properties
    assert isinstance(relationships, dict)
    for key, value in relationships.items():
        assert isinstance(value, float)
        assert -1 <= value <= 1

@pytest.mark.integration
def test_dimensionality_reduction(feature_integrator, mock_features):
    """Test dimensionality reduction."""
    results = feature_integrator.integrate_features(**mock_features)
    reduced = results['reduced_features']
    
    assert reduced.shape[1] == feature_integrator.config.n_components
    assert np.all(np.isfinite(reduced))

@pytest.mark.integration
def test_adaptive_weighting(integration_config):
    """Test adaptive weighting mechanism."""
    config = IntegrationConfig(**{
        **vars(integration_config),
        'use_adaptive_weighting': True
    })
    integrator = FeatureIntegrator(config)
    
    results = integrator.integrate_features(
        quantum_features=np.random.randn(10, 8),
        hd_features=np.random.randn(10, 8),
        topological_features=np.random.randn(10, 8),
        language_features=np.random.randn(10, 8)
    )
    
    assert 'integrated_features' in results
    assert results['integrated_features'].shape[1] == 8

@pytest.mark.integration
def test_integrated_analyzer(
    quantum_processor,
    hd_computer,
    topo_analyzer,
    language_model,
    feature_integrator
):
    """Test integrated analyzer."""
    analyzer = IntegratedAnalyzer(feature_integrator)
    sequence = "ATGCTAGCTAGCTGATCGATCG"
    
    results = analyzer.analyze_sequence(
        sequence=sequence,
        quantum_processor=quantum_processor,
        hd_computer=hd_computer,
        topological_analyzer=topo_analyzer,
        language_model=language_model
    )
    
    assert 'integrated_results' in results
    assert 'feature_relationships' in results
    assert 'feature_importance' in results

@pytest.mark.integration
def test_state_saving_loading(feature_integrator, mock_features, tmp_path):
    """Test saving and loading integrator state."""
    # Generate initial results
    initial_results = feature_integrator.integrate_features(**mock_features)
    
    # Save state
    save_path = tmp_path / "integrator_state.pt"
    feature_integrator.save_state(save_path)
    
    # Create new integrator and load state
    new_integrator = FeatureIntegrator(feature_integrator.config)
    new_integrator.load_state(save_path)
    
    # Generate results with loaded state
    new_results = new_integrator.integrate_features(**mock_features)
    
    # Compare results
    assert np.allclose(
        initial_results['integrated_features'],
        new_results['integrated_features']
    )

@pytest.mark.integration
@pytest.mark.parametrize("fusion_strategy", [
    WeightedConcatenation,
    AttentionFusion
])
def test_fusion_strategies(integration_config, mock_features, fusion_strategy):
    """Test different fusion strategies."""
    strategy = fusion_strategy(integration_config)
    integrator = FeatureIntegrator(
        config=integration_config,
        fusion_strategy=strategy
    )
    
    results = integrator.integrate_features(**mock_features)
    assert 'integrated_features' in results
    assert isinstance(results['integrated_features'], np.ndarray)
