"""
Shared test configurations and fixtures for QTL-H framework.
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from qtlh.quantum import QuantumProcessor, QuantumConfig
from qtlh.hd import HDComputing, HDConfig
from qtlh.topology import TopologicalAnalyzer, TopologyConfig
from qtlh.language import GenomicTransformer, LanguageConfig
from qtlh.integration import FeatureIntegrator, IntegrationConfig
from qtlh.validation import Validator, ValidationConfig

@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "data"

@pytest.fixture(scope="session")
def sample_sequences():
    """Generate sample genomic sequences for testing."""
    sequences = [
        "ATGCTAGCTAGCTGATCGATCG",  # Standard sequence
        "ATGCTAGCTAGCTGATCGATCGATCG",  # Longer sequence
        "ATGC",  # Short sequence
        "NNNN"  # Unknown bases
    ]
    return sequences

@pytest.fixture(scope="session")
def quantum_config():
    """Quantum processor configuration for testing."""
    return QuantumConfig(
        n_qubits=4,
        n_layers=2,
        learning_rate=0.01,
        error_threshold=1e-6
    )

@pytest.fixture(scope="session")
def hd_config():
    """Hyperdimensional computing configuration for testing."""
    return HDConfig(
        dimension=1000,  # Smaller dimension for testing
        min_kmer=3,
        max_kmer=5,
        tensor_order=2,
        fractal_levels=2
    )

@pytest.fixture(scope="session")
def topology_config():
    """Topological analysis configuration for testing."""
    return TopologyConfig(
        max_dimension=2,
        max_scale=1.0,
        n_bins=20,
        overlap_fraction=0.2
    )

@pytest.fixture(scope="session")
def language_config():
    """Language model configuration for testing."""
    return LanguageConfig(
        vocab_size=1024,  # Smaller vocabulary for testing
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=512
    )

@pytest.fixture(scope="session")
def integration_config():
    """Feature integration configuration for testing."""
    return IntegrationConfig(
        feature_dimension=128,
        n_components=32,
        fusion_temperature=0.1,
        attention_heads=4
    )

@pytest.fixture(scope="session")
def validation_config():
    """Validation configuration for testing."""
    return ValidationConfig(
        n_splits=3,
        test_size=0.2,
        n_bootstrap=100,
        confidence_level=0.95
    )

@pytest.fixture
def quantum_processor(quantum_config):
    """Initialize quantum processor for testing."""
    return QuantumProcessor(quantum_config)

@pytest.fixture
def hd_computer(hd_config):
    """Initialize HD computing module for testing."""
    return HDComputing(hd_config)

@pytest.fixture
def topo_analyzer(topology_config):
    """Initialize topological analyzer for testing."""
    return TopologicalAnalyzer(topology_config)

@pytest.fixture
def language_model(language_config):
    """Initialize language model for testing."""
    return GenomicTransformer(language_config)

@pytest.fixture
def feature_integrator(integration_config):
    """Initialize feature integrator for testing."""
    return FeatureIntegrator(integration_config)

@pytest.fixture
def validator(validation_config, tmp_path):
    """Initialize validator for testing."""
    return Validator(validation_config, save_dir=tmp_path)

@pytest.fixture
def mock_features():
    """Generate mock features for testing."""
    return {
        'quantum': np.random.randn(10, 8),
        'hd': np.random.randn(10, 16),
        'topological': np.random.randn(10, 4),
        'language': np.random.randn(10, 12)
    }

@pytest.fixture
def mock_labels():
    """Generate mock labels for testing."""
    return np.random.randint(0, 2, size=10)

@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")

def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers and conditions."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
