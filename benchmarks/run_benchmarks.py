#!/usr/bin/env python3
"""
Benchmarking suite for QTL-H Framework components.
Run with: python -m pytest benchmarks/run_benchmarks.py --benchmark-only
"""

import numpy as np
import pytest
import torch
from qtlh.quantum import QuantumProcessor
from qtlh.hd import HDComputing
from qtlh.topology import TopologicalAnalyzer
from qtlh.language import GenomicTransformer
from qtlh.integration import FeatureIntegrator

# Test data generation
def generate_test_sequence(length):
    """Generate random genomic sequence."""
    bases = ['A', 'T', 'G', 'C']
    return ''.join(np.random.choice(bases) for _ in range(length))

@pytest.fixture
def quantum_processor():
    """Initialize quantum processor."""
    return QuantumProcessor()

@pytest.fixture
def hd_computer():
    """Initialize HD computing module."""
    return HDComputing()

@pytest.fixture
def topo_analyzer():
    """Initialize topological analyzer."""
    return TopologicalAnalyzer()

@pytest.fixture
def language_model():
    """Initialize language model."""
    return GenomicTransformer()

@pytest.fixture
def integrator():
    """Initialize feature integrator."""
    return FeatureIntegrator()

@pytest.mark.benchmark(
    group="quantum",
    min_rounds=50,
    warmup=True
)
def test_quantum_processing(benchmark, quantum_processor):
    """Benchmark quantum processing."""
    sequence = generate_test_sequence(1000)
    benchmark(quantum_processor.process_sequence, sequence)

@pytest.mark.benchmark(
    group="hd",
    min_rounds=50,
    warmup=True
)
def test_hd_encoding(benchmark, hd_computer):
    """Benchmark hyperdimensional computing."""
    sequence = generate_test_sequence(1000)
    benchmark(hd_computer.encode_sequence, sequence)

@pytest.mark.benchmark(
    group="topology",
    min_rounds=50,
    warmup=True
)
def test_topology_analysis(benchmark, topo_analyzer):
    """Benchmark topological analysis."""
    data = np.random.randn(100, 10)
    benchmark(topo_analyzer.analyze_sequence, data)

@pytest.mark.benchmark(
    group="language",
    min_rounds=50,
    warmup=True
)
def test_language_model(benchmark, language_model):
    """Benchmark language model processing."""
    sequence = generate_test_sequence(1000)
    tokens = language_model.tokenizer.encode(sequence).unsqueeze(0)
    benchmark(language_model.forward, tokens)

@pytest.mark.benchmark(
    group="integration",
    min_rounds=50,
    warmup=True
)
def test_feature_integration(benchmark, integrator):
    """Benchmark feature integration."""
    features = {
        'quantum': np.random.randn(10, 32),
        'hd': np.random.randn(10, 32),
        'topology': np.random.randn(10, 32),
        'language': np.random.randn(10, 32)
    }
    benchmark(integrator.integrate_features, **features)

@pytest.mark.benchmark(
    group="pipeline",
    min_rounds=10,
    warmup=True
)
def test_full_pipeline(
    benchmark,
    quantum_processor,
    hd_computer,
    topo_analyzer,
    language_model,
    integrator
):
    """Benchmark complete analysis pipeline."""
    def run_pipeline():
        sequence = generate_test_sequence(1000)
        
        # Quantum processing
        quantum_features = quantum_processor.process_sequence(sequence)
        
        # HD computing
        hd_features = hd_computer.encode_sequence(sequence)
        
        # Topological analysis
        topo_features = topo_analyzer.analyze_sequence(quantum_features.reshape(-1, 1))
        
        # Language model
        tokens = language_model.tokenizer.encode(sequence).unsqueeze(0)
        lang_features = language_model(tokens)['pooled_features'].detach().numpy()
        
        # Integration
        return integrator.integrate_features(
            quantum_features=quantum_features,
            hd_features=hd_features[1],
            topological_features=topo_features['persistence_results']['persistence_features'],
            language_features=lang_features
        )
    
    benchmark(run_pipeline)

@pytest.mark.benchmark(
    group="gpu",
    min_rounds=50,
    warmup=True
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_gpu_acceleration(benchmark, language_model):
    """Benchmark GPU acceleration."""
    sequence = generate_test_sequence(2000)
    tokens = language_model.tokenizer.encode(sequence).unsqueeze(0).cuda()
    language_model.cuda()
    
    benchmark(language_model.forward, tokens)

def print_benchmark_results(results):
    """Print benchmark results in a formatted table."""
    print("\nQTL-H Framework Benchmark Results")
    print("=" * 80)
    print(f"{'Component':<20} {'Mean Time (ms)':<15} {'Min Time (ms)':<15} {'Max Time (ms)':<15}")
    print("-" * 80)
    
    for result in results:
        name = result.group
        mean = result.stats['mean'] * 1000
        min_time = result.stats['min'] * 1000
        max_time = result.stats['max'] * 1000
        print(f"{name:<20} {mean:<15.2f} {min_time:<15.2f} {max_time:<15.2f}")
    
    print("=" * 80)

if __name__ == "__main__":
    pytest.main([
        "--benchmark-only",
        "--benchmark-autosave",
        __file__
    ])
