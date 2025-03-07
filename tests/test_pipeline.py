"""
Integration tests for the complete QTL-H pipeline.
"""

import pytest
import numpy as np
from pathlib import Path
from Bio import SeqIO

from qtlh.quantum import QuantumProcessor, QuantumConfig
from qtlh.hd import HDComputing, HDConfig
from qtlh.topology import TopologicalAnalyzer, TopologyConfig
from qtlh.language import GenomicTransformer, LanguageConfig
from qtlh.integration import FeatureIntegrator, IntegrationConfig
from qtlh.validation import Validator, ValidationConfig

@pytest.fixture
def sample_sequences():
    """Load sample sequences from FASTA file."""
    sequences = []
    fasta_path = Path(__file__).parent / "data" / "sample_sequences.fasta"
    
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append(str(record.seq))
        
    return sequences

@pytest.fixture
def pipeline_components(
    quantum_config,
    hd_config,
    topology_config,
    language_config,
    integration_config,
    validation_config
):
    """Initialize all pipeline components."""
    return {
        'quantum': QuantumProcessor(quantum_config),
        'hd': HDComputing(hd_config),
        'topology': TopologicalAnalyzer(topology_config),
        'language': GenomicTransformer(language_config),
        'integration': FeatureIntegrator(integration_config),
        'validation': Validator(validation_config)
    }

def test_full_pipeline(pipeline_components, sample_sequences, tmp_path):
    """Test complete analysis pipeline."""
    # Process each sequence through all modules
    results = []
    
    for sequence in sample_sequences:
        # 1. Quantum Processing
        quantum_features = pipeline_components['quantum'].process_sequence(sequence)
        
        # 2. Hyperdimensional Computing
        hd_features = pipeline_components['hd'].encode_sequence(sequence)
        
        # 3. Topological Analysis
        topo_features = pipeline_components['topology'].analyze_sequence(
            quantum_features.reshape(-1, 1)
        )
        
        # 4. Language Model Analysis
        lang_features = pipeline_components['language'](
            pipeline_components['language'].tokenizer.encode(sequence).unsqueeze(0)
        )['pooled_features'].detach().numpy()
        
        # 5. Feature Integration
        integrated = pipeline_components['integration'].integrate_features(
            quantum_features=quantum_features,
            hd_features=hd_features[1],  # Using fractal encoding
            topological_features=topo_features['persistence_results']['persistence_features'],
            language_features=lang_features
        )
        
        results.append(integrated)
        
    # Validate results
    X = np.array([r['integrated_features'] for r in results])
    y = np.array([1 if 'promoter' in seq.lower() else 0 for seq in sample_sequences])
    
    validation_results = pipeline_components['validation'].validate_model(
        model=pipeline_components['language'],  # Using language model as classifier
        X=X,
        y=y
    )
    
    # Check pipeline outputs
    assert len(results) == len(sample_sequences)
    assert all('integrated_features' in r for r in results)
    assert all('feature_importance' in r for r in results)
    assert 'cv_results' in validation_results
    assert 'feature_importance' in validation_results

@pytest.mark.slow
def test_pipeline_persistence(pipeline_components, sample_sequences, tmp_path):
    """Test saving and loading of pipeline results."""
    # Process a sequence
    sequence = sample_sequences[0]
    
    # Generate results
    quantum_features = pipeline_components['quantum'].process_sequence(sequence)
    hd_features = pipeline_components['hd'].encode_sequence(sequence)
    topo_features = pipeline_components['topology'].analyze_sequence(
        quantum_features.reshape(-1, 1)
    )
    lang_features = pipeline_components['language'](
        pipeline_components['language'].tokenizer.encode(sequence).unsqueeze(0)
    )['pooled_features'].detach().numpy()
    
    # Save component states
    quantum_path = tmp_path / "quantum_state.pt"
    hd_path = tmp_path / "hd_state.pt"
    topo_path = tmp_path / "topo_state.pkl"
    lang_path = tmp_path / "lang_state.pt"
    
    pipeline_components['quantum'].save_parameters(quantum_path)
    pipeline_components['hd'].save_state(hd_path)
    pipeline_components['topology'].save_results(topo_features, topo_path)
    pipeline_components['language'].save_state_dict(lang_path)
    
    # Create new components and load states
    new_quantum = QuantumProcessor(pipeline_components['quantum'].config)
    new_hd = HDComputing(pipeline_components['hd'].config)
    new_topo = TopologicalAnalyzer(pipeline_components['topology'].config)
    new_lang = GenomicTransformer(pipeline_components['language'].config)
    
    new_quantum.load_parameters(quantum_path)
    new_hd.load_state(hd_path)
    loaded_topo = new_topo.load_results(topo_path)
    new_lang.load_state_dict(lang_path)
    
    # Compare results
    new_quantum_features = new_quantum.process_sequence(sequence)
    new_hd_features = new_hd.encode_sequence(sequence)
    new_lang_features = new_lang(
        new_lang.tokenizer.encode(sequence).unsqueeze(0)
    )['pooled_features'].detach().numpy()
    
    assert np.allclose(quantum_features, new_quantum_features)
    assert np.allclose(hd_features[1], new_hd_features[1])
    assert np.allclose(lang_features, new_lang_features)
    assert loaded_topo.keys() == topo_features.keys()

def test_error_handling(pipeline_components):
    """Test pipeline error handling."""
    # Test invalid sequence
    invalid_sequence = "ATGCX" * 10
    
    with pytest.warns(UserWarning):
        # Should handle invalid bases
        quantum_features = pipeline_components['quantum'].process_sequence(invalid_sequence)
        hd_features = pipeline_components['hd'].encode_sequence(invalid_sequence)
        
    # Test empty sequence
    empty_sequence = ""
    
    with pytest.raises(ValueError):
        pipeline_components['quantum'].process_sequence(empty_sequence)
        
    # Test sequence too short
    short_sequence = "AT"
    
    with pytest.raises(ValueError):
        pipeline_components['hd'].encode_sequence(short_sequence)

@pytest.mark.gpu
def test_gpu_acceleration(pipeline_components, sample_sequences):
    """Test GPU acceleration if available."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
        
    # Move models to GPU
    pipeline_components['language'].cuda()
    
    sequence = sample_sequences[0]
    tokens = pipeline_components['language'].tokenizer.encode(sequence).unsqueeze(0).cuda()
    
    outputs = pipeline_components['language'](tokens)
    assert all(tensor.is_cuda for tensor in outputs.values() if isinstance(tensor, torch.Tensor))

def test_pipeline_reproducibility(pipeline_components, sample_sequences):
    """Test pipeline reproducibility."""
    sequence = sample_sequences[0]
    
    # First run
    np.random.seed(42)
    torch.manual_seed(42)
    
    results1 = {
        'quantum': pipeline_components['quantum'].process_sequence(sequence),
        'hd': pipeline_components['hd'].encode_sequence(sequence),
        'topo': pipeline_components['topology'].analyze_sequence(
            np.random.randn(10, 1)  # Fixed random input
        ),
        'language': pipeline_components['language'](
            pipeline_components['language'].tokenizer.encode(sequence).unsqueeze(0)
        )['pooled_features'].detach().numpy()
    }
    
    # Second run
    np.random.seed(42)
    torch.manual_seed(42)
    
    results2 = {
        'quantum': pipeline_components['quantum'].process_sequence(sequence),
        'hd': pipeline_components['hd'].encode_sequence(sequence),
        'topo': pipeline_components['topology'].analyze_sequence(
            np.random.randn(10, 1)  # Fixed random input
        ),
        'language': pipeline_components['language'](
            pipeline_components['language'].tokenizer.encode(sequence).unsqueeze(0)
        )['pooled_features'].detach().numpy()
    }
    
    # Compare results
    assert np.allclose(results1['quantum'], results2['quantum'])
    assert np.allclose(results1['hd'][1], results2['hd'][1])
    assert np.allclose(results1['language'], results2['language'])
    assert results1['topo'].keys() == results2['topo'].keys()
