"""
Tests for the hyperdimensional computing module.
"""

import numpy as np
import torch
import pytest
from qtlh.hd import HDComputing, HDConfig, TensorEncoder, FractalEncoder

@pytest.mark.hd
def test_hd_config():
    """Test HD configuration initialization."""
    config = HDConfig(
        dimension=1000,
        min_kmer=3,
        max_kmer=5,
        tensor_order=2,
        fractal_levels=2
    )
    assert config.dimension == 1000
    assert config.min_kmer == 3
    assert config.max_kmer == 5
    assert config.tensor_order == 2
    assert config.fractal_levels == 2

@pytest.mark.hd
def test_tensor_encoder_initialization(hd_config):
    """Test tensor encoder initialization."""
    encoder = TensorEncoder(hd_config)
    
    # Check base tensors
    assert isinstance(encoder.base_tensors, dict)
    assert all(base in encoder.base_tensors for base in ['A', 'T', 'G', 'C', 'N'])
    assert all(isinstance(t, torch.Tensor) for t in encoder.base_tensors.values())
    
    # Check tensor shapes
    expected_shape = tuple([hd_config.dimension] * hd_config.tensor_order)
    assert all(t.shape == expected_shape for t in encoder.base_tensors.values())

@pytest.mark.hd
def test_fractal_encoder_initialization(hd_config):
    """Test fractal encoder initialization."""
    encoder = FractalEncoder(hd_config)
    
    # Check base vectors
    assert isinstance(encoder.base_vectors, dict)
    assert all(base in encoder.base_vectors for base in ['A', 'T', 'G', 'C', 'N'])
    assert all(isinstance(v, np.ndarray) for v in encoder.base_vectors.values())
    
    # Check vector shapes
    assert all(v.shape == (hd_config.dimension,) for v in encoder.base_vectors.values())

@pytest.mark.hd
def test_sequence_encoding(hd_computer, sample_sequences):
    """Test sequence encoding functionality."""
    for sequence in sample_sequences:
        tensor_encoding, fractal_encoding = hd_computer.encode_sequence(sequence)
        
        # Check tensor encoding
        assert isinstance(tensor_encoding, torch.Tensor)
        assert len(tensor_encoding.shape) == hd_computer.config.tensor_order
        
        # Check fractal encoding
        assert isinstance(fractal_encoding, np.ndarray)
        assert fractal_encoding.shape[0] == hd_computer.config.dimension * hd_computer.config.fractal_levels

@pytest.mark.hd
def test_tensor_encoding(hd_config):
    """Test tensor-based encoding."""
    encoder = TensorEncoder(hd_config)
    sequence = "ATGC"
    
    # Encode sequence
    encoding = encoder.encode(sequence)
    
    # Check properties
    assert isinstance(encoding, torch.Tensor)
    assert len(encoding.shape) == hd_config.tensor_order
    assert torch.allclose(torch.norm(encoding), torch.tensor(1.0), atol=1e-6)

@pytest.mark.hd
def test_fractal_encoding(hd_config):
    """Test fractal encoding."""
    encoder = FractalEncoder(hd_config)
    sequence = "ATGC"
    
    # Encode sequence
    encoding = encoder.encode(sequence)
    
    # Check properties
    assert isinstance(encoding, np.ndarray)
    assert np.allclose(np.linalg.norm(encoding), 1.0, atol=1e-6)

@pytest.mark.hd
def test_similarity_computation(hd_computer):
    """Test sequence similarity computation."""
    seq1 = "ATGC"
    seq2 = "ATGG"  # One base different
    
    tensor_sim, fractal_sim = hd_computer.compute_similarity(seq1, seq2)
    
    # Check similarity properties
    assert isinstance(tensor_sim, float)
    assert isinstance(fractal_sim, float)
    assert -1 <= tensor_sim <= 1
    assert -1 <= fractal_sim <= 1
    
    # Self-similarity should be maximum
    self_tensor_sim, self_fractal_sim = hd_computer.compute_similarity(seq1, seq1)
    assert np.isclose(self_tensor_sim, 1.0, atol=1e-6)
    assert np.isclose(self_fractal_sim, 1.0, atol=1e-6)

@pytest.mark.hd
def test_caching(hd_computer):
    """Test caching functionality."""
    sequence = "ATGC"
    
    # First encoding
    encoding1 = hd_computer.encode_sequence(sequence, use_cache=True)
    
    # Second encoding (should use cache)
    encoding2 = hd_computer.encode_sequence(sequence, use_cache=True)
    
    # Check if both encodings are identical
    assert isinstance(encoding1, tuple)
    assert isinstance(encoding2, tuple)
    assert torch.allclose(encoding1[0], encoding2[0])
    assert np.allclose(encoding1[1], encoding2[1])

@pytest.mark.hd
def test_cache_size_limit(hd_config):
    """Test cache size limiting."""
    config = HDConfig(cache_size=2, **{k: v for k, v in vars(hd_config).items() if k != 'cache_size'})
    computer = HDComputing(config)
    
    # Add three sequences to cache (should only keep last two)
    sequences = ["ATGC", "GCTA", "TACG"]
    for seq in sequences:
        computer.encode_sequence(seq, use_cache=True)
    
    assert len(computer.cache) == 2
    assert "ATGC" not in computer.cache
    assert "GCTA" in computer.cache
    assert "TACG" in computer.cache

@pytest.mark.hd
@pytest.mark.parametrize("sequence", [
    "ATGCX",  # Invalid base
    "",       # Empty sequence
    "N" * 10  # All unknown bases
])
def test_edge_cases(hd_computer, sequence):
    """Test edge cases and error handling."""
    # Should not raise exceptions
    encoding = hd_computer.encode_sequence(sequence)
    assert isinstance(encoding, tuple)
    assert len(encoding) == 2

@pytest.mark.hd
def test_state_saving_loading(hd_computer, tmp_path):
    """Test saving and loading encoder states."""
    # Generate some encodings
    sequence = "ATGC"
    original_encoding = hd_computer.encode_sequence(sequence)
    
    # Save state
    save_path = tmp_path / "hd_state.pt"
    hd_computer.save_state(save_path)
    
    # Create new computer and load state
    new_computer = HDComputing(hd_computer.config)
    new_computer.load_state(save_path)
    
    # Check if encodings match
    new_encoding = new_computer.encode_sequence(sequence)
    assert torch.allclose(original_encoding[0], new_encoding[0])
    assert np.allclose(original_encoding[1], new_encoding[1])

@pytest.mark.hd
def test_parallel_processing(hd_computer):
    """Test parallel processing of sequences."""
    sequences = ["ATGC"] * 10  # Multiple identical sequences for testing
    
    # Process sequences
    encodings = []
    for seq in sequences:
        encoding = hd_computer.encode_sequence(seq)
        encodings.append(encoding)
    
    # Check consistency
    for enc in encodings[1:]:
        assert torch.allclose(encodings[0][0], enc[0])
        assert np.allclose(encodings[0][1], enc[1])
