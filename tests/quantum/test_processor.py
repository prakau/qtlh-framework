"""
Tests for the quantum processing module.
"""

import numpy as np
import pytest
from qtlh.quantum import QuantumProcessor, QuantumConfig
from qtlh.quantum.processor import ErrorMitigator, QuantumCircuit

@pytest.fixture
def quantum_processor():
    """Create a quantum processor instance for testing."""
    config = QuantumConfig(n_qubits=4, n_layers=2)
    return QuantumProcessor(config)

@pytest.fixture
def sample_sequence():
    """Create a sample DNA sequence for testing."""
    return "ATGCTAGCTAGCTGATCGATCG"

@pytest.mark.quantum
def test_quantum_processor_initialization(quantum_processor):
    """Test quantum processor initialization."""
    assert isinstance(quantum_processor, QuantumProcessor)
    assert quantum_processor.config.n_qubits == 4
    assert quantum_processor.config.n_layers == 2
    assert isinstance(quantum_processor.circuit, QuantumCircuit)
    assert isinstance(quantum_processor.weights, np.ndarray)

@pytest.mark.quantum
def test_encode_base(quantum_processor):
    """Test base encoding functionality."""
    # Test each nucleotide
    bases = ['A', 'T', 'G', 'C', 'N']
    for base in bases:
        encoding = quantum_processor._encode_base(base)
        assert isinstance(encoding, np.ndarray)
        assert encoding.shape == (2,)
        assert np.all((encoding >= 0) & (encoding <= 1))

@pytest.mark.quantum
def test_process_sequence(quantum_processor, sample_sequence):
    """Test sequence processing."""
    features = quantum_processor.process_sequence(
        sequence=sample_sequence,
        window_size=10,
        stride=5
    )
    
    # Check output shape and properties
    assert isinstance(features, np.ndarray)
    n_windows = (len(sample_sequence) - 10) // 5 + 1
    assert features.shape[0] == n_windows
    assert features.shape[1] == quantum_processor.config.n_qubits

@pytest.mark.quantum
def test_process_window(quantum_processor):
    """Test window processing."""
    window = "ATGC"
    features = quantum_processor._process_window(window)
    
    # Check output properties
    assert isinstance(features, np.ndarray)
    assert features.shape == (quantum_processor.config.n_qubits,)
    assert np.all(np.abs(features) <= 1)  # Features should be normalized

@pytest.mark.quantum
def test_error_mitigation(quantum_processor):
    """Test error mitigation functionality."""
    # Create test data
    test_measurements = np.random.randn(quantum_processor.config.n_qubits)
    
    # Apply error mitigation
    mitigated = quantum_processor.circuit.error_mitigator.mitigate(test_measurements)
    
    # Check output
    assert isinstance(mitigated, np.ndarray)
    assert mitigated.shape == test_measurements.shape

@pytest.mark.quantum
@pytest.mark.parametrize("window_size,stride", [
    (10, 5),
    (20, 10),
    (50, 25)
])
def test_different_window_sizes(quantum_processor, sample_sequence, window_size, stride):
    """Test processing with different window sizes and strides."""
    features = quantum_processor.process_sequence(
        sequence=sample_sequence,
        window_size=window_size,
        stride=stride
    )
    
    # Check output shape
    n_windows = (len(sample_sequence) - window_size) // stride + 1
    assert features.shape[0] == n_windows
    assert features.shape[1] == quantum_processor.config.n_qubits

@pytest.mark.quantum
def test_circuit_optimization(quantum_processor):
    """Test quantum circuit optimization."""
    # Create sample training data
    sequences = [
        "ATGC" * 5,
        "GCTA" * 5,
        "TGCA" * 5
    ]
    
    # Test optimization step
    quantum_processor.optimize_circuit(
        train_sequences=sequences,
        n_epochs=2
    )
    
    # Check if weights were updated
    assert isinstance(quantum_processor.weights, np.ndarray)
    assert not np.allclose(quantum_processor.weights, np.zeros_like(quantum_processor.weights))

@pytest.mark.quantum
def test_parameter_saving_loading(quantum_processor, tmp_path):
    """Test saving and loading of quantum circuit parameters."""
    # Save parameters
    save_path = tmp_path / "quantum_params.npy"
    quantum_processor.save_parameters(save_path)
    
    # Load parameters in a new processor
    new_processor = QuantumProcessor(quantum_processor.config)
    new_processor.load_parameters(save_path)
    
    # Check if parameters match
    assert np.allclose(quantum_processor.weights, new_processor.weights)

@pytest.mark.quantum
def test_invalid_sequence_handling(quantum_processor):
    """Test handling of invalid sequences."""
    invalid_sequence = "ATGCX"  # X is not a valid nucleotide
    
    with pytest.warns(UserWarning):
        features = quantum_processor.process_sequence(invalid_sequence)
        
    assert isinstance(features, np.ndarray)

@pytest.mark.quantum
@pytest.mark.gpu
def test_gpu_acceleration():
    """Test GPU acceleration if available."""
    pytest.importorskip("torch.cuda")
    import torch
    
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
        
    config = QuantumConfig(n_qubits=4, n_layers=2)
    processor = QuantumProcessor(config)
    
    sequence = "ATGC" * 10
    features = processor.process_sequence(sequence)
    
    assert isinstance(features, np.ndarray)

@pytest.mark.quantum
def test_error_mitigator_calibration():
    """Test error mitigator calibration."""
    config = QuantumConfig(n_qubits=4)
    mitigator = ErrorMitigator(config)
    
    # Test calibration
    mitigator.calibrate()
    
    # Check if calibration matrix was updated
    assert isinstance(mitigator.calibration_matrix, np.ndarray)
    assert mitigator.calibration_matrix.shape == (16, 16)  # 2^n_qubits

@pytest.mark.quantum
def test_quantum_circuit_execution(quantum_processor):
    """Test quantum circuit execution."""
    # Create input features
    features = np.random.rand(quantum_processor.config.n_qubits)
    
    # Execute circuit
    outputs = quantum_processor.circuit._circuit(features, quantum_processor.weights)
    
    # Check outputs
    assert isinstance(outputs, list)
    assert len(outputs) == quantum_processor.config.n_qubits
    assert all(isinstance(x, float) for x in outputs)
    assert all(-1 <= x <= 1 for x in outputs)  # Expectation values should be in [-1, 1]
