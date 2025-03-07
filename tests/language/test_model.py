"""
Tests for the language modeling module.
"""

import numpy as np
import torch
import pytest
from qtlh.language import (
    LanguageConfig,
    GenomicTransformer,
    GenomicTokenizer,
    MotifAttentionLayer,
    SequenceDataset
)

@pytest.fixture
def batch_size():
    """Batch size for testing."""
    return 4

@pytest.fixture
def sequence_length():
    """Sequence length for testing."""
    return 128

@pytest.fixture
def mock_batch(batch_size, sequence_length):
    """Create mock batch for testing."""
    return {
        'input_ids': torch.randint(0, 1000, (batch_size, sequence_length)),
        'attention_mask': torch.ones(batch_size, sequence_length),
        'regulatory_labels': torch.randint(0, 2, (batch_size, sequence_length)).float(),
        'element_labels': torch.randint(0, 4, (batch_size, sequence_length))
    }

@pytest.mark.language
def test_language_config():
    """Test language model configuration."""
    config = LanguageConfig(
        vocab_size=1024,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4
    )
    assert config.vocab_size == 1024
    assert config.hidden_size == 128
    assert config.num_hidden_layers == 2
    assert config.num_attention_heads == 4

@pytest.mark.language
def test_tokenizer():
    """Test genomic sequence tokenizer."""
    tokenizer = GenomicTokenizer(kmer_size=3)
    
    # Test vocabulary creation
    assert len(tokenizer.vocab) == 64  # 4^3 for 3-mers
    
    # Test encoding
    sequence = "ATGCATGC"
    tokens = tokenizer.encode(sequence)
    assert isinstance(tokens, torch.Tensor)
    assert tokens.ndim == 1
    assert len(tokens) == len(sequence) - tokenizer.kmer_size + 1
    
    # Test decoding
    kmers = tokenizer.decode(tokens)
    assert isinstance(kmers, list)
    assert all(len(kmer) == tokenizer.kmer_size for kmer in kmers)

@pytest.mark.language
def test_motif_attention(language_config):
    """Test motif attention layer."""
    attention = MotifAttentionLayer(language_config)
    batch_size = 4
    seq_length = 10
    hidden_states = torch.randn(batch_size, seq_length, language_config.hidden_size)
    
    output = attention(hidden_states)
    assert isinstance(output, torch.Tensor)
    assert output.shape == hidden_states.shape

@pytest.mark.language
def test_transformer_initialization(language_config):
    """Test transformer model initialization."""
    model = GenomicTransformer(language_config)
    
    assert isinstance(model.transformer, torch.nn.Module)
    assert isinstance(model.motif_attention, MotifAttentionLayer)
    assert isinstance(model.regulatory_head, torch.nn.Linear)
    assert isinstance(model.element_classifier, torch.nn.Linear)

@pytest.mark.language
def test_transformer_forward(language_config, mock_batch):
    """Test transformer forward pass."""
    model = GenomicTransformer(language_config)
    outputs = model(
        input_ids=mock_batch['input_ids'],
        attention_mask=mock_batch['attention_mask']
    )
    
    # Check outputs
    assert 'sequence_features' in outputs
    assert 'pooled_features' in outputs
    assert 'motif_features' in outputs
    assert 'regulatory_scores' in outputs
    assert 'element_logits' in outputs
    assert 'all_hidden_states' in outputs

@pytest.mark.language
def test_transformer_training_step(language_config, mock_batch):
    """Test transformer training step."""
    model = GenomicTransformer(language_config)
    loss = model.training_step(mock_batch, batch_idx=0)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert loss.dim() == 0  # scalar

@pytest.mark.language
def test_sequence_dataset(sample_sequences):
    """Test sequence dataset."""
    labels = np.random.randint(0, 2, size=len(sample_sequences))
    dataset = SequenceDataset(sample_sequences, labels)
    
    assert len(dataset) == len(sample_sequences)
    
    item = dataset[0]
    assert 'input_ids' in item
    assert 'attention_mask' in item
    assert 'labels' in item

@pytest.mark.language
def test_dataloader_creation(sample_sequences):
    """Test dataloader creation."""
    from qtlh.language import create_dataloader
    
    dataloader = create_dataloader(
        sequences=sample_sequences,
        batch_size=2,
        shuffle=True
    )
    
    batch = next(iter(dataloader))
    assert isinstance(batch, dict)
    assert 'input_ids' in batch
    assert 'attention_mask' in batch

@pytest.mark.language
@pytest.mark.gpu
def test_gpu_support(language_config, mock_batch):
    """Test GPU support if available."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
        
    model = GenomicTransformer(language_config).cuda()
    mock_batch = {k: v.cuda() for k, v in mock_batch.items()}
    
    outputs = model(
        input_ids=mock_batch['input_ids'],
        attention_mask=mock_batch['attention_mask']
    )
    
    assert all(tensor.is_cuda for tensor in outputs.values() if isinstance(tensor, torch.Tensor))

@pytest.mark.language
def test_motif_detection(language_config):
    """Test motif detection capabilities."""
    model = GenomicTransformer(language_config)
    
    # Create sequence with a known motif pattern
    sequence = "ATGC" * 10  # Repeated pattern
    tokenizer = GenomicTokenizer()
    tokens = tokenizer.encode(sequence).unsqueeze(0)
    attention_mask = torch.ones_like(tokens)
    
    outputs = model(tokens, attention_mask)
    motif_features = outputs['motif_features']
    
    # Check if motif features show periodicity
    feature_correlation = torch.corrcoef(motif_features[0].T)
    assert torch.any(feature_correlation > 0.5)  # Should detect repeated pattern

@pytest.mark.language
def test_attention_visualization(language_config, mock_batch):
    """Test attention pattern visualization."""
    model = GenomicTransformer(language_config)
    outputs = model(
        input_ids=mock_batch['input_ids'],
        attention_mask=mock_batch['attention_mask']
    )
    
    # Get attention weights from the last layer
    attention = model.transformer.encoder.layer[-1].attention.self.get_attention_map()
    
    assert isinstance(attention, torch.Tensor)
    assert attention.dim() == 4  # (batch, heads, seq_len, seq_len)
    assert attention.shape[1] == language_config.num_attention_heads

@pytest.mark.language
def test_model_saving_loading(language_config, tmp_path):
    """Test model saving and loading."""
    model = GenomicTransformer(language_config)
    
    # Save model
    save_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), save_path)
    
    # Load model
    new_model = GenomicTransformer(language_config)
    new_model.load_state_dict(torch.load(save_path))
    
    # Check if parameters match
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)

@pytest.mark.language
def test_long_sequence_handling(language_config):
    """Test handling of long sequences."""
    model = GenomicTransformer(language_config)
    long_sequence = "ATGC" * 1000  # Very long sequence
    tokenizer = GenomicTokenizer()
    
    # Should handle sequence by chunking
    tokens = tokenizer.encode(long_sequence)
    chunks = torch.split(tokens, language_config.max_position_embeddings)
    
    for chunk in chunks:
        chunk = chunk.unsqueeze(0)
        attention_mask = torch.ones_like(chunk)
        outputs = model(chunk, attention_mask)
        assert all(isinstance(v, (torch.Tensor, tuple)) for v in outputs.values())

@pytest.mark.language
def test_optimizer_configuration(language_config):
    """Test optimizer configuration."""
    model = GenomicTransformer(language_config)
    optimizer_dict = model.configure_optimizers()
    
    assert 'optimizer' in optimizer_dict
    assert 'lr_scheduler' in optimizer_dict
    assert isinstance(optimizer_dict['optimizer'], torch.optim.Optimizer)
