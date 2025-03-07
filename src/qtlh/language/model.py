"""
Genomic Language Model Module
===========================

This module implements transformer-based language modeling for genomic sequences,
providing advanced semantic analysis and regulatory element prediction.
"""

import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import BertConfig, BertModel
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader

@dataclass
class LanguageConfig:
    """Configuration for genomic language modeling."""
    vocab_size: int = 4096  # Large vocab for k-mer tokenization
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 2048
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12

class MotifAttentionLayer(nn.Module):
    """Specialized attention layer for motif detection."""
    
    def __init__(self, config: LanguageConfig):
        super().__init__()
        self.config = config
        
        # Multi-head attention for motif detection
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Motif-specific CNN layers
        self.motif_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=7,
            padding=3,
            groups=config.num_attention_heads
        )
        
        self.output_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with motif-aware attention."""
        batch_size = hidden_states.size(0)
        
        # Regular self-attention
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, -1, self.config.num_attention_heads, 
                         self.config.hidden_size // self.config.num_attention_heads)
        key = key.view(batch_size, -1, self.config.num_attention_heads,
                      self.config.hidden_size // self.config.num_attention_heads)
        value = value.view(batch_size, -1, self.config.num_attention_heads,
                         self.config.hidden_size // self.config.num_attention_heads)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(
            self.config.hidden_size // self.config.num_attention_heads
        )
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, value)
        
        # Motif detection using CNN
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, self.config.hidden_size, -1)
        motif_features = self.motif_conv(context)
        motif_features = motif_features.view(
            batch_size, self.config.num_attention_heads, -1,
            self.config.hidden_size // self.config.num_attention_heads
        )
        
        # Combine attention and motif features
        output = motif_features.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.config.hidden_size)
        
        # Final processing
        output = self.output_layer(output)
        output = self.dropout(output)
        output = self.layer_norm(output + hidden_states)
        
        return output

class GenomicTokenizer:
    """Tokenizer for genomic sequences."""
    
    def __init__(self, kmer_size: int = 6):
        self.kmer_size = kmer_size
        self.vocab = self._initialize_vocab()
        
    def _initialize_vocab(self) -> Dict[str, int]:
        """Initialize k-mer vocabulary."""
        bases = ['A', 'T', 'G', 'C']
        kmers = [''.join(p) for p in self._generate_kmers(bases, self.kmer_size)]
        return {kmer: idx for idx, kmer in enumerate(kmers)}
    
    def _generate_kmers(self, bases: List[str], k: int) -> List[str]:
        """Generate all possible k-mers."""
        if k == 0:
            return ['']
        return [b + mer for b in bases for mer in self._generate_kmers(bases, k-1)]
    
    def encode(self, sequence: str) -> torch.Tensor:
        """Encode sequence to token IDs."""
        kmers = [
            sequence[i:i+self.kmer_size] 
            for i in range(len(sequence) - self.kmer_size + 1)
        ]
        return torch.tensor([
            self.vocab.get(kmer, len(self.vocab))  # Use last index for unknown
            for kmer in kmers
        ])
    
    def decode(self, token_ids: torch.Tensor) -> List[str]:
        """Decode token IDs back to k-mers."""
        id_to_kmer = {v: k for k, v in self.vocab.items()}
        return [id_to_kmer.get(int(id), 'N' * self.kmer_size) for id in token_ids]

class GenomicTransformer(pl.LightningModule):
    """Main transformer model for genomic sequences."""
    
    def __init__(self, config: Optional[LanguageConfig] = None):
        super().__init__()
        self.config = config or LanguageConfig()
        
        # Initialize BERT-based model
        bert_config = BertConfig(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            intermediate_size=self.config.intermediate_size,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
            max_position_embeddings=self.config.max_position_embeddings,
            layer_norm_eps=self.config.layer_norm_eps
        )
        
        self.transformer = BertModel(bert_config)
        self.motif_attention = MotifAttentionLayer(self.config)
        
        # Additional heads for specific tasks
        self.regulatory_head = nn.Linear(self.config.hidden_size, 1)
        self.element_classifier = nn.Linear(self.config.hidden_size, 4)  # Different regulatory elements
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        # Transform sequence through BERT
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        
        # Apply motif attention
        motif_features = self.motif_attention(sequence_output)
        
        # Generate predictions
        regulatory_scores = self.regulatory_head(motif_features)
        element_logits = self.element_classifier(motif_features)
        
        return {
            'sequence_features': sequence_output,
            'pooled_features': pooled_output,
            'motif_features': motif_features,
            'regulatory_scores': regulatory_scores,
            'element_logits': element_logits,
            'all_hidden_states': outputs.hidden_states
        }
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """Training step logic."""
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        
        # Calculate losses
        regulatory_loss = nn.BCEWithLogitsLoss()(
            outputs['regulatory_scores'].squeeze(-1),
            batch['regulatory_labels']
        )
        
        element_loss = nn.CrossEntropyLoss()(
            outputs['element_logits'].view(-1, 4),
            batch['element_labels'].view(-1)
        )
        
        total_loss = regulatory_loss + element_loss
        
        self.log('train_loss', total_loss)
        return total_loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'train_loss'
        }

class SequenceDataset(Dataset):
    """Dataset for genomic sequences."""
    
    def __init__(
        self,
        sequences: List[str],
        labels: Optional[np.ndarray] = None,
        tokenizer: Optional[GenomicTokenizer] = None
    ):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer or GenomicTokenizer()
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset."""
        sequence = self.sequences[idx]
        
        # Tokenize sequence
        tokens = self.tokenizer.encode(sequence)
        attention_mask = torch.ones_like(tokens)
        
        item = {
            'input_ids': tokens,
            'attention_mask': attention_mask
        }
        
        # Add labels if available
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
            
        return item

def create_dataloader(
    sequences: List[str],
    labels: Optional[np.ndarray] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """Create a DataLoader for genomic sequences."""
    dataset = SequenceDataset(sequences, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
