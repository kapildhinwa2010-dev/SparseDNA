"""
SparseDNA - Sparse matrix generation with DNA

This module provides functionality to generate sparse matrices using
DNA-based methods, enabling efficient storage and computation for
large-scale neural networks.

Features:
- Generate sparse weight matrices
- Configurable sparsity levels
- Memory-efficient storage
- GPU and CPU support
- Integration with PyTorch models
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple


class SparseDNA(nn.Module):
    """
    SparseDNA - Generate sparse matrices using DNA.
    
    This class generates sparse weight matrices by combining DNA-based
    weight generation with sparsity masks.
    
    Example:
        >>> sparse_dna = SparseDNA(num_freqs=64, hidden_size=128, sparsity=0.5)
        >>> # Generate a sparse weight matrix
        >>> W = sparse_dna.generate_sparse_weights(1000, 1000)
        >>> print(W.shape)  # torch.Size([1000, 1000])
    
    Args:
        num_freqs: Number of Fourier frequencies (default: 64)
        hidden_size: Size of hidden layer (default: 128)
        sparsity: Sparsity level (0.0 to 1.0, default: 0.5)
    """
    
    def __init__(
        self,
        num_freqs: int = 64,
        hidden_size: int = 128,
        sparsity: float = 0.5
    ):
        super().__init__()
        if not 0.0 <= sparsity <= 1.0:
            raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")
        self.num_freqs = num_freqs
        self.hidden_size = hidden_size
        self.sparsity = sparsity
        
        # Fourier projection matrix (fixed)
        self.register_buffer('B', torch.randn(num_freqs, 2) * 20.0)
        input_dim = num_freqs * 2
        
        # DNA network
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Learnable scale
        self.scale = nn.Parameter(torch.tensor(0.01))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with Xavier uniform"""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def _generate_fourier_features(self, coords: torch.Tensor) -> torch.Tensor:
        """Generate Fourier features from coordinates"""
        proj = 2 * np.pi * coords @ self.B.T
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
    
    def generate_weights(
        self,
        out_dim: int,
        in_dim: int,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Generate a dense weight matrix.
        
        Args:
            out_dim: Output dimension
            in_dim: Input dimension
            device: Device to use (optional)
            
        Returns:
            Weight matrix of shape (out_dim, in_dim)
        """
        if device is None:
            device = self.B.device
        
        # Generate coordinate grid
        i = torch.linspace(-1, 1, out_dim, device=device)
        j = torch.linspace(-1, 1, in_dim, device=device)
        grid_i, grid_j = torch.meshgrid(i, j, indexing='ij')
        coords = torch.stack([grid_i.flatten(), grid_j.flatten()], dim=1)
        
        # Generate Fourier features
        feats = self._generate_fourier_features(coords)
        
        # Generate weights
        raw_W = self.net(feats).view(out_dim, in_dim)
        
        # Apply scaling
        xavier_scale = (2.0 / (in_dim + out_dim)) ** 0.5
        W = raw_W * self.scale * xavier_scale
        
        return W
    
    def generate_sparse_weights(
        self,
        out_dim: int,
        in_dim: int,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Generate a sparse weight matrix.
        
        Args:
            out_dim: Output dimension
            in_dim: Input dimension
            device: Device to use (optional)
            
        Returns:
            Sparse weight matrix of shape (out_dim, in_dim)
        """
        # Generate dense weights
        W = self.generate_weights(out_dim, in_dim, device)
        
        # Apply sparsity mask
        mask = torch.rand_like(W) > self.sparsity
        sparse_W = W * mask
        
        return sparse_W
    
    def generate_sparse_layer_weights(
        self,
        layer_config: Dict[str, Tuple[int, int]]
    ) -> Dict[str, torch.Tensor]:
        """
        Generate sparse weights for multiple layers.
        
        Args:
            layer_config: Dictionary mapping layer names to (out_dim, in_dim)
            
        Returns:
            Dictionary mapping layer names to sparse weight matrices
        """
        weights = {}
        for name, (out_dim, in_dim) in layer_config.items():
            weights[name] = self.generate_sparse_weights(out_dim, in_dim)
        return weights
    
    def set_sparsity(self, sparsity: float):
        """Set the sparsity level (must be in [0, 1])."""
        if not 0.0 <= sparsity <= 1.0:
            raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")
        self.sparsity = sparsity
    
    def count_parameters(self) -> int:
        """Count parameters in DNA network"""
        return sum(p.numel() for p in self.parameters())
    
    def get_actual_sparsity(self, W: torch.Tensor) -> float:
        """Calculate actual sparsity of a matrix"""
        return (W == 0).float().mean().item()


# Example usage
if __name__ == "__main__":
    print("SparseDNA - Sparse Matrix Generation")
    print("=" * 60)
    
    # Create SparseDNA instance
    sparse_dna = SparseDNA(num_freqs=64, hidden_size=128, sparsity=0.5)
    print(f"SparseDNA created with {sparse_dna.count_parameters():,} parameters")
    print(f"Target sparsity: {sparse_dna.sparsity:.1%}")
    
    # Test weight generation
    print("\nGenerating sparse weight matrices...")
    
    # Small matrix
    W1 = sparse_dna.generate_sparse_weights(100, 100)
    actual_sparsity = sparse_dna.get_actual_sparsity(W1)
    print(f"Generated 100x100 sparse matrix: {W1.shape}")
    print(f"  Actual sparsity: {actual_sparsity:.1%}")
    print(f"  Non-zero elements: {(W1 != 0).sum()}")
    
    # Medium matrix
    W2 = sparse_dna.generate_sparse_weights(200, 200)
    actual_sparsity = sparse_dna.get_actual_sparsity(W2)
    print(f"Generated 200x200 sparse matrix: {W2.shape}")
    print(f"  Actual sparsity: {actual_sparsity:.1%}")
    print(f"  Non-zero elements: {(W2 != 0).sum()}")
    
    print("\n✓ SparseDNA test completed successfully!")
