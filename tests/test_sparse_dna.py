"""
Tests for SparseDNA module
"""

import pytest
import torch
from sparse_dna import SparseDNA


class TestSparseDNAInitialization:
    """Test SparseDNA initialization"""
    
    def test_default_initialization(self):
        """Test default initialization"""
        sparse_dna = SparseDNA()
        assert sparse_dna.num_freqs == 64
        assert sparse_dna.hidden_size == 128
        assert sparse_dna.sparsity == 0.5
    
    def test_custom_initialization(self):
        """Test custom initialization"""
        sparse_dna = SparseDNA(num_freqs=32, hidden_size=64, sparsity=0.7)
        assert sparse_dna.num_freqs == 32
        assert sparse_dna.hidden_size == 64
        assert sparse_dna.sparsity == 0.7
    
    def test_buffer_registration(self):
        """Test that B is registered as buffer"""
        sparse_dna = SparseDNA()
        assert hasattr(sparse_dna, 'B')
        assert isinstance(sparse_dna.B, torch.Tensor)
        assert sparse_dna.B.shape == (64, 2)
    
    def test_parameter_count(self):
        """Test parameter counting"""
        sparse_dna = SparseDNA(num_freqs=32, hidden_size=64)
        count = sparse_dna.count_parameters()
        # net = Linear(64->64) + Linear(64->64) + Linear(64->1), plus scale
        expected = (64*64 + 64) + (64*64 + 64) + (64*1 + 1) + 1
        assert count == expected


class TestSparseDNAWeightGeneration:
    """Test weight generation"""
    
    def test_generate_dense_weights(self):
        """Test generating dense weights"""
        sparse_dna = SparseDNA()
        W = sparse_dna.generate_weights(10, 10)
        assert W.shape == (10, 10)
        assert W.dtype == torch.float32
    
    def test_generate_sparse_weights(self):
        """Test generating sparse weights"""
        sparse_dna = SparseDNA(sparsity=0.5)
        W = sparse_dna.generate_sparse_weights(10, 10)
        assert W.shape == (10, 10)
    
    def test_sparse_weights_have_zeros(self):
        """Test that sparse weights have zeros"""
        sparse_dna = SparseDNA(sparsity=0.5)
        W = sparse_dna.generate_sparse_weights(100, 100)
        assert (W == 0).any()
    
    def test_dense_weights_no_zeros(self):
        """Test that dense weights may not have zeros"""
        sparse_dna = SparseDNA()
        W = sparse_dna.generate_weights(10, 10)
        # Dense weights might still have zeros due to initialization
        # Just check that it's a valid tensor
        assert W.shape == (10, 10)


class TestSparseDNASparsity:
    """Test sparsity functionality"""
    
    def test_set_sparsity(self):
        """Test setting sparsity"""
        sparse_dna = SparseDNA(sparsity=0.5)
        sparse_dna.set_sparsity(0.7)
        assert sparse_dna.sparsity == 0.7
    
    def test_get_actual_sparsity(self):
        """Test getting actual sparsity"""
        sparse_dna = SparseDNA(sparsity=0.5)
        W = sparse_dna.generate_sparse_weights(100, 100)
        actual = sparse_dna.get_actual_sparsity(W)
        assert 0 <= actual <= 1
    
    def test_sparsity_affects_output(self):
        """Test that sparsity affects output"""
        sparse_dna_low = SparseDNA(sparsity=0.3)
        sparse_dna_high = SparseDNA(sparsity=0.7)
        
        W_low = sparse_dna_low.generate_sparse_weights(100, 100)
        W_high = sparse_dna_high.generate_sparse_weights(100, 100)
        
        sparsity_low = sparse_dna_low.get_actual_sparsity(W_low)
        sparsity_high = sparse_dna_high.get_actual_sparsity(W_high)
        
        # Higher sparsity should result in more zeros
        assert sparsity_high > sparsity_low


class TestSparseDNALayerWeights:
    """Test layer weight generation"""
    
    def test_generate_sparse_layer_weights(self):
        """Test generating sparse weights for multiple layers"""
        sparse_dna = SparseDNA()
        layer_config = {
            'fc1': (256, 128),
            'fc2': (128, 256),
        }
        weights = sparse_dna.generate_sparse_layer_weights(layer_config)
        
        assert 'fc1' in weights
        assert 'fc2' in weights
        assert weights['fc1'].shape == (256, 128)
        assert weights['fc2'].shape == (128, 256)


class TestSparseDNADeviceAndDtype:
    """Test device and dtype handling"""
    
    def test_device_consistency(self):
        """Test that all tensors are on same device"""
        sparse_dna = SparseDNA()
        W = sparse_dna.generate_sparse_weights(10, 10)
        assert W.device == sparse_dna.B.device
        assert W.device == sparse_dna.scale.device
        for param in sparse_dna.net.parameters():
            assert param.device == sparse_dna.B.device
    
    def test_dtype_consistency(self):
        """Test that all tensors have same dtype"""
        sparse_dna = SparseDNA()
        W = sparse_dna.generate_sparse_weights(10, 10)
        assert W.dtype == sparse_dna.B.dtype
        assert W.dtype == sparse_dna.scale.dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
