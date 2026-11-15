# SparseDNA

[![CI](https://github.com/kapildhinwa2010-dev/SparseDNA/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/SparseDNA/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/SparseDNA/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/SparseDNA/actions/workflows/codeql.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview

**SparseDNA** is a sparse matrix generation framework that combines DNA-based weight generation with sparsity masks, enabling efficient storage and computation for large-scale neural networks.

### Key Features

- ✅ **Sparse Matrices**: Generate weight matrices with configurable sparsity
- ✅ **DNA-Based**: Uses Fourier features for weight generation
- ✅ **Memory Efficient**: Sparse storage reduces memory usage
- ✅ **GPU Accelerated**: Full CUDA support
- ✅ **PyTorch Integration**: Seamless integration with PyTorch models
- ✅ **Configurable**: Adjustable sparsity levels

## Installation

```bash
pip install torch numpy pytest
```

Or install from source:

```bash
git clone https://github.com/kapildhinwa2010-dev/SparseDNA.git
cd SparseDNA
pip install -e .
```

## Usage

### Basic Sparse Matrix Generation

```python
from sparse_dna import SparseDNA

# Create SparseDNA instance
sparse_dna = SparseDNA(num_freqs=64, hidden_size=128, sparsity=0.5)

# Generate a sparse weight matrix
W = sparse_dna.generate_sparse_weights(1000, 1000)
print(f"Generated sparse matrix: {W.shape}")
```

### Generate for Multiple Layers

```python
# Generate sparse weights for multiple layers
layer_config = {
    'fc1': (256, 128),
    'fc2': (128, 256),
}
weights = sparse_dna.generate_sparse_layer_weights(layer_config)

for name, W in weights.items():
    print(f"{name}: {W.shape}, sparsity: {sparse_dna.get_actual_sparsity(W):.1%}")
```

### Adjust Sparsity

```python
# Change sparsity level
sparse_dna.set_sparsity(0.7)  # 70% sparsity

# Generate new sparse matrix
W = sparse_dna.generate_sparse_weights(100, 100)
print(f"Actual sparsity: {sparse_dna.get_actual_sparsity(W):.1%}")
```

## Architecture

### Sparse Weight Generation

SparseDNA combines DNA-based weight generation with sparsity masks:

```
Coordinates → Fourier Features → DNA Network → Dense Weights → Sparsity Mask → Sparse Weights
```

### Sparsity Mechanism

The sparsity mask is applied element-wise to the generated weights:
- Random mask with probability `sparsity`
- Elements below threshold are set to zero
- Resulting matrix has approximately `sparsity` fraction of zeros

### Advantages

- **Memory Efficiency**: Sparse matrices use less memory
- **Computational Efficiency**: Sparse operations can be faster
- **Regularization**: Sparsity acts as a form of regularization
- **Interpretability**: Easier to interpret sparse connections

## Sparsity Levels

| Sparsity | Storage Savings | Typical Use Case |
|----------|-----------------|------------------|
| 0.3      | ~30%            | Fine-tuning      |
| 0.5      | ~50%            | General purpose  |
| 0.7      | ~70%            | Inference        |
| 0.9      | ~90%            | Extreme compression |

## API Reference

### SparseDNA

```python
SparseDNA(
    num_freqs: int = 64,
    hidden_size: int = 128,
    sparsity: float = 0.5
)
```

**Methods:**
- `generate_weights(out_dim: int, in_dim: int) -> Tensor`: Generate dense weights
- `generate_sparse_weights(out_dim: int, in_dim: int) -> Tensor`: Generate sparse weights
- `generate_sparse_layer_weights(layer_config: Dict) -> Dict`: Generate sparse weights for layers
- `set_sparsity(sparsity: float) -> None`: Set sparsity level
- `count_parameters() -> int`: Count DNA network parameters
- `get_actual_sparsity(W: Tensor) -> float`: Calculate actual sparsity of matrix

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds upon research in:
- Sparse neural networks
- Network pruning
- Fourier feature networks
- Efficient deep learning
