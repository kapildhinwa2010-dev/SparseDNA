# SparseDNA

SparseDNA is a small PyTorch prototype that generates weight matrices from a coordinate-based neural network and optionally applies a random zero mask. It returns ordinary dense `torch.Tensor` objects; the code does not provide sparse storage or sparse-kernel execution.

## Status

The repository contains an implementation, unit tests, and GitHub Actions workflow definitions. The tests were run locally on the checked-out commit and passed (14 tests). No model-accuracy study, performance benchmark, or compression evaluation is provided, so this project should be treated as an experimental matrix generator rather than a validated replacement for trained model weights.

## Implemented functionality

- `SparseDNA` is a `torch.nn.Module` with a fixed random Fourier-projection buffer, a three-linear-layer SiLU network, and a learnable output scale.
- `generate_weights(out_dim, in_dim, device=None)` generates a dense matrix with shape `(out_dim, in_dim)` from a normalized coordinate grid and applies a dimension-based scale.
- `generate_sparse_weights(...)` multiplies those values by an independently sampled Bernoulli mask. The `sparsity` argument is the probability of masking an entry to zero; the realized zero fraction can vary from the requested probability.
- `generate_sparse_layer_weights(config)` generates a dictionary of masked matrices from names mapped to `(out_dim, in_dim)` pairs.
- `set_sparsity(p)` changes the mask probability, `get_actual_sparsity(W)` measures the zero fraction of a tensor, and `count_parameters()` counts the module's learnable parameters.

## Requirements

Python 3.10 or newer, PyTorch 2.0 or newer, and NumPy 1.24 or newer are declared in the project metadata. The requirements file also includes pytest for running tests.

From the repository root, install the declared requirements:

```bash
python -m pip install -r requirements.txt
```

## Usage

Run the source-file demonstration from the repository root:

```bash
python sparse_dna.py
```

Or import the module directly while working from the repository root:

```python
from sparse_dna import SparseDNA

model = SparseDNA(num_freqs=64, hidden_size=128, sparsity=0.5)
weights = model.generate_sparse_weights(out_dim=256, in_dim=128)
print(weights.shape)
print("Measured zero fraction:", model.get_actual_sparsity(weights))
```

The returned tensor remains dense even where values have been set to zero. No console-script entry point is declared.

## Tests and CI

Run the test suite with:

```bash
python -m pytest -q
```

On the checked-out commit, this command completed successfully: **14 passed**. This is a local test result, not a claim about a remote GitHub Actions run. The CI workflow defines test jobs for Python 3.10, 3.11, and 3.12 and includes compile and import/generation smoke checks; the CodeQL workflow defines Python security analysis. Their configuration is present in the repository, but remote workflow results were not checked as part of this README review.

## Limitations

- The mask is random and unstructured; it is not learned or based on weight magnitude.
- Outputs use dense tensor storage. Zeroing entries alone does not reduce the returned tensor's storage or provide sparse-operation speedups.
- The repository does not include a training or fitting procedure, task-accuracy evaluation, or evidence that generated matrices preserve a model's behavior.
- No benchmark or end-to-end compression result is provided.
- Since masking is random, measured sparsity is not guaranteed to equal the configured probability, particularly for small matrices.

## License

The repository's `LICENSE` file contains the MIT License, copyright (c) 2026 Kapil Dhinwa. See [LICENSE](LICENSE) for the complete terms.
