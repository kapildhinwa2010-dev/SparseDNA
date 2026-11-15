# SparseDNA

[![CI](https://github.com/kapildhinwa2010-dev/SparseDNA/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/SparseDNA/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/SparseDNA/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/SparseDNA/actions/workflows/codeql.yml)

Fourier-feature weight generator with a random sparsity mask (PyTorch). Research prototype.

## What it does

- `generate_weights(out_dim, in_dim)`: dense matrix from a small coordinate-MLP: Fourier features of an (i, j) grid, 3-layer MLP, one scalar per entry, learnable scale plus Xavier-style scaling.
- `generate_sparse_weights(...)`: same, then zeroes each entry independently with probability `sparsity` (Bernoulli mask).
- `generate_sparse_layer_weights(config)`: several named matrices in one call.
- `set_sparsity(p)` / `get_actual_sparsity(W)`: validated to [0, 1]; measured zero-fraction.

## Honest limitations and negative results

- The mask is random, not learned. Bernoulli masking, not magnitude pruning. No evidence it preserves task accuracy at high sparsity.
- No speed or memory win: output is a dense tensor with zeros. No sparse kernels, no CSR storage.
- An untrained DNA produces a smooth random field, not useful pretrained weights. It must be fitted to a target.
- Actual sparsity fluctuates around the target (binomial sampling); small matrices deviate noticeably.
- Compression ratios quoted in this project family are storage arithmetic of generator vs materialized matrix, not verified end-to-end compression with preserved quality.

## Install

```bash
pip install -r requirements.txt
```

## Use

```python
from sparse_dna import SparseDNA
dna = SparseDNA(num_freqs=64, hidden_size=128, sparsity=0.5)
W = dna.generate_sparse_weights(256, 128)
print(dna.get_actual_sparsity(W))
```

## Tests

```bash
python -m pytest tests/ -q
```

CI: Python 3.10-3.12. CodeQL security-extended on every push.

## Status

Research prototype. API may change. No warranty.

## License

MIT - see LICENSE.
