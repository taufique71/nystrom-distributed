# Data

This directory holds dataset preparation scripts for generating input matrices
that the `nystrom` binary can consume. All generated outputs are binary,
column-major, double-precision (float64) — the format `nystrom` expects.

## CIFAR-10 Gram Matrix

`CIFAR10binary.py` downloads CIFAR-10 (via torchvision) and computes a Gram
matrix using either a linear or RBF kernel, then writes it to disk.

### Usage

```bash
# Linear kernel: cifar10-linear.bin
python CIFAR10binary.py -kernel linear

# RBF kernel with sigma=1 (default): cifar10-rbf-sig1.bin
python CIFAR10binary.py -kernel rbf -sigma 1
```

### Arguments

| Flag | Default | Description |
|---|---|---|
| `-path` | `./data` | Output directory |
| `-kernel` | `rbf` | `linear` or `rbf` |
| `-sigma` | `1` | RBF bandwidth (ignored for linear) |

The output matrix is `n × n` where `n = 50000` (CIFAR-10 training set size). At
float64, the file is ~20 GB.

### Output naming

- Linear: `cifar10-linear.bin`
- RBF: `cifar10-rbf-sig<sigma>.bin`

### Dependencies

- `numpy`
- `torchvision` (for CIFAR-10 download)
