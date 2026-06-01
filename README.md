# Parallel Distributed Nyström Approximation

This code provides parallel implementations of the Nyström low-rank approximation for large symmetric matrices on HPC systems, offering multiple algorithm variants that explore different distribution and communication strategies.

Given an n×n **symmetric** matrix A and a target rank r, compute factors Y = A·Ω and Z = Ωᵀ·Y using a single random sketch matrix Ω of shape n×r. The corresponding low-rank approximation is A ≈ Y · pinv(Z) · Yᵀ. Only the symmetric variant is offered here; generalized Nyström for non-symmetric matrices uses different sketches (Ω_left ≠ Ω_right) and is not supported.

Note: although the symmetry of A is what makes this approximation valid, the current implementation does not exploit symmetry — A is stored and multiplied as a full n×n dense matrix.

For algorithmic details, analysis, and benchmark results, see the paper: [arXiv:2603.20966](https://arxiv.org/abs/2603.20966).

## Algorithm variants

**`nystrom-1d-noredist-1d`**
Process grid is `(nprocs, 1, 1)` for both matmuls. A is distributed by rows across all ranks. The first matmul `Y = A·Ω` is local (no communication on A); the second matmul `Z = Ωᵀ·Y` is followed by a single reduce-scatter on Z. No redistribution of Y between the two matmuls.

**`nystrom-1d-redist-1d`**
Process grid is `(nprocs, 1, 1)` for the first matmul and `(1, 1, nprocs)` for the second. A is row-distributed for `Y = A·Ω`. Y is then redistributed via an all-to-all between the two matmuls so the second matmul `Z = Ωᵀ·Y` can run locally on the new grid orientation. No reduce-scatter at the end of the second matmul (the redistribution moves the cost up front instead).

Note: a third variant, `nystrom-2d-noredist-1d` (2D process grid for the first matmul), was benchmarked in the paper but offers little practical benefit over the 1D variants, so it is not exposed through the public `nystrom` binary. Interested readers can see its usage in `paper-experiments/c_matmul/nystrom.cpp`.

## Implementations

Two independent implementations are provided:

- **C++** (`cpp/`): with both CPU and GPU implementations of the algorithm.
- **Python** (`python/`): implementation in a productivity-oriented language, using `mpi4py`. CPU only — no GPU support.

## Repository structure

```
nystrom-distributed/
├── cpp/                   # C++ implementation
├── python/                # Python implementation
├── data/                  # Dataset preparation scripts (e.g., CIFAR-10 Gram matrix)
├── paper-experiments/     # Research codebase that produced the paper benchmarks
└── README.md              # This file
```

## Build instructions

### Python

The Python implementation is import-and-run; the only setup is installing two dependencies. The experiments in the paper were run on NERSC Perlmutter following this sequence:

```bash
module load python
pip install --user randomgen
```

`mpi4py` and `numpy` are already provided by the `python` module. `randomgen` (the third-party PRNG used for the sketch matrix Ω in the CPU path) is installed once into your user site-packages.

For other systems, ensure you have a working `mpi4py` linked against your system MPI, plus `numpy` and `randomgen`.

### C++

The C++ implementation builds with CMake (≥ 3.10) and requires MPI, OpenMP, and Intel MKL for the CPU path. The GPU path additionally requires CUDA and a GPU with cuBLAS/cuRAND support.

MKL is used for both BLAS (DGEMM in the matmul steps) and random number generation (MKL-VSL for the sketch matrix Ω) on the CPU path. cuBLAS and cuRAND play the same two roles on the GPU path.

The experiments in the paper were run on NERSC Perlmutter following these sequences:

**CPU build:**

```bash
cd cpp
module swap PrgEnv-gnu PrgEnv-intel
module load python
mkdir -p build_cpu
cd build_cpu
cmake ..
make
```

Produces `build_cpu/nystrom` and `build_cpu/matmul`.

**GPU build:**

```bash
cd cpp
module swap PrgEnv-gnu PrgEnv-nvidia
module load cudatoolkit
module load craype-accel-nvidia80
mkdir -p build_gpu
cd build_gpu
cmake ..
make
```

Produces `build_gpu/nystrom` and `build_gpu/matmul`.

For other systems, ensure the required toolchain is available and adapt the module commands accordingly. CMake auto-enables CUDA when `PE_ENV=NVIDIA`; on other systems pass `-DUSE_CUDA=ON` to `cmake` explicitly.

## Usage

Both implementations share the same CLI. For full help:

```bash
./cpp/build_cpu/nystrom --help    # or build_gpu/nystrom for the GPU build
python python/nystrom.py --help
```

Example Slurm scripts targeting NERSC Perlmutter are provided at:

- `cpp/scripts/nystrom-run.sh`
- `python/scripts/nystrom-run.sh`

Edit the *User parameters* block at the bottom of either script and submit with `sbatch`.

All input/output matrices are binary, column-major, double precision. See `data/README.md` for how to generate the CIFAR-10 Gram matrix used in the paper benchmarks.

### Input and output files

The binary accepts three optional file-related flags:

- `-afile <path>` — path to the input matrix A. If omitted, A is generated synthetically (a deterministic test pattern, for benchmarking only).
- `-yfile <path>` — path to write the Y factor. If omitted, Y is computed but not written.
- `-zfile <path>` — path to write the Z factor. If omitted, Z is computed but not written.

All three files use the same format: a flat binary stream of `double` values in column-major (Fortran) order — the natural format for parallel MPI-IO. A is n×n, Y is n×r, Z is r×r. **The implementation supports double precision (float64) only**; matrices in any other precision must be converted before being passed in.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{daas2026communication,
  title={Communication Lower Bounds and Algorithms for Sketching with Random Dense Matrices},
  author={Daas, Hussam Al and Ballard, Grey and Grigori, Laura and Hussain, Md Taufique and Kumar, Suraj and Rahman, Mohammad Marufur and Rouse, Kathryn},
  booktitle={Proceedings of the 38th ACM Symposium on Parallelism in Algorithms and Architectures (SPAA '26)},
  year={2026}
}
```

Preprint: [arXiv:2603.20966](https://arxiv.org/abs/2603.20966)
