# CLAUDE.md — Project Knowledge Base

This file documents the project for Claude to use across sessions.
It is built iteratively — sections marked [NEEDS VERIFICATION] have not yet
been confirmed with the author and should be treated as hypotheses.

---

## What This Project Is

A research codebase implementing parallel distributed Nyström low-rank
approximation of large symmetric matrices (e.g. Gram/kernel matrices).
Published in a paper (accepted). Being prepared for public release.

---

## Repository Structure

The repo root (`$HOME/Codes/nystrom-distributed/`) hosts two parallel trees:

- `paper-experiments/` — frozen snapshot of the code used for the paper, with
  the original messy layout (Python at top level, `c_matmul/` for C++, etc.).
  Receives occasional fixes but is otherwise stable. This `CLAUDE.md` lives
  inside `paper-experiments/` and describes only this tree.
- Cleaned-up public release at the repo root (`cpp/`, `python/`, `data/`)
  alongside `paper-experiments/`. Intended for end-user use. See the root
  `README.md` for orientation.

All file paths in this document are relative to `paper-experiments/` unless
stated otherwise.

**The algorithm:**
Given an n×n symmetric matrix A and target rank r, compute low-rank factors:
  Y = A · Ω       (first matrix multiplication)
  Z = Ωᵀ · Y      (second matrix multiplication)
where Ω is an n×r random sketch matrix. These factors can then be used to
form the Nyström approximation A ≈ Y · pinv(Z) · Yᵀ for error analysis,
but the primary output is Y and Z themselves.

The key research contribution is studying different parallel communication
strategies for computing Y and Z efficiently across many MPI processes on
HPC clusters.

---

## Two Independent Implementations

Both implementations are intended to be published and used independently —
neither is a reference for the other.

### Python (`*.py` at the top of `paper-experiments/`)
- Uses `mpi4py`
- Entry point: `nystrom.py` (run with `mpirun python nystrom.py ...`)
- Key files:
  - `communicator.py` — `ProcGrid` class: 3D process grid and sub-communicators
  - `matrix.py` — `ParMat` class (distributed matrix) + `matmul`, `matmul1_gen`,
    `matmul1_comm` functions
  - `nystrom.py` — all Nyström algorithm variants + main entry point
  - `utils.py` — MPI communication primitives, matrix I/O, distribution utilities
  - `CIFAR10bianry.py` — standalone data prep script (example dataset only)
- Tests: `tests/matmul-correctness-test.py`

### C++ (`c_matmul/`)
- Uses MPI + optional CUDA/cuBLAS
- Entry points: `nystrom` binary (`nystrom.cpp`) and `matmul` binary (`matmul.cpp`)
- Has more communication variants than Python
- Supports CPU-only (MKL) and GPU (cuBLAS/cuRAND) via `#ifdef USE_CUBLAS`
- Key files:
  - `procgrid.h` — `ProcGrid` struct: 3D process grid and sub-communicators
  - `matrix.h` — `ParMat` class + matmul functions + parallel binary I/O
  - `nystrom.h` — all Nyström algorithm variants (header-only)
  - `nystrom.cpp` — main entry point for nystrom binary
  - `matmul.cpp` — main entry point for matmul binary
  - `utils.h` — `findSplits` distribution utility
  - `prng.h` — Xoroshiro128+ CPU PRNG (used in tests; commented out in main algorithms)
  - `pack_unpack.cu` — CUDA kernels for data packing/unpacking during redistribution
  - `performance.cu` — [NEEDS VERIFICATION: purpose not fully understood]

---

## The 3D Process Grid

Both implementations organize MPI processes into a logical p1 × p2 × p3 grid.

**Rank assignment (C-order, fibRank changes fastest, rowRank slowest):**
- `rowRank = myrank / (nProcCol * nProcFib)`
- `colRank = (myrank / nProcFib) % nProcCol`
- `fibRank = myrank % nProcFib`

**Three sub-communicators** — note the counterintuitive but consistent naming:
- `rowWorld`: groups processes with same `rowRank` AND `fibRank`, varying `colRank`
  → **`rankInRowWorld = colRank`**
- `colWorld`: groups processes with same `colRank` AND `fibRank`, varying `rowRank`
  → **`rankInColWorld = rowRank`**
- `fibWorld`: groups processes with same `rowRank` AND `colRank`, varying `fibRank`
  → **`rankInFibWorld = fibRank`**

This means `rankInColWorld` equals `rowRank` and `rankInRowWorld` equals `colRank`
— not what the names suggest. This is important to keep in mind when reading
distribution logic in `matrix.py` and `matrix.h`.

Both Python and C++ implement this identically.

---

## Matrix Distribution: Faces A, B, C

Each matrix is distributed across the process grid according to a "face" (A, B, or C).
Using the confirmed knowledge that `rankInColWorld = rowRank` and `rankInRowWorld = colRank`:

| Face | Rows split by | Cols split by (stage 1) | Cols split by (stage 2) |
|------|--------------|------------------------|------------------------|
| A    | p1 (rowRank) | p2 (colRank)           | p3 (fibRank)           |
| B    | p2 (colRank) | p3 (fibRank)           | p1 (rowRank)           |
| C    | p1 (rowRank) | p3 (fibRank)           | p2 (colRank)           |

Both Python and C++ agree exactly on all three faces.

---

## Algorithm Variants

Each variant takes six process grid arguments:
`-matmul1p1 -matmul1p2 -matmul1p3 -matmul2p1 -matmul2p2 -matmul2p3`

Total MPI processes must equal both `p1×p2×p3` products.

| Variant | First matmul grid | Second matmul grid | Y redistributed? |
|---|---|---|---|
| `nystrom-1d-noredist-1d` | 1D (p×1×1) | same grid | no |
| `nystrom-1d-redist-1d` | 1D (p×1×1) | different 1D grid | yes |
| `nystrom-1d-redist-2d` | 1D (p×1×1) | 2D grid | yes |
| `nystrom-2d-redist-1d` | 2D grid | 1D (p×1×1) | yes |

The C++ implementation also has `nystrom-2d-noredist-1d` which is not in Python.

"Redistribution" means Y is explicitly communicated between the two matmuls
to change its layout from one process grid to another.

---

## Random Sketch Matrix Ω

Ω is generated redundantly on every process — no communication needed.

| Implementation | Library | Algorithm | Seed |
|---|---|---|---|
| Python | `randomgen` | Xoroshiro128+ | 1234 |
| C++ CPU | Intel MKL-VSL | Philox4×32×10 | `1234 + thread_id` |
| C++ GPU | cuRAND | Philox4×32×10 | per-rank |

The C++ CPU and GPU paths use the same PRNG variant (Philox4×32×10) intentionally,
so they produce the same Ω. Python uses Xoroshiro128+ and produces a different Ω.
This difference is **intentional and accepted** — both produce valid random sketch
matrices. Results are not bit-for-bit reproducible across Python and C++.

Note: Xoroshiro128+ code exists in `prng.h` and is commented out in `matrix.h`
and `nystrom.h` — it was the original C++ implementation before the switch to Philox.

---

## Data Format

- Always `float64` (double precision)
- Always square (n×n) — the algorithm targets symmetric matrices
- Column-major (Fortran order) binary layout
- Read/written using MPI-IO (`MPI_File_Open`, `MPI_Type_create_subarray`,
  `Read_all`/`Write_all`) with `MPI_ORDER_FORTRAN`

The C++ reader (`parallelReadBinary` in `matrix.h`) correctly uses `MPI_ORDER_FORTRAN`.
The Python reader (`getCIFAR10GramMatrix` in `utils.py`) has `MPI_ORDER_FORTRAN`
commented out — potential bug P4 in SESSION_LOG.md, to be verified with author.

`CIFAR10bianry.py` is an example data preparation script — downloads CIFAR-10,
computes a Gram matrix (linear or RBF kernel), and writes it in this format.

---

## Build System

CMake-based. Root `CMakeLists.txt` only calls `add_subdirectory(c_matmul)`.
All build logic is in `c_matmul/CMakeLists.txt`.

**CPU build** (default, `USE_CUDA=OFF`): requires MPI, OpenMP, Intel MKL.
**GPU build** (`USE_CUDA=ON`): requires MPI, OpenMP, CUDA, cuBLAS, cuRAND.
`USE_CUDA` can be set manually or auto-detected from `$PE_ENV=NVIDIA` (NERSC Perlmutter).

- Target GPU architecture: `nvidia80` (A100) — intentional for target systems
- BLAS: Intel MKL only — intentional for now

Known build issues to be fixed (see SESSION_LOG.md):
- `find_package(CUDA)` is deprecated
- `test-cublas` and `test-cuda-aware-mpi` always compiled regardless of `USE_CUDA`
- Minor: OpenMP status message prints wrong variable

---

## Scripts

All in `scripts/`. Mix of Slurm job scripts and analysis/parsing scripts.
Written for NERSC Perlmutter. Scripts and paper experiment scripts coexist —
no separation needed.

**Path assumptions (hardcoded, to be replaced with a variable later):**
- Repo root: `$HOME/Codes/nystrom-distributed/` (will host the public release too)
- Paper-experiments tree: `$REPO/paper-experiments/`
- Python entry point: `$REPO/paper-experiments/nystrom.py`
- C++ CPU binary: `$REPO/paper-experiments/build_cpu/c_matmul/nystrom` (or `matmul`)
- C++ GPU binary: `$REPO/paper-experiments/build_gpu/c_matmul/nystrom` (or `matmul`)
- Outputs: `$SCRATCH/nystrom/`

**Three system configurations in scripts:**
- `perlmutter-cpu`: CPU nodes, Intel env, 128 cores/node
- `perlmutter-gpu`: GPU nodes, NVIDIA env, 4 GPUs/node, CUDA-enabled
- `perlmutter-gpu-cpu`: GPU nodes but running CPU build (Intel env, no CUDA) — intentional

---

## SLATE matmul baseline

`slate-matmul/` is a third-party-library baseline that runs SLATE's
distributed DGEMM on the same matrix size and total MPI process count as
`c_matmul/matmul`, to show that our implementation is competitive with a
modern, expert-tuned dense linear algebra library. CPU-only,
`perlmutter-gpu-cpu`. See `slate-matmul/README.md` for the full story:
SLATE install, client build, slurm sweep, output format, parameter rationale
with documentation pointers, and where the CSV lands.

Parser: `scripts/parse-slate-experiment.py`. Output CSV is kept separate
from `matmul-results.csv` because SLATE reports a single wallclock number
whereas `c_matmul/matmul` decomposes its time into four phases.

---

## Constraints for Working on This Codebase

- Code runs on NERSC Perlmutter. Cannot be compiled or run in this container.
- All analysis is from code reading only. Changes must be carefully reasoned
  and confirmed with the author before treating as correct.
- Python and C++ are independent — changes to one do not need to mirror the other,
  but inconsistencies should be documented.

---

## Potential Vulnerabilities (unconfirmed — see SESSION_LOG.md)

See SESSION_LOG.md for the full list. None are confirmed bugs until
cross-checked with the author. Items are labeled P1-P8 (Python),
C1-C10 (C++), X1-X2 (cross-implementation).
