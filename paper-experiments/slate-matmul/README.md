# SLATE matmul baseline

This directory contains the SLATE-based distributed DGEMM benchmark used as a
third-party-library baseline for the paper. The goal is to show that our custom
`matmul` (general 3D distributed matrix multiplication) achieves competitive
wall-clock time against SLATE — a widely-used, modern, expert-tuned distributed
dense linear algebra library — on the same matrix size and total MPI process
count.

## Contents

| File | Purpose |
|---|---|
| `slate-matmul.cpp` | Benchmark client: a single timed `slate::gemm` call |
| `CMakeLists.txt` | Builds the client, linking against an installed SLATE |
| `slate-matmul-test.sh` | Slurm submission script for the sweep |

The parser for the output files is in `../scripts/parse-slate-experiment.py`.

## Installing SLATE

SLATE is built and installed once into `$HOME/Codes/slate/_install`; the
client below links against that install. The experiments use **PrgEnv-intel**
+ **cray-mpich** + **Intel MKL** on NERSC Perlmutter, matching the toolchain
our own `c_matmul/` build uses (`paper-experiments/BUILD.md`).

```bash
cd $HOME/Codes
git clone --recursive https://github.com/icl-utk-edu/slate
cd slate

if [ "$PE_ENV" != "INTEL" ]; then
    module swap PrgEnv-$(echo "$PE_ENV" | tr A-Z a-z) PrgEnv-intel
fi

mkdir -p _build && cd _build

CXX=CC cmake .. \
    -DCMAKE_INSTALL_PREFIX=$HOME/Codes/slate/_install \
    -Dblas=mkl \
    -DBLA_VENDOR=Intel10_64lp \
    -Dgpu_backend=none \
    -Dbuild_tests=no \
    -Dmpi=mpich \
    -DCMAKE_BUILD_TYPE=Release

make -j16 install
```

Notes:

- The `--recursive` clone pulls in **BLAS++** and **LAPACK++** as git
  submodules; SLATE's top-level CMake build chain then builds and installs
  both as part of the SLATE install
  ([`INSTALL.md`](https://github.com/icl-utk-edu/slate/blob/master/INSTALL.md)).
- `-Dblas=mkl -DBLA_VENDOR=Intel10_64lp` selects Intel MKL as the BLAS backend
  (matching what `c_matmul/CMakeLists.txt` links against).
- `-Dgpu_backend=none` because this experiment is CPU-only.
- `CXX=CC` makes CMake use Cray's compiler wrapper (which already knows about
  the MPI include/link flags).

## Building the client

```bash
cd $HOME/Codes/nystrom-distributed/paper-experiments/slate-matmul
mkdir -p _build && cd _build
CXX=CC cmake .. -DCMAKE_PREFIX_PATH=$HOME/Codes/slate/_install
make
```

The `-DCMAKE_PREFIX_PATH=$HOME/Codes/slate/_install` is how
`find_package(slate REQUIRED)` locates the SLATE package config files dropped
by the install step. The resulting binary is `_build/slate-matmul`.

## Running the sweep

The sweep matches the existing `c_matmul/matmul` experiment scope: same matrix
size (n = 50000, square A and B), same `perlmutter-gpu-cpu` node type, same
process-per-node count (4), and the same node range (1 → 32 nodes, giving
total MPI processes 4 → 128).

```bash
cd $HOME/Codes/nystrom-distributed
# Edit -N in slate-matmul-test.sh between submissions (1, 2, 4, 8, 16, 32)
sbatch paper-experiments/slate-matmul/slate-matmul-test.sh
```

For each `N_NODE`, the script runs `N_TRY = 10` trials at a 2D `p × q` process
grid selected from a lookup table (see the script).

## Output format

Each run writes one file to
`$SCRATCH/nystrom/slate-matmul_benchmarking/` with the filename pattern

```
slate-gemm_cpp_<system>_<nnode>_<nproc>_<thread>_<n>_<n>_<n>_<p>x<q>x1.<try>
```

— the same shape as `c_matmul/matmul`'s output filenames (with `p × q × 1`
standing in for the 3D `p1 × p2 × p3`), so the SLATE parser can reuse the
filename-split logic from `parse-nystrom-experiment.py`.

Each file's content is four lines:

```
testing 50000x50000 with 50000x50000 on 4x4x1 grid
SLATE tile size nb = 384
Time for SLATE gemm: 1.234 sec
Performance: 1234.56 GF/s
```

## Collecting data

After the sweep finishes:

```bash
cd $HOME/Codes/nystrom-distributed/paper-experiments/scripts
python parse-slate-experiment.py
```

Produces `slate-results.csv` in the current directory with columns:

```
alg, impl, system, nnode, nproc, thread_per_proc, m, k, n,
p, q, try, nb, gemm_time, gflops, timestamp
```

This is a *separate* CSV from `matmul-results.csv`. SLATE has one timing
(the whole `slate::gemm` call) whereas our matmul code reports four
(`gather A`, `gather B`, `local multiply`, `scatter and reduce C`). Trying
to map SLATE's single number onto one of those four columns would conflate
two different things. To make the head-to-head plot, sum the four matmul
timings into a total and join the CSVs on `(nproc, n, system)`.

## Parameter rationale

This section explains *why* each non-default knob was set the way it was,
with pointers to authoritative SLATE documentation.

### Distribution: 2D block-cyclic

SLATE's `Matrix` class is hard-coded to **2D block-cyclic distribution** —
there is no 1D or 3D variant of the underlying GEMM. Confirmed in the
`slate::Matrix` constructor documentation:
[`include/slate/Matrix.hh`](https://github.com/icl-utk-edu/slate/blob/master/include/slate/Matrix.hh)
states "creates an m-by-n matrix… with fixed mb-by-nb tile size and 2D block
cyclic distribution". The underlying algorithm is SUMMA (broadcast block
column of A horizontally, block row of B vertically) — see the SLATE design
paper, Gates et al., SC '19,
[ICL-UTK-1351-2019](https://netlib.org/utk/people/JackDongarra/PAPERS/icl-utk-1351-2019.pdf).

This mismatch (our matmul uses a 3D grid, SLATE uses 2D) is unavoidable. We
report the comparison anyway because the practically meaningful question is
"wall-clock time for the same matrix product at the same total MPI process
count," not "wall-clock time for the same grid shape."

### Tile size: `nb = 384`

SLATE has no autotuner; `nb` is a user-controlled knob. Performance is
moderately sensitive to it — too small under-utilises BLAS-3 inside MKL DGEMM,
too large reduces parallelism. The value 384 is in the typical productive
range (256-512) documented in the SLATE Users' Guide
([Gates et al., 2020, ICL-UTK-1664](https://icl.utk.edu/files/publications/2020/icl-utk-1664-2020.pdf),
Section 7.2 "Performance considerations"). For a more rigorous benchmark
you'd sweep `nb ∈ {192, 256, 320, 384, 512}` and report the best; this
experiment fixes one value to keep total compute time bounded.

### Process grid: `p ≈ q`

For SUMMA the per-step broadcast volume is balanced when the process grid is
square; rectangular grids increase the volume in the longer dimension. The
SLATE Users' Guide recommends square or near-square grids; when not square,
`q = 2 p` works well with the default `ColMajor` grid order
([ICL-UTK-1664](https://icl.utk.edu/files/publications/2020/icl-utk-1664-2020.pdf),
Section 7.2). The script's `P × Q` lookup:

| N_PROC | P × Q |
|---:|---:|
| 4 | 2 × 2 |
| 8 | 2 × 4 |
| 16 | 4 × 4 |
| 32 | 4 × 8 |
| 64 | 8 × 8 |
| 128 | 8 × 16 |

uses square grids for perfect squares (4, 16, 64) and `q = 2 p` otherwise.

### `Target::HostTask`

Selects the CPU code path. SLATE's `gemm` accepts a `Target` option that
chooses between host-task, host-nest, host-batch, and device (GPU) execution.
We are CPU-only for this experiment, so `HostTask` is the correct choice. See
the SLATE Users' Guide
([ICL-UTK-1664](https://icl.utk.edu/files/publications/2020/icl-utk-1664-2020.pdf),
Section 4 "Options").

### `Lookahead = 1`

This is SLATE's documented default and represents one block of overlap
between the SUMMA broadcast and the local DGEMM. We use the default rather
than a tuned value to make the comparison reflect "SLATE out of the box," not
SLATE tuned for this specific problem. Setting higher values is tuning that
risks giving SLATE an unfair edge.

### `MKL_NUM_THREADS = 1`

SLATE issues batched **single-threaded** BLAS calls inside OpenMP tasks; the
threading is provided by SLATE's OpenMP, not by MKL. Setting
`MKL_NUM_THREADS = OMP_NUM_THREADS` creates nested parallelism (every OpenMP
task spawns MKL threads that fight for cores) and destroys performance. The
single-threaded BLAS requirement is stated in the SLATE Users' Guide
([ICL-UTK-1664](https://icl.utk.edu/files/publications/2020/icl-utk-1664-2020.pdf),
Section 3.4 "Threading model").

Our own `matmul` is the opposite — it relies on multi-threaded MKL DGEMM for
all of its parallelism within a process, and so runs with
`MKL_NUM_THREADS = OMP_NUM_THREADS`. This is *not* an unfair comparison: each
library is configured to use the threading model it was designed for.

### `MPICH_MAX_THREAD_SAFETY = multiple`

SLATE issues MPI calls from inside OpenMP tasks (multiple threads concurrently
call MPI during the SUMMA broadcast). That requires `MPI_THREAD_MULTIPLE`,
which Cray MPICH only provides when `MPICH_MAX_THREAD_SAFETY=multiple` is set
([NERSC Cray MPICH docs](https://docs.nersc.gov/development/programming-models/mpi/mpich/)).
Without this, `MPI_Init_thread` returns a lower thread-safety level and the
benchmark client aborts at startup.

Our own `matmul` uses MPI only from the main thread (OpenMP is confined to
MKL DGEMM, which does not call MPI), so `MPI_THREAD_SINGLE` — the Cray MPICH
default — is sufficient for it. Again, each library uses the MPI threading
model it was designed for.

### `MPI_Init_thread(MPI_THREAD_MULTIPLE)`

Companion to the variable above: the benchmark client requests
`MPI_THREAD_MULTIPLE` at MPI initialization. SLATE relies on this; calling
`MPI_Init` (which implies `MPI_THREAD_SINGLE`) leads to undefined behavior in
SLATE's broadcast routines.

## Source references

- SLATE source: <https://github.com/icl-utk-edu/slate>
  - [`include/slate/Matrix.hh`](https://github.com/icl-utk-edu/slate/blob/master/include/slate/Matrix.hh) — `Matrix` constructor + 2D block-cyclic documentation
  - [`src/gemm.cc`](https://github.com/icl-utk-edu/slate/blob/master/src/gemm.cc) — `gemm` signature, `Options`, `Lookahead`, `Target` defaults
  - [`examples/ex05_blas.cc`](https://github.com/icl-utk-edu/slate/blob/master/examples/ex05_blas.cc) — canonical gemm setup pattern
  - [`INSTALL.md`](https://github.com/icl-utk-edu/slate/blob/master/INSTALL.md) — CMake flags, submodules, BLAS vendors
- [SLATE Users' Guide (Gates et al., 2020, ICL-UTK-1664)](https://icl.utk.edu/files/publications/2020/icl-utk-1664-2020.pdf) — threading model, options, performance considerations
- [SLATE: Design of a Modern Distributed and Accelerated Linear Algebra Library (Gates et al., SC '19)](https://netlib.org/utk/people/JackDongarra/PAPERS/icl-utk-1351-2019.pdf) — algorithm and architecture
- [NERSC Cray MPICH documentation](https://docs.nersc.gov/development/programming-models/mpi/mpich/) — `MPICH_MAX_THREAD_SAFETY`
