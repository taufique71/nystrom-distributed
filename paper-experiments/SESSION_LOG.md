# Session Log

---

## Session 1 — 2026-05-29

### Goal
Prepare the codebase for public release following paper acceptance.

### What we discussed

**Overall scope agreed upon:**
- Both Python and C++ implementations should be polished and usable independently
- C++ has more communication patterns than Python — both should be published
- C++ supports CPU (MKL) and GPU (cuBLAS) — both paths should work cleanly
- Repo structure stays largely as-is (scripts, tests, source all at current locations) — reorganizing would break path assumptions in scripts
- Paper experiment scripts and general-use scripts can coexist in `scripts/`
- CIFAR-10 is just one example dataset; the core code should be general

**Uncommitted changes understood:**
- `CIFAR10bianry.py`: sigma added to output filename, default sigma changed — clean improvement
- `utils.py`: unused torchvision import commented out — fine
- `matrix.h`: integer overflow fix (`(size_t)` cast for allocation sizes) — important fix
- `matrix.h`: new `generateRandom()` method added — new functionality
- `nystrom.cpp`: `-afile`, `-yfile`, `-zfile` CLI args added — significant new feature
- `nystrom.h`: `cudaDeviceSynchronize()` added after cublasDgemm in `nystrom_1d_redist_1d` — correctness fix
- `nystrom.h`: experimental allreduce block alongside reduce-scatter — **commented out this session** (benchmarking artifact, not part of algorithm; also corrupted `tReduceScatter` timing variable)
- `matmul.cpp`: `ParMat C = matmul(A, B)` commented out — regression, `matmul` alg case now does nothing
- `test-cublas.cpp`: completely replaced with GPU FP32 GEMM benchmark
- Scripts: reflect last experiment configuration (1-node debug, different matrix sizes)

**Build system issues identified (not yet fixed):**
- MKL-only on CPU path — not portable to OpenBLAS or Cray LibSci systems
- CUDA architecture hardcoded to `nvidia80` (A100 specific)
- `find_package(CUDA)` is deprecated; modern CMake uses `find_package(CUDAToolkit)`
- `test-cublas` and `test-cuda-aware-mpi` always compiled but only linked in CUDA mode
- OpenMP status message reports MPI compiler variable instead of OpenMP
- Five build directories in repo (`build_cpu`, `build_gpu`, `build_cpu_cifar10`, `build_gpu_cifar10`, `build_cpu_libsci`) — should be gitignored

### Potential vulnerabilities identified (NOT confirmed — to be cross-checked with author)

These were identified by static analysis of the code. They are hypotheses only.
Each item needs to be reviewed with the author before being treated as a confirmed bug.

---

#### Python implementation

**P1 — POTENTIAL CRASH**
`matrix.py` and `nystrom.py`: `np.matmul(..., order='F')` is used throughout.
`numpy.matmul` does not accept an `order` keyword argument. If this is true, every Python matmul call would crash. *Needs verification — the code may have been run successfully, so this understanding may be wrong.*

**P2 — POTENTIAL CRASH**
`matrix.py` `generate_rand()`: the variable `dtype` is referenced inside the function but is not in the function's parameter list. The call site passes `dtype` as a keyword argument which also doesn't exist in the signature. *Needs verification with author.*

**P3 — POTENTIAL CRASH**
`nystrom.py` `checkCorrectness()` line 13: uses `A` (a `ParMat` object from an outer scope) instead of the function parameter `Amat` (a numpy array). This function is commented out in main so it would not affect runtime currently.

**P4 — POTENTIAL BUG**
`utils.py` `getCIFAR10GramMatrix()`: `MPI_ORDER_FORTRAN` is commented out in the `Create_subarray` call, so MPI uses C-order (row-major) by default. But `CIFAR10bianry.py` writes the file with `flatten(order='F')` (column-major). This may cause every process to read the wrong block. The C++ `parallelReadBinary` correctly specifies `MPI_ORDER_FORTRAN`. *Needs author verification — the Python CIFAR10 path may not have been used in practice.*

**P5 — POTENTIAL BUG**
`utils.py` `splitAndReduceScatter()` with `split='row'`: scatters contiguous chunks of a column-major matrix, but column-major storage means rows are not contiguous. The unpack loop then assumes C-order (row-major) layout. Used in `nystrom_1d_redist_2d`. *Needs author verification.*

**P6 — POTENTIAL BUG**
`nystrom.py` `nystrom_1d_redist_1d` unpack loop: `blockRows = [0] + [rowsInOtherProc] * (nprocs-1)`. The first element is 0, which may cause the first block of rows to never be written into Y (empty slice). *Needs author verification — the indexing intent may be different from what was analyzed.*

**P7 — POTENTIAL BUG**
`nystrom.py` `nystrom_2d_redist_1d`: the loop variable `r` (iterating over rows) shadows the function parameter `r` (sketch rank). Downstream code referencing `r` after the loop would get the wrong value. Currently no downstream reference appears to use `r` after the loop, but it is a latent hazard.

**P8 — POTENTIAL BUG**
`nystrom.py` `nystrom_1d_redist_2d` line 324: `Z.localMat = np.matmul(..., order='C')`. All other local matrices use `order='F'`. Inconsistent memory layout for Z in this algorithm variant.

---

#### C++ implementation

**C1 — POTENTIAL BUG**
`nystrom.h` `nystrom_1d_noredist_1d`: Omega^T row offset into Y uses `grid2.rowRank` to index which rows of Omega correspond to the local rows of Y. But Y's row distribution (face C) uses `rankInColWorld` (= `colRank`). These may index different dimensions of the process grid, potentially causing the wrong rows of Omega^T to be multiplied against Y. *Needs author verification — likely correct for specific 1D grid configurations used in paper.*

**C2 — POTENTIAL BUG**
`nystrom.h` `nystrom_1d_redist_1d` second dgemm: `cblas_lda = Y.nRowLocal` is used as the leading dimension for Omega (n×r, column-major). The correct leading dimension for a column-major n×r matrix is `n`, not `Y.nRowLocal`. For a 1D grid where Y spans all n rows on each process, `Y.nRowLocal = n` and this would be correct — but for other configurations it would produce wrong results. *Needs author verification of which grid configurations this function is called with.*

**C3 — POTENTIAL BUG**
`nystrom.h` `nystrom_1d_redist_1d` CPU unpack: the base offset into the receive buffer uses `recvDispls[grid2.rankInFibWorld]` as a constant across all sender processes `p`. The correct base for sender `p`'s data should be `recvDispls[p]`. This would write data from the wrong parts of the receive buffer into Y. *Needs author verification.*

**C4 — POTENTIAL BUG**
`nystrom.h` `nystrom_2d_noredist_1d`: the reduce-scatter of Y contributions uses `grid2.colWorld`. The partial sums come from the fiber dimension of grid1 (A's column distribution), so the reduction communicator should likely be `grid1.fibWorld`. *Needs author verification.*

**C5 — POTENTIAL BUG**
`matrix.h` `matmul1_comm`: `cblas_ldb = B.nRowGlobal` is used as the leading dimension for gathered B. After allgather along colWorld, gathered B has `B.nRowLocal` rows (not `B.nRowGlobal`). The correct leading dimension would be `B.nRowLocal`. For a 1D grid where `B.nRowLocal = B.nRowGlobal`, this is correct. *Needs author verification.*

**C6 — CONFIRMED BUG**
`nystrom.cpp` `nystrom_2d_noredist_1d`: `A.generate()` is called unconditionally before `if(afile == "NONE") A.generate()`. This calls generate twice when no file is provided, and wastes work (but doesn't corrupt results) when a file is provided. *Agreed by author — regression from adding file I/O.*

**C7 — MISLEADING COMMENT**
`matrix.h` `parallelReadBinary` line ~301: comment says "row-major" but code correctly uses `MPI_ORDER_FORTRAN`. The code is correct; the comment is wrong and could cause someone to "fix" a correct line.

**C8 — ROBUSTNESS**
`procgrid.h` destructor: all `MPI_Group_free` and communicator free calls are commented out. MPI resources are leaked for every `ProcGrid` construction. Non-critical for short-running programs but worth noting.

**C9 — ROBUSTNESS**
`utils.h` `findSplits`: when `nsplit > n`, the last element becomes negative. No guard against this case.

**C10 — ROBUSTNESS**
`nystrom.h` and `matrix.h`: PRNG seed is `1234 + tid` (thread-indexed). Results change with different `OMP_NUM_THREADS` settings, affecting reproducibility.

---

#### Cross-implementation inconsistencies

**X1**
Python uses `Xoroshiro128` (xoroshiro128+) for Omega generation. C++ uses MKL-VSL Philox4x32x10 (CPU) or cuRAND Philox4_32_10 (GPU). The Omega matrices are different across implementations. Not a correctness bug individually, but means Python and C++ results cannot be compared element-by-element.

**X2**
`test-prng.cpp` `h_data` array is allocated but never populated on the CPU build path, then read — uninitialized memory access. This is in a test file, not the main algorithm.

---

### What was changed this session

1. `c_matmul/nystrom.h`: Commented out experimental allreduce block and `contribZred` buffer in `nystrom_1d_noredist_1d`. The `MPI_Reduce_scatter` (the correct operation) is untouched.

---

## Session 2 — 2026-05-30

### Goal
Build and verify both CPU and GPU builds on Perlmutter. Document the build process.

### What we discussed

- Reviewed matmul experiment scripts: `matmul-test.sh` (performance), `matmul-correctness-check.sh` (correctness), `cifar10-matmul-test.sh` (broken — references non-existent `build_gpu_debug`)
- Reviewed matmul-results.csv: 746 rows, 57 configs with 10-11 runs each (C++ only). Python matmul data exists but only 30 rows, all single runs — not enough for averaging.
- nystrom-results.csv is C++ only (950 rows, all 10 runs per config). No Python nystrom benchmarks exist yet.

### What was changed this session

1. `c_matmul/CMakeLists.txt`: Fixed MKL linking — replaced `${MKL_LIBRARIES}` with `MKL::MKL` imported target (Intel oneAPI 2021+ no longer populates the old variable).
2. `c_matmul/CMakeLists.txt`: Fixed `test-cublas` and `test-cua-mpi` always compiling regardless of `USE_CUDA` — wrapped their `add_executable` calls in `if (USE_CUDA)`.
3. `BUILD.md`: Created with CPU and GPU build instructions.

### Build verified on Perlmutter (2026-05-30)

**CPU build** — IntelLLVM 2025.3.1, MKL 2025.3.0, MPI 4.0: clean, no errors.
**GPU build** — NVHPC 25.5.0, CUDA 12.9: clean, no errors. Two expected cmake warnings (CMP0146, CMP0104).

### Still to do (agreed scope)

- [ ] Fix confirmed bug C6 (`A.generate()` called twice in `nystrom_2d_noredist_1d`)
- [ ] Fix regression in `matmul.cpp` (`ParMat C = matmul(A, B)` commented out)
- [ ] Cross-check all potential vulnerabilities above with author
- [ ] Build system: portability (MKL-only, CUDA architecture, deprecated find_package)
- [ ] Remove/gitignore: slurm output files, build directories
- [ ] Hardcoded paths in scripts (replace `Codes/nystrom-distributed` with variable)
- [ ] Add README and LICENSE
- [ ] Generalize matrix I/O functions (square-only, float64-only, ordering bug)
- [ ] Clean up dead code, unused imports
- [ ] Rerun matmul benchmarks (Python and C++) with multiple runs and average over runs in plots.ipynb
