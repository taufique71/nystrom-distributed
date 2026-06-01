# Build Instructions

Tested on NERSC Perlmutter.

All commands below assume the working directory is `paper-experiments/` (the root
of this paper-experiments tree, containing `CMakeLists.txt` and `c_matmul/`):

```bash
cd $HOME/Codes/nystrom-distributed/paper-experiments
```

## CPU Build

```bash
module swap PrgEnv-gnu PrgEnv-intel
module load python

mkdir -p build_cpu
cd build_cpu
cmake ..
make
```

## GPU Build

```bash
module swap PrgEnv-gnu PrgEnv-nvidia
module load cudatoolkit
module load craype-accel-nvidia80

mkdir -p build_gpu
cd build_gpu
cmake ..
make
```

Produces `matmul` and `nystrom` binaries under `build_cpu/c_matmul/` or `build_gpu/c_matmul/`.
