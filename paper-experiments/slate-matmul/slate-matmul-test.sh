#!/bin/bash -l

#SBATCH -q regular
##SBATCH -C cpu
#SBATCH -C gpu
#SBATCH --gpus-per-node=4
#SBATCH -A m4293 # Sparsitute project (A Mathematical Institute for Sparse Computations in Science and Engineering)

#SBATCH -t 0:20:00

#SBATCH -N 4
#SBATCH -J slate-matmul
#SBATCH -o slurm.slate-matmul.o%j

# https://docs.nersc.gov/systems/perlmutter/architecture/
#SYSTEM=perlmutter-cpu
SYSTEM=perlmutter-gpu-cpu
#N_NODE=${SLURM_NNODES}
N_NODE=1

if [ "$SYSTEM" == "perlmutter-cpu" ]; then
    # https://docs.nersc.gov/systems/perlmutter/architecture/#cpu-nodes
    if [ "$PE_ENV" != "INTEL" ]; then
        module swap PrgEnv-$(echo "$PE_ENV" | tr A-Z a-z) PrgEnv-intel
    fi

    CORE_PER_NODE=128 # 2 CPUs. 64 cores per CPU. Never change. Specific to the system
    PER_NODE_MEMORY=512 # Never change. Specific to the system
    PROC_PER_NODE=8 # 2 sockets for 2 CPUs. 4 NUMA domains per socket.
elif [ "$SYSTEM" == "perlmutter-gpu-cpu" ]; then
    # https://docs.nersc.gov/systems/perlmutter/architecture/#gpu-nodes
    if [ "$PE_ENV" != "INTEL" ]; then
        module swap PrgEnv-$(echo "$PE_ENV" | tr A-Z a-z) PrgEnv-intel
    fi

    CORE_PER_NODE=64 # 1 CPU, 64 cores per CPU. Never change. Specific to the system
    PER_NODE_MEMORY=256 # Never change. Specific to the system
    PROC_PER_NODE=4 # 1 process per NUMA region, 4 NUMA regions per node
fi

N_PROC=$(( $N_NODE * $PROC_PER_NODE ))
CORE_PER_PROC=$(( $CORE_PER_NODE / $PROC_PER_NODE ))
THREAD_PER_PROC=$(( $CORE_PER_PROC * 2 )) # 2 logical core per physical core. IMPORTANT for mem access within NUMA domain.
PER_PROC_MEM=$(( $PER_NODE_MEMORY / $PROC_PER_NODE - 2)) #2GB margin of error
export OMP_NUM_THREADS=$THREAD_PER_PROC
export MKL_NUM_THREADS=1
export MPICH_MAX_THREAD_SAFETY=multiple

# === User parameters ===
REPO=$HOME/Codes/nystrom-distributed
N=50000
NB=384
N_TRY=10
# =======================

# Process grid for SLATE (2D, p*q == N_PROC). Mirrors paper-experiments style:
# square when N_PROC is a perfect square, otherwise q = 2*p (per SLATE guidance).
if [ "$N_PROC" -eq 4 ]; then
    P=2; Q=2
elif [ "$N_PROC" -eq 8 ]; then
    P=2; Q=4
elif [ "$N_PROC" -eq 16 ]; then
    P=4; Q=4
elif [ "$N_PROC" -eq 32 ]; then
    P=4; Q=8
elif [ "$N_PROC" -eq 64 ]; then
    P=8; Q=8
elif [ "$N_PROC" -eq 128 ]; then
    P=8; Q=16
fi

BIN=$REPO/paper-experiments/slate-matmul/_build/slate-matmul

OUT_DIR=$SCRATCH/nystrom/slate-matmul_benchmarking
mkdir -p $OUT_DIR

for TRY in $(seq 1 $N_TRY); do
    STDOUT_FILE=$OUT_DIR/slate-gemm_cpp_${SYSTEM}_${N_NODE}_${N_PROC}_${THREAD_PER_PROC}_${N}_${N}_${N}_${P}x${Q}x1.${TRY}
    echo $STDOUT_FILE

    srun -N $N_NODE -n $N_PROC -c $THREAD_PER_PROC --ntasks-per-node=$PROC_PER_NODE --cpu-bind=cores \
        $BIN -n $N -p $P -q $Q -nb $NB &> $STDOUT_FILE
done
