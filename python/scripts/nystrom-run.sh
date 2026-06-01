#!/bin/bash -l

#SBATCH -q debug
##SBATCH -C cpu
#SBATCH -C gpu
#SBATCH --gpus-per-node=4
#SBATCH -A m4293 # Sparsitute project (A Mathematical Institute for Sparse Computations in Science and Engineering)

#SBATCH -t 0:20:00

#SBATCH -N 4
#SBATCH -J nystrom
#SBATCH -o slurm.nystrom.o%j

# https://docs.nersc.gov/systems/perlmutter/architecture/
#SYSTEM=perlmutter-cpu
SYSTEM=perlmutter-gpu-cpu
N_NODE=${SLURM_NNODES}

if [ "$SYSTEM" == "perlmutter-cpu" ]; then
	# https://docs.nersc.gov/systems/perlmutter/architecture/#cpu-nodes
	module swap PrgEnv-gnu PrgEnv-intel
	module load python

	CORE_PER_NODE=128 # 2 CPUs. 64 cores per CPU. Never change. Specific to the system
	PER_NODE_MEMORY=512 # Never change. Specific to the system
	PROC_PER_NODE=8 # 2 sockets for 2 CPUs. 4 NUMA domains per socket.
elif [ "$SYSTEM" == "perlmutter-gpu-cpu" ]; then
	# https://docs.nersc.gov/systems/perlmutter/architecture/#gpu-nodes
    module swap PrgEnv-gnu PrgEnv-intel
	module load python

	CORE_PER_NODE=64 # 1 CPU, 64 cores per CPU. Never change. Specific to the system
	PER_NODE_MEMORY=256 # Never change. Specific to the system
    PROC_PER_NODE=4 # 1 process per NUMA region, 4 NUMA regions per node
fi

N_PROC=$(( $N_NODE * $PROC_PER_NODE ))
CORE_PER_PROC=$(( $CORE_PER_NODE / $PROC_PER_NODE ))
THREAD_PER_PROC=$(( $CORE_PER_PROC * 2 )) # 2 logical core per physical core. IMPORTANT for mem access within NUMA domain.
PER_PROC_MEM=$(( $PER_NODE_MEMORY / $PROC_PER_NODE - 2)) #2GB margin of error
export OMP_NUM_THREADS=$THREAD_PER_PROC
export MKL_NUM_THREADS=$THREAD_PER_PROC

# === User parameters ===
REPO=$HOME/Codes/nystrom-distributed
ALG=nystrom-1d-redist-1d       # or nystrom-1d-noredist-1d
N=50000
R=5000
AFILE=$SCRATCH/nystrom/data/cifar10-linear.bin                       # path to A or NONE to generate synthetic
YFILE=$SCRATCH/nystrom/data/cifar10-linear-Y-r$R-$ALG-python.bin     # path to write Y or NONE to skip
ZFILE=$SCRATCH/nystrom/data/cifar10-linear-Z-r$R-$ALG-python.bin     # path to write Z or NONE to skip
# =======================

PY=$REPO/python/nystrom.py

EXTRA=""
[ "$AFILE" != "NONE" ] && EXTRA="$EXTRA -afile $AFILE"
[ "$YFILE" != "NONE" ] && EXTRA="$EXTRA -yfile $YFILE"
[ "$ZFILE" != "NONE" ] && EXTRA="$EXTRA -zfile $ZFILE"

srun -N $N_NODE -n $N_PROC -c $THREAD_PER_PROC --ntasks-per-node=$PROC_PER_NODE --cpu-bind=cores \
    python $PY -alg $ALG -n $N -r $R $EXTRA
