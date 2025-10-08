#!/bin/bash -l

#SBATCH -q regular 

#SBATCH -C gpu
#SBATCH --gpus-per-node=4

#SBATCH -A m4293 # Sparsitute project (A Mathematical Institute for Sparse Computations in Science and Engineering)

#SBATCH -t 0:02:00

#SBATCH -N 1
#SBATCH -J cua-mpi-test
#SBATCH -o slurm.cua-mpi.o%j

# https://docs.nersc.gov/systems/perlmutter/architecture/
#SYSTEM=perlmutter-gpu
SYSTEM=perlmutter-gpu-cpu
#SYSTEM=perlmutter-cpu
N_NODE=${SLURM_NNODES}
#N_NODE=4

if [ "$SYSTEM" == "perlmutter-cpu" ]; then
	# https://docs.nersc.gov/systems/perlmutter/architecture/#cpu-nodes

	module swap PrgEnv-gnu PrgEnv-intel
	module load python

	CORE_PER_NODE=128 # 2 CPUs. 64 cores per CPU. Never change. Specific to the system
	PER_NODE_MEMORY=512 # Never change. Specific to the system
	PROC_PER_NODE=4 # 2 sockets for 2 CPUs. 4 NUMA domains per socket.
elif [ "$SYSTEM" == "perlmutter-gpu" ]; then
	# https://docs.nersc.gov/systems/perlmutter/architecture/#gpu-nodes

    module swap PrgEnv-gnu PrgEnv-nvidia
    module load python
    module load cudatoolkit
    module load craype-accel-nvidia80
    
    export MPICH_GPU_SUPPORT_ENABLED=1

	CORE_PER_NODE=64 # 1 CPU, 64 cores per CPU. Never change. Specific to the system
	PER_NODE_MEMORY=256 # Never change. Specific to the system
	PROC_PER_NODE=4 # 1 process per GPU, 4 GPU per node
elif [ "$SYSTEM" == "perlmutter-gpu-cpu" ]; then

    module swap PrgEnv-gnu PrgEnv-intel
	module load python

	CORE_PER_NODE=64 # 1 CPU, 64 cores per CPU. Never change. Specific to the system
	PER_NODE_MEMORY=256 # Never change. Specific to the system
	PROC_PER_NODE=4 # 1 process per GPU, 4 GPU per node
fi

N_PROC=$(( $N_NODE * $PROC_PER_NODE ))
CORE_PER_PROC=$(( $CORE_PER_NODE / $PROC_PER_NODE )) 
THREAD_PER_PROC=$(( $CORE_PER_PROC * 2 )) # 2 logical core per physical core. IMPORTANT for mem access within NUMA domain.
PER_PROC_MEM=$(( $PER_NODE_MEMORY / $PROC_PER_NODE - 2)) #2GB margin of error
export OMP_NUM_THREADS=$THREAD_PER_PROC
export MKL_NUM_THREADS=$THREAD_PER_PROC

if [ "$SYSTEM" == "perlmutter-cpu" ]; then
	BIN=$HOME/Codes/nystrom-distributed/build_cpu/c_matmul/test-cua-mpi
elif [ "$SYSTEM" == "perlmutter-gpu-cpu" ]; then
	BIN=$HOME/Codes/nystrom-distributed/build_cpu/c_matmul/test-cua-mpi
elif [ "$SYSTEM" == "perlmutter-gpu" ]; then
	BIN=$HOME/Codes/nystrom-distributed/build_gpu/c_matmul/test-cua-mpi
fi

STDOUT_FILE=$SCRATCH/nystrom/cua_mpi_test/reduce-scatter_"$SYSTEM"_"$N_NODE"_"$N_PROC"_5000
echo $STDOUT_FILE
srun -N $N_NODE -n $N_PROC -c $THREAD_PER_PROC --ntasks-per-node=$PROC_PER_NODE --cpu-bind=cores $BIN &> $STDOUT_FILE
