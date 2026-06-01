#!/bin/bash -l

#SBATCH -q regular 

#SBATCH -C gpu
#SBATCH --gpus-per-node=4

#SBATCH -A m4293 # Sparsitute project (A Mathematical Institute for Sparse Computations in Science and Engineering)

#SBATCH -t 0:15:00

#SBATCH -N 64
#SBATCH -J matmul
#SBATCH -o slurm.matmul.o%j

# https://docs.nersc.gov/systems/perlmutter/architecture/
#SYSTEM=perlmutter-cpu
SYSTEM=perlmutter-gpu-cpu
#SYSTEM=perlmutter-gpu
#N_NODE=1
N_NODE=${SLURM_NNODES}

if [ "$SYSTEM" == "perlmutter-cpu" ]; then
	# https://docs.nersc.gov/systems/perlmutter/architecture/#cpu-nodes

	module swap PrgEnv-gnu PrgEnv-intel
	module load python

	CORE_PER_NODE=128 # 2 CPUs. 64 cores per CPU. Never change. Specific to the system
	PER_NODE_MEMORY=512 # Never change. Specific to the system
	PROC_PER_NODE=8 # 2 sockets for 2 CPUs. 4 NUMA domains per socket.
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
    PROC_PER_NODE=4 # 1 process per NUMA region, 4 NUMA regions per node
	#PROC_PER_NODE=16
fi

#N_PROC=1
#THREAD_PER_PROC=1 

N_PROC=$(( $N_NODE * $PROC_PER_NODE ))
CORE_PER_PROC=$(( $CORE_PER_NODE / $PROC_PER_NODE )) 
THREAD_PER_PROC=$(( $CORE_PER_PROC * 2 )) # 2 logical core per physical core. IMPORTANT for mem access within NUMA domain.
PER_PROC_MEM=$(( $PER_NODE_MEMORY / $PROC_PER_NODE - 2)) #2GB margin of error
export OMP_NUM_THREADS=$THREAD_PER_PROC
export MKL_NUM_THREADS=$THREAD_PER_PROC

P1=1
P2=1
P3=1
N1=500000
N2=1000000
N3=1000
N_TRY=10

#for N3 in 5000 500
for N3 in 1000
do
    #for ALG in matmul
    for ALG in matmul1gen
    #for ALG in matmul1comm
    #for ALG in matmul1gen matmul1comm 
    do
        #for IMPL in cpp python
        for IMPL in cpp
        #for IMPL in python
        do
            echo $ALG, $IMPL
            if [ "$ALG" == "matmul" ]; then
                if [ "$N_PROC" -eq 1 ]; then
                    P1=1
                    P2=1
                    P3=1
                elif [ "$N_PROC" -eq 4 ]; then
                    P1=2
                    P2=2
                    P3=1
                elif [ "$N_PROC" -eq 8 ]; then
                    P1=2
                    P2=2
                    P3=2
                elif [ "$N_PROC" -eq 16 ]; then
                    P1=4
                    P2=2
                    P3=2
                elif [ "$N_PROC" -eq 32 ]; then
                    P1=4
                    P2=4
                    P3=2
                elif [ "$N_PROC" -eq 64 ]; then
                    P1=4
                    P2=4
                    P3=4
                elif [ "$N_PROC" -eq 128 ]; then
                    P1=8
                    P2=4
                    P3=4
                elif [ "$N_PROC" -eq 256 ]; then
                    P1=8
                    P2=8
                    P3=4
                elif [ "$N_PROC" -eq 512 ]; then
                    P1=8
                    P2=8
                    P3=8
                elif [ "$N_PROC" -eq 1024 ]; then
                    P1=16
                    P2=8
                    P3=8
                fi
            elif [ "$ALG" == "matmul1gen" ]; then
                P1=$N_PROC
                P2=1
                P3=1
            elif [ "$ALG" == "matmul1comm" ]; then
                P1=$N_PROC
                P2=1
                P3=1
            fi

			for TRY in $(seq 1 $N_TRY); do
                #STDOUT_FILE=$SCRATCH/nystrom/"$ALG"_"$IMPL"_"$N_NODE"_"$N_PROC"_"$P1"x"$P2"x"$P3"
                STDOUT_FILE=$SCRATCH/nystrom/matmul_benchmarking/"$ALG"_"$IMPL"_"$SYSTEM"_"$N_NODE"_"$N_PROC"_"$THREAD_PER_PROC"_"$N1"_"$N2"_"$N3"_"$P1"x"$P2"x"$P3"."$TRY"
                echo $STDOUT_FILE

                if [ "$SYSTEM" == "perlmutter-cpu" ]; then
                    PY=$HOME/Codes/nystrom-distributed/paper-experiments/tests/matmul-test.py
                    BIN=$HOME/Codes/nystrom-distributed/paper-experiments/build_cpu/c_matmul/matmul
                elif [ "$SYSTEM" == "perlmutter-gpu" ]; then
                    PY=$HOME/Codes/nystrom-distributed/paper-experiments/tests/matmul-test.py
                    BIN=$HOME/Codes/nystrom-distributed/paper-experiments/build_gpu/c_matmul/matmul
                elif [ "$SYSTEM" == "perlmutter-gpu-cpu" ]; then
                    PY=$HOME/Codes/nystrom-distributed/paper-experiments/tests/matmul-test.py
                    BIN=$HOME/Codes/nystrom-distributed/paper-experiments/build_cpu/c_matmul/matmul
                fi

                if [ "$IMPL" == "cpp" ]; then
                    srun -N $N_NODE -n $N_PROC -c $THREAD_PER_PROC --ntasks-per-node=$PROC_PER_NODE --cpu-bind=cores \
                        $BIN -p1 $P1 -p2 $P2 -p3 $P3 -n1 $N1 -n2 $N2 -n3 $N3 -alg $ALG &> $STDOUT_FILE
                    #srun -N $N_NODE -n $N_PROC -c $THREAD_PER_PROC --ntasks-per-node=$PROC_PER_NODE --cpu-bind=cores \
                        #check-hybrid.gnu.pm | sort -k4,4n -k6,6n &> blah.txt
                elif [ "$IMPL" == "python" ]; then
                    srun -N $N_NODE -n $N_PROC -c $THREAD_PER_PROC --ntasks-per-node=$PROC_PER_NODE --cpu-bind=cores \
                        python $PY -p1 $P1 -p2 $P2 -p3 $P3 -n1 $N1 -n2 $N2 -n3 $N3 -alg $ALG &> $STDOUT_FILE
                fi
            done
        done
    done
done
