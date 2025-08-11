#!/bin/bash -l

#SBATCH -q debug 

#SBATCH -C cpu
##SBATCH --gpus-per-node=4

#SBATCH -A m4293 # Sparsitute project (A Mathematical Institute for Sparse Computations in Science and Engineering)

#SBATCH -t 0:05:00

#SBATCH -N 1
#SBATCH -J matmul
#SBATCH -o slurm.matmul.o%j

# https://docs.nersc.gov/systems/perlmutter/architecture/
SYSTEM=perlmutter-cpu
N_NODE=1

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

P1=1
P2=1
P3=1
N1=50000
N2=50000
N3=5000

#for ALG in [ "matmul", "matmul1gen", "matmul1comm" ]; do
#for ALG in matmul
for ALG in matmul1gen matmul1comm 
do
    #for IMPL in cpp python
	for IMPL in cpp
    do
        echo $ALG, $IMPL
        if [ "$ALG" == "matmul" ]; then
            if [ "$N_PROC" -eq 4 ]; then
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

        #STDOUT_FILE=$SCRATCH/nystrom/"$ALG"_"$IMPL"_"$N_NODE"_"$N_PROC"_"$P1"x"$P2"x"$P3"
        STDOUT_FILE=$SCRATCH/nystrom/matmul_benchmarking/"$ALG"_"$IMPL"_"$SYSTEM"_"$N_NODE"_"$N_PROC"_"$THREAD_PER_PROC"_"$N1"_"$N2"_"$N3"_"$P1"x"$P2"x"$P3"

        if [ "$SYSTEM" == "perlmutter-cpu" ]; then
            PY=$HOME/Codes/nystrom-distributed/tests/matmul-test.py
            BIN=$HOME/Codes/nystrom-distributed/build_cpu/c_matmul/matmul
        elif [ "$SYSTEM" == "perlmutter-gpu" ]; then
            PY=$HOME/Codes/nystrom-distributed/tests/matmul-test.py
            BIN=$HOME/Codes/nystrom-distributed/build_gpu/c_matmul/matmul
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
