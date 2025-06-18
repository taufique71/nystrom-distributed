#!/bin/bash -l

#SBATCH -q debug 
#SBATCH -C cpu
#SBATCH -A m4293 # Sparsitute project (A Mathematical Institute for Sparse Computations in Science and Engineering)

#SBATCH -t 0:10:00

#SBATCH -N 1
#SBATCH -J matmul
#SBATCH -o slurm.matmul.o%j

module swap PrgEnv-gnu PrgEnv-intel
module load python

IMPL=python

SYSTEM=perlmutter_cpu
CORE_PER_NODE=128 # Never change. Specific to the system
PER_NODE_MEMORY=256 # Never change. Specific to the system
N_NODE=1
PROC_PER_NODE=128
N_PROC=$(( $N_NODE * $PROC_PER_NODE ))
THREAD_PER_PROC=$(( $CORE_PER_NODE / $PROC_PER_NODE ))
PER_PROC_MEM=$(( $PER_NODE_MEMORY / $PROC_PER_NODE - 2)) #2GB margin of error
export OMP_NUM_THREADS=$THREAD_PER_PROC
export MKL_NUM_THREADS=$THREAD_PER_PROC

P1=1
P2=1
P3=1
N1=10000
N2=10000
N3=10000
if [ "$N_PROC" -eq 128 ]; then
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

STDOUT_FILE=$SCRATCH/nystrom/matmul_"$IMPL"_"$N_PROC"_"$P1"x"$P2"x"$P3"

PY=$HOME/Codes/nystrom-distributed/tests/matmul-test.py
BIN=$HOME/Codes/nystrom-distributed/build/c_matmul/matmul

if [ "$IMPL" == "cpp" ]; then
    srun -N $N_NODE -n $N_PROC -c $THREAD_PER_PROC --ntasks-per-node=$PROC_PER_NODE --cpu-bind=cores \
        $BIN -p1 $P1 -p2 $P2 -p3 $P3 -n1 $N1 -n2 $N2 -n3 $N3 &> $STDOUT_FILE
elif [ "$IMPL" == "python" ]; then
    srun -N $N_NODE -n $N_PROC -c $THREAD_PER_PROC --ntasks-per-node=$PROC_PER_NODE --cpu-bind=cores \
        python $PY -p1 $P1 -p2 $P2 -p3 $P3 -n1 $N1 -n2 $N2 -n3 $N3 &> $STDOUT_FILE

fi

