#!/bin/bash -l

#SBATCH -q debug 
#SBATCH -C cpu
#SBATCH -A m4293 # Sparsitute project (A Mathematical Institute for Sparse Computations in Science and Engineering)

#SBATCH -t 0:10:00

#SBATCH -N 8
#SBATCH -J matmul
#SBATCH -o slurm.matmul.o%j

module swap PrgEnv-gnu PrgEnv-intel
module load python

SYSTEM=perlmutter_cpu
CORE_PER_NODE=128 # Never change. Specific to the system
PER_NODE_MEMORY=256 # Never change. Specific to the system
N_NODE=2
PROC_PER_NODE=8
N_PROC=$(( $N_NODE * $PROC_PER_NODE ))
CORE_PER_PROC=$(( $CORE_PER_NODE / $PROC_PER_NODE )) 
THREAD_PER_PROC=$(( $CORE_PER_PROC * 2 )) # Set number of threads to be twice the number of physical cores ( using logical cores )
PER_PROC_MEM=$(( $PER_NODE_MEMORY / $PROC_PER_NODE - 2)) #2GB margin of error
export OMP_NUM_THREADS=$THREAD_PER_PROC
export MKL_NUM_THREADS=$THREAD_PER_PROC
#export OMP_PLACES=threads
#export OMP_PROC_BIND=spread

P1=1
P2=1
P3=1
N=50000
R=5000

for ALG in nystrom-1d-noredist-1d
do
    #for IMPL in cpp python
    for IMPL in cpp
    do
        echo $ALG, $IMPL
        P1=$N_PROC
        P2=1
        P3=1

        STDOUT_FILE=$SCRATCH/nystrom/nystrom_benchmarking/"$ALG"_"$IMPL"_"$N_NODE"_"$N_PROC"_"$THREAD_PER_PROC"_"$P1"x"$P2"x"$P3"_"$N"_"$R"

        PY=$HOME/Codes/nystrom-distributed/tests/matmul-test.py
        BIN=$HOME/Codes/nystrom-distributed/build/c_matmul/nystrom

        if [ "$IMPL" == "cpp" ]; then
            srun -N $N_NODE -n $N_PROC -c $THREAD_PER_PROC --ntasks-per-node=$PROC_PER_NODE --cpu-bind=cores \
                $BIN -p1 $P1 -p2 $P2 -p3 $P3 -n $N -r $R -alg $ALG &> $STDOUT_FILE
            #srun -N $N_NODE -n $N_PROC -c $THREAD_PER_PROC --ntasks-per-node=$PROC_PER_NODE --cpu-bind=cores \
                #check-hybrid.gnu.pm | sort -k4,4n -k6,6n &> blah.txt
        elif [ "$IMPL" == "python" ]; then
            echo "Not ready" &> $STDOUT_FILE
            #srun -N $N_NODE -n $N_PROC -c $THREAD_PER_PROC --ntasks-per-node=$PROC_PER_NODE --cpu-bind=cores \
                #python $PY -p1 $P1 -p2 $P2 -p3 $P3 -n1 $N1 -n2 $N2 -n3 $N3 -alg $ALG &> $STDOUT_FILE
        fi
    done
done
