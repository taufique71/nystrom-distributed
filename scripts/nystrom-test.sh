#!/bin/bash -l

#SBATCH -q debug 
#SBATCH -C cpu
#SBATCH -A m4293 # Sparsitute project (A Mathematical Institute for Sparse Computations in Science and Engineering)

#SBATCH -t 0:10:00

#SBATCH -N 8
#SBATCH -J nystrom
#SBATCH -o slurm.nystrom.o%j

module swap PrgEnv-gnu PrgEnv-intel
module load python

SYSTEM=perlmutter_cpu
CORE_PER_NODE=128 # Never change. Specific to the system
PER_NODE_MEMORY=256 # Never change. Specific to the system
PER_CORE_THREAD=2 # Never change. Specific to the system
N_NODE=1
PROC_PER_NODE=8
N_PROC=$(( $N_NODE * $PROC_PER_NODE ))
CORE_PER_PROC=$(( $CORE_PER_NODE / $PROC_PER_NODE )) 
THREAD_PER_PROC=$(( $CORE_PER_PROC * $PER_CORE_THREAD )) 
PER_PROC_MEM=$(( $PER_NODE_MEMORY / $PROC_PER_NODE - 2)) #2GB margin of error
export OMP_NUM_THREADS=$THREAD_PER_PROC
export MKL_NUM_THREADS=$THREAD_PER_PROC
#export OMP_PLACES=threads
#export OMP_PROC_BIND=spread

N=10000
R=1000

MATMUL1_P1=$N_PROC
MATMUL1_P2=1
MATMUL1_P3=1

MATMUL2_P1=$N_PROC
MATMUL2_P2=1
MATMUL2_P3=1

for ALG in nystrom-1d-noredist-1d nystrom-1d-redist-1d
#for ALG in nystrom-1d-noredist-1d
#for ALG in nystrom-1d-redist-1d
do
    #for IMPL in cpp python
    #for IMPL in cpp
    for IMPL in python 
    do
        echo $ALG, $IMPL
        if [ "$ALG" = "nystrom-1d-noredist-1d" ]; then
            MATMUL1_P1=$N_PROC
            MATMUL1_P2=1
            MATMUL1_P3=1

            MATMUL2_P1=$N_PROC
            MATMUL2_P2=1
            MATMUL2_P3=1
        elif [ "$ALG" = "nystrom-1d-redist-1d" ]; then
            MATMUL1_P1=$N_PROC
            MATMUL1_P2=1
            MATMUL1_P3=1

            MATMUL2_P1=1
            MATMUL2_P2=1
            MATMUL2_P3=$N_PROC
        fi

        STDOUT_FILE=$SCRATCH/nystrom/nystrom_benchmarking/"$ALG"_"$IMPL"_"$N_NODE"_"$N_PROC"_"$THREAD_PER_PROC"_"$N"_"$R"_"$MATMUL1_P1"x"$MATMUL1_P2"x"$MATMUL1_P3"_"$MATMUL2_P1"x"$MATMUL2_P2"x"$MATMUL2_P3"

        PY=$HOME/Codes/nystrom-distributed/nystrom.py
        BIN=$HOME/Codes/nystrom-distributed/build/c_matmul/nystrom

        if [ "$IMPL" == "cpp" ]; then
            srun -N $N_NODE -n $N_PROC -c $THREAD_PER_PROC --ntasks-per-node=$PROC_PER_NODE --cpu-bind=cores \
                $BIN -n $N -r $R -alg $ALG \
                -matmul1p1 $MATMUL1_P1 -matmul1p2 $MATMUL1_P2 -matmul1p3 $MATMUL1_P3  \
                -matmul2p1 $MATMUL2_P1 -matmul2p2 $MATMUL2_P2 -matmul2p3 $MATMUL2_P3  \
                &> $STDOUT_FILE
            #srun -N $N_NODE -n $N_PROC -c $THREAD_PER_PROC --ntasks-per-node=$PROC_PER_NODE --cpu-bind=cores \
                #check-hybrid.gnu.pm | sort -k4,4n -k6,6n &> blah.txt
        elif [ "$IMPL" == "python" ]; then
            #echo "Not ready" &> $STDOUT_FILE
            srun -N $N_NODE -n $N_PROC -c $THREAD_PER_PROC --ntasks-per-node=$PROC_PER_NODE --cpu-bind=cores \
                python $PY -n $N -r $R -alg $ALG \
                -matmul1p1 $MATMUL1_P1 -matmul1p2 $MATMUL1_P2 -matmul1p3 $MATMUL1_P3  \
                -matmul2p1 $MATMUL2_P1 -matmul2p2 $MATMUL2_P2 -matmul2p3 $MATMUL2_P3  \
                &> $STDOUT_FILE
        fi
    done
done
