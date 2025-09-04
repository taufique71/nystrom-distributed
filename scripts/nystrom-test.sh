#!/bin/bash -l

#SBATCH -q debug 
##SBATCH -C cpu
#SBATCH -C gpu
#SBATCH --gpus-per-node=4
#SBATCH -A m4293 # Sparsitute project (A Mathematical Institute for Sparse Computations in Science and Engineering)

#SBATCH -t 0:10:00

#SBATCH -N 2
#SBATCH -J nystrom
#SBATCH -o slurm.nystrom.o%j

# https://docs.nersc.gov/systems/perlmutter/architecture/
#SYSTEM=perlmutter-cpu
SYSTEM=perlmutter-gpu-cpu
#SYSTEM=perlmutter-gpu
N_NODE=2

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
	# https://docs.nersc.gov/systems/perlmutter/architecture/#gpu-nodes

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

N=50000
R=500

MATMUL1_P1=$N_PROC
MATMUL1_P2=1
MATMUL1_P3=1

MATMUL2_P1=$N_PROC
MATMUL2_P2=1
MATMUL2_P3=1

#for ALG in nystrom-1d-noredist-1d nystrom-1d-redist-1d
#for ALG in nystrom-1d-noredist-1d
#for ALG in nystrom-1d-redist-1d
for ALG in nystrom-2d-noredist-1d-redundant
do
    #for IMPL in cpp python
    for IMPL in cpp
    #for IMPL in python 
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
        elif [[ $ALG == nystrom-2d-noredist-1d-* ]]; then
            # https://stackoverflow.com/questions/2172352/in-bash-how-can-i-check-if-a-string-begins-with-some-value
            # Use these setting for all variants of 2d-1d
            if [ "$N_PROC" -eq 4 ]; then
                MATMUL1_P1=2
                MATMUL1_P2=2
                MATMUL1_P3=1

                MATMUL2_P1=1
                MATMUL2_P2=$N_PROC
                MATMUL2_P3=1
            elif [ "$N_PROC" -eq 8 ]; then
                MATMUL1_P1=4
                MATMUL1_P2=2
                MATMUL1_P3=1

                MATMUL2_P1=1
                MATMUL2_P2=$N_PROC
                MATMUL2_P3=1
            elif [ "$N_PROC" -eq 16 ]; then
                MATMUL1_P1=4
                MATMUL1_P2=4
                MATMUL1_P3=1

                MATMUL2_P1=1
                MATMUL2_P2=$N_PROC
                MATMUL2_P3=1
            elif [ "$N_PROC" -eq 32 ]; then
                MATMUL1_P1=8
                MATMUL1_P2=4
                MATMUL1_P3=1

                MATMUL2_P1=1
                MATMUL2_P2=$N_PROC
                MATMUL2_P3=1
            elif [ "$N_PROC" -eq 64 ]; then
                MATMUL1_P1=8
                MATMUL1_P2=8
                MATMUL1_P3=1

                MATMUL2_P1=1
                MATMUL2_P2=$N_PROC
                MATMUL2_P3=1
            elif [ "$N_PROC" -eq 128 ]; then
                MATMUL1_P1=16
                MATMUL1_P2=8
                MATMUL1_P3=1

                MATMUL2_P1=1
                MATMUL2_P2=$N_PROC
                MATMUL2_P3=1
            fi
        fi
        
        STDOUT_FILE=$SCRATCH/nystrom/nystrom_benchmarking/"$ALG"_"$IMPL"_"$N_NODE"_"$N_PROC"_"$THREAD_PER_PROC"_"$N"_"$R"_"$MATMUL1_P1"x"$MATMUL1_P2"x"$MATMUL1_P3"_"$MATMUL2_P1"x"$MATMUL2_P2"x"$MATMUL2_P3"
        echo $STDOUT_FILE

        if [ "$SYSTEM" == "perlmutter-cpu" ]; then
            PY=$HOME/Codes/nystrom-distributed/nystrom.py
            BIN=$HOME/Codes/nystrom-distributed/build_cpu/c_matmul/nystrom
        elif [ "$SYSTEM" == "perlmutter-gpu" ]; then
            PY=$HOME/Codes/nystrom-distributed/nystrom.py
            BIN=$HOME/Codes/nystrom-distributed/build_cpu/c_matmul/nystrom
        elif [ "$SYSTEM" == "perlmutter-gpu-cpu" ]; then
            PY=$HOME/Codes/nystrom-distributed/nystrom.py
            BIN=$HOME/Codes/nystrom-distributed/build_cpu/c_matmul/nystrom
        fi

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
