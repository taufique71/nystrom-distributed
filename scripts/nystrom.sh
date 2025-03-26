#!/bin/bash -l

#SBATCH -q debug
#SBATCH -C cpu
##SBATCH -A m4293 # Sparsitute project (A Mathematical Institute for Sparse Computations in Science and Engineering)

#SBATCH -t 0:10:00

#SBATCH -N 1
#SBATCH -J nystrom 
#SBATCH -o slurm.nystrom.o%j

PY=/global/u1/t/taufique/Codes/nystrom-distributed/nystrom.py

module load python
module load pytorch
export MASTER_ADDR=$(hostname)
export MASTER_PORT=4321
srun -N 1 -n 24 -c 2 python $PY -n 1000 -r 20 -p1 4 -p2 3 -p3 2 -n1 1150 -n2 35 -n3 25
