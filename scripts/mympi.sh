#!/bin/bash -l

#SBATCH -q debug
#SBATCH -C cpu
##SBATCH -A m1982 # Aydin's project (PASSION: Scalable Graph Learning for Scientific Discovery)
##SBATCH -A m2865 # Exabiome project
##SBATCH -A m4012 # Ariful's project (GraphML: Intelligent Premitives for Graph Machine Learning)
##SBATCH -A m4293 # Sparsitute project (A Mathematical Institute for Sparse Computations in Science and Engineering)

#SBATCH -t 0:10:00

#SBATCH -N 1
#SBATCH -J mympi
#SBATCH -o slurm.mympi.o%j

PY=/global/u1/t/taufique/Codes/nystrom-distributed/mympi.py

module load python
module load pytorch
export MASTER_ADDR=$(hostname)
export MASTER_PORT=4321
srun -N 1 -n 4 -c 2 python $PY
