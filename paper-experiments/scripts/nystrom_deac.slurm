#!/bin/bash -l
#SBATCH --job-name="rahmm224"           
#SBATCH --partition=small
#SBATCH --account=ballardGrp               
#SBATCH --tasks-per-node=44
#SBATCH --nodes=1                  
#SBATCH --time=0-00:01:00          
#SBATCH --mem=128GB                   
#SBATCH --mail-user=rahmm224@wfu.edu  
#SBATCH --mail-type=FAIL,END  

#SBATCH --output="/deac/csc/ballardGrp/rahmm224/nystrom/nystrom-distributed/logs/job_name-%j.o"
#SBATCH --error="/deac/csc/ballardGrp/rahmm224/nystrom/nystrom-distributed/logs/job_name-%j.e"

module load mpi/intel/2021.10.0
conda activate /home/rahmm224/Summer2025Env
PY="/deac/csc/ballardGrp/rahmm224/nystrom/nystrom-distributed/nystrom.py"
# PY="/deac/csc/ballardGrp/rahmm224/testAlltoAll.py"
alg="nystrom-2d-redist-1d"
mpiexec -n 8 python $PY -n 50 -r 15 -matmul1p1 1 -matmul1p2 2 -matmul1p3 4 -matmul2p1 8 -matmul2p2 1 -matmul2p3 1 -alg $alg
# mpiexec -n 2 python $PY
conda deactivate