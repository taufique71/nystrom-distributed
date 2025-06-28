#!/bin/bash -l
#SBATCH --job-name="rahmm224"           
#SBATCH --partition=small
#SBATCH --account=ballardGrp               
#SBATCH --tasks-per-node=44
#SBATCH --nodes=1                  
#SBATCH --time=0-00:02:00          
#SBATCH --mem=1GB                   
#SBATCH --mail-user=rahmm224@wfu.edu  
#SBATCH --mail-type=FAIL,END  

#SBATCH --output="/deac/csc/ballardGrp/rahmm224/jobs_oe/job_name-%j.o"
#SBATCH --error="/deac/csc/ballardGrp/rahmm224/jobs_oe/job_name-%j.e"

module load mpi/intel/2021.10.0
conda activate /home/rahmm224/Summer2025Env
PY="/home/rahmm224/nystrom-distributed-3/nystrom.py"
alg="nystrom-1d-noredist-1d"
mpiexec -n 2 python $PY -n 10000 -r 1000 -p1 2 -p2 1 -p3 1 -alg $alg
conda deactivate