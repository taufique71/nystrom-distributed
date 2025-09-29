#!/bin/bash -l
#SBATCH --job-name="rahmm224"           
#SBATCH --partition=small
#SBATCH --account=ballardGrp               
#SBATCH --tasks-per-node=44
#SBATCH --nodes=1                  
#SBATCH --time=0-01:01:00          
#SBATCH --mem=128GB                   
#SBATCH --mail-user=rahmm224@wfu.edu  
#SBATCH --mail-type=FAIL,END  

#SBATCH --output="/deac/csc/ballardGrp/rahmm224/nystrom/nystrom-distributed/logs/job_name-%j.o"
#SBATCH --error="/deac/csc/ballardGrp/rahmm224/nystrom/nystrom-distributed/logs/job_name-%j.e"

module load mpi/intel/2021.10.0
conda activate /home/rahmm224/Summer2025Env
PY="/deac/csc/ballardGrp/rahmm224/new_nystrom/nystrom-distributed/CIFAR10bianry.py"
path="./data"
kernel="linear"
python $PY -path $path -kernel $kernel

conda deactivate