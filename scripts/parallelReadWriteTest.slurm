#!/bin/bash
#SBATCH --job-name="rahmm224"         # Name that appears in queue
#SBATCH --partition=yangGrp           # Resource group (small/large/gpu)
#SBATCH --account=yangGrp             # Research group
#SBATCH --nodes=1                     # Number of Nodes
#SBATCH --cpus-per-task=1             # Number of CPU Cores
#SBATCH --mem=4G                     # Requested memory (CPU)
#SBATCH --time=00-00:10:00            # Job duration in DD-HH:MM:SS
#SBATCH --mail-user=rahmm224@wfu.edu  # Email address to update on Job status
#SBATCH --mail-type=END,FAIL          # Updates to provide mail-user

#SBATCH --output="/deac/csc/ballardGrp/rahmm224/new_nystrom/logs/job_name-%j.o"
#SBATCH --error="/deac/csc/ballardGrp/rahmm224/new_nystrom/logs/job_name-%j.e"

module load mpi/intel/2021.10.0
conda activate /home/rahmm224/Summer2025Env
mpiexec -n 4 python "/deac/csc/ballardGrp/rahmm224/new_nystrom/nystrom-distributed/parallelReadWrite.py"
# mpiexec -n 2 python $PY
conda deactivate