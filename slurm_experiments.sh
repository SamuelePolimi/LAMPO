#!/usr/bin/env bash

###############################################################################
# SLURM Configurations

#SBATCH -J lampo

# Define here the Job Array 1-8%2 (1-start index; 8-end index; 2-(optional) number of elements that may run in parallel)
#SBATCH -a 0-10

#SBATCH --mail-user=samuele.tosatto@tu-darmstadt.de
#SBATCH --mail-type=NONE

#SBATCH -t 025:00:00
# -n roughly the number of cpu cores
#SBATCH -n 50
# --mem-per-cpu max memory per cpu core
#SBATCH --mem-per-cpu=2000
# -c number of cpus cores per task. This is only important for mpi jobs. Set to 1 for non-mpi jobs
#SBATCH -c 5
# Do NOT use avx2 if you do not require it! There are way less nodes with avx2
#SBATCH -C avx

#SBATCH -o /work/scratch/%u/lampo/%A_%a-out.txt
#SBATCH -e /work/scratch/%u/lampo/%A_%a-err.txt

###############################################################################
# Your PROGRAM call starts here
echo "Starting Job $SLURM_JOB_ID, Index $SLURM_ARRAY_TASK_ID"

# Activate virtual environment
source rl

python3 experiment.py $@ --id $SLURM_ARRAY_TASK_ID --load

deactivate
