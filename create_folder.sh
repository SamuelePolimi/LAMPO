#!/usr/bin/env bash

###############################################################################
# SLURM Configurations

#SBATCH -J lampo_folder

#SBATCH --mail-user=samuele.tosatto@tu-darmstadt.de
#SBATCH --mail-type=NONE

#SBATCH -t 00:1:00
#SBATCH -n 1
#SBATCH --mem-per-cpu=100
#SBATCH -c 1
#SBATCH -o /work/scratch/%u/lampo/%j-out.txt
#SBATCH -e /work/scratch/%u/lampo/%j-err.txt


###############################################################################

# Your PROGRAM call starts here
echo "Starting Job $SLURM_JOB_ID, Index $SLURM_ARRAY_TASK_ID"

# Activate virtual environment
source activate rl

python experiment_organizer.py $@ --load --save --slurm

deactivate

