#!/bin/bash

SBATCH -a 1-5
## Controls the number of replications of the job that are run
## The specific ID of the replication can be accesses with the environment variable $SLURM_ARRAY_TASK_ID
## Can be used for seeding
#SBATCH -n 1
## ALWAYS leave this value to 1. This is only used for MPI, which is not supported now.
#SBATCH -c 1
## Specify the number of cores. The maximum is 32. Please set a reasonable value to not block the cluster.
SBATCH --gres=gpu:rtx2080:1
## Leave this if you want to use a GPU per job. Please REMOVE it if you do not need it.
SBATCH --mem 3000
## Here you can control the amount of memory that will be allocated for you job. To set this,
## you should run the programm on your local computer first and observe the memory it consumes.
SBATCH -t 24:00:00
## Do not allocate more time than you REALLY need.

SBATCH -o /home/tosatto/logs/%A_%a.out
SBATCH -e /home/tosatto/logs/%A_%a.err
## Make sure to create the logs directory /home/user/Documents/projects/prog/logs, BEFORE launching the jobs.

eval "$(/home/tosatto/miniconda3/bin/conda shell.bash hook)"
conda activate rl
python deep_experiment_organizer.py sparse --task_name reach_target-state-v0 --algorithm SAC --timestep 500000 --id_start $SLURM_ARRAY_TASK_ID  --n_runs 1 --slurm
