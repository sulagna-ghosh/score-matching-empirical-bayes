#!/bin/bash

# Parameters
#SBATCH --error=/home/amberlee0516/score-matching-empirical-bayes/submitit_log/%j/%j_0_log.err
#SBATCH --gpus-per-node=1
#SBATCH --job-name=EB_g
#SBATCH --mem=24GB
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/home/amberlee0516/score-matching-empirical-bayes/submitit_log/%j/%j_0_log.out
#SBATCH --partition=general
#SBATCH --signal=USR2@90
#SBATCH --time=701
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /home/amberlee0516/score-matching-empirical-bayes/submitit_log/%j/%j_%t_log.out --error /home/amberlee0516/score-matching-empirical-bayes/submitit_log/%j/%j_%t_log.err /home/amberlee0516/miniconda3/envs/nn_env/bin/python3 -u -m submitit.core._submit /home/amberlee0516/score-matching-empirical-bayes/submitit_log/%j
