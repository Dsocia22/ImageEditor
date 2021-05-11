#!/bin/bash
# specify a partition
#SBATCH --partition=dggpu
# Request nodes
#SBATCH --nodes=1
# Request some processor cores
#SBATCH --ntasks=2
# Request GPUs
#SBATCH --gres=gpu:1
# Request memory 
#SBATCH --mem=20G
# Maximum runtime of 10 minutes
#SBATCH --time=30:00
# Name of this job
#SBATCH --job-name=Gen
# Output of this job, stderr and stdout are joined by default
# %x=job-name %j=jobid
#SBATCH --output=output/%x_%j.out

# Allow for the use of conda activate
source ~/.bashrc

# Move to submission directory
# Should be ~/scratch/deepgreen-keras-tutorial/src
cd ${SLURM_SUBMIT_DIR}

# your job execution follows:
conda activate RL2
time python ~/scratch/DS2_Photo/GeneratorTraining.py --epochs 2 
