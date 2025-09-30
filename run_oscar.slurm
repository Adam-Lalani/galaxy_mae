#!/bin/bash

#SBATCH -p gpu                # Specify the 'gpu' partition
#SBATCH -N 1
#SBATCH --gres=gpu:2          # Request 2 GPUs for DataParallel training
#SBATCH -n 1                  # Number of tasks (almost always 1 for single-node jobs)
#SBATCH -c 8                  # Number of CPUs per task (2x num_workers for data loading)
#SBATCH --mem=64G             # Request 32GB of memory. Increase if you get memory errors.
#SBATCH -t 36:00:00           # Set a time limit of 36 hours (HH:MM:SS)
#SBATCH -J galaxy-mae         # A descriptive name for your job
#SBATCH -o slurm-%j.out       # Name of the file to redirect standard output
#SBATCH -e slurm-%j.err       # Name of the file to redirect standard error

# --- Environment Setup ---
echo "Job started on $(hostname) at $(date)"
echo "Loading required modules..."

module load anaconda
module load cuda

# --- Job Execution ---
echo "Navigating to submission directory: $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR

echo "Starting Python script for MAE pre-training..."
/users/allalani/miniconda3/envs/csci1470/bin/python -u main_large.py

echo "Job finished at $(date)"
