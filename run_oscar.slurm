#!/bin/bash

#SBATCH -p gpu                # Specify the 'gpu' partition
#SBATCH --gres=gpu:2          # Request 1 GPU. Change to 'gpu:2' for two GPUs.
#SBATCH -n 1                  # Number of tasks (almost always 1 for single-node jobs)
#SBATCH -c 4                  # Number of CPUs per task. 8 is a good starting point for data loading.
#SBATCH --mem=64G             # Request 32GB of memory. Increase if you get memory errors.
#SBATCH -t 36:00:00           # Set a time limit of 12 hours (HH:MM:SS)
#SBATCH -J galaxy-mae         # A descriptive name for your job
#SBATCH -o slurm-%j.out       # Name of the file to redirect standard output
#SBATCH -e slurm-%j.err       # Name of the file to redirect standard error

# --- Environment Setup ---
echo "Job started on $(hostname) at $(date)"
echo "Loading required modules..."

module load anaconda
module load cuda

# Activate your specific conda environment
echo "Activating conda environment..."
conda activate csci1470  

# --- Job Execution ---
echo "Navigating to submission directory: $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR

echo "Starting Python script for MAE pre-training..."

# The '-u' flag is for unbuffered Python output, which is helpful for seeing
# logs in the .out file in real-time.
python -u main_large.py

echo "Job finished at $(date)"
```