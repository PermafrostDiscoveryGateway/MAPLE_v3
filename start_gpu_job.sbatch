#!/bin/bash -l

# USAGE INSTRUCTIONS:
# sbatch <this_filename>

#SBATCH --job-name=maplev3
#SBATCH --partition=gpuMI100x8
#SBATCH --account=bbou-delta-gpu
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --mem=650g
#SBATCH --cpus-per-task=4

# export local environment variables to the job
#SBATCH --export=ALL

aptainer run -t -i -v /scratch/bbou/toddn/MAPLE_v3/data:/data  maplev3:latest