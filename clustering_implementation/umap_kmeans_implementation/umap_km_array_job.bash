#!/bin/bash
#SBATCH --partition=short
#SBATCH --job-name=umap_km_array
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=job_%A_num_%a.output
#SBATCH --error=job_%A_num_%a.error
#SBATCH --array=4

#####Use after training##############
##SBATCH -p short

## Load the anaconda package
module load anaconda3/2022.05

## activate the environment
source activate nemsis_project

echo "UMAP -> K-Means Clustering Algorithm - Job array ID: $SLURM_ARRAY_JOB_ID , sub-job $SLURM_ARRAY_TASK_ID is running!"

cd umap_km$SLURM_ARRAY_TASK_ID
python umap_km$SLURM_ARRAY_TASK_ID.py
