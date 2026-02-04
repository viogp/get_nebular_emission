#!/bin/sh
# Template for SLURM job submission - GNE processing
# Placeholders: JOB_NAME, PARAM_CONTENT

#SBATCH -A 16cores
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=JOB_NAME
#SBATCH --error=output/JOB_NAME.err
#SBATCH --output=output/JOB_NAME.out
##SBATCH --mem=600000
#SBATCH --partition=all
#SBATCH --time=30-00:00:00
#
export OMP_NUM_THREADS=16
srun python -c '
PARAM_CONTENT
'
