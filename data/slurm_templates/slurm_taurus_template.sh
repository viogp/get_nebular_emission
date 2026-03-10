#!/bin/sh
# Template for SLURM job submission - GNE processing

#SBATCH -A 16cores
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=__GNE_JOB_NAME__
#SBATCH --error=__GNE_LOG_DIR__/__GNE_JOB_NAME__.err
#SBATCH --output=__GNE_LOG_DIR__/__GNE_JOB_NAME__.out
##SBATCH --mem=600000
#SBATCH --partition=all
#SBATCH --exclude=epi
#SBATCH --time=30-00:00:00
#
export OMP_NUM_THREADS=16
srun python << 'EOF_PYTHON_SCRIPT'
__GNE_PARAM_CONTENT__
EOF_PYTHON_SCRIPT
