#!/bin/bash
# Run from the repository root. Account and partition: export SBATCH_ACCOUNT / SBATCH_PARTITION or pass --account / --partition (see common.sh).
#SBATCH --job-name=lorenz_ens
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --array=0-31
#SBATCH --output=logs/%x_%A_%a.out
source experiments/slurm/common.sh
$PY experiments/scripts/lorenz_ensemble.py $SLURM_ARRAY_TASK_ID
