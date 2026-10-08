#!/bin/bash
# Run from the repository root. Account and partition: export SBATCH_ACCOUNT / SBATCH_PARTITION or pass --account / --partition (see common.sh).
#SBATCH --job-name=gc_lyap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --array=0-65
#SBATCH --output=logs/%x_%A_%a.out
source experiments/slurm/common.sh
$PY experiments/scripts/gc_lyapunov.py $SLURM_ARRAY_TASK_ID --T 1000
