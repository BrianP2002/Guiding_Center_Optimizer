#!/bin/bash
# Run from the repository root. Account and partition: export SBATCH_ACCOUNT / SBATCH_PARTITION or pass --account / --partition (see common.sh).
#SBATCH --job-name=gc_off_cand
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:40:00
#SBATCH --array=0-143
#SBATCH --output=logs/%x_%A_%a.out
source experiments/slurm/common.sh
$PY experiments/scripts/gc_official_candidates.py $SLURM_ARRAY_TASK_ID --T 200000
