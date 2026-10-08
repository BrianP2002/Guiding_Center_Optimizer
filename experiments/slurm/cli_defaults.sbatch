#!/bin/bash
# Run from the repository root. Account and partition: export SBATCH_ACCOUNT / SBATCH_PARTITION or pass --account / --partition (see common.sh).
#SBATCH --job-name=gc_cli
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
# runs legacy/app_gc.py and legacy/app_gc_opt.py with their default settings, once per parameter
source experiments/slurm/common.sh
PARS=(a0 a1 iota lam G)
P=${PARS[$SLURM_ARRAY_TASK_ID]}
$PY legacy/app_gc.py --par $P --outdir results/cli/$P
$PY legacy/app_gc_opt.py --par $P --outdir results/cli/$P
