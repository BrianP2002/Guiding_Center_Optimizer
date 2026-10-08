#!/bin/bash
# Run from the repository root. Account and partition: export SBATCH_ACCOUNT / SBATCH_PARTITION or pass --account / --partition (see common.sh).
#SBATCH --job-name=gc_pathwise
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:55:00
#SBATCH --array=0-0
#SBATCH --output=logs/%x_%A_%a.out
# One array task = one block of initial conditions of experiments/scripts/gc_pathwise.py. The array range is set at submission, the mode and
# its options through PW_ARGS, for example
#   PW_ARGS="--mode pathwise --ens E1 --n 4000 --per-task 50" sbatch --array=0-79%60 experiments/slurm/gc_pathwise.sbatch
#   PW_ARGS="--mode fd --ens E1 --n 4000 --per-task 25"       sbatch --array=0-15%16 experiments/slurm/gc_pathwise.sbatch
#   PW_ARGS="--mode e2ics"                                     sbatch experiments/slurm/gc_pathwise.sbatch
source experiments/slurm/common.sh
$PY experiments/scripts/gc_pathwise.py $SLURM_ARRAY_TASK_ID $PW_ARGS ${PW_OUT:+--outdir $PW_OUT}
