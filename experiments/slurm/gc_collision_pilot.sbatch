#!/bin/bash
# Run from the repository root. Account and partition: export SBATCH_ACCOUNT / SBATCH_PARTITION or pass --account / --partition (see common.sh).
#SBATCH --job-name=gc_coll_pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
# PILOT=timing|dt|frac|lyap (frac: --array=0-3, one collision frequency per task), DT, NPATH, COUPLING (rotation|gradient)
source experiments/slurm/common.sh
$PY experiments/scripts/gc_collision.py pilot ${PILOT:-timing} ${SLURM_ARRAY_TASK_ID:-0} --dt ${DT:-0.05} --n ${NPATH:-500} --coupling ${COUPLING:-rotation}
