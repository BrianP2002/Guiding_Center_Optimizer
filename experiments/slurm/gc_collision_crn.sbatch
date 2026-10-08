#!/bin/bash
# Run from the repository root. Account and partition: export SBATCH_ACCOUNT / SBATCH_PARTITION or pass --account / --partition (see common.sh).
#SBATCH --job-name=gc_coll_crn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=01:00:00
#SBATCH --array=0-31%100
#SBATCH --output=logs/%x_%A_%a.out
# pathwise derivative with common random numbers: 4 collision frequencies x 8 blocks of 1000 paths = 32 tasks
# DT (default 0.1), TS (default 300,1000,3000,10000), NPER, NBLOCK, OUT, COUPLING (rotation|gradient)
source experiments/slurm/common.sh
$PY experiments/scripts/gc_collision.py crn $SLURM_ARRAY_TASK_ID --dt ${DT:-0.1} --Ts ${TS:-300,1000,3000,10000} --nper ${NPER:-1000} --nblock ${NBLOCK:-8} --outdir ${OUT:-results/gc_collision_crn} --coupling ${COUPLING:-rotation}
