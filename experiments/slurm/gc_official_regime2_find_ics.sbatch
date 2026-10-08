#!/bin/bash
# Run from the repository root. Account and partition: export SBATCH_ACCOUNT / SBATCH_PARTITION or pass --account / --partition (see common.sh).
#SBATCH --job-name=gc_r2_ics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --array=0-95%48
#SBATCH --output=logs/%x_%A_%a.out
# sea initial conditions of a draw: R2_SEED=835 sbatch experiments/slurm/gc_official_regime2_find_ics.sbatch
# then: python experiments/scripts/gc_official_regime2_find_ics.py --collect --seed 835
source experiments/slurm/common.sh
$PY experiments/scripts/gc_official_regime2_find_ics.py $SLURM_ARRAY_TASK_ID --seed ${R2_SEED:-835} --T ${R2_T:-50000}
