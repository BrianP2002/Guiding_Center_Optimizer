#!/bin/bash
# Run from the repository root. Account and partition: export SBATCH_ACCOUNT / SBATCH_PARTITION or pass --account / --partition (see common.sh).
#SBATCH --job-name=gc_off_fd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --array=0-199
#SBATCH --output=logs/%x_%A_%a.out
# defaults: the original run (draw 581, eps_t, 5 points, 40 copies); e.g.
#   FD_PAR=kappa FD_H=0.01 FD_NPOINT=7 FD_NCOPY=60 FD_OUT=results/gc_official_fd_kappa sbatch --array=0-419%100 experiments/slurm/gc_official_fd.sbatch
# another draw: GC_SEED=835 GC_ICS_ARGS="--ics-file results/gc_official_regime2_seed835_ics/ics.json" FD_TSPIN=6000 ...
source experiments/slurm/common.sh
$PY experiments/scripts/gc_official_fd.py $SLURM_ARRAY_TASK_ID --par ${FD_PAR:-eps_t} --h ${FD_H:-0.01} --T ${FD_T:-200000} \
    --npoint ${FD_NPOINT:-5} --ncopy ${FD_NCOPY:-40} --outdir ${FD_OUT:-results/gc_official_fd} \
    --seed ${GC_SEED:-581} --t-spinup ${FD_TSPIN:-6000} ${GC_ICS_ARGS:-}
