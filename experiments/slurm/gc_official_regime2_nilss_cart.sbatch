#!/bin/bash
# Run from the repository root. Account and partition: export SBATCH_ACCOUNT / SBATCH_PARTITION or pass --account / --partition (see common.sh).
#SBATCH --job-name=gc_r2_cart
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=01:30:00
#SBATCH --array=0-35%36
#SBATCH --output=logs/%x_%A_%a.out
# NILSS in near-axis Cartesian coordinates (same options as experiments/slurm/gc_official_nilss.sbatch), e.g. draw 835:
#   GC_SEED=835 GC_ICS_ARGS="--ics-file results/gc_official_regime2_seed835_ics/ics.json" NILSS_PARS=eps_2,eps_h,kappa,iota \
#   NILSS_LAYOUT=main_nus2 NILSS_NMAIN=30 NILSS_NNUS2=6 NILSS_OUT=results/gc_official_nilss_cart_seed835 sbatch --array=0-35%36 experiments/slurm/gc_official_regime2_nilss_cart.sbatch
source experiments/slurm/common.sh
$PY experiments/scripts/gc_official_regime2_nilss_cart.py $SLURM_ARRAY_TASK_ID --T ${NILSS_T:-200000} --outdir ${NILSS_OUT:-results/gc_official_nilss_cart} \
    --seed ${GC_SEED:-581} --pars ${NILSS_PARS:-eps_t,eps_m,kappa,iota} --layout ${NILSS_LAYOUT:-draw581} \
    --n-main ${NILSS_NMAIN:-60} --n-nus2 ${NILSS_NNUS2:-10} --t-spinup ${NILSS_TSPIN:-6000} ${GC_ICS_ARGS:-}
