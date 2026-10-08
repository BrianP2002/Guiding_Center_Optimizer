# Sourced by the sbatch scripts of experiments/slurm. Submit them from the repository root, after
#   mkdir -p logs                                    (the --output directory has to exist when the job is submitted)
#   export SBATCH_ACCOUNT=<account> SBATCH_PARTITION=<partition>        (or pass --account / --partition to sbatch)
#   export NILSS_PYTHON=/path/to/env/bin/python      (the python of an environment where `pip install -e .` was run)
PY=${NILSS_PYTHON:-python}
export PY
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs results
