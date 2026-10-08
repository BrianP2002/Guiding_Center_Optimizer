#!/bin/bash
# Template: a Slurm array of independent NILSS orbits, then collect them.
#
#   1. Install the package in the environment the jobs will use:  pip install -e .   (from the repository root)
#   2. Say where the jobs run -- either flags on the command line
#          sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION examples/06_cluster_ensemble/submit.sbatch
#      or environment variables that sbatch reads:  export SBATCH_ACCOUNT=...  SBATCH_PARTITION=...
#      and, if the Python of that environment is not the first `python` on the PATH of the job,  export NILSS_PYTHON=/path/to/python
#   3. Submit from the repository root (the paths below are relative to it). 32 orbits, at most 8 at a time. The log directory has
#      to exist before the submission (Slurm opens the log file before the script starts):
#          mkdir -p logs runs
#          sbatch examples/06_cluster_ensemble/submit.sbatch
#   4. When the array has finished:   python examples/06_cluster_ensemble/collect.py runs
#
#SBATCH --job-name=nilss_member
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --array=0-31%8
#SBATCH --output=logs/%x_%A_%a.out

mkdir -p logs runs
# one core per task: JAX's CPU thread pools would only compete with the other tasks of the node
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
${NILSS_PYTHON:-python} examples/06_cluster_ensemble/member.py --outdir runs --T ${NILSS_T:-200}
