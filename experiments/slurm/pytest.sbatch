#!/bin/bash
# Run from the repository root. Account and partition: export SBATCH_ACCOUNT / SBATCH_PARTITION or pass --account / --partition (see common.sh).
#SBATCH --job-name=gc_pytest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
# optional: PYTEST_MARK="not slow" (marker expression), PYTEST_FILES="tests/test_a.py tests/test_b.py" (default: tests)
source experiments/slurm/common.sh
$PY -m pytest -v --durations=10 -p no:cacheprovider ${PYTEST_MARK:+-m "$PYTEST_MARK"} ${PYTEST_FILES:-tests}
