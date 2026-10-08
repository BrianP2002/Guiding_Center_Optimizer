# Many cores and clusters

The unit of work is one orbit on one core. For the state dimensions this package is meant for (3 to a few tens) the compiled
integrator does not profit from several threads, so the way to use many cores is many orbits at once: worker processes on one
machine, or a Slurm array with one orbit per task.

## One CPU core per orbit: settings

JAX's CPU runtime and the BLAS libraries start thread pools; with one orbit per core they only compete with the other orbits. Set, before
Python starts,

```bash
export JAX_PLATFORMS=cpu                       # the package is written and tested for CPU; the small tangent problems gain nothing from a GPU
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
```

## Worker processes on one machine

```python
ens = run_ensemble(nilss, p, u0s, T=300.0, T_spinup=20.0, workers=8)
```

starts 8 processes with the `spawn` method (a fork of a process that already runs JAX is not safe). Each worker imports JAX and compiles
the integrator once (about 5 to 15 s), then runs its share of the orbits; the results are identical to those of the serial run
(`tests/test_ensemble.py` checks this to 1e-10), so `workers` only pays off for runs of more than about a minute each. Two requirements:

* `rhs`, `J` (and `invariant`) must be functions defined at the top level of a module, or of the script that is run: they are sent to
  the workers by reference. A lambda or a closure raises `ValueError: workers > 1 needs rhs, J and invariant to be top-level functions ...`.
  `examples/03_custom_system_lorenz96.py` runs with `--workers 2` for this reason.
* a script that calls `run_ensemble(..., workers>1)` needs the `if __name__ == '__main__':` guard, because each worker imports it.

`run_ensemble(..., executor=...)` accepts a `concurrent.futures.Executor` instead (only the standard library's `ProcessPoolExecutor` has been tried); the
same requirements on the functions apply.

## Slurm arrays

[`examples/06_cluster_ensemble/`](../examples/06_cluster_ensemble) is a template for one orbit per array task:

* `member.py` builds the problem, takes the orbit index from `--index` or `$SLURM_ARRAY_TASK_ID` (initial condition and tangent seed both
  depend on it) and writes `run_XXXX.npz` with `NILSSResult.save`;
* `submit.sbatch` is the array script: one core, 2 GB, the thread settings above, `--array=0-31%8`. It contains no account, partition or
  path: give them as `sbatch --account=... --partition=...`, or `export SBATCH_ACCOUNT=... SBATCH_PARTITION=...`, and set
  `NILSS_PYTHON` if the Python of the environment is not the first on the `PATH` of the job; submit from the repository root;
* `collect.py` reads the directory with `load_ensemble` and prints the statistics and the reliability report.

To use your own system, replace the three lines marked `SYSTEM` in `member.py`. The array can be extended later (more indices) and
collected again: the statistics and the report are computed from whatever `run_*.npz` files are present.

## What it costs

Measured on one core of an Intel Xeon Gold 6230R (2.1 GHz), after the compilation, for the integration of an orbit with its tangents (RK4):

| system | dt | T_seg | parameters | seconds per 1000 time units |
| --- | --- | --- | --- | --- |
| Lorenz 63 | 0.005 | 0.5 | 1 | 1.3 |
| Lorenz 63 | 0.005 | 0.5 | 3 | 1.2 |
| guiding-center flow (4 variables, energy constraint) | 0.01 | 200 | 1 | 1.1 |
| guiding-center flow | 0.01 | 200 | 4 | 1.4 |

A guiding-center orbit of T = 2e5 with four parameters therefore takes about 5 minutes (the studies' runs of 6000 spin-up plus 2e5 took 5 to 8
minutes on a busy node), and a Lorenz orbit of T = 1000 about 1.3 s. Cost is proportional to the number of steps; it grows with the number of
parameters much more slowly than proportionally (three Lorenz parameters cost the same as one, four guiding-center parameters 30% more than
one). The first call of `NILSS.run` compiles for roughly 5 to 15 s; later calls on the same object do not.

## Memory

The tangent solutions are never stored: the integrator carries the current state, `nus` homogeneous and one inhomogeneous tangent per parameter,
and accumulates the integrals it needs, so the memory does not depend on the number of time steps. Peak resident memory of the whole Python
process, measured over runs of increasing length with the same settings:

| system | T = 500 | 2000 | 8000 | 32000 |
| --- | --- | --- | --- | --- |
| Lorenz 63, 1 parameter, T_seg = 0.5 (1000, 4000, 16000 segments) | 183 MB | 282 MB | 676 MB | |
| guiding-center flow, 1 parameter, T_seg = 200 | | 195 MB | 196 MB | 196 MB |
| guiding-center flow, 4 parameters, T_seg = 200 | | 205 MB | 206 MB | 206 MB |

What does grow is a small record per segment (the matrices of the least-squares problem and a few vectors, kept for the whole run, and, in the
current implementation, the JAX buffers they are read from), about 30 KB per segment in the Lorenz measurement. A run with `T / T_seg` of a few
thousand segments at most (the guiding-center studies used 1000) is then cheap whatever the number of steps; with very short segments the
record, not the integration, dominates the memory, so lengthen `T_seg` (section 4 of [`usage.md`](usage.md) says how long it may be).
