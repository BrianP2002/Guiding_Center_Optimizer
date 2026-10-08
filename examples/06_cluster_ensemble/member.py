"""One member of a cluster ensemble: one NILSS orbit, saved as run_XXXX.npz.

The orbit is chosen by --index or, if absent, by the environment variable SLURM_ARRAY_TASK_ID, so that a Slurm array of N
tasks gives N independent orbits (initial condition and tangent seed both depend on the index). Collect the files with
collect.py. This template runs Lorenz 63; replace SYSTEM below by your own NILSS object and initial conditions. rhs and J
should be top-level functions of an importable module (not needed here, since each task runs one orbit in its own process).

    python examples/06_cluster_ensemble/member.py --outdir runs --index 3 [--T 200] [--fast]
"""
import argparse
import os

from nilss_jax import NILSS
from nilss_jax.systems import lorenz63


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='runs')
    ap.add_argument('--index', type=int, default=None, help='member index (default: $SLURM_ARRAY_TASK_ID)')
    ap.add_argument('--T', type=float, default=200.0)
    ap.add_argument('--fast', action='store_true', help='T = 30')
    args = ap.parse_args()
    index = args.index if args.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
    T = 30.0 if args.fast else args.T

    # ---- SYSTEM: replace these three lines -------------------------------------------------------------------------
    nilss = NILSS(lorenz63.rhs, lorenz63.J, ('rho', 'sigma', 'beta'), dt=0.005, T_seg=0.5, nus=1)
    p, u0 = lorenz63.DEFAULTS, lorenz63.initial_condition(index)
    # -----------------------------------------------------------------------------------------------------------------
    res = nilss.run(p, u0, T=T, T_spinup=20.0, seed=index)

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f'run_{index:04d}.npz')
    res.save(path)
    print(f'member {index}: {res!r} -> {path}')


if __name__ == '__main__':
    main()
