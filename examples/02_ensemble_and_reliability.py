"""An ensemble of NILSS runs, its statistics and the reliability report, on Lorenz 63.

One NILSS run follows one trajectory. For anything that is not clearly well behaved, report the statistics of many orbits
together with the diagnostics: run_ensemble() runs one orbit per row of u0s, EnsembleResult gives mean +- SEM, the median
with a bootstrap interval and quantiles, and report() runs the checks of nilss_jax.diagnostics (resolved exponent,
ergodicity, tails of the shadowing direction, scaling of the error, and -- with a second ensemble that uses one more
homogeneous tangent on the same orbits -- consistency in nus).

Expect (16 orbits of T = 300): d<z>/d(rho) = 1.018 +- 0.0004 (the SEM is the statistical error; finite differences give 1.00).
Every check passes for rho (verdict OK with --params rho). For sigma the per-segment contributions have a heavier tail (Hill
index about 2.5), which the report flags as CAUTION; this is the parameter for which NILSS is furthest from finite differences
on Lorenz 63. The --fast run (8 orbits of T = 60) is a smoke test with little statistical power.

    python examples/02_ensemble_and_reliability.py [--fast] [--params rho sigma] [--runs 16] [--workers 4]
"""
import argparse

import numpy as np

from nilss_jax import NILSS, run_ensemble
from nilss_jax.systems import lorenz63


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--runs', type=int, default=None, help='number of orbits (default 16, 8 with --fast)')
    ap.add_argument('--params', nargs='+', default=['rho', 'sigma'], choices=['rho', 'sigma', 'beta'], help='parameters to differentiate')
    ap.add_argument('--workers', type=int, default=1, help='processes (rhs and J must be importable top-level functions)')
    args = ap.parse_args()
    T, runs = (60.0, args.runs or 8) if args.fast else (300.0, args.runs or 16)

    u0s = np.array([lorenz63.initial_condition(i) for i in range(runs)])
    common = dict(T=T, T_spinup=20.0, workers=args.workers)
    nil1 = NILSS(lorenz63.rhs, lorenz63.J, tuple(args.params), dt=0.005, T_seg=0.5, nus=1)
    nil2 = NILSS(lorenz63.rhs, lorenz63.J, tuple(args.params), dt=0.005, T_seg=0.5, nus=2)   # one tangent more, same orbits and seeds
    ens = run_ensemble(nil1, lorenz63.DEFAULTS, u0s, **common)
    ens2 = run_ensemble(nil2, lorenz63.DEFAULTS, u0s, **common)

    print(ens.summary(nus_check=ens2))
    print()
    print(ens.report(nus_check=ens2))
    print()
    print('published (Ni & Wang): d<z>/d(rho) = 1.0')


if __name__ == '__main__':
    main()
