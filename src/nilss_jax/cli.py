"""Command line: ``nilss-jax selftest | lorenz | gc``.

``selftest`` checks the installation against the published Lorenz 63 values (about 20 s on one core).
``lorenz`` and ``gc`` run an ensemble on the bundled systems and print the statistics and the reliability report.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from . import __version__


def _lorenz(args) -> int:
    from .core import NILSS
    from .ensemble import run_ensemble
    from .systems import lorenz63

    p = {**lorenz63.DEFAULTS, 'rho': args.rho}
    nilss = NILSS(lorenz63.rhs, lorenz63.J, tuple(args.par), dt=args.dt, T_seg=0.5, nus=1)
    u0s = np.array([lorenz63.initial_condition(i) for i in range(args.runs)])
    ens = run_ensemble(nilss, p, u0s, T=args.T, T_spinup=20.0, workers=args.workers, verbose=True)
    print(ens.summary())
    print()
    print(ens.report())
    if args.rho == 28.0 and 'rho' in args.par:
        print('\nreference (Ni & Wang 2017): d<z>/d(rho) = 1.0, lambda_1 = 0.9056')
    return 0


def _gc(args) -> int:
    from .ensemble import run_ensemble
    from .systems import guiding_center as gc

    draw = gc.load_draw(args.draw)
    pars = tuple(args.par) if args.par else tuple(draw['reference_nilss'].keys() - {'note'})
    nilss = gc.make_nilss(pars, dt=args.dt, T_seg=args.T_seg, nus=1)
    rng = np.random.RandomState(0)
    ics = draw['sea_ics']
    u0s = np.array([gc.project_to_energy_shell(ics[i % len(ics)] + 1e-6 * rng.randn(4), draw['params']) for i in range(args.runs)])
    ens = run_ensemble(nilss, draw['params'], u0s, T=args.T, T_spinup=args.T_spinup, workers=args.workers, verbose=True)
    print(ens.summary())
    print()
    print(ens.report())
    print(f"\nreference results for draw {args.draw} (T = 2e5, 35-48 runs; slopes of <s>):")
    for n in pars:
        fdr, nl = draw['reference_fd'].get(n), draw['reference_nilss'].get(n)
        print(f"  {n:<8} finite differences {fdr['value']:+.3g} +- {fdr['err']:.2g}" if fdr else f'  {n:<8} (no finite-difference reference)',
              f"  NILSS {nl['value']:+.3g} +- {nl['err']:.2g}" if nl else '')
    return 0


def _selftest(args) -> int:
    from .core import NILSS
    from .ensemble import run_ensemble
    from .systems import lorenz63

    nilss = NILSS(lorenz63.rhs, lorenz63.J, ('rho',), dt=0.005, T_seg=0.5, nus=1)
    u0s = np.array([lorenz63.initial_condition(i) for i in range(4)])
    ens = run_ensemble(nilss, lorenz63.DEFAULTS, u0s, T=100.0, T_spinup=20.0)
    slope, lam = ens.mean()['rho'], float(np.mean([np.max(r.lyapunov) for r in ens.used]))
    ok = abs(slope - 1.0) < 0.2 and 0.8 < lam < 1.0 and len(ens.used) == 4
    print(f'Lorenz 63, rho = 28, 4 orbits of T = 100: d<z>/d(rho) = {slope:.3f} (reference 1.0), lambda_1 = {lam:.3f} (reference 0.906)')
    print('selftest ' + ('passed' if ok else 'FAILED'))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='nilss-jax', description=__doc__.split('\n')[0])
    ap.add_argument('--version', action='version', version=f'nilss-jax {__version__}')
    sub = ap.add_subparsers(dest='cmd', required=True)

    sub.add_parser('selftest', help='check the installation against the Lorenz 63 reference values').set_defaults(func=_selftest)

    pl = sub.add_parser('lorenz', help='ensemble of NILSS runs on Lorenz 63')
    pl.add_argument('--par', nargs='+', default=['rho'], choices=['rho', 'sigma', 'beta'])
    pl.add_argument('--rho', type=float, default=28.0)
    pl.add_argument('--T', type=float, default=200.0)
    pl.add_argument('--dt', type=float, default=0.005)
    pl.add_argument('--runs', type=int, default=8)
    pl.add_argument('--workers', type=int, default=1)
    pl.set_defaults(func=_lorenz)

    pg = sub.add_parser('gc', help='ensemble of NILSS runs on a chaotic sea of the guiding-center flow (expect an UNRELIABLE verdict)')
    pg.add_argument('--draw', type=int, default=835, choices=[581, 835])
    pg.add_argument('--par', nargs='+', default=None, help='parameters to differentiate (default: all with a reference result)')
    pg.add_argument('--T', type=float, default=20000.0)
    pg.add_argument('--T-spinup', type=float, default=6000.0)
    pg.add_argument('--T-seg', type=float, default=200.0)
    pg.add_argument('--dt', type=float, default=0.01)
    pg.add_argument('--runs', type=int, default=8)
    pg.add_argument('--workers', type=int, default=1)
    pg.set_defaults(func=_gc)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == '__main__':
    sys.exit(main())
