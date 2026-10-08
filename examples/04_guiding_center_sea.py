"""NILSS on the chaotic sea of the guiding-center flow: what an UNRELIABLE verdict looks like.

The vacuum Boozer-coordinate guiding-center equations of SIMSOPT / FIRM3D (nilss_jax.systems.guiding_center) with a
symmetry-breaking field have a thin bounded chaotic sea inside a mixed phase space. Two such regimes found by a random scan are
bundled: draw 835 (only eps_2 breaks the symmetry, lambda_1 about 0.003, well mixed) and draw 581 (lambda_1 about 0.006). The
objective is J = s, the flux label, on the energy surface E = 1/2 (the tangents are put on the perturbed energy surface).

This example runs an ensemble of NILSS runs on the sea orbits, prints the statistics and the reliability report, and lists the
results of the studies (T = 2e5, 35-48 runs; finite differences over 60 copies per point). Expect an UNRELIABLE verdict: the
shadowing direction has a power-law tail (Hill index about 1), so the printed standard errors mean little, and some
parameters are off (docs/findings.md). With --fd one parameter is also checked by ensemble finite differences.

The --fast run (4 orbits of T = 2000, threshold lambda_1 T >= 3) is a smoke test; the full defaults (8 orbits of T = 20000) take
several minutes on one core (use --workers).

    python examples/04_guiding_center_sea.py [--draw 835|581] [--fast] [--runs 8] [--T 20000] [--workers 4] [--fd]
"""
import argparse

import numpy as np

from nilss_jax import fd, run_ensemble
from nilss_jax.systems import guiding_center as gc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--draw', type=int, default=835, choices=[581, 835])
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--runs', type=int, default=None)
    ap.add_argument('--T', type=float, default=None)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--fd', action='store_true', help='also check one parameter by ensemble finite differences (slow)')
    args = ap.parse_args()
    runs = args.runs or (4 if args.fast else 8)
    T = args.T or (2000.0 if args.fast else 20000.0)
    T_spinup = 600.0 if args.fast else 6000.0

    draw = gc.load_draw(args.draw)
    p = draw['params']
    pars = tuple(sorted(draw['reference_nilss'].keys() - {'note'}))
    rng = np.random.RandomState(0)
    ics = draw['sea_ics']
    # orbits on the sea: the stored initial conditions, displaced by 1e-6 and put back on the energy shell E = 1/2
    u0s = np.array([gc.project_to_energy_shell(ics[i % len(ics)] + 1e-6 * rng.randn(4), p) for i in range(runs)])

    nilss = gc.make_nilss(pars, dt=0.01, T_seg=200.0, nus=1)
    # lambda_1 of the sea is 0.003-0.006: a run needs T of several thousand to resolve the exponent (lambda_1 T >= 20, the default
    # threshold below which a run is not used); the --fast smoke test relaxes the threshold
    ens = run_ensemble(nilss, p, u0s, T=T, T_spinup=T_spinup, workers=args.workers, min_lyap_time=3.0 if args.fast else 20.0)
    if args.fast:
        print('(--fast: runs this short carry no statistics; this is a smoke test of the machinery)\n')
    print(ens.summary())
    print()
    print(ens.report())

    print(f'\nresults of the studies for draw {args.draw} (T = 2e5), slopes of <s>:')
    for n in pars:
        r, q = draw['reference_fd'].get(n), draw['reference_nilss'].get(n)
        fd_txt = f"finite differences {r['value']:+.3g} +- {r['err']:.2g}" if r else 'finite differences: -'
        print(f"  {n:<6} {fd_txt:<38} NILSS {q['value']:+.3g} +- {q['err']:.2g}")

    if args.fd:
        par = 'eps_2' if args.draw == 835 else 'eps_t'
        h = {'eps_2': 0.001, 'eps_t': 0.0025}[par]
        copies = np.tile(np.array(ics), (4, 1))[: (8 if args.fast else 48)]
        sea = lambda parts: parts.std(axis=1, ddof=1) > 0.05 * np.abs(parts.mean(axis=1))      # a relative criterion for the sea
        res = fd.finite_difference(gc.rhs4, gc.mean_radius, p, par, h, copies, T=T, dt=0.01, npoint=3 if args.fast else 7,
                                   T_spinup=T_spinup, project=gc.project_to_energy_shell, select=sea)
        print()
        print(res.summary())
        print(fd.compare(ens, {par: res}))


if __name__ == '__main__':
    main()
