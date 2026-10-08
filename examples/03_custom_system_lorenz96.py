"""A user-defined system (Lorenz 96) from scratch, cross-checked with finite differences.

The whole workflow for a system of your own: (1) write rhs(u, p) and J(u) with jax.numpy as top-level functions;
(2) count the positive Lyapunov exponents with nilss_jax.lyapunov -- NILSS needs that many homogeneous tangents, and the
paper (arXiv:1611.00880, sec. 4.2) advises slightly more, so nus = n_positive + 1 here; (3) run an ensemble of NILSS runs;
(4) check against nilss_jax.fd, ensemble finite differences that use no tangent equations at all.

System: Lorenz 96 with N = 8 variables, forcing F = 8, J = mean(u^2), parameter F.

The last lines say whether NILSS and the finite differences agree within three combined standard errors. Nothing is tuned
to make them agree: if they do not, that is the result (see docs/reliability.md for how to read it).

Expect (16 orbits of T = 500; 2 positive exponents, nus = 3): the finite differences give d<J>/dF = 3.74 +- 0.02 (3.69 with
dt = 0.005) and the reliability verdict for NILSS is UNRELIABLE: the shadowing direction of this 8-variable Lorenz 96 has a power-law
tail (Hill index about 0.8-1.1), so the ensemble scatters widely -- with dt = 0.01 a median of +4.7 but a 10-90% range from -10 to
+6 and a mean of +1.7 +- 1.7; with --dt 0.005 a median of +3.2, a 10-90% range from +1.4 to +4.6 and a mean of +3.0 +- 0.3 -- and the
dependence of the statistics on dt is itself a symptom of a system that is not uniformly hyperbolic. The "agree" printed for the
standard error is then no evidence. Run with --workers to use several processes (rhs and J are top-level functions of this
file, which is enough for the worker processes).

    python examples/03_custom_system_lorenz96.py [--fast] [--dt 0.005] [--workers 4]
"""
import argparse

import numpy as np
import jax.numpy as jnp

from nilss_jax import NILSS, fd, lyapunov, run_ensemble

N, F = 8, 8.0


def rhs(u, p):
    return (jnp.roll(u, -1) - jnp.roll(u, 2)) * jnp.roll(u, 1) - u + p['F']


def J(u):
    return jnp.mean(u ** 2)


def flow(u, forcing):                      # the signature of nilss_jax.lyapunov: f(u, *args)
    return rhs(u, {'F': forcing})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--dt', type=float, default=0.01)
    args = ap.parse_args()
    fast = args.fast
    p = {'F': F}
    u0s = F + 0.5 * np.random.RandomState(0).randn(16 if not fast else 6, N)

    spec = lyapunov.lyapunov_spectrum(flow, u0s[0], args.dt, 50.0, 400.0 if fast else 3000.0, args=(F,))
    npos = int(np.sum(spec['lyap'] > 0.05))
    print('Lyapunov spectrum:', np.array2string(spec['lyap'], precision=3), f'-> {npos} positive exponents, nus = {npos + 1}')

    nilss = NILSS(rhs, J, ('F',), dt=args.dt, T_seg=0.5, nus=npos + 1)
    ens = run_ensemble(nilss, p, u0s, T=100.0 if fast else 500.0, T_spinup=50.0, workers=args.workers)
    print(ens.summary())

    res = fd.finite_difference(rhs, J, p, 'F', h=0.25, u0s=np.tile(u0s, (2, 1)), T=200.0 if fast else 1000.0, dt=args.dt,
                               npoint=3 if fast else 5, T_spinup=50.0)
    print()
    print(res.summary())
    print()
    print(fd.compare(ens, {'F': res}))
    z = (ens.mean()['F'] - res.slope) / np.hypot(ens.sem()['F'], res.slope_err)
    verdict = ens.report().verdict
    print(f'NILSS and finite differences {"agree" if abs(z) < 3 else "DISAGREE"} ({z:+.1f} combined standard errors); reliability verdict {verdict.upper()}.')
    if verdict != 'ok':
        print('The verdict is not OK: the NILSS standard error is then not a reliable yardstick, so neither agreement nor disagreement '
              'says much -- more or longer runs, or the finite differences, decide.')


if __name__ == '__main__':
    main()
