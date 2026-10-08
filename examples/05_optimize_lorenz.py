"""Optimisation with ensemble sensitivities: steer the long-time mean of z in Lorenz 63 by rho.

nilss_jax.optimize.minimize wraps scipy's L-BFGS-B. At every evaluation it runs the same ensemble of orbits (same initial
conditions and seeds, so the objective is a smoother function of the parameter), takes <J> and its sensitivity, and checks
the reliability report. The objective here is (<z> - 30.5)^2 for rho in [26, 42]; <z> is close to rho - 4.5 there, so the
optimum is near rho = 35.

Two gradients: the ensemble mean of the NILSS sensitivities, and central finite differences of the ensemble average
(gradient='fd', fd_h = 1.0), which is what to use where the NILSS verdict is UNRELIABLE (it costs 2 extra evaluations of <J> per
free parameter and needs no tangent equations).

Expect: both end within about 0.1 of <z> = 30.5 at rho close to 34.9. The finite-time average of a chaotic orbit is a rough
function of rho (about +-0.05 in <z> for T = 100, whatever the step), so L-BFGS-B may end by reporting a line-search failure
('ABNORMAL') once it is at that noise floor: the answer is then as good as the ensemble allows. The verdicts are CAUTION
only because 4 orbits are fewer than the 8 the report asks for.

    python examples/05_optimize_lorenz.py [--fast]
"""
import argparse

import numpy as np

from nilss_jax import NILSS, optimize
from nilss_jax.systems import lorenz63


def show(res):
    for h in res.history[:3]:
        print(f"  evaluation: rho = {h['x']['rho']:.4f}  <z> = {h['J']:.4f}  d<z>/drho = {h['grad'][0]:.4f}" + (f"  [{h['verdict']}]" if h['verdict'] else ''))
    print(f"  ... {len(res.history)} evaluations in all")
    print(f"result: rho = {res.x['rho']:.3f}, <z> = {res.J:.3f} (target 30.5); scipy: {res.message.strip() or 'no message'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fast', action='store_true')
    args = ap.parse_args()
    T, maxiter = (40.0, 4) if args.fast else (100.0, 10)

    nilss = NILSS(lorenz63.rhs, lorenz63.J, ('rho',), dt=0.01, T_seg=0.5, nus=1)
    u0s = np.array([lorenz63.initial_condition(i) for i in range(4)])
    common = dict(free=['rho'], u0s=u0s, T=T, T_spinup=10.0, bounds={'rho': (26.0, 42.0)}, target=30.5, maxiter=maxiter)

    print('--- NILSS gradients (ensemble mean) ---')
    res = optimize.minimize(nilss, lorenz63.DEFAULTS, **common)
    show(res)

    print('--- finite-difference gradients ---')
    res = optimize.minimize(nilss, lorenz63.DEFAULTS, gradient='fd', fd_h=1.0, **common)
    show(res)


if __name__ == '__main__':
    main()
