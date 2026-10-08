"""Exploratory: does breaking the helical symmetry of B make the guiding-center flow chaotic?

NOT the model of the project. legacy/app_gc.py has B = 1 + a0 sqrt(x) cos(y - a1 z), which depends on the single
angle chi = y - a1 z: an exactly quasi-helically symmetric field, whose guiding-center flow reduces to an
autonomous 2D flow in (x, chi) and cannot be chaotic. Here a second helicity is added,

    B = 1 + sqrt(x) [ a0 cos(y - a1 z) + b0 cos(y - b1 z) ],

with the same drift equations as legacy/app_gc.f_ode (written with autodiff derivatives of B).
One slurm array task = one (b1, b0, initial condition).

    python experiments/scripts/gc_symmetry_breaking.py TASK_ID
"""
import argparse
import json
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

from nilss_jax import lyapunov as lyap_tools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'legacy'))   # legacy/app_gc.py, the model of the project (legacy/ is not a package)
import app_gc

B1S = [0.0, 2.0]
B0S = [0.0, 0.02, 0.05, 0.1]
N_IC = 3
N_TASKS = len(B1S) * len(B0S) * N_IC


def make_f(params, b0, b1):
    a0, a1, iota, G, lam = (params[k] for k in ('a0', 'a1', 'iota', 'G', 'lam'))

    def B(x, y, z):
        return 1.0 + jnp.sqrt(x) * (a0 * jnp.cos(y - a1 * z) + b0 * jnp.cos(y - b1 * z))

    Bx, By = jax.grad(B, argnums=0), jax.grad(B, argnums=1)

    def f(u):
        x, y, z = u
        Bv = B(x, y, z)
        factor = 2.0 / lam - Bv
        V = jnp.sqrt(1.0 - lam * Bv)
        return jnp.array([-By(x, y, z) * factor / Bv,
                          Bx(x, y, z) * factor / Bv + iota * V * Bv / G,
                          Bv * V / G])
    return f


def main():
    p = argparse.ArgumentParser()
    p.add_argument('task', type=int)
    p.add_argument('--T', type=float, default=1000.0)
    p.add_argument('--T-spinup', type=float, default=50.0)
    p.add_argument('--dt', type=float, default=1e-3)
    p.add_argument('--outdir', default='results/gc_symmetry_breaking')
    args = p.parse_args()

    ic_seed = args.task % N_IC
    b0 = B0S[(args.task // N_IC) % len(B0S)]
    b1 = B1S[args.task // (N_IC * len(B0S))]
    rng = np.random.RandomState(1000 + ic_seed)
    u0 = np.array([0.1 * rng.rand() + 0.01, 2 * np.pi * rng.rand(), 2 * np.pi * rng.rand()])

    f = make_f(app_gc.default_params, b0, b1)
    if b0 == 0.0:  # consistency with the model of the project
        ref = app_gc.f_ode_wrapper(jnp.array(u0), app_gc.default_params)
        assert np.allclose(np.asarray(f(jnp.array(u0))), np.asarray(ref), rtol=1e-10)

    res = lyap_tools.lyapunov_spectrum(f, u0, args.dt, args.T_spinup, args.T)
    out = {'task': args.task, 'b0': b0, 'b1': b1, 'ic_seed': ic_seed, 'u0': u0.tolist(), 'dt': args.dt,
           **{k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in res.items()}}
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, f'task_{args.task:03d}.json'), 'w') as fh:
        json.dump(out, fh)
    print(f"b1={b1} b0={b0} ic={ic_seed} lyap={np.round(res['lyap'], 4)} <x> parts={np.round(res['x_mean_parts'], 3)} "
          f"xmax={res['x_max']:.2f} finite={res['finite']}")


if __name__ == '__main__':
    main()
