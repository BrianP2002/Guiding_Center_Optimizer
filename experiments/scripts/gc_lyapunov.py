"""Is the guiding-center flow chaotic, and is <x> independent of the initial condition?

NILSS needs (i) a positive Lyapunov exponent and (ii) ergodicity (the long-time average must not
depend on the initial condition). One slurm array task = one (parameter setting, initial condition).

    python experiments/scripts/gc_lyapunov.py TASK_ID [--T 1000] [--ntask-only]
"""
import argparse
import json
import os
import sys

import numpy as np
import jax.numpy as jnp

from nilss_jax import lyapunov as lyap_tools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'legacy'))   # legacy/app_gc.py, the model of the project (legacy/ is not a package)
import app_gc

N_IC = 6
CONFIGS = [('default', None, None)]
for _par, (_lo, _hi) in app_gc.PARAM_BOUNDS.items():
    CONFIGS += [(_par, 'lo', _lo), (_par, 'hi', _hi)]
N_TASKS = len(CONFIGS) * N_IC


def initial_condition(ic_seed):
    rng = np.random.RandomState(1000 + ic_seed)
    return np.array([0.1 * rng.rand() + 0.01, 2 * np.pi * rng.rand(), 2 * np.pi * rng.rand()])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('task', type=int)
    p.add_argument('--T', type=float, default=1000.0)
    p.add_argument('--T-spinup', type=float, default=50.0)
    p.add_argument('--dt', type=float, default=1e-3)
    p.add_argument('--outdir', default='results/gc_lyapunov')
    args = p.parse_args()

    name, which, value = CONFIGS[args.task // N_IC]
    ic_seed = args.task % N_IC
    params = dict(app_gc.default_params)
    if value is not None:
        params[name] = value
    u0 = initial_condition(ic_seed)

    res = lyap_tools.lyapunov_spectrum(lambda u: app_gc.f_ode_wrapper(u, params), u0, args.dt, args.T_spinup, args.T)
    out = {'task': args.task, 'par': name, 'which': which, 'value': value, 'params': params, 'ic_seed': ic_seed,
           'u0': u0.tolist(), 'dt': args.dt, **{k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in res.items()}}
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, f'task_{args.task:03d}.json'), 'w') as fh:
        json.dump(out, fh)
    print(f"{name:8s} {str(which):5s} ic={ic_seed}  lyap={np.round(res['lyap'], 4)}  <x> parts={np.round(res['x_mean_parts'], 3)}"
          f"  xmax={res['x_max']:.2f} finite={res['finite']}")


if __name__ == '__main__':
    main()
