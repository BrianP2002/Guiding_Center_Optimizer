"""Are the bounded chaotic draws of gc_official_scan.py really chaotic and ergodic?

The 9 draws that experiments/scripts/summarize_gc_official_scan.py calls chaotic (T = 4000) are re-run for a much longer time
from IC_PER_SEED initial conditions each (the original one and random ones on the same energy / mu surface, both signs
of v_par). Ergodicity needs: the same long-time <s> from every initial condition inside the chaotic sea, and a
positive lambda_1 that does not decay with the length of the run. One slurm array task = one (seed, initial condition).

    python experiments/scripts/gc_official_candidates.py TASK_ID [--T 200000]
"""
import argparse
import json
import os

import numpy as np
import jax.numpy as jnp

from nilss_jax import lyapunov as lyap_tools
from nilss_jax.systems import guiding_center as go
from gc_official_scan import sample

SEEDS = [95, 951, 34, 581, 959, 835, 663, 873, 792]
IC_PER_SEED = 16
N_TASKS = len(SEEDS) * IC_PER_SEED


def initial_condition(seed, k, p, u_scan):
    """k = 0: the initial condition of the scan; k > 0: random position and sign of v_par, same lam (same E, mu)."""
    if k == 0:
        return u_scan
    rng = np.random.RandomState(10_000 * seed + k)
    while True:
        s0, th0, ze0 = rng.uniform(0.02, 0.4), 2 * np.pi * rng.rand(), 2 * np.pi * rng.rand()
        v2 = 1.0 - p['lam'] * float(go.B_derivs(s0, th0, ze0, p)[0])
        if v2 > 0.0:
            return np.array([s0, th0, ze0, np.sign(rng.rand() - 0.5) * np.sqrt(v2)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task', type=int)
    ap.add_argument('--T', type=float, default=200000.0)
    ap.add_argument('--dt', type=float, default=0.005)
    ap.add_argument('--outdir', default='results/gc_official_candidates')
    args = ap.parse_args()

    seed, k = SEEDS[args.task // IC_PER_SEED], args.task % IC_PER_SEED
    p, u_scan, _ = sample(seed)
    u0 = initial_condition(seed, k, p, u_scan)
    runner = lyap_tools.make_runner(go.rhs4, args.dt, 100, obs=lambda u: u[jnp.array([0, 3])])
    res = lyap_tools.lyapunov_spectrum(go.rhs4, u0, args.dt, 0.0, args.T, checkpoints=20, args=(p,), runner=runner)
    dE = float(go.energy(jnp.array(res['u_final']), p)) - float(go.energy(jnp.array(u0), p)) if res['finite'] else float('nan')
    out = {'task': args.task, 'seed': seed, 'k': k, 'params': p, 'u0': u0.tolist(), 'dt': args.dt, 'dE': dE,
           'v_min': float(res['obs_min'][1]), 'v_max': float(res['obs_max'][1]),
           **{k_: (v.tolist() if hasattr(v, 'tolist') else v) for k_, v in res.items() if k_ not in ('obs_min', 'obs_max', 'u_final')}}
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, f'task_{args.task:03d}.json'), 'w') as fh:
        json.dump(out, fh)
    print(f"seed={seed} k={k:2d} lyap1={res['lyap'][0]:.5f} lyap1*T={res['lyap'][0] * res['T']:.1f} <s>={np.mean(res['x_mean_parts']):.4f} "
          f"s in [{res['x_min']:.3f},{res['x_max']:.3f}] v in [{res['obs_min'][1]:+.2f},{res['obs_max'][1]:+.2f}] dE={dE:.1e} finite={res['finite']}")


if __name__ == '__main__':
    main()
