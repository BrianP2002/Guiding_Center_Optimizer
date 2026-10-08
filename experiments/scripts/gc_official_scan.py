"""Does the OFFICIAL vacuum Boozer guiding-center system (guiding_center.rhs4) have a bounded chaotic regime?

Random draws of the field (helical amplitude and helicity, and three symmetry-breaking terms), of the drift
strength kappa, the rotational transform and the initial condition (position and pitch v_par/v, so passing and
trapped particles are both covered). 15 % of the draws have the exact quasi-helical field as a control: there the
flow is integrable (E and phi_qh are conserved), so any positive exponent would be numerical.
For each draw: the Lyapunov spectrum of the 4D flow, the trajectory average of s in four parts, the range of s and
v_par, the energy drift. One slurm array task = DRAWS_PER_TASK draws.

    python experiments/scripts/gc_official_scan.py TASK_ID [--T 4000]
"""
import argparse
import json
import os

import numpy as np
import jax.numpy as jnp

from nilss_jax import lyapunov as lyap_tools
from nilss_jax.systems import guiding_center as go

DRAWS_PER_TASK = 8
N_TASKS = 120
LOG10 = np.log10


def sample(seed):
    rng = np.random.RandomState(seed)
    p = dict(go.default_params)
    p['N'] = float(rng.choice([1, 2, 3, 4, 5]))
    p['N2'] = p['N'] + float(rng.choice([-2, -1, 1, 2]))
    p['eps_h'] = float(10 ** rng.uniform(LOG10(0.03), LOG10(0.2)))
    broken = bool(rng.rand() < 0.85)
    if broken:
        for key in ('eps_t', 'eps_m', 'eps_2'):
            if rng.rand() < 0.5:
                p[key] = float(10 ** rng.uniform(LOG10(0.005), LOG10(0.1)))
        if p['eps_t'] == p['eps_m'] == p['eps_2'] == 0.0:
            p['eps_t'] = float(10 ** rng.uniform(LOG10(0.005), LOG10(0.1)))
    p['iota'] = float(rng.uniform(0.3, 1.8))
    p['kappa'] = float(10 ** rng.uniform(LOG10(0.03), LOG10(20.0)))
    s0, th0, ze0 = rng.uniform(0.02, 0.3), 2 * np.pi * rng.rand(), 2 * np.pi * rng.rand()
    xi0 = rng.uniform(-1.0, 1.0)                     # v_par / v at the start
    B0 = float(go.B_derivs(s0, th0, ze0, p)[0])
    p['lam'] = float((1.0 - xi0 ** 2) / B0)          # v_perp^2 / (v^2 B)
    return p, np.array([s0, th0, ze0, xi0]), broken


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task', type=int)
    ap.add_argument('--T', type=float, default=4000.0)
    ap.add_argument('--dt', type=float, default=0.005)
    ap.add_argument('--outdir', default='results/gc_official_scan')
    args = ap.parse_args()

    runner = lyap_tools.make_runner(go.rhs4, args.dt, 100, obs=lambda u: u[jnp.array([0, 3])])   # observables: s, v_par
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f'task_{args.task:03d}.jsonl')
    with open(path, 'w') as fh:
        for k in range(DRAWS_PER_TASK):
            seed = args.task * DRAWS_PER_TASK + k
            p, u0, broken = sample(seed)
            res = lyap_tools.lyapunov_spectrum(go.rhs4, u0, args.dt, 0.0, args.T, args=(p,), runner=runner)
            E0 = float(go.energy(jnp.array(u0), p))
            dE = float(go.energy(jnp.array(res['u_final']), p)) - E0 if res['finite'] else float('nan')
            out = {'seed': seed, 'broken': broken, 'params': p, 'u0': u0.tolist(), 'dt': args.dt, 'dE': dE,
                   'v_min': float(res['obs_min'][1]), 'v_max': float(res['obs_max'][1]),
                   **{k_: (v.tolist() if hasattr(v, 'tolist') else v) for k_, v in res.items()
                      if k_ not in ('obs_min', 'obs_max', 'u_final')}}
            fh.write(json.dumps(out) + '\n')
            fh.flush()
            print(f"seed={seed:4d} broken={broken!s:5s} kappa={p['kappa']:7.3f} xi0={u0[3]:+.2f} lyap1={res['lyap'][0]:8.4f}"
                  f" lyap1*T={res['lyap'][0] * res['T']:8.2f} s in [{res['x_min']:.3f},{res['x_max']:.3f}]"
                  f" v in [{res['obs_min'][1]:+.2f},{res['obs_max'][1]:+.2f}] dE={dE:.1e} finite={res['finite']}", flush=True)


if __name__ == '__main__':
    main()
