"""Ground truth for NILSS in the chaotic sea of the official guiding-center flow: the long-time <s> as a function of a
parameter, by independent long trajectories (no tangent equations at all).

The parameters are those of the draw SEED of experiments/scripts/gc_official_scan.py (default 581, whose chaotic sea
s in [0.008, 0.28] was found by experiments/scripts/gc_official_candidates.py to hold the long-time <s> of the chaotic initial
conditions at 0.110 +- 0.011). Copy c of the ensemble starts from one of the SEA_ICS of that run, displaced by a random
1e-6, and is integrated for T_SPINUP (not counted) and then for T. The same copies are used at every parameter value.
One slurm array task = one (parameter value, copy).

    python experiments/scripts/gc_official_fd.py TASK_ID [--par eps_t --h 0.01 --T 200000 --npoint 5 --ncopy 40 --outdir DIR]
        [--seed 581 --sea-ics 0,1,3,6,13 | --ics-file ICS.json --t-spinup 6000]

task = point * ncopy + copy, with the parameter value p0 + h * (point - npoint // 2). The defaults are draw 581. Another draw:
--seed N with either --sea-ics (k of gc_official_candidates.initial_condition) or --ics-file (the ics.json written by
gc_official_regime2_find_ics.py: explicit initial states).
"""
import argparse
import json
import os
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from nilss_jax.systems import guiding_center as go
from gc_official_candidates import initial_condition
from gc_official_scan import sample

SEED = 581
SEA_ICS = [0, 1, 3, 6, 13]          # initial conditions k of gc_official_candidates.py that are in the chaotic sea
NPOINT = 5                          # p0 + h * (-2, -1, 0, 1, 2)
NCOPY = 40
N_TASKS = NPOINT * NCOPY
T_SPINUP = 6000.0


def load_ics(path):
    """[(k, u0)] from an ics.json written by gc_official_regime2_find_ics.py."""
    with open(path) as fh:
        return [(d['k'], d['u0']) for d in json.load(fh)['ics']]


def copy_initial_condition(seed, sea_ics, copy, p, ics=None):
    """(k, u0) of copy `copy`: the chaotic initial condition (index sea_ics[copy % n] of gc_official_candidates, or the explicit
    list `ics` of (k, u0)), displaced by a random 1e-6, with v_par re-projected onto the energy shell of the parameters p."""
    p0, u_scan, _ = sample(seed)
    if ics is None:
        k = sea_ics[copy % len(sea_ics)]
        u0 = initial_condition(seed, k, p0, u_scan)
    else:
        k, u0 = ics[copy % len(ics)]
    u0 = np.array(u0) + 1e-6 * np.random.RandomState(777 + copy).randn(4)
    u0[3] = np.sign(u0[3]) * np.sqrt(1.0 - p['lam'] * float(go.B_derivs(u0[0], u0[1], u0[2], p)[0]))
    return k, u0


def parse_ints(text):
    return [int(x) for x in text.split(',')]


@partial(jax.jit, static_argnames=['nblock', 'steps_per_block'])
def block_means(u0, p, dt, nblock, steps_per_block):
    """Means of s over nblock blocks, the extremes of s, and the energy drift, along the RK4 orbit of rhs4."""
    def step(u, _):
        k1 = go.rhs4(u, p)
        k2 = go.rhs4(u + 0.5 * dt * k1, p)
        k3 = go.rhs4(u + 0.5 * dt * k2, p)
        k4 = go.rhs4(u + dt * k3, p)
        u = u + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return u, u[0]

    def block(u, _):
        u, xs = jax.lax.scan(step, u, None, length=steps_per_block)
        return u, (xs.mean(), xs.max(), xs.min())
    u, (m, mx, mn) = jax.lax.scan(block, u0, None, length=nblock)
    return m, mx.max(), mn.min(), go.energy(u, p) - go.energy(u0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task', type=int)
    ap.add_argument('--par', default='eps_t')
    ap.add_argument('--h', type=float, default=0.01)
    ap.add_argument('--npoint', type=int, default=NPOINT)
    ap.add_argument('--ncopy', type=int, default=NCOPY)
    ap.add_argument('--T', type=float, default=200000.0)
    ap.add_argument('--dt', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--sea-ics', type=parse_ints, default=SEA_ICS)
    ap.add_argument('--ics-file', default=None)
    ap.add_argument('--t-spinup', type=float, default=T_SPINUP)
    ap.add_argument('--outdir', default='results/gc_official_fd')
    args = ap.parse_args()

    ipoint, copy = divmod(args.task, args.ncopy)
    p = sample(args.seed)[0]
    value = p[args.par] + args.h * (ipoint - args.npoint // 2)
    p = {**p, args.par: value}
    k, u0 = copy_initial_condition(args.seed, args.sea_ics, copy, p, load_ics(args.ics_file) if args.ics_file else None)

    spb = 100
    n_spin = int(round(args.t_spinup / (args.dt * spb)))
    nblock = int(round(args.T / (args.dt * spb)))
    # spin-up and the run in one go (the first n_spin blocks are dropped)
    m, mx, mn, dE = block_means(jnp.array(u0), p, args.dt, n_spin + nblock, spb)
    m = np.asarray(m)
    run = m[n_spin:]
    parts = run.reshape(20, -1).mean(axis=1)
    out = {'task': args.task, 'par': args.par, 'value': value, 'copy': copy, 'ic_k': k, 'u0': u0.tolist(), 'T': args.T,
           'dt': args.dt, 'mean_s': float(run.mean()), 'parts': parts.tolist(), 'x_max': float(mx), 'x_min': float(mn),
           'dE': float(dE), 'finite': bool(np.all(np.isfinite(m))), 'seed': args.seed, 't_spinup': args.t_spinup}
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, f'task_{args.task:03d}.json'), 'w') as fh:
        json.dump(out, fh)
    print(f"{args.par}={value:.5f} copy={copy:2d} <s>={out['mean_s']:.4f} parts std={parts.std(ddof=1):.4f} "
          f"s in [{mn:.4f},{mx:.3f}] dE={dE:.1e} finite={out['finite']}")


if __name__ == '__main__':
    main()
