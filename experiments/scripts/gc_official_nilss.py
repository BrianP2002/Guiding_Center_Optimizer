"""NILSS (streaming, several parameters at once) in the chaotic sea of the official guiding-center flow.

Same setting as experiments/scripts/gc_official_fd.py: the parameters of draw 581 of experiments/scripts/gc_official_scan.py (trapped and barely
passing particles, symmetry broken, a bounded chaotic sea at s in [0.008, 0.28]) and the same initial conditions.
NILSS is applied to the 4D system guiding_center.rhs4 with J = s, on the energy surface (the initial tangents satisfy
grad E . w = 0, grad E . v* + dE/dp = 0), with nus homogeneous tangents; the sensitivities with respect to eps_t, eps_m,
kappa and iota share them. The ground truth for eps_t is the slope of the long-time <s> of the sea orbits from
experiments/scripts/gc_official_fd.py (summarize_gc_official_fd.py). One slurm array task = one run.

    python experiments/scripts/gc_official_nilss.py TASK_ID [--T 200000]
        [--seed 581 --sea-ics 0,1,3,6,13 | --ics-file ICS.json] [--pars eps_t,eps_m,kappa,iota] [--t-spinup 6000]
        [--layout draw581 | --layout main_nus2 --n-main 60 --n-nus2 10]

The defaults are draw 581 with its 70 runs. Layout main_nus2 (another draw): n-main runs with nus = 1 and T_seg = 200 on the copies
0 .. n-main-1, then n-nus2 runs with nus = 2 on the copies 0 .. n-nus2-1 (the same orbits, for the paired comparison).
"""
import argparse
import json
import os
import time

import numpy as np
import jax.numpy as jnp

from nilss_jax.systems import guiding_center as go
from nilss_jax import NILSSStreamer
from gc_official_candidates import initial_condition
from gc_official_scan import sample
from gc_official_fd import SEED, SEA_ICS, T_SPINUP, copy_initial_condition, load_ics, parse_ints

PARS = ('eps_t', 'eps_m', 'kappa', 'iota')
# (copy, nus, T_seg): 40 runs of the main setting, 10 each of three variations
RUNS = ([(c, 1, 200.0) for c in range(40)] + [(c, 2, 200.0) for c in range(10)]
        + [(c, 1, 50.0) for c in range(10)] + [(c, 1, 800.0) for c in range(10)])
N_TASKS = len(RUNS)


def runs_for(layout, n_main, n_nus2):
    if layout == 'draw581':
        return RUNS
    return [(c, 1, 200.0) for c in range(n_main)] + [(c, 2, 200.0) for c in range(n_nus2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task', type=int)
    ap.add_argument('--T', type=float, default=200000.0)
    ap.add_argument('--dt', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--sea-ics', type=parse_ints, default=SEA_ICS)
    ap.add_argument('--ics-file', default=None)
    ap.add_argument('--pars', default=','.join(PARS))
    ap.add_argument('--t-spinup', type=float, default=T_SPINUP)
    ap.add_argument('--layout', default='draw581', choices=['draw581', 'main_nus2'])
    ap.add_argument('--n-main', type=int, default=60)
    ap.add_argument('--n-nus2', type=int, default=10)
    ap.add_argument('--outdir', default='results/gc_official_nilss')
    args = ap.parse_args()
    pars = tuple(args.pars.split(','))

    copy, nus, T_seg = runs_for(args.layout, args.n_main, args.n_nus2)[args.task]
    p = sample(args.seed)[0]
    k, u0 = copy_initial_condition(args.seed, args.sea_ics, copy, p, load_ics(args.ics_file) if args.ics_file else None)

    streamer = NILSSStreamer(go.rhs4, lambda u: u[0], pars, args.dt, T_seg, nus, invariant=go.energy)
    nseg, nseg_ps = int(round(args.T / T_seg)), int(round(args.t_spinup / T_seg))
    t0 = time.time()
    Javg, dJdp, info = streamer.run_segments(p, u0, nseg, nseg_ps, seed=1000 + args.task)
    seg = info['dJdp_segments']
    blocks = seg[:nseg // 20 * 20].reshape(20, -1, len(pars)).sum(axis=1)       # 20 block sums, for the error of the run
    out = {'task': args.task, 'copy': copy, 'ic_k': k, 'nus': nus, 'T_seg': T_seg, 'T': info['T'], 'dt': args.dt,
           'pars': pars, 'Javg': float(Javg), 'dJdp': np.asarray(dJdp).tolist(), 'lyap': np.asarray(info['lyap']).tolist(),
           'blocks': blocks.tolist(), 'segments': seg.tolist(), 'vnorm': info['vnorm'].tolist(), 'elapsed': time.time() - t0, 'seed': args.seed, 't_spinup': args.t_spinup}
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, f'task_{args.task:03d}.json'), 'w') as fh:
        json.dump(out, fh)
    print(f"task {args.task} copy={copy} nus={nus} T_seg={T_seg:g}: <s>={Javg:.4f} lyap={np.round(info['lyap'], 5)} "
          f"dJ/d{pars} = {np.round(dJdp, 3)}  ({time.time() - t0:.0f} s)")


if __name__ == '__main__':
    main()
