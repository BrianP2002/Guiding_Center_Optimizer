"""NILSS on Lorenz 63 over an ensemble of initial conditions (one slurm array task = one initial condition).

    python experiments/scripts/lorenz_ensemble.py TASK_ID

Reference (Ni & Wang 2017, fig. 4): d<z>/d(rho) ~ 1 for chaotic rho, with 50 segments of 2 time units.
"""
import argparse
import json
import os

import numpy as np

from nilss_jax.reference import nilss
from nilss_jax.systems import lorenz63

DT = 0.01
RHOS = [26.0, 28.0, 30.0, 35.0, 40.0]
SETTINGS = [  # (nus, number of segments, segment length)
    (1, 50, 2.0),
    (2, 50, 2.0),
    (1, 200, 2.0),
    (1, 50, 0.5),  # shorter segments, same number: T = 25
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('task', type=int)
    p.add_argument('--outdir', default='results/lorenz_ensemble')
    args = p.parse_args()

    rng = np.random.RandomState(args.task)
    u0 = np.array([12.0, 6.8, 36.5]) + rng.rand(3)
    integ, fjj = lorenz63.make_problem(DT)
    rows = []
    for rho in RHOS:
        for nus, nseg, T_seg in SETTINGS:
            np.random.seed(10 * args.task + nus)
            J, dJ, info = nilss(DT, nseg, T_seg, 20, u0, nus, 'rho', rho, integ, fjj, return_info=True)
            rows.append({'rho': rho, 'nus': nus, 'nseg': nseg, 'T_seg': T_seg, 'J': J, 'dJdrho': dJ,
                         'lyap': info['lyap'].tolist(), 'T': info['T']})
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, f'task_{args.task:03d}.json'), 'w') as fh:
        json.dump({'task': args.task, 'rows': rows}, fh)
    print(f'task {args.task}: done {len(rows)} runs')


if __name__ == '__main__':
    main()
