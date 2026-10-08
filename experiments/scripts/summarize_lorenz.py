"""Summarize results/lorenz_ensemble/*.json: mean +- standard error over initial conditions."""
import glob
import json
import sys
from collections import defaultdict

import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else 'results/lorenz_ensemble'
groups = defaultdict(list)
for fn in sorted(glob.glob(f'{d}/task_*.json')):
    for r in json.load(open(fn))['rows']:
        groups[(r['rho'], r['nus'], r['nseg'], r['T_seg'])].append(r)
n = len(glob.glob(f'{d}/task_*.json'))
print(f'{n} initial conditions; reference d<z>/drho ~ 1.0 (arXiv:1611.00880 fig. 4), lambda_1 = 0.9056 at rho=28')
print(f'{"rho":>5} {"nus":>3} {"nseg":>5} {"T_seg":>6} {"T":>6} | {"dJ/drho mean":>12} {"+-sem":>7} {"min":>7} {"max":>7} | {"lambda_1":>8}')
for (rho, nus, nseg, T_seg), rows in sorted(groups.items()):
    g = np.array([r['dJdrho'] for r in rows])
    l1 = np.mean([r['lyap'][0] for r in rows])
    print(f'{rho:5.1f} {nus:3d} {nseg:5d} {T_seg:6.2f} {nseg*T_seg:6.0f} | {g.mean():12.4f} {g.std(ddof=1)/np.sqrt(len(g)):7.4f} {g.min():7.3f} {g.max():7.3f} | {l1:8.3f}')
