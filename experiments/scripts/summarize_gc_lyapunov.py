"""Summarize results/gc_lyapunov/*.json (one line per parameter setting, statistics over initial conditions)."""
import glob
import json
import sys
from collections import defaultdict

import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else 'results/gc_lyapunov'
groups = defaultdict(list)
for fn in sorted(glob.glob(f'{d}/task_*.json')):
    r = json.load(open(fn))
    groups[(r['par'], r['which'], r['value'])].append(r)
T = next(iter(groups.values()))[0]['T']
print(f'T = {T:g} per trajectory. A resolved positive exponent needs lambda_1 * T >> 1; lambda_1 ~ log(T)/T ~ {np.log(T)/T:.4f} is what a regular flow gives.')
print('x_ic = std over initial conditions of the trajectory average <x>; x_in = mean std between the parts of one trajectory.')
print(f'{"par":>8} {"which":>5} {"value":>7} {"n":>2} | {"lam1 mean":>9} {"lam1 max":>9} {"lam1*T":>7} {"lam2 mean":>9} {"lam3 mean":>9} | {"<x> mean":>8} {"x_ic":>6} {"x_in":>6} {"xmax":>6} {"fin":>4}')
for (par, which, value), rows in groups.items():
    ly = np.array([r['lyap'] for r in rows])
    xm = np.array([np.mean(r['x_mean_parts']) for r in rows])
    xin = np.mean([np.std(r['x_mean_parts']) for r in rows])
    val = '-' if value is None else f'{value:.3f}'
    print(f'{par:>8} {str(which):>5} {val:>7} {len(rows):2d} | {ly[:,0].mean():9.4f} {ly[:,0].max():9.4f} {ly[:,0].max()*T:7.2f} {ly[:,1].mean():9.4f} {ly[:,2].mean():9.4f} |'
          f' {xm.mean():8.3f} {xm.std():6.3f} {xin:6.3f} {max(r["x_max"] for r in rows):6.2f} {all(r["finite"] for r in rows)!s:>4}')
