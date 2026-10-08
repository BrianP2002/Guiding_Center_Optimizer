"""Summarize results/gc_symmetry_breaking/*.json (statistics over initial conditions)."""
import glob
import json
import sys
from collections import defaultdict

import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else 'results/gc_symmetry_breaking'
groups = defaultdict(list)
for fn in sorted(glob.glob(f'{d}/task_*.json')):
    r = json.load(open(fn))
    groups[(r['b1'], r['b0'])].append(r)
T = next(iter(groups.values()))[0]['T']
print(f'T = {T:g}; regular flow gives lambda_1 ~ log(T)/T ~ {np.log(T)/T:.4f}')
print(f'{"b1":>4} {"b0":>6} {"n":>2} | {"lam1 per IC":>34} | {"<x> per IC (mean over parts)":>34} {"x_in":>6} {"fin":>4}')
for (b1, b0), rows in sorted(groups.items()):
    l1 = ' '.join(f'{r["lyap"][0]:7.4f}' for r in rows)
    xm = ' '.join(f'{np.mean(r["x_mean_parts"]):7.3f}' for r in rows)
    xin = np.mean([np.std(r['x_mean_parts']) for r in rows])
    print(f'{b1:4.1f} {b0:6.3f} {len(rows):2d} | {l1:>34} | {xm:>34} {xin:6.3f} {all(r["finite"] for r in rows)!s:>4}')
