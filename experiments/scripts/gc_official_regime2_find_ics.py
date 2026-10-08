"""Initial conditions in the chaotic sea of a draw of experiments/scripts/gc_official_scan.py (second regime for the NILSS test).

Every k of experiments/scripts/gc_official_candidates.initial_condition (k = 0: the scan initial condition; k > 0: random position and
sign of v_par on the same energy / mu surface) is classified by a Lyapunov run of length T (dt = 0.01, the step of the finite
differences and of NILSS): chaotic if the orbit stays bounded with s > 0, |dE| < 1e-6, lambda_1 * T > 20 and lambda_1 does not
decay by more than 30 % between the first half and the whole run (the rule of summarize_gc_official_candidates.py).
For a chaotic orbit the point of largest v_par^2 along 2000 time units is stored as `u0` too: re-projecting an initial
condition onto the energy shell of a shifted parameter, v_par = sign * sqrt(1 - lam B), needs 1 - lam B > 0, and the sea of a
weakly passing regime has v_par^2 of a few 1e-3 at an arbitrary point of the orbit but ~ 4e-2 at the bottom of the well.

    python experiments/scripts/gc_official_regime2_find_ics.py TASK_ID --seed 835 [--k0 0 --T 30000]       # one slurm task = k0 + TASK_ID
    python experiments/scripts/gc_official_regime2_find_ics.py --collect --seed 835 [--outdir DIR]         # writes DIR/ics.json
"""
import argparse
import glob
import json
import os

import numpy as np
import jax.numpy as jnp

from nilss_jax import lyapunov as lyap_tools
from nilss_jax.systems import guiding_center as go
from gc_official_candidates import initial_condition
from gc_official_scan import sample

DT = 0.01


def default_outdir(seed):
    return f'results/gc_official_regime2_seed{seed}_ics'


def classify(res, dE):
    ok = res['finite'] and res['x_max'] <= 1.0 and res['x_min'] > 0 and abs(dE) < 1e-6
    lam, lam_half = res['lyap'][0], res['lyap_finite_time'][len(res['lyap_finite_time']) // 2 - 1][0]
    if not ok:
        return 'escaped'
    return 'chaotic' if lam * res['T'] > 20 and lam > 0.7 * lam_half else 'regular'


def run_one(seed, k, T):
    p, u_scan, _ = sample(seed)
    u0 = np.array(initial_condition(seed, k, p, u_scan))
    runner = lyap_tools.make_runner(go.rhs4, DT, 100, obs=lambda u: u[jnp.array([0, 3])])
    res = lyap_tools.lyapunov_spectrum(go.rhs4, u0, DT, 0.0, T, checkpoints=4, args=(p,), runner=runner)
    dE = float(go.energy(jnp.array(res['u_final']), p)) - float(go.energy(jnp.array(u0), p)) if res['finite'] else float('nan')
    cls = classify(res, dE)
    out = {'seed': seed, 'k': k, 'class': cls, 'T': T, 'dt': DT, 'u_scan_k': u0.tolist(), 'dE': dE,
           'lyap1': res['lyap'][0], 'lyap1_half': res['lyap_finite_time'][len(res['lyap_finite_time']) // 2 - 1][0],
           'x_mean_parts': [float(x) for x in res['x_mean_parts']], 'x_max': float(res['x_max']), 'x_min': float(res['x_min']),
           'v_min': float(res['obs_min'][1]), 'v_max': float(res['obs_max'][1])}
    if cls == 'chaotic':
        tr = np.asarray(go.trajectory(go.rhs4, jnp.array(u0), p, DT, 200000, 10))
        j = int(np.argmax(tr[:, 3] ** 2))
        out['u0'] = tr[j].tolist()
        out['v2_safe'] = float(tr[j, 3] ** 2)
        out['v2_start'] = float(u0[3] ** 2)
    return out


def collect(seed, outdir):
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(outdir, 'task_*.json')))]
    rows.sort(key=lambda r: r['k'])
    print(f'{len(rows)} initial conditions of seed {seed} classified (T = {rows[0]["T"]:g}, dt = {rows[0]["dt"]})')
    for cls in ('chaotic', 'regular', 'escaped'):
        print(f'  {cls}: {sum(r["class"] == cls for r in rows)}')
    sea = [r for r in rows if r['class'] == 'chaotic']
    for r in sea:
        xm = np.array(r['x_mean_parts'])
        print(f"  k={r['k']:3d} lambda_1={r['lyap1']:.5f} <s>={xm.mean():.4f} s in [{r['x_min']:.4f},{r['x_max']:.3f}] "
              f"v_par^2: start {r['v2_start']:.4f}, bottom of the well {r['v2_safe']:.4f}")
    m = np.array([np.mean(r['x_mean_parts']) for r in sea])
    if len(m) > 1:
        print(f'  <s> over the chaotic initial conditions: mean {m.mean():.4f}, std {m.std(ddof=1):.4f}')
    path = os.path.join(outdir, 'ics.json')
    with open(path, 'w') as fh:
        json.dump({'seed': seed, 'T_classify': rows[0]['T'], 'ics': [{'k': r['k'], 'u0': r['u0'], 'u_scan_k': r['u_scan_k'],
                                                                        'lyap1': r['lyap1'], 'mean_s': float(np.mean(r['x_mean_parts']))} for r in sea]}, fh)
    print('wrote', path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task', type=int, nargs='?', default=0)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--k0', type=int, default=0)
    ap.add_argument('--T', type=float, default=30000.0)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--collect', action='store_true')
    args = ap.parse_args()
    outdir = args.outdir or default_outdir(args.seed)
    if args.collect:
        collect(args.seed, outdir)
        return
    k = args.k0 + args.task
    out = run_one(args.seed, k, args.T)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f'task_{k:03d}.json'), 'w') as fh:
        json.dump(out, fh)
    print(f"seed={args.seed} k={k:3d} {out['class']:8s} lambda_1={out['lyap1']:.5f} <s>={np.mean(out['x_mean_parts']):.4f} "
          f"s in [{out['x_min']:.4f},{out['x_max']:.3f}] dE={out['dE']:.1e}" +
          (f" v2 bottom {out['v2_safe']:.4f}" if out['class'] == 'chaotic' else ''))


if __name__ == '__main__':
    main()
