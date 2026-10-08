"""NILSS in near-axis Cartesian coordinates: is the failure of NILSS in the chaotic sea an artefact of the coordinates?

guiding_center.rhs4 is written in (s, theta, zeta, v_par), where (s, theta) are polar coordinates of the poloidal plane
(r = sqrt(s)). NILSS minimises the integral of |v|^2 in the coordinates of the state, so a trajectory that passes near the magnetic
axis (draw 835: s_min = 0.0035) has a large theta-component of the shadowing direction for a purely geometric reason, and
|v| is not a coordinate-free quantity. Here the same flow is integrated in X = sqrt(s) cos(theta), Y = sqrt(s) sin(theta),
zeta, v_par (B is linear in X, Y: the vector field is smooth through the axis), J = s = X^2 + Y^2, the energy is the same
function. Same orbits as experiments/scripts/gc_official_nilss.py (same initial states, perturbation 1e-6 mapped to the plane), same
output format, so experiments/scripts/summarize_gc_official_nilss.py reads the results; |v_perp| is now the norm in the Cartesian metric.

    python experiments/scripts/gc_official_regime2_nilss_cart.py TASK_ID [same options as gc_official_nilss.py]
    python experiments/scripts/gc_official_regime2_nilss_cart.py 0 --selftest        # 200 time units in both coordinates
"""
import argparse
import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

from nilss_jax.systems import guiding_center as go
from nilss_jax import NILSSStreamer
from gc_official_scan import sample
from gc_official_fd import SEED, SEA_ICS, T_SPINUP, copy_initial_condition, load_ics, parse_ints
from gc_official_nilss import PARS, runs_for


def to_polar(u):
    return jnp.array([u[0] ** 2 + u[1] ** 2, jnp.arctan2(u[1], u[0]), u[2], u[3]])


def to_cart(v):
    r = np.sqrt(v[0])
    return np.array([r * np.cos(v[1]), r * np.sin(v[1]), v[2], v[3]])


def rhs_cart(u, p):
    s, th = u[0] ** 2 + u[1] ** 2, jnp.arctan2(u[1], u[0])
    ds, dth, dze, dv = go.rhs4(jnp.array([s, th, u[2], u[3]]), p)
    r = jnp.sqrt(s)
    return jnp.array([jnp.cos(th) * ds / (2 * r) - r * jnp.sin(th) * dth,
                      jnp.sin(th) * ds / (2 * r) + r * jnp.cos(th) * dth, dze, dv])


def energy_cart(u, p):
    return go.energy(to_polar(u), p)


def selftest(args):
    p = sample(args.seed)[0]
    k, u0 = copy_initial_condition(args.seed, args.sea_ics, 0, p, load_ics(args.ics_file) if args.ics_file else None)
    n = 20000
    tr_p = np.asarray(go.trajectory(go.rhs4, jnp.array(u0), p, args.dt, n, 100))
    tr_c = np.asarray(go.trajectory(rhs_cart, jnp.array(to_cart(u0)), p, args.dt, n, 100))
    s_c = tr_c[:, 0] ** 2 + tr_c[:, 1] ** 2
    E = np.array([float(energy_cart(jnp.array(x), p)) for x in tr_c])
    print(f'selftest, {n * args.dt:g} time units from the IC k={k}: max |s_cart - s_polar| = {np.max(np.abs(s_c - tr_p[:, 0])):.2e}, '
          f'max |v_par difference| = {np.max(np.abs(tr_c[:, 3] - tr_p[:, 3])):.2e}, energy drift in Cartesian coordinates {np.max(np.abs(E - E[0])):.1e}, '
          f's in [{tr_p[:, 0].min():.4f}, {tr_p[:, 0].max():.4f}]')


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
    ap.add_argument('--outdir', default='results/gc_official_nilss_cart')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        selftest(args)
        return
    pars = tuple(args.pars.split(','))

    copy, nus, T_seg = runs_for(args.layout, args.n_main, args.n_nus2)[args.task]
    p = sample(args.seed)[0]
    k, u0 = copy_initial_condition(args.seed, args.sea_ics, copy, p, load_ics(args.ics_file) if args.ics_file else None)
    u0c = to_cart(u0)

    streamer = NILSSStreamer(rhs_cart, lambda u: u[0] ** 2 + u[1] ** 2, pars, args.dt, T_seg, nus, invariant=energy_cart)
    nseg, nseg_ps = int(round(args.T / T_seg)), int(round(args.t_spinup / T_seg))
    t0 = time.time()
    Javg, dJdp, info = streamer.run_segments(p, u0c, nseg, nseg_ps, seed=1000 + args.task)
    seg = info['dJdp_segments']
    blocks = seg[:nseg // 20 * 20].reshape(20, -1, len(pars)).sum(axis=1)
    out = {'task': args.task, 'copy': copy, 'ic_k': k, 'nus': nus, 'T_seg': T_seg, 'T': info['T'], 'dt': args.dt,
           'pars': pars, 'Javg': float(Javg), 'dJdp': np.asarray(dJdp).tolist(), 'lyap': np.asarray(info['lyap']).tolist(),
           'blocks': blocks.tolist(), 'segments': seg.tolist(), 'vnorm': info['vnorm'].tolist(), 'elapsed': time.time() - t0,
           'seed': args.seed, 't_spinup': args.t_spinup, 'coords': 'cartesian'}
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, f'task_{args.task:03d}.json'), 'w') as fh:
        json.dump(out, fh)
    print(f"task {args.task} copy={copy} nus={nus} T_seg={T_seg:g} (Cartesian): <s>={Javg:.4f} lyap={np.round(info['lyap'], 5)} "
          f"dJ/d{pars} = {np.round(dJdp, 3)}  ({time.time() - t0:.0f} s)")


if __name__ == '__main__':
    main()
