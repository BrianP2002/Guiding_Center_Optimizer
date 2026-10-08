"""Pitch-angle scattering added to the official guiding-center flow (gc_collisions.py), parameters of draw 581 of
experiments/scripts/gc_official_scan.py, finite-time ensemble objectives E[g(s_T)] and their derivative with respect to eps_t.

Initial states (the same for every parameter value and every collision frequency): s ~ U[0.02, 0.20], theta and zeta
uniform, sign +-1, xi0 = sign sqrt(1 - lam B(p0)), points with 1 - lam B(p0) < 0.02 dropped (gc_collisions.make_ics).
Collision frequencies NUS = 0, 1e-4, 1e-3, 1e-2 (a deeply trapped particle, xi ~ 0.2, is detrapped in about xi^2 / nu =
400, 40, 4 time units; the bounce time is of the order of 50 .. 100). Horizons TS = 300, 1000, 3000, 10000.

    python experiments/scripts/gc_collision.py resp TASK     response curves: independent paths at every eps_t of a grid, one slurm task =
                                                 (nu, grid point, block of paths); task = (inu * npoint + ipoint) * nblock + iblock
    python experiments/scripts/gc_collision.py crn  TASK     pathwise derivative with common random numbers, one task = (nu, block of paths);
                                                 task = inu * nblock + iblock
    python experiments/scripts/gc_collision.py pilot timing|dt|frac|lyap [NU_INDEX]     sizes the experiments, Lyapunov exponent of the noisy flow
"""
import argparse
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

from nilss_jax.systems import guiding_center as go
import gc_collisions as gcc
from gc_official_scan import sample

SEED = 581
NUS = (0.0, 1e-4, 1e-3, 1e-2)
TS = (300.0, 1000.0, 3000.0, 10000.0)
IC_SEED = 12345
NOISE_SEED = 20261007
S_C, S_W = 0.15, 0.02               # g2 = logistic((s - S_C) / S_W)


def base():
    return sample(SEED)[0]


def keys_for(ids):
    return gcc.path_keys(jax.random.PRNGKey(NOISE_SEED), np.asarray(ids))


def save(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)


def run_resp(args):
    inu, rest = divmod(args.task, args.npoint * args.nblock)
    ipoint, iblock = divmod(rest, args.nblock)
    p, nu = base(), NUS[inu]
    eps = p['eps_t'] + args.h * (ipoint - args.npoint // 2)
    ics = gcc.make_ics(args.nblock * args.nper, p, seed=IC_SEED)[iblock * args.nper:(iblock + 1) * args.nper]
    ids = (inu * args.npoint + ipoint) * 1_000_000 + iblock * args.nper + np.arange(args.nper)      # independent noise per grid point
    t0 = time.time()
    out = gcc.run_paths(ics, keys_for(ids), p, eps, nu, args.dt, args.Ts)
    save(os.path.join(args.outdir, f'task_{args.task:04d}.npz'), inu=inu, nu=nu, ipoint=ipoint, iblock=iblock, eps_t=eps,
         dt=args.dt, elapsed=time.time() - t0, T=out['T'], s=out['s'], xi=out['xi'], s_min=out['s_min'], s_max=out['s_max'])
    print(f"resp task {args.task}: nu={nu:g} eps_t={eps:.5f} block {iblock}: {args.nper} paths, T={args.Ts[-1]:g}, dt={args.dt} "
          f"({time.time() - t0:.0f} s); <s_T> = {np.round(out['s'].mean(axis=1), 4)}, finite {int(np.isfinite(out['s']).all(axis=0).sum())}")


def run_crn(args):
    inu, iblock = divmod(args.task, args.nblock)
    p, nu = base(), NUS[inu]
    ics = gcc.make_ics(args.nblock * args.nper, p, seed=IC_SEED)[iblock * args.nper:(iblock + 1) * args.nper]
    ids = 500_000_000 + inu * 1_000_000 + iblock * args.nper + np.arange(args.nper)
    t0 = time.time()
    out = gcc.run_paths(ics, keys_for(ids), p, p['eps_t'], nu, args.dt, args.Ts, tangent=True, coupling=args.coupling)
    save(os.path.join(args.outdir, f'task_{args.task:04d}.npz'), inu=inu, nu=nu, iblock=iblock, eps_t=p['eps_t'], dt=args.dt,
         coupling=args.coupling, elapsed=time.time() - t0, T=out['T'], s=out['s'], xi=out['xi'], ds=out['ds'], dxi=out['dxi'], s_min=out['s_min'],
         s_max=out['s_max'])
    print(f"crn task {args.task} ({args.coupling} coupling): nu={nu:g} block {iblock}: {args.nper} paths, T={args.Ts[-1]:g}, dt={args.dt} ({time.time() - t0:.0f} s); "
          f"mean ds/deps_t = {np.round(np.mean(out['ds'], axis=1), 3)}, finite {int(np.isfinite(out['ds']).all(axis=0).sum())}")


# ---- pilots --------------------------------------------------------------------------------------------------------

def pilot_timing(args):
    p = base()
    ics = gcc.make_ics(1000, p, seed=IC_SEED)
    for nu, tangent in ((0.0, False), (1e-3, False), (0.0, True), (1e-3, True)):
        keys = keys_for(np.arange(1000))
        gcc.run_paths(ics, keys, p, p['eps_t'], nu, args.dt, [50.0], tangent=tangent)         # compile
        t0 = time.time()
        gcc.run_paths(ics, keys, p, p['eps_t'], nu, args.dt, [50.0, 100.0], tangent=tangent)
        el = time.time() - t0
        print(f"nu={nu:g} tangent={tangent!s:5s} dt={args.dt}: {el:.1f} s for 1000 paths x {int(100 / args.dt)} steps = "
              f"{el / (1000 * 100 / args.dt) * 1e6:.2f} us per path-step", flush=True)


def mu_drift(u0, uf, p):
    f = jax.vmap(lambda u: gcc.mu(u, p))
    m0, m1 = np.asarray(f(jnp.asarray(u0))), np.asarray(f(jnp.asarray(uf)))
    return m1 / m0 - 1.0


def pilot_dt(args):
    p = base()
    n = args.n
    ics = gcc.make_ics(n, p, seed=IC_SEED)
    keys = keys_for(np.arange(n))
    pe = {k: float(v) for k, v in p.items()}
    Ts = [300.0, 1000.0]
    dts = [0.025, 0.05, 0.1, 0.2]
    res = {}
    for dt in dts:
        t0 = time.time()
        res[dt] = gcc.run_paths(ics, keys, p, p['eps_t'], 0.0, dt, Ts)
        print(f'dt={dt}: {time.time() - t0:.0f} s', flush=True)
    ref = res[dts[0]]
    print(f'collisionless, {n} paths, differences from dt={dts[0]} (T = 300 / 1000):')
    for dt in dts[1:]:
        for k, T in enumerate(Ts):
            d = np.abs(res[dt]['s'][k] - ref['s'][k])
            sem = np.std(res[dt]['s'][k] - ref['s'][k], ddof=1) / np.sqrt(n)
            print(f"  dt={dt} T={T:g}: |ds| median {np.median(d):.2e}, 90% {np.quantile(d, .9):.2e}, max {d.max():.2e};  "
                  f"mean(s) {res[dt]['s'][k].mean():.5f} vs {ref['s'][k].mean():.5f}, difference/SEM {np.mean(res[dt]['s'][k] - ref['s'][k]) / sem:+.1f}")
    print('mu(T = 1000) / mu(0) - 1 (conserved by the flow):')
    for dt in dts:
        d = mu_drift(ics, res[dt]['u_final'], pe)
        print(f"  dt={dt}: median |.| {np.median(np.abs(d)):.1e}, 99% {np.quantile(np.abs(d), .99):.1e}, max {np.abs(d).max():.1e}")
    nu = 1e-3
    print(f'nu = {nu:g}, independent noise for every dt: ensemble means at T = 300 / 1000')
    stoch = {}
    for dt in dts:
        stoch[dt] = gcc.run_paths(ics, keys, p, p['eps_t'], nu, dt, Ts)
    for k, T in enumerate(Ts):
        for name, f in (('s', lambda r: r['s'][k]), ('xi^2', lambda r: r['xi'][k] ** 2),
                        ('g2', lambda r: 1 / (1 + np.exp(-(r['s'][k] - S_C) / S_W)))):
            print(f'  T={T:g} {name:5s}: ' + ', '.join(
                f"dt={dt}: {f(stoch[dt]).mean():.5f} (SEM {f(stoch[dt]).std(ddof=1) / np.sqrt(n):.5f})" for dt in dts))


def pilot_frac(args):
    p = base()
    inu = args.nu_index
    n = args.n
    ics = gcc.make_ics(n, p, seed=IC_SEED)
    keys = keys_for(10_000_000 + np.arange(n))
    t0 = time.time()
    out = gcc.run_paths(ics, keys, p, p['eps_t'], NUS[inu], args.dt, args.Ts)
    print(f'nu={NUS[inu]:g}, {n} paths, dt={args.dt}: {time.time() - t0:.0f} s')
    print(f"initial: <s> {ics[:, 0].mean():.4f}, fraction above S_C {np.mean(ics[:, 0] > S_C):.3f}, <|xi|> {np.abs(ics[:, 5]).mean():.3f}")
    for k, T in enumerate(out['T']):
        s = out['s'][k]
        ok = np.isfinite(s)
        print(f"T={T:g}: finite {ok.mean():.4f}; <s> {s[ok].mean():.4f}, quantiles(1,10,50,90,99%) {np.round(np.quantile(s[ok], [.01, .1, .5, .9, .99]), 4)}; "
              f"fraction s>S_C {np.mean(s[ok] > S_C):.3f}, s>0.25 {np.mean(s[ok] > 0.25):.3f}; <xi^2> {np.mean(out['xi'][k][ok] ** 2):.3f}")
    print(f"s_min over the run: quantiles(0,1,10,50%) {np.round(np.quantile(out['s_min'], [0, .01, .1, .5]), 5)}; fraction below 2e-3 {np.mean(out['s_min'] < 2e-3):.4f}; "
          f"s_max quantiles(50,99,100%) {np.round(np.quantile(out['s_max'], [.5, .99, 1.0]), 3)}")


def pilot_lyap(args):
    """Top Lyapunov exponent of the noisy flow under the same noise (synchronous coupling): two copies of every path, the
    second displaced by delta in s, the same random numbers; growth of the separation sqrt(ds^2 + dxi^2)."""
    p = base()
    n, delta = args.n, 1e-10
    ics = gcc.make_ics(n, p, seed=IC_SEED)
    ics2 = ics.copy()
    ics2[:, 0] += delta
    keys = keys_for(20_000_000 + np.arange(n))
    Ts = [50.0 * k for k in range(1, 11)]
    for inu, nu in enumerate(NUS):
        a = gcc.run_paths(ics, keys, p, p['eps_t'], nu, args.dt, Ts, coupling=args.coupling)
        b = gcc.run_paths(ics2, keys, p, p['eps_t'], nu, args.dt, Ts, coupling=args.coupling)
        r = np.sqrt((a['s'] - b['s']) ** 2 + (a['xi'] - b['xi']) ** 2)                    # [nT, n]
        ok = np.all(np.isfinite(r), axis=0) & np.all(r > 0, axis=0)
        lr = np.log(r[:, ok] / delta)
        T = np.array(Ts)
        sel = (T >= 150) & (T <= 450)
        gam = lambda y: np.polyfit(T[sel], y[sel], 1)[0]
        print(f'nu={nu:g}: {int(ok.sum())} paths; log(separation/delta) at T = {Ts[1]:g}, {Ts[4]:g}, {Ts[8]:g}: median '
              f'{np.round(np.median(lr, axis=1)[[1, 4, 8]], 2)}, 90% {np.round(np.quantile(lr, .9, axis=1)[[1, 4, 8]], 2)}; '
              f'growth rate of the median {gam(np.median(lr, axis=1)):.4f}, of the 90% quantile {gam(np.quantile(lr, .9, axis=1)):.4f}, '
              f'mean of log (the Lyapunov exponent) {gam(lr.mean(axis=1)):.4f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['resp', 'crn', 'pilot'])
    ap.add_argument('what', nargs='?', default='0', help='task id (resp, crn) or timing|dt|frac (pilot)')
    ap.add_argument('nu_index', nargs='?', type=int, default=0)
    ap.add_argument('--dt', type=float, default=0.05)
    ap.add_argument('--nper', type=int, default=1000, help='paths per task')
    ap.add_argument('--nblock', type=int, default=None, help='blocks of paths per (nu, point) (resp default 4) or per nu (crn default 8)')
    ap.add_argument('--npoint', type=int, default=11)
    ap.add_argument('--h', type=float, default=0.0025, help='grid spacing in eps_t')
    ap.add_argument('--Ts', type=lambda s: tuple(float(x) for x in s.split(',')), default=TS)
    ap.add_argument('--n', type=int, default=500, help='paths of a pilot')
    ap.add_argument('--coupling', choices=['rotation', 'gradient'], default='rotation', help='coupling of the collision noise of nearby paths')
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()
    if args.mode == 'resp':
        args.task, args.nblock = int(args.what), args.nblock or 4
        args.outdir = args.outdir or 'results/gc_collision_resp'
        run_resp(args)
    elif args.mode == 'crn':
        args.task, args.nblock = int(args.what), args.nblock or 8
        args.outdir = args.outdir or 'results/gc_collision_crn'
        run_crn(args)
    else:
        {'timing': pilot_timing, 'dt': pilot_dt, 'frac': pilot_frac, 'lyap': pilot_lyap}[args.what](args)


if __name__ == '__main__':
    main()
