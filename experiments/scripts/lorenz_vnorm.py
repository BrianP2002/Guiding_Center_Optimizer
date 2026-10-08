"""Control for experiments/scripts/summarize_gc_official_nilss.py: the norm of the shadowing direction along a long Lorenz 63 orbit and the
tails of the per-segment contributions to dJ/drho, the same quantities that are heavy-tailed for the guiding-center sea.

    python experiments/scripts/lorenz_vnorm.py [--T 4000] [--rho 28]
"""
import argparse

import numpy as np
import jax.numpy as jnp

from nilss_jax.systems import lorenz63
from nilss_jax import NILSSStreamer


def rhs(u, p):
    return jnp.array([p['sigma'] * (u[1] - u[0]), u[0] * (p['rho'] - u[2]) - u[1], u[0] * u[1] - p['beta'] * u[2]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--T', type=float, default=4000.0)
    ap.add_argument('--rho', type=float, default=28.0)
    ap.add_argument('--nrun', type=int, default=6)
    ap.add_argument('--dt', type=float, default=0.005)
    ap.add_argument('--T_seg', type=float, default=0.5)
    args = ap.parse_args()
    p = {**lorenz63.DEFAULTS, 'rho': args.rho}
    streamer = NILSSStreamer(rhs, lambda u: u[2], ('rho',), args.dt, args.T_seg, 1)
    nseg = int(round(args.T / args.T_seg))
    print(f'Lorenz 63, rho = {args.rho}, T = {args.T:g}, T_seg = {args.T_seg}, {nseg} segments, {args.nrun} runs')
    for k in range(args.nrun):
        rng = np.random.RandomState(k)
        u0 = np.array([1.0, 1.0, 20.0]) + 0.1 * rng.randn(3)
        J, dJ, info = streamer.run_segments(p, u0, nseg, int(round(20.0 / args.T_seg)), seed=k)
        v, s = info['vnorm'][:, 0], info['dJdp_segments'][:, 0]
        a = np.sort(np.abs(s))
        z = (s - s.mean()) / s.std()
        print(f'  run {k}: dJ/drho = {dJ[0]:+.3f}  |v_perp|: median {np.median(v):.2f}, 99% {np.quantile(v, .99):.1f}, '
              f'99.9% {np.quantile(v, .999):.1f}, max {v.max():.1f};  contributions: top 1% carry {a[-nseg // 100:].sum() / a.sum():.2f}, '
              f'kurtosis {np.mean(z ** 4):.1f}')


if __name__ == '__main__':
    main()
