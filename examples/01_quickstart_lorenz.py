"""Quickstart: one NILSS run on Lorenz 63, with three parameters differentiated at once.

The smallest complete use of the library: a flow written with jax.numpy (here the bundled Lorenz 63), the objective J = z,
one call of NILSS.run, and the NILSSResult. Lorenz 63 is the validation problem of Ni & Wang (arXiv:1611.00880): at
rho = 28 the sensitivity d<z>/d(rho) is close to 1.0 and the leading Lyapunov exponent is 0.9056.

Expect: d<z>/d(rho) = 1.02 +- 0.01 and lambda_1 = 0.90 +- 0.01 (one trajectory of T = 1000: about 15 s, most of it the
compilation). sigma and beta are differentiated in the same run, sharing the homogeneous tangent: d<z>/d(sigma) about 0.13
and d<z>/d(beta) about -1.66. Finite differences of the ensemble average give 1.00, 0.14-0.15 and -1.65 (docs/usage.md, section 7):
NILSS agrees to about 2% for rho and beta and is further off for sigma, whose sensitivity is small.

    python examples/01_quickstart_lorenz.py [--fast]
"""
import argparse

from nilss_jax import NILSS
from nilss_jax.systems import lorenz63


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fast', action='store_true', help='shorter run (T = 100) for a quick check')
    args = ap.parse_args()
    T = 100.0 if args.fast else 1000.0

    # rhs(u, p) and J(u) are plain jax.numpy functions; the parameters named here are differentiated, all share one tangent.
    nilss = NILSS(lorenz63.rhs, lorenz63.J, params=('rho', 'sigma', 'beta'), dt=0.005, T_seg=0.5, nus=1)
    res = nilss.run(lorenz63.DEFAULTS, lorenz63.initial_condition(0), T=T, T_spinup=20.0)

    print(res.summary())
    print()
    print(f"published (Ni & Wang): d<z>/d(rho) = 1.0 and lambda_1 = 0.9056;  this run: {res.dJdp['rho']:.3f} and {res.lyapunov[0]:.3f}")
    print('one run is one trajectory: see examples/02 for the statistics of many runs and the reliability report,')
    print('and nilss_jax.fd (docs/usage.md) for the finite-difference cross-check')


if __name__ == '__main__':
    main()
