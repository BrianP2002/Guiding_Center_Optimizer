import os
import argparse
import warnings
from functools import partial
import numpy as np
import jax
jax.config.update('jax_enable_x64', True) # float32 is not enough for the tangent solves and the xi checks
import jax.numpy as jnp
from jax import jit, jacobian
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nilss_jax.reference import nilss

default_params = {'a0': 0.1, 'a1': 1.0, 'iota': 0.5, 'G': 1.0, 'lam': 0.1}
# one table for the parameter names and the ranges used by app_gc.py and app_gc_opt.py
PARAM_BOUNDS = {
    'a0': (0.05, 0.25),
    'a1': (0.9, 1.1),
    'iota': (0.4, 0.6),
    'lam': (0.01, 0.21),
    'G': (0.9, 1.1),
}
PARAM_ALIASES = {'lambda': 'lam'}
DEFAULT_DT = 0.001
# NILSS needs a positive Lyapunov exponent; if lambda_1 * T is below this, the exponent
# is not resolved by the trajectory (or the system is not chaotic at all)
LYAP_RESOLVED_GROWTH = 3.0


def canon_par(par):
    par = PARAM_ALIASES.get(par, par)
    if par not in default_params:
        raise KeyError(f'unknown parameter {par!r}, valid: {sorted(default_params)} (alias: {PARAM_ALIASES})')
    return par


@jit
def B_func(x, y, z, a0, a1):
    # Compute the B field
    return 1.0 + a0 * jnp.sqrt(x) * jnp.cos(y - a1 * z)

@jit
def V_func(lam, B):
    # Compute V based on B
    return jnp.sqrt(1.0 - lam * B)


def f_ode(x, y, z, a0, a1, iota, G, lam):
    B = B_func(x, y, z, a0, a1)
    invB = 1.0 / B
    factor = 2.0 / lam - B
    dBdy = -a0 * jnp.sqrt(x) * jnp.sin(y - a1 * z)
    dfdx = -invB * dBdy * factor
    dBdx = a0 * (1.0 / (2.0 * jnp.sqrt(x))) * jnp.cos(y - a1 * z)
    V = V_func(lam, B)
    dfdy = invB * dBdx * factor + (iota * V * B) / G
    dfdz = (B * V) / G
    return jnp.array([dfdx, dfdy, dfdz])

def f_ode_wrapper(u, params):
    x, y, z = u
    return f_ode(x, y, z, params['a0'], params['a1'], params['iota'], params['G'], params['lam'])


@partial(jit, static_argnames=['par'])
def ddt(uwvs, params, par):
    if par not in params:
        # new_params[par] below would silently add an unused key and give dfdpar = 0
        raise KeyError(f'unknown parameter {par!r}, valid: {sorted(params)}')
    u, w, vstar = uwvs
    dudt = f_ode_wrapper(u, params)
    Df = jacobian(f_ode_wrapper, argnums=0)(u, params)
    dwdt = jnp.dot(Df, w.T).T
    def f_par(p_val):
        new_params = params.copy()
        new_params[par] = p_val
        return f_ode_wrapper(u, new_params)
    dfdpar = jacobian(f_par)(params[par])
    dvstardt = jnp.dot(Df, vstar) + dfdpar
    return (dudt, dwdt, dvstardt)


def fJJu(u, params):
    f_val = f_ode_wrapper(u, params)
    return f_val, u[0], jnp.array([1.0, 0.0, 0.0])


@partial(jax.jit, static_argnames=['par'])
def RK4(uwvs, params, par, dt):
    k0 = tuple(dt * comp for comp in ddt(uwvs, params, par))
    uwvs_half = tuple(uwvs[i] + 0.5 * k0[i] for i in range(3))
    k1 = tuple(dt * comp for comp in ddt(uwvs_half, params, par))
    uwvs_half2 = tuple(uwvs[i] + 0.5 * k1[i] for i in range(3))
    k2 = tuple(dt * comp for comp in ddt(uwvs_half2, params, par))
    uwvs_full = tuple(uwvs[i] + k2[i] for i in range(3))
    k3 = tuple(dt * comp for comp in ddt(uwvs_full, params, par))
    new_uwvs = tuple(uwvs[i] + (k0[i] + 2 * k1[i] + 2 * k2[i] + k3[i]) / 6.0 for i in range(3))
    return new_uwvs


def make_problem(dt=DEFAULT_DT, base_params=None):
    """
    The (integrator, fJJu) pair in the interface of nilss.py:
    integrator(u, w, vstar, par, s) and fJJu(u, par, s), with parameter `par` set to s
    and all other parameters from base_params (default_params if None).
    The time step is bound here, once, and checked against the dt given to nilss().
    """
    base = {**default_params, **(base_params or {})}

    def integrator(u, w, vstar, par, s):
        par = canon_par(par)
        return RK4((u, w, vstar), {**base, par: float(s)}, par, dt)

    def fJJu_par(u, par, s):
        par = canon_par(par)
        return fJJu(u, {**base, par: float(s)})

    integrator.dt = dt
    return integrator, fJJu_par


def sample_u0(key):
    kx, ky, kz = jax.random.split(key, 3)
    return jnp.array([0.1 * jax.random.uniform(kx),
                      2.0 * jnp.pi * jax.random.uniform(ky),
                      2.0 * jnp.pi * jax.random.uniform(kz)])


def add_nilss_args(parser):
    parser.add_argument('--par', type=str, required=True,
                        choices=sorted(set(PARAM_BOUNDS) | set(PARAM_ALIASES)),
                        help='Parameter to vary')
    parser.add_argument('--nseg', type=int, default=200, help='number of segments')
    parser.add_argument('--T-seg', type=float, default=0.01, help='time length of each segment')
    parser.add_argument('--nseg-ps', type=int, default=200, help='number of segments to reach the attractor')
    parser.add_argument('--nus', type=int, default=1, help='number of homogeneous tangents')
    parser.add_argument('--dt', type=float, default=DEFAULT_DT, help='time step')
    parser.add_argument('--outdir', type=str, default='results')
    return parser


def main():
    parser = add_nilss_args(argparse.ArgumentParser(description='Run NILSS sensitivity analysis.'))
    parser.add_argument('--step-size', type=float, default=0.01, help='spacing of the parameter grid')
    parser.add_argument('--seed', type=int, default=20250301, help='seed of the random homogeneous tangents')
    parser.add_argument('--key-seed', type=int, default=20250407, help='seed of the initial condition')
    args = parser.parse_args()

    par = canon_par(args.par)
    np.random.seed(args.seed)
    par_lb, par_ub = PARAM_BOUNDS[par]
    num_steps = int(np.round((par_ub - par_lb) / args.step_size)) + 1
    par_arr = np.linspace(par_lb, par_ub, num_steps)

    # the same initial condition for every parameter value
    u0 = sample_u0(jax.random.PRNGKey(args.key_seed))
    integrator, fJJu_par = make_problem(args.dt)

    J_arr, dJdpar_arr, lyap_arr = [], [], []
    for par_value in par_arr:
        print(f'{par} = {par_value:.4f}, u0 = {np.asarray(u0)}', flush=True)
        J_val, dJdpar_val, info = nilss(args.dt, args.nseg, args.T_seg, args.nseg_ps, u0, args.nus,
                                        par, float(par_value), integrator, fJJu_par, return_info=True)
        J_arr.append(J_val)
        dJdpar_arr.append(dJdpar_val)
        lyap_arr.append(info['lyap'][0])
    J_arr, dJdpar_arr, lyap_arr = np.array(J_arr), np.array(dJdpar_arr), np.array(lyap_arr)
    T = info['T']

    unresolved = lyap_arr * T < LYAP_RESOLVED_GROWTH
    if unresolved.any():
        warnings.warn(f'lambda_1 * T < {LYAP_RESOLVED_GROWTH} for {unresolved.sum()} of {len(par_arr)} values of {par} '
                      f'(T = {T:g}, largest lambda_1 = {lyap_arr.max():.3g}): no positive Lyapunov exponent is resolved '
                      f'by the trajectory, so the shadowing assumptions of NILSS are not supported by this run.')

    os.makedirs(args.outdir, exist_ok=True)
    np.savez(os.path.join(args.outdir, f'guiding_center_{par}.npz'),
             par_arr=par_arr, J_arr=J_arr, dJdpar_arr=dJdpar_arr, lyap_arr=lyap_arr, T=T)

    plt.figure(figsize=[12, 12])
    plt.subplot(2, 1, 1)
    plt.plot(par_arr, J_arr, marker='o')
    plt.xlabel(rf'${par}$')
    plt.ylabel(r'$\langle x \rangle$')

    plt.subplot(2, 1, 2)
    plt.plot(par_arr, dJdpar_arr, marker='s')
    plt.xlabel(rf'${par}$')
    plt.ylabel(rf'$\frac{{d\langle J\rangle}}{{d {par}}}$')

    plt.savefig(os.path.join(args.outdir, f'guiding_center_{par}.png'))

if __name__ == '__main__':
    main()
