"""Guiding-center motion in Boozer coordinates, vacuum field (I = 0, K = 0, G = const), JAX, float64.

The equations are the "gc_vac" mode of SIMSOPT (docs: Field line and particle tracing) and FIRM3D (docs:
Guiding Center Integration), which come from the Littlejohn Lagrangian:

    ds/dt     = -B_theta W
    dtheta/dt =  B_s W + iota v_par B / G
    dzeta/dt  =  v_par B / G
    dv_par/dt = -(iota B_theta + B_zeta) mu B / G,         W = kappa (v_par^2 / B + mu),  kappa = m / (q psi0)

with mu = v_perp^2 / (2 B) conserved and E = v_par^2 / 2 + mu B conserved. Units: speed v = 1 (so mu = lam / 2,
lam = v_perp^2 / (v^2 B) the pitch variable of legacy/app_gc.py), B0 = 1, time in G/(B0 v). kappa is the
dimensionless drift strength (m v G / (q B0 psi0) in the original units): it multiplies the radial and poloidal drift
relative to the streaming along the field.

The state is u = (s, theta, zeta, v_par) in rhs4 (the official system, v_par may change sign: trapped
particles). rhs3 eliminates v_par through the energy, v_par = +sqrt(1 - lam B) (passing particles only), as the
model of legacy/app_gc.py does. That model is rhs3 with eps_t = eps_m = eps_2 = 0 and kappa = 2 / lam.

The field is B = 1 + sqrt(s) [eps_h cos(theta - N zeta) + eps_t cos(theta) + eps_2 cos(theta - N2 zeta)]
                   + eps_m cos(N zeta).
eps_h alone is the first-order near-axis field of SIMSOPT's BoozerAnalytic (B0 = 1, eps_h = etabar sqrt(2 psi0 / Bbar),
s = normalised toroidal flux); it is quasi-helically symmetric. eps_t, eps_m, eps_2 break the symmetry.
"""
import json
from functools import partial
from importlib import resources

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax import jit, jacobian

default_params = {'eps_h': 0.1, 'N': 1.0, 'eps_t': 0.0, 'eps_m': 0.0, 'eps_2': 0.0, 'N2': 2.0,
                  'iota': 0.5, 'G': 1.0, 'lam': 0.1, 'kappa': 20.0}
DEFAULT_DT = 0.001


def teacher_params(a0=0.1, a1=1.0, iota=0.5, G=1.0, lam=0.1):
    """The parameters of legacy/app_gc.py in the notation of this module (kappa is tied to lam there)."""
    return {**default_params, 'eps_h': a0, 'N': a1, 'iota': iota, 'G': G, 'lam': lam, 'kappa': 2.0 / lam}


def B_derivs(s, theta, zeta, p):
    """B and its partial derivatives (B, B_s, B_theta, B_zeta)."""
    r = jnp.sqrt(s)
    c1, s1 = jnp.cos(theta - p['N'] * zeta), jnp.sin(theta - p['N'] * zeta)
    c2, s2 = jnp.cos(theta), jnp.sin(theta)
    c3, s3 = jnp.cos(theta - p['N2'] * zeta), jnp.sin(theta - p['N2'] * zeta)
    cm, sm = jnp.cos(p['N'] * zeta), jnp.sin(p['N'] * zeta)
    harm = p['eps_h'] * c1 + p['eps_t'] * c2 + p['eps_2'] * c3
    B = 1.0 + r * harm + p['eps_m'] * cm
    B_s = harm / (2.0 * r)
    B_theta = -r * (p['eps_h'] * s1 + p['eps_t'] * s2 + p['eps_2'] * s3)
    B_zeta = r * (p['eps_h'] * p['N'] * s1 + p['eps_2'] * p['N2'] * s3) - p['eps_m'] * p['N'] * sm
    return B, B_s, B_theta, B_zeta


def pitch_vpar(u3, p):
    """v_par = sqrt(1 - lam B) at the point (s, theta, zeta) of an orbit with v = 1."""
    return jnp.sqrt(1.0 - p['lam'] * B_derivs(u3[0], u3[1], u3[2], p)[0])


def rhs4(u, p):
    """The official system for u = (s, theta, zeta, v_par)."""
    s, theta, zeta, v = u
    B, B_s, B_theta, B_zeta = B_derivs(s, theta, zeta, p)
    mu = 0.5 * p['lam']
    W = p['kappa'] * (v * v / B + mu)
    return jnp.array([-B_theta * W,
                      B_s * W + p['iota'] * v * B / p['G'],
                      v * B / p['G'],
                      -(p['iota'] * B_theta + B_zeta) * mu * B / p['G']])


def rhs3(u, p):
    """The same flow with v_par = +sqrt(1 - lam B) eliminated, u = (s, theta, zeta)."""
    s, theta, zeta = u
    B, B_s, B_theta, _ = B_derivs(s, theta, zeta, p)
    v = jnp.sqrt(1.0 - p['lam'] * B)
    W = p['kappa'] * (1.0 - 0.5 * p['lam'] * B) / B       # kappa (v_par^2 / B + mu)
    return jnp.array([-B_theta * W,
                      B_s * W + p['iota'] * v * B / p['G'],
                      v * B / p['G']])


def energy(u4, p):
    """v_par^2 / 2 + mu B, conserved by rhs4."""
    return 0.5 * u4[3] ** 2 + 0.5 * p['lam'] * B_derivs(u4[0], u4[1], u4[2], p)[0]


def phi_qh(s, theta, zeta, v, p):
    """G v_par / B - (iota - N) s / kappa. It is conserved when B depends on theta - N zeta only
    (eps_t = eps_m = eps_2 = 0): the flow is then integrable (two invariants, E and phi_qh)."""
    B = B_derivs(s, theta, zeta, p)[0]
    return p['G'] * v / B - (p['iota'] - p['N']) * s / p['kappa']


@partial(jit, static_argnames=['rhs', 'nstep', 'stride'])
def trajectory(rhs, u0, p, dt, nstep, stride=1):
    """RK4 states every `stride` steps (nstep // stride of them) of du/dt = rhs(u, p)."""
    def step(u, _):
        k1 = rhs(u, p)
        k2 = rhs(u + 0.5 * dt * k1, p)
        k3 = rhs(u + 0.5 * dt * k2, p)
        k4 = rhs(u + dt * k3, p)
        return u + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), None

    def block(u, _):
        u, _ = jax.lax.scan(step, u, None, length=stride)
        return u, u
    return jax.lax.scan(block, jnp.asarray(u0, dtype=float), None, length=nstep // stride)[1]


# ---- the interface of nilss_jax.reference, for rhs3 (J = s, the mean flux label) -----------------------------------

@partial(jit, static_argnames=['par'])
def ddt(uwvs, params, par):
    if par not in params:
        raise KeyError(f'unknown parameter {par!r}, valid: {sorted(params)}')
    u, w, vstar = uwvs
    dudt = rhs3(u, params)
    Df = jacobian(rhs3, argnums=0)(u, params)
    dwdt = jnp.dot(Df, w.T).T

    def f_par(p_val):
        return rhs3(u, {**params, par: p_val})
    dvstardt = jnp.dot(Df, vstar) + jacobian(f_par)(params[par])
    return (dudt, dwdt, dvstardt)


@partial(jit, static_argnames=['par'])
def RK4(uwvs, params, par, dt):
    k0 = tuple(dt * c for c in ddt(uwvs, params, par))
    k1 = tuple(dt * c for c in ddt(tuple(uwvs[i] + 0.5 * k0[i] for i in range(3)), params, par))
    k2 = tuple(dt * c for c in ddt(tuple(uwvs[i] + 0.5 * k1[i] for i in range(3)), params, par))
    k3 = tuple(dt * c for c in ddt(tuple(uwvs[i] + k2[i] for i in range(3)), params, par))
    return tuple(uwvs[i] + (k0[i] + 2 * k1[i] + 2 * k2[i] + k3[i]) / 6.0 for i in range(3))


def make_problem(dt=DEFAULT_DT, base_params=None):
    """(integrator, fJJu) in the interface of nilss_jax.reference for the passing-particle flow rhs3; any key of
    default_params can be the parameter. The time step is bound here and checked by nilss()."""
    base = {**default_params, **(base_params or {})}

    def check(par):
        if par not in base:
            raise KeyError(f'unknown parameter {par!r}, valid: {sorted(base)}')

    def integrator(u, w, vstar, par, s):
        check(par)
        return RK4((u, w, vstar), {**base, par: float(s)}, par, dt)

    def fJJu(u, par, s):
        check(par)
        return rhs3(u, {**base, par: float(s)}), u[0], jnp.array([1.0, 0.0, 0.0])

    integrator.dt = dt
    return integrator, fJJu


# ---- convenience for NILSS runs on the official flow ------------------------------------------------------------------

def mean_radius(u):
    """The objective used throughout the studies: the flux label s of the orbit, J = s (u = (s, theta, zeta, v_par))."""
    return u[0]


def project_to_energy_shell(u, p, sign=None):
    """Return ``u`` with v_par replaced by ``sign * sqrt(1 - lam B)``: the point of the shell E = 1/2 (v = 1) above the position
    (s, theta, zeta). ``sign`` defaults to the sign of the current v_par. Raises ValueError where the position is forbidden."""
    u = np.array(u, dtype=float)
    v2 = 1.0 - p['lam'] * float(B_derivs(u[0], u[1], u[2], p)[0])
    if v2 <= 0.0:
        raise ValueError('the position is not accessible: 1 - lam * B <= 0')
    u[3] = (np.sign(u[3]) if sign is None else sign) * np.sqrt(v2)
    return u


def make_nilss(pars, dt=0.01, T_seg=200.0, nus=1):
    """:class:`nilss_jax.NILSS` for the 4D official flow with J = s, on the energy surface (the invariant is the energy)."""
    from ..core import NILSS
    return NILSS(rhs4, mean_radius, pars, dt, T_seg, nus, invariant=energy)


def load_draw(seed):
    """The two bounded chaotic regimes found by the random scan of the official flow (``seed`` 581 or 835).

    Returns a dict with ``params`` (a full parameter dict), ``sea_ics`` (initial conditions on the chaotic sea, arrays of
    4 numbers) and the finite-difference and NILSS reference results of the studies in ``docs/findings.md``
    (``reference_fd`` and ``reference_nilss``: slopes d<s>/dp with their errors).
    """
    with resources.files('nilss_jax.systems').joinpath('data/gc_draws.json').open() as fh:
        draws = json.load(fh)
    if str(seed) not in draws:
        raise KeyError(f'unknown draw {seed!r}, available: {sorted(draws)}')
    d = draws[str(seed)]
    d['sea_ics'] = [np.array(u) for u in d['sea_ics']]
    return d
