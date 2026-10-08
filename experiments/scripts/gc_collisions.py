"""The official vacuum Boozer guiding-center flow (nilss_jax.systems.guiding_center) with pitch-angle scattering, JAX float64.

Prototype of the "finite-time objective + collisions" setting. The collisionless flow is guiding_center.rhs4 written for the
pitch xi = v_par / v at speed v = 1, so that mu = (1 - xi^2) / (2 B) is a function of the state instead of a parameter:

    ds/dt     = -B_theta W,                W = kappa (v_par^2 / B + mu) = kappa (1 + xi^2) / (2 B)
    dtheta/dt =  B_s W + iota xi B / G
    dzeta/dt  =  xi B / G
    dxi/dt    = -(iota B_theta + B_zeta) (1 - xi^2) / (2 G)

(for a collisionless orbit mu is constant and this is guiding_center.rhs4 with lam = (1 - xi0^2) / B0). The Lorentz operator
scatters the direction of the velocity, n = (n_1, n_2, xi), on the unit sphere and keeps the speed, hence the energy:

    d xi = -nu xi dt + sqrt(nu (1 - xi^2)) dW,        generator (nu / 2) d/dxi (1 - xi^2) d/dxi
         = (nu / 2) x Laplace-Beltrami of the sphere restricted to functions of xi.

The state is u = (s, theta, zeta, n_1, n_2, n_3 = xi) with |n| = 1. n_1, n_2 hold the gyrophase and are only carried along
(dn/dt = c (xi n - e_3), c = (iota B_theta + B_zeta) / (2 G), whose third component is the equation of xi above). With this
choice the flow is smooth on the whole sphere, and a collision step is a random rotation of n, exp([a]_x) n with
a = sqrt(nu dt) Z, Z ~ N(0, I_3): an isometry of the sphere. The derivative of a path with respect to a parameter, for the
same random numbers (CRN), therefore has no stochastic term and is not amplified by the collisions (an additive noise in
the pitch angle chi would give a Jacobian 1 - nu dt / (2 sin^2 chi), singular at the poles).

Time stepping: Lie splitting, one RK4 step of the flow, then one collision rotation. The random numbers of a path are a
function of (key, chunk index), so that a path can be restarted and two parameter values can share the noise.

The law of a single path does not depend on how the noise of two nearby paths is coupled, but the pathwise derivative does.
coupling='rotation' (default) is the isometric coupling above (the same ambient rotation for every path: no contraction or
expansion from the collisions); coupling='gradient' is the gradient Brownian motion of the sphere,
n' = normalise(n + sqrt(nu dt) (z - (z.n) n)), whose nearby paths contract on average (rate of order nu).
"""
from functools import partial

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from nilss_jax.systems import guiding_center as go

CHUNK = 250                 # steps drawn together from the random number generator


def rhs(u, p):
    """Deterministic part, u = (s, theta, zeta, n1, n2, xi)."""
    s, theta, zeta, n1, n2, xi = u
    B, B_s, B_theta, B_zeta = go.B_derivs(s, theta, zeta, p)
    W = p['kappa'] * (1.0 + xi * xi) / (2.0 * B)
    c = (p['iota'] * B_theta + B_zeta) / (2.0 * p['G'])
    return jnp.array([-B_theta * W,
                      B_s * W + p['iota'] * xi * B / p['G'],
                      xi * B / p['G'],
                      c * xi * n1,
                      c * xi * n2,
                      c * (xi * xi - 1.0)])


def mu(u, p):
    """Magnetic moment (1 - xi^2) / (2 B) at the speed v = 1: conserved by rhs, changed by collisions."""
    return (1.0 - u[5] ** 2) / (2.0 * go.B_derivs(u[0], u[1], u[2], p)[0])


def rk4_step(u, p, dt):
    k1 = rhs(u, p)
    k2 = rhs(u + 0.5 * dt * k1, p)
    k3 = rhs(u + 0.5 * dt * k2, p)
    k4 = rhs(u + dt * k3, p)
    return u + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def rotate(n, a):
    """Rotation of n about the axis a / |a| by the angle |a| (Rodrigues); smooth at a = 0."""
    th2 = a @ a
    small = th2 < 1e-8
    th2s = jnp.where(small, 1.0, th2)
    th = jnp.sqrt(th2s)
    A = jnp.where(small, 1.0 - th2 / 2.0, jnp.cos(th))
    Bc = jnp.where(small, 1.0 - th2 / 6.0, jnp.sin(th) / th)
    Cc = jnp.where(small, 0.5 - th2 / 24.0, (1.0 - jnp.cos(th)) / th2s)
    return A * n + Bc * jnp.cross(a, n) + Cc * (a @ n) * a


def normalise(u):
    n = u[3:]
    return jnp.concatenate([u[:3], n / jnp.sqrt(n @ n)])


def collide(u, a):
    """One collision: rotate the velocity direction by the vector a (|a| = angle)."""
    return normalise(jnp.concatenate([u[:3], rotate(u[3:], a)]))


def step(u, z, p, dt, sq):
    """RK4 step of the flow followed by a collision rotation a = sq * z (sq = sqrt(nu dt), z ~ N(0, I_3))."""
    return collide(rk4_step(u, p, dt), sq * z)


def step_gradient(u, z, p, dt, sq):
    """RK4 step followed by a collision of the gradient-Brownian-motion coupling: n' = normalise(n + sq P_perp(n) z)."""
    u = rk4_step(u, p, dt)
    n = u[3:]
    m = n + sq * (z - (z @ n) * n)
    return jnp.concatenate([u[:3], m / jnp.sqrt(m @ m)])


STEPS = {'rotation': step, 'gradient': step_gradient}


def step_collisionless(u, p, dt):
    return normalise(rk4_step(u, p, dt))


def initial_state(s, theta, zeta, xi):
    """u0 with the gyrophase 0."""
    return jnp.array([s, theta, zeta, jnp.sqrt(1.0 - xi * xi), 0.0, xi])


def simulate(u0, key, p, nu, dt, nstep, noise=True, coupling='rotation'):
    """One path of nstep steps, the final state (random numbers: normal(fold_in(key, chunk), (CHUNK, 3)); nstep must be
    a multiple of CHUNK when noise=True). Mostly for tests; the experiments use run_paths."""
    sq = jnp.sqrt(nu * dt)
    stepf = STEPS[coupling]
    if not noise:
        return jax.lax.scan(lambda u, _: (step_collisionless(u, p, dt), None), jnp.asarray(u0, float), None, length=nstep)[0]
    assert nstep % CHUNK == 0

    def chunk(u, j):
        Z = jax.random.normal(jax.random.fold_in(key, j), (CHUNK, 3))
        return jax.lax.scan(lambda uu, z: (stepf(uu, z, p, dt, sq), None), u, Z)[0], None
    return jax.lax.scan(chunk, jnp.asarray(u0, float), jnp.arange(nstep // CHUNK))[0]


# ---- many paths, with checkpoints and (optionally) the pathwise derivative with respect to eps_t -----------------------

_CACHE = {}


def _advance(dt, nchunk, noise, tangent, coupling='rotation'):
    """Compiled function: advance a batch of paths by nchunk * CHUNK steps.
    (u [N,6], du [N,6], key [N], chunk0, eps_t, nu, p) -> (u, du, s_min, s_max)  (extremes of s at the chunk ends).
    du is the derivative of u with respect to eps_t along the path for the same random numbers (tangent=True)."""
    sig = (dt, nchunk, noise, tangent, coupling)
    if sig in _CACHE:
        return _CACHE[sig]
    stepf = STEPS[coupling]

    def one(u, du, key, chunk0, eps_t, nu, p):
        def f(u, eps):
            q = {**p, 'eps_t': eps}
            sq = jnp.sqrt(nu * dt)

            def chunk_fn(carry, j):
                u, smin, smax = carry
                if noise:
                    Z = jax.random.normal(jax.random.fold_in(key, chunk0 + j), (CHUNK, 3))
                    u = jax.lax.scan(lambda uu, z: (stepf(uu, z, q, dt, sq), None), u, Z)[0]
                else:
                    u = jax.lax.scan(lambda uu, _: (step_collisionless(uu, q, dt), None), u, None, length=CHUNK)[0]
                return (u, jnp.minimum(smin, u[0]), jnp.maximum(smax, u[0])), None
            (u, smin, smax), _ = jax.lax.scan(chunk_fn, (u, u[0], u[0]), jnp.arange(nchunk))
            return u, smin, smax
        if tangent:
            (u1, smin, smax), (du1, _, _) = jax.jvp(f, (u, eps_t), (du, 1.0))
            return u1, du1, smin, smax
        u1, smin, smax = f(u, eps_t)
        return u1, du, smin, smax

    fn = jax.jit(jax.vmap(one, in_axes=(0, 0, 0, None, None, None, None)))
    _CACHE[sig] = fn
    return fn


def path_keys(master, ids):
    """One key per path id (an integer array)."""
    return jax.vmap(lambda i: jax.random.fold_in(master, i))(jnp.asarray(ids))


def run_paths(u0, keys, p, eps_t, nu, dt, Ts, tangent=False, unit_T=50.0, coupling='rotation'):
    """Integrate the paths u0 [N,6] (independent noise per path from `keys` [N]) and record the state at the times Ts.
    Returns a dict: 'T' (the times), 's' and 'xi' [len(Ts), N], 's_min', 's_max' [N] over the whole run, 'u_final' [N, 6], and
    with tangent=True 'ds' and 'dxi' [len(Ts), N]: d s / d eps_t and d xi / d eps_t along each path for the same random numbers."""
    steps_unit = int(round(unit_T / dt))
    assert steps_unit % CHUNK == 0, (unit_T, dt)
    nchunk = steps_unit // CHUNK
    noise = bool(nu > 0)
    fn = _advance(float(dt), nchunk, noise, bool(tangent), coupling)
    u = jnp.asarray(u0, dtype=float)
    du = jnp.zeros_like(u)
    p = {k: float(v) for k, v in p.items() if k != 'eps_t'}
    out = {'T': [], 's': [], 'xi': [], 'ds': [], 'dxi': []}
    smin, smax = np.full(u.shape[0], np.inf), np.full(u.shape[0], -np.inf)
    done, chunk0 = 0, 0
    for T in Ts:
        n_units = int(round(T / unit_T)) - done
        assert n_units >= 1 and abs(T - (done + n_units) * unit_T) < 1e-9, (T, unit_T)
        for _ in range(n_units):
            u, du, a, b = fn(u, du, keys, chunk0, float(eps_t), float(nu), p)
            smin, smax = np.minimum(smin, np.asarray(a)), np.maximum(smax, np.asarray(b))
            chunk0 += nchunk
        done += n_units
        out['T'].append(T)
        out['s'].append(np.asarray(u[:, 0]))
        out['xi'].append(np.asarray(u[:, 5]))
        if tangent:
            out['ds'].append(np.asarray(du[:, 0]))
            out['dxi'].append(np.asarray(du[:, 5]))
    out = {k: np.array(v) for k, v in out.items() if len(v)}
    out['s_min'], out['s_max'] = smin, smax
    out['u_final'] = np.asarray(u)
    return out


def make_ics(n, p0, seed=12345, s_range=(0.02, 0.20), margin=0.02):
    """n initial states: s uniform in s_range, theta and zeta uniform in [0, 2 pi), sign +-1 with equal probability,
    xi0 = sign * sqrt(1 - lam B(p0)) for the points with 1 - lam B(p0) >= margin (the others are redrawn; lam = p0['lam'])."""
    rng = np.random.RandomState(seed)
    rows = []
    while len(rows) < n:
        m = 4096                                         # fixed batch size: make_ics(n) is a prefix of make_ics(n') for n < n'
        s = rng.uniform(*s_range, m)
        th, ze = 2 * np.pi * rng.rand(m), 2 * np.pi * rng.rand(m)
        sgn = np.where(rng.rand(m) < 0.5, -1.0, 1.0)
        B = np.asarray(jax.vmap(lambda a, b, c: go.B_derivs(a, b, c, p0)[0])(jnp.asarray(s), jnp.asarray(th), jnp.asarray(ze)))
        q = 1.0 - p0['lam'] * B
        for i in np.nonzero(q >= margin)[0]:
            rows.append([s[i], th[i], ze[i], sgn[i] * np.sqrt(q[i])])
            if len(rows) == n:
                break
    s, th, ze, xi = np.array(rows).T
    return np.stack([s, th, ze, np.sqrt(1.0 - xi ** 2), np.zeros(n), xi], axis=1)
