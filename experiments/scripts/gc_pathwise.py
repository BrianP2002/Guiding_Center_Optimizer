"""Pathwise (forward-tangent) derivative of a finite-time ensemble objective on the official guiding-center flow.

Direction B of the research plan: Phi_T(p) = E_{rho0}[ g(s_T) ] for a fixed, smooth initial distribution rho0 of the
particles and a fixed horizon T (here: the flow guiding_center.rhs4 with the parameters of draw SEED = 581 of
gc_official_scan.py, a mixed phase space with a chaotic sea, lambda_1 about 0.006). For every initial condition the
flow map is smooth in the parameters, so d Phi_T / d p = E[ g'(s_T) v_s(T) ] exactly, v = d u / d p the forward
tangent (no shadowing, no ergodicity). The question is the variance of that estimator as a function of T, in
particular whether the 1/x tail found for the NILSS shadowing direction is also there.

An initial condition is a position (s, theta, zeta) and a sign sigma of v_par; v_par = sigma sqrt(1 - lam B(p)) puts
it on the energy shell E = 1/2 of the parameter values p, so the initial map depends on p and the initial tangent is
its derivative (computed by jax.jacfwd, not by hand). The four parameters eps_t, eps_m, kappa, iota are
differentiated together (one tangent each); a homogeneous tangent W (renormalised every time unit) gives the finite
time Lyapunov exponent of the orbit, and int s dt with its derivative the time-averaged observable. Everything is
read out at several horizons TIMES from ONE integration. The integrators are vmapped over the initial conditions.

Modes (one slurm array task = one block of --per-task initial conditions):
    pathwise : RK4 orbit + tangents                         -> pw_<ens>_<task>.npz  (a task that is cut off by the time limit leaves
               pw_<ens>_<task>.part.npz with the state at its last horizon; running the same command again continues from there)
    fd       : orbits at q +- h e_n (same initial positions) -> fd_<ens>_<task>.npz   (central differences of the sample mean)
    pop      : orbits at q +- h e_n with INDEPENDENT initial positions for + and - -> pop_<task>.npz
    e2ics    : snapshots of the chaotic-sea orbits of draw 581 -> E2_ics.npz (the second ensemble E2)

    python experiments/scripts/gc_pathwise.py TASK --mode pathwise --ens E1 --n 4000 --per-task 100 [--times 100 300 ...]
"""
import argparse
import os
import time
from functools import lru_cache

import numpy as np
import jax
import jax.numpy as jnp

from nilss_jax.systems import guiding_center as go
from gc_official_scan import sample
from gc_official_candidates import initial_condition

SEED = 581
PARS = ('eps_t', 'eps_m', 'kappa', 'iota')
NPAR = len(PARS)
TIMES = (100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0)
DT = 0.01
STEPS_PER_CHUNK = 100                  # one time unit; the homogeneous tangent is renormalised and s, v_par sampled once per chunk
S_RANGE = (0.02, 0.20)                 # uniform initial s of ensemble E1
VPAR2_MIN = 0.02                       # initial conditions with 1 - lam B(p0) below this are dropped (v_par = 0 singularity)
SEA_ICS = [0, 1, 3, 6, 13]             # initial conditions k of gc_official_candidates.py that lie in the chaotic sea of draw 581
S_C, W_C = 0.15, 0.02                  # smooth loss proxy g2(s) = logistic((s - S_C) / W_C)
H_REL = (1e-4, 1e-3, 1e-2)             # relative steps of the finite differences (h_n = H_REL * |q0_n|)
NZ = 4 + 4 * NPAR + 4 + 1 + NPAR       # u, V (parameter tangents), W (homogeneous tangent), A = int s dt, dA/dp
NAUX = 5                               # log|W| accumulated, sign changes of v_par, max s, min s, previous sign of v_par


# ---- parameters, initial conditions ------------------------------------------------------------------------------

def base_parameters(seed=SEED):
    """(parameters of the draw as Python floats, the values of PARS as an array)."""
    p, _, _ = sample(seed)
    p = {k: float(v) for k, v in p.items()}
    return p, np.array([p[n] for n in PARS])


def with_q(pfix, q):
    return {**pfix, **{n: q[i] for i, n in enumerate(PARS)}}


def ic_map(pos, q, pfix):
    """(s, theta, zeta, sigma) -> (s, theta, zeta, sigma sqrt(1 - lam B)) on the energy shell E = 1/2 of the parameters q."""
    p = with_q(pfix, q)
    B = go.B_derivs(pos[0], pos[1], pos[2], p)[0]
    return jnp.array([pos[0], pos[1], pos[2], pos[3] * jnp.sqrt(1.0 - p['lam'] * B)])


def sample_positions(seed, n, pfix, q0, s_range=S_RANGE):
    """n positions (s, theta, zeta, sigma) of ensemble E1: uniform in s, theta, zeta, sigma = +-1, accepted if the particle can be
    there with v_par^2 >= VPAR2_MIN at the parameters q0 (the rule does not depend on the parameter that is differentiated)."""
    rng = np.random.RandomState(seed)
    p0 = with_q(pfix, q0)
    out = np.zeros((0, 4))
    while len(out) < n:
        m = 4 * n
        pos = np.stack([rng.uniform(*s_range, m), 2 * np.pi * rng.rand(m), 2 * np.pi * rng.rand(m), rng.choice([-1.0, 1.0], m)], axis=1)
        B = np.asarray(go.B_derivs(pos[:, 0], pos[:, 1], pos[:, 2], p0)[0])
        out = np.concatenate([out, pos[1.0 - p0['lam'] * B >= VPAR2_MIN]])
    return out[:n]


# ---- integrators -------------------------------------------------------------------------------------------------

def _flow(pfix):
    def f(x):                                           # x = (u, q)
        return go.rhs4(x[:4], with_q(pfix, x[4:]))
    return f


@lru_cache(maxsize=None)
def _pathwise_advance(pfix_items, dt):
    """Compiled advance(z, aux, q, nchunk) for a batch of initial conditions: RK4 of (u, V, W, A, dA) over nchunk time units."""
    f = _flow(dict(pfix_items))

    def deriv(z, q):
        u, V, W = z[:4], z[4:4 + 4 * NPAR].reshape(NPAR, 4), z[4 + 4 * NPAR:8 + 4 * NPAR]
        x = jnp.concatenate([u, q])
        J = jax.jacfwd(f)(x)                                                                   # [4, 4 + NPAR]: d f / d (u, q)
        dV = V @ J[:, :4].T + J[:, 4:].T                                                       # row n: Df V_n + df/dq_n
        return jnp.concatenate([f(x), dV.ravel(), J[:, :4] @ W, u[:1], V[:, 0]])               # ..., Df W, dA/dt = s, dA_n/dt = V_n[s]

    def rk4(z, q):
        k1 = deriv(z, q)
        k2 = deriv(z + 0.5 * dt * k1, q)
        k3 = deriv(z + 0.5 * dt * k2, q)
        k4 = deriv(z + dt * k3, q)
        return z + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    iw = 4 + 4 * NPAR

    def chunk(i, carry, q):
        z, aux = carry
        z = jax.lax.fori_loop(0, STEPS_PER_CHUNK, lambda j, zz: rk4(zz, q), z)
        nw = jnp.linalg.norm(z[iw:iw + 4])
        z = z.at[iw:iw + 4].set(z[iw:iw + 4] / nw)
        sg = jnp.sign(z[3])
        aux = jnp.stack([aux[0] + jnp.log(nw), aux[1] + (sg * aux[4] < 0), jnp.maximum(aux[2], z[0]), jnp.minimum(aux[3], z[0]), sg])
        return z, aux

    def advance_one(z, aux, q, nchunk):
        return jax.lax.fori_loop(0, nchunk, lambda i, c: chunk(i, c, q), (z, aux))

    return jax.jit(jax.vmap(advance_one, in_axes=(0, 0, None, None)))


@lru_cache(maxsize=None)
def _primal_advance(pfix_items, dt):
    """Compiled advance(z, q, nchunk) for the orbit only: z = (u, A = int s dt)."""
    f = _flow(dict(pfix_items))

    def deriv(z, q):
        return jnp.concatenate([f(jnp.concatenate([z[:4], q])), z[:1]])

    def rk4(z, q):
        k1 = deriv(z, q)
        k2 = deriv(z + 0.5 * dt * k1, q)
        k3 = deriv(z + 0.5 * dt * k2, q)
        k4 = deriv(z + dt * k3, q)
        return z + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    def advance_one(z, q, nchunk):
        return jax.lax.fori_loop(0, nchunk * STEPS_PER_CHUNK, lambda i, zz: rk4(zz, q), z)

    return jax.jit(jax.vmap(advance_one, in_axes=(0, None, None)))


def _nchunk(t_from, t_to, dt):
    n = (t_to - t_from) / (dt * STEPS_PER_CHUNK)
    assert abs(n - round(n)) < 1e-9, (t_from, t_to)
    return int(round(n))


def initial_pathwise_state(pos, pfix, q, w_seed=0):
    """z0 [n, NZ] and aux0 [n, NAUX] for positions pos [n, 4] at the parameters q."""
    pos, q = jnp.asarray(pos), jnp.asarray(q)
    u0 = jax.vmap(lambda x: ic_map(x, q, pfix))(pos)
    J = jax.vmap(lambda x: jax.jacfwd(ic_map, argnums=1)(x, q, pfix))(pos)                      # [n, 4 (state), NPAR]
    V0 = jnp.swapaxes(J, 1, 2).reshape(len(pos), 4 * NPAR)                                      # row n of V = d u0 / d q_n
    W0 = np.random.RandomState(w_seed).randn(len(pos), 4)
    W0 = jnp.asarray(W0 / np.linalg.norm(W0, axis=1, keepdims=True))
    z0 = jnp.concatenate([u0, V0, W0, jnp.zeros((len(pos), 1 + NPAR))], axis=1)
    aux0 = jnp.stack([jnp.zeros(len(pos)), jnp.zeros(len(pos)), u0[:, 0], u0[:, 0], jnp.sign(u0[:, 3])], axis=1)
    return z0, aux0


def unpack_pathwise(Z, AUX, times):
    """The dict of run_pathwise from the raw states Z [n, nT, NZ] and AUX [n, nT, NAUX] read out at `times`."""
    n, nT = Z.shape[:2]
    iw = 4 + 4 * NPAR
    return {'times': np.array(times), 'u': Z[..., :4], 'V': Z[..., 4:iw].reshape(n, nT, NPAR, 4), 'A': Z[..., iw + 4], 'dA': Z[..., iw + 5:],
            'logw': AUX[..., 0], 'nflip': AUX[..., 1], 'smax': AUX[..., 2], 'smin': AUX[..., 3]}


def run_pathwise(pos, pfix, q, times=TIMES, dt=DT, w_seed=0, state=None, t_start=0.0, callback=None, raw=False):
    """Forward tangents for all positions at the horizons `times` (ascending). Returns a dict of arrays [n, nT, ...]:
    u [4], V [NPAR, 4], A, dA [NPAR] (A = int_0^T s dt), logw (log norm growth of the homogeneous tangent), nflip (sign changes of
    v_par, sampled once per time unit), smax, smin. To continue an orbit that has been integrated to t_start, pass its
    state = (z, aux); callback(T, z, aux, Z_list, AUX_list) is called at every horizon (to save partial results);
    raw=True also returns (Z, AUX)."""
    advance = _pathwise_advance(tuple(sorted(pfix.items())), dt)
    z, aux = initial_pathwise_state(pos, pfix, q, w_seed) if state is None else (jnp.asarray(state[0]), jnp.asarray(state[1]))
    q = jnp.asarray(q)
    Zs, AUXs, t_prev = [], [], t_start
    for T in times:
        z, aux = advance(z, aux, q, _nchunk(t_prev, T, dt))
        Zs.append(np.asarray(z))
        AUXs.append(np.asarray(aux))
        t_prev = T
        if callback is not None:
            callback(T, z, aux, Zs, AUXs)
    Z, AUX = np.stack(Zs, axis=1), np.stack(AUXs, axis=1)                                        # [n, nT, NZ], [n, nT, NAUX]
    res = unpack_pathwise(Z, AUX, times)
    return (res, Z, AUX) if raw else res


def run_primal(pos, pfix, q, times=TIMES, dt=DT):
    """Orbits (no tangents) from the positions pos at the parameters q: u [n, nT, 4] and A = int_0^T s dt [n, nT]."""
    advance = _primal_advance(tuple(sorted(pfix.items())), dt)
    q_j = jnp.asarray(q)
    u0 = jax.vmap(lambda x: ic_map(x, q_j, pfix))(jnp.asarray(pos))
    z = jnp.concatenate([u0, jnp.zeros((len(pos), 1))], axis=1)
    out, t_prev = [], 0.0
    for T in times:
        z = advance(z, q_j, _nchunk(t_prev, T, dt))
        out.append(np.asarray(z))
        t_prev = T
    Z = np.stack(out, axis=1)
    return {'times': np.array(times), 'u': Z[..., :4], 'A': Z[..., 4]}


def sea_snapshots(pfix, q0, nper=2600, spacing=200.0, t_spinup=6000.0, dt=DT, jitter_seed=777):
    """Positions (s, theta, zeta, sigma) of snapshots of the five chaotic-sea orbits of the draw (ensemble E2): every `spacing` time
    units after t_spinup, nper per orbit (the five orbits advance together). Snapshots with v_par^2 < VPAR2_MIN are dropped, as in E1;
    most of the time a sea orbit is closer to its turning points than that, so about one snapshot in twelve is kept."""
    p, u_scan, _ = sample(SEED)
    p0 = with_q(pfix, q0)
    advance = _primal_advance(tuple(sorted(pfix.items())), dt)
    qj = jnp.asarray(q0)
    u0s = []
    for c, k in enumerate(SEA_ICS):
        u0 = np.array(initial_condition(SEED, k, p, u_scan)) + 1e-6 * np.random.RandomState(jitter_seed + c).randn(4)
        u0[3] = np.sign(u0[3]) * np.sqrt(1.0 - p0['lam'] * float(go.B_derivs(u0[0], u0[1], u0[2], p0)[0]))
        u0s.append(u0)
    z = jnp.asarray(np.concatenate([np.array(u0s), np.zeros((len(u0s), 1))], axis=1))
    z = advance(z, qj, _nchunk(0.0, t_spinup, dt))
    pos = []
    for _ in range(nper):
        z = advance(z, qj, _nchunk(0.0, spacing, dt))
        U = np.asarray(z[:, :4])
        B = np.asarray(go.B_derivs(U[:, 0], U[:, 1], U[:, 2], p0)[0])
        keep = 1.0 - p0['lam'] * B >= VPAR2_MIN
        pos.extend([[u[0], u[1], u[2], np.sign(u[3])] for u in U[keep]])
    return np.array(pos)


# ---- command line ------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task', type=int)
    ap.add_argument('--mode', default='pathwise', choices=['pathwise', 'fd', 'pop', 'e2ics'])
    ap.add_argument('--ens', default='E1', choices=['E1', 'E2'])
    ap.add_argument('--n', type=int, default=4000, help='size of the ensemble')
    ap.add_argument('--per-task', type=int, default=100)
    ap.add_argument('--seed', type=int, default=1, help='seed of the positions of E1')
    ap.add_argument('--times', type=float, nargs='+', default=list(TIMES))
    ap.add_argument('--par', default='eps_t', help='pop mode: the parameter that is shifted')
    ap.add_argument('--h-rel', type=float, nargs='+', default=list(H_REL), help='fd / pop mode: relative steps')
    ap.add_argument('--nper', type=int, default=2600, help='e2ics mode: snapshots per sea orbit (about 8 % are kept)')
    ap.add_argument('--dt', type=float, default=DT)
    ap.add_argument('--outdir', default='results/gc_pathwise')
    args = ap.parse_args()

    pfix, q0 = base_parameters()
    pfix_items = {k: v for k, v in pfix.items()}
    for n in PARS:
        pfix_items.pop(n)                                   # the differentiated parameters are traced, not constants
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()

    if args.mode == 'e2ics':
        pos = sea_snapshots(pfix_items, q0, nper=args.nper, dt=args.dt)
        np.savez(os.path.join(args.outdir, 'E2_ics.npz'), pos=pos, q0=q0, pars=np.array(PARS))
        print(f'E2: {len(pos)} snapshots of the sea orbits, s in [{pos[:, 0].min():.3f}, {pos[:, 0].max():.3f}]  ({time.time() - t0:.0f} s)')
        return

    def block_of(pos_all):
        lo, hi = args.task * args.per_task, min((args.task + 1) * args.per_task, len(pos_all))
        return lo, hi, pos_all[lo:hi]

    def positions(ens, n, seed):
        if ens == 'E1':
            return sample_positions(seed, n, pfix_items, q0)
        return np.load(os.path.join(args.outdir, 'E2_ics.npz'))['pos'][:n]

    if args.mode == 'pathwise':
        lo, hi, pos = block_of(positions(args.ens, args.n, args.seed))
        final = os.path.join(args.outdir, f'pw_{args.ens}_{args.task:04d}.npz')
        part = os.path.join(args.outdir, f'pw_{args.ens}_{args.task:04d}.part.npz')
        done_Z = done_AUX = None
        state, t_start, times = None, 0.0, list(args.times)
        if os.path.exists(part):                                    # a task that was cut off by the time limit continues from its last horizon
            prev = np.load(part)
            keep = [k for k, T in enumerate(prev['times']) if T in args.times]
            if keep and np.array_equal(prev['idx'], np.arange(lo, hi)):
                done_Z, done_AUX = prev['Z'][:, keep], prev['AUX'][:, keep]
                t_start = float(prev['times'][keep[-1]])
                state, times = (prev['z_last'], prev['aux_last']), [T for T in args.times if T > t_start]
                print(f'resuming from T = {t_start:g} ({len(keep)} horizons done)', flush=True)

        def save_part(T, z, aux, Zs, AUXs):
            Z = np.stack(Zs, axis=1)
            AUX = np.stack(AUXs, axis=1)
            tt = np.array([t for t in args.times if t <= T][-Z.shape[1]:])
            if done_Z is not None:
                Z, AUX = np.concatenate([done_Z, Z], axis=1), np.concatenate([done_AUX, AUX], axis=1)
                tt = np.array([t for t in args.times if t <= T])
            np.savez(part, times=tt, Z=Z, AUX=AUX, idx=np.arange(lo, hi), z_last=np.asarray(z), aux_last=np.asarray(aux))

        if times:
            _, Z, AUX = run_pathwise(pos, pfix_items, q0, tuple(times), args.dt, w_seed=1000 + args.task, state=state, t_start=t_start,
                                     callback=save_part, raw=True)
            if done_Z is not None:
                Z, AUX = np.concatenate([done_Z, Z], axis=1), np.concatenate([done_AUX, AUX], axis=1)
        else:
            Z, AUX = done_Z, done_AUX
        res = unpack_pathwise(Z, AUX, args.times)
        np.savez(final, pos=pos, idx=np.arange(lo, hi), q0=q0, pars=np.array(PARS), dt=args.dt, **res)
        if os.path.exists(part):
            os.remove(part)
        print(f'pathwise {args.ens} task {args.task}: ICs {lo}..{hi - 1}, T up to {args.times[-1]:g}, '
              f'finite {np.mean(np.isfinite(res["V"]).all(axis=(2, 3))[:, -1]):.2f}, {time.time() - t0:.0f} s', flush=True)

    elif args.mode == 'fd':
        lo, hi, pos = block_of(positions(args.ens, args.n, args.seed))
        U = np.zeros((NPAR, len(args.h_rel), 2, len(pos), len(args.times), 4))
        A = np.zeros(U.shape[:-1])
        for n in range(NPAR):
            for ih, r in enumerate(args.h_rel):
                for isg, sg in enumerate((+1.0, -1.0)):
                    q = q0.copy()
                    q[n] += sg * r * abs(q0[n])
                    res = run_primal(pos, pfix_items, q, tuple(args.times), args.dt)
                    U[n, ih, isg], A[n, ih, isg] = res['u'], res['A']
        np.savez(os.path.join(args.outdir, f'fd_{args.ens}_{args.task:04d}.npz'), pos=pos, idx=np.arange(lo, hi), q0=q0, pars=np.array(PARS),
                 h=np.array([[r * abs(q0[n]) for r in args.h_rel] for n in range(NPAR)]), times=np.array(args.times), u=U, A=A)
        print(f'fd {args.ens} task {args.task}: ICs {lo}..{hi - 1}, {NPAR * len(args.h_rel) * 2} orbits each, {time.time() - t0:.0f} s', flush=True)

    elif args.mode == 'pop':
        # independent initial positions for the + and - orbits: task -> (block, sign index) ; two tasks per block
        block, isg = divmod(args.task, 2)
        n = PARS.index(args.par)
        seed = 100_000 + 2 * block + isg
        pos = sample_positions(seed, args.per_task, pfix_items, q0)
        out = {}
        for ih, r in enumerate(args.h_rel):
            q = q0.copy()
            q[n] += (+1.0 if isg == 0 else -1.0) * r * abs(q0[n])
            res = run_primal(pos, pfix_items, q, tuple(args.times), args.dt)
            out[f'u_{ih}'], out[f'A_{ih}'] = res['u'], res['A']
        np.savez(os.path.join(args.outdir, f'pop_{args.par}_{args.task:04d}.npz'), pos=pos, par=args.par, sign=(+1 if isg == 0 else -1),
                 h=np.array([r * abs(q0[n]) for r in args.h_rel]), times=np.array(args.times), **out)
        print(f'pop {args.par} task {args.task}: block {block} sign {"+-"[isg]}, {args.per_task} ICs x {len(args.h_rel)} steps, '
              f'{time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
