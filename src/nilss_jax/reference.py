"""Dense reference implementation of NILSS (one parameter, tangent form), written from the equations of the papers.

This module exists to cross-check :mod:`nilss_jax.core`, and is deliberately different from it in structure: every time step of the
whole run is kept in memory, and the least-squares problem with the continuity constraints is solved as the full
Karush-Kuhn-Tucker system with a dense solver (``core`` accumulates the integrals on the fly and uses the sparse Schur complement
of arXiv:1711.06633). The tests require the two to agree to 1e-10 on Lorenz 63, on the guiding-center flow and on an
8-dimensional Lorenz 96 system.

Equation numbers refer to Ni & Wang, arXiv:1611.00880v9: the projection (8), the least-squares problem (24), the Lyapunov exponents
(25), the matrices C and d (29, 33), the sensitivity (41). The algorithm is theirs; their public Python implementation
(https://github.com/niangxiu/nilss) was the starting point of the first version of this repository, and this module is a rewrite
that does not use their code.

Interface of the problem definition
-----------------------------------
integrator(u, w, vstar, par, s) -> (u_next, w_next, vstar_next)
    one time step of the state u [nc], the homogeneous tangents w [nus, nc] and the inhomogeneous tangent vstar [nc] for the parameter
    named ``par`` set to the value ``s``. If it has a ``dt`` attribute it must equal the ``dt`` given to :func:`nilss`.
fJJu(u, par, s) -> (f, J, dJdu)
    the right-hand side f(u) [nc], the instantaneous objective J(u) and its gradient dJ/du [nc]. (A dependence of J on ``s`` is not supported.)
"""
import numpy as np

__all__ = ['nilss', 'solve_shadowing_coeffs']


def _remove_component(x, f):
    """x minus its component along f (eq. 8); f has shape [..., nc], x has shape [..., k, nc] or [..., nc]."""
    if x.ndim == f.ndim + 1:
        f = f[..., None, :]
    coeff = np.sum(x * f, axis=-1, keepdims=True) / np.sum(f * f, axis=-1, keepdims=True)
    return x - coeff * f


def _march(u, w, v, nsteps, par, s, integrator, fJJu):
    """Advance ``nsteps`` steps and return the samples (nsteps + 1 of them, both ends included) of u, w, v*, f, J and dJ/du."""
    us, ws, vs = [u], [w], [v]
    for _ in range(nsteps):
        u, w, v = integrator(us[-1], ws[-1], vs[-1], par, s)
        us.append(np.asarray(u, dtype=float))
        ws.append(np.asarray(w, dtype=float))
        vs.append(np.asarray(v, dtype=float))
    fjd = [fJJu(x, par, s) for x in us]
    f = np.array([np.asarray(a[0], dtype=float) for a in fjd])
    J = np.array([float(a[1]) for a in fjd])
    dJ = np.array([np.asarray(a[2], dtype=float) for a in fjd])
    return np.array(us), np.array(ws), np.array(vs), f, J, dJ


def _interface(w_perp_end, v_perp_end):
    """Renormalise at the end of a segment: W_perp = Q R and v*_perp = Q b + (the rest), which starts the next segment."""
    Q, R = np.linalg.qr(w_perp_end.T)
    b = Q.T @ v_perp_end
    return Q, R, b, v_perp_end - Q @ b


def solve_shadowing_coeffs(Cs, ds, Rs, bs):
    """
    Minimise sum_i (a_i^T C_i a_i / 2 + d_i^T a_i) subject to a_{i+1} = R_i a_i + b_i (eqs. 24, 37) by solving the full KKT system
        [C  B^T] [a     ]   [-d]
        [B   0 ] [lambda] = [ b]       with the rows of B: a_{i+1} - R_i a_i.
    Cs, ds: per segment (nseg entries); Rs, bs: per interface (nseg - 1 entries). Returns a with shape [nseg, nus].
    """
    nseg, nus = len(Cs), Cs[0].shape[0]
    n, m = nseg * nus, (nseg - 1) * nus
    K = np.zeros((n + m, n + m))
    rhs = np.zeros(n + m)
    for i in range(nseg):
        K[i * nus:(i + 1) * nus, i * nus:(i + 1) * nus] = Cs[i]
        rhs[i * nus:(i + 1) * nus] = -np.asarray(ds[i])
    for i in range(nseg - 1):
        rows = slice(n + i * nus, n + (i + 1) * nus)
        K[rows, (i + 1) * nus:(i + 2) * nus] = np.eye(nus)
        K[rows, i * nus:(i + 1) * nus] = -np.asarray(Rs[i])
        K[:n, rows] = K[rows, :n].T
        rhs[rows] = np.asarray(bs[i])
    return np.linalg.solve(K, rhs)[:n].reshape(nseg, nus)


def nilss(dt, nseg, T_seg, nseg_ps, u0, nus, par, s, integrator, fJJu, w0=None, return_info=False):
    """
    Long-time average of J and its derivative with respect to the parameter ``par`` at the value ``s``.

    ``nseg`` segments of time ``T_seg`` are used after ``nseg_ps`` segments of spin-up (0: start from u0 as it is). ``nus`` homogeneous
    tangents; ``w0`` [nus, nc] fixes their initial values (random unit vectors from ``numpy.random`` by default).
    Returns ``Javg, dJds``; with ``return_info=True`` also a dict with
      'lyap'          the nus exponents of the projected tangent space (eq. 25),
      'T'             total time of the shadowing trajectory,
      'dJds_segments' contribution of each segment to dJds,
      'a'             coefficients of the homogeneous tangents, [nseg, nus].
    """
    u0 = np.asarray(u0, dtype=float)
    nc = u0.size
    nsteps = int(round(T_seg / dt))
    if hasattr(integrator, 'dt') and not np.isclose(integrator.dt, dt):
        raise ValueError(f'nilss got dt={dt} but the integrator steps with dt={integrator.dt}')
    if w0 is None:
        w0 = np.random.rand(nus, nc)
        w0 = w0 / np.linalg.norm(w0, axis=1, keepdims=True)
    w = np.array(w0, dtype=float).reshape(nus, nc)
    v = np.zeros(nc)

    u = u0
    if nseg_ps > 0:
        # spin-up: only the interface data are needed, they carry u, w and v* over to the recorded part
        for _ in range(nseg_ps):
            us, ws, vs, f, _, _ = _march(u, w, v, nsteps, par, s, integrator, fJJu)
            Q, _, _, v = _interface(_remove_component(ws[-1], f[-1]), _remove_component(vs[-1], f[-1]))
            u, w = us[-1], Q.T
    else:
        # start from perpendicular, orthonormal tangents so that xi = 0 at t = 0
        f0 = np.asarray(fJJu(u, par, s)[0], dtype=float)
        w = np.linalg.qr(_remove_component(w, f0).T)[0].T

    # the recorded part: everything is stored
    seg = []
    for _ in range(nseg):
        us, ws, vs, f, J, dJ = _march(u, w, v, nsteps, par, s, integrator, fJJu)
        wp, vp = _remove_component(ws, f), _remove_component(vs, f)
        Q, R, b, v = _interface(wp[-1], vp[-1])
        seg.append(dict(w=ws, v=vs, f=f, J=J, dJ=dJ, wp=wp, vp=vp, R=R, b=b))
        u, w = us[-1], Q.T

    weight = np.ones(nsteps + 1)
    weight[[0, -1]] = 0.5                                   # trapezoid rule
    N = nseg * nsteps
    Javg = sum(weight @ sg['J'] for sg in seg) / N          # long-time average of J

    Cs = [np.einsum('t,tic,tjc->ij', weight, sg['wp'], sg['wp']) for sg in seg]        # eq. 29
    ds = [np.einsum('t,tic,tc->i', weight, sg['wp'], sg['vp']) for sg in seg]          # eq. 33
    a = solve_shadowing_coeffs(Cs, ds, [sg['R'] for sg in seg[:-1]], [sg['b'] for sg in seg[:-1]])

    contributions = []
    for i, sg in enumerate(seg):
        vfull = sg['v'] + np.einsum('i,tic->tc', a[i], sg['w'])                          # v = v* + sum_j a_j w_j
        xi_end = vfull[-1] @ sg['f'][-1] / (sg['f'][-1] @ sg['f'][-1])                   # coefficient along the flow at the end of the segment
        xi_start = vfull[0] @ sg['f'][0] / (sg['f'][0] @ sg['f'][0])
        assert abs(xi_start) <= 1e-5, f'xi is {xi_start:.3e} at the start of segment {i}, expected 0'
        part1 = np.sum(weight[:, None] * sg['dJ'] * vfull) / N
        part2 = xi_end * (Javg - sg['J'][-1]) / N / dt                                   # the time-dilation term of eq. 41
        contributions.append(part1 + part2)
    dJds = float(np.sum(contributions))

    if not return_info:
        return Javg, dJds
    T_actual = nsteps * dt
    lyap = np.array([np.sum(np.log(np.abs([sg['R'][j, j] for sg in seg]))) for j in range(nus)]) / (nseg * T_actual)
    return Javg, dJds, {'lyap': lyap, 'T': nseg * T_actual, 'dJds_segments': np.array(contributions), 'a': a}
