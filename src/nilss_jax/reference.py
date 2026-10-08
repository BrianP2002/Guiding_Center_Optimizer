"""Non-Intrusive Least Squares Shadowing (NILSS), one parameter, tangent version: the dense reference implementation.

This is the straightforward NumPy version that keeps every time step in memory. :mod:`nilss_jax.core` is the version to use;
this one is kept as an independent implementation that the tests compare it against (they agree to 1e-10).

Equation numbers refer to Ni & Wang, arXiv:1611.00880v9 (NILSS); the Schur
complement solve is the one of Ni et al., arXiv:1711.06633 (FD-NILSS).

Interface expected from the problem definition
----------------------------------------------
integrator(u, w, vstar, par, s) -> (u_next, w_next, vstar_next)
    One time step of the primal state u [nc], the homogeneous tangents
    w [nus, nc] and the inhomogeneous tangent vstar [nc], for parameter `par`
    (a name) set to the value `s`. If the integrator has a `dt` attribute it must
    equal the `dt` given to nilss(); otherwise the time normalisation of the
    averages and of the time-dilation term would be silently wrong.
fJJu(u, par, s) -> (f, J, dJdu)
    right-hand side f(u) [nc], instantaneous objective J(u) and dJ/du [nc].
    (An explicit dependence of J on s, the dJ/ds term of eq. 41, is not supported.)
"""
import numpy as np
from scipy.linalg import block_diag


def _perp(p, f):
    """Component of p perpendicular to f (eq. 8). The last axis is the state axis."""
    return p - (np.sum(p * f, axis=-1) / np.sum(f * f, axis=-1))[..., None] * f


def pushSeg(nseg, nstep, nus, nc, dt, u0, vstar0, w0, par, s, integrator, fJJu):
    """
    find u, w, vstar, f, J, dJdu on each segment.
    Here we store all values at all time steps for better intuition,
    but this implementation costs a lot of memory.
    See the paper on FD-NILSS for discussion on how to reduce memory cost
    by computing inner products via only snapshots.

    Rs[i], bs[i] (i = 0..nseg-1) are the R and b obtained at the end of segment i
    (R_{i+1}, b_{i+1} in the paper); the last pair has no following segment.
    """

    J = np.zeros([nseg, nstep])
    u = np.zeros([nseg, nstep, nc])
    f = np.zeros(u.shape)
    dJdu = np.zeros(u.shape)
    vstar = np.zeros(u.shape)
    vstar_perp = np.zeros(u.shape)
    w = np.zeros([nseg, nstep, nus, nc])
    w_perp = np.zeros(w.shape)
    Rs = [] #Rs[0] in code = R_1 in paper
    bs = [] #bs[0] in code = b_1 in paper

    # assign initial value, u[0,0], v*[0,0], w[0,0]
    u[0,0] = u0
    vstar[0,0] = vstar0
    w[0,0] = w0

    # push forward
    for iseg in range(0, nseg):

        # compute u, w, vstar, f, J, dJdu
        for istep in range(0, nstep-1):
            u[iseg, istep+1], w[iseg, istep+1], vstar[iseg,istep+1]\
                = integrator(u[iseg, istep], w[iseg, istep], vstar[iseg, istep], par, s)
        for istep in range(0, nstep):
            f[iseg, istep], J[iseg, istep], dJdu[iseg, istep] = fJJu(u[iseg, istep], par, s)

        # calculate vstar_perp and w_perp
        vstar_perp[iseg] = _perp(vstar[iseg], f[iseg])
        w_perp[iseg] = _perp(w[iseg], f[iseg][:, np.newaxis, :])

        # renormalize at interfaces
        Q_temp, R_temp = np.linalg.qr(w_perp[iseg,-1].T, 'reduced')
        Rs.append(R_temp)
        b_temp = Q_temp.T @ vstar_perp[iseg,-1]
        bs.append(b_temp)
        p_temp = vstar_perp[iseg,-1] - Q_temp @ b_temp
        if iseg < nseg - 1:
            u[iseg+1, 0] = u[iseg, -1]
            w[iseg+1, 0] = Q_temp.T
            vstar[iseg+1,0] = p_temp

    return [u, w, vstar, w_perp, vstar_perp, f, J, dJdu, Rs, bs, Q_temp, p_temp]


def solve_shadowing_coeffs(Cs, ds, Rs, bs):
    """
    Minimize sum_i (a_i^T C_i a_i / 2 + d_i^T a_i)  s.t.  a_{i+1} = R_{i+1} a_i + b_{i+1}
    (eqs. 24, 37) through the Schur complement of the Lagrange multiplier system.
    Cs, ds: per segment (nseg entries); Rs, bs: per interface (nseg-1 entries).
    Returns a with shape [nseg, nus].
    """
    nseg, nus = len(Cs), Cs[0].shape[0]
    Cinvs = [np.linalg.inv(C) for C in Cs]
    if nseg == 1:
        return (-Cinvs[0] @ ds[0])[np.newaxis, :]
    Cinv = block_diag(*Cinvs)
    d = np.ravel(ds)

    # construct B, first off diagonal I, then add Rs
    B = np.eye((nseg-1)*nus, nseg*nus, k=nus)
    B[:, :-nus] -= block_diag(*Rs)
    b = np.ravel(bs)

    # solve
    lbd = np.linalg.solve(-B @ Cinv @ B.T, B @ Cinv @ d + b)
    a = -Cinv @ (B.T @ lbd + d)
    return a.reshape([nseg, nus])


def nilss(dt, nseg, T_seg, nseg_ps, u0, nus, par, s, integrator, fJJu, w0=None, return_info=False):
    """
    Returns Javg, dJds. With return_info=True also returns a dict with
      'lyap'  : the nus leading Lyapunov exponents of the projected tangent space (eq. 25)
      'T'     : total time of the shadowing trajectory
      'dJds_segments' : contribution of each segment to dJds
      'a'     : coefficients of the homogeneous tangents, [nseg, nus]
    w0 [nus, nc] optionally fixes the initial homogeneous tangents (random by default).
    nseg_ps = 0 skips the push-forward to the attractor, u0 is used as is.
    """

    u0 = np.asarray(u0, dtype=float)
    nc = len(u0)
    nstep = int(round(T_seg / dt)) + 1 # number of step + 1 in each time segment
    if hasattr(integrator, 'dt') and not np.isclose(integrator.dt, dt):
        raise ValueError(f'nilss got dt={dt} but the integrator steps with dt={integrator.dt}')

    # initial homogeneous tangents, random unit vectors unless given
    if w0 is None:
        w0 = np.random.rand(nus, nc)
        w0 = w0 / np.linalg.norm(w0, axis=1, keepdims=True)
    else:
        w0 = np.array(w0, dtype=float).reshape(nus, nc)
    vstar0 = np.zeros(nc)

    if nseg_ps > 0:
        # push forward u to a stable attractor
        u_ps, w_ps, vstar_ps, _, _, _, _, _, _, _, Q_ps, p_ps= pushSeg(nseg_ps, nstep, nus, nc, dt, u0, vstar0, w0, par, s, integrator, fJJu)
        u0 = u_ps[-1,-1]
        w0 = Q_ps.T
        vstar0 = p_ps
    else:
        # start from a perpendicular, orthonormal set of tangents so that xi = 0 at t = 0
        f0 = np.asarray(fJJu(u0, par, s)[0], dtype=float)
        w0 = np.linalg.qr(_perp(w0, f0).T, 'reduced')[0].T

    # find u, w, vstar on all segments
    u, w, vstar, w_perp, vstar_perp, f, J, dJdu, Rs, bs, _, _ = pushSeg(nseg, nstep, nus, nc, dt,u0, vstar0, w0, par, s, integrator, fJJu)

    # a weight matrix for integration, 0.5 at interfaces
    weight = np.ones(nstep)
    weight[0] = weight[-1] = 0.5

    # compute Javg
    Javg = np.sum(J*weight[np.newaxis,:]) / (nstep-1) / nseg

    # covariance matrices C (eq. 29) and d (eq. 33)
    Cs = [np.einsum('t,tic,tjc->ij', weight, w_perp[iseg], w_perp[iseg]) for iseg in range(nseg)]
    ds = [np.einsum('t,tic,tc->i', weight, w_perp[iseg], vstar_perp[iseg]) for iseg in range(nseg)]

    # solve the least squares problem with the continuity constraints
    a = solve_shadowing_coeffs(Cs, ds, Rs[:-1], bs[:-1])

    # calculate v and vperp
    v = np.zeros([nseg, nstep, nc])
    for iseg in range(nseg):
        v[iseg] = vstar[iseg] + np.einsum('i,tic->tc', a[iseg], w[iseg])


    # calculate ksi, only need to use first and last step in each segment
    ksi = np.zeros([nseg, nstep])
    for iseg in range(nseg):
        for i in (0, -1):
            ksi[iseg,i] = np.dot(v[iseg, i], f[iseg, i]) / np.dot(f[iseg, i], f[iseg, i])
        assert abs(ksi[iseg, 0]) <= 1e-5, f'xi is {ksi[iseg, 0]:.3e} at the start of segment {iseg}, expected 0'


    # compute dJds
    dJdss = []
    for iseg in range(nseg):
        t1 = np.sum(dJdu[iseg] * v[iseg] * weight[:,np.newaxis]) / (nstep-1) / nseg
        t2 = ksi[iseg,-1] * (Javg - J[iseg,-1]) / (nstep-1) / nseg / dt
        dJdss.append(t1+t2)
    dJds = np.sum(dJdss)

    if not return_info:
        return Javg, dJds
    T_seg_actual = (nstep - 1) * dt
    info = {'lyap': np.array([np.sum(np.log(np.abs([R[j, j] for R in Rs]))) for j in range(nus)]) / (nseg * T_seg_actual),
            'T': nseg * T_seg_actual,
            'dJds_segments': np.array(dJdss),
            'a': a}
    return Javg, dJds, info
