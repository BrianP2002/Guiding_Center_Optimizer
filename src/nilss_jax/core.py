"""Streaming, multi-parameter NILSS in JAX.

The least-squares shadowing problem of Ni & Wang (arXiv:1611.00880, "NILSS") with the Schur-complement solve of Ni et al.
(arXiv:1711.06633, "FD-NILSS"), written so that

* the time integrals of the tangent solutions are accumulated while integrating (the matrices C and d of eqs. 29 and 33 and the
  integrals of dJ/du . v of eq. 41), so the memory does not depend on the length of the trajectory;
* any number of parameters share the homogeneous tangents: one W (``nus`` tangents), one inhomogeneous tangent per parameter.
  C, B and the Schur matrix are common, only d and b change;
* the tangent equations use Jacobian-vector products (``jax.linearize``), so the cost per step grows with the number of
  tangents, not with the dimension of the state;
* an optional conserved quantity (the energy of a Hamiltonian flow) puts the tangents on the perturbed invariant surface.

Typical use::

    from nilss_jax import NILSS

    def rhs(u, p):                      # du/dt, written in jax.numpy; p is a dict of floats
        ...
    def J(u):                           # instantaneous objective, a scalar function of the state only
        return u[2]

    nilss = NILSS(rhs, J, params=("rho",), dt=0.005, T_seg=0.5, nus=1)
    res = nilss.run({"sigma": 10.0, "rho": 28.0, "beta": 8 / 3}, u0, T=400.0, T_spinup=20.0)
    res.J, res.dJdp["rho"], res.lyapunov

One run is one trajectory. In chaotic systems that are not uniformly hyperbolic the estimate of one trajectory can be far
from the truth and its error bar misleading: use :func:`nilss_jax.run_ensemble` and :func:`nilss_jax.reliability_report`.

Segment and step conventions are those of :mod:`nilss_jax.reference` (trapezoid rule with weight 1/2 at the interfaces).
"""
from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

__all__ = ['NILSS', 'NILSSStreamer', 'NILSSResult', 'solve_shadowing_coeffs_multi']


def _perp(a, f):
    """Component of the rows of a perpendicular to f."""
    return a - (a @ f)[:, None] * f[None, :] / (f @ f)


def solve_shadowing_coeffs_multi(Cs, Ds, Rs, Bs):
    """
    Minimise sum_i (a_i^T C_i a_i / 2 + d_i^T a_i) s.t. a_{i+1} = R_i a_i + b_i, for several right-hand sides at once
    (the Schur complement of the Lagrange multiplier system, block tridiagonal, sparse).
    Cs [nseg, nus, nus], Ds [nseg, nus, npar], Rs [nseg-1, nus, nus], Bs [nseg-1, nus, npar].
    Returns a [nseg, nus, npar].
    """
    nseg, nus, npar = Ds.shape
    Cinv = sp.block_diag([sp.coo_matrix(np.linalg.inv(C)) for C in Cs], format='csr')
    d = Ds.reshape(nseg * nus, npar)
    if nseg == 1:
        return (-Cinv @ d).reshape(1, nus, npar)
    B = sp.lil_matrix(((nseg - 1) * nus, nseg * nus))
    for i in range(nseg - 1):
        B[i * nus:(i + 1) * nus, i * nus:(i + 1) * nus] = -Rs[i]
        B[i * nus:(i + 1) * nus, (i + 1) * nus:(i + 2) * nus] = np.eye(nus)
    B = B.tocsr()
    S = (B @ Cinv @ B.T).tocsc()
    lbd = spla.splu((-S).tocsc()).solve(B @ (Cinv @ d) + Bs.reshape((nseg - 1) * nus, npar))
    return (-(Cinv @ (B.T @ lbd + d))).reshape(nseg, nus, npar)


@dataclass(repr=False)
class NILSSResult:
    """Outcome of one NILSS run (one trajectory).

    Attributes
    ----------
    J : long-time average of the objective along the trajectory.
    dJdp : sensitivities d<J>/dp, keyed by parameter name.
    lyapunov : the ``nus`` leading exponents of the projected tangent space (largest first is not guaranteed for nus > 1).
    T : total time of the shadowing trajectory (without the spin-up).
    segments : contribution of each segment to dJdp, [nseg, npar]; ``segments.sum(axis=0)`` is the sensitivity.
    vnorm : norm of the shadowing direction perpendicular to the flow at the end of each segment, [nseg, npar].
    J_segments : mean of J over each segment, [nseg].
    coeffs : coefficients a of the homogeneous tangents, [nseg, nus, npar].
    finite : False if the integration produced non-finite numbers (then the numeric fields are NaN).
    """
    J: float
    dJdp: dict
    lyapunov: np.ndarray
    T: float
    params: tuple
    nus: int
    T_seg: float
    dt: float
    segments: np.ndarray
    vnorm: np.ndarray
    J_segments: np.ndarray
    coeffs: np.ndarray
    finite: bool = True
    seed: int = 0
    meta: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        sens = ', '.join(f'{n}: {v:+.4g}' for n, v in self.dJdp.items())
        return f'NILSSResult(J={self.J:.6g}, dJdp={{{sens}}}, T={self.T:g}, nseg={self.nseg}, nus={self.nus}, finite={self.finite})'

    @property
    def nseg(self) -> int:
        return int(self.segments.shape[0])

    @property
    def dJdp_array(self) -> np.ndarray:
        """The sensitivities in the order of ``params``."""
        return np.array([self.dJdp[n] for n in self.params], dtype=float)

    @property
    def lyapunov_time_product(self) -> float:
        """lambda_1 * T: how many e-foldings of the leading exponent the trajectory has seen."""
        return float(np.max(self.lyapunov) * self.T)

    def summary(self) -> str:
        lines = [f'NILSS run: T = {self.T:g}, {self.nseg} segments of {self.T_seg:g}, nus = {self.nus}, <J> = {self.J:.6g}',
                 f'  leading exponents: {np.array2string(np.asarray(self.lyapunov), precision=5)}  (lambda_1 T = {self.lyapunov_time_product:.3g})']
        lines += [f'  d<J>/d{n} = {self.dJdp[n]:+.6g}' for n in self.params]
        if not self.finite:
            lines.append('  WARNING: non-finite numbers in the integration (the orbit left the domain or dt is too large)')
        elif self.lyapunov_time_product < 3.0:
            lines.append('  WARNING: lambda_1 * T < 3: no positive exponent is resolved, the NILSS assumptions are not supported by this run')
        return '\n'.join(lines)

    def save(self, path) -> None:
        """Write the result to an ``.npz`` file (read back with :meth:`NILSSResult.load`)."""
        meta = {'params': list(self.params), 'nus': self.nus, 'T_seg': self.T_seg, 'dt': self.dt, 'T': self.T,
                'finite': self.finite, 'seed': self.seed, 'meta': self.meta}
        np.savez_compressed(path, J=self.J, dJdp=self.dJdp_array, lyapunov=self.lyapunov, segments=self.segments,
                            vnorm=self.vnorm, J_segments=self.J_segments, coeffs=self.coeffs, meta=json.dumps(meta))

    @classmethod
    def load(cls, path) -> 'NILSSResult':
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z['meta']))
        params = tuple(meta['params'])
        return cls(J=float(z['J']), dJdp=dict(zip(params, map(float, z['dJdp']))), lyapunov=z['lyapunov'], T=float(meta['T']),
                   params=params, nus=int(meta['nus']), T_seg=float(meta['T_seg']), dt=float(meta['dt']),
                   segments=z['segments'], vnorm=z['vnorm'], J_segments=z['J_segments'], coeffs=z['coeffs'],
                   finite=bool(meta['finite']), seed=int(meta['seed']), meta=meta.get('meta', {}))


class NILSS:
    """Compiled NILSS integrator for one problem; :meth:`run` can be called repeatedly (other parameter values, other
    orbits) without recompiling.

    Parameters
    ----------
    rhs : ``rhs(u, p) -> du/dt``, written with ``jax.numpy`` (it is traced and differentiated). ``p`` is a dict of
        floats; the entries named in ``params`` are the parameters differentiated, the others are constants.
    J : ``J(u) -> scalar``, the instantaneous objective. A dependence of J on the parameters is not supported.
    params : names of the parameters to differentiate (keys of ``p``); they must be scalars.
    dt : RK4 time step.
    T_seg : time length of one segment (rounded to a multiple of ``dt``). Segments only set where the tangents are
        re-orthonormalised: the result does not depend on them as long as they are not too long for the tangents
        to stay well conditioned (the exponent times ``T_seg`` should be modest).
    nus : number of homogeneous tangents; at least the number of positive Lyapunov exponents of the flow (one for the
        Lorenz attractor). The flow direction itself is projected out and needs no tangent.
    invariant : optional ``E(u, p) -> scalar`` conserved by the flow for every ``p`` (the energy of a Hamiltonian
        system). Then the tangents start on the perturbed invariant surface (grad E . w = 0, grad E . v* + dE/dp = 0),
        which the tangent equations preserve. Without it a conserved quantity is a second neutral direction and the
        shadowing least-squares problem has the wrong structure.
    """

    def __init__(self, rhs: Callable, J: Callable, params: Sequence[str], dt: float, T_seg: float, nus: int = 1,
                 invariant: Optional[Callable] = None):
        if jnp.zeros(1, dtype=float).dtype != jnp.dtype('float64'):
            raise RuntimeError('NILSS needs double precision: set jax.config.update("jax_enable_x64", True) before any JAX computation')
        params = (params,) if isinstance(params, str) else tuple(params)
        if not params or len(set(params)) != len(params) or not all(isinstance(n, str) for n in params):
            raise ValueError(f'params must be a non-empty list of distinct parameter names, got {params!r}')
        if dt <= 0 or T_seg < dt:
            raise ValueError(f'need 0 < dt <= T_seg, got dt={dt}, T_seg={T_seg}')
        if int(nus) < 1:
            raise ValueError('nus must be at least 1')
        self.rhs, self.J, self.pars, self.dt, self.nus, self.invariant = rhs, J, params, float(dt), int(nus), invariant
        self._spec = (rhs, J, params, float(dt), float(T_seg), int(nus), invariant)       # to rebuild this object in a worker process
        ratio = T_seg / dt
        if abs(ratio - round(ratio)) > 1e-6 * max(1.0, ratio):
            warnings.warn(f'T_seg = {T_seg} is not a multiple of dt = {dt}; using {round(ratio)} steps per segment')
        self.nstep = int(round(ratio))
        self.T_seg = self.nstep * self.dt
        self._segment = jax.jit(self._make_segment())
        self._start = jax.jit(self._make_start())

    # ------------------------------------------------------------------------------------------------------------
    def _make_start(self):
        rhs, pars, invariant = self.rhs, self.pars, self.invariant

        def start(u, W, p):
            V = jnp.zeros((len(pars), u.shape[0]))
            if invariant is not None:
                g = jax.grad(invariant, 0)(u, p)
                dEdp = jnp.stack([jax.grad(lambda x, n=n: invariant(u, {**p, n: x}))(p[n]) for n in pars])
                W = W - (W @ g)[:, None] * g[None, :] / (g @ g)
                V = -(dEdp / (g @ g))[:, None] * g[None, :]
            f = rhs(u, p)
            W, V = _perp(W, f), _perp(V, f)
            return jnp.linalg.qr(W.T)[0].T, V
        return start

    def _make_segment(self):
        rhs, J, pars, dt, nstep = self.rhs, self.J, self.pars, self.dt, self.nstep

        def derivs(u, W, V, p):
            f, jvp = jax.linearize(lambda x: rhs(x, p), u)             # jvp(t) = Df(u) t
            dfdp = jnp.stack([jax.jvp(lambda x, n=n: rhs(u, {**p, n: x}), (p[n],), (jnp.ones_like(p[n]),))[1] for n in pars])
            return f, jax.vmap(jvp)(W), jax.vmap(jvp)(V) + dfdp

        def rk4(u, W, V, p):
            y = (u, W, V)
            k0 = derivs(*y, p)
            k1 = derivs(*(a + 0.5 * dt * k for a, k in zip(y, k0)), p)
            k2 = derivs(*(a + 0.5 * dt * k for a, k in zip(y, k1)), p)
            k3 = derivs(*(a + dt * k for a, k in zip(y, k2)), p)
            return tuple(a + dt / 6.0 * (c0 + 2 * c1 + 2 * c2 + c3) for a, c0, c1, c2, c3 in zip(y, k0, k1, k2, k3))

        def terms(u, W, V, p):
            f = rhs(u, p)
            Wp, Vp = _perp(W, f), _perp(V, f)
            dJ = jax.grad(J)(u)
            return {'C': Wp @ Wp.T, 'D': Vp @ Wp.T, 'Jv': V @ dJ, 'Jw': W @ dJ, 'Jsum': J(u)}

        def segment(u, W, V, p):
            acc0 = jax.tree_util.tree_map(jnp.zeros_like, terms(u, W, V, p))

            def step(carry, i):
                u, W, V, acc = carry
                t = terms(u, W, V, p)
                weight = jnp.where(i == 0, 0.5, 1.0)
                acc = jax.tree_util.tree_map(lambda a, x: a + weight * x, acc, t)
                u, W, V = rk4(u, W, V, p)
                return (u, W, V, acc), None
            (u, W, V, acc), _ = jax.lax.scan(step, (u, W, V, acc0), jnp.arange(nstep))
            acc = jax.tree_util.tree_map(lambda a, x: a + 0.5 * x, acc, terms(u, W, V, p))
            f = rhs(u, p)
            end = {'vf': V @ f / (f @ f), 'wf': W @ f / (f @ f), 'Jend': J(u)}
            # renormalise at the interface: W_perp = Q R, v*_perp = Q b + (the rest, which starts the next segment)
            Wp, Vp = _perp(W, f), _perp(V, f)
            end['Wp'], end['Vp'] = Wp, Vp                        # kept to evaluate the shadowing direction itself
            Q, R = jnp.linalg.qr(Wp.T)
            b = Q.T @ Vp.T                                     # [nus, npar]
            return u, Q.T, Vp - (Q @ b).T, acc, R, b, end
        return segment

    # ------------------------------------------------------------------------------------------------------------
    def _check_inputs(self, p, u0, w0):
        p = dict(p)
        missing = [n for n in self.pars if n not in p]
        if missing:
            raise KeyError(f'parameters {missing} are not in p (keys: {sorted(p)})')
        for n in self.pars:
            if np.ndim(p[n]) != 0:
                raise ValueError(f'parameter {n!r} must be a scalar, got shape {np.shape(p[n])}')
        p = {k: (float(v) if np.ndim(v) == 0 else v) for k, v in p.items()}
        u = jnp.asarray(u0, dtype=float)
        if u.ndim != 1 or not bool(jnp.all(jnp.isfinite(u))):
            raise ValueError('u0 must be a finite 1-D state vector')
        f0 = jnp.asarray(self.rhs(u, p))
        if f0.shape != u.shape:
            raise ValueError(f'rhs(u, p) has shape {f0.shape}, expected the shape of the state {u.shape}')
        if jnp.shape(self.J(u)) != ():
            raise ValueError('J(u) must return a scalar')
        if self.nus >= u.shape[0]:
            raise ValueError(f'nus = {self.nus} must be smaller than the dimension of the state ({u.shape[0]})')
        return p, u

    def run_segments(self, p: Mapping, u0, nseg: int, nseg_ps: int = 0, w0=None, seed: int = 0, verbose: bool = False):
        """
        Low-level entry point (see :meth:`run`): ``nseg`` segments are accumulated after ``nseg_ps`` segments of spin-up.
        Returns ``Javg, dJdp [npar], info`` where info has 'lyap' (nus), 'T', 'a' [nseg, nus, npar],
        'dJdp_segments' [nseg, npar], 'vnorm' [nseg, npar] and 'J_segments' [nseg].
        """
        p, u = self._check_inputs(p, u0, w0)
        nc, nus, npar, nstep = u.shape[0], self.nus, len(self.pars), self.nstep
        if w0 is None:
            w0 = np.random.RandomState(seed).rand(nus, nc)
            w0 = w0 / np.linalg.norm(w0, axis=1, keepdims=True)
        W, V = self._start(u, jnp.asarray(w0, dtype=float).reshape(nus, nc), p)
        tick = max(1, (nseg_ps + nseg) // 10)
        for i in range(nseg_ps):
            u, W, V = self._segment(u, W, V, p)[:3]
            if verbose and (i + 1) % tick == 0:
                print(f'  spin-up segment {i + 1}/{nseg_ps}', file=sys.stderr, flush=True)
        out = []
        for i in range(nseg):
            u, W, V, acc, R, b, end = self._segment(u, W, V, p)
            out.append(jax.tree_util.tree_map(np.asarray, (acc, R, b, end)))
            if verbose and (i + 1) % tick == 0:
                print(f'  segment {i + 1}/{nseg}', file=sys.stderr, flush=True)
        acc = {k: np.array([o[0][k] for o in out]) for k in out[0][0]}
        Rs, bs = np.array([o[1] for o in out]), np.array([o[2] for o in out])
        end = {k: np.array([o[3][k] for o in out]) for k in out[0][3]}
        T = nseg * self.T_seg
        if not (np.all(np.isfinite(acc['C'])) and np.all(np.isfinite(acc['D']))):
            nan = lambda *shape: np.full(shape, np.nan)
            return np.nan, np.full(npar, np.nan), {'lyap': nan(nus), 'T': T, 'a': nan(nseg, nus, npar),
                                                   'dJdp_segments': nan(nseg, npar), 'vnorm': nan(nseg, npar),
                                                   'J_segments': nan(nseg)}

        a = solve_shadowing_coeffs_multi(acc['C'], np.swapaxes(acc['D'], 1, 2), Rs[:-1], bs[:-1])    # [nseg, nus, npar]
        N = nseg * nstep
        Javg = acc['Jsum'].sum() / N
        # xi at the end of each segment, v = v* + sum_j a_j w_j (eq. 41 of Ni & Wang)
        xi = end['vf'] + np.einsum('ijk,ij->ik', a, end['wf'])                                          # [nseg, npar]
        t1 = (acc['Jv'] + np.einsum('ijk,ij->ik', a, acc['Jw'])) / N
        t2 = xi * (Javg - end['Jend'])[:, None] / N / self.dt
        # the shadowing direction (perpendicular to the flow) at the end of each segment: v* + sum_j a_j w_j
        vperp = end['Vp'] + np.einsum('ijk,ijc->ikc', a, end['Wp'])                                     # [nseg, npar, nc]
        info = {'lyap': np.log(np.abs(np.diagonal(Rs, axis1=1, axis2=2))).sum(axis=0) / (nseg * self.T_seg),
                'T': T, 'a': a, 'dJdp_segments': t1 + t2, 'vnorm': np.linalg.norm(vperp, axis=2),
                'J_segments': acc['Jsum'] / nstep}
        return Javg, (t1 + t2).sum(axis=0), info

    def run(self, p: Mapping, u0, T: float, *, T_spinup: float = 0.0, w0=None, seed: int = 0,
            verbose: bool = False) -> NILSSResult:
        """Shadowing sensitivities along the trajectory that starts at ``u0``.

        ``T_spinup`` of the trajectory is integrated first, without accumulation, to reach the attractor; the sensitivities
        are accumulated over the next ``T``. ``w0`` [nus, n] fixes the initial homogeneous tangents (random unit
        vectors from ``seed`` by default). Returns a :class:`NILSSResult`.
        """
        nseg, nseg_ps = int(round(T / self.T_seg)), int(round(T_spinup / self.T_seg))
        if nseg < 1:
            raise ValueError(f'T = {T} is shorter than one segment ({self.T_seg})')
        Javg, dJdp, info = self.run_segments(p, u0, nseg, nseg_ps, w0=w0, seed=seed, verbose=verbose)
        finite = bool(np.isfinite(Javg) and np.all(np.isfinite(dJdp)))
        return NILSSResult(J=float(Javg), dJdp={n: float(v) for n, v in zip(self.pars, dJdp)}, lyapunov=np.asarray(info['lyap']),
                           T=float(info['T']), params=self.pars, nus=self.nus, T_seg=self.T_seg, dt=self.dt,
                           segments=info['dJdp_segments'], vnorm=info['vnorm'], J_segments=info['J_segments'],
                           coeffs=info['a'], finite=finite, seed=seed)


NILSSStreamer = NILSS            # the name used in earlier versions
