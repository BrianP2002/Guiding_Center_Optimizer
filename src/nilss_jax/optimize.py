"""Gradient-based optimisation of a long-time average, with NILSS or finite-difference gradients.

``minimize`` wraps ``scipy.optimize.minimize`` (L-BFGS-B by default). At every evaluation the ensemble of orbits (the same
initial conditions and seeds every time, so that the objective is a smoother function of the parameters) gives <J> and its
sensitivities. The reliability report of :mod:`nilss_jax.diagnostics` is evaluated each time: where NILSS cannot be trusted
(``on_unreliable``), either use ``gradient='fd'`` -- central finite differences of the ensemble average, at about
``2 * len(free) + 1`` times the cost of one evaluation of <J> -- or accept a warning.

    res = minimize(nilss, p0, free=['rho'], u0s=u0s, T=100., T_spinup=20., bounds={'rho': (24, 40)}, target=30.0)
    res.x, res.J
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from . import fd as _fd
from .core import NILSS
from .ensemble import run_ensemble

__all__ = ['OptimizationResult', 'minimize']


@dataclass
class OptimizationResult:
    x: dict                         # optimal values of the free parameters
    fun: float                      # objective at x
    J: float                        # ensemble mean of <J> at x
    success: bool
    message: str
    nit: int
    history: list = field(default_factory=list)     # one dict per evaluation: x, J, grad (d<J>/dx), objective, verdict
    scipy_result: object = None


def minimize(nilss: NILSS, p0: Mapping, free: Sequence[str], u0s, T: float, *, bounds: Optional[Mapping] = None,
             target: Optional[float] = None, maximize: bool = False, T_spinup: float = 0.0, gradient: str = 'nilss',
             aggregate: str = 'mean', fd_h=None, fd_project: Optional[Callable] = None, method: str = 'L-BFGS-B', maxiter: int = 20,
             min_lyap_time: float = 20.0, on_unreliable: str = 'warn', workers: int = 1, seeds=None,
             verbose: bool = False) -> OptimizationResult:
    """Minimise <J> (or ``-<J>`` with ``maximize``, or ``(<J> - target)**2`` with ``target``) over the parameters ``free``.

    Parameters
    ----------
    nilss : the compiled :class:`NILSS` problem; ``free`` must be among its ``params`` when ``gradient='nilss'``.
    p0 : all parameters (starting values of the free ones).
    u0s : [n_orbits, n] initial conditions of the ensemble, reused at every evaluation.
    gradient : ``'nilss'`` (ensemble ``aggregate`` = ``'mean'`` or ``'median'`` of the NILSS sensitivities) or ``'fd'`` (central
        finite differences of the ensemble average with step ``fd_h``, a float or a dict by parameter name; ``fd_project`` as in
        :func:`nilss_jax.fd.finite_difference`).
    on_unreliable : ``'warn'`` (once), ``'raise'`` or ``'ignore'`` when the reliability verdict of a NILSS evaluation is 'unreliable'.
    """
    from scipy.optimize import minimize as scipy_minimize
    free = list(free)
    if gradient not in ('nilss', 'fd'):
        raise ValueError("gradient must be 'nilss' or 'fd'")
    if aggregate not in ('mean', 'median'):
        raise ValueError("aggregate must be 'mean' or 'median'")
    if on_unreliable not in ('warn', 'raise', 'ignore'):
        raise ValueError("on_unreliable must be 'warn', 'raise' or 'ignore'")
    if gradient == 'nilss':
        missing = [n for n in free if n not in nilss.pars]
        if missing:
            raise ValueError(f'free parameters {missing} are not differentiated by this NILSS object (params = {nilss.pars})')
    else:
        if fd_h is None:
            raise ValueError("gradient='fd' needs fd_h")
        fd_h = {n: float(fd_h[n] if isinstance(fd_h, Mapping) else fd_h) for n in free}
    u0s = np.atleast_2d(np.asarray(u0s, dtype=float))
    history, warned = [], [False]
    bounds_list = [tuple((bounds or {}).get(n, (None, None))) for n in free]

    def ensemble_J(p):
        proj = (lambda pv: np.array([fd_project(u, pv) for u in u0s])) if fd_project is not None else (lambda pv: u0s)
        return float(_fd.time_average(nilss.rhs, nilss.J, p, proj(p), T, nilss.dt, T_spinup=T_spinup).mean())

    def evaluate(x):
        p = {**p0, **dict(zip(free, map(float, x)))}
        verdict = None
        if gradient == 'nilss':
            ens = run_ensemble(nilss, p, u0s, T, T_spinup=T_spinup, seeds=seeds, min_lyap_time=min_lyap_time, workers=workers)
            if not ens.kept.any():
                raise RuntimeError('no usable NILSS run at ' + str(p) + ' (lambda_1 T too small or non-finite)')
            Jbar = float(ens.J.mean())
            g = ens.mean() if aggregate == 'mean' else ens.median()
            dJ = np.array([g[n] for n in free])
            verdict = ens.report().verdict
            if verdict == 'unreliable' and on_unreliable != 'ignore':
                msg = 'the NILSS reliability verdict is UNRELIABLE (heavy tails or unresolved exponent): see ensemble.report(); consider gradient="fd"'
                if on_unreliable == 'raise':
                    raise RuntimeError(msg)
                if not warned[0]:
                    warnings.warn(msg)
                    warned[0] = True
        else:
            Jbar = ensemble_J(p)
            dJ = np.array([(ensemble_J({**p, n: p[n] + fd_h[n]}) - ensemble_J({**p, n: p[n] - fd_h[n]})) / (2 * fd_h[n]) for n in free])
        if target is not None:
            f, df = (Jbar - target) ** 2, 2 * (Jbar - target)
        else:
            f, df = (-Jbar, -1.0) if maximize else (Jbar, 1.0)
        history.append({'x': dict(zip(free, map(float, x))), 'J': Jbar, 'grad': dJ.copy(), 'objective': f, 'verdict': verdict})
        if verbose:
            print(f'  x = {np.array2string(np.asarray(x), precision=5)}  <J> = {Jbar:.6g}  objective = {f:.6g}  d<J>/dx = {np.array2string(dJ, precision=4)}'
                  + (f'  [{verdict}]' if verdict else ''), flush=True)
        return f, df * dJ

    x0 = np.array([float(p0[n]) for n in free])
    res = scipy_minimize(evaluate, x0, jac=True, method=method, bounds=bounds_list, options={'maxiter': maxiter})
    match = [h for h in history if np.allclose([h['x'][n] for n in free], res.x, rtol=0, atol=1e-12)]
    best = match[-1] if match else history[-1]
    return OptimizationResult(x=dict(zip(free, map(float, res.x))), fun=float(res.fun), J=float(best['J']), success=bool(res.success),
                              message=str(res.message), nit=int(res.nit), history=history, scipy_result=res)
