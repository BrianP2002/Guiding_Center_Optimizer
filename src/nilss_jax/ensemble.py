"""Ensembles of NILSS runs: robust statistics over many orbits, and the reliability report.

One NILSS run follows one trajectory. For a system that is not uniformly hyperbolic the run-to-run scatter is large and
heavy-tailed, so a sensitivity is only meaningful as the statistic of many runs from different initial conditions on the same
attractor, together with the diagnostics of :mod:`nilss_jax.diagnostics`.

    ens = run_ensemble(nilss, p, u0s, T=200.0, T_spinup=20.0, workers=4)
    print(ens.summary())          # mean +- SEM, median with bootstrap interval, quantiles, reliability verdict
    print(ens.report())           # the individual checks

For clusters, write one result per task with :meth:`NILSSResult.save` and collect with :func:`load_ensemble`.
"""
from __future__ import annotations

import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

import numpy as np

from .core import NILSS, NILSSResult
from .diagnostics import ReliabilityReport, reliability_report

__all__ = ['EnsembleResult', 'run_ensemble', 'load_ensemble']

_CACHE: dict = {}


def _run_one(spec, p, u0, T, T_spinup, seed):
    """Worker: build (once per process) the NILSS object of ``spec`` and run one orbit."""
    nil = _CACHE.get(spec)
    if nil is None:
        rhs, J, params, dt, T_seg, nus, invariant = spec
        nil = _CACHE[spec] = NILSS(rhs, J, params, dt, T_seg, nus, invariant)
    return nil.run(p, u0, T, T_spinup=T_spinup, seed=seed)


@dataclass(repr=False)
class EnsembleResult:
    """The runs of an ensemble and the statistics over those that are usable (finite, lambda_1 T >= ``min_lyap_time``)."""
    results: list
    min_lyap_time: float = 20.0

    def __repr__(self) -> str:
        return f'EnsembleResult(n_runs={len(self.results)}, n_used={int(self.kept.sum())}, params={self.params}, min_lyap_time={self.min_lyap_time:g})'

    @property
    def params(self) -> tuple:
        return self.results[0].params

    @property
    def kept(self) -> np.ndarray:
        return np.array([r.finite and r.lyapunov_time_product >= self.min_lyap_time for r in self.results], dtype=bool)

    def __len__(self) -> int:
        return len(self.results)

    @property
    def used(self) -> list:
        return [r for r, k in zip(self.results, self.kept) if k]

    def values(self) -> np.ndarray:
        """Sensitivities of the usable runs, [n_used, npar] in the order of ``params``."""
        return np.array([r.dJdp_array for r in self.used])

    @property
    def J(self) -> np.ndarray:
        return np.array([r.J for r in self.used])

    def mean(self) -> dict:
        return dict(zip(self.params, self.values().mean(axis=0)))

    def sem(self) -> dict:
        v = self.values()
        return dict(zip(self.params, v.std(axis=0, ddof=1) / np.sqrt(len(v)))) if len(v) > 1 else {n: float('nan') for n in self.params}

    def median(self) -> dict:
        return dict(zip(self.params, np.median(self.values(), axis=0)))

    def median_interval(self, level: float = 0.95, nboot: int = 2000, seed: int = 0) -> dict:
        """Bootstrap interval of the median over the runs."""
        v = self.values()
        rng = np.random.RandomState(seed)
        idx = rng.randint(0, len(v), (nboot, len(v)))
        meds = np.median(v[idx], axis=1)                                  # [nboot, npar]
        lo, hi = np.quantile(meds, [(1 - level) / 2, (1 + level) / 2], axis=0)
        return {n: (float(a), float(b)) for n, a, b in zip(self.params, lo, hi)}

    def trimmed_mean(self, fraction: float = 0.1) -> dict:
        """Mean after cutting ``fraction`` of the runs from each end (robust to the outliers of heavy tails, but biased
        when the tails carry part of the answer)."""
        from scipy.stats import trim_mean
        return dict(zip(self.params, trim_mean(self.values(), fraction, axis=0)))

    def report(self, nus_check=None) -> ReliabilityReport:
        """:func:`nilss_jax.reliability_report` of this ensemble."""
        return reliability_report(self, nus_check=nus_check, min_lyap_time=self.min_lyap_time)

    def summary(self, nus_check=None) -> str:
        v = self.values()
        if len(v) == 0:
            return f'{len(self.results)} runs, none usable (lambda_1 T < {self.min_lyap_time:g} or non-finite): ' + str(self.report(nus_check))
        mean, sem, med, ci = self.mean(), self.sem(), self.median(), self.median_interval()
        q = np.quantile(v, [0.1, 0.9], axis=0)
        lines = [f'NILSS ensemble: {len(v)} usable runs of {len(self.results)} (T = {self.used[0].T:g} each), '
                 f'<J> = {self.J.mean():.5g} +- {self.J.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan"):.2g}',
                 '  parameter       mean +- SEM            median [95% bootstrap]          10-90% of runs        fraction > 0']
        for i, n in enumerate(self.params):
            lines.append(f'  {n:<12} {mean[n]:+10.4g} +- {sem[n]:<9.3g}  {med[n]:+10.4g} [{ci[n][0]:+.4g}, {ci[n][1]:+.4g}]   '
                         f'[{q[0, i]:+.3g}, {q[1, i]:+.3g}]   {np.mean(v[:, i] > 0):.2f}')
        rep = self.report(nus_check)
        lines.append(f'reliability: {rep.verdict.upper()} -- {rep.advice}')
        return '\n'.join(lines)

    def save(self, directory) -> None:
        """One ``run_XXXX.npz`` per run in ``directory``."""
        os.makedirs(directory, exist_ok=True)
        for i, r in enumerate(self.results):
            r.save(os.path.join(directory, f'run_{i:04d}.npz'))


def load_ensemble(directory, min_lyap_time: float = 20.0) -> EnsembleResult:
    """Collect the ``run_*.npz`` files written by :meth:`EnsembleResult.save` or by cluster tasks."""
    files = sorted(glob.glob(os.path.join(directory, 'run_*.npz')))
    if not files:
        raise FileNotFoundError(f'no run_*.npz in {directory}')
    return EnsembleResult([NILSSResult.load(f) for f in files], min_lyap_time=min_lyap_time)


def run_ensemble(nilss: NILSS, p: Mapping, u0s, T: float, *, T_spinup: float = 0.0, seeds=None, min_lyap_time: float = 20.0,
                 workers: int = 1, executor=None, verbose: bool = False, save_dir: Optional[str] = None) -> EnsembleResult:
    """Run :meth:`NILSS.run` from every row of ``u0s`` and collect the results.

    Parameters
    ----------
    u0s : [n_runs, n] initial conditions, on the attractor or close to it (``T_spinup`` brings them there).
    seeds : integer seeds of the initial homogeneous tangents (default 0, 1, 2, ...).
    min_lyap_time : runs with lambda_1 * T below this are not used in the statistics (no resolved positive exponent).
    workers : > 1 runs the orbits in that many processes (``spawn``; ``rhs``, ``J`` and ``invariant`` must then be functions
        defined at the top level of an importable module, not lambdas or closures). ``executor`` takes any
        ``concurrent.futures.Executor`` instead. The default runs the orbits one after the other in this process.
    save_dir : if given, every result is written there as it finishes (``run_XXXX.npz``).
    """
    u0s = np.atleast_2d(np.asarray(u0s, dtype=float))
    seeds = list(range(len(u0s))) if seeds is None else list(seeds)
    if len(seeds) != len(u0s):
        raise ValueError('need one seed per initial condition')
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    def done(i, r):
        if save_dir is not None:
            r.save(os.path.join(save_dir, f'run_{i:04d}.npz'))
        if verbose:
            print(f'  run {i + 1}/{len(u0s)}: <J> = {r.J:.5g}, lambda_1 = {np.max(r.lyapunov):.4g}, dJdp = {np.array2string(r.dJdp_array, precision=4)}',
                  file=sys.stderr, flush=True)
        return r

    if workers > 1 or executor is not None:
        import multiprocessing
        pool = executor or ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'))
        try:
            futs = [pool.submit(_run_one, nilss._spec, dict(p), u0, T, T_spinup, s) for u0, s in zip(u0s, seeds)]
            results = [done(i, f.result()) for i, f in enumerate(futs)]
        except Exception as e:
            msg = str(e).lower()
            if type(e).__name__ == 'PicklingError' or any(t in msg for t in ('pickl', 'local object', "can't get", 'lambda')):
                raise ValueError('workers > 1 needs rhs, J and invariant to be top-level functions of an importable module '
                                 '(lambdas and closures cannot be sent to other processes)') from e
            raise
        finally:
            if executor is None:
                pool.shutdown()
    else:
        results = [done(i, nilss.run(p, u0, T, T_spinup=T_spinup, seed=s)) for i, (u0, s) in enumerate(zip(u0s, seeds))]
    return EnsembleResult(results, min_lyap_time=min_lyap_time)
