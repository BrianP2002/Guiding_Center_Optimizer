"""Can a NILSS estimate be trusted? Diagnostics computed from the output of one or many runs.

NILSS differentiates long-time averages of chaotic flows under the shadowing assumptions (uniform hyperbolicity). In systems
that are only partly chaotic -- Hamiltonian flows with a mixed phase space, intermittent flows -- the shadowing direction
can have a power-law tail, the sample mean of the per-segment contributions then has infinite variance, and the error bar of
the sample is meaningless however long the run. These checks look for that and for the other things that are known to go wrong:

``resolved exponent``   lambda_1 * T of every run (no positive exponent resolved => the assumptions are not supported).
``ergodicity``          the runs must agree on <J> within their own sampling error (one ergodic component).
``exponent spread``     runs whose exponent is far below the others are probably on another component (e.g. a regular torus).
``tail of |v|``         Hill tail index of the norm of the shadowing direction; index < 2 means infinite variance.
``tail of segments``    the same for the per-segment contributions to the sensitivity.
``error scaling``       how fast the spread of the estimate shrinks with the averaging time (1/2 for a finite variance).
``nus consistency``     (optional) the estimate must not change when one more homogeneous tangent is used.

The thresholds are heuristics calibrated on the cases of ``docs/reliability.md``: Lorenz 63 at rho = 28 passes every check; the
chaotic seas of the guiding-center flow fail the tail checks with an index of 1.0. A verdict of ``ok`` means that none of the
known failure signatures is present, not that the answer is right: compare with finite differences (:mod:`nilss_jax.fd`)
for any sensitivity that matters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

OK, CAUTION, UNRELIABLE, SKIPPED = 'ok', 'caution', 'unreliable', 'skipped'
_RANK = {SKIPPED: -1, OK: 0, CAUTION: 1, UNRELIABLE: 2}

# thresholds (see the module docstring)
RESOLVED_OK, RESOLVED_MIN = 20.0, 3.0          # lambda_1 * T
TAIL_UNRELIABLE, TAIL_CAUTION = 2.0, 4.0       # Hill index: infinite variance below 2, infinite kurtosis below 4
SCALING_UNRELIABLE, SCALING_CAUTION = 0.2, 0.35   # exponent a of IQR ~ L^-a (0.5 expected)
ERGODIC_CAUTION = 3.0                          # spread of <J> over the runs / their sampling error (about 1 expected)
EXPONENT_SPREAD = 0.3                          # a run with lambda_1 below this fraction of the median is suspicious


def hill_index(x, k: int) -> float:
    """Hill estimator of the tail index alpha, P(|X| > t) ~ t^-alpha, from the k largest values of |x|.

    alpha < 2: infinite variance; alpha < 1: infinite mean. Light-tailed samples give large values (> 5).
    Returns NaN if the sample is too small.
    """
    x = np.sort(np.abs(np.asarray(x, dtype=float).ravel()))[::-1]
    x = x[np.isfinite(x) & (x > 0)]
    if k < 2 or len(x) < k + 1:
        return float('nan')
    return float(1.0 / np.mean(np.log(x[:k] / x[k])))


def tail_index(x, fraction: float = 0.05, kmin: int = 20, kmax: int = 1000) -> float:
    """Hill index with k = ``fraction`` of the sample, limited to [kmin, kmax]."""
    n = np.size(x)
    k = int(min(max(fraction * n, kmin), kmax))
    return hill_index(x, k)


def convergence_exponent(contrib, min_blocks: int = 20, ladder=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)) -> float:
    """Exponent a of the interquartile range of the sensitivity estimated from sub-runs of L segments, IQR(L) ~ L^-a.

    ``contrib`` is [n_runs, nseg] (or [nseg]) of per-segment contributions to the sensitivity. A finite-variance estimator
    with decaying correlations has a = 1/2; a Cauchy-like one has a = 0. NaN if there are too few blocks.
    """
    c = np.atleast_2d(np.asarray(contrib, dtype=float))
    n_runs, nseg = c.shape
    Ls, iqrs = [], []
    for L in ladder:
        nb = nseg // L
        if nb * n_runs < min_blocks:
            break
        est = c[:, :nb * L].reshape(n_runs, nb, L).sum(axis=2).ravel() * (nseg / L)
        q1, q3 = np.quantile(est, [0.25, 0.75])
        if q3 - q1 > 0:
            Ls.append(L)
            iqrs.append(q3 - q1)
    if len(Ls) < 3:
        return float('nan')
    return float(-np.polyfit(np.log(Ls), np.log(iqrs), 1)[0])


@dataclass
class Check:
    name: str
    status: str
    value: str
    detail: str = ''


@dataclass
class ReliabilityReport:
    verdict: str
    checks: List[Check] = field(default_factory=list)
    n_runs: int = 0
    n_used: int = 0
    advice: str = ''

    def __str__(self) -> str:
        w = max(len(c.name) for c in self.checks) if self.checks else 10
        lines = [f'NILSS reliability report: {self.n_used} of {self.n_runs} runs used']
        for c in self.checks:
            tag = {OK: 'ok        ', CAUTION: 'CAUTION   ', UNRELIABLE: 'UNRELIABLE', SKIPPED: 'skipped   '}[c.status]
            lines.append(f'  {tag}  {c.name:<{w}}  {c.value}' + (f'   ({c.detail})' if c.detail else ''))
        lines.append(f'verdict: {self.verdict.upper()}. {self.advice}')
        return '\n'.join(lines)

    def to_dict(self) -> dict:
        return {'verdict': self.verdict, 'n_runs': self.n_runs, 'n_used': self.n_used, 'advice': self.advice,
                'checks': [c.__dict__ for c in self.checks]}

    @property
    def ok(self) -> bool:
        return self.verdict == OK


def _worst(statuses):
    return max(statuses, key=lambda s: _RANK[s]) if statuses else SKIPPED


def _tail_status(alpha):
    if not np.isfinite(alpha):
        return SKIPPED
    return UNRELIABLE if alpha < TAIL_UNRELIABLE else CAUTION if alpha < TAIL_CAUTION else OK


def reliability_report(runs, *, nus_check=None, min_lyap_time: float = 20.0) -> ReliabilityReport:
    """Run the checks on a list of :class:`~nilss_jax.NILSSResult` (or an :class:`~nilss_jax.EnsembleResult`).

    ``nus_check``: optionally a second ensemble computed with one more homogeneous tangent from the same initial conditions
    (same order); the estimates of each orbit are compared.
    Runs with lambda_1 * T below ``min_lyap_time`` (or non-finite) are left out, as in :func:`nilss_jax.run_ensemble`.
    """
    results = list(getattr(runs, 'results', runs))
    n_all = len(results)
    used = [r for r in results if r.finite and r.lyapunov_time_product >= min_lyap_time]
    checks: List[Check] = []
    rep = ReliabilityReport(verdict=SKIPPED, checks=checks, n_runs=n_all, n_used=len(used))
    if not used:
        checks.append(Check('resolved exponent', UNRELIABLE, 'no run has lambda_1 * T >= %g' % min_lyap_time,
                            'the trajectories show no resolved positive exponent: NILSS does not apply'))
        rep.verdict = UNRELIABLE
        rep.advice = 'Check the system with nilss_jax.lyapunov and use longer runs.'
        return rep
    params = used[0].params
    lyap1 = np.array([np.max(r.lyapunov) for r in used])
    T = used[0].T

    # number of runs
    if len(used) < 8:
        checks.append(Check('number of runs', CAUTION, f'{len(used)}', 'fewer than 8 runs: no reliable statistics of the run-to-run scatter'))
    else:
        checks.append(Check('number of runs', OK, f'{len(used)}'))

    # resolved exponent
    prod = float(np.median(lyap1) * T)
    st = OK if prod >= RESOLVED_OK else CAUTION if prod >= RESOLVED_MIN else UNRELIABLE
    checks.append(Check('resolved exponent', st, f'median lambda_1 = {np.median(lyap1):.4g}, lambda_1 T = {prod:.3g}',
                        'want lambda_1 T >= %g' % RESOLVED_OK if st != OK else ''))

    # exponent spread
    if len(used) >= 4:
        low = lyap1 < EXPONENT_SPREAD * np.median(lyap1)
        st = CAUTION if low.any() else OK
        checks.append(Check('exponent spread', st, f'{int(low.sum())} of {len(used)} runs below {EXPONENT_SPREAD:g} x the median exponent',
                            'those orbits are probably on another component (regular torus?)' if low.any() else ''))

    # ergodicity: <J> must agree between the runs within their own sampling error
    if len(used) >= 4 and used[0].nseg >= 40:
        Js = np.array([r.J for r in used])
        se = []
        for r in used:
            nb = 20
            blocks = r.J_segments[:r.nseg // nb * nb].reshape(nb, -1).mean(axis=1)
            se.append(blocks.std(ddof=1) / np.sqrt(nb))
        ratio = float(Js.std(ddof=1) / np.mean(se)) if np.mean(se) > 0 else float('nan')
        st = SKIPPED if not np.isfinite(ratio) else CAUTION if ratio > ERGODIC_CAUTION else OK
        checks.append(Check('ergodicity', st, f'spread of <J> over the runs / sampling error of one run = {ratio:.2f}',
                            'about 1 for one ergodic component; large: the runs sample different parts of the phase space or T is too short'
                            if st == CAUTION else ''))

    # tails and scaling, per parameter
    pooled_v = lambda i: np.concatenate([r.vnorm[:, i] for r in used])
    pooled_c = lambda i: np.concatenate([np.abs(r.segments[:, i]) for r in used])
    for i, name in enumerate(params):
        a_v, a_c = tail_index(pooled_v(i)), tail_index(pooled_c(i))
        s_v, s_c = _tail_status(a_v), _tail_status(a_c)
        top = np.sort(pooled_c(i))[::-1]
        share = float(top[:max(1, len(top) // 100)].sum() / top.sum()) if top.sum() > 0 else float('nan')
        checks.append(Check(f'tail of |v| [{name}]', s_v, f'Hill index {a_v:.2f}',
                            'index < 2: infinite variance, the standard errors are meaningless' if s_v == UNRELIABLE else
                            'finite variance but heavy tail' if s_v == CAUTION else ''))
        checks.append(Check(f'tail of segments [{name}]', s_c, f'Hill index {a_c:.2f}, top 1% of the segments carry {share:.0%} of sum |contributions|'))
        contrib = np.array([r.segments[:, i] for r in used])
        a = convergence_exponent(contrib)
        st = SKIPPED if not np.isfinite(a) else UNRELIABLE if a < SCALING_UNRELIABLE else CAUTION if a < SCALING_CAUTION else OK
        checks.append(Check(f'error scaling [{name}]', st, 'not enough segments' if not np.isfinite(a) else f'spread ~ L^-{a:.2f} (0.5 expected)',
                            'the estimate converges much more slowly than 1/sqrt(T)' if st in (CAUTION, UNRELIABLE) else ''))

    # nus consistency
    if nus_check is not None:
        other = list(getattr(nus_check, 'results', nus_check))
        pairs = [(a, b) for a, b in zip(results, other) if a.finite and b.finite and a.lyapunov_time_product >= min_lyap_time]
        if len(pairs) >= 4:
            worst = 0
            for i, name in enumerate(params):
                x = np.array([a.dJdp_array[i] for a, _ in pairs])
                d = np.array([b.dJdp_array[i] - a.dJdp_array[i] for a, b in pairs])
                sigma = 1.4826 * np.median(np.abs(x - np.median(x)))
                bad = int(np.sum(np.abs(d) > 3 * max(sigma, 1e-12))) if sigma > 0 else 0
                worst = max(worst, bad / len(pairs))
                if bad:
                    checks.append(Check(f'nus consistency [{name}]', CAUTION, f'{bad} of {len(pairs)} orbits change by more than 3 robust sigma with one more tangent'))
            if worst == 0:
                checks.append(Check('nus consistency', OK, f'{len(pairs)} orbits unchanged with one more tangent'))
        else:
            checks.append(Check('nus consistency', SKIPPED, 'too few common orbits'))

    rep.verdict = _worst([c.status for c in checks])
    rep.advice = {
        OK: 'No known failure signature found. This does not prove the sensitivity right: check the ones you rely on against finite differences (nilss_jax.fd).',
        CAUTION: 'Treat the error bars as optimistic and compare with finite differences (nilss_jax.fd) before using the sensitivities.',
        UNRELIABLE: 'The sample mean has infinite variance or the assumptions are not met: the printed standard errors are meaningless and the values '
                    'can be biased. Use NILSS output here only as a hint and verify with ensemble finite differences (nilss_jax.fd).',
        SKIPPED: 'Not enough data for a verdict.'}[rep.verdict]
    return rep
