"""Calibration of the reliability verdicts of ``nilss_jax.diagnostics`` (the evidence behind docs/reliability.md).

The checks of ``reliability_report`` are heuristics; this script measures what they do on

* systems where NILSS is known to work (Lorenz 63 at rho = 28) and on systems that are less well behaved (Lorenz 63 at rho = 35
  and 40, Lorenz 96 with 8 variables, the Roessler attractor),
* the stored NILSS runs on the chaotic seas of the guiding-center flow (draws 581 and 835; ``results/gc_official_nilss*``),
* synthetic samples with known tails (Gaussian, Pareto with index 1 and 1.5).

For each case: the verdict and every check, the numbers behind the tail and scaling checks, and the stability of the verdict under
sub-sampling (random subsets of 8 and 16 orbits, ``--draws`` draws each; the first 25 % and 50 % of the segments of every orbit).
Orbits of the ordinary systems are computed once and cached in ``--cache`` (``EnsembleResult.save``), so the analysis can be rerun.

    python experiments/scripts/calibrate_reliability.py [--cache results/calibration] [--quick] [--cases NAME ...] > calibration.txt

Output is plain text. Run it from the repository root (the stored guiding-center runs are read from ``results/``).
"""
import argparse
import collections
import glob
import json
import os
import sys
import time

import numpy as np
import jax.numpy as jnp

from nilss_jax import NILSS, NILSSResult, fd, load_ensemble, run_ensemble, reliability_report, lyapunov
from nilss_jax.diagnostics import OK, CAUTION, UNRELIABLE, convergence_exponent, hill_index, tail_index
from nilss_jax.systems import guiding_center, lorenz63


# ---- two more systems ------------------------------------------------------------------------------------------------

def lorenz96_rhs(u, p):
    return (jnp.roll(u, -1) - jnp.roll(u, 2)) * jnp.roll(u, 1) - u + p['F']


def lorenz96_J(u):
    return jnp.mean(u ** 2)


def rossler_rhs(u, p):
    x, y, z = u
    return jnp.array([-y - z, x + p['a'] * y, p['b'] + z * (x - p['c'])])


def rossler_J(u):
    return u[2]


# ---- cases --------------------------------------------------------------------------------------------------------------

def _cached(cache, name, compute):
    d = os.path.join(cache, name)
    if os.path.isdir(d) and glob.glob(os.path.join(d, 'run_*.npz')):
        return load_ensemble(d).results
    t0 = time.time()
    ens = compute()
    ens.save(d)
    print(f'# computed {name} in {time.time() - t0:.0f} s', file=sys.stderr, flush=True)
    return ens.results


def lorenz_case(rho, T, n, cache):
    def compute():
        nil = NILSS(lorenz63.rhs, lorenz63.J, ('rho',), dt=0.005, T_seg=0.5, nus=1)
        u0s = np.array([lorenz63.initial_condition(i) for i in range(n)])
        return run_ensemble(nil, {**lorenz63.DEFAULTS, 'rho': rho}, u0s, T=T, T_spinup=20.0)
    def compute2():
        nil = NILSS(lorenz63.rhs, lorenz63.J, ('rho',), dt=0.005, T_seg=0.5, nus=2)
        u0s = np.array([lorenz63.initial_condition(i) for i in range(n)])
        return run_ensemble(nil, {**lorenz63.DEFAULTS, 'rho': rho}, u0s, T=T, T_spinup=20.0)
    u0s = np.array([lorenz63.initial_condition(i) for i in range(n)])
    fd_spec = dict(rhs=lorenz63.rhs, J=lorenz63.J, p={**lorenz63.DEFAULTS, 'rho': rho}, par='rho', h=1.0, u0s=u0s, T=2000.0, dt=0.005, T_spinup=50.0)
    return _cached(cache, f'lorenz63_rho{rho:g}_T{T:g}', compute), {'what': f'Lorenz 63, rho = {rho:g}', 'T': T, 'nus': 1, 'fd': fd_spec,
                                                                  'second': (lambda: _cached(cache, f'lorenz63_rho{rho:g}_T{T:g}_nus2', compute2))}


def lorenz96_case(T, n, cache):
    f = lambda u: lorenz96_rhs(u, {'F': 8.0})
    spec = lyapunov.lyapunov_spectrum(f, 8.0 + 0.5 * np.random.RandomState(0).randn(8), 0.01, 50.0, 4000.0)
    nus = int(np.sum(spec['lyap'] > 0.05))

    def compute():
        nil = NILSS(lorenz96_rhs, lorenz96_J, ('F',), dt=0.01, T_seg=0.5, nus=nus)
        u0s = np.array([8.0 + 0.5 * np.random.RandomState(100 + i).randn(8) for i in range(n)])
        return run_ensemble(nil, {'F': 8.0}, u0s, T=T, T_spinup=50.0)
    def compute2():
        nil = NILSS(lorenz96_rhs, lorenz96_J, ('F',), dt=0.01, T_seg=0.5, nus=nus + 1)
        u0s = np.array([8.0 + 0.5 * np.random.RandomState(100 + i).randn(8) for i in range(n)])
        return run_ensemble(nil, {'F': 8.0}, u0s, T=T, T_spinup=50.0)
    runs = _cached(cache, f'lorenz96_N8_F8_T{T:g}', compute)
    u0s = np.array([8.0 + 0.5 * np.random.RandomState(100 + i).randn(8) for i in range(n)])
    fd_spec = dict(rhs=lorenz96_rhs, J=lorenz96_J, p={'F': 8.0}, par='F', h=0.5, u0s=u0s, T=2000.0, dt=0.01, T_spinup=50.0)
    return runs, {'what': f'Lorenz 96 (8 variables, F = 8, J = mean u^2); exponents {np.array2string(spec["lyap"], precision=3)}', 'T': T, 'nus': nus, 'fd': fd_spec,
                  'second': (lambda: _cached(cache, f'lorenz96_N8_F8_T{T:g}_nus{nus + 1}', compute2))}


def rossler_case(T, n, cache):
    def compute():
        nil = NILSS(rossler_rhs, rossler_J, ('c',), dt=0.01, T_seg=1.0, nus=1)
        u0s = np.array([np.array([1.0, 1.0, 0.0]) + 0.5 * np.random.RandomState(200 + i).randn(3) for i in range(n)])
        return run_ensemble(nil, {'a': 0.2, 'b': 0.2, 'c': 5.7}, u0s, T=T, T_spinup=300.0)
    u0s = np.array([np.array([1.0, 1.0, 0.0]) + 0.5 * np.random.RandomState(200 + i).randn(3) for i in range(n)])
    fd_spec = dict(rhs=rossler_rhs, J=rossler_J, p={'a': 0.2, 'b': 0.2, 'c': 5.7}, par='c', h=0.05, u0s=u0s, T=4000.0, dt=0.01, T_spinup=300.0)
    return _cached(cache, f'rossler_T{T:g}', compute), {'what': 'Roessler (a = b = 0.2, c = 5.7, J = z, parameter c)', 'T': T, 'nus': 1, 'fd': fd_spec}


def _result_from_json(r):
    pars = tuple(r['pars'])
    nseg = len(r['segments'])
    return NILSSResult(J=r['Javg'], dJdp=dict(zip(pars, r['dJdp'])), lyapunov=np.array(r['lyap']), T=r['T'], params=pars, nus=r['nus'],
                       T_seg=r['T_seg'], dt=r['dt'], segments=np.array(r['segments']), vnorm=np.array(r['vnorm']),
                       J_segments=np.full(nseg, np.nan), coeffs=np.zeros((nseg, r['nus'], len(pars))))   # J per segment was not stored: NaN skips the ergodicity check


def gc_case(draw, directory, lyap_min):
    allrows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(directory, 'task_*.json')))]
    rows = [r for r in allrows if r['nus'] == 1 and r['T_seg'] == 200.0 and r['lyap'][0] >= lyap_min and 'vnorm' in r]
    runs = [_result_from_json(r) for r in rows]
    two = {r['copy']: r for r in allrows if r['nus'] == 2 and r['T_seg'] == 200.0 and r['lyap'][0] >= lyap_min and 'vnorm' in r}
    common = [r for r in rows if r['copy'] in two]
    second = (lambda: ([_result_from_json(r) for r in common], [_result_from_json(two[r['copy']]) for r in common]))
    ref = guiding_center.load_draw(draw)['reference_fd']
    return runs, {'what': f'guiding-center flow, chaotic sea of draw {draw} (stored runs, parameters {", ".join(runs[0].params)})' if runs else '',
                  'T': rows[0]['T'] if rows else 0, 'nus': 1, 'fd_ref': {k: (v['value'], v['err']) for k, v in ref.items()}, 'second': second, 'paired': True}


def synthetic_case(kind, n=16, nseg=1000, seed=0):
    rng = np.random.RandomState(seed)
    runs = []
    for _ in range(n):
        if kind == 'gaussian':
            v, c = np.abs(rng.randn(nseg, 1)) + 1.0, 0.01 * rng.randn(nseg, 1)
        else:
            alpha = {'pareto1': 1.0, 'pareto1.5': 1.5}[kind]
            v = rng.pareto(alpha, (nseg, 1)) + 1.0
            c = 0.01 * np.sign(rng.randn(nseg, 1)) * (rng.pareto(alpha, (nseg, 1)) + 1.0)
        jseg = 1.0 + 0.1 * rng.randn(nseg)                 # one ergodic component: every run averages the same noisy J
        runs.append(NILSSResult(J=float(jseg.mean()), dJdp={'p': float(c.sum())}, lyapunov=np.array([0.9]), T=500.0, params=('p',), nus=1,
                                T_seg=0.5, dt=0.005, segments=c, vnorm=v, J_segments=jseg, coeffs=np.zeros((nseg, 1, 1))))
    return runs, {'what': {'gaussian': 'synthetic: |v| = |Gauss| + 1, Gaussian contributions', 'pareto1': 'synthetic: Pareto tails, index 1 (|v| and contributions)',
                           'pareto1.5': 'synthetic: Pareto tails, index 1.5'}[kind], 'T': 500.0, 'nus': 1}


_FD_CACHE = {}


# ---- analysis -----------------------------------------------------------------------------------------------------------

CATEGORIES = [('number of runs', 'number of runs'), ('resolved exponent', 'resolved exponent'), ('exponent spread', 'exponent spread'),
              ('ergodicity', 'ergodicity'), ('tail of |v|', 'tail |v|'), ('tail of segments', 'tail segments'), ('error scaling', 'scaling'),
              ('nus consistency', 'nus')]


def category(check_name):
    for prefix, label in CATEGORIES:
        if check_name.startswith(prefix):
            return label
    return check_name


def truncate(r, frac):
    n = max(2, int(round(r.nseg * frac)))
    return NILSSResult(J=r.J, dJdp=r.dJdp, lyapunov=r.lyapunov, T=r.T * n / r.nseg, params=r.params, nus=r.nus, T_seg=r.T_seg, dt=r.dt,
                       segments=r.segments[:n], vnorm=r.vnorm[:n], J_segments=r.J_segments[:n], coeffs=r.coeffs[:n], finite=r.finite)


def numbers(runs):
    """Tail indices, error-scaling exponents and the top-1% share per parameter (the quantities behind the checks), over the usable runs."""
    used = [r for r in runs if r.finite and r.lyapunov_time_product >= 20.0]
    out = {}
    for i, name in enumerate(runs[0].params):
        pv = np.concatenate([r.vnorm[:, i] for r in used])
        pc = np.concatenate([np.abs(r.segments[:, i]) for r in used])
        top = np.sort(pc)[::-1]
        out[name] = {'alpha_v': tail_index(pv), 'alpha_seg': tail_index(pc), 'scaling': convergence_exponent(np.array([r.segments[:, i] for r in used])),
                     'top1': float(top[:max(1, len(top) // 100)].sum() / top.sum())}
    return out


def fractions(runs, m, ndraw, rng):
    verdicts, flagged = collections.Counter(), collections.Counter()
    for _ in range(ndraw):
        idx = rng.choice(len(runs), m, replace=False)
        rep = reliability_report([runs[i] for i in idx])
        verdicts[rep.verdict] += 1
        for cat in {category(c.name) for c in rep.checks if c.status in (CAUTION, UNRELIABLE)}:
            flagged[cat] += 1
    return {k: v / ndraw for k, v in verdicts.items()}, {k: v / ndraw for k, v in flagged.items()}


def fmt_fr(fr):
    return ' '.join(f'{k} {fr.get(k, 0.0):.2f}' for k in (OK, CAUTION, UNRELIABLE))


def analyse(name, runs, meta, ndraw, rng):
    print('=' * 118)
    print(f'{name}: {meta["what"]}')
    print(f'  T = {meta["T"]:g} per orbit, {len(runs)} orbits, nus = {meta["nus"]}')
    sens = np.array([r.dJdp_array for r in runs if r.finite])
    lam = np.array([np.max(r.lyapunov) for r in runs])
    print(f'  lambda_1: median {np.median(lam):.4g} (min {lam.min():.4g}, max {lam.max():.4g}); lambda_1 T median {np.median(lam) * meta["T"]:.4g}')
    for i, p in enumerate(runs[0].params):
        print(f'  d<J>/d{p}: mean {sens[:, i].mean():+.4g} +- {sens[:, i].std(ddof=1) / np.sqrt(len(sens)):.2g}, median {np.median(sens[:, i]):+.4g}, '
              f'10-90% [{np.quantile(sens[:, i], 0.1):+.3g}, {np.quantile(sens[:, i], 0.9):+.3g}]')
    mean = {p: sens[:, i].mean() for i, p in enumerate(runs[0].params)}
    sem = {p: sens[:, i].std(ddof=1) / np.sqrt(len(sens)) for i, p in enumerate(runs[0].params)}
    median = {p: np.median(sens[:, i]) for i, p in enumerate(runs[0].params)}
    refs = {}
    if 'fd' in meta:                       # finite differences of the ensemble average, same initial conditions at every value
        sp = meta['fd']
        t0 = time.time()
        key = (sp['par'], tuple(sorted(sp['p'].items())), sp['h'], sp['T'], len(sp['u0s']))
        if key not in _FD_CACHE:
            _FD_CACHE[key] = fd.finite_difference(sp['rhs'], sp['J'], sp['p'], sp['par'], sp['h'], sp['u0s'], sp['T'], sp['dt'], npoint=5, T_spinup=sp['T_spinup'])
        res = _FD_CACHE[key]
        refs[sp['par']] = (res.slope, res.slope_err)
        print(f'  finite differences in {sp["par"]} (h = {sp["h"]:g}, 5 points, {len(sp["u0s"])} copies of T = {sp["T"]:g}; {time.time() - t0:.0f} s): '
              f'{res.slope:+.4g} +- {res.slope_err:.2g}  (chi2/dof {res.chi2:.1f}/{res.dof})')
    if 'fd_ref' in meta:
        refs.update(meta['fd_ref'])
        print('  finite differences of the studies (docs/findings.md): ' + ', '.join(f'{k} {v[0]:+.3g} +- {v[1]:.2g}' for k, v in meta['fd_ref'].items()))
    for p_, (v, e) in refs.items():
        if p_ in mean:
            print(f'    {p_}: NILSS mean {mean[p_]:+.4g} +- {sem[p_]:.2g} ({(mean[p_] - v) / np.hypot(sem[p_], e):+.1f} sigma from the reference), '
                  f'NILSS median {median[p_]:+.4g} ({(median[p_] - v) / max(e, 1e-12):+.1f} reference errors away)')
    rep = reliability_report(runs)
    print(str(rep).replace('\n', '\n  '))
    nums = numbers(runs)
    pooled0 = np.concatenate([r.vnorm[:, 0] for r in runs if r.finite])
    print(f'  Hill index of |v| [{runs[0].params[0]}] for k = 50 / 100 / 300 / 1000: ' + ' / '.join(f'{hill_index(pooled0, k):.2f}' for k in (50, 100, 300, 1000)))
    for p, v in nums.items():
        print(f'  numbers [{p}]: Hill index of |v| {v["alpha_v"]:.2f}, of segments {v["alpha_seg"]:.2f}, error-scaling exponent {v["scaling"]:.2f}, top 1% of segments carry {v["top1"]:.0%}')
    zs = [abs((mean[p_] - v) / np.hypot(sem[p_], e)) for p_, (v, e) in refs.items() if p_ in mean]
    if 'second' in meta:                   # one more homogeneous tangent on the same orbits
        sec = meta['second']()
        a_runs, b_runs = sec if meta.get('paired') else (runs, sec[:len(runs)])
        nrep = reliability_report(a_runs, nus_check=b_runs)
        nus_checks = [c for c in nrep.checks if c.name.startswith('nus consistency')]
        for c in nus_checks:
            print(f'  nus -> nus + 1 on {len(a_runs)} common orbits: {c.status.upper()}  {c.name}: {c.value}')
        sa = np.array([r.dJdp_array for r in a_runs]); sb = np.array([r.dJdp_array for r in b_runs])
        print('  nus -> nus + 1: mean sensitivity ' + ', '.join(f'{p}: {sa[:, i].mean():+.4g} -> {sb[:, i].mean():+.4g}' for i, p in enumerate(runs[0].params))
              + f';  largest change of one orbit {np.max(np.abs(sb - sa)):.3g}')
    row = {'name': name, 'T': meta['T'], 'n': len(runs), 'verdict': rep.verdict, 'zfd': (max(zs) if zs else float('nan')), 'alpha_v': min(v['alpha_v'] for v in nums.values()),
           'alpha_seg': min(v['alpha_seg'] for v in nums.values()), 'scaling': min(v['scaling'] for v in nums.values())}
    for m in (8, 16):
        if len(runs) > m:
            fr, fl = fractions(runs, m, ndraw, rng)
            print(f'  subsets of {m} orbits ({ndraw} draws): {fmt_fr(fr)}   checks flagging: ' + (', '.join(f'{k} {v:.2f}' for k, v in sorted(fl.items())) or 'none'))
            row[f'sub{m}'] = fr
    for frac in (0.25, 0.5):
        tr = [truncate(r, frac) for r in runs]
        rp = reliability_report(tr)
        flagged = ', '.join(f'{c.name}: {c.status}' for c in rp.checks if c.status in (CAUTION, UNRELIABLE)) or 'none'
        print(f'  first {frac:.0%} of the segments (T = {tr[0].T:g}): {rp.verdict.upper()}   flagged: {flagged}')
        row[f'trunc{int(frac * 100)}'] = rp.verdict
    print(flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--cache', default='results/calibration')
    ap.add_argument('--quick', action='store_true', help='fewer and shorter orbits (smoke test)')
    ap.add_argument('--cases', nargs='*', default=None)
    ap.add_argument('--draws', type=int, default=200)
    ap.add_argument('--gc581', default='results/gc_official_nilss')
    ap.add_argument('--gc835', default='results/gc_official_nilss_seed835')
    args = ap.parse_args()
    q = args.quick
    T1, T2, N = (30.0, 60.0, 8) if q else (200.0, 1000.0, 32)
    cases = {
        'lorenz63_rho28_T200': lambda: lorenz_case(28.0, T1, N, args.cache),
        'lorenz63_rho28_T1000': lambda: lorenz_case(28.0, T2, N, args.cache),
        'lorenz63_rho35_T200': lambda: lorenz_case(35.0, T1, N, args.cache),
        'lorenz63_rho35_T1000': lambda: lorenz_case(35.0, T2, N, args.cache),
        'lorenz63_rho40_T200': lambda: lorenz_case(40.0, T1, N, args.cache),
        'lorenz63_rho40_T1000': lambda: lorenz_case(40.0, T2, N, args.cache),
        'lorenz96': lambda: lorenz96_case(100.0 if q else 500.0, N, args.cache),
        'rossler': lambda: rossler_case(300.0 if q else 1000.0, N, args.cache),
        'gc_draw581': lambda: gc_case(581, args.gc581, 0.003),
        'gc_draw835': lambda: gc_case(835, args.gc835, 0.001),
        'synthetic_gaussian': lambda: synthetic_case('gaussian'),
        'synthetic_pareto1.5': lambda: synthetic_case('pareto1.5'),
        'synthetic_pareto1': lambda: synthetic_case('pareto1'),
    }
    names = args.cases or list(cases)
    rng = np.random.RandomState(12345)
    rows = []
    print('Calibration of nilss_jax.reliability_report (command: python experiments/scripts/calibrate_reliability.py)')
    print('thresholds: Hill index < 2 unreliable, < 4 caution; error-scaling exponent < 0.2 unreliable, < 0.35 caution (0.5 expected); lambda_1 T >= 20; ergodicity ratio <= 3')
    print()
    for name in names:
        runs, meta = cases[name]()
        if not runs:
            print(f'{name}: no stored runs found, skipped\n')
            continue
        # 16 and 32 orbit versions of the ordinary systems
        rows.append(analyse(name, runs, meta, args.draws, rng))
        if name.startswith(('lorenz63_rho28', 'lorenz96', 'rossler')) and len(runs) >= 32:
            rows.append(analyse(name + ' [first 16 orbits]', runs[:16], meta, args.draws, rng))

    print('=' * 118)
    print('SUMMARY (alpha_v, alpha_seg: smallest Hill index over the parameters; scal: smallest error-scaling exponent;')
    print('         |NILSS-FD|: largest deviation of the NILSS mean from the finite-difference reference, in combined standard errors)')
    print(f'{"case":<42}{"T":>8}{"orbits":>7}  {"verdict":<11}{"alpha_v":>8}{"alpha_seg":>10}{"scal":>6}  {"|NILSS-FD|":>10}   subsets of 8: ok/caution/unreliable   25%  50%')
    for r in rows:
        s8 = r.get('sub8')
        print(f'{r["name"]:<42}{r["T"]:>8g}{r["n"]:>7}  {r["verdict"]:<11}{r["alpha_v"]:>8.2f}{r["alpha_seg"]:>10.2f}{r["scaling"]:>6.2f}  {r["zfd"]:>8.1f}sg   '
              + (f'{s8.get(OK, 0):.2f} / {s8.get(CAUTION, 0):.2f} / {s8.get(UNRELIABLE, 0):.2f}' if s8 else '-'.ljust(20)) + f'   {r["trunc25"][:4]:<5}{r["trunc50"][:4]}')
    print()
    well = [r for r in rows if r['name'].startswith(('lorenz63_rho28', 'synthetic_gaussian'))]
    bad = [r for r in rows if r['name'].startswith(('gc_draw', 'synthetic_pareto'))]
    print('FALSE ALARMS (well-behaved cases not OK): ' + (', '.join(f'{r["name"]} ({r["verdict"]})' for r in well if r['verdict'] != OK) or 'none at full size'))
    for r in well:
        for m in (8, 16):
            fr = r.get(f'sub{m}')
            if fr:
                print(f'  {r["name"]}: subsets of {m}: flagged (caution or unreliable) in {1 - fr.get(OK, 0):.1%} of the draws')
    print('MISSES (known-bad cases not UNRELIABLE): ' + (', '.join(f'{r["name"]} ({r["verdict"]})' for r in bad if r['verdict'] != UNRELIABLE) or 'none at full size'))
    for r in bad:
        for m in (8, 16):
            fr = r.get(f'sub{m}')
            if fr:
                print(f'  {r["name"]}: subsets of {m}: UNRELIABLE in {fr.get(UNRELIABLE, 0):.1%}, OK in {fr.get(OK, 0):.1%} of the draws')


if __name__ == '__main__':
    main()
