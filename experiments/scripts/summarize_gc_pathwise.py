"""Summary of experiments/scripts/gc_pathwise.py: how good is the plain pathwise (forward-tangent) derivative of a finite-time ensemble
objective Phi_T = E[g(s_T)] as a function of the horizon T?

For every parameter, observable and horizon: ensemble mean +- SEM of the pathwise derivative D_i = g'(s_T) v_s,i(T),
median, relative SEM at N = 2000, Hill tail index of |D| and of the norm of the whole tangent |v|, the share of sum |D|
carried by the top 1 %, how the spread of the batch mean shrinks with the batch size (n^-1/2 for a finite variance),
the "usable horizon" T*, and checks against central finite differences (same initial conditions: the tangent code;
independent initial conditions: the population derivative).

    python experiments/scripts/summarize_gc_pathwise.py [results/gc_pathwise] [--ens E1 E2] [--json out.json]
"""
import argparse
import glob
import json
import os

import numpy as np

PARS = ('eps_t', 'eps_m', 'kappa', 'iota')
S_C, W_C = 0.15, 0.02                 # loss proxy g2(s) = logistic((s - S_C) / W_C), the default of experiments/scripts/gc_pathwise.py
LAM_CHAOS = 0.003                     # finite-time exponent log|W|/T (at the longest horizon) above which an orbit counts as chaotic
LAM_REGULAR = 0.0005                  # ... and below which it counts as regular (log|W| grows like log T); in between: weakly chaotic / sticky
REL_SEM_MAX, INDEX_MIN, N_REF = 0.3, 2.0, 2000
BATCH_SIZES = (10, 30, 100, 300, 1000)
OBSERVABLES = ('g1=s', 'g2=loss proxy', 'g3=<s>_T')


# ---- statistics --------------------------------------------------------------------------------------------------

def hill(x, k):
    """Hill estimator of the tail index of the positive sample x from its k largest values (P(X > t) ~ t^-index)."""
    x = np.sort(np.asarray(x, dtype=float))[::-1]
    x = x[x > 0]
    if len(x) <= k + 1 or k < 2:
        return float('nan')
    return float(1.0 / np.mean(np.log(x[:k] / x[k])))


def batch_dispersion(x, sizes, nboot=2000, rng=None):
    """Robust spread (interquartile range / 1.349) of the mean of n values drawn from the sample x, for n in sizes."""
    x = np.asarray(x, dtype=float)
    rng = rng or np.random.RandomState(0)
    out = {}
    for n in sizes:
        means = x[rng.randint(0, len(x), (nboot, n))].mean(axis=1)
        q1, q3 = np.quantile(means, [0.25, 0.75])
        out[n] = (q3 - q1) / 1.349
    return out


def scaling_exponent(disp):
    n = np.array(sorted(disp))
    return float(np.polyfit(np.log(n), np.log([disp[k] for k in n]), 1)[0])


def usable_horizon(table):
    """Largest T such that at every horizon up to T the relative SEM is below REL_SEM_MAX and the tail index above INDEX_MIN."""
    best = None
    for row in sorted(table, key=lambda r: r['T']):
        if row['rel_sem'] < REL_SEM_MAX and row['index'] > INDEX_MIN:
            best = row['T']
        else:
            break
    return best


def index_horizon(table):
    """Largest T such that the tail index of |D| (k = 100) is above INDEX_MIN at every horizon up to T (None if already at the first)."""
    best = None
    for row in sorted(table, key=lambda r: r['T']):
        if row['index'] > INDEX_MIN:
            best = row['T']
        else:
            break
    return best


def logistic(s):
    return 1.0 / (1.0 + np.exp(-(s - S_C) / W_C))


def observables(u, A, V, dA, times):
    """Values g(s_T) [n, nT] and pathwise derivatives D [3 observables][n, nT, NPAR] from the stored fields."""
    s = u[..., 0]
    g2 = logistic(s)
    D1 = V[..., 0]                                                       # [n, nT, NPAR]
    D2 = (g2 * (1 - g2) / W_C)[..., None] * D1
    D3 = dA / times[None, :, None]
    return [s, g2, A / times[None, :]], [D1, D2, D3]


# ---- data --------------------------------------------------------------------------------------------------------

def load_pathwise(outdir, ens):
    files = [f for f in sorted(glob.glob(os.path.join(outdir, f'pw_{ens}_*.npz'))) if '.part.' not in f]      # not the partial files of cut-off tasks
    if not files:
        return None
    parts = [dict(np.load(f)) for f in files]
    idx = np.concatenate([p['idx'] for p in parts])
    order = np.argsort(idx)
    cat = {k: np.concatenate([p[k] for p in parts])[order] for k in ('pos', 'idx', 'u', 'V', 'A', 'dA', 'logw', 'nflip', 'smax', 'smin')}
    cat['times'] = parts[0]['times']
    cat['q0'] = parts[0]['q0']
    return cat


def stats_block(D, V, n_ref=N_REF):
    """Statistics of the sample D [n] (derivatives) and v [n] (norm of the whole tangent)."""
    ok = np.isfinite(D) & np.isfinite(V)
    D, V = D[ok], V[ok]
    n = len(D)
    mean, sd = D.mean(), D.std(ddof=1)
    a = np.sort(np.abs(D))[::-1]
    return {'n': n, 'mean': mean, 'sem': sd / np.sqrt(n), 'median': float(np.median(D)),
            'rel_sem': (sd / np.sqrt(n_ref)) / max(abs(mean), 1e-300),            # at N = n_ref (extrapolated by n^-1/2 where n differs)
            'index': hill(np.abs(D), 100), 'index50': hill(np.abs(D), 50), 'index300': hill(np.abs(D), 300), 'index_v': hill(V, 100),
            'top1': float(a[:max(n // 100, 1)].sum() / a.sum()) if a.sum() > 0 else float('nan')}


def pathwise_report(data, label, summary):
    times, n = data['times'], len(data['idx'])
    vnorm = np.linalg.norm(data['V'], axis=3)                           # [n, nT, NPAR]
    G, D = observables(data['u'], data['A'], data['V'], data['dA'], times)
    lam = data['logw'] / times[None, :]
    print(f'\n===== ensemble {label}: {n} initial conditions, horizons {list(times)} =====')
    fin = np.isfinite(data['V']).all(axis=(2, 3))
    print('fraction with finite tangents per horizon:', np.round(fin.mean(axis=0), 4))
    print('mean of s_T per horizon:', np.round(np.nanmean(G[0], axis=0), 4), '   fraction with s_T > s_c =', S_C, ':',
          np.round((G[0] > S_C).mean(axis=0), 3))
    chaotic = lam[:, -1] > LAM_CHAOS
    print(f'finite-time exponent log|W|/T at T = {times[-1]:g}: median {np.median(lam[:, -1]):.5f}; chaotic fraction (> {LAM_CHAOS}): '
          f'{chaotic.mean():.3f}   [histogram of 1000*exponent: {np.histogram(1000 * lam[:, -1], bins=[-1, 0.5, 1, 2, 3, 4, 5, 6, 8, 12, 100])[0].tolist()} '
          f'for bins 0.5 1 2 3 4 5 6 8 12]')
    for T in times:
        k = list(times).index(T)
        flips = data['nflip'][:, k]
        if T == times[min(3, len(times) - 1)]:
            print(f'bounces: sign changes of v_par up to T = {T:g}: median {np.median(flips):.0f} (bounce period about {2 * T / max(np.median(flips), 1):.0f}); '
                  f'fraction of orbits that never change the sign of v_par: {(flips == 0).mean():.2f}')
    summary[label] = {'n': n, 'times': times.tolist(), 'chaotic_fraction': float(chaotic.mean()), 'stats': {}}
    for io, name in enumerate(OBSERVABLES):
        print(f'\n--- observable {name}: Phi_T = E[g(s_T)] = {np.round(np.nanmean(G[io], axis=0), 4).tolist()}')
        print('  parameter     T    mean +- SEM            median     relSEM(N=2000)  index|D| (k=50/100/300)  index|v|  top1%  usable')
        for ip, par in enumerate(PARS):
            table = []
            for k, T in enumerate(times):
                st = stats_block(D[io][:, k, ip], vnorm[:, k, ip])
                st['T'] = float(T)
                table.append(st)
                summary[label]['stats'][f'{name}|{par}|{T:g}'] = st
            tstar, tidx = usable_horizon(table), index_horizon(table)
            summary[label]['stats'][f'{name}|{par}|usable'] = tstar
            summary[label]['stats'][f'{name}|{par}|index_horizon'] = tidx
            for st in table:
                print(f"  {par:6s} {st['T']:8g}  {st['mean']:+9.4f} +- {st['sem']:.4f}   {st['median']:+9.4f}   {st['rel_sem']:10.2f}     "
                      f"{st['index50']:5.2f} {st['index']:5.2f} {st['index300']:5.2f}      {st['index_v']:5.2f}   {st['top1']:.2f}")
            print(f'  {par:6s} usable horizon T* = {tstar}' + ('' if tstar is None else f'  (lambda_1 T = {0.006 * tstar:.1f} for lambda_1 = 0.006)')
                  + f';  tail index of |D| above {INDEX_MIN:g} up to T = {tidx}')
    # spread of the batch mean versus the batch size, observable g1
    print('\n--- spread of the mean of n initial conditions (IQR/1.349 of bootstrap batch means), observable g1; exponent -0.5 for a finite variance')
    for ip, par in enumerate(PARS):
        for k, T in enumerate(times):
            d = D[0][:, k, ip]
            d = d[np.isfinite(d)]
            disp = batch_dispersion(d, BATCH_SIZES)
            print(f'  {par:6s} T={T:8g}  ' + ' '.join(f'n={n_}: {disp[n_]:.4f}' for n_ in BATCH_SIZES) + f'   exponent {scaling_exponent(disp):+.2f}')
    # chaotic versus regular orbits (classified by the finite-time exponent at the longest horizon)
    regular = lam[:, -1] < LAM_REGULAR
    middle = ~chaotic & ~regular
    print(f'\n--- orbit classes by log|W|/T at T = {times[-1]:g}: regular (< {LAM_REGULAR}) {regular.sum()}, weakly chaotic / sticky {middle.sum()}, '
          f'chaotic (> {LAM_CHAOS}) {chaotic.sum()}; observable g1, pathwise derivative at each horizon (mean +- SEM, median, Hill index k=50)')
    summary[label]['classes'] = {'regular': int(regular.sum()), 'middle': int(middle.sum()), 'chaotic': int(chaotic.sum())}
    for ip, par in enumerate(PARS):
        for k, T in enumerate(times):
            cells = []
            for name, mask in (('regular', regular), ('middle', middle), ('chaotic', chaotic)):
                d = D[0][mask, k, ip]
                d = d[np.isfinite(d)]
                if len(d) < 3:
                    cells.append(f'{name}: -')
                    continue
                cells.append(f'{name}: {d.mean():+9.3g} +- {d.std(ddof=1) / np.sqrt(len(d)):.2g} med {np.median(d):+9.3g} idx {hill(np.abs(d), 50):.2f}')
                summary[label]['stats'][f'class|{name}|{par}|{T:g}'] = {'mean': float(d.mean()), 'sem': float(d.std(ddof=1) / np.sqrt(len(d))),
                                                                      'median': float(np.median(d)), 'index50': hill(np.abs(d), 50), 'n': int(len(d))}
            print(f'  {par:6s} T={T:8g}  ' + ' | '.join(cells))
    return G, D


def fd_report(outdir, ens, data, summary):
    files = sorted(glob.glob(os.path.join(outdir, f'fd_{ens}_*.npz')))
    if not files or data is None:
        return
    parts = [dict(np.load(f)) for f in files]
    idx = np.concatenate([p['idx'] for p in parts])
    U = np.concatenate([p['u'] for p in parts], axis=3)                 # [NPAR, nh, 2, n, nT, 4]
    A = np.concatenate([p['A'] for p in parts], axis=3)
    h = parts[0]['h']                                                    # [NPAR, nh]
    times = parts[0]['times']
    sel = np.searchsorted(data['idx'], idx)
    ktimes = [list(data['times']).index(t) for t in times]
    G, D = observables(data['u'][sel][:, ktimes], data['A'][sel][:, ktimes], data['V'][sel][:, ktimes], data['dA'][sel][:, ktimes], times)
    print(f'\n===== finite differences of the sample mean (same {len(idx)} initial conditions) against the pathwise sample mean, ensemble {ens} =====')
    print('  columns per horizon: pathwise mean +- SEM; then for each h (relative step): FD mean +- its SEM, |FD - pathwise| / SEM of the pathwise mean;')
    print('  and the fraction of single orbits whose own central difference is within 10 % of their pathwise derivative')
    summary.setdefault('fd', {})
    for io, name in enumerate(OBSERVABLES[:3]):
        print(f'\n--- observable {name}')
        for ip, par in enumerate(PARS):
            for k, T in enumerate(times):
                d = D[io][:, k, ip]
                row = f'  {par:6s} T={T:8g}  pathwise {d.mean():+9.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f} |'
                for ih in range(h.shape[1]):
                    if io == 0:
                        gp_, gm_ = U[ip, ih, 0, :, k, 0], U[ip, ih, 1, :, k, 0]
                    elif io == 1:
                        gp_, gm_ = logistic(U[ip, ih, 0, :, k, 0]), logistic(U[ip, ih, 1, :, k, 0])
                    else:
                        gp_, gm_ = A[ip, ih, 0, :, k] / T, A[ip, ih, 1, :, k] / T
                    fdi = (gp_ - gm_) / (2 * h[ip, ih])
                    good = np.mean(np.abs(fdi - d) <= 0.1 * np.abs(d) + 1e-9)
                    sem = d.std(ddof=1) / np.sqrt(len(d))
                    fsem = fdi.std(ddof=1) / np.sqrt(len(fdi))
                    row += (f' h={h[ip, ih] / abs(parts[0]["q0"][ip]):.0e}: {fdi.mean():+9.4f} +- {fsem:.4f} ({abs(fdi.mean() - d.mean()) / sem:5.2f} pw-SEM, '
                            f'{good:.2f}) |')
                    summary['fd'][f'{name}|{par}|{T:g}|{ih}'] = {'fd_mean': float(fdi.mean()), 'fd_sem': float(fsem), 'pw_mean': float(d.mean()),
                                                                 'sem': float(sem), 'good': float(good)}
                print(row)


def population_report(outdir, data, summary):
    for par in PARS:
        files = sorted(glob.glob(os.path.join(outdir, f'pop_{par}_*.npz')))
        if not files:
            continue
        parts = [dict(np.load(f)) for f in files]
        times, h = parts[0]['times'], parts[0]['h']
        print(f'\n===== population finite differences of E[g(s_T)] with respect to {par}: independent initial conditions at q + h and q - h ({len(files)} blocks) =====')
        ip = PARS.index(par)
        for ih in range(len(h)):
            for io, name in enumerate(OBSERVABLES[:3]):
                row = f'  h={h[ih]:.4g} {name:14s}'
                for k, T in enumerate(times):
                    vals = {}
                    for sg in (+1, -1):
                        sub = [p for p in parts if int(p['sign']) == sg]
                        u = np.concatenate([p[f'u_{ih}'][:, k] for p in sub])
                        A = np.concatenate([p[f'A_{ih}'][:, k] for p in sub])
                        g = [u[:, 0], logistic(u[:, 0]), A / T][io]
                        vals[sg] = (g.mean(), g.std(ddof=1) / np.sqrt(len(g)), len(g))
                    der = (vals[+1][0] - vals[-1][0]) / (2 * h[ih])
                    err = np.hypot(vals[+1][1], vals[-1][1]) / (2 * h[ih])
                    ref = ''
                    if data is not None:
                        kk = list(data['times']).index(T) if T in data['times'] else None
                        if kk is not None:
                            _, D = observables(data['u'], data['A'], data['V'], data['dA'], data['times'])
                            d = D[io][:, kk, ip]
                            zz = (d.mean() - der) / np.hypot(d.std(ddof=1) / np.sqrt(len(d)), err)
                            ref = f' [pathwise N={len(d)}: {d.mean():+.4g}+-{d.std(ddof=1) / np.sqrt(len(d)):.2g}, {zz:+.1f} sigma from this]'
                    row += f'\n      T={T:7g}: {der:+8.4f} +- {err:.4f} (N={vals[+1][2]} per sign){ref}'
                    summary.setdefault('pop', {})[f'{par}|{name}|{T:g}|{ih}'] = {'der': float(der), 'err': float(err)}
                print(row)


def dt_check(outdir, other, ens='E1'):
    """Per-orbit pathwise derivatives of the runs in `other` (a smaller time step, same initial conditions) against those of `outdir`."""
    a, b = load_pathwise(outdir, ens), load_pathwise(other, ens)
    if a is None or b is None:
        print(f'(no data for the time-step check: {outdir}, {other})')
        return
    common = np.intersect1d(a['idx'], b['idx'])
    ia, ib = np.searchsorted(a['idx'], common), np.searchsorted(b['idx'], common)
    print(f'\n===== time-step check: {len(common)} initial conditions of {ens}, dt of {outdir} against dt of {other} =====')
    for k, T in enumerate(b['times']):
        ka = list(a['times']).index(T)
        row = f'  T={T:7g}: '
        for ip, par in enumerate(PARS):
            da, db = a['V'][ia, ka, ip, 0], b['V'][ib, k, ip, 0]
            rel = np.abs(da - db) / np.maximum(np.abs(db), 1e-12)
            row += f'{par}: median rel diff {np.median(rel):.1e}, 90% {np.quantile(rel, .9):.1e}, max {rel.max():.1e} | '
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir', nargs='?', default='results/gc_pathwise')
    ap.add_argument('--ens', nargs='+', default=['E1', 'E2'])
    ap.add_argument('--json', default=None)
    ap.add_argument('--dt-check', default=None, help='directory of a run with a smaller time step on the same initial conditions')
    args = ap.parse_args()
    summary = {}
    datasets = {}
    for ens in args.ens:
        data = load_pathwise(args.outdir, ens)
        if data is None:
            print(f'(no pathwise data for {ens} in {args.outdir})')
            continue
        datasets[ens] = data
        pathwise_report(data, ens, summary)
        fd_report(args.outdir, ens, data, summary)
    population_report(args.outdir, datasets.get('E1'), summary)
    if args.dt_check:
        dt_check(args.outdir, args.dt_check)
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(summary, fh, default=float)


if __name__ == '__main__':
    main()
