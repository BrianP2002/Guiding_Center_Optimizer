"""Summary of experiments/scripts/gc_collision.py.

(a) response curves: Phi_T(eps_t) = E[g(s_T)] on a grid of eps_t, independent paths at every grid point (the same initial
    states). Slopes at the centre by generalised least squares on the paired differences G_k - G_centre (the paths share
    their initial state across the grid, so the means are strongly correlated; the covariance of the differences is
    estimated from the paths), chi^2 of polynomial fits as the measure of smoothness.
(b) pathwise derivative d g(s_T) / d eps_t with common random numbers: mean +- standard error, tails (Hill index, share of
    the top 1 %), scale of the estimator against T, convergence of batch means with the batch size.

The model has no wall: with collisions some paths drift to s >> 1 (B -> 0 at s ~ 36, then the state overflows). The
observables therefore treat s_T >= S_WALL = 1 (the last closed flux surface) and non-finite states as lost:
    g1 = min(s_T, 1)                       (derivative 0 for lost paths)
    g2 = logistic((s_T - S_C) / S_W)       (S_C, S_W must be those of experiments/scripts/gc_collision.py; = 1 for lost paths)
    g3 = 1[s_T >= 1]                        (loss probability; response curves only, a pathwise derivative does not exist)

    python experiments/scripts/summarize_gc_collision.py [results/gc_collision_resp [results/gc_collision_crn]]
"""
import glob
import os
import sys

import numpy as np

S_C, S_W, S_WALL = 0.15, 0.02, 1.0
NAMES = ('g1 = min(s_T, 1)', 'g2 = logistic((s_T - S_C) / S_W)', 'g3 = 1[s_T >= 1] (loss probability)')


def g_values(s, which):
    s = np.where(np.isfinite(s), s, np.inf)              # a non-finite state is a lost particle
    if which == 0:
        return np.minimum(s, S_WALL)
    if which == 1:
        with np.errstate(over='ignore'):
            return 1.0 / (1.0 + np.exp(-(s - S_C) / S_W))
    return (s >= S_WALL).astype(float)


def dg_values(s, which):
    """d g / d s (0 for lost particles)."""
    fin = np.isfinite(s)
    s = np.where(fin, s, np.inf)
    if which == 0:
        return np.where(s < S_WALL, 1.0, 0.0)
    with np.errstate(over='ignore'):
        l = 1.0 / (1.0 + np.exp(-(s - S_C) / S_W))
    return np.where(fin, l * (1.0 - l) / S_W, 0.0)


def load(outdir):
    return [dict(np.load(p)) for p in sorted(glob.glob(os.path.join(outdir, 'task_*.npz')))]


# ---- (a) -----------------------------------------------------------------------------------------------------------

def collect_resp(tasks):
    """{inu: {'nu', 'T', 'eps' [npoint], 's' [npoint, nT, N], 'xi' [..], 's_min' [npoint, N]}} with the blocks concatenated."""
    by = {}
    for t in tasks:
        by.setdefault((int(t['inu']), int(t['ipoint'])), []).append(t)
    out = {}
    for (inu, ipoint), ts in sorted(by.items()):
        ts = sorted(ts, key=lambda t: int(t['iblock']))
        d = out.setdefault(inu, {'nu': float(ts[0]['nu']), 'T': ts[0]['T'], 'eps': {}, 's': {}, 'xi': {}, 's_min': {}, 's_max': {}})
        d['eps'][ipoint] = float(ts[0]['eps_t'])
        for k in ('s', 'xi'):
            d[k][ipoint] = np.concatenate([t[k] for t in ts], axis=1)
        for k in ('s_min', 's_max'):
            d[k][ipoint] = np.concatenate([t[k] for t in ts])
    for d in out.values():
        pts = sorted(d['eps'])
        n = min(d['s'][i].shape[1] for i in pts)
        d['points'] = pts
        d['eps'] = np.array([d['eps'][i] for i in pts])
        for k in ('s', 'xi'):
            d[k] = np.stack([d[k][i][:, :n] for i in pts])            # [npoint, nT, N]
        for k in ('s_min', 's_max'):
            d[k] = np.stack([d[k][i][:n] for i in pts])
    return out


def gls_slope(G, e, deg, window=None):
    """G [N, npoint] values of g at the grid points e, paths in rows (paired across the grid). Fit of the differences
    G_k - G_c, c the centre, by sum_j beta_j x^j (x = (e_k - e_c) / h, j = 1..deg) with the covariance of the paired
    differences; returns slope at the centre (beta_1 / h), its error, chi^2, degrees of freedom."""
    npoint = G.shape[1]
    c = npoint // 2
    h = float(np.median(np.diff(e)))
    idx = [k for k in range(npoint) if k != c and (window is None or abs(k - c) <= window)]
    if len(idx) <= deg:
        return float('nan'), float('nan'), float('nan'), 0
    D = G[:, idx] - G[:, [c]]
    d = D.mean(axis=0)
    S = np.cov(D, rowvar=False).reshape(len(idx), len(idx)) / D.shape[0]
    x = (e[idx] - e[c]) / h
    X = np.vander(x, deg + 1, increasing=True)[:, 1:]
    try:
        Si = np.linalg.inv(S)
        cov = np.linalg.inv(X.T @ Si @ X)
    except np.linalg.LinAlgError:                              # e.g. nobody is lost: g3 is constant
        return float('nan'), float('nan'), float('nan'), 0
    beta = cov @ X.T @ Si @ d
    r = d - X @ beta
    return beta[0] / h, np.sqrt(cov[0, 0]) / h, float(r @ Si @ r), len(idx) - deg


def resp_report(resp):
    print('=== (a) response curves: slope d Phi_T / d eps_t at the centre of the grid ===')
    for which in (0, 1, 2):
        print(f'\n{NAMES[which]}')
        for it in range(len(next(iter(resp.values()))['T'])):
            T = next(iter(resp.values()))['T'][it]
            print(f'  T = {T:g}')
            print('    nu       N     Phi(centre) +- sem | slope: line +-1h   line +-2h   line +-3h   parabola all 11 pts   cubic all 11 pts   | chi2/dof of the fits (line, parabola, cubic; all 11)')
            for inu, d in sorted(resp.items()):
                s = d['s'][:, it, :]                                    # [npoint, N]
                G = g_values(s, which).T                                  # [N, npoint]
                e = d['eps']
                c = len(e) // 2
                row = []
                for deg, w in ((1, 1), (1, 2), (1, 3), (2, None), (3, None)):
                    sl, er, chi2, dof = gls_slope(G, e, deg, w)
                    row.append((sl, er, chi2, dof))
                fit = ' '.join(f'{r[0]:+7.3f}+-{r[1]:.3f}' for r in row[:3]) + f'   {row[3][0]:+7.3f}+-{row[3][1]:.3f}      {row[4][0]:+7.3f}+-{row[4][1]:.3f}'
                chis = gls_slope(G, e, 1)[2:], row[3][2:], row[4][2:]
                print(f"    {d['nu']:<7g} {G.shape[0]:5d}  {G[:, c].mean():.5f} +- {G[:, c].std(ddof=1) / np.sqrt(G.shape[0]):.5f} | {fit}   | "
                      + ', '.join(f'{x[0]:.1f}/{x[1]}' for x in chis))
    print('\nmean s_T and its extremes on the grid centre:')
    for inu, d in sorted(resp.items()):
        c = len(d['eps']) // 2
        print(f"  nu={d['nu']:g}: fraction s_T > S_C at T = "
              + ', '.join(f"{T:g}: {np.mean(d['s'][c][i] > S_C):.3f}" for i, T in enumerate(d['T']))
              + '; lost (s_T >= 1 or non-finite) at T = '
              + ', '.join(f"{T:g}: {np.mean(~(d['s'][c][i] < S_WALL)):.3f}" for i, T in enumerate(d['T']))
              + f"; ran past s=1 at some time: {np.mean(~(d['s_max'][c] < S_WALL)):.3f}"
              + f"; s_min over the run (finite paths): median {np.nanmedian(d['s_min'][c]):.4f}, 1% {np.nanquantile(d['s_min'][c], 0.01):.4f}, "
              + f"min {np.nanmin(d['s_min'][c]):.5f}; s_max: median {np.nanmedian(d['s_max'][c]):.3f}, 99% {np.nanquantile(d['s_max'][c], 0.99):.3f}")


# ---- (b) -----------------------------------------------------------------------------------------------------------

def hill(x, k):
    x = np.sort(np.abs(x))[::-1]
    return 1.0 / np.mean(np.log(x[:k] / x[k]))


def batch_exponent(x, sizes=(8, 32, 128, 512)):
    """IQR of the means of batches of L values against L: exponent a of IQR ~ L^-a (0.5 for a finite variance, 0 for a Cauchy law)."""
    n = len(x)
    sizes = [L for L in sizes if n // L >= 8]
    if len(sizes) < 3:
        return float('nan'), []
    iqr = []
    for L in sizes:
        nb = n // L
        m = x[:nb * L].reshape(nb, L).mean(axis=1)
        iqr.append(np.subtract(*np.quantile(m, [0.75, 0.25])))
    a = -np.polyfit(np.log(sizes), np.log(iqr), 1)[0]
    return a, iqr


def collect_crn(tasks):
    by = {}
    for t in tasks:
        by.setdefault(int(t['inu']), []).append(t)
    out = {}
    for inu, ts in sorted(by.items()):
        ts = sorted(ts, key=lambda t: int(t['iblock']))
        out[inu] = {'nu': float(ts[0]['nu']), 'T': ts[0]['T'], 'eps': float(ts[0]['eps_t'])}
        for k in ('s', 'xi', 'ds', 'dxi'):
            out[inu][k] = np.concatenate([t[k] for t in ts], axis=1)
        for k in ('s_min', 's_max'):
            out[inu][k] = np.concatenate([t[k] for t in ts])
    return out


def crn_report(crn, resp=None):
    print('\n=== (b) pathwise derivative with common random numbers, eps_t = centre ===')
    for which in (0, 1):
        print(f'\n{NAMES[which]}: D_i = g\'(s_T) d s_T / d eps_t')
        for it in range(len(next(iter(crn.values()))['T'])):
            T = next(iter(crn.values()))['T'][it]
            print(f'  T = {T:g}')
            print('    nu       N     mean +- sem        FD slope (parabola, 11 pts)  z   | robust sigma  std    max|D|/sigma  Hill(k=50/150)  top1% share  batch exp. a (IQR~L^-a)')
            for inu, d in sorted(crn.items()):
                dg = dg_values(d['s'][it], which)
                D = np.where(dg == 0.0, 0.0, dg * d['ds'][it])
                D = D[np.isfinite(D)]                                 # (non-finite tangent of a path still inside: dropped)
                n = len(D)
                m, sem = D.mean(), D.std(ddof=1) / np.sqrt(n)
                sig = 1.4826 * np.median(np.abs(D - np.median(D)))
                a = np.sort(np.abs(D))[::-1]
                share = a[:max(1, n // 100)].sum() / a.sum()
                fd = ''
                if resp is not None and inu in resp:
                    r = resp[inu]
                    sl, er, chi2, dof = gls_slope(g_values(r['s'][:, it, :], which).T, r['eps'], 2)
                    fd = f'{sl:+8.3f}+-{er:.3f}   {(m - sl) / np.hypot(sem, er):+5.1f}'
                else:
                    fd = '        n/a            '
                ex, _ = batch_exponent(D)
                print(f"    {d['nu']:<7g} {n:5d}  {m:+9.3f} +- {sem:7.3f}   {fd}  | {sig:9.3f} {D.std(ddof=1):9.2f} {np.abs(D).max() / max(sig, 1e-300):10.1f}  "
                      f"{hill(D, 50):.2f}/{hill(D, 150):.2f}     {share:.2f}        {ex:.2f}")
    print('\nscale of the estimator against T: quantiles of |D1|, D1 = d s_T / d eps_t (g1), and the growth rate of each quantile,')
    print('the slope of log(quantile) against T by least squares over all horizons (an exponential growth e^(gamma T) has gamma > 0 and a constant slope)')
    for inu, d in sorted(crn.items()):
        T = np.asarray(d['T'], float)
        rows = []
        for it in range(len(T)):
            inside = d['s'][it] < S_WALL                              # (False for NaN)
            x = d['ds'][it][inside]
            bad = int(np.sum(~np.isfinite(x)))
            x = np.abs(x[np.isfinite(x)])
            rows.append([np.median(x), np.quantile(x, 0.9), np.quantile(x, 0.99), np.quantile(x, 0.999), x.max(), x.std(ddof=1), bad])
        rows = np.array(rows)
        print(f"  nu = {d['nu']:g}: ({len(d['ds'][0])} paths; the paths that are inside s < 1 at the horizon; lost fraction "
              + ', '.join(f"{np.mean(~(d['s'][i] < S_WALL)):.3f}" for i in range(len(T))) + ')')
        print('       T      median      90%         99%         99.9%       max         std       non-finite')
        for it in range(len(T)):
            print(f"    {T[it]:7g} " + ' '.join(f'{v:11.3g}' for v in rows[it, :6]) + f'   {int(rows[it, 6])}')
        ok = rows[:, :5] > 0
        gam = [np.polyfit(T[ok[:, k]], np.log(rows[ok[:, k], k]), 1)[0] if ok[:, k].sum() >= 3 else float('nan') for k in range(5)]
        print('    growth rate gamma (per unit time): median %.4f, 90%% %.4f, 99%% %.4f, 99.9%% %.4f, max %.4f' % tuple(gam))


def main():
    rdir = sys.argv[1] if len(sys.argv) > 1 else 'results/gc_collision_resp'
    cdir = sys.argv[2] if len(sys.argv) > 2 else 'results/gc_collision_crn'
    resp = collect_resp(load(rdir)) if os.path.isdir(rdir) and glob.glob(os.path.join(rdir, 'task_*.npz')) else None
    if resp:
        resp_report(resp)
    if os.path.isdir(cdir) and glob.glob(os.path.join(cdir, 'task_*.npz')):
        crn_report(collect_crn(load(cdir)), resp)


if __name__ == '__main__':
    main()
