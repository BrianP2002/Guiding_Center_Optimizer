"""Summary of experiments/scripts/gc_official_nilss.py: NILSS sensitivities in the chaotic sea of the official guiding-center flow,
against the finite-difference ground truth of experiments/scripts/gc_official_fd.py (sea copies only, slope at the centre of the
grid; summarize_gc_official_fd.py) wherever a finite-difference scan of that parameter exists.

    python experiments/scripts/summarize_gc_official_nilss.py [results/gc_official_nilss] [--sea-lambda 0.003]
        [--fd PAR=DIR[,DIR...] ...] [--fd-sea-abs 0.01 | --fd-sea-rel R] [--fd-min-copies 10]
        [--lambda-scan 0.0003,0.001,0.002] [--table-par NAME]

Without --fd the finite-difference scans of draw 581 are used (FD_DIRS). A run is a "sea" run when its largest Lyapunov
exponent is at least --sea-lambda (draw 581: 0.003, regular tori have 1e-4 .. 3e-3; draw 835: 1e-3, regular tori have 5e-5 and the
sea 2.6e-3 .. 3.3e-3); --lambda-scan shows how much the result depends on that choice.
"""
import argparse
import glob
import json
import os

import numpy as np

from summarize_gc_official_fd import load, table, window_slopes, SEA_PART_STD, MIN_COPIES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEA_LAMBDA = 0.003        # a run whose largest exponent is smaller is on a regular torus, not in the sea
FD_DIRS = {'eps_t': ['results/gc_official_fd', 'results/gc_official_fd2_eps_t'], 'eps_m': ['results/gc_official_fd2_eps_m'],
           'kappa': ['results/gc_official_fd2_kappa'], 'iota': ['results/gc_official_fd2_iota']}
MAIN_CFG = (1, 200.0)


def mean_sem(x):
    x = np.asarray(x, dtype=float)
    return x.mean(), (x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float('nan'))


def median_ci(x, nboot=4000, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.RandomState(seed)
    med = np.median(x[rng.randint(0, len(x), (nboot, len(x)))], axis=1)
    return np.median(x), np.quantile(med, 0.025), np.quantile(med, 0.975)


def hill(x, k):
    """Hill estimator of the tail index from the k largest values (P(X > x) ~ x^-index)."""
    x = np.sort(np.asarray(x))[::-1]
    return 1.0 / np.mean(np.log(x[:k] / x[k]))


def diagnostics(rows, pars, sea_lambda):
    """The shadowing direction along the orbits (needs the 'vnorm' and 'segments' fields): is it bounded, how heavy are its tails,
    how fast does the estimate converge with the length of the run."""
    sub = [r for r in rows if (r['nus'], r['T_seg']) == MAIN_CFG and r['lyap'][0] >= sea_lambda and 'vnorm' in r]
    if not sub:
        return
    V, S = np.array([r['vnorm'] for r in sub]), np.array([r['segments'] for r in sub])          # [run, nseg, npar]
    nseg = S.shape[1]
    print(f'\nshadowing direction along the {len(sub)} sea runs of the main setting ({nseg} segments each):')
    for i, par in enumerate(pars):
        v, a = V[:, :, i], np.abs(S[:, :, i]).ravel()
        top = np.sort(a)[::-1]
        print(f'  {par}: |v_perp| median {np.median(v):.2f}, 90% {np.quantile(v, .9):.1f}, 99% {np.quantile(v, .99):.0f}, '
              f'99.9% {np.quantile(v, .999):.0f}, max {v.max():.0f};  median by quarter of the run '
              f'{[round(float(np.median(v[:, q])), 2) for q in np.array_split(np.arange(nseg), 4)]}')
        print(f'       tail index (Hill, k = 100 / 1000) of |v_perp|: {hill(v.ravel(), 100):.2f} / {hill(v.ravel(), 1000):.2f}; '
              f'of the per-segment contributions: {hill(a, 100):.2f} / {hill(a, 1000):.2f};  top 1% of the segments carry '
              f'{top[:len(top) // 100].sum() / top.sum():.2f} of the sum of |contributions|')
        d = S[:, :, i].sum(axis=1)
        print(f'       corr(|run estimate - median|, max |v_perp| of the run) = {np.corrcoef(np.abs(d - np.median(d)), v.max(axis=1))[0, 1]:.2f}')
        line, iqr = [], {}
        for L in (1, 10, 100, 1000):
            nb = nseg // L
            est = (S[:, :nb * L, i].reshape(len(sub), nb, L).sum(axis=2) * (nseg / L)).ravel()
            q1, q2, q3 = np.quantile(est, [.25, .5, .75])
            iqr[L] = q3 - q1
            line.append(f'L={L}: median {q2:+.2f}, IQR {q3 - q1:.2f}')
        print('       estimate from sub-runs of L segments:  ' + ';  '.join(line) +
              f';  IQR(L=1) / IQR(L=1000) = {iqr[1] / iqr[1000]:.1f} (31.6 if the variance were finite)')


def fd_reference(par, fd_dirs, sea_abs, sea_rel, min_copies):
    """([(label, slope, error)], [(dir, mean, sem, n)] at the centre) from the finite-difference scans of this parameter."""
    ref, centre = [], []
    for d in (os.path.join(ROOT, d) for d in fd_dirs.get(par, [])):
        if not os.path.isdir(d) or not glob.glob(os.path.join(d, 'task_*.json')):
            continue
        rows = load(d)
        values = sorted({r['value'] for r in rows})
        h = float(np.median(np.diff(values)))
        c = values[len(values) // 2]
        tab = table(rows, True, sea_abs, sea_rel, min_copies)
        for k, n, slope, err, chi2, dof in window_slopes(tab, c, h):
            ref.append((f'{os.path.basename(d)} window +-{k}h ({n} pts)', slope, err))
        for k, n, slope, err, chi2, dof in window_slopes(tab, c, h, quadratic=True):
            ref.append((f'{os.path.basename(d)} parabola +-{k}h ({n} pts)', slope, err))
        t0 = [t for t in tab if abs(t['value'] - c) < 1e-9]
        if t0:
            centre.append((os.path.basename(d), t0[0]['mean'], t0[0]['sem'], t0[0]['n']))
    return ref, centre


def parse_fd(items):
    out = {}
    for item in items:
        par, dirs = item.split('=', 1)
        out[par] = dirs.split(',')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir', nargs='?', default='results/gc_official_nilss')
    ap.add_argument('--sea-lambda', type=float, default=SEA_LAMBDA)
    ap.add_argument('--fd', action='append', default=[])
    ap.add_argument('--fd-sea-abs', type=float, default=SEA_PART_STD)
    ap.add_argument('--fd-sea-rel', type=float, default=None)
    ap.add_argument('--fd-min-copies', type=int, default=MIN_COPIES)
    ap.add_argument('--lambda-scan', default=None)
    ap.add_argument('--table-par', default=None)
    args = ap.parse_args()
    fd_dirs = parse_fd(args.fd) if args.fd else FD_DIRS
    sea_lambda = args.sea_lambda

    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(args.outdir, 'task_*.json')))]
    pars = rows[0]['pars']
    lam = np.array([r['lyap'][0] for r in rows])
    print(f'{len(rows)} runs from {args.outdir}; T = {rows[0]["T"]:g}; parameters {pars}')
    print(f'largest exponent of the runs: min {lam.min():.5f}, median {np.median(lam):.5f}, max {lam.max():.5f}; '
          f'{int(np.sum(lam < sea_lambda))} runs below {sea_lambda} (regular torus, dropped below)\n')

    for key in sorted({(r['nus'], r['T_seg']) for r in rows}):
        sub = [r for r in rows if (r['nus'], r['T_seg']) == key and r['lyap'][0] >= sea_lambda]
        D = np.array([r['dJdp'] for r in sub])
        print(f'nus = {key[0]}, T_seg = {key[1]:g}: {len(sub)} sea runs, <s> = {mean_sem([r["Javg"] for r in sub])[0]:.4f}')
        for i, par in enumerate(pars):
            m, se = mean_sem(D[:, i])
            med, lo, hi = median_ci(D[:, i])
            print(f'  dJ/d{par:6s}: mean {m:+8.3f} +- {se:6.3f}   median {med:+8.3f} [{lo:+.3f}, {hi:+.3f}]   '
                  f'10-90%: [{np.quantile(D[:, i], 0.1):+.2f}, {np.quantile(D[:, i], 0.9):+.2f}]   fraction > 0: {np.mean(D[:, i] > 0):.2f}')

    sub = [r for r in rows if (r['nus'], r['T_seg']) == MAIN_CFG and r['lyap'][0] >= sea_lambda]
    D = np.array([r['dJdp'] for r in sub])
    print('\nground truth (finite differences over sea orbits, slope in windows around the base value):')
    jm, jse = mean_sem([r['Javg'] for r in sub])
    for i, par in enumerate(pars):
        m, se = mean_sem(D[:, i])
        med = np.median(D[:, i])
        print(f'  {par}: NILSS mean {m:+.3f} +- {se:.3f}, median {med:+.3f}')
        ref, centre = fd_reference(par, fd_dirs, args.fd_sea_abs, args.fd_sea_rel, args.fd_min_copies)
        for label, slope, err in ref:
            print(f'      FD {label}: {slope:+.3f} +- {err:.3f}   (NILSS mean - FD) / sqrt(err^2) = {(m - slope) / np.hypot(se, err):+.1f}')
        if i == 0:
            for name, cm, cse, n in centre:
                print(f'      (NILSS <s> = {jm:.4f} +- {jse:.4f} against the FD ensemble at the base value {cm:.4f} +- {cse:.4f}, {name}, {n} copies)')

    if args.lambda_scan:
        print('\nsensitivity of the NILSS result (main setting) to the sea threshold on lambda_1:')
        main_runs = [r for r in rows if (r['nus'], r['T_seg']) == MAIN_CFG]
        for th in (float(x) for x in args.lambda_scan.split(',')):
            s2 = [r for r in main_runs if r['lyap'][0] >= th]
            D2 = np.array([r['dJdp'] for r in s2])
            print(f'  lambda_1 >= {th:g}: {len(s2)} of {len(main_runs)} runs;  ' +
                  ';  '.join(f'{par} mean {D2[:, i].mean():+.2f} median {np.median(D2[:, i]):+.2f}' for i, par in enumerate(pars)))

    diagnostics(rows, pars, sea_lambda)

    tpar = pars.index(args.table_par) if args.table_par else 0
    tab = {}
    for r in rows:
        tab.setdefault(r['copy'], {})[(r['nus'], r['T_seg'])] = r['dJdp'][tpar]
    keys = [MAIN_CFG] + [k for k in sorted({(r['nus'], r['T_seg']) for r in rows}) if k != MAIN_CFG]
    copies = [c for c in sorted(tab) if len(tab[c]) > 1]
    if copies and len(keys) > 1:
        print(f'\nsame orbit, different settings (d J / d {pars[tpar]}):')
        print('  copy  ' + ' '.join(f'nus{k[0]}/Tseg{k[1]:<5g}' for k in keys))
        for c in copies:
            print(f'  {c:4d}  ' + ' '.join(f'{tab[c][k]:13.3f}' if k in tab[c] else f'{"-":>13s}' for k in keys))
        for k in keys[1:]:
            both = [c for c in copies if k in tab[c] and MAIN_CFG in tab[c]]
            if both:
                diff = np.array([tab[c][k] - tab[c][MAIN_CFG] for c in both])
                print(f'  {k}: largest |difference from nus1/Tseg200| = {np.max(np.abs(diff)):.3f} over {len(both)} orbits; '
                      f'median |difference| = {np.median(np.abs(diff)):.3f}')


if __name__ == '__main__':
    main()
