"""Summary of experiments/scripts/gc_official_fd.py: the long-time <s> in the chaotic sea as a function of a parameter, and its slope
(the ground truth for experiments/scripts/gc_official_nilss.py).

A copy is called a "sea" orbit when <s> varies from one twentieth of the run to the next (standard deviation of the 20 parts above
a threshold); a copy on a regular torus has the same <s> in every part (standard deviation ~1e-4 .. 1e-3) and is left out of the sea
average, because the sea average is what a shadowing method is supposed to differentiate. The threshold is absolute (SEA_PART_STD =
0.01, tuned for the <s> ~ 0.1 of draw 581) or, with --sea-rel R, relative to the <s> of the copy (the scale of <s> differs between
draws). The "all copies" columns keep the regular ones.

    python experiments/scripts/summarize_gc_official_fd.py [results/gc_official_fd] [--sea-abs 0.01 | --sea-rel R] [--min-copies 10]
        [--scan-sea-rel 0.01,0.02,0.05,0.1]     # slope at the centre for each threshold, to show how much it matters
"""
import argparse
import glob
import json
import os

import numpy as np

S_MAX = 1.0
SEA_PART_STD = 0.01
MIN_COPIES = 10           # parameter values with fewer copies are left out


def load(outdir):
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(outdir, 'task_*.json')))]


def is_sea(r, sea_abs=SEA_PART_STD, sea_rel=None):
    """<s> varies between the 20 parts of the run: standard deviation of the parts above sea_abs, or above sea_rel * <s>."""
    return np.std(r['parts'], ddof=1) > (sea_abs if sea_rel is None else sea_rel * r['mean_s'])


def table(rows, sea_only, sea_abs=SEA_PART_STD, sea_rel=None, min_copies=MIN_COPIES):
    """Per parameter value: mean of <s> over the copies (+- standard error of the copies), number of copies, median s_max."""
    values = sorted({r['value'] for r in rows})
    out = []
    for v in values:
        sub = [r for r in rows if r['value'] == v and r['finite'] and r['x_max'] <= S_MAX and r['x_min'] > 0 and abs(r['dE']) < 1e-6]
        if sea_only:
            sub = [r for r in sub if is_sea(r, sea_abs, sea_rel)]
        if len(sub) < min_copies:
            continue
        m = np.array([r['mean_s'] for r in sub])
        out.append({'value': v, 'n': len(m), 'mean': m.mean(), 'sem': m.std(ddof=1) / np.sqrt(len(m)), 'std': m.std(ddof=1),
                    'smax': np.median([r['x_max'] for r in sub]), 'smin': np.median([r['x_min'] for r in sub]),
                    'within': float(np.mean([np.std(r['parts'], ddof=1) / np.sqrt(len(r['parts'])) for r in sub]))})
    return out


def line_slope(v, m, se):
    """Weighted straight-line fit; slope, its error, chi2 and degrees of freedom."""
    w = 1.0 / se ** 2
    X = np.vstack([np.ones_like(v), v - v.mean()]).T
    cov = np.linalg.inv(X.T @ (X * w[:, None]))
    b = cov @ (X.T @ (w * m))
    return b[1], np.sqrt(cov[1, 1]), float(np.sum(w * (m - X @ b) ** 2)), len(v) - 2


def quad_slope(v, m, se, center):
    """Derivative at `center` of the weighted parabola through the points; value, error, chi2, dof."""
    w = 1.0 / se ** 2
    X = np.vstack([np.ones_like(v), v - center, (v - center) ** 2]).T
    cov = np.linalg.inv(X.T @ (X * w[:, None]))
    b = cov @ (X.T @ (w * m))
    return b[1], np.sqrt(cov[1, 1]), float(np.sum(w * (m - X @ b) ** 2)), len(v) - 3


def window_slopes(tab, center, h, quadratic=False):
    """Slope at `center` from the points within k*h of it, k = 1 .. (all points): straight line, or parabola (>= 5 points)."""
    v = np.array([t['value'] for t in tab]); m = np.array([t['mean'] for t in tab]); se = np.array([t['sem'] for t in tab])
    res = []
    for k in range(1, int(round(np.max(np.abs(v - center)) / h)) + 1):
        sel = np.abs(v - center) <= k * h * (1 + 1e-9)
        if quadratic and sel.sum() >= 5:
            res.append((k, int(sel.sum())) + quad_slope(v[sel], m[sel], se[sel], center))
        elif not quadratic and sel.sum() >= 3:
            res.append((k, int(sel.sum())) + line_slope(v[sel], m[sel], se[sel]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir', nargs='?', default='results/gc_official_fd')
    ap.add_argument('--sea-abs', type=float, default=SEA_PART_STD)
    ap.add_argument('--sea-rel', type=float, default=None)
    ap.add_argument('--min-copies', type=int, default=MIN_COPIES)
    ap.add_argument('--scan-sea-rel', default=None)
    args = ap.parse_args()
    rows = load(args.outdir)
    par = rows[0]['par']
    values = sorted({r['value'] for r in rows})
    h = float(np.median(np.diff(values)))
    center = values[len(values) // 2]
    print(f'{len(rows)} runs from {args.outdir}: parameter {par}, T = {rows[0]["T"]:g}, dt = {rows[0]["dt"]}, '
          f'{len(values)} values with spacing {h:g}, centre {center:.5f}')
    crit = f'part-to-part std of <s> > {args.sea_rel} * <s>' if args.sea_rel is not None else f'part-to-part std of <s> > {args.sea_abs}'
    for label, sea_only in (('sea copies only', True), ('all bounded copies', False)):
        tab = table(rows, sea_only, args.sea_abs, args.sea_rel, args.min_copies)
        print(f'\n{label}' + (f' ({crit}):' if sea_only else ':'))
        for t in tab:
            print(f"  {par} = {t['value']:.5f}: n = {t['n']:2d}  <s> = {t['mean']:.5f} +- {t['sem']:.5f}  (copy std {t['std']:.4f}, "
                  f"within-run SE {t['within']:.4f})  median s_max {t['smax']:.3f}  s_min {t['smin']:.4f}")
        print(f'  slope d<s>/d{par} at the centre from the points within k*h (straight line):')
        for k, n, slope, err, chi2, dof in window_slopes(tab, center, h):
            print(f'    k = {k}: {n:2d} points  {slope:+8.3f} +- {err:.3f}   chi2/dof = {chi2:6.1f}/{dof}')
        print('  slope at the centre from a parabola through the points within k*h:')
        for k, n, slope, err, chi2, dof in window_slopes(tab, center, h, quadratic=True):
            print(f'    k = {k}: {n:2d} points  {slope:+8.3f} +- {err:.3f}   chi2/dof = {chi2:6.1f}/{dof}')
    if args.scan_sea_rel:
        print('\nsensitivity to the sea threshold (relative to <s>): copies per value (min..max) and slope at the centre')
        for r in (float(x) for x in args.scan_sea_rel.split(',')):
            tab = table(rows, True, sea_rel=r, min_copies=args.min_copies)
            if len(tab) < 3:
                print(f'  sea-rel {r:g}: fewer than 3 usable parameter values')
                continue
            ns = [t['n'] for t in tab]
            parts = [f'k={k}: {slope:+.2f} +- {err:.2f}' for k, n, slope, err, chi2, dof in window_slopes(tab, center, h)]
            quad = [f'k={k}: {slope:+.2f} +- {err:.2f}' for k, n, slope, err, chi2, dof in window_slopes(tab, center, h, quadratic=True)]
            print(f'  sea-rel {r:g}: n = {min(ns)}..{max(ns)} over {len(tab)} values;  line  ' + ', '.join(parts) +
                  ('  | parabola ' + ', '.join(quad) if quad else ''))


if __name__ == '__main__':
    main()
