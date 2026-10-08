"""Summary of experiments/scripts/gc_official_scan.py: where is the official guiding-center flow chaotic and bounded?

    python experiments/scripts/summarize_gc_official_scan.py [results/gc_official_scan]
"""
import glob
import json
import os
import sys

import numpy as np

S_MAX = 1.0                 # the near-axis field is only meaningful inside s < 1 (and s > 0 for sqrt(s))
RESOLVED = 20.0             # lambda_1 * T above this ... (a regular flow gives lambda_1 * T ~ ln T ~ 8 at T = 4000)
E_TOL = 1e-6                # energy drift above this: the integration is not trusted


def load(outdir):
    rows = []
    for path in sorted(glob.glob(os.path.join(outdir, 'task_*.jsonl'))):
        with open(path) as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    return rows


def classify(r):
    if not r['finite'] or r['x_max'] > S_MAX or r['x_min'] <= 0.0:
        return 'escaped'
    if not abs(r['dE']) < E_TOL:
        return 'inaccurate'
    lam1T = r['lyap'][0] * r['T']
    lam_half = r['lyap_finite_time'][1][0]           # estimate over the first half of the run
    # ... and the estimate must not decay like ln(t)/t, as it does for regular flows
    return 'chaotic' if lam1T > RESOLVED and r['lyap'][0] > 0.7 * lam_half else 'regular'


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else 'results/gc_official_scan'
    rows = load(outdir)
    for r in rows:
        r['cls'] = classify(r)
        r['trapped'] = r['v_min'] < 0.0 < r['v_max']
    print(f'{len(rows)} draws from {outdir}')
    for broken in (False, True):
        sub = [r for r in rows if r['broken'] == broken]
        if not sub:
            continue
        print(f"\n{'symmetry broken' if broken else 'exact quasi-helical field (control)'}: {len(sub)} draws")
        for cls in ('regular', 'chaotic', 'escaped', 'inaccurate'):
            n = sum(r['cls'] == cls for r in sub)
            print(f'  {cls:10s} {n:4d}  ({100 * n / len(sub):4.1f} %)')
        for trapped in (False, True):
            ss = [r for r in sub if r['trapped'] == trapped and r['cls'] in ('regular', 'chaotic')]
            if ss:
                n = sum(r['cls'] == 'chaotic' for r in ss)
                print(f"  bounded {'trapped' if trapped else 'passing':7s}: {n}/{len(ss)} chaotic")

    ctrl = [r for r in rows if not r['broken'] and r['cls'] in ('regular', 'chaotic')]
    if ctrl:
        print('\ncontrol: max lambda_1 * T =', max(r['lyap'][0] * r['T'] for r in ctrl), '(integrable: only numerical noise expected)')

    chaotic = sorted([r for r in rows if r['cls'] == 'chaotic'], key=lambda r: -r['lyap'][0] * r['T'])
    print(f'\nbounded chaotic draws: {len(chaotic)}')
    for r in chaotic[:15]:
        p = r['params']
        print(f"  seed={r['seed']:4d} lam1={r['lyap'][0]:.4f} lam1*T={r['lyap'][0] * r['T']:.1f} kappa={p['kappa']:.3f} "
              f"N={p['N']:.0f} eps_h={p['eps_h']:.3f} eps_t={p['eps_t']:.3f} eps_m={p['eps_m']:.3f} eps_2={p['eps_2']:.3f}(N2={p['N2']:.0f}) "
              f"iota={p['iota']:.2f} lam={p['lam']:.3f} trapped={r['trapped']} <s> parts={np.round(r['x_mean_parts'], 3)} "
              f"lam1(t)={np.round([c[0] for c in r['lyap_finite_time']], 3)}")


if __name__ == '__main__':
    main()
