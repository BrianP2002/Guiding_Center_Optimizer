"""Summary of experiments/scripts/gc_official_candidates.py: ergodicity of the chaotic draws of the official guiding-center flow.

    python experiments/scripts/summarize_gc_official_candidates.py [results/gc_official_candidates]
"""
import glob
import json
import os
import sys

import numpy as np

S_MAX, E_TOL = 1.0, 1e-6


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else 'results/gc_official_candidates'
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(outdir, 'task_*.json')))]
    print(f'{len(rows)} runs from {outdir}')
    for seed in sorted({r['seed'] for r in rows}):
        sub = sorted([r for r in rows if r['seed'] == seed], key=lambda r: r['k'])
        p = sub[0]['params']
        print(f"\nseed {seed}: kappa={p['kappa']:.3f} N={p['N']:.0f} eps_h={p['eps_h']:.3f} eps_t={p['eps_t']:.3f} eps_m={p['eps_m']:.3f} "
              f"eps_2={p['eps_2']:.3f}(N2={p['N2']:.0f}) iota={p['iota']:.2f} lam={p['lam']:.3f}   T={sub[0]['T']:g}")
        chaotic = []
        for r in sub:
            ok = r['finite'] and r['x_max'] <= S_MAX and r['x_min'] > 0 and abs(r['dE']) < E_TOL
            lam = r['lyap'][0]
            lam_half = r['lyap_finite_time'][len(r['lyap_finite_time']) // 2 - 1][0]
            cls = 'escaped' if not ok else ('chaotic' if lam * r['T'] > 20 and lam > 0.7 * lam_half else 'regular')
            xm = np.array(r['x_mean_parts'])
            half = len(xm) // 2
            print(f"  k={r['k']:2d} {cls:8s} lam1={lam:8.5f} lam1*T={lam * r['T']:7.1f} <s>={xm.mean():.4f}  "
                  f"(1st half {xm[:half].mean():.4f}, 2nd half {xm[half:].mean():.4f}, part-to-part std {xm.std(ddof=1):.4f}) "
                  f"s in [{r['x_min']:.3f},{r['x_max']:.3f}] v_par sign change={r['v_min'] < 0 < r['v_max']}")
            if cls == 'chaotic':
                chaotic.append((xm.mean(), xm.std(ddof=1) / np.sqrt(len(xm))))
        if len(chaotic) >= 2:
            m = np.array([c[0] for c in chaotic])
            sem = np.array([c[1] for c in chaotic])
            print(f'  chaotic initial conditions: {len(chaotic)}; <s> across them: mean {m.mean():.4f}, std {m.std(ddof=1):.4f}; '
                  f'typical within-run standard error {sem.mean():.4f}  ->  spread / standard error = {m.std(ddof=1) / sem.mean():.1f}'
                  f' (about 1 for an ergodic set)')


if __name__ == '__main__':
    main()
