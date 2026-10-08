"""Figures of docs/findings.md and docs/reliability.md.

Reads the stored results (``results/``) and the text summaries in ``experiments/reference_results/`` and writes
``docs/figures/*.png`` (150 dpi, light surface, solid hairline grid, colours from the validated categorical order
blue / orange / aqua; ordered quantities (the collision frequency) use one blue ramp).

    python experiments/scripts/make_figures.py [--outdir docs/figures]

Run it from the repository root. Figures whose input is missing are skipped with a message.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nilss_jax import load_ensemble
from nilss_jax.systems import guiding_center

# ---- style ----------------------------------------------------------------------------------------------------------------
SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
RAMP = {'250': '#86b6ef', '400': '#3987e5', '550': '#1c5cab'}      # sequential blue steps (light -> dark)
LW, MS = 1.5, 6.5                                                   # 2 px lines, >= 8 px markers at 150 dpi


def style():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'axes.titlecolor': INK, 'text.color': INK,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'grid.linestyle': '-',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.axisbelow': True,
        'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5, 'legend.fontsize': 8, 'legend.frameon': False,
        'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round'})


def save(fig, path):
    fig.savefig(path, dpi=150, pil_kwargs={'optimize': True})
    plt.close(fig)
    print(f'wrote {path} ({os.path.getsize(path) / 1e3:.0f} KB)')


def marker(ax, x, y, color, shape='o', **kw):
    """A marker with the 2 px surface ring."""
    return ax.plot(x, y, shape, color=color, ms=MS, mec=SURFACE, mew=1.2, ls='none', zorder=3, **kw)


# ---- figure 1: NILSS against finite differences ------------------------------------------------------------------------------

def fig_nilss_vs_fd(outdir):
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 4.4))
    for row, draw in enumerate((581, 835)):
        d = guiding_center.load_draw(draw)
        pars = [k for k in d['reference_nilss'] if k != 'note']
        for col, par in enumerate(pars):
            ax = axes[row, col]
            n, f = d['reference_nilss'][par], d['reference_fd'][par]
            ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
            ax.errorbar([0], [n['value']], yerr=[n['err']], fmt='none', ecolor=BLUE, elinewidth=LW, capsize=3, zorder=2)
            ax.errorbar([1], [f['value']], yerr=[f['err']], fmt='none', ecolor=ORANGE, elinewidth=LW, capsize=3, zorder=2)
            marker(ax, [0], [n['value']], BLUE, 'o')
            marker(ax, [1], [f['value']], ORANGE, 's')
            lo = min(n['value'] - n['err'], f['value'] - f['err'], 0.0)
            hi = max(n['value'] + n['err'], f['value'] + f['err'], 0.0)
            pad = 0.25 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(-1.0, 2.0)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['NILSS', 'finite\ndiff.'], color=INK2)
            ax.set_title(f'd⟨s⟩/d{par.replace("eps_", "ε_").replace("kappa", "κ").replace("iota", "ι")}', loc='left')
            ax.grid(axis='x', visible=False)
            ax.annotate(f'{n["value"]:+.2f}', (0, n['value']), textcoords='offset points', xytext=(-9, -3), ha='right', color=INK2, fontsize=7.5)
            ax.annotate(f'{f["value"]:+.2f}', (1, f['value']), textcoords='offset points', xytext=(9, -3), ha='left', color=INK2, fontsize=7.5)
        axes[row, 0].set_ylabel(f'draw {draw}', color=INK, fontweight='semibold')
    fig.text(0.5, 0.004, 'NILSS: mean ± SEM of 35 (draw 581) or 48 (draw 835) runs of T = 2×10⁵ on the chaotic sea.\n'
             'Finite differences: sea orbits, slope ± error (draw 581, ε_m: +1.8 … +3.5 in the studies, one value shown).', ha='center', va='bottom',
             color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, os.path.join(outdir, 'nilss_vs_fd.png'))


# ---- figure 2: tails of the shadowing direction ---------------------------------------------------------------------------------

def _pool_json(directory, par, nus=1, lyap_min=0.0):
    out = []
    for p in sorted(glob.glob(os.path.join(directory, 'task_*.json'))):
        r = json.load(open(p))
        if r['nus'] == nus and r['T_seg'] == 200.0 and r['lyap'][0] >= lyap_min and 'vnorm' in r:
            out.append(np.array(r['vnorm'])[:, r['pars'].index(par)])
    return np.concatenate(out) if out else None


def _survival(x, n=60):
    x = np.sort(np.asarray(x) / np.median(x))
    grid = np.logspace(0, np.log10(x[-1]), n)
    return grid, 1.0 - np.searchsorted(x, grid, side='right') / len(x)


def fig_tails(outdir):
    series = []
    v = _pool_json('results/gc_official_nilss', 'eps_t', lyap_min=0.003)
    if v is not None:
        series.append(('GC draw 581, ε_t', v, BLUE, 'o'))
    v = _pool_json('results/gc_official_nilss_seed835', 'eps_2')
    if v is not None:
        series.append(('GC draw 835, ε_2', v, ORANGE, 's'))
    cache = 'results/calibration/lorenz63_rho28_T1000'
    if os.path.isdir(cache):
        ens = load_ensemble(cache)
        series.append(('Lorenz 63, ρ = 28', np.concatenate([r.vnorm[:, 0] for r in ens.results]), AQUA, 'D'))
    if not series:
        print('figure tails: no input found, skipped')
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    for label, v, c, m in series:
        x, s = _survival(v)
        keep = s > 0
        ax.plot(x[keep], s[keep], '-', color=c, lw=LW, label=label)
        marker(ax, x[keep][::6], s[keep][::6], c, m)
    xs = np.array([3.0, 1e5])
    ax.plot(xs, 0.5 * 3.0 / xs * 1.0, '-', color=INK2, lw=0.9, zorder=2)
    ax.annotate('∝ 1/x  (tail index 1)', (4e3, 0.5 * 3.0 / 4e3), textcoords='offset points', xytext=(6, 4), color=INK2, fontsize=7.5)
    ax.set_xscale('log'), ax.set_yscale('log')
    ax.set_xlim(0.8, 3e6)
    ax.set_ylim(1e-6, 1.5)
    ax.set_xlabel('x = |v⊥| / median |v⊥|  (norm of the shadowing direction at segment ends)')
    ax.set_ylabel('P(|v⊥| / median > x)')
    ax.set_title('Heavy tail of the shadowing direction on the chaotic seas', loc='left')
    ax.legend(loc='lower left')
    save(fig, os.path.join(outdir, 'shadowing_tails.png'))


# ---- figure 3: pathwise gradient against horizon ----------------------------------------------------------------------------------

def parse_pathwise(path):
    """E1, g = s, parameter eps_t: per horizon (mean, SEM, Hill index of |D| at k = 100, Hill index of |v|); and the population finite differences."""
    lines = open(path).read().split('\n')
    i0 = next(i for i, l in enumerate(lines) if 'ensemble E1' in l)
    i1 = next(i for i in range(i0, len(lines)) if 'observable g1=s' in lines[i])
    table = {}
    for l in lines[i1 + 1:]:
        if l.startswith('--- observable'):
            break
        m = re.match(r'\s+eps_t\s+(\d+)\s+(\S+)\s+\+-\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$', l)
        if m:
            T = float(m.group(1))
            table[T] = {'mean': float(m.group(2)), 'sem': float(m.group(3)), 'idxD100': float(m.group(7)), 'idxv': float(m.group(9))}
    pop = {}
    j0 = next(i for i, l in enumerate(lines) if l.startswith('===== population finite differences') and 'eps_t' in l)
    cur = None
    for l in lines[j0 + 1:]:
        if l.startswith('====='):
            break
        mh = re.match(r'\s+h=(\S+) (g\d)=', l)
        if mh:
            cur = (float(mh.group(1)), mh.group(2))
            continue
        mt = re.match(r'\s+T=\s*(\d+):\s+(\S+)\s+\+-\s+(\S+)\s+\(N=', l)
        if mt and cur is not None and cur[1] == 'g1':
            pop.setdefault(cur[0], {})[float(mt.group(1))] = (float(mt.group(2)), float(mt.group(3)))
    return table, pop


def fig_pathwise(outdir):
    path = 'experiments/reference_results/pathwise_summary.txt'
    if not os.path.exists(path):
        print('figure pathwise: missing', path)
        return
    table, pop = parse_pathwise(path)
    Ts = sorted(table)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw={'width_ratios': [1.15, 1]})
    # (a) pathwise mean +- SEM and population finite differences, one symmetric-log axis (linear within +-10)
    near = [T for T in Ts if T <= 3000]
    a.axhline(0, color=AXIS, lw=0.8)
    a.errorbar(np.array(near) * 0.93, [table[T]['mean'] for T in near], yerr=[table[T]['sem'] for T in near], fmt='none', ecolor=BLUE,
               elinewidth=LW, capsize=3, zorder=2)
    marker(a, np.array(near) * 0.93, [table[T]['mean'] for T in near], BLUE, 'o', label='pathwise (forward tangent), 2400 orbits')
    hs = sorted(pop)
    for h, c, m, lab, off in zip(hs, (ORANGE, AQUA), ('s', 'D'), ('population finite differences, step 0.1 |ε_t|', 'population finite differences, step 0.3 |ε_t|'),
                                 (1.0, 1.07)):
        T2 = sorted(pop[h])
        a.errorbar(np.array(T2) * off, [pop[h][T][0] for T in T2], yerr=[pop[h][T][1] for T in T2], fmt='none', ecolor=c, elinewidth=LW, capsize=3, zorder=2)
        marker(a, np.array(T2) * off, [pop[h][T][0] for T in T2], c, m, label=lab)
    a.set_xscale('log')
    a.set_yscale('symlog', linthresh=10, linscale=3.0)
    a.set_xlim(70, 4500)
    a.set_ylim(-400, 5e10)
    a.set_yticks([-100, -10, 0, 10, 1e3, 1e6, 1e9])
    a.set_yticklabels(['−100', '−10', '0', '10', '10³', '10⁶', '10⁹'])
    a.set_xlabel('horizon T  (λ₁T = 0.006 T; one bounce ≈ 34)')
    a.set_ylabel('∂E[s_T]/∂ε_t   (symmetric log axis)')
    a.set_title('Gradient of the finite-time ensemble average', loc='left')
    a.legend(loc='upper left', fontsize=7)
    # (b) tail index versus T
    b.axhline(2, color=INK2, lw=0.9)
    b.annotate('2: infinite variance below', (80, 2.0), textcoords='offset points', xytext=(0, 4), color=INK2, fontsize=7.5)
    b.axhline(1, color=INK2, lw=0.9)
    b.annotate('1: infinite mean below', (80, 1.0), textcoords='offset points', xytext=(0, -11), color=INK2, fontsize=7.5)
    b.plot(Ts, [table[T]['idxD100'] for T in Ts], '-', color=BLUE, lw=LW)
    marker(b, Ts, [table[T]['idxD100'] for T in Ts], BLUE, 'o')
    b.set_xscale('log'), b.set_yscale('log')
    b.set_xlim(70, 4e4)
    b.set_xlabel('horizon T')
    b.set_ylabel('Hill tail index of the per-orbit derivative (k = 100)')
    b.set_title('Tail of the pathwise estimator', loc='left')
    fig.tight_layout()
    save(fig, os.path.join(outdir, 'pathwise_horizon.png'))


# ---- figure 4: collisions ---------------------------------------------------------------------------------------------------------

def parse_growth(path):
    """gamma quantiles per nu from 'growth rate gamma (per unit time): median ..., 90% ..., 99% ...' lines."""
    out, nu = {}, None
    for l in open(path):
        m = re.match(r'\s+nu = ([\d.e-]+): \(', l)
        if m:
            nu = float(m.group(1))
        g = re.match(r'\s+growth rate gamma \(per unit time\): median ([\d.]+), 90% ([\d.]+), 99% ([\d.]+), 99.9% ([\d.]+), max ([\d.]+)', l)
        if g and nu is not None:
            out[nu] = {'median': float(g.group(1)), '99': float(g.group(3))}
    return out


def parse_relsem(path):
    """SEM / |mean| of the g1 pathwise estimator per horizon and nu, from the 'pathwise derivative with common random numbers' block."""
    txt = open(path).read().split('g2 =')[0]
    out = {}
    for b in re.split(r'\n  T = ', txt)[1:]:
        T = float(b.split('\n')[0])
        for line in b.split('\n')[2:]:
            m = re.match(r'\s+(\S+)\s+(\d+)\s+([+-][\d.]+(?:e[+-]?\d+)?)\s+\+-\s+([\d.]+(?:e[+-]?\d+)?)', line)
            if m:
                out.setdefault(float(m.group(1)), {})[T] = float(m.group(4)) / abs(float(m.group(3)))
    return out


def fig_collisions(outdir):
    pg, pf = 'experiments/reference_results/collision_summary.txt', 'experiments/reference_results/collision_crn_fine.txt'
    if not (os.path.exists(pg) and os.path.exists(pf)):
        print('figure collisions: missing input, skipped')
        return
    growth, rel = parse_growth(pg), parse_relsem(pf)
    nus = sorted(growth)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.5))
    xs = np.arange(len(nus))
    a.plot(xs, [growth[n]['99'] for n in nus], '-', color=ORANGE, lw=LW, label='99th percentile of the paths')
    marker(a, xs, [growth[n]['99'] for n in nus], ORANGE, 's')
    a.plot(xs, [growth[n]['median'] for n in nus], '-', color=BLUE, lw=LW, label='median path')
    marker(a, xs, [growth[n]['median'] for n in nus], BLUE, 'o')
    a.set_xticks(xs)
    a.set_xticklabels(['0' if n == 0 else f'{n:.0e}'.replace('e-0', 'e-') for n in nus])
    a.set_xlabel('collision frequency ν')
    a.set_ylabel('growth rate γ of |∂s_T/∂ε_t|, per unit time')
    a.set_ylim(0, 0.032)
    a.set_title('The tangent grows faster with collisions', loc='left')
    a.legend(loc='upper left')
    a.grid(axis='x', visible=False)
    colors = {0.0: INK2, 1e-4: RAMP['250'], 1e-3: RAMP['400'], 1e-2: RAMP['550']}
    shapes = {0.0: 'o', 1e-4: 's', 1e-3: 'D', 1e-2: '^'}
    b.axhline(0.1, color=INK2, lw=0.9)
    b.annotate('10 %', (50, 0.1), textcoords='offset points', xytext=(0, 3), color=INK2, fontsize=7.5)
    for n in nus:
        Ts = sorted(rel[n])
        y = [min(rel[n][T], 30) for T in Ts]
        b.plot(Ts, y, '-', color=colors[n], lw=LW, label='ν = 0' if n == 0 else f'ν = {n:.0e}'.replace('e-0', 'e-'))
        marker(b, Ts, y, colors[n], shapes[n])
    b.set_yscale('log')
    b.set_xlim(40, 540)
    b.set_ylim(0.01, 60)
    b.set_xlabel('horizon T')
    b.set_ylabel('relative error of the pathwise gradient')
    b.set_title('Usable horizon: error below 10 %  (2000 paths)', loc='left')
    b.legend(loc='lower right', ncol=2)
    fig.tight_layout()
    save(fig, os.path.join(outdir, 'collision_horizon.png'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='docs/figures')
    ap.add_argument('--only', nargs='*', default=None, choices=['nilss_vs_fd', 'tails', 'pathwise', 'collisions'])
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    style()
    todo = {'nilss_vs_fd': fig_nilss_vs_fd, 'tails': fig_tails, 'pathwise': fig_pathwise, 'collisions': fig_collisions}
    for name in args.only or todo:
        todo[name](args.outdir)


if __name__ == '__main__':
    main()
