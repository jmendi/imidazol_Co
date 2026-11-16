#!/usr/bin/env python
"""Evolution of the cell during the constant-pressure runs (send_npt_heat.sh).

One row per system, columns a, b, c (A) and V (A^3) against time, one colour per
temperature.  Thin lines are every barostat update (every 20 fs); bold lines a
running mean over --window updates.  Dashed: the CIF value.  Dotted: end of the
1 ps warm-up.  The barostat is isotropic (the cell shape is kept), so a, b and c
change by the same factor; they are drawn separately to show the absolute lengths.

    python fig_cell_npt.py                  # two-body D3 series (npt_<T>)
    python fig_cell_npt.py --series npt3b   # with three-body D3 (npt3b_<T>)
    python fig_cell_npt.py --window 25      # smoother bold lines (default 10 = 200 fs)

Writes cif_runs/fig_cell_<series>.png/.pdf and prints, per run, the mean cell over
the production so far (or the last 200 fs of warm-up if production has not begun).
Reads only cell.dat, so it is cheap and safe while jobs write.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cif_runs')
SYSTEMS = ['Co_1MIm', 'Co_12dMIm', 'CoSCN2bIm2']
WARM = 1000
T_COL = {300: '#86b6ef', 350: '#3987e5', 400: '#256abf', 450: '#184f95', 500: '#0d366b'}
INK, INK2, MUTED, GRID = '#1f1f1e', '#52514e', '#898781', '#e4e2dc'


def load_cell(d):
    rows = []
    for st in ('warmup', 'prod'):
        p = os.path.join(d, st, 'cell.dat')
        if os.path.isfile(p):
            for line in open(p):
                f = line.split()
                if line.startswith('#') or len(f) != 8:
                    continue
                try:
                    rows.append([float(x) for x in f])
                except ValueError:
                    pass
    if not rows:
        return None
    c = np.array(rows)
    _, i = np.unique(c[:, 0], return_index=True)
    return c[i]          # step, V, a, b, c, P_vir, P_kin, P_int


def load_angles(d):
    """step, alpha, beta, gamma from cellmat.dat (flexible-cell runs), or None."""
    rows = []
    for st in ('warmup', 'prod'):
        p = os.path.join(d, st, 'cellmat.dat')
        if os.path.isfile(p):
            for line in open(p):
                f = line.split()
                if line.startswith('#') or len(f) < 13:
                    continue
                rows.append([float(f[0])] + [float(x) for x in f[10:13]])
    if not rows:
        return None
    a = np.array(rows)
    _, i = np.unique(a[:, 0], return_index=True)
    return a[i]


def smooth(y, w):
    if w <= 1 or len(y) < 2:
        return y
    k = np.ones(min(w, len(y))) / min(w, len(y))
    return np.convolve(y, k, mode='same') / np.convolve(np.ones_like(y), k, mode='same')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--series', default='npt', help='npt (default), npt3b or nptaniso')
    ap.add_argument('--base', default=None,
                    help='runs directory (default cif_runs)')
    ap.add_argument('--window', type=int, default=10,
                    help='running mean over this many barostat updates (default 10 = 200 fs)')
    args = ap.parse_args()
    global RUNS
    RUNS = args.base or RUNS

    cols = [('a', 2, 'a (Å)'), ('b', 3, 'b (Å)'), ('c', 4, 'c (Å)'), ('V', 1, 'V (Å$^3$)')]
    angles = any(load_angles(d) is not None
                 for s in SYSTEMS for d in glob.glob(os.path.join(RUNS, s, f'{args.series}_*')))
    if angles:     # flexible cell: the angles change too
        cols += [('alpha', 1, 'α (°)'), ('beta', 2, 'β (°)'), ('gamma', 3, 'γ (°)')]
    fig, axes = plt.subplots(len(SYSTEMS), len(cols), figsize=(3.4 * len(cols), 8.2),
                             sharex=True)
    tmax = 0.0
    print(f"{'run':24s} {'over':>12s} {'a':>8s} {'b':>8s} {'c':>8s} {'V':>9s} {'dV %':>7s}")
    for i, s in enumerate(SYSTEMS):
        runs = sorted(glob.glob(os.path.join(RUNS, s, f'{args.series}_*')),
                      key=lambda p: int(p.rsplit('_', 1)[1]))
        if not runs:
            continue
        cif = np.loadtxt(os.path.join(runs[0], 'warmup', 'input.lvs'))
        from ase.geometry import cell_to_cellpar
        par = cell_to_cellpar(cif)
        ref = {'a': par[0], 'b': par[1], 'c': par[2], 'V': abs(np.linalg.det(cif)),
               'alpha': par[3], 'beta': par[4], 'gamma': par[5]}
        for d in runs:
            T = int(d.rsplit('_', 1)[1])
            c = load_cell(d)
            if c is None:
                continue
            t = c[:, 0] / 1000
            tmax = max(tmax, t[-1])
            col = T_COL.get(T, MUTED)
            ang = load_angles(d) if angles else None
            for j, (key, k, _) in enumerate(cols):
                ax = axes[i, j]
                if key in ('alpha', 'beta', 'gamma'):
                    if ang is None:
                        continue
                    tt, y = ang[:, 0] / 1000, ang[:, k]
                else:
                    tt, y = t, c[:, k]
                ax.plot(tt, y, color=col, lw=0.5, alpha=0.3, zorder=2)
                ax.plot(tt, smooth(y, args.window), color=col, lw=1.6, zorder=3,
                        label=f'{T} K')
            prod = c[c[:, 0] > WARM]
            win, lab = (prod, 'production') if len(prod) else (c[-10:], 'last 200 fs')
            m = win.mean(axis=0)
            print(f'{s + "/" + os.path.basename(d):24s} {lab:>12s} {m[2]:8.3f} {m[3]:8.3f} '
                  f'{m[4]:8.3f} {m[1]:9.1f} {100 * (m[1] / ref["V"] - 1):+7.2f}')
        for j, (key, _, ylab) in enumerate(cols):
            ax = axes[i, j]
            ax.axhline(ref[key], color=INK2, lw=0.9, ls=(0, (5, 3)), zorder=1)
            ax.axvline(WARM / 1000, color=MUTED, lw=0.7, ls=':', zorder=1)
            if i == 0:
                ax.set_title(ylab, loc='left', fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel(f'{s}', fontsize=10, color=INK)
            if key in ('alpha', 'beta', 'gamma'):
                continue          # angles: absolute degrees, no % axis
            # secondary axis in % of the CIF value, so all panels read the same way
            sec = ax.secondary_yaxis('right', functions=(lambda y, r=ref[key]: 100 * (y / r - 1),
                                                         lambda p, r=ref[key]: r * (1 + p / 100)))
            sec.tick_params(labelsize=7, colors=MUTED)
            if j == len(cols) - 1:
                sec.set_ylabel('% vs CIF', fontsize=8, color=MUTED)
    for ax in axes.flat:
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ('top',):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('time (ps; warm-up 0-1, production from 1)', color=INK2)
    axes[0, 0].set_xlim(0, max(tmax, 1.05))
    handles = [Line2D([], [], color=T_COL[T], lw=1.6, label=f'{T} K') for T in sorted(T_COL)]
    handles += [Line2D([], [], color=INK2, lw=0.9, ls=(0, (5, 3)), label='CIF cell'),
                Line2D([], [], color=MUTED, lw=0.7, ls=':', label='end of warm-up')]
    fig.legend(handles=handles, frameon=False, loc='lower center', ncol=7, fontsize=8)
    d3 = 'two- + three-body D3' if args.series == 'npt3b' else 'two-body D3'
    kind = 'flexible cell' if angles else 'isotropic'
    fig.suptitle(f'Cell evolution, constant pressure (P = 0, {kind}), periodic {d3} '
                 f'— series {args.series}; bold: {args.window * 20} fs running mean',
                 x=0.01, ha='left', fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out = os.path.join(RUNS, f'fig_cell_{args.series}')
    fig.savefig(out + '.png', dpi=170)
    fig.savefig(out + '.pdf')
    print(f'\nwrote {out}.png and .pdf')


if __name__ == '__main__':
    main()
