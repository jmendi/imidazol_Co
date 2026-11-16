#!/usr/bin/env python
"""Progress of the constant-pressure series (send_npt_heat.sh).

Prints one line per run - stage, step, speed, time left, and running averages of
T, volume (relative to the CIF cell) and pressure - and draws npt_status.png:
volume, pressure and temperature against time for every system and T.

    python npt_status.py                # table + figure
    python npt_status.py --no-plot
    python npt_status.py --window 500   # running-average window, steps (default 200)
    python npt_status.py --series npt3b # the three-body D3 series (figure npt3b_status.png)
    python npt_status.py --cells        # table of T and cell lengths/volume instead

Reads only energy.dat and cell.dat, so it is cheap and safe while jobs write.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

PROJ = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(PROJ, 'cif_runs')
SYSTEMS = ['Co_1MIm', 'Co_12dMIm', 'CoSCN2bIm2']
TOTAL = 11000
T_COL = {300: '#86b6ef', 350: '#3987e5', 400: '#256abf', 450: '#184f95', 500: '#0d366b'}
INK, INK2, GRID = '#1f1f1e', '#5f5e5a', '#e4e2dc'


def load(path, ncol):
    """Numeric rows with exactly ncol columns (skips a half-written last line)."""
    rows = []
    try:
        with open(path) as fh:
            for line in fh:
                f = line.split()
                if not line.startswith('#') and len(f) == ncol:
                    try:
                        rows.append([float(x) for x in f])
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    return np.array(rows).reshape(-1, ncol)


def run_data(d):
    e = np.vstack([load(os.path.join(d, s, 'energy.dat'), 7) for s in ('warmup', 'prod')])
    c = np.vstack([load(os.path.join(d, s, 'cell.dat'), 8) for s in ('warmup', 'prod')])
    if len(e):
        _, i = np.unique(e[:, 0], return_index=True)     # prod repeats step 1000
        e = e[i]
    cell = np.loadtxt(os.path.join(d, 'warmup', 'input.lvs'))
    return e, c, abs(np.linalg.det(cell))


def tail_mean(x, n):
    return x[-n:].mean() if len(x) else np.nan


def cells_table(args):
    for s in SYSTEMS:
        runs = sorted(glob.glob(os.path.join(RUNS, s, f'{args.series}_*')),
                      key=lambda p: int(p.rsplit('_', 1)[1]))
        if not runs:
            continue
        cif = np.loadtxt(os.path.join(runs[0], 'warmup', 'input.lvs'))
        a0, b0, c0 = np.linalg.norm(cif, axis=1)
        v0 = abs(np.linalg.det(cif))
        print(f'\n{s}  ({args.series})   CIF: a {a0:.3f}  b {b0:.3f}  c {c0:.3f} A   V {v0:.1f} A^3')
        print(f"  {'T set':>5s} {'<T>':>6s} {'step':>6s}  {'a (A)':>7s} {'b (A)':>7s} {'c (A)':>7s} "
              f"{'V (A^3)':>8s}  {'da %':>6s} {'db %':>6s} {'dc %':>6s} {'dV %':>6s}")
        for d in runs:
            T = int(d.rsplit('_', 1)[1])
            e, c, _ = run_data(d)
            if not len(e) or not len(c):
                print(f'  {T:5d}   (no barostat update yet)')
                continue
            w = c[-args.nbaro:]
            a, b, cc, v = w[:, 2].mean(), w[:, 3].mean(), w[:, 4].mean(), w[:, 1].mean()
            tm = e[e[:, 0] > e[-1, 0] - 200, 5].mean()
            print(f'  {T:5d} {tm:6.0f} {int(e[-1, 0]):6d}  {a:7.3f} {b:7.3f} {cc:7.3f} {v:8.1f}  '
                  f'{100 * (a / a0 - 1):+6.2f} {100 * (b / b0 - 1):+6.2f} {100 * (cc / c0 - 1):+6.2f} '
                  f'{100 * (v / v0 - 1):+6.2f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--no-plot', action='store_true')
    ap.add_argument('--window', type=int, default=200)
    ap.add_argument('--series', default='npt', help="folder prefix: npt (default), npt3b or nptaniso")
    ap.add_argument('--base', default=None,
                    help='runs directory (default cif_runs)')
    ap.add_argument('--cells', action='store_true',
                    help='print target/current T and cell a, b, c, V (mean of the last '
                         '--nbaro barostat updates) next to the CIF cell, then exit')
    ap.add_argument('--nbaro', type=int, default=5,
                    help='barostat updates averaged for --cells (default 5 = 100 steps)')
    args = ap.parse_args()
    global RUNS
    RUNS = args.base or os.path.join(PROJ, 'cif_runs')

    if args.cells:
        cells_table(args)
        return

    data = {}
    print(f"{'run':24s} {'stage':6s} {'step':>6s} {'s/step':>7s} {'left':>7s} "
          f"{'<T> K':>7s} {'V/V_cif':>8s} {'<P> GPa':>8s}")
    for s in SYSTEMS:
        for d in sorted(glob.glob(os.path.join(RUNS, s, f'{args.series}_*')),
                        key=lambda p: int(p.rsplit('_', 1)[1])):
            T = int(d.rsplit('_', 1)[1])
            e, c, v0 = run_data(d)
            data[(s, T)] = (e, c, v0)
            if not len(e):
                print(f'{s + "/" + os.path.basename(d):24s} not started')
                continue
            step = int(e[-1, 0])
            stage = 'prod' if step > 1000 else 'warmup'
            m = e[:, 0] > step - args.window
            # speed from wall-clock column of the current stage's last rows
            w = e[-min(len(e), 50):]
            sps = (np.diff(w[:, 6]) / np.diff(w[:, 0]))
            sps = np.median(sps[sps > 0]) if (sps > 0).any() else np.nan
            try:    # a run may carry its own production length (extended runs: 30000)
                total = 1000 + int(open(os.path.join(d, 'prod_steps')).read())
            except (OSError, ValueError):
                total = TOTAL
            left = (total - step) * sps / 3600
            nwin = max(1, args.window // 20)
            print(f'{s + "/" + os.path.basename(d):24s} {stage:6s} {step:6d} {sps:7.1f} '
                  f'{left:6.0f}h {e[m, 5].mean():7.1f} '
                  f'{tail_mean(c[:, 1], nwin) / v0 if len(c) else np.nan:8.4f} '
                  f'{tail_mean(c[:, 7], nwin):8.3f}')

    if args.no_plot:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
    for j, s in enumerate(SYSTEMS):
        for (ss, T), (e, c, v0) in sorted(data.items()):
            if ss != s or not len(e):
                continue
            col = T_COL.get(T, INK2)
            if len(c):
                t = c[:, 0] / 1000
                k = max(1, min(10, len(c) // 5))
                sm = lambda y: np.convolve(y, np.ones(k) / k, mode='valid')
                axes[0, j].plot(t, 100 * (c[:, 1] / v0 - 1), color=col, lw=0.6, alpha=0.35)
                axes[0, j].plot(t[k - 1:], 100 * (sm(c[:, 1]) / v0 - 1), color=col, lw=1.6,
                                label=f'{T} K')
                axes[1, j].plot(t[k - 1:], sm(c[:, 7]), color=col, lw=1.6)
            k = max(1, min(100, len(e) // 5))
            axes[2, j].plot(e[k - 1:, 0] / 1000,
                            np.convolve(e[:, 5], np.ones(k) / k, mode='valid'),
                            color=col, lw=1.6)
        axes[0, j].set_title(s, loc='left', fontsize=10, color=INK)
    axes[0, 0].set_ylabel('V vs CIF cell (%)')
    axes[1, 0].set_ylabel('P (GPa, running mean)')
    axes[2, 0].set_ylabel('T (K, running mean)')
    for ax in axes.flat:
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)
    for ax in axes[1]:
        ax.axhline(0, color=INK2, lw=0.8, ls=':')
    for ax in axes[0]:
        ax.axhline(0, color=INK2, lw=0.8, ls=':')
    for ax in axes[2]:
        ax.set_xlabel('time (ps; warm-up 0-1, production from 1)')
        ax.axvline(1, color=INK2, lw=0.8, ls=':')
    axes[0, 0].legend(fontsize=8, frameon=False)
    d3 = 'two-body + three-body D3' if args.series == 'npt3b' else 'two-body D3'
    fig.suptitle(f'Constant-pressure series {args.series} (P = 0, periodic {d3}): cell volume, '
                 'pressure, temperature', x=0.01, ha='left', fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(RUNS, f'{args.series}_status.png')
    fig.savefig(out, dpi=150)
    print(f'\nfigure: {out}')


if __name__ == '__main__':
    main()
