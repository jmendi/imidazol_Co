#!/usr/bin/env python
"""Co-N first-shell distances along the 300-500 K CIF-cell MD (send_cif_heat.sh).

Grid: one row per system, one column per temperature.  Each panel joins the
warm-up (steps 0-1000) and the production (1000-11000) into one time axis,
using the step number written in every answer.xyz frame, so a run still in
progress is plotted up to wherever it has got to.  The vertical line marks the
end of the warm-up.

Thin lines are individual Co-N bonds; bold lines are a running mean (WINDOW
fs) of each distance sub-group -- at 300-500 K a single bond is too noisy to
read.  Bonds are picked once, in the first frame, with RCUT = 3.2 A (same
choice as fig_coN_all.py: the second shell starts at ~4.1 A) and then followed
by index, so a bond that breaks is still drawn as it leaves.  Dashed lines are
the start of the run (the last frame of the 100 K MD).  All panels in a row
share the y-axis.

    python fig_coN_heat.py                 # all 15 runs
    python fig_coN_heat.py --window 200    # smoother mean lines
    python fig_coN_heat.py --sc 211        # the 2x1x1 supercells (send_sc_heat.sh)
    python fig_coN_heat.py --sc 221        # the 2x2x1 supercells
    python fig_coN_heat.py --series npt    # constant-pressure runs, two-body D3 (send_npt_heat.sh)
    python fig_coN_heat.py --series npt3b  # constant-pressure runs, with three-body D3

Constant-pressure runs change the cell every 20 steps (isotropic: shape kept,
volume from <stage>/cell.dat).  The cell of each frame is the input.lvs cell
scaled by (V/V0)^(1/3), V being the volume after the last barostat move before
that frame, and every minimum-image distance uses it.

A second figure, fig_CN_heat, shows the Co-N coordination number n(RCN) per Co
against time, on the same grid: every N within RCN = 2.6 A of the Co is
counted in every frame (not only the bonds picked in frame 0), so an N that
comes in from the second shell is counted too.  2.6 A is the project's
convention (1MIm 4+2, CoBIm 3+1 analyses): it sits in the gap between bound
(<= 2.3 A) and detached (2.8-3.0 A) N.  Thin lines are each Co's running mean,
the step line the instantaneous CN of the whole cell averaged over Co.

Writes cif_runs/fig_coN_heat.pdf/.png and fig_CN_heat.pdf/.png, and prints,
per run and group, the mean, sd and max over the last half of what has been
run so far, then the same for the CN of each Co.
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# works on the cluster (~/imidazol_Co) and on the workstation (~/fast05/imidazol_Co)
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cif_runs')

SYSTEMS = [   # (row label, directory, run-name pattern with {T})
    ('Co_1MIm',    'Co_1MIm',    'md_{T}_corrected'),
    ('Co_12dMIm',  'Co_12dMIm',  'md_{T}'),
    ('CoSCN2bIm2', 'CoSCN2bIm2', 'md_{T}_restrained'),
]
TEMPS = [300, 350, 400, 450, 500]
WARM_STEPS = 1000

RCUT, R_NC, R_CS, DT_PS, SPLIT = 3.2, 1.60, 1.90, 0.001, 0.15
RCN = 2.6     # Co-N coordination cutoff, A (see docstring)

C_NCS, C_IM = '#2a78d6', '#eb6834'      # reference palette, slots 1 and 2
INK_PRIMARY, INK_SECONDARY, INK_MUTED = '#0b0b0b', '#52514e', '#898781'
GRIDLINE, BASELINE, SURFACE = '#e1e0d9', '#c3c2b7', '#fcfcfb'

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.edgecolor': BASELINE, 'axes.linewidth': 0.6,
    'xtick.color': INK_MUTED, 'ytick.color': INK_MUTED,
    'xtick.labelcolor': INK_SECONDARY, 'ytick.labelcolor': INK_SECONDARY,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'pdf.fonttype': 42, 'font.family': 'sans-serif',
})

STEP_RE = re.compile(r'step\s+(\d+)')


def read_traj(path):
    """-> (symbols, steps, xyz).  A partly written last frame is dropped."""
    with open(path) as fp:
        lines = fp.read().splitlines()
    natoms = int(lines[0].split()[0])
    block = natoms + 2
    nf = len(lines) // block
    sym = np.array([lines[2 + i].split()[0] for i in range(natoms)])
    steps = np.empty(nf, dtype=int)
    xyz = np.empty((nf, natoms, 3))
    for f in range(nf):
        off = f * block
        m = STEP_RE.search(lines[off + 1])
        steps[f] = int(m.group(1)) if m else f
        try:
            xyz[f] = [l.split()[1:4] for l in lines[off + 2:off + block]]
        except ValueError:                      # truncated frame being written
            return sym, steps[:f], xyz[:f]
    return sym, steps, xyz


def read_run(rundir):
    """Warm-up + production, ordered by step, duplicates (restart overlap) removed."""
    sym, parts_s, parts_x = None, [], []
    for stage in ('warmup', 'prod'):
        p = os.path.join(rundir, stage, 'answer.xyz')
        if os.path.isfile(p) and os.path.getsize(p):
            sym, s, x = read_traj(p)
            parts_s.append(s)
            parts_x.append(x)
    if not parts_s:
        return None
    steps, xyz = np.concatenate(parts_s), np.concatenate(parts_x)
    # keep the LAST occurrence of each step (a resumed run rewrites from its checkpoint)
    _, idx = np.unique(steps[::-1], return_index=True)
    keep = np.sort(len(steps) - 1 - idx)
    steps, xyz = steps[keep], xyz[keep]
    cell0 = np.loadtxt(os.path.join(rundir, 'warmup', 'input.lvs'))
    # flexible-cell runs (md_fireballpy_aniso.py): full cell per barostat move in cellmat.dat
    mats = [np.loadtxt(os.path.join(rundir, st, 'cellmat.dat'), ndmin=2, usecols=range(10))
            for st in ('warmup', 'prod') if os.path.isfile(os.path.join(rundir, st, 'cellmat.dat'))]
    mats = [m for m in mats if m.size]
    if mats:
        hist = np.vstack(mats)
        hist = hist[np.argsort(hist[:, 0], kind='stable')]
        i = np.searchsorted(hist[:, 0], steps, side='left') - 1     # see the note below
        cells = np.where((i >= 0)[:, None, None],
                         hist[np.clip(i, 0, None), 1:10].reshape(-1, 3, 3), cell0[None])
        return sym, steps, xyz, cells
    # constant-pressure runs: one cell per frame from cell.dat (isotropic scaling)
    rows = [np.loadtxt(os.path.join(rundir, st, 'cell.dat'), ndmin=2, usecols=(0, 1))
            for st in ('warmup', 'prod') if os.path.isfile(os.path.join(rundir, st, 'cell.dat'))]
    rows = [r for r in rows if r.size]
    if not rows:
        return sym, steps, xyz, cell0
    hist = np.vstack(rows)
    hist = hist[np.argsort(hist[:, 0], kind='stable')]
    v0 = abs(np.linalg.det(cell0))
    # the row of step k is the cell AFTER the move made at the end of step k:
    # frame k itself still has the previous cell
    i = np.searchsorted(hist[:, 0], steps, side='left') - 1
    vol = np.where(i >= 0, hist[np.clip(i, 0, None), 1], v0)
    cells = cell0[None] * ((vol / v0) ** (1.0 / 3.0))[:, None, None]
    return sym, steps, xyz, cells


def mic(d, cell):
    """Minimum image of d (..., 3).  cell (3, 3), or (F, 3, 3) with d (F, ..., 3)."""
    if cell.ndim == 2:
        f = d @ np.linalg.inv(cell)
        f -= np.round(f)
        return f @ cell
    inv = np.linalg.inv(cell)
    sh = d.shape
    dd = d.reshape(sh[0], -1, 3)
    f = np.einsum('fni,fij->fnj', dd, inv)
    f -= np.round(f)
    return np.einsum('fni,fij->fnj', f, cell).reshape(sh)


def thiocyanate_n(sym, pos, cell, n_idx):
    c_idx, s_idx = np.where(sym == 'C')[0], np.where(sym == 'S')[0]
    if not len(s_idx):
        return False
    dnc = np.linalg.norm(mic(pos[c_idx] - pos[n_idx], cell), axis=1)
    for c in c_idx[dnc < R_NC]:
        if (np.linalg.norm(mic(pos[s_idx] - pos[c], cell), axis=1) < R_CS).any():
            return True
    return False


def analyse(rundir):
    run = read_run(rundir)
    if run is None:
        return None
    sym, steps, xyz, cell = run
    co, nn, p0 = np.where(sym == 'Co')[0], np.where(sym == 'N')[0], xyz[0]
    c0 = cell if cell.ndim == 2 else cell[0]
    bonds = []
    for c in co:
        dist = np.linalg.norm(mic(p0[nn] - p0[c], c0), axis=1)
        for n in nn[dist < RCUT]:
            bonds.append((c, n, thiocyanate_n(sym, p0, c0, n)))
    r = np.empty((len(xyz), len(bonds)))
    for k, (c, n, _) in enumerate(bonds):
        r[:, k] = np.linalg.norm(mic(xyz[:, n] - xyz[:, c], cell), axis=1)
    # coordination number: all N, every frame
    cn = np.empty((len(xyz), len(co)), dtype=int)
    for k, c in enumerate(co):
        d = np.linalg.norm(mic(xyz[:, nn] - xyz[:, c][:, None], cell), axis=2)
        cn[:, k] = (d < RCN).sum(axis=1)
    return steps * DT_PS, r, np.array([b[2] for b in bonds]), cn, co


def running_mean(y, w):
    if w <= 1 or len(y) < w:
        return y
    k = np.ones(w) / w
    out = np.convolve(y, k, mode='same')
    # 'same' pads with zeros; renormalise the edges
    norm = np.convolve(np.ones_like(y), k, mode='same')
    return out / norm


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--window', type=int, default=100,
                    help='running-mean window for the bold lines, in frames (fs)')
    ap.add_argument('--sc', choices=('111', '211', '221'), default='111',
                    help='supercell: 111 = the unit-cell runs md_<T>..., '
                         '211/221 = sc211_md_<T>... / sc221_md_<T>...')
    ap.add_argument('--series', choices=('nvt', 'npt', 'npt3b', 'nptaniso', 'nptanisoimp'), default='nvt',
                    help='nvt = the fixed-cell md_<T>... runs (default); npt / npt3b = '
                         'the constant-pressure runs npt_<T> / npt3b_<T>')
    ap.add_argument('--base', default=None,
                    help='runs directory (default cif_runs)')
    ap.add_argument('--out', default=None,
                    help='output stem (default <base>/fig_coN_heat[_sc<rep>])')
    args = ap.parse_args()
    if args.base is None:
        args.base = BASE
    args.prefix = '' if args.sc == '111' else f'sc{args.sc}_'
    args.label = '' if args.sc == '111' else f', {args.sc[0]}×{args.sc[1]}×{args.sc[2]} supercell'
    suffix = '' if args.sc == '111' else f'_sc{args.sc}'
    args.pattern = None
    if args.series != 'nvt':
        if args.sc != '111':
            ap.error('--series npt/npt3b is only for the unit cells')
        args.pattern = args.series + '_{T}'
        suffix += '_' + args.series
        args.label += {'npt': ', constant pressure (P = 0), periodic two-body D3',
                       'npt3b': ', constant pressure (P = 0), periodic two- + three-body D3',
                       'nptaniso': ', constant pressure (P = 0), flexible cell, two-body D3, legacy dipoles',
                       'nptanisoimp': ', constant pressure (P = 0), flexible cell, two-body D3, improved dipoles'}[args.series]
    if args.out is None:
        args.out = os.path.join(args.base, 'fig_coN_heat' + suffix)
    args.cn_out = os.path.join(os.path.dirname(args.out), 'fig_CN_heat' + suffix)

    fig, axes = plt.subplots(len(SYSTEMS), len(TEMPS), figsize=(13.5, 7.6),
                             sharex=True, squeeze=False)
    summary, tmax, results = [], 0.0, {}

    for i, (label, sysdir, pattern) in enumerate(SYSTEMS):
        row_lo, row_hi = np.inf, -np.inf
        for j, T in enumerate(TEMPS):
            ax = axes[i, j]
            name = args.prefix + (args.pattern or pattern).format(T=T)
            res = analyse(os.path.join(args.base, sysdir, name))
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            ax.grid(axis='y', color=GRIDLINE, lw=0.5, zorder=0)
            ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(f'{T} K', color=INK_PRIMARY, pad=5)
            if j == 0:
                ax.set_ylabel(f'{label}\nCo–N (Å)', color=INK_SECONDARY)
            if res is None:
                ax.text(0.5, 0.5, 'no frames yet', transform=ax.transAxes,
                        ha='center', va='center', color=INK_MUTED)
                continue
            t, r, ncs, cn, co = res
            results[i, j] = (t, cn, co)
            tmax = max(tmax, t[-1])
            row_lo, row_hi = min(row_lo, r.min()), max(row_hi, r.max())
            for cls, colour, tag in ((True, C_NCS, 'NCS'), (False, C_IM, 'Im')):
                sub = r[:, ncs == cls]
                if not sub.size:
                    continue
                ax.plot(t, sub, color=colour, lw=0.3, alpha=0.25, zorder=2)
                means = sub.mean(axis=0)
                order = np.argsort(means)
                cut = [0] + [k for k in range(1, len(order))
                             if means[order[k]] - means[order[k - 1]] > SPLIT] + [len(order)]
                for a, b in zip(cut[:-1], cut[1:]):
                    g = sub[:, order[a:b]]
                    ax.axhline(g[0].mean(), color=colour, lw=0.8, ls=(0, (5, 3)),
                               alpha=0.6, zorder=1)
                    ax.plot(t, running_mean(g.mean(axis=1), args.window),
                            color=colour, lw=1.6, zorder=4, solid_capstyle='round')
                    h = g[len(g) // 2:]
                    summary.append((f'{sysdir}/{name}', tag, g.shape[1],
                                    h.mean(), h.std(), g.max(), int(t[-1] / DT_PS)))
            ax.axvline(WARM_STEPS * DT_PS, color=INK_MUTED, lw=0.7, ls=':', zorder=1)
        if np.isfinite(row_lo):
            for ax in axes[i]:
                ax.set_ylim(row_lo - 0.05, row_hi + 0.05)
        else:                                   # nothing run yet in this row
            for ax in axes[i]:
                ax.set_yticks([])

    for ax in axes[-1]:
        ax.set_xlabel('time (ps)', color=INK_SECONDARY)
    axes[0, 0].set_xlim(0, max(tmax, WARM_STEPS * DT_PS * 1.05))

    handles = [Line2D([], [], color=C_NCS, lw=1.6, label='Co–N (thiocyanate)'),
               Line2D([], [], color=C_IM, lw=1.6, label='Co–N (imidazole)'),
               Line2D([], [], color=INK_MUTED, lw=1.6,
                      label=f'group mean, {args.window} fs running average'),
               Line2D([], [], color=INK_MUTED, lw=0.8, ls=(0, (5, 3)),
                      label='start (last 100 K frame)'),
               Line2D([], [], color=INK_MUTED, lw=0.7, ls=':', label='end of warm-up')]
    fig.legend(handles=handles, frameon=False, loc='lower center', ncol=5,
               labelcolor=INK_SECONDARY, handlelength=2.6)
    fig.suptitle(f'Co–N first-shell distances, 300–500 K Langevin MD from the 100 K '
                 f'frames (cutoff {RCUT} Å){args.label}',
                 color=INK_PRIMARY, fontsize=9.5, x=0.008, ha='left', y=0.997)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(args.out + '.pdf')
    fig.savefig(args.out + '.png', dpi=200)
    plt.close(fig)

    print(f'{"run":<32} {"cls":<4}{"n":>3} {"mean":>8} {"sd":>7} {"max":>8} {"step":>7}'
          '   (2nd half so far, A)')
    for row in summary:
        print('%-32s %-4s%3d %8.3f %7.3f %8.3f %7d' % row)
    print(f'\nwrote {args.out}.pdf and .png\n')
    plot_cn(results, tmax, args)


def plot_cn(results, tmax, args):
    fig, axes = plt.subplots(len(SYSTEMS), len(TEMPS), figsize=(13.5, 7.6),
                             sharex=True, squeeze=False)
    rows = []
    for i, (label, sysdir, pattern) in enumerate(SYSTEMS):
        lo, hi = np.inf, -np.inf
        for j, T in enumerate(TEMPS):
            ax = axes[i, j]
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            ax.grid(axis='y', color=GRIDLINE, lw=0.5, zorder=0)
            ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(f'{T} K', color=INK_PRIMARY, pad=5)
            if j == 0:
                ax.set_ylabel(f'{label}\nn$_{{Co-N}}$({RCN} Å)', color=INK_SECONDARY)
            if (i, j) not in results:
                ax.text(0.5, 0.5, 'no frames yet', transform=ax.transAxes,
                        ha='center', va='center', color=INK_MUTED)
                continue
            t, cn, co = results[i, j]
            lo, hi = min(lo, cn.min()), max(hi, cn.max())
            for k in range(cn.shape[1]):
                ax.plot(t, running_mean(cn[:, k].astype(float), args.window),
                        color=C_IM, lw=0.9, alpha=0.55, zorder=3)
            ax.plot(t, cn.mean(axis=1), color=INK_PRIMARY, lw=0.7,
                    drawstyle='steps-post', zorder=4)
            ax.axvline(WARM_STEPS * DT_PS, color=INK_MUTED, lw=0.7, ls=':', zorder=1)
            h = cn[len(cn) // 2:]
            rows.append((f'{sysdir}/{args.prefix}{(args.pattern or pattern).format(T=T)}', int(t[-1] / DT_PS),
                         ' '.join(f'{c}:{h[:, k].mean():.2f}' for k, c in enumerate(co)),
                         ' '.join(f'{c}:{cn[0, k]}' for k, c in enumerate(co))))
        if np.isfinite(lo):
            for ax in axes[i]:
                ax.set_ylim(lo - 0.3, hi + 0.3)
                ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        else:
            for ax in axes[i]:
                ax.set_yticks([])
    for ax in axes[-1]:
        ax.set_xlabel('time (ps)', color=INK_SECONDARY)
    axes[0, 0].set_xlim(0, max(tmax, WARM_STEPS * DT_PS * 1.05))

    handles = [Line2D([], [], color=C_IM, lw=0.9, alpha=0.7,
                      label=f'one Co, {args.window} fs running average'),
               Line2D([], [], color=INK_PRIMARY, lw=0.7, drawstyle='steps-post',
                      label='mean over Co, instantaneous'),
               Line2D([], [], color=INK_MUTED, lw=0.7, ls=':', label='end of warm-up')]
    fig.legend(handles=handles, frameon=False, loc='lower center', ncol=3,
               labelcolor=INK_SECONDARY, handlelength=2.6)
    fig.suptitle(f'Co–N coordination number n({RCN} Å), all N counted every frame — '
                 f'300–500 K Langevin MD from the 100 K frames{args.label}',
                 color=INK_PRIMARY, fontsize=9.5, x=0.008, ha='left', y=0.997)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out = args.cn_out
    fig.savefig(out + '.pdf')
    fig.savefig(out + '.png', dpi=200)
    plt.close(fig)

    print(f'{"run":<32} {"step":>6}   CN per Co at start   |   mean CN per Co, 2nd half so far')
    for name, step, mean, start in rows:
        print(f'{name:<32} {step:>6}   {start}   |   {mean}')
    print(f'\nwrote {out}.pdf and .png')


if __name__ == '__main__':
    main()
