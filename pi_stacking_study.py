#!/usr/bin/env python
"""pi-pi contacts between the azole ligands (constant-pressure runs).

CoSCN2bIm2: every benzimidazole (9 ring atoms) gets, per frame, a best-fit
plane normal and the centroids of its imidazole (5) and benzene (6) rings.
Co_1MIm: every 1-methylimidazole gets its ring plane and ring centroid (the
methyl C is left out).  Every pair of
ligands, including a ligand with its own periodic images, is checked over the
27 neighbouring cell translations; for each pair/translation the closest
pair of ring centroids is kept.

  contact   centroid-centroid d_cc <= DCC (4.2 A) and interplanar angle <= 30 deg
  perp      distance of the partner centroid from the plane of the ligand
  slip      lateral offset sqrt(d_cc^2 - perp^2)

The contacts present in the first frame (the 100 K structure) are followed by
identity (ligand pair, translation, rings) through the run, and contacts that
form later are counted too.  Ligands whose Co-N bond ends above 2.6 A (see
bond_break_study.py) are marked '*', above 3.5 A (ligand gone) '**'.

Writes <out>/pi_<series>.txt and fig_pi_<series>.png/.pdf:
  top row     contacts per cell vs time, one panel per T;
  middle row  d_cc of each 100 K contact vs time;
  bottom      production means vs T (count, d_cc, angle, slip).

    python pi_stacking_study.py                       # CoSCN2bIm2, nptaniso
    python pi_stacking_study.py --system Co_1MIm
    python pi_stacking_study.py --series nptanisoimp
"""
from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fig_coN_heat import read_run, mic, running_mean, BASE, DT_PS
from bond_break_study import molecules, ring_of, angle

TEMPS = [300, 350, 400, 450, 500]
DCC, ANG = 4.2, 30.0
SHIFTS = np.array(list(itertools.product((-1, 0, 1), repeat=3)))


def ligands(sym, p0, c0):
    """Azole ligands -> dicts: plane atoms, rings (list of atom lists), ring N."""
    adj, comp = molecules(sym, p0, c0)
    out, seen = [], set()
    for n in np.where(sym == 'N')[0]:
        r5 = ring_of(n, sym, adj)
        if r5 is None or comp[n] in seen:
            continue
        seen.add(comp[n])
        heavy = [a for a in range(len(sym)) if comp[a] == comp[n] and sym[a] != 'H']
        fused = [a for a in r5 if sym[a] == 'C' and
                 sum(sym[b] != 'H' and comp[b] == comp[n] for b in adj[a]) == 3]
        ring_atoms = [a for a in heavy if a in r5 or
                      sum(sym[b] != 'H' for b in adj[a]) >= 2]      # drops methyl C
        rings = [r5]
        if len(fused) == 2 and len(ring_atoms) == 9:                 # benzimidazole
            rings.append([a for a in ring_atoms if a not in r5] + fused)
        else:
            ring_atoms = r5
        out.append(dict(heavy=ring_atoms, rings=rings, n=[a for a in r5 if sym[a] == 'N']))
    return out


def geometry(xyz, cell, L):
    """-> ring centroids (F, nring, 3), unit normal (F, 3)."""
    a0 = L['heavy'][0]
    rel = {a: mic(xyz[:, a] - xyz[:, a0], cell) for a in L['heavy']}
    pts = np.stack([rel[a] for a in L['heavy']], axis=1)            # (F, 9, 3)
    ctr = pts.mean(axis=1, keepdims=True)
    _, _, vt = np.linalg.svd(pts - ctr)
    nrm = vt[:, 2]
    cs = [np.mean([rel[a] for a in r], axis=0) + xyz[:, a0] for r in L['rings']]
    return np.stack(cs, axis=1), nrm


def analyse(rundir, stride, system):
    run = read_run(rundir)
    if run is None:
        return None
    sym, steps, xyz, cell = run
    if cell.ndim == 2:
        cell = np.repeat(cell[None], len(xyz), axis=0)
    steps, xyz, cell = steps[::stride], xyz[::stride], cell[::stride]
    t = steps * DT_PS
    ligs = ligands(sym, xyz[0], cell[0])
    geo = [geometry(xyz, cell, L) for L in ligs]
    # which ligands lose their Co (Co-N > 2.6 A over the last 10 % of the run)
    co = np.where(sym == 'Co')[0]
    broken, gone = set(), set()
    for k, L in enumerate(ligs):
        for n in L['n']:
            d = np.linalg.norm(mic(xyz[:, co] - xyz[:, n][:, None], cell), axis=2).min(axis=1)
            end = d[-max(1, len(d) // 10):].mean()
            if d[0] < 2.6 and end > 2.6:
                broken.add(k)
            if d[0] < 2.6 and end > 3.5:
                gone.add(k)
    F = len(t)
    pairs = {}    # (i, j, shift index) -> (d_cc, angle, perp, slip) arrays (F,)
    for i, j in itertools.combinations_with_replacement(range(len(ligs)), 2):
        ci, ni = geo[i]
        cj, nj = geo[j]
        ang = angle(ni, nj)
        ang = np.minimum(ang, 180 - ang)
        for s, sh in enumerate(SHIFTS):
            if i == j and not sh.any():
                continue
            if i == j and tuple(-sh) < tuple(sh):     # self-image pairs counted once
                continue
            T = np.einsum('k,fkl->fl', sh, cell)
            best = None
            for a in range(ci.shape[1]):
                for b in range(cj.shape[1]):
                    d = cj[:, b] + T - ci[:, a]
                    dd = np.linalg.norm(d, axis=1)
                    if best is None:
                        best, vec = dd, d
                    else:
                        m = dd < best
                        best = np.where(m, dd, best)
                        vec = np.where(m[:, None], d, vec)
            if best.min() > 6.0:
                continue
            perp = np.abs(np.einsum('fi,fi->f', vec, ni))
            slip = np.sqrt(np.maximum(best ** 2 - perp ** 2, 0))
            pairs[(i, j, s)] = (best, ang, perp, slip)
    return dict(t=t, ligs=ligs, broken=broken, gone=gone, pairs=pairs, F=F)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--series', default='nptaniso')
    ap.add_argument('--system', default='CoSCN2bIm2', choices=('CoSCN2bIm2', 'Co_1MIm'))
    ap.add_argument('--stride', type=int, default=10)
    ap.add_argument('--out', default='analysis_2026-11-16/pi_stacking')
    args = ap.parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(root, args.out)
    os.makedirs(out, exist_ok=True)

    res, lines = {}, []
    for T in TEMPS:
        r = analyse(os.path.join(BASE, args.system, f'{args.series}_{T}'), args.stride, args.system)
        if r is None:
            continue
        res[T] = r
        t, F = r['t'], r['F']
        half = t >= t[-1] / 2
        lines.append(f'== {args.system} {args.series} {T} K: {F} frames to {t[-1]:.1f} ps, '
                     f'{len(r["ligs"])} azole ligands, Co-N > 2.6 A at the end: '
                     f'{sorted(r["broken"]) or "none"}, > 3.5 A: {sorted(r["gone"]) or "none"}')
        is_c = {k: (v[0] <= DCC) & (v[1] <= ANG) for k, v in r['pairs'].items()}
        r['count'] = np.sum(list(is_c.values()), axis=0)
        r['init'] = [k for k, c in is_c.items() if c[:max(1, F // 200)].all()]
        lines.append(f'   contacts per cell: 100 K frame {r["count"][0]}, 2nd half mean '
                     f'{r["count"][half].mean():.2f} +/- {r["count"][half].std():.2f}')
        lines.append('   pair (ligands, image)      present    d_cc   angle   perp   slip  (2nd half; '
                     'present = fraction of 2nd half in contact)  100 K: d_cc angle')
        for k, c in sorted(is_c.items(), key=lambda kv: -kv[1][half].mean()):
            frac = c[half].mean()
            if frac < 0.05 and k not in r['init']:
                continue
            d, a, p, s = (v[half] for v in r['pairs'][k])
            i, j, sh = k
            mk = lambda x: '**' if x in r['gone'] else ('*' if x in r['broken'] else '')
            tag = f'L{i}{mk(i)}-L{j}{mk(j)} ({",".join(str(int(v)) for v in SHIFTS[sh])})'
            lines.append(f'   {tag:26s} {frac:7.2f}  {d.mean():6.2f} {a.mean():6.1f} {p.mean():6.2f} '
                         f'{s.mean():6.2f}      {r["pairs"][k][0][0]:5.2f} {r["pairs"][k][1][0]:5.1f}'
                         + ('   (100 K contact)' if k in r['init'] else '   (new)'))
        r['shown'] = sorted(set(r['init']) | {k for k, c in is_c.items() if c[half].mean() >= 0.2})
        r['summary'] = []
        for k in r['shown']:
            d, a, p, s = (v[half] for v in r['pairs'][k])
            r['summary'].append((d.mean(), a.mean(), s.mean(), is_c[k][half].mean()))
    txt = '\n'.join(lines)
    print(txt)
    with open(os.path.join(out, f'pi_{args.system}_{args.series}.txt'), 'w') as fp:
        fp.write(txt + '\n')

    Ts = list(res)
    fig = plt.figure(figsize=(2.6 * len(Ts), 7.2))
    gs = fig.add_gridspec(3, len(Ts), height_ratios=(1, 1.3, 1.2))
    cols = plt.cm.tab10.colors
    for c, T in enumerate(Ts):
        r = res[T]
        w = max(1, 100 // args.stride)
        ax = fig.add_subplot(gs[0, c])
        ax.plot(r['t'], r['count'], lw=0.4, color='#c3c2b7')
        ax.plot(r['t'], running_mean(r['count'].astype(float), w), lw=1, color='#0b0b0b')
        ax.set_title(f'{T} K', fontsize=8)
        ax.set_ylim(-0.3, max(rr['count'].max() for rr in res.values()) + 0.5)
        if c == 0:
            ax.set_ylabel(f'π contacts per cell\n(d_cc ≤ {DCC} Å, ≤ {ANG:.0f}°)', fontsize=7)
        ax2 = fig.add_subplot(gs[1, c], sharex=ax)
        mk = lambda x: '**' if x in r['gone'] else ('*' if x in r['broken'] else '')
        for q, k in enumerate(r['shown']):
            i, j, sh = k
            star = i in r['broken'] or j in r['broken']
            img = ','.join(str(int(v)) for v in SHIFTS[sh])
            ax2.plot(r['t'], running_mean(r['pairs'][k][0], w), lw=1.0, color=cols[q % 10],
                     ls='--' if star else '-',
                     label=f'L{i}{mk(i)}-L{j}{mk(j)} ({img})' + ('' if k in r['init'] else ' new'))
        ax2.axhline(DCC, color='#898781', lw=0.6, ls=':')
        ax2.set_xlabel('time (ps)', fontsize=7)
        ax2.legend(fontsize=5, ncol=2, frameon=False)
        if c == 0:
            ax2.set_ylabel('d_cc of the contacts (Å)\n(at 100 K or ≥ 20 % of 2nd half)\ndashed: a ligand off its Co', fontsize=7)
        for a_ in (ax, ax2):
            a_.tick_params(labelsize=6)
    # vs T
    keys = [('count', 'contacts per cell'), (0, 'd_cc (Å)'), (1, 'angle (°)'), (2, 'slip (Å)')]
    for q, (key, lab) in enumerate(keys):
        ax = fig.add_subplot(gs[2, q * len(Ts) // 4:(q + 1) * len(Ts) // 4] if len(Ts) >= 4 else gs[2, :])
        for T in Ts:
            r = res[T]
            half = r['t'] >= r['t'][-1] / 2
            if key == 'count':
                ax.errorbar(T, r['count'][half].mean(), r['count'][half].std(), fmt='o', color='#0b0b0b', ms=3)
            else:
                for v in r['summary']:
                    ax.plot(T, v[key], 'o', ms=2.5, color='#2a78d6', alpha=0.7)
        ax.set_xlabel('T (K)', fontsize=7)
        ax.set_xlim(280, 520)
        ax.set_ylabel(lab + ('' if key == 'count' else '\ncontacts shown above, 2nd half'), fontsize=6)
        ax.tick_params(labelsize=6)
    fig.suptitle(f'Azole π–π contacts, {args.system} {args.series}', fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out, f'fig_pi_{args.system}_{args.series}.{ext}'), dpi=150)
    print('wrote', out)


if __name__ == '__main__':
    main()
