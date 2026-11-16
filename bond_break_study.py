#!/usr/bin/env python
"""Origin of the long Co-N bonds in Co_1MIm and CoSCN2bIm2 (flexible-cell runs).

For every run <system>/<series>_<T> (default series nptaniso) each Co-N bond of
the first frame is followed (same reader and minimum image as fig_coN_heat.py).
A bond is an "event" when its 100-fs running mean ends above RBREAK (2.6 A, the
project's coordination cutoff).  For each event the script records, against
time,

  r        the Co-N distance;
  rcen     Co to the centroid of the ligand's ring (ligand leaves as a whole, or
           only pivots about the ring);
  tilt     angle between Co->N and the N lone-pair direction (bisector of the
           two ring neighbours of N, pointing away from the ring): 0 = Co on the
           lone-pair axis; a sliding/rotating ligand shows a growing tilt;
  rtrans   for Co_1MIm the Co-N distance of the ligand trans to the leaving one;
  tau4     for 4-coordinate Co (CoSCN2bIm2) the Houser tau4 of the 4 initial
           donors (1 tetrahedron, 0 square plane), and the angle sum of the
           three retained donors (360 = Co in their plane);
  near     the closest atom to Co that is not one of its initial donors;
  hN       the closest H to the leaving N (other molecules), i.e. whether the
           freed lone pair is picked up by an X-H...N contact;
  nhS      for benzimidazole, the H...S distance of the leaving ligand's own
           N-H (its N-H...S hydrogen bond).

Writes analysis dir <out>/bond_breaking/: events_<series>.txt (one block per
event: onset time, before/after averages of every descriptor), and
fig_bond_break_<series>.png/.pdf (one row per event, columns = descriptors),
and a per-run summary of all bonds (max running mean) for the runs without
events, so the 300-400 K references are in the same table.

    python bond_break_study.py                       # nptaniso, both systems
    python bond_break_study.py --series nptanisoimp
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fig_coN_heat import read_run, mic, running_mean, BASE, DT_PS

SYSTEMS = ['Co_1MIm', 'CoSCN2bIm2']
TEMPS = [300, 350, 400, 450, 500]
RBOND0, RBREAK, WIN = 3.2, 2.6, 100       # initial bond cutoff, break cutoff, running mean (frames = fs)
RCOV = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'S': 1.05, 'Co': 1.26}


def molecules(sym, pos, cell):
    """Covalent graph without Co -> (adjacency list, component id per atom)."""
    n = len(sym)
    d = np.linalg.norm(mic(pos[None] - pos[:, None], cell), axis=2)
    rc = np.array([RCOV[s] for s in sym])
    bond = (d < 1.2 * (rc[:, None] + rc[None])) & (d > 0.1)
    isco = sym == 'Co'
    bond[isco] = False
    bond[:, isco] = False
    adj = [list(np.where(bond[i])[0]) for i in range(n)]
    comp = -np.ones(n, dtype=int)
    c = 0
    for i in range(n):
        if comp[i] >= 0 or isco[i]:
            continue
        stack = [i]
        comp[i] = c
        while stack:
            j = stack.pop()
            for k in adj[j]:
                if comp[k] < 0:
                    comp[k] = c
                    stack.append(k)
        c += 1
    return adj, comp


def ring_of(n_idx, sym, adj):
    """5-membered ring containing N (imidazole) -> list of atom indices, or None."""
    heavy = lambda a: [b for b in adj[a] if sym[b] != 'H']
    for a in heavy(n_idx):
        for b in heavy(a):
            if b == n_idx:
                continue
            for c in heavy(b):
                if c in (a, n_idx):
                    continue
                for d in heavy(c):
                    if d in (a, b, n_idx):
                        continue
                    if n_idx in heavy(d):
                        return [n_idx, a, b, c, d]
    return None


def angle(u, v):
    cu = np.einsum('...i,...i', u, v) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    return np.degrees(np.arccos(np.clip(cu, -1, 1)))


def analyse_run(rundir, stride):
    run = read_run(rundir)
    if run is None:
        return None
    sym, steps, xyz, cell = run
    if cell.ndim == 2:
        cell = np.repeat(cell[None], len(xyz), axis=0)
    steps, xyz, cell = steps[::stride], xyz[::stride], cell[::stride]
    t = steps * DT_PS
    p0, c0 = xyz[0], cell[0]
    adj, comp = molecules(sym, p0, c0)
    co = np.where(sym == 'Co')[0]
    nn = np.where(sym == 'N')[0]
    hh = np.where(sym == 'H')[0]
    ss = np.where(sym == 'S')[0]
    win = max(1, WIN // stride)

    def dvec(i, j):       # (F, 3) minimum-image vector i -> j
        return mic(xyz[:, j] - xyz[:, i], cell)

    donors = {}
    for c in co:
        dist = np.linalg.norm(mic(p0[nn] - p0[c], c0), axis=1)
        donors[c] = list(nn[dist < RBOND0])
    events, allbonds = [], []
    for c in co:
        for n in donors[c]:
            r = np.linalg.norm(dvec(c, n), axis=1)
            rm = running_mean(r, win)
            ring = ring_of(n, sym, adj)
            kind = 'Im' if ring is not None else 'NCS'
            allbonds.append((c, n, kind, r[0], rm.max(), rm[len(rm) // 2:].mean()))
            if rm[-max(1, len(rm) // 10):].mean() < RBREAK:
                continue
            above = rm > RBREAK
            k = len(above) - 1
            while k > 0 and above[k - 1]:
                k -= 1                       # start of the final stretch above RBREAK
            first = int(np.argmax(above))    # first excursion
            ev = dict(co=c, n=n, kind=kind, t=t, r=r, onset=t[k], first=t[first])
            # ligand geometry: lone-pair axis, centroid, orientation
            vnc = dvec(n, c)
            if ring is not None:
                nb = [ring[1], ring[-1]]                  # the two ring neighbours of N
                vn1, vn2 = dvec(n, nb[0]), dvec(n, nb[1])
                lp = -(vn1 / np.linalg.norm(vn1, axis=1)[:, None] + vn2 / np.linalg.norm(vn2, axis=1)[:, None])
                body = ring
                nrm = np.cross(vn1, vn2)
            else:                                         # N=C=S: lone pair along C->N
                cc = [a for a in adj[n] if sym[a] == 'C'][0]
                lp = -dvec(n, cc)
                body = [a for a in range(len(sym)) if comp[a] == comp[n]]
                nrm = lp
            ev['tilt'] = angle(lp, vnc)
            cen = np.mean([dvec(n, a) for a in body], axis=0)        # N -> ligand centroid
            ev['rcen'] = np.linalg.norm(vnc - cen, axis=1)
            rot = angle(nrm, np.repeat(nrm[:1], len(nrm), 0))
            ev['ringrot'] = np.minimum(rot, 180 - rot) if ring is not None else rot
            # coordination polyhedron
            others = [m for m in donors[c] if m != n]
            vs = {m: dvec(c, m) for m in donors[c]}
            if len(donors[c]) == 4:
                angs = [angle(vs[a], vs[b]) for i, a in enumerate(donors[c]) for b in donors[c][i + 1:]]
                angs = np.sort(np.array(angs).T, axis=1)
                ev['tau4'] = (360 - angs[:, -1] - angs[:, -2]) / 141.0
                ev['angsum3'] = sum(angle(vs[a], vs[b]) for i, a in enumerate(others) for b in others[i + 1:])
            else:
                # trans partner = the donor with the largest N-Co-N angle at frame 0
                tr = max(others, key=lambda m: angle(vs[n][0], vs[m][0]))
                ev['rtrans'] = np.linalg.norm(vs[tr], axis=1)
                ev['trans_kind'] = 'Im' if ring_of(tr, sym, adj) is not None else 'NCS'
                ev['ang_trans'] = angle(vs[n], vs[tr])
            # nearest atom to Co outside its own ligands (does anything take the vacated site?)
            own = {comp[m] for m in donors[c]}
            rest = np.array([a for a in range(len(sym)) if a != c and comp[a] not in own])
            dr = np.linalg.norm(mic(xyz[:, rest] - xyz[:, c][:, None], cell), axis=2)
            j = dr.argmin(axis=1)
            ev['near_r'], ev['near_el'] = dr[np.arange(len(j)), j], sym[rest[j]]
            # Co out of the plane of its other donors (4-coordinate: the three others)
            if len(donors[c]) == 4:
                a, b, d = others
                nv = np.cross(vs[b] - vs[a], vs[d] - vs[a])
                ev['coplane'] = np.abs(np.einsum('fi,fi->f', -vs[a], nv)) / np.linalg.norm(nv, axis=1)
            # freed N to the other Co (does the ligand migrate?)
            oc = [x for x in co if x != c]
            if oc:
                ev['nOtherCo'] = np.min([np.linalg.norm(dvec(n, x), axis=1) for x in oc], axis=0)
            # stacking: leaving ring centroid to the nearest other ring centroid
            if ring is not None:
                rings = []
                for m in nn:
                    if comp[m] != comp[n]:
                        rr = ring_of(m, sym, adj)
                        if rr is not None and sorted(rr) not in [sorted(x) for x in rings]:
                            rings.append(rr)
                cen_abs = xyz[:, n] + cen
                dd = [np.linalg.norm(mic(xyz[:, rr[0]] + np.mean([dvec(rr[0], q) for q in rr], axis=0)
                                         - cen_abs, cell), axis=1) for rr in rings]
                ev['stack'] = np.min(dd, axis=0)
            # closest H (other molecules) to the leaving N
            hx = hh[comp[hh] != comp[n]]
            dh = np.linalg.norm(mic(xyz[:, hx] - xyz[:, n][:, None], cell), axis=2)
            jh = dh.argmin(axis=1)
            ev['hN'] = dh[np.arange(len(jh)), jh]
            ev['hN_parent'] = [sym[adj[hx[x]][0]] if adj[hx[x]] else '?' for x in jh]
            # own N-H...S of benzimidazole
            nh = [h for a in (ring or []) if sym[a] == 'N' and a != n for h in adj[a] if sym[h] == 'H']
            if nh and len(ss):
                ds = np.linalg.norm(mic(xyz[:, ss] - xyz[:, nh[0]][:, None], cell), axis=2)
                ev['nhS'] = ds.min(axis=1)
            # cell angles
            ev['cell'] = cell
            events.append(ev)
    return dict(sym=sym, t=t, events=events, allbonds=allbonds, co=co)


def window_mean(ev, key, lo, hi):
    t = ev['t']
    m = (t >= lo) & (t < hi)
    return np.nanmean(ev[key][m]) if m.any() else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--series', default='nptaniso')
    ap.add_argument('--stride', type=int, default=5)
    ap.add_argument('--fig-tmin', type=int, default=450, help='plot only events at T >= this')
    ap.add_argument('--out', default='analysis_2026-11-16/bond_breaking')
    args = ap.parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(root, args.out)
    os.makedirs(out, exist_ok=True)

    lines, allev = [], []
    for s in SYSTEMS:
        for T in TEMPS:
            rd = os.path.join(BASE, s, f'{args.series}_{T}')
            res = analyse_run(rd, args.stride)
            if res is None:
                continue
            lab = f'{s} {T} K'
            lines.append(f'== {lab}: {len(res["t"])} frames to {res["t"][-1]:.1f} ps; '
                         f'{len(res["events"])} broken bond(s)')
            for c, n, kind, r0, rmax, r2 in res['allbonds']:
                lines.append(f'   Co{c:<3d} N{n:<3d} {kind:3s}  r0 {r0:5.2f}  max(100fs mean) {rmax:5.2f}  '
                             f'2nd-half mean {r2:5.2f}' + ('   <-- event' if any(
                                 e['co'] == c and e['n'] == n for e in res['events']) else ''))
            for ev in res['events']:
                ev['label'] = f'{lab}  Co{ev["co"]}-N{ev["n"]} ({ev["kind"]})'
                o = ev['onset']
                b, a = (max(0, o - 2.0), o), (o + 0.5, ev['t'][-1] + 1e-9)
                lines.append(f'   EVENT Co{ev["co"]}-N{ev["n"]} {ev["kind"]}: first > {RBREAK} A at '
                             f'{ev["first"]:.2f} ps, stays above from {o:.2f} ps')
                for key, fmt in (('r', '%.2f A'), ('rcen', '%.2f A'), ('tilt', '%.0f deg'),
                                 ('ringrot', '%.0f deg'), ('rtrans', '%.2f A'), ('ang_trans', '%.0f deg'),
                                 ('tau4', '%.2f'), ('angsum3', '%.0f deg'), ('near_r', '%.2f A'),
                                 ('hN', '%.2f A'), ('nhS', '%.2f A'), ('coplane', '%.2f A'),
                                 ('nOtherCo', '%.2f A'), ('stack', '%.2f A')):
                    if key in ev:
                        first2 = (ev['t'][0], ev['t'][0] + 0.1)
                        lines.append(f'      {key:9s} 100 K {fmt % window_mean(ev, key, *first2):>9s}   '
                                     f'2 ps before {fmt % window_mean(ev, key, *b):>9s}   '
                                     f'after {fmt % window_mean(ev, key, *a):>9s}')
                m = ev['t'] >= o + 0.5
                from numpy.linalg import norm
                def abc(cc):
                    L = norm(cc, axis=-1)
                    ang = [np.degrees(np.arccos(np.einsum('fi,fi->f', cc[:, i], cc[:, k]) / (L[:, i] * L[:, k])))
                           for i, k in ((1, 2), (0, 2), (0, 1))]
                    return np.c_[L, np.array(ang).T].mean(axis=0)
                tb = (ev['t'] < o) & (ev['t'] >= max(0, o - 2))
                tb = tb if tb.any() else ev['t'] <= ev['t'][0]
                lines.append('      cell a b c al be ga  before ' + ' '.join(f'{x:6.2f}' for x in abc(ev['cell'][tb])) +
                             '   after ' + ' '.join(f'{x:6.2f}' for x in abc(ev['cell'][m])))
                el, cnt = np.unique(ev['near_el'][m], return_counts=True)
                lines.append('      nearest non-donor to Co after: ' +
                             ', '.join(f'{e} {100 * k / m.sum():.0f}%' for e, k in zip(el, cnt)))
                par, cnt = np.unique(np.array(ev['hN_parent'])[m], return_counts=True)
                lines.append('      closest H to freed N belongs to: ' +
                             ', '.join(f'{e}-H {100 * k / m.sum():.0f}%' for e, k in zip(par, cnt)))
                ev['T'] = T
                allev.append(ev)
    txt = '\n'.join(lines)
    print(txt)
    with open(os.path.join(out, f'events_{args.series}.txt'), 'w') as fp:
        fp.write(txt + '\n')

    allev = [e for e in allev if e['T'] >= args.fig_tmin]
    if not allev:
        return
    cols = [('r', 'Co–N (Å)'), ('rcen', 'Co–ring centroid (Å)'), ('tilt', 'off lone-pair axis (°)'),
            ('poly', 'polyhedron'), ('stack', 'ring–nearest ring centroid (Å)'), ('hN', 'H···freed N (Å)')]
    fig, axs = plt.subplots(len(allev), len(cols), figsize=(2.6 * len(cols), 1.7 * len(allev) + 0.6),
                            squeeze=False, sharex='row')
    for i, ev in enumerate(allev):
        w = max(1, WIN // args.stride)
        for j, (key, lab) in enumerate(cols):
            ax = axs[i, j]
            if key == 'poly':
                if 'tau4' in ev:
                    ax.plot(ev['t'], running_mean(ev['tau4'], w), lw=1, color='#2a78d6', label='τ4')
                    ax2 = ax.twinx()
                    ax2.plot(ev['t'], running_mean(ev['angsum3'], w), lw=1, color='#eb6834')
                    ax2.axhline(360, ls=':', color='#eb6834', lw=0.6)
                    ax2.tick_params(labelsize=6, colors='#eb6834')
                    ax.set_ylabel('τ4 (blue) / Σ3 angles (°, orange)', fontsize=6)
                else:
                    ax.plot(ev['t'], running_mean(ev['ringrot'], w), lw=1, color='#2a78d6')
                    ax.set_ylabel('ring rotation from start (°)', fontsize=6)
            else:
                ax.plot(ev['t'], ev[key], lw=0.3, color='#c3c2b7')
                ax.plot(ev['t'], running_mean(ev[key], w), lw=1, color='#0b0b0b')
                if key == 'r' and 'rtrans' in ev:
                    ax.plot(ev['t'], running_mean(ev['rtrans'], w), lw=1, color='#2a78d6')
                    lab = 'Co–N (k) / trans Co–N (b), Å'
                if key == 'hN' and 'nhS' in ev:
                    ax.plot(ev['t'], running_mean(ev['nhS'], w), lw=1, color='#2a78d6')
                    lab = 'H···freed N (k) / own N–H···S (b)'
                if i == 0 or True:
                    ax.set_ylabel(lab, fontsize=6)
            ax.axvline(ev['onset'], color='#eb6834', lw=0.8, ls='--')
            ax.tick_params(labelsize=6)
            if i == len(allev) - 1:
                ax.set_xlabel('time (ps)', fontsize=7)
        axs[i, 0].set_title(ev['label'], fontsize=7, loc='left')
    fig.suptitle(f'Long Co–N bonds, {args.series}: dashed = start of the final stretch above {RBREAK} Å',
                 fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out, f'fig_bond_break_{args.series}.{ext}'), dpi=150)
    print('wrote', out)


if __name__ == '__main__':
    main()
