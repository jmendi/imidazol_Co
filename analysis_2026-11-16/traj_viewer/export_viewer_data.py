#!/usr/bin/env python
"""Export nptaniso trajectories for the web 3D viewer (artifact "Co–N Trajectory Viewer").

For each run: every STRIDE-th frame (warm-up + production, steps in fs), the
cell of each frame, and positions with molecules kept whole and continuous in
time (each covalent fragment is unwrapped about its anchor atom, and the anchor
takes the periodic image closest to where it was in the previous frame; frame 0
follows the Co-N bonds so every complex starts in one piece).  Positions are
stored as int16 in units of 0.01 A relative to the frame-0 centroid, base64.

Also stored: the Co-N bonds worth watching (final 100-fs mean > 2.4 A or
maximum > 2.45 A) and their distance at every exported frame.

    python export_viewer_data.py OUTDIR [--stride 40]
"""
import argparse
import base64
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from fig_coN_heat import read_run, mic, running_mean, BASE, DT_PS    # noqa: E402
from bond_break_study import molecules                               # noqa: E402

SYSTEMS = ('Co_1MIm', 'Co_12dMIm', 'CoSCN2bIm2')
TEMPS = (300, 350, 400, 450, 500)


def export(system, T, stride):
    sym, steps, xyz, cell = read_run(os.path.join(BASE, system, f'nptaniso_{T}'))
    if cell.ndim == 2:
        cell = np.repeat(cell[None], len(xyz), axis=0)
    co = np.where(sym == 'Co')[0]
    nn = np.where(sym == 'N')[0]
    # Co-N distances on the full trajectory (for the watched bonds)
    d0 = np.linalg.norm(mic(xyz[0][nn] - xyz[0][co][:, None], cell[0]), axis=2)
    bonds = [(int(c), int(n)) for i, c in enumerate(co) for j, n in enumerate(nn) if d0[i, j] < 3.2]
    watched, rcurves = [], []
    for c, n in bonds:
        r = np.linalg.norm(mic(xyz[:, n] - xyz[:, c], cell), axis=1)
        rm = running_mean(r, 100)
        if rm[-len(rm) // 10:].mean() > 2.4 or rm.max() > 2.45:
            watched.append((c, n))
            rcurves.append(rm[::stride])
    steps, xyz, cell = steps[::stride], xyz[::stride], cell[::stride]
    F, N = xyz.shape[:2]

    adj, comp = molecules(sym, xyz[0], cell[0])
    frags = {}
    for a in range(N):
        frags.setdefault(int(comp[a]) if comp[a] >= 0 else -1 - a, []).append(a)   # Co atoms alone
    # frame 0: BFS over covalent + Co-N bonds so complexes are whole
    out = np.empty_like(xyz)
    nbr = [list(adj[a]) for a in range(N)]
    for c, n in bonds:
        if np.linalg.norm(mic(xyz[0][n] - xyz[0][c], cell[0])) < 2.6:
            nbr[c].append(n)
            nbr[n].append(c)
    seen = np.zeros(N, bool)
    for a0 in list(co) + list(range(N)):
        if seen[a0]:
            continue
        out[0, a0] = xyz[0, a0]
        seen[a0] = True
        stack = [a0]
        while stack:
            i = stack.pop()
            for j in nbr[i]:
                if not seen[j]:
                    out[0, j] = out[0, i] + mic(xyz[0, j] - xyz[0, i], cell[0])
                    seen[j] = True
                    stack.append(j)
    # later frames: each fragment rigidly unwrapped about its anchor, anchor continuous in time
    for f in range(1, F):
        for atoms in frags.values():
            a0 = atoms[0]
            prev = out[f - 1, a0]
            out[f, a0] = prev + mic(xyz[f, a0] - prev, cell[f])
            for a in atoms[1:]:
                out[f, a] = out[f, a0] + mic(xyz[f, a] - xyz[f, a0], cell[f])
    origin = out[0].mean(axis=0)
    q = np.round((out - origin) * 100).astype('<i2')
    lo = -origin  # cell box drawn from this corner (cell origin relative to the centroid)
    return dict(system=system, T=T, natoms=int(N), nframes=int(F), sym=list(map(str, sym)),
                t=[round(float(s) * DT_PS, 3) for s in steps],
                cells=[[round(float(x), 3) for x in c.ravel()] for c in cell],
                origin=[round(float(x), 3) for x in lo],
                pos=base64.b64encode(q.tobytes()).decode(),
                watched=[dict(co=c, n=n, label=f'Co{list(co).index(c)}–N{n}',
                              r=[round(float(x), 3) for x in rc]) for (c, n), rc in zip(watched, rcurves)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir')
    ap.add_argument('--stride', type=int, default=40)
    ap.add_argument('--systems', nargs='+', default=list(SYSTEMS),
                    help='systems to (re)export; the index always lists every file present')
    ap.add_argument('--temps', nargs='+', type=int, default=list(TEMPS))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    for s in args.systems:
        for T in args.temps:
            d = export(s, T, args.stride)
            name = f'{s}_{T}.json'
            with open(os.path.join(args.outdir, name), 'w') as fp:
                json.dump(d, fp, separators=(',', ':'))
            print(name, d['nframes'], 'frames', [w['label'] for w in d['watched']],
                  f'{os.path.getsize(os.path.join(args.outdir, name)) / 1e6:.2f} MB')
    index = []
    for s in SYSTEMS:
        for T in TEMPS:
            path = os.path.join(args.outdir, f'{s}_{T}.json')
            if not os.path.isfile(path):
                continue
            with open(path) as fp:
                d = json.load(fp)
            index.append(dict(file=f'{s}_{T}.json', system=s, T=T, nframes=d['nframes'], tmax=d['t'][-1],
                              watched=[w['label'] for w in d['watched']]))
    with open(os.path.join(args.outdir, 'index.json'), 'w') as fp:
        json.dump(index, fp, indent=1)


if __name__ == '__main__':
    main()
