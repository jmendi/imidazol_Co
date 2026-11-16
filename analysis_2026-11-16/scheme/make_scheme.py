#!/usr/bin/env python
"""Cartoon of the Co-N bond events in the nptaniso runs (Co_1MIm, CoSCN2bIm2).

Numbers come from ../bond_breaking/ and ../pi_stacking/.  Rings are drawn
edge-on (bars) so that pi stacking reads as two parallel bars.

    python make_scheme.py      -> scheme_nptaniso.png / .pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Polygon

C_CO, C_N, C_S = '#3b6fb6', '#2a78d6', '#d9a400'
C_NCS, C_IM, C_NB = '#2a78d6', '#eb6834', '#b8b6ad'
INK, INK2, MUTED, PANEL = '#0b0b0b', '#52514e', '#898781', '#f3f2ee'
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9})

fig = plt.figure(figsize=(16, 11.0))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 160)
ax.set_ylim(-6, 104)
ax.set_aspect('equal')
ax.axis('off')


def panel(x0, y0, w, h, title, sub=None):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle='round,pad=0.4,rounding_size=1.5',
                                fc=PANEL, ec='#d6d4cc', lw=0.8))
    ax.text(x0 + 1.2, y0 + h - 1.6, title, fontsize=10, weight='bold', color=INK, va='top')
    if sub:
        ax.text(x0 + 1.2, y0 + h - 4.6, sub, fontsize=8.2, color=INK2, va='top')


def atom(x, y, r, c, label=None, fs=8, tc='white'):
    ax.add_patch(Circle((x, y), r, fc=c, ec='white', lw=0.8, zorder=5))
    if label:
        ax.text(x, y, label, ha='center', va='center', fontsize=fs, color=tc, zorder=6, weight='bold')


def bond(p, q, style='-', c=INK2, lw=1.6, z=3):
    ax.plot([p[0], q[0]], [p[1], q[1]], ls=style, color=c, lw=lw, zorder=z, solid_capstyle='round')


def ring(cx, cy, ang, length, c, lw=7, alpha=1.0, z=4):
    """Ring seen edge-on: a thick bar centred at (cx, cy), direction ang (deg)."""
    a = np.radians(ang)
    dx, dy = 0.5 * length * np.cos(a), 0.5 * length * np.sin(a)
    ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color=c, lw=lw, alpha=alpha,
            solid_capstyle='round', zorder=z)


def ncs(x, y, direction):
    """N=C=S along +-x from the N at (x, y)."""
    s = 1 if direction > 0 else -1
    atom(x, y, 0.9, C_N, 'N', 6.5)
    bond((x, y), (x + s * 3.0, y), c=C_NCS, lw=2.2)
    atom(x + s * 3.0, y, 0.7, '#5f5e5a')
    bond((x + s * 3.0, y), (x + s * 5.8, y), c=C_NCS, lw=2.2)
    atom(x + s * 5.8, y, 1.0, C_S, 'S', 6.5, tc=INK)


def dim(p, q, text, off=(0, 0), c=INK, fs=8, side='right'):
    ax.annotate('', p, q, arrowprops=dict(arrowstyle='<->', color=c, lw=0.8, shrinkA=0, shrinkB=0), zorder=7)
    mx, my = (p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1]
    ax.text(mx, my, text, fontsize=fs, color=c, ha='left' if side == 'right' else 'right', va='center', zorder=8)


def arrow(x0, x1, y, top, bottom):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle='-|>,head_width=5,head_length=8',
                                 color=INK, lw=2, zorder=2))
    ax.text((x0 + x1) / 2, y + 2.2, top, ha='center', va='bottom', fontsize=8.6, weight='bold', color=INK)
    ax.text((x0 + x1) / 2, y - 2.2, bottom, ha='center', va='top', fontsize=7.8, color=INK2, linespacing=1.3)


# ---------------------------------------------------------------- Co_1MIm
ax.text(2, 101.5, 'Co_1MIm  —  Co(NCS)$_2$(1-MeIm)$_4$, octahedral', fontsize=12, weight='bold', color=INK, va='top')
Y0, H, W = 53, 44, 40
X = [2, 60, 118]

# stage 1
panel(X[0], Y0, W, H, '1. Start (100 K frame)', 'six-coordinate, all Co–N 2.0–2.2 Å')
cx, cy = X[0] + 20, Y0 + 19
for s in (-1, 1):
    bond((cx, cy), (cx + s * 4.5, cy))
    ncs(cx + s * 4.5, cy, s)
for (dx, dy) in ((-3.2, -2.6), (3.2, 2.6)):        # equatorial Im (perspective)
    bond((cx, cy), (cx + dx, cy + dy), c=MUTED, lw=1.2)
    atom(cx + dx, cy + dy, 0.75, C_N)
    ring(cx + dx * 1.9, cy + dy * 1.9, 39, 3.4, C_IM, lw=5, alpha=0.55)
for s in (1, -1):                                  # axial Im pair
    bond((cx, cy), (cx, cy + s * 4.6))
    atom(cx, cy + s * 4.6, 0.9, C_N, 'N', 6.5)
    ring(cx, cy + s * 8.4, 90, 5.2, C_IM)
atom(cx, cy, 1.6, C_CO, 'Co', 8)
dim((cx - 2.6, cy), (cx - 2.6, cy + 4.6), '2.0–2.2 Å', off=(-0.6, 0), side='left')
ring(cx + 12.5, cy + 12.5, 0, 5.0, C_NB, lw=7)                    # neighbour Im, edge-to-face
ax.text(cx + 12.5, cy + 15.3, 'Im of neighbour Co', fontsize=7, ha='center', color=MUTED)
ax.text(cx + 8.0, cy + 7.5, 'edge-to-face\n64–80°, 5 Å\nno π stacking', fontsize=7, color=INK2, ha='center')
ax.text(X[0] + 1.2, Y0 + 2.2, 'Closed-shell model puts one electron in d$_{z^2}$:\none axis is already destined to elongate',
        fontsize=7.5, color=INK2, va='bottom')

# stage 2
panel(X[1], Y0, W, H, '2. Axial elongation  ("4+2")', 'every T, 300–500 K')
cx, cy = X[1] + 20, Y0 + 19
for s in (-1, 1):
    bond((cx, cy), (cx + s * 4.5, cy))
    ncs(cx + s * 4.5, cy, s)
for (dx, dy) in ((-3.2, -2.6), (3.2, 2.6)):
    bond((cx, cy), (cx + dx, cy + dy), c=MUTED, lw=1.2)
    atom(cx + dx, cy + dy, 0.75, C_N)
    ring(cx + dx * 1.9, cy + dy * 1.9, 39, 3.4, C_IM, lw=5, alpha=0.55)
for s in (1, -1):
    bond((cx, cy), (cx, cy + s * 6.6), style=(0, (2, 1.5)), c=C_IM, lw=1.4)
    atom(cx, cy + s * 6.6, 0.9, C_N, 'N', 6.5)
    ring(cx, cy + s * 10.4, 90, 5.2, C_IM)
atom(cx, cy, 1.6, C_CO, 'Co', 8)
dim((cx - 2.6, cy), (cx - 2.6, cy + 6.6), '≈ 3.0 Å', off=(-0.6, -1.5), side='left')
ax.text(cx - 3.2, cy + 12.5, 'trans pair\nmoves together', fontsize=7.2, color=INK2, ha='right')
ring(cx + 11.5, cy + 12.5, 21, 5.0, C_NB, lw=7)
ax.text(cx + 11.5, cy + 15.6, '400 K: slipped contact\n3.95 Å, 21°', fontsize=7, ha='center', color=INK2)
ax.text(X[1] + 1.2, Y0 + 2.2, 'Ligand still points its N at Co (tilt 10–20°).\n300 K: the NCS pair elongates instead.',
        fontsize=7.5, color=INK2, va='bottom')

# stage 3
panel(X[2], Y0, W, H, '3. Axial pair leaves, π-stacks', 'only 450 and 500 K, one Co of two')
cx, cy = X[2] + 12, Y0 + 19
for s in (-1, 1):
    bond((cx, cy), (cx + s * 4.5, cy))
    ncs(cx + s * 4.5, cy, s)
for (dx, dy) in ((-3.2, -2.6), (3.2, 2.6)):
    bond((cx, cy), (cx + dx, cy + dy), c=MUTED, lw=1.2)
    atom(cx + dx, cy + dy, 0.75, C_N)
    ring(cx + dx * 1.9, cy + dy * 1.9, 39, 3.4, C_IM, lw=5, alpha=0.55)
atom(cx, cy, 1.6, C_CO, 'Co', 8)
ax.text(cx, cy - 4.2, 'square-planar CoN$_4$\nvacant sites stay empty', fontsize=7, ha='center', va='top', color=INK2)
# departed rings, rotated ~60 deg, stacked on a neighbour ring
for s in (1, -1):
    ny = cy + s * 9.6
    ring(cx + 3.0, ny, 30, 5.2, C_IM)
# explicit neighbour rings (parallel, offset along the ring normal)
nrm = np.array([-np.sin(np.radians(30)), np.cos(np.radians(30))])
for s in (1, -1):
    c0 = np.array([cx + 3.0, cy + s * 9.6])
    c1 = c0 + 3.6 * nrm * s
    ring(*c1, 30, 5.2, C_NB)
    dim(tuple(c0 + 0.0), tuple(c1), '3.7 Å' if s > 0 else '', off=(0.8, 0.4))
bond((cx, cy), (cx + 0.5, cy + 6.2), style=':', c=MUTED, lw=1)
bond((cx, cy), (cx + 0.5, cy - 6.2), style=':', c=MUTED, lw=1)
ax.text(cx - 1.0, cy + 6.0, '4.2–4.5 Å', fontsize=7.5, color=INK, ha='right')
ax.text(X[2] + 26.5, cy + 3, 'ring rotates ≈60°,\nN no longer points\nat Co (tilt 30–37°)\n\nstacks face-to-face\non an Im of the\nother Co (3.7 Å, 12–15°);\nthe two leavers also\nstack on each other\n(7.8 → 3.9 Å)',
        fontsize=7.2, color=INK2, va='center')
ax.text(X[2] + 1.2, Y0 + 2.2, 'Leaving and stacking happen together:\na weak 3 Å Co–N bond is traded for π stacking.',
        fontsize=7.5, color=INK2, va='bottom')

arrow(X[0] + W + 1.2, X[1] - 1.2, Y0 + 24, 'heating', 'all T\n≈ 1 ps')
arrow(X[1] + W + 1.2, X[2] - 1.2, Y0 + 24, '≥ 450 K', '4–8 ps (450 K)\n2–5 ps (500 K)\n400 K: 1 bond\nreaches 3.5 Å\nat 16 ps')

# ---------------------------------------------------------------- CoSCN2bIm2
ax.text(2, 50.5, 'CoSCN2bIm2  —  Co(NCS)$_2$(benzimidazole)$_2$, tetrahedral', fontsize=12, weight='bold', color=INK, va='top')
Y1 = 2


def cobim(cx, cy, h_co, r4, hb, hb_txt, stack_txt, broken=False):
    """Side view: three donors in a plane (ellipse), Co above it by h_co, 4th N above Co."""
    from matplotlib.patches import Ellipse
    ax.add_patch(Ellipse((cx, cy), 15, 4.0, fc='#e6e4dc', ec='#c3c2b7', lw=0.8, zorder=1))
    pts = [(cx - 6.0, cy - 0.6), (cx + 5.0, cy - 1.2), (cx + 3.9, cy + 1.5)]
    co = (cx, cy + h_co)
    for k, p in enumerate(pts):
        bond(co, p)
        atom(*p, 0.85, C_N, 'N', 6)
    ax.text(cx + 8.4, cy - 0.4, 'plane of the\nother 3 N\n(2 NCS + 1 bIm)', fontsize=6.6, color=MUTED, va='center')
    n4 = (cx, co[1] + r4)
    bond(co, n4, style=(0, (2, 1.5)) if broken else '-', c=C_IM if broken else INK2, lw=1.6)
    atom(*n4, 0.9, C_N, 'N', 6.5)
    # benzimidazole edge-on: 5-ring + 6-ring, N-H at the far end
    ring(cx, n4[1] + 3.0, 90, 4.2, C_IM, lw=7)
    ring(cx, n4[1] + 7.0, 90, 3.6, '#c9763f', lw=7)
    h = (cx + 2.0, n4[1] + 3.6)
    nh = (cx + 0.9, n4[1] + 3.3)
    bond(nh, h, c=INK2, lw=1.2)
    atom(*h, 0.55, '#ffffff')
    ax.add_patch(Circle(h, 0.55, fc='white', ec=INK2, lw=0.8, zorder=6))
    s_pos = (h[0] + hb, h[1] + 0.4)
    atom(*s_pos, 1.0, C_S, 'S', 6.5, tc=INK)
    bond(h, s_pos, style=':', c=C_S, lw=1.6)
    ax.text((h[0] + s_pos[0]) / 2, h[1] + 1.6, hb_txt, fontsize=7, ha='center', color=INK2)
    # pi column partner
    ring(cx - 3.5, n4[1] + 3.6, 90, 4.2, C_NB, lw=7)
    ax.text(cx - 5.0, n4[1] + 3.6, stack_txt, fontsize=6.8, color=INK2, ha='right', va='center')
    atom(*co, 1.6, C_CO, 'Co', 8)
    return co, n4


panel(X[0], Y1, W, H, '1. Start (100 K frame)', 'already a flattened tetrahedron, τ$_4$ ≈ 0.55')
co, n4 = cobim(X[0] + 17, Y1 + 13, 2.2, 4.4, 4.2, 'N–H···S 2.3 Å', 'π column\nalong a\n3.4–3.8 Å')
dim((co[0] - 9.8, Y1 + 13), (co[0] - 9.8, co[1]), 'Co 0.45 Å\nabove plane', off=(-0.5, 0), side='left', fs=7)
dim((co[0] + 2.4, co[1]), (co[0] + 2.4, n4[1]), '2.0 Å', off=(0.5, 0))
ax.text(X[0] + 1.2, Y1 + 2.2, 'Σ of the 3 N–Co–N angles ≈ 340°', fontsize=7.5, color=INK2, va='bottom')

panel(X[1], Y1, W, H, '2. Co drops into the plane  ("3+1")', '450 and 500 K, 2 of 4 Co, during warm-up')
co, n4 = cobim(X[1] + 17, Y1 + 13, 0.0, 6.4, 4.2, 'N–H···S 2.3 Å', 'π column\nunchanged')
dim((co[0] + 2.4, co[1]), (co[0] + 2.4, n4[1]), '2.9 Å', off=(0.5, 0))
ax.annotate('', (co[0] - 2.6, co[1] - 0.2), (co[0] - 2.6, co[1] + 2.4),
            arrowprops=dict(arrowstyle='-|>', color=C_CO, lw=1.6))
ax.text(co[0] - 3.2, co[1] + 1.0, 'Co moves,\nnot the ligand', fontsize=7, color=C_CO, ha='right', va='center')
ax.text(X[1] + 1.2, Y1 + 2.2, 'Co 0.06 Å from plane, Σ angles 359°, τ$_4$ 0.72.\nRing keeps pointing at Co (tilt 10–15°), no rotation.',
        fontsize=7.5, color=INK2, va='bottom')

panel(X[2], Y1, W, H, '3. The N–H···S hydrogen bond breaks', 'the ligand moved 0.3–0.7 Å away')
co, n4 = cobim(X[2] + 14, Y1 + 13, 0.0, 6.4, 6.0, 'N–H···S 3.3 Å', 'π kept\n3.4 / 3.8 Å')
ax.text(X[2] + 1.2, Y1 + 2.2, 'The 3+1 geometry lasts the whole run (14 ps);\nnothing enters the 4th site (nearest H 3.1 Å).',
        fontsize=7.5, color=INK2, va='bottom')

arrow(X[0] + W + 1.2, X[1] - 1.2, Y1 + 24, '≥ 450 K', '0.6–0.9 ps\n(400 K: one bond\nto 2.54 Å, returns)')
arrow(X[1] + W + 1.2, X[2] - 1.2, Y1 + 24, 'then', '0.1–0.6 ps later')

ax.text(80, -1.5,
         'Flexible-cell runs with legacy dipoles (nptaniso). With improved dipoles (nptanisoimp) CoSCN2bIm2 keeps four bonds at every T '
         'and the Co_1MIm axial pair stays at ≈3 Å (≤ 3.9 Å).\nBoth start from a Co(II) geometry the closed-shell model makes unstable; a spin-polarised '
         '(S = 3/2) check is needed.',
         ha='center', va='top', fontsize=8, color=INK2)

here = os.path.dirname(os.path.abspath(__file__))
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(here, f'scheme_nptaniso.{ext}'), dpi=170, facecolor='#fcfcfb')
print('ok')
