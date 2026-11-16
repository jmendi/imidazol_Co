# Co–azole crystals: flexible-cell Fireball MD at 300–500 K (nptaniso)

Molecular dynamics of three Co(II) thiocyanate–azole crystals, run with Fireball (fireballpy) at 300, 350, 400, 450 and 500 K:
- **Co_1MIm**: Co(NCS)₂(1-methylimidazole)₄, octahedral;
- **Co_12dMIm**: the 1,2-dimethylimidazole analogue, tetrahedral;
- **CoSCN2bIm2**: Co(NCS)₂(benzimidazole)₂, tetrahedral.

This repository contains the **nptaniso** series only:
- constant pressure (P = 0) with a fully flexible cell;
- two-body D3 dispersion, legacy dipoles;
- Langevin thermostat, 1 fs time step;
- 1 ps warm-up from a 100 K structure, then production. The runs are being extended to 30 ps; this snapshot (2026-11-16) has 13–16 ps.

## Contents

| path | what |
|---|---|
| `analysis_2026-11-16/scheme/scheme_nptaniso.png` | summary scheme of what happens to the Co coordination |
| `analysis_2026-11-16/bond_breaking/` | origin of the long Co–N bonds (README, per-bond table, figure) |
| `analysis_2026-11-16/pi_stacking/` | π–π contacts between the azole rings (README, tables, figures) |
| `analysis_2026-11-16/coN/`, `CN/` | Co–N distances and coordination numbers against time |
| `analysis_2026-11-16/cells/`, `status/` | cell parameters, volume, pressure and temperature against time |
| `analysis_2026-11-16/traj_viewer/` | 3D trajectory viewer (see below) |
| `cif_runs/<system>/nptaniso_<T>/` | run folders: inputs, `energy.dat`, `cell.dat`/`cellmat.dat` |
| `*.cif` | experimental crystal structures |
| `*.py`, `*.sh` | MD driver, job scripts and the analysis scripts used here; each script's header says how to run it |

## Main results

- **Co_1MIm:** one trans pair of imidazoles lengthens to about 3 Å at every T. At 450 and 500 K that pair leaves one Co (Co–N 4.2–4.5 Å), rotates and π-stacks with neighbouring imidazoles.
- **CoSCN2bIm2:** at 450 and 500 K two of the four Co drop into the plane of three of their N ("3+1"). The fourth benzimidazole stays 2.9 Å away and its N–H···S hydrogen bond breaks.
- **Co_12dMIm:** keeps all four ligands at every T.
- **No melting** in any system.
- Both distortions likely come from the closed-shell (spin-restricted) description of high-spin Co(II); see `analysis_2026-11-16/bond_breaking/README.md`.

## 3D viewer

**Online:** https://jmendi.github.io/imidazol_Co/analysis_2026-11-16/traj_viewer/

To run it locally instead:

```
cd analysis_2026-11-16/traj_viewer
python -m http.server 8000
```
Then open http://localhost:8000. The viewer shows every run at one frame per 40 fs, with molecules kept whole, the cell box, periodic images and the long Co–N bonds against time.

## Trajectories

**Download:** https://github.com/jmendi/imidazol_Co/releases/tag/traj-2026-11-16

The full trajectories (one frame per fs) are attached to that release, compressed with xz. It holds 30 files: warm-up and production for each of the 15 runs. Each asset is named after its path with `/` replaced by `__`. For example:

```
gh release download traj-2026-11-16 -R jmendi/imidazol_Co -p 'Co_1MIm__nptaniso_450__*'
xz -d Co_1MIm__nptaniso_450__prod__answer.xyz.xz
```

- Each frame's comment line holds its MD step (fs).
- The cell of each frame is in `cellmat.dat`, in the same run folder. Each row is the step followed by the cell vectors a1, a2, a3 in Å.
