# Long Co–N bonds in the `nptaniso` runs (Co_1MIm and CoSCN2bIm2, 450–500 K)

Script: `../../bond_break_study.py` (default `--series nptaniso`; `--fig-tmin 450` for the figure).
- `events_nptaniso.txt` lists every Co–N bond of every run. A bond is an event when its 100 fs running mean ends above 2.6 Å. For each event the file gives the descriptors at three points: the 100 K start frame, the 2 ps before the break, and after the break.
- `fig_bond_break_nptaniso.png/.pdf` shows the 450/500 K events.

## Co_1MIm: the bond breaks in two steps

1. **Step 1, at 0.8–1.7 ps (axial elongation).** One trans pair of the six donors lengthens from 2.0–2.2 Å to about 3 Å, and both bonds of the pair move together.
   - This happens at **every** temperature:
     - 300 K: the NCS pair on Co1;
     - 350 and 400 K: Im pairs, on Co0 at 350 K and on both Co at 400 K.
   - The ligand stays on the Co lone-pair axis (tilt 10–20°, ring not rotated).
   - This is the closed-shell Jahn–Teller axis described in `../../analysis_2026-10-29/spin_check/`.
2. **Step 2, only at 450/500 K, at 4–8 ps.** On one Co, the elongated Im pair leaves completely, to 4.2–4.5 Å.
   - The Co–ring-centroid distance goes from 4.2 to 5.3 Å.
   - The ring rotates by 45–60° and its lone pair no longer points at the Co (tilt 30–37°).
   - **At the same moment the ring stacks on a neighbouring imidazole:** the centroid–centroid distance drops from 4.5–4.9 to 3.6–3.8 Å.
   - Nothing takes the empty site: the nearest atom from another molecule is an H at 3.6 Å. The Co is left square-planar CoN4 (2 NCS + 2 Im).
   - The other Co keeps its elongated pair at about 2.5 Å.
   - The cell is not the trigger: *c* grows by 1–1.9 Å in the first 2–3 ps at 400 K as well, where step 2 does not happen. At 400 K one bond reached 3.5 Å at 16.4 ps, so step 2 may only be slower there.
   - Interpretation: once the axial bond has gone, the ligand is held only by packing and dispersion. At ≥ 450 K it has enough energy to rotate into a π-stacked position, which is a better non-bonded contact than staying 3 Å from the Co.

## CoSCN2bIm2: the Co flattens to a 3+1 geometry

- **Which bonds:** one benzimidazole on each of two of the four Co. At 450 K these are Co2 and Co3; at 500 K, Co0 and Co1.
- **When:** at 0.6–0.9 ps, while the system is still heating up in the warm-up. Both bonds of a pair break within about 0.2 ps of each other.
- **The Co moves, not the ligand.** The Co drops into the plane of its other three donors:
  - Co height above that plane: 0.4–0.5 → 0.06 Å;
  - sum of the three N–Co–N angles: 340° → 359°;
  - τ4: 0.55 → 0.72.
  - This happens in the same frames where Co–N crosses 2.6 Å.
  - The benzimidazole only translates. Co–N goes from 2.0 to 2.9 Å, but the tilt stays at 10–15° and the ring does not rotate (8–15°).
- **Hydrogen bond:** the ligand's own N–H···S bond breaks 0.1–0.6 ps **after** the Co–N bond (2.3 → 3.3 Å). It is a consequence of the break, not its cause.
- **Cell:** the change in shape (*b* +1.3 Å, *a* −0.4 Å, β 96 → 101°) follows over the next 1–4 ps.
- **Persistence:** nothing enters the fourth site (nearest atom: H at 3.1 Å), and the 3+1 geometry lasts for the rest of the run, 14 ps. At 400 K one bond reached 2.54 Å and came back.
- The model already has these Co flattened at 100 K (τ4 about 0.55, not 1.0 as in a tetrahedron).

## Both systems

- In both systems the starting point is a **Co(II) coordination that the closed-shell model makes unstable**: an elongated axial pair in the octahedron, a flattened tetrahedron.
- Temperature only decides whether the system crosses into the distorted state (CoSCN2bIm2), or whether the loosened ligand can then wander off (Co_1MIm).
- **The dipole treatment matters.** With improved dipoles (`nptanisoimp`):
  - CoSCN2bIm2 stays CN 4 on all four Co at every temperature (maximum Co–N 2.59 Å at 500 K);
  - Co_1MIm keeps the axial elongation (about 3.1 Å) but no ligand goes beyond 3.9 Å.
- Both points suggest a model artefact rather than real chemistry. A spin-polarised (S = 3/2) single-point check of the before/after geometries would settle it.
