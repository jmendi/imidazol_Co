# π–π contacts between the azole ligands (`nptaniso`)

Script: `../../pi_stacking_study.py`. Run it as `--system CoSCN2bIm2` (the default) or `--system Co_1MIm`; `--series nptanisoimp` analyses the other flexible-cell series.

Method:
- Each ligand gets a ring plane and ring centroid(s): the 5- and 6-ring of benzimidazole, the 5-ring of 1-methylimidazole.
- A contact is two centroids within 4.2 Å with planes at most 30° apart. Periodic images are included, so a ligand stacking with its own image along *a* counts.
- Ligands whose Co–N bond ends above 2.6 Å are marked `*`; above 3.5 Å, `**`.
- Output: `pi_<system>_nptaniso.txt` (every contact) and `fig_pi_<system>_nptaniso.png/.pdf`.

## Co_1MIm: stacking appears with the bond breaking

- **At 100 K there is no π stacking.** Neighbouring imidazoles are edge-to-face (64–80°, 4.9–5.1 Å apart).
- **300 and 350 K:** no contacts.
- **400 K:**
  - The axial imidazoles elongated to about 2.9 Å (L2, L3) form slipped contacts with a neighbour along *a*: d_cc 3.95 Å, 21°, slip 2.0 Å.
  - These contacts are present for about 80 % of the time; on average 1.6 per cell.
- **450 and 500 K:**
  - The two imidazoles that leave their Co (`**`, at 4.2–4.5 Å) each stack face-to-face with an imidazole on the neighbouring Co: d_cc 3.67–3.72 Å, 12–15°, slip 1.0–1.3 Å.
  - They also stack with each other: 7.8 Å apart at 100 K, 3.85–3.91 Å now.
  - The result is a stacked triad, with on average 2.8 contacts per cell.
  - The contacts form at 5–8 ps (450 K) and 3.5–5 ps (500 K). These are the same times at which Co–N passes 4 Å and the rings rotate by about 60° (`../bond_breaking/`).
- The departure and the stacking are one event. The ligand trades its weak axial Co–N bond (about 3 Å) for π stacking.

## CoSCN2bIm2: the stacks stay; the extra columns at 300/350 K come from the cell shape

- **At 100 K** the benzimidazoles L0/L2 and L1/L3 already form columns along *a*. Each ligand stacks with a partner and with that partner's *a*-image (3.7–4.2 Å). That gives 4 contacts per cell.
  - L0 and L1 belong to Co0 and Co1; L2 and L3 to Co2 and Co3.
- **400–500 K:** the same 4 contacts all the time, with nothing new.
  - The ligands that leave their Co (L2 and L3 at 450 K, L0 and L1 at 500 K) are in these columns, and stay stacked.
  - Their columns become slightly uneven: alternating 3.4 and 3.8 Å, slip 0.7 and 1.75 Å. At 400 K, without any bond breaking, the values are 3.4 and 3.6 Å.
  - In the 450/500 K runs that Co–N > 2.6 Å happens at 0.6–0.9 ps, and the stacking does not change at that moment. Stacking is therefore not what drives the CoSCN2bIm2 break.
- **300 and 350 K:** two new stacks form: L4–L6 and L5–L7, 5.6 Å apart at 100 K, then 3.4–3.5 Å.
  - At 350 K they form at about 5 ps and stay (6 contacts per cell).
  - At 300 K they form late, at 13 ps.
  - These are the runs whose flexible cell sheared (*a* 7.5 → 6.4 Å; see `../cells/`). In both runs the new stacks appear when β drops to about 80°: at 5–6 ps at 350 K, and at 13 ps at 300 K (β 82.8° → 79.4°).
  - The extra π stacking therefore belongs to that shear, which is the different cell shape behind the "step" between 350 and 400 K.
- **All temperatures:** the column contacts loosen slowly with T. d_cc goes from 3.3 to 3.45 Å and the interplanar angle from 4° to 7.5°, which is ordinary thermal motion.
