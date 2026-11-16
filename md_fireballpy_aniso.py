#!/usr/bin/env python
"""Constant-pressure MD with a fully flexible cell (anisotropic barostat).

Companion of md_fireballpy.py (same options, output files and checkpoints, so
run_cif_heat.sh can drive it), for fireballpy builds that provide a stress tensor
- the branch in /home/jesus/fireball_qmmm_idipole1_pbc (commit ec3098b or later).
md_fireballpy.py itself is left untouched because the production series use it.

Stress, every --baro-every steps:
  * Fireball part: fireballpy's own stress, stress_method = 'numerical' by default
    (central differences of the energy for the 6 symmetric strains, 12 SCF
    energies, full charge response included), or 'analytic' (cheap, with the
    branch's charge-response correction).
  * D3 part: exact, from the dftd3 virial (fireballpy still never applies D3, so
    FireballD3S adds D3 energy, forces and stress itself, periodic images included).
  * kinetic part: sum_i m_i v_i (x) v_i / V.

Barostat: the volume follows isotropic stochastic cell rescaling (Bussi & Bernetti,
J. Chem. Phys. 153, 114107, 2020), exactly as md_fireballpy.py --barostat crescale;
the shape follows the deviatoric part of the pressure tensor with Berendsen-type
relaxation (same coupling beta/taup), i.e. the mean shape is right but shape
fluctuations are not sampled exactly.  The strain is symmetric, so the cell does
not rotate: cell -> cell @ mu, positions -> positions @ mu, velocities -> v @ mu^-1,
mu = expm(eps).

Output, in addition to md_fireballpy.py's: cell.dat keeps its 8-column format
(lengths of a, b, c; pressures are the isotropic parts) so npt_status.py works,
and cellmat.dat holds the full cell (9 numbers), the angles alpha beta gamma, the
total pressure tensor P_int and the static (virial) part -sigma, Voigt order
xx yy zz yz xz xy, GPa.

    PYTHONPATH=<branch build> python md_fireballpy_aniso.py <rundir> --temp 400 \\
        --d3 periodic --d3-s9 0 --pressure 0 --baro-every 50 --steps 1000
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
from ase import units
from ase.calculators.calculator import all_changes
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from numpy.linalg import LinAlgError

from fireballpy import Fireball
from md_fireballpy import CORRECTION, FDATA, append_frame, make_atoms

VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def voigt_to_mat(s):
    m = np.empty((3, 3))
    for k, (i, j) in enumerate(VOIGT):
        m[i, j] = m[j, i] = s[k]
    return m


def mat_to_voigt(m):
    return np.array([m[i, j] for i, j in VOIGT])


class FireballD3S(Fireball):
    """Fireball (with stress) + periodic DFT-D3(BJ) energy, forces and stress.

    fireballpy returns its own energy every call, but forces and stress only when
    they are recomputed; a call that asks only for the stress leaves the previous
    (already corrected) forces in self.results.  The corrected arrays are tracked
    by identity so the D3 gradient and virial are added exactly once.
    """

    def __init__(self, *args, d3_params=None, **kwargs):
        super().__init__(*args, correction=False, **kwargs)
        import dftd3.interface as d3i
        self._d3i = d3i
        self._d3par = d3i.RationalDampingParam(**d3_params)
        self._d3 = None
        self._f_out = None
        self._s_out = None

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos = self.atoms.get_positions() / units.Bohr
        cell = np.array(self.atoms.get_cell()) / units.Bohr
        if self._d3 is None:
            self._d3 = self._d3i.DispersionModel(
                self.atoms.get_atomic_numbers(), pos, lattice=cell,
                periodic=np.asarray(self.atoms.pbc))
        else:
            self._d3.update(pos, cell)
        r = self._d3.get_dispersion(param=self._d3par, grad=True)
        e = r['energy'] * units.Hartree
        self.results['energy'] = self.results['energy'] + e
        self.results['free_energy'] = self.results['free_energy'] + e
        self.results['d3_energy'] = e
        f = self.results.get('forces')
        if f is not None and f is not self._f_out:
            self._f_out = f - r['gradient'] * units.Hartree / units.Bohr
            self.results['forces'] = self._f_out
        s = self.results.get('stress')
        if s is not None and s is not self._s_out:
            # stress = (1/V) dE/d(eps); the dftd3 virial is dE/d(eps)
            sig_d3 = np.asarray(r['virial']) * units.Hartree / self.atoms.get_volume()
            self._s_out = np.asarray(s) + mat_to_voigt(sig_d3)
            self.results['stress'] = self._s_out
            self.results['d3_stress'] = mat_to_voigt(sig_d3)


def aniso_crescale(at, rng, sigma, temp, p0, beta, taup, dtb):
    """One barostat move.  sigma: static stress (3x3, eV/A^3, ASE sign).
    Volume: stochastic cell rescaling; shape: Berendsen on the deviatoric pressure.
    Returns (P_int 3x3, P_static 3x3) before the move, eV/A^3."""
    v = at.get_volume()
    kt = units.kB * temp
    m = at.get_masses()
    vel = at.get_velocities()
    kin = np.einsum('i,ij,ik->jk', m, vel, vel) / v
    p_static = -sigma
    p_int = p_static + kin
    p_iso = np.trace(p_int) / 3
    deps_vol = (-(beta / taup) * (p0 - p_iso - kt / v) * dtb
                + np.sqrt(2 * kt * beta * dtb / (v * taup)) * rng.standard_normal())
    dev = p_int - p_iso * np.eye(3)
    eps = (deps_vol / 3) * np.eye(3) + (beta / taup) * dtb * dev
    w, u = np.linalg.eigh(eps)
    mu = u @ np.diag(np.exp(w)) @ u.T
    cell = np.array(at.get_cell()) @ mu
    pos = at.get_positions() @ mu
    at.set_cell(cell, scale_atoms=False)
    at.set_positions(pos)
    at.set_velocities(vel @ np.linalg.inv(mu))
    return p_int, p_static


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('rundir')
    ap.add_argument('--temp', type=float, required=True)
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--dt', type=float, default=1.0)
    ap.add_argument('--thermostat', choices=['langevin'], default='langevin')
    ap.add_argument('--friction', type=float, default=0.01)
    ap.add_argument('--kpts', nargs=3, type=int, default=[1, 1, 1])
    ap.add_argument('--fmax-start', type=float, default=0.5)
    ap.add_argument('--restart', action='store_true')
    ap.add_argument('--mixer', choices=['anderson', 'johnson'], default='johnson')
    ap.add_argument('--beta', type=float, default=0.1)
    ap.add_argument('--mix-order', type=int, default=6)
    ap.add_argument('--max-iter', type=int, default=300)
    ap.add_argument('--ejump', type=float, default=50.0)
    ap.add_argument('--checkpoint', type=int, default=50)
    ap.add_argument('--d3', choices=['periodic'], default='periodic',
                    help='only periodic: a flexible cell without dispersion just expands')
    ap.add_argument('--d3-s9', type=float, default=1.0)
    ap.add_argument('--barostat', choices=['crescale-aniso'], default='crescale-aniso')
    ap.add_argument('--pressure', type=float, default=0.0, help='GPa, hydrostatic')
    ap.add_argument('--taup', type=float, default=1000.0, help='fs')
    ap.add_argument('--compress', type=float, default=0.1, help='GPa^-1 (coupling only)')
    ap.add_argument('--baro-every', type=int, default=50,
                    help='steps between stress evaluations / cell moves (default 50; '
                         'a numerical stress costs 12 SCF energies)')
    ap.add_argument('--dipole', choices=['legacy', 'improved'], default='legacy',
                    help="dipole_method; 'legacy' (default) is what fireballpy 0.8.9 used for "
                         "periodic cells, so results compare with the npt/npt3b series")
    ap.add_argument('--stress-method', choices=['numerical', 'analytic'], default='numerical')
    ap.add_argument('--stress-strain', type=float, default=1e-3,
                    help='strain of the numerical stress (default 1e-3)')
    args = ap.parse_args()
    if args.checkpoint % args.baro_every:
        raise SystemExit('--checkpoint must be a multiple of --baro-every')
    if 'stress' not in getattr(Fireball, 'implemented_properties', []):
        raise SystemExit('this fireballpy has no stress tensor: put a build of '
                         '/home/jesus/fireball_qmmm_idipole1_pbc (>= ec3098b) on PYTHONPATH')

    rundir = os.path.abspath(args.rundir)
    xyz = os.path.join(rundir, 'answer.xyz')
    elog = os.path.join(rundir, 'energy.dat')
    clog = os.path.join(rundir, 'cell.dat')
    mlog = os.path.join(rundir, 'cellmat.dat')
    ckpt = os.path.join(rundir, 'restart.npz')

    at = make_atoms(rundir)
    step0 = 0
    if args.restart:
        if not os.path.exists(ckpt):
            raise SystemExit(f'--restart given but {ckpt} does not exist')
        d = np.load(ckpt)
        if 'cell' in d:
            at.set_cell(d['cell'], scale_atoms=False)
        at.set_positions(d['positions'])
        at.set_velocities(d['velocities'])
        step0 = int(d['step'])
        print(f'resuming from step {step0}', flush=True)
    else:
        for f in (xyz, elog, clog, mlog):
            if os.path.exists(f):
                os.remove(f)
        MaxwellBoltzmannDistribution(at, temperature_K=args.temp)
        Stationary(at)
        with open(elog, 'w') as fh:
            fh.write('#   step      time_fs         Epot_eV         Ekin_eV'
                     '        Etot_eV      T_inst_K   walltime_s\n')
    for path, header in (
            (clog, '#   step       V_A3        a_A        b_A        c_A'
                   '   P_vir_GPa  P_kin_GPa  P_int_GPa\n'),
            (mlog, '#   step  cell a1x a1y a1z a2x a2y a2z a3x a3y a3z (A)  alpha beta gamma (deg)'
                   '  P_int xx yy zz yz xz xy (GPa)  P_static xx yy zz yz xz xy (GPa)\n')):
        if args.restart and os.path.exists(path):
            keep = [l for l in open(path) if l.startswith('#') or int(l.split()[0]) <= step0]
            open(path, 'w').writelines(keep)
        else:
            with open(path, 'w') as fh:
                fh.write(header)

    mixer_kws = {'method': args.mixer, 'beta': args.beta,
                 'mix_order': args.mix_order, 'max_iter': args.max_iter}
    at.calc = FireballD3S('custom', fdata_path=FDATA, charges_method='weighted_lowdin',
                          d3_params={**CORRECTION['params_tweaks'], 's9': args.d3_s9},
                          kpts=tuple(args.kpts), mixer_kws=mixer_kws,
                          dipole_method=args.dipole, stress_method=args.stress_method,
                          stress_strain=args.stress_strain)
    import fireballpy
    print(f'  fireballpy: {fireballpy.__file__}', flush=True)
    print(f'  mixer: {mixer_kws}', flush=True)
    print(f'  D3: periodic, s9 = {args.d3_s9}; dipole_method = {args.dipole}', flush=True)
    print(f'  barostat: anisotropic (volume c-rescale, shape Berendsen), P0 = {args.pressure} GPa, '
          f'taup = {args.taup} fs, compressibility {args.compress} GPa^-1, every '
          f'{args.baro_every} steps, stress {args.stress_method}'
          + (f' (strain {args.stress_strain})' if args.stress_method == 'numerical' else ''),
          flush=True)
    rng = np.random.default_rng()
    dyn = Langevin(at, timestep=args.dt * units.fs, temperature_K=args.temp,
                   friction=args.friction / units.fs)
    nat = len(at)
    print(f'{rundir}: {nat} atoms, T = {args.temp} K, dt = {args.dt} fs, langevin, '
          f'friction {args.friction} fs^-1, steps {step0} -> {step0 + args.steps}', flush=True)

    fmax0 = np.sqrt((at.get_forces() ** 2).sum(axis=1)).max()
    print(f'  initial Fmax = {fmax0:.4f} eV/A', flush=True)
    if not args.restart and fmax0 > args.fmax_start:
        raise SystemExit(f'initial Fmax = {fmax0:.3f} eV/A exceeds --fmax-start '
                         f'({args.fmax_start} eV/A)')

    t_start = time.time()
    last_e: list[float] = []

    def save_ckpt(step):
        np.savez(ckpt + '.tmp.npz', step=step, positions=at.get_positions(),
                 velocities=at.get_velocities(), cell=np.array(at.get_cell()))
        os.replace(ckpt + '.tmp.npz', ckpt)

    def observe() -> None:
        step = step0 + dyn.nsteps
        epot = at.get_potential_energy()
        if last_e and abs(epot - last_e[-1]) > args.ejump:
            raise SystemExit(f'\nSCF produced a spurious solution at step {step}: '
                             f'Epot = {epot:.3f} eV against {last_e[-1]:.3f} eV one step '
                             f'earlier.\nFrames up to step {step - 1} in {xyz} are good.')
        last_e.append(epot)
        ekin = at.get_kinetic_energy()
        tinst = ekin / (1.5 * nat * units.kB)
        append_frame(xyz, at, step, epot)
        with open(elog, 'a') as fh:
            fh.write(f'{step:8d}{step * args.dt:14.2f}{epot:16.6f}{ekin:16.6f}'
                     f'{epot + ekin:16.6f}{tinst:14.3f}{time.time() - t_start:12.1f}\n')
        if step % args.checkpoint == 0:
            save_ckpt(step)

    dyn.attach(observe, interval=1)

    for _ in range(args.steps):
        try:
            dyn.run(1)
        except LinAlgError as exc:
            raise SystemExit(f'SCF failed at step {step0 + dyn.nsteps}: {exc}')
        step = step0 + dyn.nsteps
        if step % args.baro_every == 0:
            t0 = time.time()
            sigma = voigt_to_mat(at.get_stress(include_ideal_gas=False))
            t_stress = time.time() - t0
            p_int, p_stat = aniso_crescale(at, rng, sigma, args.temp,
                                           args.pressure * units.GPa,
                                           args.compress / units.GPa, args.taup,
                                           args.baro_every * args.dt)
            g = 1 / units.GPa
            lens = at.cell.lengths()
            ang = at.cell.angles()
            p_kin_iso = np.trace(p_int - p_stat) / 3
            with open(clog, 'a') as fh:
                fh.write(f'{step:8d}{at.get_volume():12.3f}{lens[0]:11.5f}{lens[1]:11.5f}'
                         f'{lens[2]:11.5f}{np.trace(p_stat) / 3 * g:12.4f}'
                         f'{p_kin_iso * g:11.4f}{np.trace(p_int) / 3 * g:11.4f}\n')
            with open(mlog, 'a') as fh:
                fh.write(f'{step:8d} ' + ' '.join(f'{x:10.5f}' for x in np.array(at.get_cell()).ravel())
                         + ' ' + ' '.join(f'{x:8.3f}' for x in ang)
                         + ' ' + ' '.join(f'{x:8.4f}' for x in mat_to_voigt(p_int) * g)
                         + ' ' + ' '.join(f'{x:8.4f}' for x in mat_to_voigt(p_stat) * g) + '\n')
            print(f'  step {step}: stress in {t_stress:.0f} s; P_int diag '
                  f'{np.round(np.diag(p_int) * g, 3)} GPa, offdiag '
                  f'{np.round(mat_to_voigt(p_int)[3:] * g, 3)} GPa; V {at.get_volume():.1f}, '
                  f'abc {np.round(lens, 4)}, angles {np.round(ang, 2)}', flush=True)
            if step % args.checkpoint == 0:
                save_ckpt(step)        # post-move state
        tinst = at.get_kinetic_energy() / (1.5 * nat * units.kB)
        if not np.isfinite(tinst) or tinst > 10 * args.temp:
            raise SystemExit(f'thermostat diverged at step {step}: T_inst = {tinst:.1f} K')

    dt_wall = time.time() - t_start
    print(f'done: {args.steps} steps in {dt_wall / 3600:.2f} h '
          f'({dt_wall / max(args.steps, 1):.1f} s/step)', flush=True)


if __name__ == '__main__':
    main()
