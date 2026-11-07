#  Copyright (c) 2025. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# ******************************************************************************
# sim.gromacs module (experimental)
# ******************************************************************************
"""GROMACS backend for explicitly selected portable GAFF calculations.

Supported: single-point energies/forces, energy minimization and NVE/NVT/NPT
molecular dynamics with optional hydrogen bond constraints. Native
top/mdp/tpr/edr/trr/xtc/cpt files are retained beside a JSON run manifest, and
an interrupted run can be resumed from its checkpoint. Public results use
RadonPy's real units (angstrom, fs, kcal/mol, atm). GROMACS is an optional
executable, selected with GROMACS_EXEC or solver_path.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ..core import calc, utils
from . import lammps
from .profiles import apply_profile

__version__ = '0.2.0'

KCAL = 4.184
ATM = 1.01325


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Presets build stage settings with numpy (e.g. temperatures from np.arange).
    path.write_text(json.dumps(value, indent=2, allow_nan=False,
                               default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist()) + '\n')


def _write_molecule(mol, path):
    data = utils.MolToJSON_dict(mol)
    for record, conformer in zip(data['molecules'][0]['conformers'], mol.GetConformers()):
        record['coords'] = conformer.GetPositions().tolist()
    data['gromacs_conformer_ids'] = [c.GetId() for c in mol.GetConformers()]
    _json(path, data)


def _read_molecule(path):
    # RDKit's interchange reader casts coordinates to float32. Restore the
    # original doubles after reading graph and RadonPy parameter extensions.
    data = json.loads(Path(path).read_text())
    mol = utils.JSONToMol_dict(data)
    ids = data.get('gromacs_conformer_ids', range(mol.GetNumConformers()))
    for cid, conformer, record in zip(ids, mol.GetConformers(), data['molecules'][0]['conformers']):
        conformer.SetId(cid)
        for i, xyz in enumerate(record['coords']):
            conformer.SetAtomPosition(i, xyz)
    return mol


def clear_run(manifest_path):
    """Remove only artifacts owned by this run, for the explicit tmp_clear API."""
    manifest_path = Path(manifest_path)
    data = json.loads(manifest_path.read_text())
    base = Path(data['work_dir'])
    opts = data['options']
    paths = [Path(data['molecule']), base/(opts['log_file']+'.gmx.json'),
             base/((opts['outstr'] or 'final_state')+'.npz')]
    paths += [base/opts[k] for k in ('write_data', 'xtc_file') if opts.get(k)]
    for path in paths:
        path.unlink(missing_ok=True)
    shutil.rmtree(str(manifest_path)+'.gmx', ignore_errors=True)
    manifest_path.unlink()


def _pairs14(mol):
    """Shortest graph distance, including rings; do not infer from torsions."""
    neighbors = [set(a.GetIdx() for a in atom.GetNeighbors()) for atom in mol.GetAtoms()]
    for start in range(mol.GetNumAtoms()):
        seen, frontier = {start}, {start}
        for _ in range(3):
            frontier = set().union(*(neighbors[i] for i in frontier)) - seen
            seen.update(frontier)
        for end in sorted(frontier):
            if end > start:
                yield start, end


def validate_molecule(mol):
    if not hasattr(mol, 'cell'):
        raise NotImplementedError('GROMACS currently requires a periodic orthorhombic cell')
    if mol.HasProp('ff_name') and mol.GetProp('ff_name').lower() not in ('gaff', 'gaff2', 'gaff2_mod'):
        raise NotImplementedError('Only GAFF, GAFF2 and GAFF2_mod are supported')
    for key, expected in [('pair_style', 'lj'), ('bond_style', 'harmonic'),
                          ('angle_style', 'harmonic'), ('dihedral_style', 'fourier'),
                          ('improper_style', 'cvff')]:
        if mol.HasProp(key) and mol.GetProp(key) != expected:
            raise NotImplementedError('%s=%s is unsupported' % (key, mol.GetProp(key)))
    if getattr(mol, 'cmaps', {}):
        raise NotImplementedError('CMAP is not supported')
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'H' and atom.GetIsotope() >= 3:
            raise NotImplementedError('Linker/isotope-labelled hydrogens require explicit mass handling')
        if any(atom.HasProp(k) and atom.GetBoolProp(k) for k in ('CL_react', 'CL_remove')):
            raise NotImplementedError('Reactive atoms are not supported')
        for key in ('AtomicCharge', 'ff_sigma', 'ff_epsilon'):
            if not atom.HasProp(key) or not np.isfinite(atom.GetDoubleProp(key)):
                raise ValueError('Missing or invalid atom parameter: %s' % key)
        if atom.GetMass() <= 0 or atom.GetDoubleProp('ff_sigma') < 0 or atom.GetDoubleProp('ff_epsilon') < 0:
            raise ValueError('Invalid mass or LJ parameters')
    if abs(sum(a.GetDoubleProp('AtomicCharge') for a in mol.GetAtoms())) > 1e-6:
        raise NotImplementedError('Only charge-neutral cells are supported')
    if not all(np.isfinite(c.GetPositions()).all() for c in mol.GetConformers()):
        raise ValueError('Non-finite coordinates')




def is_constrained(mol, i, j):
    # LAMMPS "fix shake ... m 1.0" selects bonds by a mass window (0.1), not by element.
    return any(abs(mol.GetAtomWithIdx(n).GetMass()-1.0) <= 0.1 for n in (i, j))


def topology(mol, shake=False, electrostatics=True):
    """Convert assigned parameters without retyping atoms or assigning charges.

    electrostatics=False zeroes charges, for LAMMPS-style charge-free lj/cut packing.
    """
    validate_molecule(mol)
    rows = ['; Generated from RadonPy parameters; nm, kJ/mol', '[ defaults ]',
            '1 2 yes 0.5 0.8333333333333333', '', '[ atomtypes ]']
    types, atom_types = {}, []
    for atom in mol.GetAtoms():
        key = (atom.GetMass(), atom.GetDoubleProp('ff_sigma'), atom.GetDoubleProp('ff_epsilon'))
        if key not in types:
            name = 'R%d' % len(types)
            types[key] = name
            rows.append('%s %.16g 0 A %.16g %.16g' % (name, key[0], key[1]/10, key[2]*KCAL))
        atom_types.append(types[key])
    rows += ['', '[ moleculetype ]', 'RADON 3', '', '[ atoms ]']
    for i, atom in enumerate(mol.GetAtoms()):
        charge = atom.GetDoubleProp('AtomicCharge') if electrostatics else 0.0
        rows.append('%d %s 1 MOL %s %d %.16g %.16g' %
                    (i+1, atom_types[i], atom.GetSymbol(), i+1, charge, atom.GetMass()))
    rows += ['', '[ bonds ]']
    constraints = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        r0, k = bond.GetDoubleProp('ff_r0')/10, bond.GetDoubleProp('ff_k')*2*KCAL*100
        if shake and is_constrained(mol, i, j):
            # Function 5 keeps connectivity and exclusions without a bond force.
            rows.append('%d %d 5' % (i+1, j+1))
            constraints.append('%d %d 1 %.16g' % (i+1, j+1, r0))
        else:
            rows.append('%d %d 1 %.16g %.16g' % (i+1, j+1, r0, k))
    rows += ['', '[ pairs ]']
    rows += ['%d %d 1' % (a+1, b+1) for a, b in _pairs14(mol)]
    rows += ['', '[ angles ]']
    for term in getattr(mol, 'angles', {}).values():
        rows.append('%d %d %d 1 %.16g %.16g' %
                    (term.a+1, term.b+1, term.c+1, term.ff.theta0, term.ff.k*2*KCAL))
    rows += ['', '[ dihedrals ]']
    for term in getattr(mol, 'dihedrals', {}).values():
        for k, n, phase in zip(term.ff.k, term.ff.n, term.ff.d0):
            rows.append('%d %d %d %d 9 %.16g %.16g %d' %
                        (term.a+1, term.b+1, term.c+1, term.d+1, phase, k*KCAL, n))
    for term in getattr(mol, 'impropers', {}).values():
        if term.ff.d0 not in (-1, 1):
            raise ValueError('CVFF improper sign must be -1 or 1')
        rows.append('%d %d %d %d 4 %g %.16g %d' %
                    (term.a+1, term.b+1, term.c+1, term.d+1,
                     180 if term.ff.d0 == -1 else 0, term.ff.k*KCAL, term.ff.n))
    if constraints:
        rows += ['', '[ constraints ]'] + constraints
    rows += ['', '[ system ]', 'RadonPy portable cell', '', '[ molecules ]', 'RADON 1', '']
    return '\n'.join(rows)


def write_gro(mol, path, confId=0):
    """Extended precision GRO input; stock three-decimal GRO loses force accuracy."""
    cell = mol.cell
    origin = np.array([cell.xlo, cell.ylo, cell.zlo])
    xyz = (mol.GetConformer(confId).GetPositions()-origin)/10
    rows = ['RadonPy', str(mol.GetNumAtoms())]
    for i, atom in enumerate(mol.GetAtoms()):
        vel = [atom.GetDoubleProp(k)*100 if atom.HasProp(k) else 0 for k in ('vx', 'vy', 'vz')]
        rows.append('%5d%-5s%5s%5d' % (1, 'MOL', atom.GetSymbol(), (i+1) % 100000)
                    + ''.join('%15.10f' % x for x in xyz[i])
                    + ''.join('%15.11f' % x for x in vel))
    rows.append(' '.join('%.12f' % (x/10) for x in (cell.dx, cell.dy, cell.dz)))
    Path(path).write_text('\n'.join(rows)+'\n')


def read_g96(path):
    """Return positions (angstrom), velocities (angstrom/fs) and orthorhombic box lengths."""
    blocks, section = {}, None
    for line in Path(path).read_text().splitlines():
        if line.strip() == 'END':
            section = None
        elif line.strip() in ('POSITION', 'POSITIONRED', 'VELOCITY', 'VELOCITYRED', 'BOX'):
            section = line.strip()
            blocks[section] = []
        elif section and line.strip() and not line.startswith('#'):
            blocks[section].append([float(x) for x in line.split()[-3:]])
    pos = np.asarray(blocks.get('POSITION', blocks.get('POSITIONRED')))*10
    vel = blocks.get('VELOCITY', blocks.get('VELOCITYRED'))
    vel = None if vel is None else np.asarray(vel)/100
    box = np.asarray(blocks['BOX']).ravel()*10
    if len(box) != 3:
        raise NotImplementedError('Only orthorhombic boxes are supported')
    return pos, vel, box


def read_xvg(path):
    names, data = {}, []
    for line in Path(path).read_text().splitlines():
        match = re.match(r'@\s+s(\d+)\s+legend\s+"(.*)"', line)
        if match:
            names[int(match[1])] = match[2]
        elif line.strip() and not line.startswith(('#', '@', '&')):
            data.append([float(x) for x in line.split()])
    if not data:
        raise ValueError('No numeric data in %s' % path)
    array = np.asarray(data)
    return array, [names.get(i, str(i)) for i in range(array.shape[1]-1)]


def unwrap_trajectory(xtc, initial, output=None, observer=None):
    """Follow each atom's periodic image through a native trajectory.

    initial: unwrapped coordinates (angstrom, GROMACS frame) at the first
    frame. Frames must be close enough that no atom moves half a box length
    between them. Optionally writes the unwrapped frames to another XTC and
    calls observer(time_ps, unwrapped_angstrom) for every frame.
    Returns the unwrapped coordinates of the last frame.
    """
    from mdtraj.formats import XTCTrajectoryFile
    previous = np.asarray(initial, dtype=float)
    writer = XTCTrajectoryFile(str(output), 'w') if output else None
    try:
        with XTCTrajectoryFile(str(xtc)) as trajectory:
            while True:
                xyz, time, step, box = trajectory.read(n_frames=1000)
                if len(xyz) == 0:
                    break
                frames = []
                for t, x, b in zip(time, xyz.astype(float)*10, box.astype(float)*10):
                    lengths = np.diag(b)
                    x += lengths*np.rint((previous-x)/lengths)
                    previous = x
                    frames.append(x)
                    if observer is not None:
                        observer(float(t), x)
                if writer is not None:
                    writer.write(np.asarray(frames)/10, time=time, step=step, box=box)
    finally:
        if writer is not None:
            writer.close()
    return previous


class Observables:
    """Rg, MSD and dipole from unwrapped trajectory frames, written like LAMMPS eq presets.

    LAMMPS averages every step over windows (rg_ave_length steps for
    gyration/chunk, thermo_freq for "msd com yes"); here, the trajectory
    frames inside each window are averaged. Rg is mass weighted per molecule;
    MSD is relative to the stage start with the center-of-mass drift removed.
    The total dipole (e*angstrom) is the instantaneous sum of q*r over whole
    molecules, which for neutral molecules is the sum of LAMMPS dipole/chunk.
    """
    def __init__(self, mol, initial, wf, md, elapsed):
        from rdkit import Chem
        self.masses = np.array([a.GetMass() for a in mol.GetAtoms()])
        self.fragments = [np.asarray(f) for f in Chem.GetMolFrags(mol)]
        self.initial = np.asarray(initial, dtype=float)
        self.com0 = np.average(self.initial, axis=0, weights=self.masses)
        self.time_step, self.elapsed = wf.time_step, elapsed
        self.rg_window = wf.rg_ave_length if getattr(wf, 'rg', False) else None
        self.msd_window = md.thermo_freq if getattr(wf, 'msd', False) else None
        self.rg, self.msd = [], []
        self._rg, self._msd = [], []
        self.charges = (np.array([a.GetDoubleProp('AtomicCharge') for a in mol.GetAtoms()])
                        if getattr(wf, 'dipole', False) else None)
        self.dipole = []

    def __call__(self, time_ps, xyz):
        step = int(round((time_ps*1000-self.elapsed)/self.time_step))
        if self.charges is not None:
            self.dipole.append((step, (self.charges@xyz).tolist()))
        if step == 0:
            return
        if self.rg_window:
            values = []
            for f in self.fragments:
                w = self.masses[f]
                center = np.average(xyz[f], axis=0, weights=w)
                values.append(np.sqrt(np.average(np.sum((xyz[f]-center)**2, axis=1), weights=w)))
            self._rg.append(values)
            if step % self.rg_window == 0:
                self.rg.append((step, np.mean(self._rg, axis=0).tolist()))
                self._rg = []
        if self.msd_window:
            delta = xyz-self.initial
            delta -= np.average(xyz, axis=0, weights=self.masses)-self.com0
            self._msd.append(np.mean(np.sum(delta**2, axis=1)))
            if step % self.msd_window == 0:
                self.msd.append((step, float(np.mean(self._msd))))
                self._msd = []

    def write_rg(self, path, step_offset):
        rows = ['# Time-averaged data from GROMACS trajectory frames',
                '# TimeStep Number-of-rows', '# Row c_gyr']
        for step, values in self.rg:
            rows.append('%d %d' % (step+step_offset, len(values)))
            rows += ['%d %.10g' % (i+1, v) for i, v in enumerate(values)]
        Path(path).write_text('\n'.join(rows)+'\n')


ALLOWED_COMMANDS = ('comm_modify', 'neigh_modify')  # LAMMPS-only tuning, no GROMACS meaning


def _only_tuning(commands):
    return all(str(c).split()[0] in ALLOWED_COMMANDS for c in commands if str(c).strip())


THERMOSTATS = {'Nose-Hoover': 'nose-hoover', 'csvr': 'v-rescale', 'Langevin': None}
BAROSTATS = ('Parrinello-Rahman', 'C-rescale')
ATM_A3_KCAL = 101325e-30*6.02214076e23/4184  # 1 atm*angstrom^3 in kcal/mol


class Gromacs:
    def __init__(self, work_dir=None, solver_path=None, **kwargs):
        self.work_dir = str(Path(work_dir or '.').resolve())
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)
        self.solver_path = str(solver_path or os.environ.get('GROMACS_EXEC', 'gmx'))
        idx = kwargs.get('idx')
        self.input_file = 'radon_gmx%s.json' % ('' if idx is None else '_%d' % idx)
        self.output_file = 'gromacs.stdout'
        self.dat_file = None
        self.confId = 0

    @property
    def get_name(self):
        return 'GROMACS'

    def _path(self, name):
        return Path(self.work_dir) / name

    def _command(self, args, cwd, stdin=None, mpi=0, omp=1):
        executable = shutil.which(self.solver_path)
        if executable is None:
            raise FileNotFoundError('GROMACS executable not found: %s; set GROMACS_EXEC' % self.solver_path)
        command = [executable] + [str(a) for a in args]
        if mpi > 1:
            command = ['mpirun', '-np', str(mpi)] + command
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(max(1, omp))
        cp = subprocess.run(command, cwd=cwd, input=stdin, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        with (Path(cwd)/'commands.log').open('a') as stream:
            stream.write(json.dumps(command)+'\n'+cp.stdout+'\n')
        if cp.returncode:
            raise RuntimeError('GROMACS failed (%s); see %s\n%s' %
                               (cp.returncode, Path(cwd)/'commands.log', cp.stdout[-4000:]))
        return cp

    def get_version(self):
        return self._command(['--version'], self.work_dir).stdout

    def make_dat(self, mol, confId=0, file_name=None, dir_name=None, velocity=True, temp=300, **kwargs):
        """
        Gromacs.make_dat

        Save an exact snapshot of the parameterized molecule. Native topology
        and coordinate files are generated per stage when running.

        Optional args:
            velocity: Generate velocities at temp if any atom lacks them, like LAMMPS (boolean)
        """
        validate_molecule(mol)
        mol.GetConformer(confId)
        mol = utils.deepcopy_mol(mol)
        if velocity and not all(a.HasProp(k) for a in mol.GetAtoms() for k in ('vx', 'vy', 'vz')):
            calc.set_velocity(mol, temp)
        name = file_name or 'radon_gmx_molecule'
        self.dat_file = str(Path(dir_name or self.work_dir) / (name+'.json'))
        _write_molecule(mol, self.dat_file)
        self.confId = confId
        return self.dat_file

    def make_input(self, md, file_name=None, dir_name=None):
        """
        Gromacs.make_input

        Validate an MD object and save it as a JSON run manifest.
        """
        self._check(md)
        if self.dat_file is None:
            raise ValueError('Call make_dat before make_input')
        name = str(Path(dir_name or self.work_dir) / (file_name or self.input_file))
        if Path(name).resolve() == Path(self.dat_file).resolve():
            raise ValueError('Input manifest and molecule snapshot must use different paths')
        options = {k: v for k, v in vars(md).items() if k not in ('mol', 'wf') and not k.startswith('_')}
        _json(name, {'schema': 1, 'engine': 'gromacs', 'molecule': self.dat_file,
                     'work_dir': self.work_dir, 'confId': self.confId, 'options': options,
                     'workflow': [vars(w) for w in md.wf]})
        self.input_file = name
        md._gromacs_manifest = name
        return name

    def _check(self, md):
        """Validate an MD object (and apply its interaction profile)."""
        apply_profile(md)
        if not md.pbc or md.units != 'real' or md.boundary != 'p p p':
            raise NotImplementedError('Only periodic real-unit simulations are supported')
        if md.drude or not _only_tuning(md.add) or md.add_f:
            raise NotImplementedError('Drude and raw LAMMPS commands are unsupported')
        if md.atom_style != 'full' or md.bc_flag or any(getattr(md, key) != value for key, value in
                [('bond_style', 'harmonic'), ('angle_style', 'harmonic'),
                 ('dihedral_style', 'fourier'), ('improper_style', 'cvff')]):
            raise NotImplementedError('Unsupported topology or bonded interaction settings')
        # The default profile maps the LAMMPS CHARMM switch onto a GROMACS potential switch
        # (not the same polynomial); 'portable' is the exactly reproducible validation model.
        pair_styles = ('lj/cut/coul/long', 'lj/cut') if md.interaction_profile else ('lj/charmm/coul/long', 'lj/cut')
        if md.pair_style not in pair_styles or md.dielectric != 1.0:
            raise NotImplementedError('Unsupported nonbonded settings')
        if md.pair_style == 'lj/charmm/coul/long' and not 0 < float(md.cutoff_in) < float(md.cutoff_out):
            raise ValueError('The LJ switch needs 0 < cutoff_in < cutoff_out')
        if md.thermo_freq < 1 or md.dump_freq < 1:
            raise ValueError('Output frequencies must be positive')
        # Presets may give floats (AutoMD ef_dp); LAMMPS writes them as thermo %i.
        md.thermo_freq, md.dump_freq = int(md.thermo_freq), int(md.dump_freq)
        if not _is_temperature(md.set_init_velocity, allow_none=True):
            raise ValueError('set_init_velocity must be a temperature or None')
        for wf in md.wf:
            self._validate_workflow(wf, md)

    def rerun(self, md, mol, xyz, lengths, name='rerun', omp=1, gpu=0):
        """
        Gromacs.rerun

        Energies of mol at the given frames, like a LAMMPS rerun: xyz (frames x atoms x 3) and
        orthorhombic box lengths (frames x 3) in angstrom, with the interactions of md (no workflow).
        Returns one row per frame with LAMMPS thermo names (energies in kcal/mol, volume in
        angstrom^3). The run is kept in <name>.gmx under the work directory; reusing a name raises.
        """
        self._check(md)
        if md.wf:
            raise ValueError('A rerun takes the interactions only; remove the MD stages')
        xyz, lengths = np.asarray(xyz, dtype=float), np.asarray(lengths, dtype=float)
        if xyz.ndim != 3 or xyz.shape[1:] != (mol.GetNumAtoms(), 3) or lengths.shape != (len(xyz), 3):
            raise ValueError('Frames must be (frames, atoms, 3) with (frames, 3) box lengths')
        root = self._path(name+'.gmx')
        if root.exists():
            raise FileExistsError('Run already exists: %s' % root)
        root.mkdir(parents=True)
        (root/'system.top').write_text(topology(mol, electrostatics=md.pair_style != 'lj/cut'))
        write_gro(mol, root/'input.gro')
        # Single-point settings, energies of every frame and no forces.
        (root/'run.mdp').write_text(self._mdp(md, None, mol, 0.0, gpu=bool(gpu)).replace('nstfout = 1\n', 'nstfout = 0\n'))
        self._command(['grompp', '-f', 'run.mdp', '-c', 'input.gro', '-p', 'system.top', '-o', 'run.tpr'], root)
        import mdtraj
        boxes = np.zeros((len(xyz), 3, 3))
        boxes[:, [0, 1, 2], [0, 1, 2]] = lengths/10
        with mdtraj.formats.TRRTrajectoryFile(str(root/'frames.trr'), 'w') as trr:
            trr.write(xyz/10, time=np.arange(len(xyz), dtype=float), step=np.arange(len(xyz)), box=boxes)
        command = ['mdrun', '-s', 'run.tpr', '-rerun', 'frames.trr', '-deffnm', 'run', '-ntomp', str(max(1, omp))]
        command += ['-nb', 'gpu'] if gpu else ['-nb', 'cpu', '-pme', 'cpu', '-bonded', 'cpu']
        if 'thread_mpi' in self._command(['--version'], root).stdout:
            command += ['-ntmpi', '1']
        self._command(command, root, omp=omp)
        frame = self._energy_frame(root, mol, None, 0.0, 0)
        if len(frame) != len(xyz):
            raise RuntimeError('GROMACS evaluated %i of %i frames' % (len(frame), len(xyz)))
        # The box of each frame (the energy file of a rerun has no box terms).
        for axis, length in zip('xyz', lengths.T):
            frame['L'+axis] = length
        frame['Volume'] = np.prod(lengths, axis=1)
        frame['Density'] = calc.mol_mass(mol)/6.02214076e23/frame['Volume']*1e24
        return frame

    @staticmethod
    def _validate_workflow(wf, md):
        if wf.type == 'minimize':
            if wf.min_style not in ('cg', 'sd'):
                raise NotImplementedError('GROMACS minimization supports cg and sd')
            if wf.maxiter < 1 or not wf.ftol > 0:
                raise ValueError('Minimization requires positive maxiter and ftol')
            return
        if wf.ensemble not in ('nve', 'nvt', 'npt'):
            raise NotImplementedError('GROMACS supports nve, nvt and npt, not %s' % wf.ensemble)
        for flag in ('bcreate', 'variable', 'timeave', 'rerun', 'rattle'):
            if getattr(wf, flag, False):
                raise NotImplementedError('Unsupported workflow option for GROMACS: %s' % flag)
        if getattr(wf, 'efield', False):
            if wf.efield_rate:
                raise NotImplementedError('GROMACS electric fields are constant or cosine; '
                                          'linear ramps (efield rate) are unsupported')
            _efield(wf)
        if wf.deform == 'erate':
            # GROMACS deform scales the coordinates with the box, as LAMMPS remap x.
            if wf.deform_axis not in ('x', 'y', 'z') or wf.ensemble not in ('nvt', 'npt') or wf.deform_remap != 'x':
                raise NotImplementedError('Strain-rate deformation is supported along one axis with remap x, '
                                          'in NVT or NPT')
        elif wf.deform and (wf.deform != 'final' or wf.deform_axis != 'xyz' or wf.ensemble != 'nvt'):
            raise NotImplementedError('Only isotropic NVT deformation to a final box, or strain-rate '
                                      'deformation along one axis, is supported')
        if wf.nve_limit or not _only_tuning(wf.add) or wf.add_f:
            raise NotImplementedError('nve/limit and raw LAMMPS commands are unsupported')
        if (wf.thermo_freq is not None and wf.thermo_freq != md.thermo_freq) or wf.thermo_style:
            raise NotImplementedError('Set output frequency on MD; per-stage thermo overrides are unsupported')
        # As in LAMMPS, per-axis pressures and couple apply only with p_aniso.
        axes = _pressure_axes(wf)
        if axes is not None:
            if wf.p_couple:
                raise NotImplementedError('Coupled anisotropic pressure axes are unsupported')
            if wf.barostat != 'Parrinello-Rahman':
                raise NotImplementedError("Anisotropic pressure control needs barostat='Parrinello-Rahman'; "
                                          'GROMACS C-rescale has no anisotropic coupling')
            if any(start != stop for start, stop, _ in axes.values()):
                raise NotImplementedError('Pressure ramps are unsupported')
            if len({dump for _, _, dump in axes.values()}) > 1:
                raise NotImplementedError('GROMACS has one pressure damping time for all axes')
            if wf.deform == 'erate' and wf.deform_axis in axes:
                raise ValueError('The deformed axis cannot also be pressure controlled')
        if wf.ensemble != 'nve' and wf.thermostat not in THERMOSTATS:
            raise NotImplementedError('GROMACS supports the %s thermostats, not %s'
                                      % (', '.join(THERMOSTATS), wf.thermostat))
        if wf.ensemble == 'npt':
            if wf.barostat not in BAROSTATS:
                raise ValueError("GROMACS has no MTK barostat; select barostat='Parrinello-Rahman' "
                                 "or 'C-rescale' explicitly")
            if wf.p_start != wf.p_stop:
                raise NotImplementedError('Pressure ramps are unsupported')
            compressibility = getattr(wf, 'compressibility', 4.5e-5)
            if not np.isfinite(compressibility) or compressibility <= 0:
                raise ValueError('Compressibility must be finite and positive (bar^-1)')
        if wf.ensemble != 'nve' and (wf.t_dump <= 0 or min(wf.t_start, wf.t_stop) <= 0):
            raise ValueError('Thermostat temperatures and damping must be positive')
        if wf.step < 1 or wf.time_step <= 0:
            raise ValueError('MD step count and timestep must be positive')
        if not _is_temperature(wf.set_init_velocity, allow_none=True):
            raise ValueError('set_init_velocity must be a temperature or None')

    @staticmethod
    def _mdp(md, wf, mol, elapsed, gpu=False):
        apply_profile(md)
        portable = md.interaction_profile == 'portable'
        switched = md.pair_style == 'lj/charmm/coul/long'
        cutoff = float(md.cutoff_out if switched else md.cutoff_in)/10
        lengths = np.array([mol.cell.dx, mol.cell.dy, mol.cell.dz])/10
        if np.min(lengths) <= 2*cutoff:
            raise ValueError('Each box length must exceed twice the cutoff')
        # lj/cut: LAMMPS-style charge-free short-range packing (charges are zeroed).
        coulomb = 'PME' if md.pair_style in ('lj/cut/coul/long', 'lj/charmm/coul/long') else 'Cut-off'
        opts = {'cutoff-scheme': 'Verlet', 'pbc': 'xyz', 'rlist': cutoff,
                'verlet-buffer-tolerance': -1, 'nstlist': 1,
                'coulombtype': coulomb, 'rcoulomb': cutoff, 'coulomb-modifier': 'None',
                'vdwtype': 'Cut-off', 'vdw-modifier': 'None', 'rvdw': cutoff,
                'DispCorr': 'no', 'fourierspacing': 0.08, 'pme-order': 6,
                'ewald-rtol': 1e-8, 'constraints': 'none', 'comm-mode': 'None',
                'nstcalcenergy': 1, 'nstxout': 0, 'nstvout': 0,
                'nstxout-compressed': 0}
        if switched:
            # LAMMPS lj/charmm switches the LJ energy between cutoff_in and cutoff_out.
            opts.update({'vdw-modifier': 'Potential-switch', 'rvdw-switch': float(md.cutoff_in)/10})
        # Production numerics (the default profile, and any GPU run): PME order 4, a buffered
        # pair list, and coupling and energies every 10-100 steps instead of every step. The
        # portable CPU numerics stay exact for the static and trajectory parity tests.
        fast = gpu or not portable
        if not portable:
            # The LAMMPS PPPM accuracy (relative force error) as the Ewald tolerance.
            opts.update({'ewald-rtol': float(md.kspace_style_accuracy or 1e-6), 'fourierspacing': 0.1})
        if fast:
            opts['pme-order'] = 4
        # Minimization and sub-fs packing start far from equilibrium, where the buffer estimate
        # does not hold (a PMMA cell minimized to 2e9 kJ/(mol nm) forces and exploded), so the
        # default profile keeps the exact pair list there; GPU runs always buffer.
        settled = wf is not None and wf.type == 'md' and wf.time_step >= 1.0
        if gpu or (fast and settled):
            opts.update({'verlet-buffer-tolerance': 0.005, 'nstlist': 10})
        if wf is None:
            # A single point is a zero-step MD run that writes forces.
            opts.update(integrator='md', nsteps=0, dt=0.001, nstenergy=1,
                        nstlog=1, nstfout=1, **{'gen-vel': 'no'})
        elif wf.type == 'minimize':
            # LAMMPS etol and maxeval have no GROMACS equivalents.
            opts.update(integrator='cg' if wf.min_style == 'cg' else 'steep',
                        nsteps=wf.maxiter, emtol=wf.ftol*KCAL*10, emstep=0.001,
                        nstenergy=md.thermo_freq, nstlog=md.thermo_freq, nstfout=0)
        else:
            # Couple every step like LAMMPS fixes; the trajectory is always
            # written because it is needed to follow periodic images.
            # Step counts may be floats (e.g. 5e5); LAMMPS writes them as run %i.
            opts.update({'integrator': 'md-vv' if wf.ensemble == 'nve' else 'md',
                         'nsteps': int(wf.step), 'dt': wf.time_step/1000, 'tinit': elapsed/1000,
                         'nstenergy': md.thermo_freq, 'nstlog': md.thermo_freq, 'nstfout': 0,
                         'nstxout-compressed': Gromacs._frame_stride(md, wf), 'comm-mode': 'Linear', 'nstcomm': 100,
                         'gen-vel': 'no', 'continuation': 'no',
                         'constraint-algorithm': 'lincs', 'lincs-order': 8, 'lincs-iter': 2})
            if fast:
                opts['nstcalcenergy'] = int(np.gcd(md.thermo_freq, 100))
            def couple(tau, every_step=False):
                # Up to every 10 steps, within GROMACS' 20 coupling steps per tau.
                if not fast or every_step:
                    return 1
                return int(np.clip(tau/(20*wf.time_step), 1, 10))
            if wf.ensemble != 'nve':
                opts.update({'tc-grps': 'System', 'tau-t': wf.t_dump/1000, 'ref-t': wf.t_start,
                             'nsttcouple': couple(wf.t_dump, wf.thermostat == 'Nose-Hoover'), 'ld-seed': np.random.randint(1000, 999999)})
                if wf.thermostat == 'Langevin':
                    opts['integrator'] = 'sd'
                else:
                    opts['tcoupl'] = THERMOSTATS[wf.thermostat]
                    if wf.thermostat == 'Nose-Hoover':
                        # Leapfrog supports no chains. Coupling Nose-Hoover only
                        # every 5 fs heated unconstrained octane (density -9%).
                        opts['nh-chain-length'] = 1
                if wf.t_start != wf.t_stop:
                    end = (elapsed+wf.step*wf.time_step)/1000
                    opts.update({'annealing': 'single', 'annealing-npoints': 2,
                                 'annealing-time': '%.12g %.12g' % (elapsed/1000, end),
                                 'annealing-temp': '%.12g %.12g' % (wf.t_start, wf.t_stop)})
            if wf.deform == 'erate':
                # LAMMPS erate units box: constant box velocity L0*rate (rate in 1/fs), in nm/ps.
                velocity = np.zeros(6)
                velocity['xyz'.index(wf.deform_axis)] = lengths['xyz'.index(wf.deform_axis)]*wf.deform_rate*1000
                opts['deform'] = ' '.join('%.12g' % x for x in velocity)
            elif wf.deform:
                target = (wf.deform_fin_hi-wf.deform_fin_lo)/10
                if target <= 2*cutoff:
                    raise ValueError('The final deformed box is smaller than twice the cutoff')
                rates = (target-lengths)/(wf.step*wf.time_step/1000)
                opts['deform'] = ' '.join('%.12g' % x for x in [*rates, 0, 0, 0])
            if getattr(wf, 'dipole', False):
                # The dipole comes from the trajectory; LAMMPS uses full-precision coordinates,
                # while the default XTC precision (0.001 nm) would add about 0.01 e*angstrom per frame.
                opts['compressed-x-precision'] = 100000
            for axis, value in (_efield(wf) if getattr(wf, 'efield', False) else {}).items():
                # LAMMPS efac = E cos(2 pi f time); GROMACS E0 cos(omega (t - t0)) in V/nm and 1/ps.
                opts['electric-field-'+axis] = '%.12g %.12g 0 0' % (value*10, 2*np.pi*wf.efield_freq*1e-12)
            if wf.ensemble == 'npt':
                opts.update({'pcoupl': wf.barostat, 'pcoupltype': 'isotropic', 'nstpcouple': couple(wf.p_dump),
                             'tau-p': wf.p_dump/1000, 'ref-p': wf.p_start*ATM,
                             'compressibility': getattr(wf, 'compressibility', 4.5e-5)})
                axes = _pressure_axes(wf)
                if axes is not None:
                    # Axes without a pressure (e.g. the deformed one) keep zero compressibility.
                    dump = next(iter(axes.values()))[2]
                    compressibility = getattr(wf, 'compressibility', 4.5e-5)
                    opts.update({'pcoupltype': 'anisotropic', 'nstpcouple': couple(dump), 'tau-p': dump/1000,
                                 'ref-p': ' '.join('%.12g' % (axes[a][0]*ATM if a in axes else 0) for a in 'xyz')+' 0 0 0',
                                 'compressibility': ' '.join('%.12g' % (compressibility if a in axes else 0)
                                                             for a in 'xyz')+' 0 0 0'})
        return ''.join('%s = %s\n' % item for item in opts.items())

    @staticmethod
    def _frame_stride(md, wf):
        stride = md.dump_freq
        if getattr(wf, 'msd', False) or getattr(wf, 'dipole', False):
            stride = np.gcd(stride, md.thermo_freq)
        if getattr(wf, 'rg', False):
            stride = np.gcd(stride, wf.rg_ave_length)
        return int(stride)

    def exec(self, input_file=None, output_file=None, omp=1, mpi=0, gpu=0,
             return_cmd=False, intel='off', opt='off', resume=False):
        """
        Gromacs.exec

        Run the stages of a manifest. A single point follows a final (or
        absent) minimization to provide forces. With resume=True, completed
        stages are reused and an interrupted stage continues from its
        checkpoint; the manifest, molecule snapshot and gpu must be unchanged.
        gpu=1 offloads to one GPU with GPU numerics (see _mdp): PME order 4,
        a buffered pair list, and coupling every 10 steps; use a
        mixed-precision build. gpu=0 keeps the exact CPU reference numerics.
        Returns a CompletedProcess or raises on failure.
        """
        if gpu not in (0, 1):
            raise NotImplementedError('The GROMACS backend supports zero or one GPU')
        if return_cmd:
            raise NotImplementedError('Use the manifest and native stage files for GROMACS execution')
        if isinstance(input_file, list):
            raise NotImplementedError('Run conformers one at a time with GROMACS')
        manifest_path = self._path(input_file or self.input_file)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('schema') != 1 or manifest.get('engine') != 'gromacs':
            raise ValueError('Unsupported GROMACS manifest schema')
        md = SimpleNamespace(**manifest['options'])
        mol = _read_molecule(manifest['molecule'])
        cid = manifest['confId']
        root = Path(str(manifest_path)+'.gmx')
        digest = hashlib.sha256(manifest_path.read_bytes()+Path(manifest['molecule']).read_bytes()
                                +(b'gpu' if gpu else b'')).hexdigest()
        if resume:
            if not root.exists() or (root/'input.sha256').read_text() != digest:
                raise ValueError('Cannot resume: no previous run, or its inputs have changed')
        elif root.exists():
            raise FileExistsError('Run already exists: %s; use resume=True or a new directory' % root)
        else:
            root.mkdir()
            (root/'input.sha256').write_text(digest)
        # Never read a previous successful run's result after a failed new run.
        result_path = self._path(md.log_file+'.gmx.json')
        result_path.unlink(missing_ok=True)
        version = self._command(['--version'], root).stdout
        (root/'version.txt').write_text(version)
        if mpi > 1 and 'thread_mpi' in version:
            raise ValueError('External MPI requires an MPI-enabled GROMACS executable')

        stages = [SimpleNamespace(**w) for w in manifest['workflow']]
        if not stages or stages[-1].type == 'minimize':
            stages.append(None)
        dfs, xtcs, elapsed, step = [], [], 0.0, 0
        for index, wf in enumerate(stages):
            stage = root/('stage_%03d' % index if wf is not None else 'single_point')
            completed = stage/'completed.json'
            if completed.exists():
                mol = _read_molecule(stage/'final.json')
                dfs.append(self._energy_frame(stage, mol, wf, elapsed, step))
                state = json.loads(completed.read_text())
                elapsed, step = state['elapsed_fs'], state['step']
                if (stage/'unwrapped.xtc').exists():
                    xtcs.append(stage/'unwrapped.xtc')
                continue
            continuing = (stage/'run.cpt').exists()
            if not continuing:
                shutil.rmtree(stage, ignore_errors=True)
                stage.mkdir()
                temp = getattr(wf, 'set_init_velocity', None)
                if index == 0 and md.set_init_velocity is not None:
                    calc.set_velocity(mol, md.set_init_velocity)
                if temp is not None:
                    calc.set_velocity(mol, temp)
                _write_molecule(mol, stage/'initial.json')
                (stage/'system.top').write_text(topology(mol, shake=getattr(wf, 'shake', False),
                                                         electrostatics=md.pair_style != 'lj/cut'))
                write_gro(mol, stage/'input.gro', cid)
                (stage/'run.mdp').write_text(self._mdp(md, wf, mol, elapsed, gpu=bool(gpu)))
                self._command(['grompp', '-f', 'run.mdp', '-c', 'input.gro', '-p', 'system.top',
                               '-o', 'run.tpr'], stage)
            else:
                mol = _read_molecule(stage/'initial.json')
            command = ['mdrun', '-s', 'run.tpr', '-deffnm', 'run', '-c', 'final.g96',
                       '-ntomp', str(max(1, omp))]
            # With a GPU, mdrun also offloads PME, bonded and update when supported.
            command += ['-nb', 'gpu'] if gpu else ['-nb', 'cpu', '-pme', 'cpu', '-bonded', 'cpu']
            if 'thread_mpi' in version:
                command += ['-ntmpi', '1']
            if continuing:
                command += ['-cpi', 'run.cpt', '-append']
            self._command(command, stage, mpi=mpi, omp=omp)
            df = self._energy_frame(stage, mol, wf, elapsed, step)
            if wf is not None and wf.type == 'md':
                end = elapsed+wf.step*wf.time_step
                if not (stage/'final.g96').exists() or abs(df['Time'].iat[-1]-end) > wf.time_step/4:
                    raise RuntimeError('GROMACS stopped before the requested final time; '
                                       'rerun with resume=True')
            if wf is not None:
                mol = self._final_state(stage, mol, cid, wf, md, elapsed, step)
                df = self._add_observables(df, stage, step, wf)
            _write_molecule(mol, stage/'final.json')
            if wf is not None and wf.type == 'md':
                elapsed, step = end, step+wf.step
                xtcs.append(stage/'unwrapped.xtc')
            elif wf is not None:
                step = int(df.index[-1])
            _json(completed, {'elapsed_fs': elapsed, 'step': step})
            dfs.append(df)

        origin = np.array([mol.cell.xlo, mol.cell.ylo, mol.cell.zlo])
        lengths = np.array([mol.cell.dx, mol.cell.dy, mol.cell.dz])
        force = np.full((mol.GetNumAtoms(), 3), np.nan)
        if stages[-1] is None:
            self._command(['traj', '-f', 'run.trr', '-s', 'run.tpr', '-of', 'force.xvg', '-fp'],
                          stage, stdin='System\n')
            data, _ = read_xvg(stage/'force.xvg')
            force = data[-1, 1:].reshape(-1, 3)/(KCAL*10)
        xyz = mol.GetConformer(cid).GetPositions()
        velocity = np.array([[a.GetDoubleProp(k) if a.HasProp(k) else 0.0
                              for k in ('vx', 'vy', 'vz')] for a in mol.GetAtoms()])
        state_path = self._path((md.outstr or 'final_state')+'.npz')
        np.savez(state_path, unwrapped=xyz, wrapped=(xyz-origin) % lengths + origin,
                 cell=np.column_stack([origin, origin+lengths]), velocity=velocity, force=force)
        if md.write_data:
            # A real LAMMPS data export supports existing structure/charge readers.
            lammps.MolToLAMMPSdata(mol, str(self._path(md.write_data)), confId=cid,
                                   velocity=all(a.HasProp('vx') for a in mol.GetAtoms()))
        trajectory = None
        if md.xtc_file and xtcs:
            trajectory = str(self._path(md.xtc_file))
            self._concatenate(xtcs, trajectory, origin, md.dump_freq)
        payload = {'schema': 1, 'engine': 'gromacs', 'version': version,
                   'interaction_profile': md.interaction_profile, 'gpu': bool(gpu),
                   'manifest': str(manifest_path),
                   'charges': [a.GetDoubleProp('AtomicCharge') for a in mol.GetAtoms()],
                   'frames': [df.reset_index().to_dict(orient='list') for df in dfs],
                   'state': str(state_path), 'trajectory': trajectory,
                   'molecule': str(stage/'final.json')}
        _json(result_path, payload)
        return subprocess.CompletedProcess([self.solver_path, 'mdrun'], 0)

    def _final_state(self, stage, mol, cid, wf, md, elapsed, step_offset):
        """Copy the final native state into mol, preserving unwrapped coordinates."""
        xyz, vel, lengths = read_g96(stage/'final.g96')
        origin = np.array([mol.cell.xlo, mol.cell.ylo, mol.cell.zlo])
        initial = mol.GetConformer(cid).GetPositions()-origin
        if wf.type == 'minimize':
            # Mixed-precision builds write the box rounded to single precision.
            if not np.allclose(lengths, [mol.cell.dx, mol.cell.dy, mol.cell.dz], rtol=1e-6, atol=0):
                raise RuntimeError('GROMACS changed the box during minimization')
            # Minimization displacements are far below half a box length.
            reference = initial
        else:
            observer = None
            if getattr(wf, 'rg', False) or getattr(wf, 'msd', False) or getattr(wf, 'dipole', False):
                observer = Observables(mol, initial, wf, md, elapsed)
            reference = unwrap_trajectory(stage/'run.xtc', initial, stage/'unwrapped.xtc', observer)
            if observer is not None:
                if observer.rg_window:
                    observer.write_rg(self._path(wf.rg_file), step_offset)
                if observer.msd_window:
                    _json(stage/'msd.json', observer.msd)
                if observer.charges is not None:
                    _json(stage/'dipole.json', observer.dipole)
        xyz += lengths*np.rint((reference-xyz)/lengths)
        xyz += origin
        for i, atom in enumerate(mol.GetAtoms()):
            mol.GetConformer(cid).SetAtomPosition(i, xyz[i].tolist())
            if wf.type == 'md':
                for axis, value in zip(('vx', 'vy', 'vz'), vel[i]):
                    atom.SetDoubleProp(axis, float(value))
        mol.cell = utils.Cell(origin[0]+lengths[0], origin[0], origin[1]+lengths[1], origin[1],
                              origin[2]+lengths[2], origin[2])
        return mol

    @staticmethod
    def _concatenate(xtcs, output, origin, dump_freq):
        """Write one unwrapped trajectory in RadonPy coordinates, dropping repeated stage boundaries."""
        from mdtraj.formats import XTCTrajectoryFile
        last_time = -np.inf
        with XTCTrajectoryFile(output, 'w') as writer:
            for path in xtcs:
                with XTCTrajectoryFile(str(path)) as reader:
                    while True:
                        xyz, time, step, box = reader.read(n_frames=1000)
                        if len(xyz) == 0:
                            break
                        keep = (time > last_time+1e-6) & (step % dump_freq == 0)
                        if keep.any():
                            writer.write(xyz[keep]+np.asarray(origin, dtype=np.float32)/10,
                                         time=time[keep], step=step[keep], box=box[keep])
                            last_time = time[keep][-1]

    def _energy_frame(self, stage, mol, wf, elapsed, step_offset):
        if not (stage/'energy.xvg').exists() or not (stage/'completed.json').exists():
            menu = self._command(['energy', '-f', 'run.edr', '-o', 'energy.xvg', '-dp'],
                                 stage, stdin='Potential\n0\n').stdout
            terms = re.findall(r'(?<!\S)(\d+)\s+(?=[A-Za-z])',
                               menu.split('Select the terms')[1].split('Last energy frame')[0])
            self._command(['energy', '-f', 'run.edr', '-o', 'energy.xvg', '-dp'],
                          stage, stdin=' '.join(terms)+'\n0\n')
        values, names = read_xvg(stage/'energy.xvg')
        raw = pd.DataFrame(values[:, 1:], columns=names)

        def val(name, default=0.0):
            return raw[name].to_numpy() if name in raw else np.full(len(raw), default)

        dynamics = wf is not None and wf.type == 'md'
        # For minimization, the xvg "time" column is the iteration count.
        result = pd.DataFrame({'Time': values[:, 0]*1000 if dynamics else np.full(len(raw), elapsed)})
        mapping = {'Bond': 'E_bond', 'Angle': 'E_angle', 'Proper Dih.': 'E_dihed',
                   'Per. Imp. Dih.': 'E_impro', 'Potential': 'PotEng', 'Coul. recip.': 'E_long'}
        for key, target in mapping.items():
            result[target] = val(key)/KCAL
        result['E_vdwl'] = (val('LJ (SR)')+val('LJ-14'))/KCAL
        result['E_coul'] = (val('Coulomb (SR)')+val('Coulomb-14'))/KCAL
        result['KinEng'] = val('Kinetic En.')/KCAL
        result['TotEng'] = result['PotEng']+result['KinEng']
        result['Temp'] = val('Temperature')
        result['Press'] = val('Pressure')/ATM
        result['Volume'] = val('Volume', mol.cell.volume/1000)*1000
        result['Enthalpy'] = result['TotEng']+result['Press']*result['Volume']*ATM_A3_KCAL
        result['Density'] = calc.mol_mass(mol)/6.02214076e23/result['Volume']*1e24
        for axis, length in zip('XYZ', (mol.cell.dx, mol.cell.dy, mol.cell.dz)):
            result['L'+axis.lower()] = val('Box-'+axis, length/10)*10
        for a, b in [('X', 'X'), ('Y', 'Y'), ('Z', 'Z'), ('X', 'Y'), ('X', 'Z'), ('Y', 'Z')]:
            result['P'+a.lower()+b.lower()] = val('Pres-'+a+b)/ATM
        if dynamics:
            result.index = np.rint((values[:, 0]*1000-elapsed)/wf.time_step).astype(int)+step_offset
        else:
            result.index = np.rint(values[:, 0]).astype(int)+step_offset
        result.index.name = 'Step'
        return self._add_observables(result, stage, step_offset, wf)

    @staticmethod
    def _add_observables(frame, stage, step_offset, wf):
        if (stage/'msd.json').exists():
            # As LAMMPS prints f_msd: zero before the first window, then the last window average.
            msd = dict((int(k)+step_offset, v) for k, v in json.loads((stage/'msd.json').read_text()))
            frame['v_msd'] = pd.Series(msd, dtype=float).reindex(frame.index).ffill().fillna(0.0).to_numpy()
        if (stage/'dipole.json').exists():
            # Frames are written at every thermo step, so each row has its instantaneous dipole.
            dipole = pd.DataFrame({int(k)+step_offset: v for k, v in json.loads((stage/'dipole.json').read_text())},
                                  index=['v_mux', 'v_muy', 'v_muz']).T.reindex(frame.index)
            for column in dipole:
                frame[column] = dipole[column].to_numpy()
            frame['v_mu'] = np.sqrt(np.sum(dipole.to_numpy()**2, axis=1))
        if getattr(wf, 'efield', False) and (wf.efield_freq or wf.efield_rate):
            frame['v_efac'] = wf.efield_value*np.cos(2*np.pi*wf.efield_freq*frame['Time'].to_numpy()*1e-15)
        return frame

    def read_traj_simple(self, filename):
        """Return unwrapped/wrapped coordinates, cell, velocity and force, like LAMMPS."""
        with np.load(str(filename)+'.npz') as data:
            return tuple(data[k].copy() for k in ('unwrapped', 'wrapped', 'cell', 'velocity', 'force'))

    read_traj_last = read_traj_simple

    def run(self, md, mol=None, confId=0, input_file=None, output_file=None, last_data=None,
            last_str=None, omp=1, mpi=0, gpu=0, intel='off', opt='off', resume=False):
        """
        Gromacs.run

        Run a manifest and copy the final coordinates, velocities and cell of
        confId into mol. With resume=True, continue the existing manifest.
        """
        if last_data:
            md.write_data = last_data
        if last_str:
            md.outstr = last_str
        if not resume:
            # Like LAMMPS, minimization-only workflows carry no velocities.
            dynamics = any(wf.type == 'md' for wf in md.wf)
            self.make_dat(mol, confId=confId, file_name=md.dat_file, velocity=dynamics)
            self.make_input(md, file_name=input_file)
        else:
            md._gromacs_manifest = str(self._path(input_file or self.input_file))
        self.exec(input_file=input_file, omp=omp, mpi=mpi, gpu=gpu, resume=resume)
        uwstr, _, cell, vel, _ = self.read_traj_simple(self._path(md.outstr or 'final_state'))
        for i, atom in enumerate(mol.GetAtoms()):
            mol.GetConformer(confId).SetAtomPosition(i, uwstr[i].tolist())
            for axis, value in zip(('vx', 'vy', 'vz'), vel[i]):
                atom.SetDoubleProp(axis, float(value))
        mol.cell = utils.Cell(cell[0, 1], cell[0, 0], cell[1, 1], cell[1, 0], cell[2, 1], cell[2, 0])
        return calc.mol_trans_in_cell(mol, confId=confId)


def _pressure_axes(wf):
    """Per-axis (start, stop, damping) pressures of an anisotropic NPT stage as LAMMPS writes them, or None."""
    if wf.ensemble != 'npt':
        return None
    if not wf.p_aniso:
        # LAMMPS barostats the other axes of a strain-rate deformation (make_input_deform).
        if wf.deform == 'erate':
            return {a: (wf.p_start, wf.p_stop, wf.p_dump) for a in 'xyz' if a != wf.deform_axis}
        return None
    axes = {a: tuple(getattr(wf, 'p%s_%s' % (a, k), None) for k in ('start', 'stop', 'dump')) for a in 'xyz'}
    axes = {a: v for a, v in axes.items() if None not in v}
    # LAMMPS aniso: all three axes at the isotropic settings.
    return axes or {a: (wf.p_start, wf.p_stop, wf.p_dump) for a in 'xyz'}


def _efield(wf):
    """Electric field components (V/angstrom) by axis, as LAMMPS make_input_efield applies them."""
    if wf.efield_axis not in ('x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz'):
        raise ValueError('Unknown electric field axis: %s' % wf.efield_axis)
    field = {axis: wf.efield_value for axis in wf.efield_axis}
    for axis in 'xyz':
        value = getattr(wf, 'efield_'+axis, None)
        if value:
            if isinstance(value, str):
                raise NotImplementedError('Electric field variables are LAMMPS only')
            field[axis] = value
    return field


def _is_temperature(value, allow_none=False):
    if value is None:
        return allow_none
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


class Analyze(lammps.Analyze):
    """Read GROMACS results into the LAMMPS analysis interface."""
    def read_log(self, log_file, ignore_log=None):
        self.metadata = json.loads(Path(str(log_file)+'.gmx.json').read_text())
        return [pd.DataFrame(frame).set_index('Step') for frame in self.metadata['frames']]

    def get_partial_charges(self, dat_file=None):
        self.charges = np.asarray(self.metadata['charges'])
        return self.charges
