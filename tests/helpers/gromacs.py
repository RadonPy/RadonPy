"""Small GROMACS reference runner for static and NVE validation only.

This is test infrastructure, independent of RadonPy's solver dispatch.
Native inputs and command output stay in the test's temporary directory.
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
from radonpy.core import utils

KCAL = 4.184


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


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
    if not np.all(np.isfinite(mol.GetConformer().GetPositions())):
        raise ValueError('Non-finite coordinates')


def topology(mol):
    """Convert assigned parameters without retyping atoms or assigning charges."""
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
        charge = atom.GetDoubleProp('AtomicCharge')
        rows.append('%d %s 1 MOL %s %d %.16g %.16g' %
                    (i+1, atom_types[i], atom.GetSymbol(), i+1, charge, atom.GetMass()))
    rows += ['', '[ bonds ]']
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        r0, k = bond.GetDoubleProp('ff_r0')/10, bond.GetDoubleProp('ff_k')*2*KCAL*100
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


def command(executable, args, work, stdin=None):
    executable = shutil.which(str(executable))
    if executable is None:
        raise FileNotFoundError('Set GROMACS_EXEC to a GROMACS executable')
    argv = [executable, *map(str, args)]
    result = subprocess.run(argv, cwd=work, input=stdin, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env={**os.environ, 'OMP_NUM_THREADS': '1'}, timeout=120)
    log = Path(work)/'commands.log'
    with log.open('a') as stream:
        stream.write(json.dumps(argv)+'\n'+result.stdout+'\n')
    if result.returncode:
        raise RuntimeError('GROMACS failed; see %s\n%s' % (log, result.stdout[-4000:]))
    return result.stdout


def run(mol, work, executable, *, steps=0, time_step=.1, stride=1):
    """Run a single point or unconstrained NVE in a fresh directory.

    Return native energy columns in kJ/mol and, for a single point, forces
    converted to kcal/(mol angstrom). This runner uses one CPU thread.
    """
    validate_molecule(mol)
    if steps < 0 or time_step <= 0 or stride < 1 or steps % stride:
        raise ValueError('Use nonnegative steps divisible by stride and a positive timestep')
    if min(mol.cell.dx, mol.cell.dy, mol.cell.dz) <= 24:
        raise ValueError('Each box length must exceed twice the 12 angstrom cutoff')
    if steps and not all(a.HasProp(k) for a in mol.GetAtoms() for k in ('vx', 'vy', 'vz')):
        raise ValueError('NVE validation requires explicit initial velocities')
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=False)
    (work/'system.top').write_text(topology(mol))
    write_gro(mol, work/'input.gro')
    options = {
        'integrator': 'md-vv', 'nsteps': steps, 'dt': time_step/1000,
        'cutoff-scheme': 'Verlet', 'pbc': 'xyz', 'rlist': 1.2,
        'verlet-buffer-tolerance': -1, 'nstlist': 1,
        'coulombtype': 'PME', 'rcoulomb': 1.2, 'coulomb-modifier': 'None',
        'vdwtype': 'Cut-off', 'vdw-modifier': 'None', 'rvdw': 1.2,
        'DispCorr': 'no', 'fourierspacing': .08, 'pme-order': 6,
        'ewald-rtol': 1e-8, 'constraints': 'none', 'gen-vel': 'no',
        'comm-mode': 'Linear', 'nstcomm': 1000,
        'nstenergy': stride, 'nstcalcenergy': 1, 'nstlog': stride,
        'nstxout': 0, 'nstvout': 0, 'nstfout': 0 if steps else 1,
        'nstxout-compressed': 0,
    }
    (work/'run.mdp').write_text(''.join('%s = %s\n' % item for item in options.items()))
    version = command(executable, ['--version'], work)
    (work/'version.txt').write_text(version)
    command(executable, ['grompp', '-f', 'run.mdp', '-c', 'input.gro',
                         '-p', 'system.top', '-o', 'run.tpr'], work)
    args = ['mdrun', '-s', 'run.tpr', '-deffnm', 'run', '-ntomp', '1', '-nb', 'cpu']
    if 'thread_mpi' in version:
        args += ['-ntmpi', '1']
    command(executable, args, work)
    command(executable, ['energy', '-f', 'run.edr', '-o', 'energy.xvg', '-dp'],
            work, stdin='Potential\nTotal-Energy\n0\n')
    values, names = read_xvg(work/'energy.xvg')
    if not np.isclose(values[-1, 0], steps*time_step/1000, rtol=0, atol=1e-8):
        raise RuntimeError('GROMACS stopped before the requested final time')
    energies = dict(zip(names, values[:, 1:].T))
    force = None
    if steps == 0:
        command(executable, ['traj', '-f', 'run.trr', '-s', 'run.tpr',
                             '-of', 'force.xvg', '-fp'], work, stdin='System\n')
        values, _ = read_xvg(work/'force.xvg')
        force = values[-1, 1:].reshape(mol.GetNumAtoms(), 3)/(KCAL*10)
    return energies, force


def single_point(mol, work, executable):
    energies, force = run(mol, work, executable)
    return energies['Potential'][-1]/KCAL, force
