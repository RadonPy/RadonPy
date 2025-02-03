#  Copyright (c) 2024. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# ******************************************************************************
# sim.md module
# ******************************************************************************

import os
import numpy as np
from rdkit import Geometry as Geom
from .md_wrapper import MD_solver, MD_analyzer
from ..core import calc, utils

__version__ = '0.2.10'


class MD():
    def __init__(self, **kwargs):
        self.idx = kwargs.get('idx', None)

        self.input_file = kwargs.get('input_file', 'radon_lmp.in' if self.idx is None else 'radon_lmp_%i.in' % self.idx)
        self.log_file = kwargs.get('log_file', 'radon_md.log' if self.idx is None else 'radon_md_%i.log' % self.idx)
        self.dat_file = kwargs.get('dat_file', 'radon_md_lmp.data' if self.idx is None else 'radon_md_lmp_%i.data' % self.idx)
        self.dump_file = kwargs.get('dump_file', 'radon_md.dump' if self.idx is None else 'radon_md_%i.dump' % self.idx)
        self.xtc_file = kwargs.get('xtc_file', None)
        self.rst1_file = kwargs.get('rst1_file', 'radon_md_1.rst' if self.idx is None else 'radon_md_1_%i.rst' % self.idx)
        self.rst2_file = kwargs.get('rst2_file', 'radon_md_2.rst' if self.idx is None else 'radon_md_2_%i.rst' % self.idx)
        self.outstr = kwargs.get('outstr', 'radon_md_last.dump' if self.idx is None else 'radon_md_last_%i.dump' % self.idx)
        self.write_data = kwargs.get('write_data', 'radon_md_last.data' if self.idx is None else 'radon_md_last_%i.data' % self.idx)

        self.dump_freq = kwargs.get('dump_freq', 1000)
        self.dump_style = kwargs.get('dump_style', 'id type mol x y z ix iy iz vx vy vz')
        self.rst = kwargs.get('rst', True)
        self.rst_freq = kwargs.get('rst_freq', 10000)
        self.thermo_freq = kwargs.get('thermo_freq', 1000)
#        self.thermo_style = kwargs.get('thermo_style', 'custom step time temp press enthalpy etotal ke pe ebond eangle edihed eimp evdwl ecoul elong etail vol lx ly lz density pxx pyy pzz pxy pxz pyz')
        self.thermo_style = kwargs.get('thermo_style', ['step', 'time', 'temp', 'press', 'enthalpy', 'etotal', 'ke', 'pe', 'ebond', 'eangle',
                                                        'edihed', 'eimp', 'evdwl', 'ecoul', 'elong', 'etail', 'vol', 'lx', 'ly', 'lz',
                                                        'density', 'pxx', 'pyy', 'pzz', 'pxy', 'pxz', 'pyz'])
        self.boundary = kwargs.get('boundary', 'p p p')
        self.pbc = kwargs.get('pbc', True)
        self.units = kwargs.get('units', 'real')
        self.atom_style = kwargs.get('atom_style', 'full')
        self.pair_style = kwargs.get('pair_style', 'lj/charmm/coul/long')
        self.pair_style_nonpbc = kwargs.get('pair_style_nonpbc', 'lj/charmm/coul/charmm')
        self.cutoff_in = kwargs.get('cutoff_in', 8.0)
        self.cutoff_out = kwargs.get('cutoff_out', 12.0)
        self.kspace_style = kwargs.get('kspace_style', 'pppm')
        self.kspace_style_accuracy = kwargs.get('kspace_style_accuracy', '1e-6')
        self.dielectric = kwargs.get('dielectric', 1.0)
        self.bond_style = kwargs.get('bond_style', 'harmonic')
        self.angle_style = kwargs.get('angle_style', 'harmonic')
        self.dihedral_style = kwargs.get('dihedral_style', 'fourier')
        self.improper_style = kwargs.get('improper_style', 'cvff')
        self.special_bonds = kwargs.get('special_bonds', 'amber')
        self.pair_modify = kwargs.get('pair_modify', 'mix arithmetic')
        self.neighbor = kwargs.get('neighbor', '2.0 bin')
        self.neigh_modify = kwargs.get('neigh_modify', 'delay 0 every 1 check yes')
        self.time_step = kwargs.get('time_step', 1.0)
        self.drude = kwargs.get('drude', False)
        self.set_init_velocity = kwargs.get('set_init_velocity', None)
        self.log_append = kwargs.get('log_append', True)
        self.wf = []
        self.add = []
        self.add_f = []

        if kwargs.get('mol') is not None:
            mol = kwargs.get('mol')
            if mol.HasProp('pair_style'):
                if mol.GetProp('pair_style') == 'lj':
                    self.pair_style = 'lj/charmm/coul/long'
                    self.pair_style_nonpbc = 'lj/charmm/coul/charmm'
                else:
                    self.pair_style = mol.GetProp('pair_style')
                    self.pair_style_nonpbc = mol.GetProp('pair_style')
            if mol.HasProp('bond_style'):
                self.bond_style = mol.GetProp('bond_style')
            if mol.HasProp('angle_style'):
                self.angle_style = mol.GetProp('angle_style')
            if mol.HasProp('dihedral_style'):
                self.dihedral_style = mol.GetProp('dihedral_style')
            if mol.HasProp('improper_style'):
                self.improper_style = mol.GetProp('improper_style')


    def add_min(self, min_style='cg', etol=1.0e-4, ftol=1.0e-6, maxiter=10000, maxeval=100000):
        mini = Minimize(min_style=min_style, etol=etol, ftol=ftol, maxiter=maxiter, maxeval=maxeval)
        self.wf.append(mini)
        return True


    def add_md(self, ensemble, step, time_step=None, shake=False, **kwargs):
        if time_step is None: time_step = self.time_step
        dyna = Dynamics(ensemble=ensemble, step=step, time_step=time_step, shake=shake, **kwargs)
        self.wf.append(dyna)
        return True


    def add_drude(self):
        # In development
        # Polarizable drude model
        #self.drude = True
        return False


    def clear(self, work_dir):
        if os.path.exists(os.path.join(work_dir, self.input_file)):
            os.remove(os.path.join(work_dir, self.input_file))

        if os.path.exists(os.path.join(work_dir, self.log_file)):
            os.remove(os.path.join(work_dir, self.log_file))

        if os.path.exists(os.path.join(work_dir, self.dat_file)):
            os.remove(os.path.join(work_dir, self.dat_file))

        if os.path.exists(os.path.join(work_dir, self.dump_file)):
            os.remove(os.path.join(work_dir, self.dump_file))

        if self.xtc_file is not None:
            if os.path.exists(os.path.join(work_dir, self.xtc_file)):
                os.remove(os.path.join(work_dir, self.xtc_file))

        if self.rst:
            if os.path.exists(os.path.join(work_dir, self.rst1_file)):
                os.remove(os.path.join(work_dir, self.rst1_file))
            if os.path.exists(os.path.join(work_dir, self.rst2_file)):
                os.remove(os.path.join(work_dir, self.rst2_file))

        if self.outstr is not None:
            if os.path.exists(os.path.join(work_dir, self.outstr)):
                os.remove(os.path.join(work_dir, self.outstr))

        if self.write_data is not None:
            if os.path.exists(os.path.join(work_dir, self.write_data)):
                os.remove(os.path.join(work_dir, self.write_data))


class Minimize():
    def __init__(self, min_style='cg', etol=1.0e-4, ftol=1.0e-6, maxiter=10000, maxeval=100000):
        self.type = 'minimize'
        self.min_style = min_style
        self.etol = etol
        self.ftol = ftol
        self.maxiter = maxiter
        self.maxeval = maxeval


class Dynamics():
    def __init__(self, ensemble='nve', step=1000, time_step=None, shake=False, **kwargs):
        self.type = 'md'
        self.ensemble = ensemble
        self.nve_limit = kwargs.get('nve_limit', 0.0)
        self.step = step
        self.time_step = self.time_step if time_step is None else time_step
        self.t_start = kwargs.get('t_start', 300.0)
        self.t_stop = kwargs.get('t_stop', 300.0)
        self.t_dump = kwargs.get('t_dump', 100.0)
        self.p_start = kwargs.get('p_start', 1.0)
        self.p_stop = kwargs.get('p_stop', 1.0)
        self.p_dump = kwargs.get('p_dump', 1000.0)
        self.p_aniso = kwargs.get('p_aniso', False)
        self.px_start = kwargs.get('px_start', None)
        self.px_stop = kwargs.get('px_stop', None)
        self.px_dump = kwargs.get('px_dump', None)
        self.py_start = kwargs.get('py_start', None)
        self.py_stop = kwargs.get('py_stop', None)
        self.py_dump = kwargs.get('py_dump', None)
        self.pz_start = kwargs.get('pz_start', None)
        self.pz_stop = kwargs.get('pz_stop', None)
        self.pz_dump = kwargs.get('pz_dump', None)
        self.p_couple = kwargs.get('p_couple', None)
        self.p_nreset = kwargs.get('p_nreset', 1000)
        self.shake = shake
        self.rattle = kwargs.get('rattle', False)
        self.thermostat = kwargs.get('thermostat', 'Nose-Hoover')
        self.barostat = kwargs.get('barostat', 'Nose-Hoover')

        self.set_init_velocity = kwargs.get('set_init_velocity', None)
        self.chunk_mol = kwargs.get('chunk_mol', False)
        self.deform = kwargs.get('deform', None)
        self.efield = kwargs.get('efield', False)
        self.dipole = kwargs.get('dipole', False)
        self.rg = kwargs.get('rg', False)
        self.msd = kwargs.get('msd', False)
        self.momentum = kwargs.get('momentum', False)
        self.variable = kwargs.get('variable', False)
        self.timeave = kwargs.get('timeave', False)
        self.thermo_style = kwargs.get('thermo_style', [])
        self.thermo_freq = kwargs.get('thermo_freq', None)
        self.rerun = kwargs.get('rerun', False)

        self.add = kwargs.get('add', [])
        self.add_f = kwargs.get('add_f', [])


    def add_deform(self, dftype='scale', axis='x', **kwargs):
        # fix deform
        self.deform = dftype
        self.deform_scale = kwargs.get('deform_scale', 2.0)
        self.deform_fin_lo = kwargs.get('deform_fin_lo', -10.0)
        self.deform_fin_hi = kwargs.get('deform_fin_hi', 10.0)
        self.deform_axis = axis
        self.deform_remap = kwargs.get('remap', 'v')
        return False


    def add_efield(self, evalue=1.0, axis='x', freq=0.0, **kwargs):
        # fix efield
        self.efield = True
        self.efield_value = evalue
        self.efield_axis = axis
        self.efield_freq = freq
        self.efield_x = kwargs.get('ex', None)
        self.efield_y = kwargs.get('ey', None)
        self.efield_z = kwargs.get('ez', None)
        return False


    def add_dipole(self, **kwargs):
        # compute dipole/chunk
        self.dipole = True
        self.chunk_mol = True
        return False


    def add_rg(self, ave_length=1000, file='rg.profile', **kwargs):
        # compute gyration/chunk
        self.rg = True
        self.rg_ave_length = ave_length
        self.rg_file = file
        self.chunk_mol = True
        return False


    def add_msd(self, msd_freq=1000, **kwargs):
        # compute msd
        self.msd = True
        self.msd_freq = msd_freq
        return False


    def add_variable(self, var):
        # variable
        self.variable = True
        self.variable_name = []
        self.variable_style = []
        self.variable_args = []

        for v in var:
            self.variable_name.append(v[0])
            self.variable_style.append(v[1])
            self.variable_args.append(v[2:])

        return False


    def add_timeave(self, name=None, var=[], nevery=1, nfreq=1000, nstep=1000, nounfix=False):
        # fix ave/time
        self.timeave = True
        self.timeave_name = name
        self.timeave_var = var
        self.timeave_nevery = nevery
        self.timeave_nfreq = nfreq
        self.timeave_nstep = nstep
        self.timeave_nounfix = nounfix
        return False


    def add_atomave(self):
        # Under development
        # fix ave/atom
        return False


    def add_rerun(self, dump_file, keyword='dump x y z ix iy iz vx vy vz'):
        self.rerun = True
        self.rerun_dump = dump_file
        self.rerun_keyword = keyword
        return False
        

    def add_user(self, strings):
        self.add.append(strings)
        return False


    def unfix_user(self, strings):
        self.add_f.append(strings)
        return False


def quick_energy(mol, confId=0, force=True, idx=None, tmp_clear=False,
                solver='lammps', solver_path=None, work_dir=None, omp=1, mpi=0, gpu=0):
    """
    MD.quick_energy

    Calculate potential energy and force by MD solver

    Args:
        mol: RDKit Mol object
    
    Optional args:
        confId: Target conformer ID (int)
        force: Compute force (boolean)
        solver: lammps (str) 
        solver_path: File path of solver (str) 
        work_dir: Path of work directory (str) 

    Returns:
        energy (float, kcal/mol)
        force (float, numpy.ndarray, kcal/(mol angstrom))

    """

    sol = MD_solver(md_solver=solver, work_dir=work_dir, solver_path=solver_path, idx=idx)

    md = MD(idx=idx, mol=mol)
    if not hasattr(mol, 'cell'):
        md.pbc = False
        calc.centering_mol(mol, confId=confId)
        
    sol.make_dat(mol, confId=confId, file_name=md.dat_file, velocity=False)
    sol.make_input(md)
    cp = sol.exec(omp=omp, mpi=mpi, gpu=gpu)
    if cp.returncode != 0 and (
                (md.write_data is not None and not os.path.exists(os.path.join(work_dir, md.write_data)))
                or (md.outstr is not None and not os.path.exists(os.path.join(work_dir, md.outstr)))
            ):
        utils.radon_print('Error termination of %s. Return code = %i' % (sol.get_name, cp.returncode), level=3)
        return None

    anal = MD_analyzer(log_file=os.path.join(sol.work_dir, md.log_file))
    energy = anal.dfs[-1]['PotEng'].iat[-1]

    if force:
        _, _, _, _, force = sol.read_traj_simple(os.path.join(sol.work_dir, md.outstr))
        if tmp_clear: md.clear(work_dir)
        return energy, force
    else:
        if tmp_clear: md.clear(work_dir)
        return energy


def quick_min(mol, confId=0, min_style='cg', idx=None, tmp_clear=False,
            solver='lammps', solver_path=None, work_dir=None, omp=1, mpi=0, gpu=0, **kwargs):
    """
    MD.quick_min

    Geometry optimization by MD solver

    Args:
        mol: RDKit Mol object
    
    Optional args:
        confId: Target conformer ID (int)
        min_style: cg, sd, hftn, fire (str)
        solver: lammps (str) 
        solver_path: File path of solver (str) 
        work_dir: Path of work directory (str) 

    Returns:
        energy (float, kcal/mol)
        Unwrapped coordinates (float, numpy.ndarray, angstrom)
    """
    mol_copy = utils.deepcopy_mol(mol)

    sol = MD_solver(md_solver=solver, work_dir=work_dir, solver_path=solver_path, idx=idx)

    md = MD(idx=idx, mol=mol_copy)
    if not hasattr(mol_copy, 'cell'):
        md.pbc = False
        calc.centering_mol(mol_copy, confId=confId)

    etol = kwargs.get('etol', 1.0e-4)
    ftol = kwargs.get('ftol', 1.0e-6)
    maxiter = kwargs.get('maxiter', 10000)
    maxeval = kwargs.get('maxeval', 100000)
    md.add_min(min_style=min_style, etol=etol, ftol=ftol, maxiter=maxiter, maxeval=maxeval)

    sol.make_dat(mol_copy, confId=confId, file_name=md.dat_file, velocity=False)

    mol_copy = sol.run(md, mol=mol_copy, confId=confId, last_data=md.write_data, last_str=md.outstr, omp=omp, mpi=mpi, gpu=gpu)

    anal = MD_analyzer(log_file=os.path.join(sol.work_dir, md.log_file))
    energy = anal.dfs[-1]['PotEng'].iat[-1]
    uwstr, wstr, _, _, _ = sol.read_traj_simple(os.path.join(sol.work_dir, md.outstr))

    if tmp_clear: md.clear(work_dir)

    return mol_copy, energy, uwstr


def quick_min_all(mol, min_style='cg', tmp_clear=False,
            solver='lammps', solver_path=None, work_dir=None, omp=1, mpi=1, gpu=0, mp=1, **kwargs):
    """
    MD.quick_min_all

    Geometry optimization by MD solver for all conformers

    Args:
        mol: RDKit Mol object
    
    Optional args:
        min_style: cg, sd, hftn, fire (str)
        solver: lammps (str) 
        solver_path: File path of solver (str) 
        work_dir: Path of work directory (str) 

    Returns:
        energy (float, kcal/mol)
        Unwrapped coordinates (float, numpy.ndarray, angstrom)
    """
    mol_copy = utils.deepcopy_mol(mol)
    input_files = []
    md_list = []
    uwstr_list = []
    energies = []

    exec_i = np.append(np.arange(mp, mol_copy.GetNumConformers(), mp), mol_copy.GetNumConformers()).tolist()

    for i in range(mol_copy.GetNumConformers()):
        sol = MD_solver(md_solver=solver, work_dir=work_dir, solver_path=solver_path, idx=i)

        md = MD(idx=i, mol=mol_copy)
        if not hasattr(mol_copy, 'cell'):
            md.pbc = False
            calc.centering_mol(mol_copy, confId=i)

        etol = kwargs.get('etol', 1.0e-4)
        ftol = kwargs.get('ftol', 1.0e-6)
        maxiter = kwargs.get('maxiter', 10000)
        maxeval = kwargs.get('maxeval', 100000)
        md.add_min(min_style=min_style, etol=etol, ftol=ftol, maxiter=maxiter, maxeval=maxeval)

        sol.make_dat(mol_copy, confId=i, file_name=md.dat_file, velocity=False)
        sol.make_input(md)

        input_files.append(sol.input_file)
        md_list.append(md)

        if int(i+1) in exec_i:
            init = i - mp + 1
            last = i + 1

            cp = sol.exec(input_file=input_files[init:last], omp=omp, mpi=mpi, gpu=gpu)

            if cp.returncode != 0 and (
                    (md_list[init].write_data is not None and not os.path.exists(os.path.join(work_dir, md_list[init].write_data)))
                    or (md_list[init].outstr is not None and not os.path.exists(os.path.join(work_dir, md_list[init].outstr)))
                ):
                if os.path.isfile(os.path.join(sol.work_dir, sol.output_file)):
                    with open(os.path.join(sol.work_dir, sol.output_file), 'r') as fh:
                        utils.radon_print('%s' % fh.read())
                utils.radon_print('Error termination of %s. Return code = %i' % (sol.get_name, cp.returncode), level=3)
                return None

            for confId in range(init, last):
                md = md_list[confId]
                uwstr, wstr, _, _, _ = sol.read_traj_simple(os.path.join(sol.work_dir, md.outstr))

                for j in range(mol_copy.GetNumAtoms()):
                    mol_copy.GetConformer(confId).SetAtomPosition(j, Geom.Point3D(uwstr[j, 0], uwstr[j, 1], uwstr[j, 2]))

                if hasattr(mol_copy, 'cell'):
                    mol_copy = calc.mol_trans_in_cell(mol_copy, confId=confId)

                anal = MD_analyzer(log_file=os.path.join(sol.work_dir, md.log_file))
                energy = anal.dfs[-1]['PotEng'].iat[-1]

                uwstr_list.append(uwstr)
                energies.append(energy)

                if tmp_clear:
                    md.clear(work_dir)

    return mol_copy, energies, uwstr_list


def quick_rw(mol, confId=0, step=1000, time_step=0.2, limit=0.1, shake=False, idx=None, tmp_clear=False,
            solver='lammps', solver_path=None, work_dir=None, omp=1, mpi=0, gpu=0, **kwargs):
    """
    MD.quick_rw

    Geometry optimization by MD solver for random walk process

    Args:
        mol: RDKit Mol object
    
    Optional args:
        confId: Target conformer ID (int)
        step: Number of MD steps (int)
        solver: lammps (str) 
        solver_path: File path of solver (str) 
        work_dir: Path of work directory (str) 

    Returns:
        Unwrapped coordinates (float, numpy.ndarray, angstrom)
    """
    mol_copy = utils.deepcopy_mol(mol)

    sol = MD_solver(md_solver=solver, work_dir=work_dir, solver_path=solver_path, idx=idx)

    md = MD(idx=idx, mol=mol_copy)
    if not hasattr(mol_copy, 'cell'):
        md.pbc = False
        calc.centering_mol(mol_copy, confId=confId)

    md.dielectric = kwargs.get('dielectric', 1.0)
    md.add_min(min_style='cg')
    md.add_md('nve', step, time_step=time_step, shake=shake, nve_limit=limit, **kwargs)
    #md.add_min(min_style='fire')

    sol.make_dat(mol_copy, confId=confId, file_name=md.dat_file, velocity=False)
    sol.make_input(md)
    cp = sol.exec(omp=omp, mpi=mpi, gpu=gpu)
    if cp.returncode != 0 and (
                (md.write_data is not None and not os.path.exists(os.path.join(work_dir, md.write_data)))
                or (md.outstr is not None and not os.path.exists(os.path.join(work_dir, md.outstr)))
            ):
        utils.radon_print('Error termination of %s. Return code = %i' % (sol.get_name, cp.returncode), level=3)
        return None

    uwstr, wstr, _, _, _ = sol.read_traj_simple(os.path.join(sol.work_dir, md.outstr))

    for i in range(mol_copy.GetNumAtoms()):
        mol_copy.GetConformer(confId).SetAtomPosition(i, Geom.Point3D(uwstr[i, 0], uwstr[i, 1], uwstr[i, 2]))

    if hasattr(mol_copy, 'cell'):
        mol_copy = calc.mol_trans_in_cell(mol_copy, confId=confId)

    if tmp_clear: md.clear(work_dir)

    return mol_copy, uwstr


def quick_nve(mol, confId=0, step=2000, time_step=None, limit=0.0, shake=False, idx=None, tmp_clear=False,
            solver='lammps', solver_path=None, work_dir=None, omp=1, mpi=0, gpu=0, **kwargs):
    """
    MD.quick_nve

    MD simulation with NVE ensemble

    Args:
        mol: RDKit Mol object
    
    Optional args:
        confId: Target conformer ID (int)
        step: Number of MD steps (int)
        time_step: Set timestep of MD (float or None, fs)
        limit: NVE limit (float)
        shake: Use SHAKE (boolean)
        solver: lammps (str)
        solver_path: File path of solver (str)
        work_dir: Path of work directory (str)

    Returns:
        Unwrapped coordinates (float, numpy.ndarray, angstrom)
    """
    mol_copy = utils.deepcopy_mol(mol)

    sol = MD_solver(md_solver=solver, work_dir=work_dir, solver_path=solver_path, idx=idx)

    md = MD(idx=idx, mol=mol_copy)
    if not hasattr(mol_copy, 'cell'):
        md.pbc = False
        calc.centering_mol(mol_copy, confId=confId)

    md.add_md('nve', step, time_step=time_step, shake=shake, nve_limit=limit, **kwargs)

    sol.make_dat(mol_copy, confId=confId, file_name=md.dat_file)
    sol.make_input(md)
    cp = sol.exec(omp=omp, mpi=mpi, gpu=gpu)
    if cp.returncode != 0 and (
                (md.write_data is not None and not os.path.exists(os.path.join(work_dir, md.write_data)))
                or (md.outstr is not None and not os.path.exists(os.path.join(work_dir, md.outstr)))
            ):
        utils.radon_print('Error termination of %s. Return code = %i' % (sol.get_name, cp.returncode), level=3)
        return None

    uwstr, wstr, _, vel, _ = sol.read_traj_simple(os.path.join(sol.work_dir, md.outstr))

    for i in range(mol_copy.GetNumAtoms()):
        mol_copy.GetConformer(confId).SetAtomPosition(i, Geom.Point3D(uwstr[i, 0], uwstr[i, 1], uwstr[i, 2]))
        mol_copy.GetAtomWithIdx(i).SetDoubleProp('vx', vel[i, 0])
        mol_copy.GetAtomWithIdx(i).SetDoubleProp('vy', vel[i, 1])
        mol_copy.GetAtomWithIdx(i).SetDoubleProp('vz', vel[i, 2])

    if hasattr(mol_copy, 'cell'):
        mol_copy = calc.mol_trans_in_cell(mol_copy, confId=confId)

    if tmp_clear: md.clear(work_dir)

    return mol_copy, uwstr


def quick_nvt(mol, confId=0, step=2000, time_step=None, temp=300.0, f_temp=None, shake=False, idx=None, tmp_clear=False,
            solver='lammps', solver_path=None, work_dir=None, omp=1, mpi=0, gpu=0, **kwargs):
    """
    MD.quick_nvt

    MD simulation with NVT ensemble

    Args:
        mol: RDKit Mol object
    
    Optional args:
        confId: Target conformer ID (int)
        step: Number of MD steps (int)
        time_step: Set timestep of MD (float or None, fs)
        temp: Initial temperature (float, K)
        f_temp: Final temperature (float or None, K)
        shake: Use SHAKE (boolean)
        solver: lammps (str)
        solver_path: File path of solver (str)
        work_dir: Path of work directory (str)
        thermostat: Nose-Hoover, Langevin, Berendsen, csvr, or csld (str, default:Nose-Hoover)

    Returns:
        Unwrapped coordinates (float, numpy.ndarray, angstrom)
    """
    mol_copy = utils.deepcopy_mol(mol)

    sol = MD_solver(md_solver=solver, work_dir=work_dir, solver_path=solver_path, idx=idx)

    md = MD(idx=idx, mol=mol_copy)
    if not hasattr(mol_copy, 'cell'):
        md.pbc = False
        calc.centering_mol(mol_copy, confId=confId)

    if f_temp is None: f_temp = temp
    md.add_md('nvt', step, time_step=time_step, shake=shake, t_start=temp, t_stop=f_temp, **kwargs)

    sol.make_dat(mol_copy, confId=confId, file_name=md.dat_file)
    sol.make_input(md)
    cp = sol.exec(omp=omp, mpi=mpi, gpu=gpu)
    if cp.returncode != 0 and (
                (md.write_data is not None and not os.path.exists(os.path.join(work_dir, md.write_data)))
                or (md.outstr is not None and not os.path.exists(os.path.join(work_dir, md.outstr)))
            ):
        utils.radon_print('Error termination of %s. Return code = %i' % (sol.get_name, cp.returncode), level=3)
        return None

    uwstr, wstr, _, vel, _ = sol.read_traj_simple(os.path.join(sol.work_dir, md.outstr))

    for i in range(mol_copy.GetNumAtoms()):
        mol_copy.GetConformer(confId).SetAtomPosition(i, Geom.Point3D(uwstr[i, 0], uwstr[i, 1], uwstr[i, 2]))
        mol_copy.GetAtomWithIdx(i).SetDoubleProp('vx', vel[i, 0])
        mol_copy.GetAtomWithIdx(i).SetDoubleProp('vy', vel[i, 1])
        mol_copy.GetAtomWithIdx(i).SetDoubleProp('vz', vel[i, 2])

    if hasattr(mol_copy, 'cell'):
        mol_copy = calc.mol_trans_in_cell(mol_copy, confId=confId)

    if tmp_clear: md.clear(work_dir)

    return mol_copy, uwstr


def quick_npt(mol, confId=0, step=2000, time_step=None, temp=300.0, f_temp=None, press=1.0, f_press=None, shake=False, idx=None, tmp_clear=False,
            solver='lammps', solver_path=None, work_dir=None, omp=1, mpi=0, gpu=0, **kwargs):
    """
    MD.quick_npt

    MD simulation with NPT ensemble

    Args:
        mol: RDKit Mol object
    
    Optional args:
        confId: Target conformer ID (int)
        step: Number of MD steps (int)
        time_step: Set timestep of MD (float or None, fs)
        temp: Initial temperature (float, K)
        f_temp: Final temperature (float or None, K)
        press: Initial pressure (float, atm)
        f_press: Final pressure (float or None, atm)
        shake: Use SHAKE (boolean)
        solver: lammps (str)
        solver_path: File path of solver (str)
        work_dir: Path of work directory (str)
        thermostat: Nose-Hoover, Langevin, Berendsen, csvr, or csld (str, default:Nose-Hoover)
        barostat: Nose-Hoover, or Berendsen (str, default:Nose-Hoover)

    Returns:
        Unwrapped coordinates (float, numpy.ndarray, angstrom)
        Cell coordinates (float, numpy.ndarray, angstrom)
    """
    mol_copy = utils.deepcopy_mol(mol)

    sol = MD_solver(md_solver=solver, work_dir=work_dir, solver_path=solver_path, idx=idx)

    md = MD(idx=idx, mol=mol_copy)
    if not hasattr(mol_copy, 'cell'):
        md.pbc = False
        calc.centering_mol(mol_copy, confId=confId)

    if f_temp is None: f_temp = temp
    if f_press is None: f_press = press
    md.add_md('npt', step, time_step=time_step, shake=shake, t_start=temp, t_stop=f_temp, p_start=press, p_stop=f_press, **kwargs)

    sol.make_dat(mol_copy, confId=confId, file_name=md.dat_file)
    sol.make_input(md)
    cp = sol.exec(omp=omp, mpi=mpi, gpu=gpu)
    if cp.returncode != 0 and (
                (md.write_data is not None and not os.path.exists(os.path.join(work_dir, md.write_data)))
                or (md.outstr is not None and not os.path.exists(os.path.join(work_dir, md.outstr)))
            ):
        utils.radon_print('Error termination of %s. Return code = %i' % (sol.get_name, cp.returncode), level=3)
        return None

    uwstr, wstr, cell, vel, _ = sol.read_traj_simple(os.path.join(sol.work_dir, md.outstr))

    for i in range(mol_copy.GetNumAtoms()):
        mol_copy.GetConformer(confId).SetAtomPosition(i, Geom.Point3D(uwstr[i, 0], uwstr[i, 1], uwstr[i, 2]))
        mol_copy.GetAtomWithIdx(i).SetDoubleProp('vx', vel[i, 0])
        mol_copy.GetAtomWithIdx(i).SetDoubleProp('vy', vel[i, 1])
        mol_copy.GetAtomWithIdx(i).SetDoubleProp('vz', vel[i, 2])

    setattr(mol_copy, 'cell', utils.Cell(cell[0, 1], cell[0, 0], cell[1, 1], cell[1, 0], cell[2, 1], cell[2, 0]))
    mol_copy = calc.mol_trans_in_cell(mol_copy, confId=confId)

    if tmp_clear: md.clear(work_dir)

    return mol_copy, uwstr, cell


def quick_nph(mol, confId=0, step=2000, time_step=None, press=1.0, f_press=None, shake=False, idx=None, tmp_clear=False,
            solver='lammps', solver_path=None, work_dir=None, omp=1, mpi=0, gpu=0, **kwargs):
    """
    MD.quick_nph

    MD simulation with NPH ensemble

    Args:
        mol: RDKit Mol object
    
    Optional args:
        confId: Target conformer ID (int)
        step: Number of MD steps (int)
        time_step: Set timestep of MD (float or None, fs)
        press: Initial pressure (float, atm)
        f_press: Final pressure (float or None, atm)
        shake: Use SHAKE (boolean)
        solver: lammps (str)
        solver_path: File path of solver (str)
        work_dir: Path of work directory (str)
        barostat: Nose-Hoover, or Berendsen (str, default:Nose-Hoover)

    Returns:
        Unwrapped coordinates (float, numpy.ndarray, angstrom)
        Cell coordinates (float, numpy.ndarray, angstrom)
    """
    mol_copy = utils.deepcopy_mol(mol)

    sol = MD_solver(md_solver=solver, work_dir=work_dir, solver_path=solver_path, idx=idx)

    md = MD(idx=idx, mol=mol_copy)
    if not hasattr(mol_copy, 'cell'):
        md.pbc = False
        calc.centering_mol(mol_copy, confId=confId)

    if f_press is None: f_press = press
    md.add_md('nph', step, time_step=time_step, shake=shake, p_start=press, p_stop=f_press, **kwargs)

    sol.make_dat(mol_copy, confId=confId, file_name=md.dat_file)
    sol.make_input(md)
    cp = sol.exec(omp=omp, mpi=mpi, gpu=gpu)
    if cp.returncode != 0 and (
                (md.write_data is not None and not os.path.exists(os.path.join(work_dir, md.write_data)))
                or (md.outstr is not None and not os.path.exists(os.path.join(work_dir, md.outstr)))
            ):
        utils.radon_print('Error termination of %s. Return code = %i' % (sol.get_name, cp.returncode), level=3)
        return None

    uwstr, wstr, cell, vel, _ = sol.read_traj_simple(os.path.join(sol.work_dir, md.outstr))

    for i in range(mol_copy.GetNumAtoms()):
        mol_copy.GetConformer(confId).SetAtomPosition(i, Geom.Point3D(uwstr[i, 0], uwstr[i, 1], uwstr[i, 2]))
        mol_copy.GetAtomWithIdx(i).SetDoubleProp('vx', vel[i, 0])
        mol_copy.GetAtomWithIdx(i).SetDoubleProp('vy', vel[i, 1])
        mol_copy.GetAtomWithIdx(i).SetDoubleProp('vz', vel[i, 2])

    setattr(mol_copy, 'cell', utils.Cell(cell[0, 1], cell[0, 0], cell[1, 1], cell[1, 0], cell[2, 1], cell[2, 0]))
    mol_copy = calc.mol_trans_in_cell(mol_copy, confId=confId)

    if tmp_clear: md.clear(work_dir)

    return mol_copy, uwstr, cell

