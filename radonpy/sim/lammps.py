#  Copyright (c) 2024. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# ******************************************************************************
# sim.lammps module
# ******************************************************************************

import subprocess
import os
import math
import numpy as np
from scipy import stats
import pandas as pd
from matplotlib import pyplot as pp
from rdkit import Chem
from rdkit import Geometry as Geom
from ..core import calc, poly, const, utils
from ..ff import ff_class

__version__ = '0.2.10'

mdtraj_avail = True
try:
    import mdtraj
except ImportError:
    mdtraj_avail = False

check_package = {}


class LAMMPS():
    def __init__(self, work_dir=None, solver_path=None, check_lammps_package=True, **kwargs):
        self.work_dir = work_dir if work_dir else './'
        self.solver_path = solver_path if solver_path else const.lammps_exec

        self.idx = kwargs.get('idx', None)
        self.dat_file = kwargs.get('dat_file', 'radon_md_lmp.data' if self.idx is None else 'radon_md_lmp_%i.data' % self.idx)
        self.input_file = kwargs.get('input_file', 'radon_lmp.in' if self.idx is None else 'radon_lmp_%i.in' % self.idx)
        self.output_file = kwargs.get('output_file', 'log.lammps')

        self.package = {
            'omp': False,
            'intel': False,
            'opt': False,
            'gpu': False
        }

        global check_package
        if const.check_package_disable or not check_lammps_package:
            for k in self.package.keys():
                self.package[k] = True
        elif self.solver_path in check_package.keys():
            self.package = check_package[self.solver_path]
        else:
            self.check_package()
            check_package[self.solver_path] = self.package


    @property
    def get_name(self):
        return 'LAMMPS'


    def exec(self, input_file=None, output_file=None, omp=0, mpi=0, gpu=0, return_cmd=False, intel='off', opt='off'):
        """
        LAMMPS.exec

        Execute LAMMPS

        Optional args:
            input_file: Input file path (str)
            output_file: Output file (str)
            omp: Number of openMP thread (int)
            mpi: Number of MPI process (int)
            gpu: Num ber of GPU (int)

        Return:
            Return args of subprocess.run()
        """

        input_file = input_file if input_file else self.input_file
        output_file = output_file if output_file else self.output_file

        if omp != 0 and not self.package['omp']:
            omp = 0
            utils.radon_print('OPENMP package is not available. Parallel number of OPENMP is changed to zero.', level=2)
        if gpu != 0 and not self.package['gpu']:
            gpu = 0
            utils.radon_print('GPU package is not available. Parallel number of GPU is changed to zero.', level=2)

        if mpi == -1:
            mpi = utils.cpu_count()
            omp = 0
        elif omp == -1:
            omp = utils.cpu_count()

        intel_flag = False
        if intel == 'on':
            intel_flag = True
        elif intel == 'auto':
            intel_flag = self.package['intel']
        else:
            intel_flag = False

        opt_flag = False
        if opt == 'on':
            opt_flag = True
        elif opt == 'auto':
            opt_flag = self.package['opt']
        else:
            opt_flag = False

        if mpi > 0:
            mpi_cmd = const.mpi_cmd % (mpi)
        elif mpi == 0:
            mpi_cmd = ''
        elif mpi < 0:
            mpi_cmd = const.mpi_cmd

        if omp == 0:
            os.environ['OMP_NUM_THREADS'] = str(1)
        else:
            os.environ['OMP_NUM_THREADS'] = str(omp)

        if gpu > 0 and omp > 0:
            acc_str = '-sf gpu -pk gpu %i omp %i' % (gpu, omp)
        elif gpu > 0:
            acc_str = '-sf gpu -pk gpu %i' % (gpu)
        elif omp > 0:
            if intel_flag:
                acc_str = '-sf hybrid intel omp -pk omp %i' % (omp)
            elif opt_flag:
                acc_str = '-sf opt -pk omp %i' % (omp)
            else:
                acc_str = '-sf omp -pk omp %i' % (omp)
        else:
            if intel_flag:
                acc_str = '-sf intel'
            elif opt_flag:
                acc_str = '-sf opt'
            else:
                acc_str = ''

        if type(input_file) is list:
            for i, infile in enumerate(input_file):
                if i == 0:
                    lmp_cmd  = const.lmp_cmd % (self.solver_path, acc_str, infile, output_file)
                else:
                    lmp_cmd += str(' : -n %i ' % mpi) + str(const.lmp_cmd % (self.solver_path, acc_str, infile, output_file))
        else:
            lmp_cmd = const.lmp_cmd % (self.solver_path, acc_str, input_file, output_file)

        cmd = '%s %s' % (mpi_cmd, lmp_cmd)
        utils.radon_print(cmd)
        if return_cmd:
            return cmd

        cwd = os.getcwd()
        os.chdir(self.work_dir)
        cp = subprocess.run([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='UTF-8')
        with open(output_file, 'a') as fh:
            fh.write(cmd+'\n')
            fh.write(cp.stdout+'\n')
            fh.write(cp.stderr+'\n')
            fh.write('LAMMPS returncode = %s \n' % (str(cp.returncode)))

        os.chdir(cwd)

        return cp


    def run(self, md, mol=None, confId=0, input_file=None, output_file=None, last_data=None, last_str=None,
            omp=0, mpi=0, gpu=0, intel='off', opt='off'):

        input_file = input_file if input_file else self.input_file
        output_file = output_file if output_file else self.output_file
        last_data = last_data if last_data else md.write_data
        last_str = last_str if last_str else md.outstr

        self.make_input(md, file_name=input_file)

        if not os.path.isfile(os.path.join(self.work_dir, input_file)):
            utils.radon_print('Cannot write LAMMPS input file.', level=2)

        cp = self.exec(input_file=input_file, output_file=output_file, omp=omp, mpi=mpi, gpu=gpu, intel=intel, opt=opt)

        if cp.returncode != 0 and (
                    (last_data is not None and not os.path.exists(os.path.join(self.work_dir, last_data)))
                    or (last_str is not None and not os.path.exists(os.path.join(self.work_dir, last_str)))
                ):
            if os.path.isfile(os.path.join(self.work_dir, output_file)):
                with open(os.path.join(self.work_dir, output_file), 'r') as fh:
                    utils.radon_print('%s' % fh.read())
            utils.radon_print('Error termination of %s. Input file = %s; Data file = %s; Return code = %i;'
                % (self.get_name, input_file, md.dat_file, cp.returncode), level=3)
            return cp.returncode

        if isinstance(mol, Chem.Mol) and last_str is not None:
            uwstr, wstr, cell, vel, _ = self.read_traj_simple(os.path.join(self.work_dir, last_str))

            for i in range(mol.GetNumAtoms()):
                mol.GetConformer(confId).SetAtomPosition(i, Geom.Point3D(uwstr[i, 0], uwstr[i, 1], uwstr[i, 2]))
                mol.GetAtomWithIdx(i).SetDoubleProp('vx', vel[i, 0])
                mol.GetAtomWithIdx(i).SetDoubleProp('vy', vel[i, 1])
                mol.GetAtomWithIdx(i).SetDoubleProp('vz', vel[i, 2])

            if hasattr(mol, 'cell'):
                setattr(mol, 'cell', utils.Cell(cell[0, 1], cell[0, 0], cell[1, 1], cell[1, 0], cell[2, 1], cell[2, 0]))
                mol = calc.mol_trans_in_cell(mol, confId=confId)

        return mol


    def check_package(self):
        check_omp = False
        check_intel = False
        check_opt = False
        check_gpu = False

        try:
            mpi_cmd = const.mpi_cmd % 1
            mpi_cmd = mpi_cmd.split()[0]

            cmd = '%s %s -h' % (mpi_cmd, self.solver_path)
            cp = subprocess.run([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='UTF-8')
            lines = str(cp.stdout).splitlines()

            flag = False
            for l in lines:
                if 'Installed packages:' in l:
                    flag = True
                elif flag:
                    if 'OPENMP' in l or 'USER-OMP' in l:
                        utils.radon_print('OPENMP package is available.', level=1)
                        check_omp = True
                    if 'INTEL' in l:
                        utils.radon_print('INTEL package is available.', level=1)
                        check_intel = True
                    if 'OPT' in l:
                        utils.radon_print('OPT package is available.', level=1)
                        check_opt = True
                    if 'GPU' in l:
                        utils.radon_print('GPU package is available.', level=1)
                        check_gpu = True

        except:
            try:
                cmd = '%s -h' % self.solver_path
                cp = subprocess.run([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='UTF-8')
                lines = str(cp.stdout).splitlines()

                flag = False
                for l in lines:
                    if 'Installed packages:' in l:
                        flag = True
                    elif flag:
                        if 'OPENMP' in l or 'USER-OMP' in l:
                            utils.radon_print('OPENMP package is available.', level=1)
                            check_omp = True
                        if 'INTEL' in l:
                            utils.radon_print('INTEL package is available.', level=1)
                            check_intel = True
                        if 'OPT' in l:
                            utils.radon_print('OPT package is available.', level=1)
                            check_opt = True
                        if 'GPU' in l:
                            utils.radon_print('GPU package is available.', level=1)
                            check_gpu = True

            except:
                utils.radon_print('Could not obtain package information of LAMMPS.', level=2)
                return False

        self.package['omp'] = check_omp
        self.package['intel'] = check_intel
        self.package['opt'] = check_opt
        self.package['gpu'] = check_gpu

        return True


    def get_version(self):

        ver = None
        try:
            mpi_cmd = const.mpi_cmd % 1
            cmd = '%s %s -h' % (mpi_cmd, self.solver_path)
            cp = subprocess.run([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='UTF-8')
            lines = str(cp.stdout).splitlines()

            for l in lines:
                if 'Large-scale Atomic/Molecular Massively Parallel Simulator' in l:
                    ver = '%s%s%s' % (str(l.split()[6]), str(l.split()[7]), str(l.split()[8]))
        except:
            try:
                cmd = '%s -h' % self.solver_path
                cp = subprocess.run([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='UTF-8')
                lines = str(cp.stdout).splitlines()

                for l in lines:
                    if 'Large-scale Atomic/Molecular Massively Parallel Simulator' in l:
                        ver = '%s%s%s' % (str(l.split()[6]), str(l.split()[7]), str(l.split()[8]))
            except:
                return ver

        return ver
        

    def make_dat(self, mol, confId=0, file_name=None, dir_name=None, velocity=True, temp=300, drude=False):
        """
        LAMMPS.make_dat

        Make a LAMMPS data file from MD object

        Args:
            mol: RDKit Mol object

        Optional args:
            confId: Target conformer ID
            file_name: Name of a topology file (str)
            dir_name: Directory name (str)
            velocity: Generate initial velocity (boolean)
            temp: Temperature of initial velocity (float, K)
            drude: Polarizable Drude model (boolean)

        Return:
            Path of data file
        """

        file_name = file_name if file_name is not None else self.dat_file
        dir_name = dir_name if dir_name is not None else self.work_dir
        dat_file = os.path.join(dir_name, file_name)
        MolToLAMMPSdata(mol, dat_file, confId=confId, velocity=velocity, temp=temp, drude=drude)

        return dat_file


    def make_input(self, md, file_name=None, dir_name=None):
        """
        LAMMPS.make_input

        Make a LAMMPS input file from MD object

        Args:
            md: MD object

        Optional args:
            file_name: Name of an input file
            dir_name: Directory name

        Return:
            Strings of content in input files
        """

        indata = []
        if md.log_append:
            indata.append('log %s append' % (md.log_file))
        else:
            indata.append('log %s' % (md.log_file))
        indata.append('units %s' % (md.units))
        indata.append('atom_style %s' % (md.atom_style))
        if md.pbc:
            indata.append('boundary %s' % (md.boundary))
        else:
            indata.append('boundary f f f')

        indata.append('')
        if md.pbc:
            indata.append('pair_style %s %s %s' % (md.pair_style, str(md.cutoff_in), str(md.cutoff_out)))
            indata.append('kspace_style %s %s' % (md.kspace_style, md.kspace_style_accuracy))
        else:
            indata.append('pair_style %s %s %s' % (md.pair_style_nonpbc, str(md.cutoff_in), str(md.cutoff_out)))

        indata.append('dielectric %f' % (md.dielectric))
        indata.append('bond_style %s' % (md.bond_style))
        indata.append('angle_style %s' % (md.angle_style))
        indata.append('dihedral_style %s' % (md.dihedral_style))
        indata.append('improper_style %s' % (md.improper_style))
        indata.append('special_bonds %s' % (md.special_bonds))
        indata.append('pair_modify %s' % (md.pair_modify))
        indata.append('neighbor %s' % (md.neighbor))
        indata.append('neigh_modify %s' % (md.neigh_modify))
        indata.append('read_data %s' % (md.dat_file))
        indata.append('')
        indata.append('thermo_style custom %s' % ' '.join(md.thermo_style))
        indata.append('thermo_modify flush yes')
        indata.append('thermo %i' % (md.thermo_freq))

        if len(md.add) > 0:
            indata.append('')
            indata.extend(md.add)

        if md.dump_file:
            indata.append('')
            indata.append('dump dump0 all custom %i %s %s' % (md.dump_freq, md.dump_file, md.dump_style))

        if md.xtc_file:
            indata.append('dump xtc0 all xtc %i %s' % (md.dump_freq, md.xtc_file))
            indata.append('dump_modify xtc0 unwrap yes')

        if md.rst:
            indata.append('restart %i %s %s' % (md.rst_freq, md.rst1_file, md.rst2_file))

        # Generate initial velocity
        if type(md.set_init_velocity) is float or type(md.set_init_velocity) is int:
            indata.append('')
            indata.append('velocity all create %f %i mom yes rot yes dist gaussian' % (md.set_init_velocity, np.random.randint(1000, 999999)))

        # Generate "fix drude"
        if md.drude:
            self.make_input_drude(md, indata)

        for i, wf in enumerate(md.wf):
            if wf.type == 'minimize':
                indata.append('')

                if md.dump_file:
                    indata.append('undump dump0')
                if md.xtc_file:
                    indata.append('undump xtc0')

                indata.append('min_style %s' % (wf.min_style))
                indata.append('minimize %f %f %i %i' % (wf.etol, wf.ftol, wf.maxiter, wf.maxeval))
                indata.append('reset_timestep 0')

                if md.dump_file:
                    indata.append('dump dump0 all custom %i %s %s' % (md.dump_freq, md.dump_file, md.dump_style))
                if md.xtc_file:
                    indata.append('dump xtc0 all xtc %i %s' % (md.dump_freq, md.xtc_file))
                    indata.append('dump_modify xtc0 unwrap yes')

                
            elif wf.type == 'md':
                unfix = []
                seed = np.random.randint(1000, 999999)
                indata.append('')
                indata.append('timestep %f' % (wf.time_step))

                # Generate initial velocity
                if type(wf.set_init_velocity) is float or type(wf.set_init_velocity) is int:
                    indata.append('')
                    indata.append('velocity all create %f %i mom yes rot yes dist gaussian' % (wf.set_init_velocity, np.random.randint(1000, 999999)))

                # Generate compute chunk/atom molecule
                if wf.chunk_mol:
                    indata.append('compute cmol%i all chunk/atom molecule nchunk once limit 0 ids once compress no' % (i+1))
                    unfix.append('uncompute cmol%i' % (i+1))

                # Generate "fix efield"
                if wf.efield:
                    indata.append('')
                    indata.append('# efield')
                    self.make_input_efield(md, wf, i, indata, unfix)

                # Generate "compute dipole/chunk"
                if wf.dipole:
                    indata.append('')
                    indata.append('# dipole')
                    self.make_input_dipole(md, wf, i, indata, unfix)

                # Generate "compute gyration/chunk"
                if wf.rg:
                    indata.append('')
                    indata.append('# rg')
                    self.make_input_rg(md, wf, i, indata, unfix)

                # Generate "compute msd"
                if wf.msd:
                    indata.append('')
                    indata.append('# msd')
                    self.make_input_msd(md, wf, i, indata, unfix)

                # Generate "variable"
                if wf.variable:
                    indata.append('')
                    indata.append('# variable')
                    self.make_input_variable(md, wf, i, indata, unfix)

                # Generate "fix ave/time"
                if wf.timeave:
                    indata.append('')
                    indata.append('# ave/time')
                    self.make_input_timeave(md, wf, i, indata, unfix)


                if len(wf.add) > 0:
                    indata.extend(wf.add)

                # Generate SHAKE constraint
                if wf.shake:
                    indata.append('fix shake%i all shake 1e-4 1000 0 m 1.0' % (i+1))
                    unfix.append('unfix shake%i' % (i+1))

                # Generate "fix deform"
                if wf.deform:
                    indata.append('')
                    indata.append('# deform')
                    self.make_input_deform(md, wf, i, indata, unfix)

                # Generate anisotropic pressure
                if wf.ensemble in ['npt', 'nph']:
                    if wf.p_aniso:
                        if wf.px_start is None and wf.py_start is None and wf.pz_start is None:
                            p_str = 'aniso %f %f %f ' % (wf.p_start, wf.p_stop, wf.p_dump)
                        else:
                            p_str = ''
                            if wf.px_start is not None and wf.px_stop is not None and wf.px_dump is not None:
                                p_str += 'x %f %f %f ' % (wf.px_start, wf.px_stop, wf.px_dump)
                            if wf.py_start is not None and wf.py_stop is not None and wf.py_dump is not None:
                                p_str += 'y %f %f %f ' % (wf.py_start, wf.py_stop, wf.py_dump)
                            if wf.pz_start is not None and wf.pz_stop is not None and wf.pz_dump is not None:
                                p_str += 'z %f %f %f ' % (wf.pz_start, wf.pz_stop, wf.pz_dump)
                        if wf.p_couple:
                            p_str += 'couple %s ' % (wf.p_couple)
                    else:
                        p_str = 'iso %f %f %f ' % (wf.p_start, wf.p_stop, wf.p_dump)
                    p_str += 'nreset %i ' % wf.p_nreset


                # Generate time integration, thermostat, and barostat
                if wf.ensemble == 'nve':
                    indata.append('')
                    indata.append('# nve')
                    if wf.nve_limit == 0:
                        indata.append('fix md%i all nve' % (i+1))
                    else:
                        indata.append('fix md%i all nve/limit %f' % (i+1, wf.nve_limit))

                elif wf.ensemble == 'nvt':
                    indata.append('')
                    indata.append('# nvt')
                    if wf.thermostat == 'Nose-Hoover':
                        indata.append('fix md%i all nvt temp %f %f %f' % (i+1, wf.t_start, wf.t_stop, wf.t_dump))

                    elif wf.thermostat == 'Langevin':
                        indata.append('fix thermostat%i all langevin %f %f %f %i' % (i+1, wf.t_start, wf.t_stop, wf.t_dump, seed))
                        indata.append('fix md%i all nve' % (i+1))
                        unfix.append('unfix thermostat%i' % (i+1))

                    elif wf.thermostat == 'Berendsen':
                        indata.append('fix thermostat%i all temp/berendsen %f %f %f' % (i+1, wf.t_start, wf.t_stop, wf.t_dump))
                        indata.append('fix md%i all nve' % (i+1))
                        unfix.append('unfix thermostat%i' % (i+1))

                    elif wf.thermostat == 'csvr':
                        indata.append('fix thermostat%i all temp/csvr %f %f %f %i' % (i+1, wf.t_start, wf.t_stop, wf.t_dump, seed))
                        indata.append('fix md%i all nve' % (i+1))
                        unfix.append('unfix thermostat%i' % (i+1))

                    elif wf.thermostat == 'csld':
                        indata.append('fix thermostat%i all temp/csld %f %f %f %i' % (i+1, wf.t_start, wf.t_stop, wf.t_dump, seed))
                        indata.append('fix md%i all nve' % (i+1))
                        unfix.append('unfix thermostat%i' % (i+1))

                    else:
                        utils.radon_print('%s thermostat does not support in RadonPy' % (wf.thermostat), level=3)
                        return None

                elif wf.ensemble == 'npt':
                    indata.append('')
                    indata.append('# npt')
                    if wf.thermostat == 'Nose-Hoover' and wf.barostat == 'Nose-Hoover':
                        indata.append('fix md%i all npt temp %f %f %f %s' %
                                    (i+1, wf.t_start, wf.t_stop, wf.t_dump, p_str))

                    elif wf.thermostat == 'Langevin' and wf.barostat == 'Nose-Hoover':
                        indata.append('fix thermostat%i all langevin %f %f %f %i' % (i+1, wf.t_start, wf.t_stop, wf.t_dump, seed))
                        indata.append('fix md%i all nph %s' % (i+1, p_str))
                        unfix.append('unfix thermostat%i' % (i+1))

                    elif wf.thermostat == 'Berendsen' and wf.barostat == 'Nose-Hoover':
                        indata.append('fix thermostat%i all temp/berendsen %f %f %f' % (i+1, wf.t_start, wf.t_stop, wf.t_dump))
                        indata.append('fix md%i all nph %s' % (i+1, p_str))
                        unfix.append('unfix thermostat%i' % (i+1))

                    elif wf.thermostat == 'csvr' and wf.barostat == 'Nose-Hoover':
                        indata.append('fix thermostat%i all temp/csvr %f %f %f %i' % (i+1, wf.t_start, wf.t_stop, wf.t_dump, seed))
                        indata.append('fix md%i all nph %s' % (i+1, p_str))
                        unfix.append('unfix thermostat%i' % (i+1))

                    elif wf.thermostat == 'csld' and wf.barostat == 'Nose-Hoover':
                        indata.append('fix thermostat%i all temp/csld %f %f %f %i' % (i+1, wf.t_start, wf.t_stop, wf.t_dump, seed))
                        indata.append('fix md%i all nph %s' % (i+1, p_str))
                        unfix.append('unfix thermostat%i' % (i+1))

                    elif wf.thermostat == 'Nose-Hoover' and wf.barostat == 'Berendsen':
                        indata.append('fix barostat%i all press/berendsen %s' % (i+1, p_str))
                        indata.append('fix md%i all nvt temp %f %f %f' % (i+1, wf.t_start, wf.t_stop, wf.t_dump))
                        unfix.append('unfix barostat%i' % (i+1))

                    elif wf.thermostat == 'Langevin' and wf.barostat == 'Berendsen':
                        indata.append('fix thermostat%i all langevin %f %f %f %i' % (i+1, wf.t_start, wf.t_stop, wf.t_dump, seed))
                        indata.append('fix barostat%i all press/berendsen %s' % (i+1, p_str))
                        indata.append('fix md%i all nve' % (i+1))
                        unfix.append('unfix thermostat%i' % (i+1))
                        unfix.append('unfix barostat%i' % (i+1))

                    elif wf.thermostat == 'Berendsen' and wf.barostat == 'Berendsen':
                        indata.append('fix thermostat%i all temp/berendsen %f %f %f' % (i+1, wf.t_start, wf.t_stop, wf.t_dump))
                        indata.append('fix barostat%i all press/berendsen %s' % (i+1, p_str))
                        indata.append('fix md%i all nve' % (i+1))
                        unfix.append('unfix thermostat%i' % (i+1))
                        unfix.append('unfix barostat%i' % (i+1))

                    elif wf.thermostat == 'csvr' and wf.barostat == 'Berendsen':
                        indata.append('fix thermostat%i all temp/csvr %f %f %f %i' % (i+1, wf.t_start, wf.t_stop, wf.t_dump, seed))
                        indata.append('fix barostat%i all press/berendsen %s' % (i+1, p_str))
                        indata.append('fix md%i all nve' % (i+1))
                        unfix.append('unfix thermostat%i' % (i+1))
                        unfix.append('unfix barostat%i' % (i+1))

                    elif wf.thermostat == 'csld' and wf.barostat == 'Berendsen':
                        indata.append('fix thermostat%i all temp/csld %f %f %f %i' % (i+1, wf.t_start, wf.t_stop, wf.t_dump, seed))
                        indata.append('fix barostat%i all press/berendsen %s' % (i+1, p_str))
                        indata.append('fix md%i all nve' % (i+1))
                        unfix.append('unfix thermostat%i' % (i+1))
                        unfix.append('unfix barostat%i' % (i+1))

                    else:
                        utils.radon_print('%s thermostat and/or %s barostat does not support in RadonPy' % (wf.thermostat, wf.barostat), level=3)
                        return None

                elif wf.ensemble == 'nph':
                    indata.append('')
                    indata.append('# nph')
                    if wf.barostat == 'Nose-Hoover':
                        indata.append('fix md%i all nph %s' % (i+1, p_str))

                    elif wf.barostat == 'Berendsen':
                        indata.append('fix barostat%i all press/berendsen %s' % (i+1, p_str))
                        indata.append('fix md%i all nve' % (i+1))
                        unfix.append('unfix barostat%i' % (i+1))

                    else:
                        utils.radon_print('%s barostat does not support in RadonPy' % (wf.barostat), level=3)
                        return None

                # Generate RATTLE constraints
                if wf.rattle:
                    indata.append('fix rattle%i all rattle 1e-4 1000 0 m 1.0' % (i+1))
                    unfix.append('unfix rattle%i' % (i+1))

                if wf.momentum:
                    indata.append('fix momentum%i all momentum 1000 linear 1 1 1 rescale' % (i+1))
                    unfix.append('unfix momentum%i' % (i+1))

                # Update thermo_style
                indata.append('')
                indata.append('# Update thermo_style')
                self.update_thermo_style(md, wf, i, indata, unfix)

                if wf.rerun:
                    indata.append('')
                    indata.append('rerun %s %s' % (wf.rerun_dump, wf.rerun_keyword))
                else:
                    indata.append('')
                    indata.append('run %i' % (wf.step))
                    
                indata.append('unfix md%i' % (i+1))    
                indata.extend(unfix)
                if len(wf.add_f) > 0:
                    indata.extend(wf.add_f)

        if len(md.wf) == 0:
            indata.append('fix md0 all nve')
            indata.append('run 0')

        indata.append('')
        if md.outstr:
            indata.append('write_dump all custom %s id x y z xu yu zu vx vy vz fx fy fz modify sort id' % (md.outstr))
        if md.write_data:
            indata.append('write_data %s' % (md.write_data))

        if len(md.add_f) > 0:
            indata.append('')
            indata.extend(md.add_f)

        indata.append('quit')

        file_name = file_name if file_name is not None else self.input_file
        dir_name = dir_name if dir_name is not None else self.work_dir
        in_file = os.path.join(dir_name, file_name)
        with open(in_file, 'w') as fh:
            fh.write('\n'.join(indata)+'\n')
            fh.flush()
            if hasattr(os, 'fdatasync'):
                os.fdatasync(fh.fileno())
            else:
                os.fsync(fh.fileno())

        return in_file


    def update_thermo_style(self, md, wf, i, indata, unfix):
        indata.append('thermo_style custom %s' % ' '.join([*md.thermo_style, *wf.thermo_style]))
        indata.append('thermo_modify flush yes')
        if wf.thermo_freq is not None:
            indata.append('thermo %i' % (wf.thermo_freq))
        else:
            indata.append('thermo %i' % (md.thermo_freq))


    def make_input_drude(self, md, indata):
        # Under development

        return indata


    def make_input_deform(self, md, wf, i, indata, unfix):
        if wf.deform == 'scale':
            if wf.ensemble in ['nvt', 'nve']:
                if wf.deform_axis == 'x':
                    indata.append('fix DEF%i all deform 1 x scale %f y volume z volume' % (i+1, wf.deform_scale))
                elif wf.deform_axis == 'y':
                    indata.append('fix DEF%i all deform 1 y scale %f x volume z volume' % (i+1, wf.deform_scale))
                elif wf.deform_axis == 'z':
                    indata.append('fix DEF%i all deform 1 z scale %f x volume y volume' % (i+1, wf.deform_scale))
            elif wf.ensemble in ['npt', 'nph']:
                indata.append('fix DEF%i all deform 1 %s scale %f' % (i+1, wf.deform_axis, wf.deform_scale))
                if not wf.p_aniso:
                    wf.p_aniso = True
                    if wf.deform_axis == 'x':
                        wf.py_start, wf.py_stop, wf.py_dump = wf.p_start, wf.p_stop, wf.p_dump
                        wf.pz_start, wf.pz_stop, wf.pz_dump = wf.p_start, wf.p_stop, wf.p_dump
                        wf.p_couple = 'yz'
                    elif wf.deform_axis == 'y':
                        wf.px_start, wf.px_stop, wf.px_dump = wf.p_start, wf.p_stop, wf.p_dump
                        wf.pz_start, wf.pz_stop, wf.pz_dump = wf.p_start, wf.p_stop, wf.p_dump
                        wf.p_couple = 'xz'
                    elif wf.deform_axis == 'z':
                        wf.px_start, wf.px_stop, wf.px_dump = wf.p_start, wf.p_stop, wf.p_dump
                        wf.py_start, wf.py_stop, wf.py_dump = wf.p_start, wf.p_stop, wf.p_dump
                        wf.p_couple = 'xy'
                    elif wf.deform_axis == 'xy':
                        wf.pz_start, wf.pz_stop, wf.pz_dump = wf.p_start, wf.p_stop, wf.p_dump
                    elif wf.deform_axis == 'xz':
                        wf.py_start, wf.py_stop, wf.py_dump = wf.p_start, wf.p_stop, wf.p_dump
                    elif wf.deform_axis == 'yz':
                        wf.px_start, wf.px_stop, wf.px_dump = wf.p_start, wf.p_stop, wf.p_dump
            unfix.append('unfix DEF%i' % (i+1))

        elif wf.deform == 'final':
            if wf.ensemble in ['nvt', 'nve']:
                if wf.deform_axis == 'x':
                    indata.append('fix DEF%i all deform 1 x final %f %f y volume z volume' % (i+1, wf.deform_fin_lo, wf.deform_fin_hi))
                elif wf.deform_axis == 'y':
                    indata.append('fix DEF%i all deform 1 y final %f %f x volume z volume' % (i+1, wf.deform_fin_lo, wf.deform_fin_hi))
                elif wf.deform_axis == 'z':
                    indata.append('fix DEF%i all deform 1 z final %f %f x volume y volume' % (i+1, wf.deform_fin_lo, wf.deform_fin_hi))
                elif wf.deform_axis == 'xy':
                    indata.append('fix DEF%i all deform 1 x final %f %f y final %f %f z volume'
                            % (i+1, wf.deform_fin_lo, wf.deform_fin_hi, wf.deform_fin_lo, wf.deform_fin_hi))
                elif wf.deform_axis == 'xz':
                    indata.append('fix DEF%i all deform 1 x final %f %f y volume z final %f %f'
                            % (i+1, wf.deform_fin_lo, wf.deform_fin_hi, wf.deform_fin_lo, wf.deform_fin_hi))
                elif wf.deform_axis == 'yz':
                    indata.append('fix DEF%i all deform 1 x volume y final %f %f z final %f %f'
                            % (i+1, wf.deform_fin_lo, wf.deform_fin_hi, wf.deform_fin_lo, wf.deform_fin_hi))
                elif wf.deform_axis == 'xyz':
                    indata.append('fix DEF%i all deform 1 x final %f %f y final %f %f z final %f %f'
                            % (i+1, wf.deform_fin_lo, wf.deform_fin_hi, wf.deform_fin_lo, wf.deform_fin_hi, wf.deform_fin_lo, wf.deform_fin_hi))
            elif wf.ensemble in ['npt', 'nph']:
                if wf.deform_axis in ['x', 'y', 'z']:
                    indata.append('fix DEF%i all deform 1 %s final %f %f' % (i+1, wf.deform_axis, wf.deform_fin_lo, wf.deform_fin_hi))
                elif wf.deform_axis == 'xy':
                    indata.append('fix DEF%i all deform 1 x final %f %f y final %f %f'
                            % (i+1, wf.deform_fin_lo, wf.deform_fin_hi, wf.deform_fin_lo, wf.deform_fin_hi))
                elif wf.deform_axis == 'xz':
                    indata.append('fix DEF%i all deform 1 x final %f %f z final %f %f'
                            % (i+1, wf.deform_fin_lo, wf.deform_fin_hi, wf.deform_fin_lo, wf.deform_fin_hi))
                elif wf.deform_axis == 'yz':
                    indata.append('fix DEF%i all deform 1 y final %f %f z final %f %f'
                            % (i+1, wf.deform_fin_lo, wf.deform_fin_hi, wf.deform_fin_lo, wf.deform_fin_hi))
                if not wf.p_aniso:
                    wf.p_aniso = True
                    if wf.deform_axis == 'x':
                        wf.py_start, wf.py_stop, wf.py_dump = wf.p_start, wf.p_stop, wf.p_dump
                        wf.pz_start, wf.pz_stop, wf.pz_dump = wf.p_start, wf.p_stop, wf.p_dump
                        wf.p_couple = 'yz'
                    elif wf.deform_axis == 'y':
                        wf.px_start, wf.px_stop, wf.px_dump = wf.p_start, wf.p_stop, wf.p_dump
                        wf.pz_start, wf.pz_stop, wf.pz_dump = wf.p_start, wf.p_stop, wf.p_dump
                        wf.p_couple = 'xz'
                    elif wf.deform_axis == 'z':
                        wf.px_start, wf.px_stop, wf.px_dump = wf.p_start, wf.p_stop, wf.p_dump
                        wf.py_start, wf.py_stop, wf.py_dump = wf.p_start, wf.p_stop, wf.p_dump
                        wf.p_couple = 'xy'
                    elif wf.deform_axis == 'xy':
                        wf.pz_start, wf.pz_stop, wf.pz_dump = wf.p_start, wf.p_stop, wf.p_dump
                    elif wf.deform_axis == 'xz':
                        wf.py_start, wf.py_stop, wf.py_dump = wf.p_start, wf.p_stop, wf.p_dump
                    elif wf.deform_axis == 'yz':
                        wf.px_start, wf.px_stop, wf.px_dump = wf.p_start, wf.p_stop, wf.p_dump
            
            if wf.deform_remap:
                indata[-1] += ' remap %s' % wf.deform_remap

            unfix.append('unfix DEF%i' % (i+1))

        return indata, unfix


    def make_input_efield(self, md, wf, i, indata, unfix, nounfix=False):
        evalue = wf.efield_value if wf.efield_freq == 0.0 else 'v_efac'
        if wf.efield_axis == 'x':
            ex = evalue
            ey = 0.0
            ez = 0.0
        elif wf.efield_axis == 'y':
            ex = 0.0
            ey = evalue
            ez = 0.0
        elif wf.efield_axis == 'z':
            ex = 0.0
            ey = 0.0
            ez = evalue
        elif wf.efield_axis == 'xy':
            ex = evalue
            ey = evalue
            ez = 0.0
        elif wf.efield_axis == 'xz':
            ex = evalue
            ey = 0.0
            ez = evalue
        elif wf.efield_axis == 'yz':
            ex = 0.0
            ey = evalue
            ez = evalue
        elif wf.efield_axis == 'xyz':
            ex = evalue
            ey = evalue
            ez = evalue

        if wf.efield_x:
            ex = wf.efield_x
        if wf.efield_y:
            ey = wf.efield_y
        if wf.efield_z:
            ez = wf.efield_z

        if wf.efield_freq != 0.0:
            indata.append('variable efac equal %f*sin(2*PI*%f*time*1e-15)' % (wf.efield_value, wf.efield_freq))
        indata.append('fix EF%i all efield %s %s %s' % (i+1, ex, ey, ez))
        if not nounfix:
            unfix.append('unfix EF%i' % (i+1))

        wf.thermo_style += ['v_efac']
        # indata.append('thermo_style %s' % (md.thermo_style))
        # indata.append('thermo_modify flush yes')
        # indata.append('thermo %i' % (md.thermo_freq))

        return indata, unfix


    def make_input_dipole(self, md, wf, i, indata, unfix):
        indata.append('compute dipole%i all dipole/chunk cmol%i' % (i+1, i+1))
        indata.append('variable mux equal sum(c_dipole%i[1])' % (i+1))
        indata.append('variable muy equal sum(c_dipole%i[2])' % (i+1))
        indata.append('variable muz equal sum(c_dipole%i[3])' % (i+1))
        indata.append('variable mu equal sqrt(v_mux^2+v_muy^2+v_muz^2)')

        unfix.append('uncompute dipole%i' % (i+1))

        wf.thermo_style += ['v_mux', 'v_muy', 'v_muz', 'v_mu']
        # indata.append('thermo_style %s' % (md.thermo_style))
        # indata.append('thermo_modify flush yes')
        # indata.append('thermo %i' % (md.thermo_freq))

        return indata, unfix


    def make_input_rg(self, md, wf, i, indata, unfix):
        indata.append('compute gyr%i all gyration/chunk cmol%i' % (i+1, i+1))
        indata.append('fix rg%i all ave/time 1 %i %i c_gyr%i file %s mode vector' % (i+1, wf.rg_ave_length, wf.rg_ave_length, i+1, wf.rg_file))

        unfix.append('unfix rg%i' % (i+1))
        unfix.append('uncompute gyr%i' % (i+1))

        return indata, unfix


    def make_input_msd(self, md, wf, i, indata, unfix):
        indata.append('compute msd%i all msd com yes average no' % (i+1))
        indata.append('fix msd%i all ave/time 1 %i %i c_msd%i[4] mode scalar' % (i+1, md.thermo_freq, md.thermo_freq, i+1))
        indata.append('variable msd equal f_msd%i' % (i+1))

        #indata.append('fix msd%i all vector %i c_msd%i[4]' % (i+1, wf.msd_freq, i+1))
        #indata.append('variable msd equal ave(f_msd%i)' % (i+1))
        #indata.append('variable diffc equal slope(f_msd%i)/6/(%i*dt)' % (i+1, wf.msd_freq))

        unfix.append('unfix msd%i' % (i+1))
        unfix.append('uncompute msd%i' % (i+1))

        wf.thermo_style += ['v_msd']
        # indata.append('thermo_style %s' % (md.thermo_style))
        # indata.append('thermo_modify flush yes')
        # indata.append('thermo %i' % (md.thermo_freq))

        return indata, unfix


    def make_input_variable(self, md, wf, i, indata, unfix):
        for n, s, a in zip(wf.variable_name, wf.variable_style, wf.variable_args):
            indata.append('variable %s %s %s' % (n, s, ' '.join(a)))

        return indata, unfix


    def make_input_timeave(self, md, wf, i, indata, unfix, nounfix=False):
        if wf.timeave_name is None:
            wf.timeave_name = 'ave%i' % (i+1)

        if not nounfix and not wf.timeave_nounfix:
            unfix.append('unfix %s' % wf.timeave_name)

        var_str = ' '.join(wf.timeave_var)
        indata.append('fix %s all ave/time %i %i %i %s' % (wf.timeave_name, wf.timeave_nevery, wf.timeave_nfreq, wf.timeave_nstep, var_str))

        wf.thermo_style += ['f_%s[%i]' % (wf.timeave_name, (j+1)) for j in range(len(wf.timeave_var))]
        # fix_name = ' '.join(['f_%s[%i]' % (wf.timeave_name, (j+1)) for j in range(len(wf.timeave_var))])
        # md.thermo_style += ' %s' % fix_name
        # indata.append('thermo_style %s' % (md.thermo_style))
        # indata.append('thermo_modify flush yes')
        # indata.append('thermo %i' % (md.thermo_freq))

        return indata, unfix


    def make_input_atomave(self, md, wf, i, indata, unfix, nounfix=False):
        # Under development
        return indata, unfix


    def read_traj_last(self, filename, b_size=64):
        """
        LAMMPS.read_traj_last

        tail command like reading routine of trajectory data

        Args:
            filename: Path of trajectory file

        Optional args:
            b_size: buffer size

        Return:
            Unwrapped atomic coordinates
            Wrapped atomic coordinates
            Cell lengths
            Atomic velocities
            Force on atoms
        """

        line = b''

        with open(filename, 'rb') as fh:
            fh.seek(0, 2)
            n = int(fh.tell() / b_size)

            for i in range(n):
                fh.seek(-b_size, 1)
                data = fh.read(b_size)
                line = data + line
                text = line.decode()
                fh.seek(-b_size, 1)

                if 'ITEM: TIMESTEP' in text: break

        flag_cell = False
        flag_atoms = False
        cell = []
        atoms = []
        lines = text.split('\n')
        for line in lines:
            if line == '':
                continue
            elif 'ITEM: BOX BOUNDS' in line:
                pbc = line.split(' ')[3:]
                flag_cell = True
                flag_atoms = False
            elif 'ITEM: ATOMS' in line:
                atoms_column = line.split(' ')[2:]
                flag_cell = False
                flag_atoms = True
            elif flag_cell:
                cell.append([float(f) for f in line.split(' ')])
            elif flag_atoms:
                atoms.append(line.split(' '))

        if 'id' in atoms_column:
            id_idx = atoms_column.index('id')
        else:
            return False
        x_idx = atoms_column.index('x') if 'x' in atoms_column else None
        y_idx = atoms_column.index('y') if 'y' in atoms_column else None
        z_idx = atoms_column.index('z') if 'z' in atoms_column else None
        xu_idx = atoms_column.index('xu') if 'xu' in atoms_column else None
        yu_idx = atoms_column.index('yu') if 'yu' in atoms_column else None
        zu_idx = atoms_column.index('zu') if 'zu' in atoms_column else None
        vx_idx = atoms_column.index('vx') if 'vx' in atoms_column else None
        vy_idx = atoms_column.index('vy') if 'vy' in atoms_column else None
        vz_idx = atoms_column.index('vz') if 'vz' in atoms_column else None
        fx_idx = atoms_column.index('fx') if 'fx' in atoms_column else None
        fy_idx = atoms_column.index('fy') if 'fy' in atoms_column else None
        fz_idx = atoms_column.index('fz') if 'fz' in atoms_column else None

        num = len(atoms)
        uwstr = np.array([[0] * 3 for i in range(num)], dtype=float)
        wstr = np.array([[0] * 3 for i in range(num)], dtype=float)
        v = np.array([[0] * 3 for i in range(num)], dtype=float)
        f = np.array([[0] * 3 for i in range(num)], dtype=float)

        for atom in atoms:
            atom_id = int(atom[id_idx])-1
            wstr[atom_id, 0] = float(atom[x_idx]) if x_idx is not None else 0
            wstr[atom_id, 1] = float(atom[y_idx]) if y_idx is not None else 0
            wstr[atom_id, 2] = float(atom[z_idx]) if z_idx is not None else 0
            uwstr[atom_id, 0] = float(atom[xu_idx]) if xu_idx is not None else 0
            uwstr[atom_id, 1] = float(atom[yu_idx]) if yu_idx is not None else 0
            uwstr[atom_id, 2] = float(atom[zu_idx]) if zu_idx is not None else 0
            v[atom_id, 0] = float(atom[vx_idx]) if vx_idx is not None else 0
            v[atom_id, 1] = float(atom[vy_idx]) if vy_idx is not None else 0
            v[atom_id, 2] = float(atom[vz_idx]) if vz_idx is not None else 0
            f[atom_id, 0] = float(atom[fx_idx]) if fx_idx is not None else 0
            f[atom_id, 1] = float(atom[fy_idx]) if fy_idx is not None else 0
            f[atom_id, 2] = float(atom[fz_idx]) if fz_idx is not None else 0

        return uwstr, wstr, np.array(cell), v, f


    def read_traj_simple(self, filename):
        """
        LAMMPS.read_traj_simple

        Simple reading routine of trajectory data

        Args:
            filename: Path of trajectory file

        Return:
            Unwrapped atomic coordinates
            Wrapped atomic coordinates
            Cell lengths
            Atomic velocities
            Force on atoms
        """

        with open(filename, "r") as fh:
            lines = [s.replace('\n', '').replace('\r', '') for s in fh.readlines()]

        flag_cell = False
        flag_atoms = False
        cell = []
        atoms = []
        for line in lines:
            if line == '':
                continue
            elif 'ITEM: TIMESTEP' in line:
                flag_cell = False
                flag_atoms = False
            elif 'ITEM: BOX BOUNDS' in line:
                pbc = line.split(' ')[3:]
                flag_cell = True
                flag_atoms = False
            elif 'ITEM: ATOMS' in line:
                atoms_column = line.split(' ')[2:]
                flag_cell = False
                flag_atoms = True
            elif flag_cell:
                cell.append([float(f) for f in line.split(' ')])
            elif flag_atoms:
                atoms.append(line.split(' '))

        if 'id' in atoms_column:
            id_idx = atoms_column.index('id')
        else:
            return False
        x_idx = atoms_column.index('x') if 'x' in atoms_column else None
        y_idx = atoms_column.index('y') if 'y' in atoms_column else None
        z_idx = atoms_column.index('z') if 'z' in atoms_column else None
        xu_idx = atoms_column.index('xu') if 'xu' in atoms_column else None
        yu_idx = atoms_column.index('yu') if 'yu' in atoms_column else None
        zu_idx = atoms_column.index('zu') if 'zu' in atoms_column else None
        vx_idx = atoms_column.index('vx') if 'vx' in atoms_column else None
        vy_idx = atoms_column.index('vy') if 'vy' in atoms_column else None
        vz_idx = atoms_column.index('vz') if 'vz' in atoms_column else None
        fx_idx = atoms_column.index('fx') if 'fx' in atoms_column else None
        fy_idx = atoms_column.index('fy') if 'fy' in atoms_column else None
        fz_idx = atoms_column.index('fz') if 'fz' in atoms_column else None

        num = len(atoms)
        uwstr = np.array([[0] * 3 for i in range(num)], dtype=float)
        wstr = np.array([[0] * 3 for i in range(num)], dtype=float)
        v = np.array([[0] * 3 for i in range(num)], dtype=float)
        f = np.array([[0] * 3 for i in range(num)], dtype=float)

        for atom in atoms:
            atom_id = int(atom[id_idx])-1
            wstr[atom_id, 0] = float(atom[x_idx]) if x_idx is not None else 0
            wstr[atom_id, 1] = float(atom[y_idx]) if y_idx is not None else 0
            wstr[atom_id, 2] = float(atom[z_idx]) if z_idx is not None else 0
            uwstr[atom_id, 0] = float(atom[xu_idx]) if xu_idx is not None else 0
            uwstr[atom_id, 1] = float(atom[yu_idx]) if yu_idx is not None else 0
            uwstr[atom_id, 2] = float(atom[zu_idx]) if zu_idx is not None else 0
            v[atom_id, 0] = float(atom[vx_idx]) if vx_idx is not None else 0
            v[atom_id, 1] = float(atom[vy_idx]) if vy_idx is not None else 0
            v[atom_id, 2] = float(atom[vz_idx]) if vz_idx is not None else 0
            f[atom_id, 0] = float(atom[fx_idx]) if fx_idx is not None else 0
            f[atom_id, 1] = float(atom[fy_idx]) if fy_idx is not None else 0
            f[atom_id, 2] = float(atom[fz_idx]) if fz_idx is not None else 0

        return uwstr, wstr, np.array(cell), v, f




###########################################
# Analysis functions
###########################################

class Analyze():
    def __init__(self, log_file='radon_md.log', ignore_log=[], **kwargs):
        self.dfs = self.read_log(log_file, ignore_log=ignore_log)
        self.log_file = log_file
        self.in_file = kwargs.get('in_file', 'radon_md.dump')
        self.dat_file = kwargs.get('dat_file', 'radon_md_lmp.data')
        self.traj_file = kwargs.get('traj_file', 'radon_md.dump')
        self.rg_file = kwargs.get('rg_file', 'rg.profile')
        self.pdb_file = kwargs.get('pdb_file', 'topology.pdb')

        self.traj = None
        self.charges = np.array([])

        self.totene_data = {}
        self.kinene_data = {}
        self.ebond_data = {}
        self.eangle_data = {}
        self.edihed_data = {}
        self.evdw_data = {}
        self.ecoul_data = {}
        self.elong_data = {}
        self.dens_data = {}
        self.temp_data = {}
        self.rg_data = {}
        self.msd_data = {}
        self.diffc_data = {}
        self.Cp_data = {}
        self.Cv_data = {}
        self.compress_T_data = {}
        self.compress_S_data = {}
        self.bulk_mod_T_data = {}
        self.bulk_mod_S_data = {}
        self.volume_exp_data = {}
        self.linear_exp_data = {}
        self.diele = {}
        self.diele_data = {}
        self.nop = {}
        self.nop_data = {}
        self.prop_df = pd.DataFrame([])
        self.conv_df = pd.DataFrame([])

        self.totene_sma_sd_crit = kwargs.get('totene_sma_sd_crit', 0.0005)
        self.kinene_sma_sd_crit = kwargs.get('kinene_sma_sd_crit', 0.0005)
        self.ebond_sma_sd_crit = kwargs.get('ebond_sma_sd_crit', 0.001)
        self.eangle_sma_sd_crit = kwargs.get('eangle_sma_sd_crit', 0.001)
        self.edihed_sma_sd_crit = kwargs.get('edihed_sma_sd_crit', 0.002)
        self.evdw_sma_sd_crit = kwargs.get('evdw_sma_sd_crit', 30.0)
        self.ecoul_sma_sd_crit = kwargs.get('ecoul_sma_sd_crit', None)
        self.elong_sma_sd_crit = kwargs.get('elong_sma_sd_crit', 0.001)
        self.dens_sma_sd_crit = kwargs.get('dens_sma_sd_crit', 0.001)
        self.rg_sd_crit = kwargs.get('rg_sd_crit', 0.01)
        self.diffc_sma_sd_crit = kwargs.get('diffc_sma_sd_crit', None)
        self.Cp_sma_sd_crit = kwargs.get('Cp_sma_sd_crit', None)
        self.compress_sma_sd_crit = kwargs.get('compress_sma_sd_crit', None)
        self.volexp_sma_sd_crit = kwargs.get('volexp_sma_sd_crit', None)


    def read_log(self, log_file, ignore_log=[]):
        """
        lammps.Analyze.read_log

        Read thermodynamic data from a log file

        Args:
            log_file: Path of log file

        Return:
            Array of Pandas Data Frame
        """

        with open(log_file, "r") as fh:
            log_data = [s.replace('\n', '').replace('\r', '') for s in fh.readlines()]

        try:
            dfs = self.parse_thermo(log_data, ignore_log=ignore_log)
            self.dfs = dfs
        except Exception as e:
            self.dfs = dfs = None
            utils.radon_print('Can not read log file. %s; %s' % (log_file, e), level=2)

        return dfs


    @classmethod
    def parse_thermo(cls, log_data, ignore_log=[]):
        """
        lammps.Analyze.parse_thermo

        Parser of thermodynamic data in a log file

        Args:
            log_data: Strings of log data

        Return:
            Array of Pandas Data Frame
        """

        d_flag = 0
        columns = []
        data = []
        dfs = []

        for line in log_data:
            if line.find('Per MPI rank memory allocation') == 0 or line.find('Memory usage per processor') == 0:
                d_flag = 1
            elif d_flag == 1:
                columns = line.split()
                d_flag = 2
            elif d_flag >= 1 and (line.find('Loop time of') == 0 or line.find('ERROR') == 0 or line == ''):
                df = pd.DataFrame(data, columns=columns)
                if 'Step' in columns:
                    df = df.set_index('Step')
                dfs.append(df)
                d_flag = 0
                columns = []
                data = []
            elif d_flag == 2:
                try:
                    ignore_flag = False
                    for ignore_line in ignore_log:
                        if ignore_line in line:
                            ignore_flag = True
                            break
                    if ignore_flag:
                        continue
                    data.append([float(f) for f in line.split()])
                except ValueError as e:
                    utils.radon_print('Can not parse thermodynamic data. %s; Skip to parse the data of %i step.' % (e, len(dfs)), level=1)
                    d_flag = 0
                    columns = []
                    data = []

        if d_flag >= 1:
            df = pd.DataFrame(data, columns=columns)
            if 'Step' in columns:
                df = df.set_index('Step')
            dfs.append(df)

        return dfs


    def analyze_thermo(self, target, init=2000, last=None, width=2000,
                    ylabel=None, conv_a=1.0, conv_b=0.0, printout=False, save=None):
        """
        LAMMPS.analyze_thermo

        Analyze thermodynamic propeties in a log file

        Args:
            target: Target property

        Optional args:
            init: Initial step (int)
            last: Last step (int)
            width: Width of steps to use for the average, variance, and covariance calculations (int)
            ylabel: Label of y-axis in the output plot (str)
            timestep: Conversion factor of timestep -> ps (float)
            conv_a, conv_b: Conversion factor of property: prop_c = conv_a * prop + conv_b (float)
            printout: Printout analyzed data for STDOUT (boolean)
            save: Dir path to save analyzed data (str)

        Return:
            Analyzed results
        """

        thermo_df = self.dfs[-1]
        data = {}
        ps = 1e-3
        if not last: last = 0
        if len(thermo_df) < last: last = 0
        
        if init >= len(thermo_df):
            utils.radon_print('init=%i is out of range. Require init < %i' % (init, len(thermo_df)), level=3)
            return None

        data_conv = thermo_df[target] * conv_a + conv_b
        data_sma = data_conv.rolling(width).mean()
        data_sd = data_conv.rolling(width).std()
        data_se = data_sd / np.sqrt(width)

        data['init'] = thermo_df['Time'].values[init] * ps
        data['last'] = thermo_df['Time'].values[last-1] * ps
        data['width'] = width
        data['mean'] = data_sma.values[int(last-1)]
        data['sd'] = data_sd.values[int(last-1)]
        data['se'] = data_se.values[int(last-1)]
        data['sma_sd'] = data_sma.values[last-width-1:last].std() if last else data_sma.values[-width-1:].std()
        data['sma_se'] = data['sma_sd'] / np.sqrt(width)

        if printout or save:
            if not last: last = None
            fig, ax = pp.subplots(figsize=(6, 6))
            ax.ticklabel_format(style="sci",  axis="y", scilimits=(0,0))
            ax.plot(thermo_df['Time'].values[init:last]*ps, data_conv.values[init:last], linewidth=0.1)
            ax.errorbar(thermo_df['Time'].values[init:last]*ps, data_sma.values[init:last], yerr=data_se.values[init:last]*2, linewidth=2.0)
            ax.set_xlabel('Time [ps]', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            output = 'Accumulation of %f - %f ps\n' % (data['init'], data['last'])
            output += '%s = %e     SD = %e    SE = %e\n' % (ylabel, data['mean'], data['sd'], data['se'])
            output += 'SMA_SD = %e     SMA_SE = %e\n' % (data['sma_sd'], data['sma_se'])

            if printout:
                pp.show()
                print(output)

            if save:
                if not os.path.exists(save):
                    os.makedirs(save)
                fig.savefig(os.path.join(save, target+'.png'))
                with open(os.path.join(save, target+'.txt'), mode='w') as f:
                    f.write(output)
            
            pp.close(fig)

        return data


    def analyze_thermo_fluctuation(self, func, target=-1, temp=None, press=1.0, mass=None, f_width=2000,
            init=2000, last=None, width=2000, name='img', ylabel=None, conv_a=1.0, conv_b=0.0,
            printout=False, save=None):
        """
        lammps.Analyze.analyze_thermo_fluctuation

        Analyze thermodynamic fluctiation propeties in a log file

        Args:
            func: Function object of thermodynamic fluctiation propeties
                lammps.Analyze.heat_capacity_Cp
                lammps.Analyze.heat_capacity_Cv
                lammps.Analyze.heat_capacity_Cv_NVT
                lammps.Analyze.isothermal_compressibility
                lammps.Analyze.isentropic_compressibility
                lammps.Analyze.bulk_modulus
                lammps.Analyze.isentropic_bulk_modulus
                lammps.Analyze.speed_of_sound
                lammps.Analyze.volume_expansion
                lammps.Analyze.linear_expansion

        Optional args:
            temp: Temperature (If None, temperature in thermodynamic data is used) (float, K)
            press: Pressure (float, atm)
            mass: Mass (If None, mass in thermodynamic data (density*volume) is used) (float, kg)
            init: Initial step (int)
            last: Last step (int)
            width: Width of steps to use for the average, variance, and covariance calculations (int)
            name: Output file name (str)
            ylabel: Label of y-axis in the output plot (str)
            timestep: Conversion factor of timestep -> ps (float)
            conv_a, conv_b: Conversion factor of property: prop_c = conv_a * prop + conv_b (float)
            printout: Printout analyzed data for STDOUT (boolean)
            save: Dir path to save analyzed data (str)

        Return:
            Analyzed results
        """

        thermo_df = self.dfs[target]
        law_data = []
        data = {}
        ps = 1e-3

        if last and len(thermo_df) < last: last = None

        n = len(thermo_df) - init
        if n <= 0:
            utils.radon_print('init=%i is out of range. Require init > %i' % (init, len(thermo_df)), level=3)
            return None

        if init < f_width:
            utils.radon_print('init=%i, f_width=%i is out of range. Require init >= f_width' % (init, f_width), level=3)
            return None

        for i in range(n):
            f_last = i + init
            prop = func(thermo_df, temp=temp, press=press, mass=mass, init=init-f_width, last=f_last)
            law_data.append([thermo_df['Time'].values[f_last], prop])

        if not last: last = 0

        prop_df = pd.DataFrame(law_data, columns=['Time', 'Prop'])
        data_conv = prop_df['Prop'] * conv_a + conv_b
        data_sma = data_conv.rolling(width).mean()
        data_sd = data_conv.rolling(width).std()
        data_se = data_sd / np.sqrt(width)

        data['init'] = thermo_df['Time'].values[init] * ps
        data['last'] = thermo_df['Time'].values[last-1] * ps
        data['width'] = width
        data['mean'] = data_sma.values[int(last-1)]
        data['sd'] = data_sd.values[int(last-1)]
        data['se'] = data_se.values[int(last-1)]
        data['sma_sd'] = data_sma.values[last-width-1:last].std() if last else data_sma.values[-width-1:].std()
        data['sma_se'] = data['sma_sd'] / np.sqrt(width)

        if printout or save:
            if not last: last = None
            fig, ax = pp.subplots(figsize=(6, 6))
            ax.ticklabel_format(style="sci",  axis="y", scilimits=(0,0))
            ax.plot(prop_df['Time'].values[:last]*ps, data_conv.values[:last], linewidth=1.0)
            ax.errorbar(prop_df['Time'].values[:last]*ps, data_sma.values, yerr=data_se.values*2, linewidth=2.0)
            ax.set_xlabel('Time [ps]', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            output = 'Accumulation of %f - %f ps\n' % (data['init'], data['last'])
            output += '%s = %e     SD = %e    SE = %e\n' % (ylabel, data['mean'], data['sd'], data['se'])
            output += 'SMA_SD = %e     SMA_SE = %e\n' % (data['sma_sd'], data['sma_se'])

            if printout:
                pp.show()
                print(output)

            if save:
                if not os.path.exists(save):
                    os.makedirs(save)
                fig.savefig(os.path.join(save, name+'.png'))
                with open(os.path.join(save, name+'.txt'), mode='w') as f:
                    f.write(output)
            
            pp.close(fig)

        return data


    @classmethod
    def heat_capacity_Cp(cls, thermo_df, temp=None, press=1.0, mass=None, init=0, last=None):
        """
        LAMMPS.heat_capacity_Cp

        Calculate isobaric specific heat capacity from thermodynamic data in a log file
        Cp = Var(H)/(m*kB*T**2)

        Args:
            thermo_df: Pandas Data Frame of thermodynamic data

        Optional args:
            temp: Temperature (If None, temperature in thermodynamic data is used) (float, K)
            press: Pressure (float, atm)
            mass: Mass (If None, mass in thermodynamic data (density*volume) is used) (float, kg)
            init: Initial step (int)
            last: Last step (int)

        Return:
            isobaric heat capacity (float, J/(kg K))
        """

        if 'Volume' in thermo_df.columns:
            V = thermo_df['Volume'].to_numpy() * 1e-30 # Angstrom**3 -> m**3
        else:
            V = thermo_df['Lx'].to_numpy() * thermo_df['Ly'].to_numpy() * thermo_df['Lz'].to_numpy() * 1e-30 # Angstrom**3 -> m**3

        M = V * (thermo_df['Density'].to_numpy() * 1e+3) # m**3 * (g/cm**3) -> m**3 * (kg/m**3) = kg

        T = thermo_df['Temp'].to_numpy() # K

        U = thermo_df['TotEng'].to_numpy() * const.cal2j * 1000 / const.NA # kcal/mol -> J
        P = press * const.atm2pa # Pa = J / m**3
        H = U + P * V

        mM = M[init:last].mean() if mass is None else mass
        mT = T[init:last].mean() if temp is None else temp
        H_var = np.var(H[init:last])

        Cp = H_var / (mM * const.kB * mT**2) # J**2 / (kg * J/K * K**2) -> J/(kg K)

        return Cp


    @classmethod
    def heat_capacity_Cv(cls, thermo_df, temp=None, press=1.0, mass=None, init=0, last=None):
        """
        LAMMPS.heat_capacity_Cv

        Calculate isochoric specific heat capacity from thermodynamic data in a log file
        Cv = Cp - V*T*alpha_P**2 / (beta_T * m)

        Args:
            thermo_df: Pandas Data Frame of thermodynamic data

        Optional args:
            temp: Temperature (If None, temperature in thermodynamic data is used) (float, K)
            press: Pressure (float, atm)
            mass: Mass (If None, mass in thermodynamic data (density*volume) is used) (float, kg)
            init: Initial step (int)
            last: Last step (int)

        Return:
            isochoric heat capacity (float, J/(kg K))
        """

        if 'Volume' in thermo_df.columns:
            V = thermo_df['Volume'].to_numpy() * 1e-30 # Angstrom**3 -> m**3
        else:
            V = thermo_df['Lx'].to_numpy() * thermo_df['Ly'].to_numpy() * thermo_df['Lz'].to_numpy() * 1e-30 # Angstrom**3 -> m**3

        M = V * (thermo_df['Density'].to_numpy() * 1e+3) # m**3 * (g/cm**3) -> m**3 * (kg/m**3) = kg

        T = thermo_df['Temp'].to_numpy() # K

        mV = V[init:last].mean()
        mM = M[init:last].mean() if mass is None else mass
        mT = T[init:last].mean() if temp is None else temp

        Cp = cls.heat_capacity_Cp(thermo_df, temp=temp, press=press, mass=mass, init=init, last=last)
        alpha_P = cls.volume_expansion(thermo_df, temp=temp, press=press, init=init, last=last)
        beta_T = cls.isothermal_compressibility(thermo_df, temp=temp, init=init, last=last)

        Cv = Cp - mV * mT * alpha_P**2 / beta_T / mM # m**3 * K * K**-2 / (m s**2 / kg) / kg = m**2 * K**-1 * s**-2 * kg * kg**-1 = J/(kg K)

        return Cv


    @classmethod
    def heat_capacity_Cv_NVT(cls, thermo_df, temp=None, press=1.0, mass=None, init=0, last=None):
        """
        LAMMPS.heat_capacity_Cv_NVT

        Calculate isochoric specific heat capacity from thermodynamic data in a log file (for NVT)
        Cv = Var(U)/(m*kB*T)

        Args:
            thermo_df: Pandas Data Frame of thermodynamic data

        Optional args:
            temp: Temperature (If None, temperature in thermodynamic data is used) (float, K)
            mass: Mass (If None, mass in thermodynamic data (density*volume) is used) (float, kg)
            init: Initial step (int)
            last: Last step (int)

        Return:
            isobaric heat capacity (float, J/(kg K))
        """

        if 'Volume' in thermo_df.columns:
            V = thermo_df['Volume'].to_numpy() * 1e-30 # Angstrom**3 -> m**3
        else:
            V = thermo_df['Lx'].to_numpy() * thermo_df['Ly'].to_numpy() * thermo_df['Lz'].to_numpy() * 1e-30 # Angstrom**3 -> m**3

        M = V * (thermo_df['Density'].to_numpy() * 1e+3) # m**3 * (g/cm**3) -> m**3 * (kg/m**3) = kg

        T = thermo_df['Temp'].to_numpy() # K

        if 'TotEng' in thermo_df.columns:
            U = thermo_df['TotEng'].to_numpy() * const.cal2j * 1000 / const.NA # kcal/mol -> J
        else:
            U = (thermo_df['PotEng'].to_numpy() + thermo_df['KinEng'].to_numpy()) * const.cal2j * 1000 / const.NA # kcal/mol -> J

        mM = M[init:last].mean() if mass is None else mass
        mT = T[init:last].mean() if temp is None else temp
        U_var = np.var(U[init:last])

        Cv = U_var / (mM * const.kB * mT**2) # J**2 / (kg * J/K * K**2) -> J/(kg K)

        return Cv


    @classmethod
    def isothermal_compressibility(cls, thermo_df, temp=None, press=1.0, mass=None, init=0, last=None):
        """
        LAMMPS.isothermal_compressibility

        Calculate isothermal compressibility from thermodynamic data in a log file
        beta_T = Var(V)/(V*kB*T)

        Args:
            thermo_df: Pandas Data Frame of thermodynamic data

        Optional args:
            temp: Temperature (If None, temperature in thermodynamic data is used) (float, K)
            init: Initial step (int)
            last: Last step (int)

        Return:
            isothermal compressibility (float, Pa**-1 = m s**2 / kg)
        """

        if 'Volume' in thermo_df.columns:
            V = thermo_df['Volume'].to_numpy() * 1e-30 # Angstrom**3 -> m**3
        else:
            V = thermo_df['Lx'].to_numpy() * thermo_df['Ly'].to_numpy() * thermo_df['Lz'].to_numpy() * 1e-30 # Angstrom**3 -> m**3

        T = thermo_df['Temp'].to_numpy() # K

        mV = V[init:last].mean()
        mT = T[init:last].mean() if temp is None else temp
        V_var = np.var(V[init:last])

        beta_T = V_var / (mV * const.kB * mT) # m**6 / (m**3 * J/K * K) = m**3/J = m**3 * (s**2/kg m**2) = m s**2 / kg

        return beta_T


    @classmethod
    def isentropic_compressibility(cls, thermo_df, temp=None, press=1.0, mass=None, init=0, last=None):
        """
        LAMMPS.isentropic_compressibility

        Calculate isentropic (or adiabatic) compressibility from thermodynamic data in a log file
        beta_S = bata_T*Cv/Cp

        Args:
            thermo_df: Pandas Data Frame of thermodynamic data

        Optional args:
            temp: Temperature (If None, temperature in thermodynamic data is used) (float, K)
            press: Pressure (float, atm)
            mass: Mass (If None, mass in thermodynamic data (density*volume) is used) (float, kg)
            init: Initial step (int)
            last: Last step (int)

        Return:
            isentropic compressibility (float, Pa**-1 = m s**2 / kg)
        """

        Cp = cls.heat_capacity_Cp(thermo_df, temp=temp, press=press, mass=mass, init=init, last=last)
        Cv = cls.heat_capacity_Cv(thermo_df, temp=temp, press=press, mass=mass, init=init, last=last)
        beta_T = cls.isothermal_compressibility(thermo_df, temp=temp, init=init, last=last)

        beta_S = beta_T * Cv / Cp

        return beta_S


    @classmethod
    def bulk_modulus(cls, thermo_df, temp=None, press=1.0, mass=None, init=0, last=None):
        """
        LAMMPS.bulk_modulus

        Calculate (isothermal) bulk modulus from thermodynamic data in a log file
        K_T = 1/beta_T

        Args:
            thermo_df: Pandas Data Frame of thermodynamic data

        Optional args:
            temp: Temperature (If None, temperature in thermodynamic data is used) (float, K)
            init: Initial step (int)
            last: Last step (int)

        Return:
            bulk modulus (float, Pa = kg / (m s**2))
        """

        return 1 / cls.isothermal_compressibility(thermo_df, temp=temp, init=init, last=last)


    @classmethod
    def isentropic_bulk_modulus(cls, thermo_df, temp=None, press=1.0, mass=None, init=0, last=None):
        """
        LAMMPS.isentropic_bulk_modulus

        Calculate isentropic bulk modulus from thermodynamic data in a log file
        K_S = 1/beta_S

        Args:
            thermo_df: Pandas Data Frame of thermodynamic data

        Optional args:
            temp: Temperature (If None, temperature in thermodynamic data is used) (float, K)
            press: Pressure (float, atm)
            mass: Mass (If None, mass in thermodynamic data (density*volume) is used) (float, kg)
            init: Initial step (int)
            last: Last step (int)

        Return:
            isentropic bulk modulus (float, Pa = kg / (m s**2))
        """

        return 1 / cls.isentropic_compressibility(thermo_df, temp=temp, press=press, mass=mass, init=init, last=last)


    @classmethod
    def speed_of_sound_lq(cls, thermo_df, temp=None, press=1.0, mass=None, init=0, last=None):
        """
        LAMMPS.speed_of_sound_lq

        Calculate speed of sound as a liquid from thermodynamic data in a log file
        The Newton–Laplace equation: c = sqrt(1/(rho*beta_S)) or c = sqrt(K_S/rho)

        Args:
            thermo_df: Pandas Data Frame of thermodynamic data

        Optional args:
            temp: Temperature (If None, temperature in thermodynamic data is used) (float, K)
            press: Pressure (float, atm)
            mass: Mass (If None, mass in thermodynamic data (density*volume) is used) (float, kg)
            init: Initial step (int)
            last: Last step (int)

        Return:
            speed of sound (float, m/s)
        """

        beta_S = cls.isentropic_compressibility(thermo_df, temp=temp, press=press, mass=mass, init=init, last=last) # m s**2 / kg
        #K_S = cls.isentropic_bulk_modulus(thermo_df, temp=temp, press=press, mass=mass, init=init, last=last) # kg / m s**2
        rho = thermo_df['Density'].to_numpy() * 1e+3 # g/cm**3 -> kg/m**3
        m_rho = rho[init:last].mean()

        c = math.sqrt(1/(m_rho*beta_S)) # sqrt( (m**3/kg) * (kg / (m s**2)) ) = sqrt(m**2 / s**2) = m/s
        #c = math.sqrt(K_S/m_rho) # sqrt( (kg / (m s**2)) / (kg/m**3) ) = sqrt(m**2 / s**2) = m/s

        return c


    @classmethod
    def volume_expansion(cls, thermo_df, temp=None, press=1.0, mass=None, init=0, last=None):
        """
        LAMMPS.volume_expansion

        Calculate (isobaric volumetric) thermal expansion coefficient from thermodynamic data in a log file
        alpha_P = Cov(V, H) / (V*kB*T**2)

        Args:
            thermo_df: Pandas Data Frame of thermodynamic data

        Optional args:
            temp: Temperature (If None, temperature in thermodynamic data is used) (float, K)
            press: Pressure (float, atm)
            init: Initial step (int)
            last: Last step (int)

        Return:
            volume expansion (float, K**-1)
        """

        if 'Volume' in thermo_df.columns:
            V = thermo_df['Volume'].to_numpy() * 1e-30 # Angstrom**3 -> m**3
        else:
            V = thermo_df['Lx'].to_numpy() * thermo_df['Ly'].to_numpy() * thermo_df['Lz'].to_numpy() * 1e-30 # Angstrom**3 -> m**3

        T = thermo_df['Temp'].to_numpy() # K

        U = thermo_df['TotEng'].to_numpy() * const.cal2j * 1000 / const.NA # kcal/mol -> J
        P = press * const.atm2pa # Pa = J / m**3
        H = U + P * V

        mV = V[init:last].mean()
        mT = T[init:last].mean() if temp is None else temp
        VH_cov = np.sum((V[init:last] - mV)*(H[init:last] - H[init:last].mean())) / len(V[init:last])

        alpha_P = VH_cov / (mV * const.kB * mT**2) # m**3 * J / (m**3 * J/K * K**2) = 1/K

        return alpha_P


    @classmethod
    def linear_expansion(cls, thermo_df, temp=None, press=1.0, mass=None, init=0, last=None):
        """
        LAMMPS.linear_expansion

        Calculate (isotropic, isobaric) linear expansion coefficient from thermodynamic data in a log file
        alpha_lP = alpha_P / 3

        Args:
            thermo_df: Pandas Data Frame of thermodynamic data

        Optional args:
            temp: Temperature (If None, temperature in thermodynamic data is used) (float, K)
            press: Pressure (float, atm)
            init: Initial step (int)
            last: Last step (int)

        Return:
            linear expansion (float, K**-1)
        """

        return cls.volume_expansion(thermo_df, temp=temp, press=press, init=init, last=last) / 3


    @classmethod
    def linear_expansion_aniso(cls, thermo_df, temp=None, press=1.0, mass=None, init=0, last=None):
        """
        LAMMPS.linear_expansion_aniso

        Calculate anisotropic (isobaric) linear expansion coefficient from thermodynamic data in a log file
        alpha_lP = alpha_P / 3

        Args:
            thermo_df: Pandas Data Frame of thermodynamic data

        Optional args:
            temp: Temperature (If None, temperature in thermodynamic data is used) (float, K)
            press: Pressure (float, atm)
            init: Initial step (int)
            last: Last step (int)

        Return:
            linear expansion x, y, z (float, K**-1)
        """

        Lx = thermo_df['Lx'].to_numpy() * 1e-10 # Angstrom -> m
        Ly = thermo_df['Ly'].to_numpy() * 1e-10
        Lz = thermo_df['Lz'].to_numpy() * 1e-10

        T = thermo_df['Temp'].to_numpy() # K

        U = thermo_df['TotEng'].to_numpy() * const.cal2j * 1000 / const.NA # kcal/mol -> J
        P = press * const.atm2pa # Pa = J / m**3
        H = U + P * (Lx * Ly * Lz)

        mLx = Lx[init:last].mean()
        mLy = Ly[init:last].mean()
        mLz = Lz[init:last].mean()
        mT = T[init:last].mean() if temp is None else temp
        mH = H[init:last].mean()
        N = len(T[init:last])

        LxH_cov = np.sum((Lx[init:last] - mLx)*(H[init:last] - mH)) / N
        LyH_cov = np.sum((Ly[init:last] - mLy)*(H[init:last] - mH)) / N
        LzH_cov = np.sum((Lz[init:last] - mLz)*(H[init:last] - mH)) / N

        alpha_Lx = LxH_cov / (mLx * const.kB * mT**2) # m * J / (m * J/K * K**2) = 1/K
        alpha_Ly = LyH_cov / (mLy * const.kB * mT**2)
        alpha_Lz = LzH_cov / (mLz * const.kB * mT**2)

        return alpha_Lx, alpha_Ly, alpha_Lz


    @classmethod
    def self_diffusion(cls, thermo_df, temp=None, press=1.0, mass=None, init=0, last=None):
        """
        LAMMPS.self_diffusion

        Calculate self-diffusion coefficient from thermodynamic data in a log file
        D = 1/6 * d(MSD)/dt

        Args:
            thermo_df: Pandas Data Frame of thermodynamic data

        Optional args:
            temp: Temperature (If None, temperature in thermodynamic data is used) (float, K)
            press: Pressure (float, atm)
            init: Initial step (int)
            last: Last step (int)

        Return:
            self-diffusion coefficient (float, m**2/s)
        """

        MSD = thermo_df['v_msd'].to_numpy() * 1e-20 # Angstrom^2 -> m^2
        t = thermo_df['Time'].to_numpy() * 1e-15 # fs -> s
        grad, k, r, p, se = stats.linregress(t[init:last], MSD[init:last])
        diffc = grad / 6

        return diffc


    def read_traj(self, traj_file=None, pdb_file=None, traj_type=None):

        if not mdtraj_avail:
            utils.radon_print('mdtraj is not available. You can use read_traj by "conda install -c conda-forge mdtraj"', level=3)
            return None

        if traj_file is None:
            traj_file = self.traj_file
        if pdb_file is None:
            pdb_file = self.pdb_file

        if traj_type is None:
            if 'dump' in traj_file: traj_type = 'dump'
            elif 'xtc' in traj_file: traj_type = 'xtc'
            else:
                utils.radon_print('read_traj can not specified the format of trajectory file %s' % traj_file, level=3)
                return None

        if traj_type == 'dump':
            self.traj = mdtraj.load_lammpstrj(traj_file, top=pdb_file)
        elif traj_type == 'xtc':
            self.traj = mdtraj.load_xtc(traj_file, top=pdb_file)

        return self.traj


    def analyze_traj(self, traj, prop, conv_a=1.0, conv_b=0.0, ylabel=None, temp=300, periodic=False, charges=None, enthalpy=None,
                     init=1000, last=None, width=1000, printout=True, save=None):

        if not mdtraj_avail:
            utils.radon_print('mdtraj is not available. You can use analyze_traj by "conda install -c conda-forge mdtraj"', level=3)
            return None, None

        data = {}
        chain_prop_data = []
        chain_id_code = const.pdb_id

        if type(prop) is list:
            prop_data = pd.Series(prop, index=traj.time)
            
        elif prop == 'rmsd':
            #prop_data = pd.Series(mdtraj.rmsd(traj, traj, frame=0), index=traj.time)
            for i in range(traj.n_chains):
                atom_list = traj.topology.select("chainid %s" % (i))
                traj_tmp = traj.atom_slice(atom_list)
                prop_tmp = mdtraj.rmsd(traj_tmp, traj_tmp, frame=0)
                chain_prop_data.append(prop_tmp)

            chain_prop_data = np.array(chain_prop_data)
            prop_data = np.mean(chain_prop_data, axis=0)
            prop_data = pd.Series(prop_data, index=traj.time)

        elif prop == 'r2':
            res1 = 0
            res2 = 0
            for i in range(traj.n_chains):
                atom_list = traj.topology.select("chainid %s" % (i))
                traj_tmp = traj.atom_slice(atom_list)

                for j, r in enumerate(traj_tmp.topology.residues):
                    if r.name == 'TU0':
                        res1 = j
                    elif r.name == 'TU1':
                        res2 = j

                prop_tmp = mdtraj.compute_contacts(traj_tmp, contacts=[(res1, res2)], scheme='closest', periodic=periodic)
                chain_prop_data.append(prop_tmp[0].flatten())
            
            chain_prop_data = np.array(chain_prop_data)
            prop_data = np.mean(chain_prop_data**2, axis=0)
            prop_data = pd.Series(prop_data, index=traj.time)

        elif prop == 'rg':
            #prop_data = pd.Series(mdtraj.compute_rg(traj), index=traj.time)
            for i in range(traj.n_chains):
                atom_list = traj.topology.select("chainid %s" % (i))
                traj_tmp = traj.atom_slice(atom_list)
                prop_tmp = mdtraj.compute_rg(traj_tmp)
                chain_prop_data.append(prop_tmp)
            
            chain_prop_data = np.array(chain_prop_data)
            prop_data = np.mean(chain_prop_data, axis=0)
            prop_data = pd.Series(prop_data, index=traj.time)
            
        elif prop == 'density': 
            prop_data = pd.Series(mdtraj.density(traj), index=traj.time)
            
        elif prop == 'order_param':
            prop_data = pd.Series(mdtraj.compute_nematic_order(traj, indices='residues'), index=traj.time)
            data['director'] = mdtraj.compute_directors(traj, indices='residues')
            
        elif prop == 'dipole_moments': # nm * e
            data['dipole'] = pd.DataFrame(mdtraj.dipole_moments(traj, charges), columns=['x', 'y', 'z'], index=traj.time)
            dipole_data = np.sqrt(data['dipole'].x.values**2 + data['dipole'].y.values**2 + data['dipole'].z.values**2)
            prop_data = pd.Series(dipole_data, index=traj.time)
            
        elif prop == 'dielectric': 
            n = traj[init:last:width].n_frames
            prop_data_tmp = []
            time = []
            for i in range(n):
                diele = np.mean(mdtraj.static_dielectric(traj[init-width:init+i*width], charges, temp))
                prop_data_tmp.append(diele)
                time.append(traj.time[init+i*width])
            if traj.n_frames > init+(n-1)*width:
                diele = np.mean(mdtraj.static_dielectric(traj[-width:], charges, temp))
                prop_data_tmp.append(diele)
                time.append(traj.time[-1])
            prop_data = pd.Series(prop_data_tmp, index=time)

        elif prop == 'diffusion_coeff': # m**2/s
            n = traj[init:last].n_frames
            msd = mdtraj.rmsd(traj, traj, frame=0) ** 2
            prop_data_tmp = []
            for i in range(n):
                time = traj.time[init-width+i:init+1+i] - traj.time[init-width+i]
                grad, k, r, p, std = stats.linregress(msd[init-width+i:init+1+i], time)
                prop_data_tmp.append(grad/6 * 1e-8)  # angstrom**2/ps -> m**2/s
            prop_data = pd.Series(prop_data_tmp, index=traj.time[init:last])
            
        elif prop == 'compressibility': # bar^-1
            n = traj[init:last:width].n_frames
            prop_data_tmp = []
            time = []
            for i in range(n):
                kappa_T = np.mean(mdtraj.isothermal_compressability_kappa_T(traj[init-width:init+i*width], temp))
                prop_data_tmp.append(kappa_T)
                time.append(traj.time[init+i*width])
            if traj.n_frames > init+(n-1)*width:
                kappa_T = np.mean(mdtraj.isothermal_compressability_kappa_T(traj[-width:], temp))
                prop_data_tmp.append(kappa_T)
                time.append(traj.time[-1])
            prop_data = pd.Series(prop_data_tmp, index=time)
            
        elif prop == 'expansion': # K^-1
            n = traj[init:last:width].n_frames
            prop_data_tmp = []
            time = []
            for i in range(n):
                alpha_P = np.mean(mdtraj.thermal_expansion_alpha_P(traj[init-width:init+i*width], temp, enthalpy))
                prop_data_tmp.append(alpha_P)
                time.append(traj.time[init+i*width])
            if traj.n_frames > init+(n-1)*width:
                alpha_P = np.mean(mdtraj.thermal_expansion_alpha_P(traj[-width:], temp, enthalpy))
                prop_data_tmp.append(alpha_P)
                time.append(traj.time[-1])
            prop_data = pd.Series(prop_data_tmp, index=time)
            
        prop_data = prop_data * conv_a + conv_b
        data_sma = prop_data.rolling(width).mean()
        data_sd = prop_data.rolling(width).std()
        data_se = data_sd / np.sqrt(width)

        if not last: last = 0
        if traj.n_frames < last: last = 0

        data['init'] = traj.time[last-width-1]
        data['last'] = traj.time[last-1]
        data['width'] = width
        if prop in ['dielectric', 'compressibility', 'expansion']:
            data['mean'] = prop_data.values[-1]
        else:
            data['mean'] = data_sma.values[int(last-1)]
            data['sd'] = data_sd.values[int(last-1)]
            data['se'] = data_se.values[int(last-1)]
            data['sma_sd'] = data_sma.values[last-width-1:last].std() if last else data_sma.values[-width-1:].std()
            data['sma_se'] = data['sma_sd'] / np.sqrt(width)

        if printout or save:
            if not last: last = None
            fig, ax = pp.subplots(figsize=(6, 6))
            ax.ticklabel_format(style="sci",  axis="y", scilimits=(0,0))

            if prop in ['dielectric', 'compressibility', 'expansion']:
                ax.plot(time, prop_data.values, linewidth=2.0)
            else:
                ax.plot(traj.time[init:last], prop_data.values[init:last], linewidth=0.1)

            if prop in ['dielectric', 'compressibility', 'expansion']:
                pass
            elif prop in ['diffusion_coeff']:
                ax.errorbar(data_sma.index[init:last], data_sma.values[init:last], yerr=data_se.values[init+width:last]*2, linewidth=2.0)
            else:
                ax.errorbar(data_sma.index[init:last], data_sma.values[init:last], yerr=data_se.values[init:last]*2, linewidth=2.0)
            ax.set_xlabel('Time [ps]', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            output = 'Accumulation of %f - %f ps\n' % (data['init'], data['last'])
            if prop in ['dielectric', 'compressibility', 'expansion']:
                output += '%s = %e' % (ylabel, data['mean'])
            else:
                output += '%s = %e     SD = %e    SE = %e' % (ylabel, data['mean'], data['sd'], data['se'])
                output += 'SMA_SD = %e     SMA_SE = %e' % (data['sma_sd'], data['sma_se'])
            
            if printout:
                pp.show()
                print(output)

            if save:
                if not os.path.exists(save):
                    os.makedirs(save)
                label = ylabel if type(prop) is list else prop
                fig.savefig(os.path.join(save, label)+'.png')
                with open(os.path.join(save, label)+'.txt', mode='w') as f:
                    f.write(output)
            
            pp.close(fig)

        return prop_data, data


    @classmethod
    def read_ave(cls, ave_file):
        """
        LAMMPS.read_ave

        Read average data file

        Args:
            ave_file: Path of average data file

        Return:
            Array of Pandas Data Frame
        """

        with open(ave_file, "r") as fh:
            ave_data = [s.replace('\n', '').replace('\r', '') for s in fh.readlines()]

        try:
            dfs = cls.parse_ave(ave_data)
        except Exception as e:
            dfs = None
            utils.radon_print('Can not read average file. %s; %s' % (ave_file, e), level=2)

        return dfs


    @classmethod
    def parse_ave(cls, ave_data):
        """
        LAMMPS.parse_ave

        Parser of average data file

        Args:
            ave_data: Strings of average data

        Return:
            Array of Pandas Data Frame or Panel
        """
        header = 0
        header1 = []
        header2 = []
        df_index = []
        columns = []
        nrow_index = 1
        dim = 2
        row_count = 0
        data = []

        for line in ave_data:
            if len(line) > 0 and line[0] == '#':
                if header == 0:
                    header = 1
                    ave_type = line.split()[1]
                elif header == 1:
                    header = 2
                    header1 = line.split()[1:]
                    columns = ['Timestep', *header1[1:]]
                    df_index = 'Timestep'
                elif header == 2:
                    header = 3
                    header2 = line.split()[1:]
                    if 'Number-of-rows' in header1:
                        nrow_index = header1.index('Number-of-rows')
                    elif 'Number-of-chunks' in header1:
                        nrow_index = header1.index('Number-of-chunks')
                    elif 'Number-of-time-windows' in header1:
                        nrow_index = header1.index('Number-of-time-windows')
                    df_index = ['Timestep', 'Row']
                    columns = ['Timestep', 'Row', *header2[1:]]
                    dim = 3
                continue
            
            m = line.split()
            if dim == 2:
                data.append(np.array(m, dtype=np.float64))

            elif dim == 3:
                if row_count == 0:
                    timestep = m[0]
                    nrow = int(m[nrow_index])
                    row_count = 1
                elif row_count < nrow:
                    data.append([timestep, *np.array(m, dtype=np.float64)])
                    row_count += 1
                elif row_count == nrow:
                    data.append([timestep, *np.array(m, dtype=np.float64)])
                    row_count = 0

        df = pd.DataFrame(data, columns=columns, copy=True)
        df = df.set_index(df_index)

        return df


    def calc_rg(self, rg_file=None, ave_type=0, init=-2000, last=None):

        if rg_file is None: rg_file = self.rg_file
        df = self.read_ave(rg_file)
        if df is None:
            utils.radon_print('Fail to calculate radius of gyration. %s' % (rg_file), level=2)
            return None

        rg = []
        for index1 in df.index.unique(level=0):
            data = df.loc[index1].to_numpy(dtype=np.float64)
            rg.append(data)

        rg = np.array(rg)
        rg_mean = np.mean(rg[init:last], axis=ave_type).flatten()
        rg_sd = np.std(rg[init:last], axis=ave_type).flatten()
        length = len(rg[init:last])

        rg_data = {
            'mean': rg_mean,
            'sd': rg_sd,
            'se': rg_sd / np.sqrt(length),
            'sd_max': np.max(rg_sd),
            'se_max': np.max(rg_sd) / np.sqrt(length),
            'mean_mean': np.mean(rg_mean),
            'mean_sd': np.std(rg_mean),
            'mean_se': np.std(rg_mean) / np.sqrt(len(rg_mean))
        }
    
        return rg_data


    def get_partial_charges(self, dat_file=None):

        if dat_file is None:
            dat_file = self.dat_file
        with open(dat_file, "r") as f:
            dat_data = [s.replace('\n', '').replace('\r', '') for s in f.readlines()]

        flag = 0
        charges = []
        for line in dat_data:
            if line.find('atoms') >= 0:
                n_atom = int(line.split()[0])
                charges = [0.0 for x in range(n_atom)]
            elif line.find('Atoms') == 0:
                flag = 1
            elif line.find('Velocities') == 0:
                break
            elif flag and line:
                charges[int(line.split()[0]) - 1] = float(line.split()[3])

        self.charges = np.array(charges, dtype="float")

        return self.charges


    def get_all_prop(self, temp=300.0, press=1.0, width=2000, init=2000, last=None, f_width=2000,
                printout=False, save=False, save_name='analyze', do_traj=True):

        thermo_df = self.dfs[-1]

        if save:
            save_dir = os.path.join(os.path.dirname(self.log_file), save_name)
        else:
            save_dir = None

        self.totene_data = self.analyze_thermo('TotEng', conv_a=const.cal2j, ylabel='Total energy [kJ/mol]',
                            width=width, init=init, last=last, printout=printout, save=save_dir)

        self.kinene_data = self.analyze_thermo('KinEng', conv_a=const.cal2j, ylabel='Kinetic energy [kJ/mol]',
                            width=width, init=init, last=last, printout=printout, save=save_dir)

        self.ebond_data = self.analyze_thermo('E_bond', conv_a=const.cal2j, ylabel='Potential energy of bonds [kJ/mol]',
                            width=width, init=init, last=last, printout=printout, save=save_dir)

        self.eangle_data = self.analyze_thermo('E_angle', conv_a=const.cal2j, ylabel='Potential energy of angles [kJ/mol]',
                            width=width, init=init, last=last, printout=printout, save=save_dir)

        self.edihed_data = self.analyze_thermo('E_dihed', conv_a=const.cal2j, ylabel='Potential energy of dihedrals [kJ/mol]',
                            width=width, init=init, last=last, printout=printout, save=save_dir)

        self.evdw_data = self.analyze_thermo('E_vdwl', conv_a=const.cal2j, ylabel='Potential energy of vdW [kJ/mol]',
                            width=width, init=init, last=last, printout=printout, save=save_dir)

        self.ecoul_data = self.analyze_thermo('E_coul', conv_a=const.cal2j, ylabel='Potential energy of coulomb [kJ/mol]',
                            width=width, init=init, last=last, printout=printout, save=save_dir)

        self.elong_data = self.analyze_thermo('E_long', conv_a=const.cal2j, ylabel='Potential energy of Kspace [kJ/mol]',
                            width=width, init=init, last=last, printout=printout, save=save_dir)

        self.temp_data = self.analyze_thermo('Temp', ylabel='Temperature [K]',
                            width=width, init=init, last=last, printout=printout, save=save_dir)

        self.dens_data = self.analyze_thermo('Density', ylabel='Density [g/cm^3]',
                            width=width, init=init, last=last, printout=printout, save=save_dir)

        if os.path.exists(self.rg_file):
            self.rg_data = self.calc_rg(rg_file=self.rg_file, init=-width, last=last)

        if 'v_msd' in thermo_df.columns.tolist():
            self.msd_data = self.analyze_thermo('v_msd', ylabel='MSD [Angstrome^2]', 
                            width=width, init=init, last=last, printout=printout, save=save_dir)
            self.diffc_data = self.analyze_thermo_fluctuation(self.self_diffusion,
                            name='self_diffusion', ylabel='Self-diffusion coeffisient [m^2/s]', f_width=f_width,
                            temp=temp, press=press, init=init, width=width, last=last, printout=printout, save=save_dir)

        self.Cp_data = self.analyze_thermo_fluctuation(self.heat_capacity_Cp,
                            name='heat_capacity_Cp', ylabel='Heat capacity Cp [J/(kg K)]', f_width=f_width,
                            temp=temp, press=press, init=init, width=width, last=last, printout=printout, save=save_dir)
        
        self.Cv_data = self.analyze_thermo_fluctuation(self.heat_capacity_Cv,
                            name='heat_capacity_Cv', ylabel='Heat capacity Cv [J/(kg K)]', f_width=f_width,
                            temp=temp, press=press, init=init, width=width, last=last, printout=printout, save=save_dir)
        
        self.compress_T_data = self.analyze_thermo_fluctuation(self.isothermal_compressibility,
                            name='isothermal_compressibility', ylabel='Isothermal compressibility [1/Pa]', f_width=f_width,
                            temp=temp, press=press, init=init, width=width, last=last, printout=printout, save=save_dir)
        
        self.compress_S_data = self.analyze_thermo_fluctuation(self.isentropic_compressibility,
                            name='isentropic_compressibility', ylabel='Isentropic compressibility [1/Pa]', f_width=f_width,
                            temp=temp, press=press, init=init, width=width, last=last, printout=printout, save=save_dir)
        
        self.bulk_mod_T_data = self.analyze_thermo_fluctuation(self.bulk_modulus,
                            name='bulk_modulus', ylabel='Bulk modulus [Pa]', f_width=f_width,
                            temp=temp, press=press, init=init, width=width, last=last, printout=printout, save=save_dir)
        
        self.bulk_mod_S_data = self.analyze_thermo_fluctuation(self.isentropic_bulk_modulus,
                            name='isentropic_bulk_modulus', ylabel='Isentropic bulk modulus [Pa]', f_width=f_width,
                            temp=temp, press=press, init=init, width=width, last=last, printout=printout, save=save_dir)
        
        self.volume_exp_data = self.analyze_thermo_fluctuation(self.volume_expansion,
                            name='volume_expansion', ylabel='Volume expansion [1/K]', f_width=f_width,
                            temp=temp, press=press, init=init, width=width, last=last, printout=printout, save=save_dir)
        
        self.linear_exp_data = self.analyze_thermo_fluctuation(self.linear_expansion,
                            name='linear_expansion', ylabel='Linear expansion [1/K]', f_width=f_width,
                            temp=temp, press=press, init=init, width=width, last=last, printout=printout, save=save_dir)

        prop_data = {
            'density': self.dens_data.get('mean', np.nan),
            'Rg': self.rg_data.get('mean_mean', np.nan),
            'self-diffusion': self.diffc_data.get('mean', np.nan),
            'Cp': self.Cp_data.get('mean', np.nan),
            'Cv': self.Cv_data.get('mean', np.nan),
            'compressibility': self.compress_T_data.get('mean', np.nan),
            'isentropic_compressibility': self.compress_S_data.get('mean', np.nan),
            'bulk_modulus': self.bulk_mod_T_data.get('mean', np.nan),
            'isentropic_bulk_modulus': self.bulk_mod_S_data.get('mean', np.nan),
            'volume_expansion': self.volume_exp_data.get('mean', np.nan),
            'linear_expansion': self.linear_exp_data.get('mean', np.nan),
        }

        conv_data = {
            'totene': self.totene_data.get('mean', np.nan),
            'totene_sd': self.totene_data.get('sd', np.nan),
            'totene_se': self.totene_data.get('se', np.nan),
            'totene_sma_sd': self.totene_data.get('sma_sd', np.nan),
            'totene_sma_se': self.totene_data.get('sma_se', np.nan),
            'kinene': self.kinene_data.get('mean', np.nan),
            'kinene_sd': self.kinene_data.get('sd', np.nan),
            'kinene_se': self.kinene_data.get('se', np.nan),
            'kinene_sma_sd': self.kinene_data.get('sma_sd', np.nan),
            'kinene_sma_se': self.kinene_data.get('sma_se', np.nan),
            'ebond': self.ebond_data.get('mean', np.nan),
            'ebond_sd': self.ebond_data.get('sd', np.nan),
            'ebond_se': self.ebond_data.get('se', np.nan),
            'ebond_sma_sd': self.ebond_data.get('sma_sd', np.nan),
            'ebond_sma_se': self.ebond_data.get('sma_se', np.nan),
            'eangle': self.eangle_data.get('mean', np.nan),
            'eangle_sd': self.eangle_data.get('sd', np.nan),
            'eangle_se': self.eangle_data.get('se', np.nan),
            'eangle_sma_sd': self.eangle_data.get('sma_sd', np.nan),
            'eangle_sma_se': self.eangle_data.get('sma_se', np.nan),
            'edihed': self.edihed_data.get('mean', np.nan),
            'edihed_sd': self.edihed_data.get('sd', np.nan),
            'edihed_se': self.edihed_data.get('se', np.nan),
            'edihed_sma_sd': self.edihed_data.get('sma_sd', np.nan),
            'edihed_sma_se': self.edihed_data.get('sma_se', np.nan),
            'evdw': self.evdw_data.get('mean', np.nan),
            'evdw_sd': self.evdw_data.get('sd', np.nan),
            'evdw_se': self.evdw_data.get('se', np.nan),
            'evdw_sma_sd': self.evdw_data.get('sma_sd', np.nan),
            'evdw_sma_se': self.evdw_data.get('sma_se', np.nan),
            'ecoul': self.ecoul_data.get('mean', np.nan),
            'ecoul_sd': self.ecoul_data.get('sd', np.nan),
            'ecoul_se': self.ecoul_data.get('se', np.nan),
            'ecoul_sma_sd': self.ecoul_data.get('sma_sd', np.nan),
            'ecoul_sma_se': self.ecoul_data.get('sma_se', np.nan),
            'elong': self.elong_data.get('mean', np.nan),
            'elong_sd': self.elong_data.get('sd', np.nan),
            'elong_se': self.elong_data.get('se', np.nan),
            'elong_sma_sd': self.elong_data.get('sma_sd', np.nan),
            'elong_sma_se': self.elong_data.get('sma_se', np.nan),
            'density_sd': self.dens_data.get('sd', np.nan),
            'density_se': self.dens_data.get('se', np.nan),
            'density_sma_sd': self.dens_data.get('sma_sd', np.nan),
            'density_sma_se': self.dens_data.get('sma_se', np.nan),
            'Rg_sd_max': self.rg_data.get('sd_max', np.nan),
            'Rg_se_max': self.rg_data.get('se_max', np.nan),
            'Rg_mean_sd': self.rg_data.get('mean_sd', np.nan),
            'Rg_mean_se': self.rg_data.get('mean_se', np.nan),
            'self-diff_sd': self.diffc_data.get('sd', np.nan),
            'self-diff_se': self.diffc_data.get('se', np.nan),
            'self-diff_sma_sd': self.diffc_data.get('sma_sd', np.nan),
            'self-diff_sma_se': self.diffc_data.get('sma_se', np.nan),
            'Cp_sd': self.Cp_data.get('sd', np.nan),
            'Cp_se': self.Cp_data.get('se', np.nan),
            'Cp_sma_sd': self.Cp_data.get('sma_sd', np.nan),
            'Cp_sma_se': self.Cp_data.get('sma_se', np.nan),
            'Cv_sd': self.Cv_data.get('sd', np.nan),
            'Cv_se': self.Cv_data.get('se', np.nan),
            'Cv_sma_sd': self.Cv_data.get('sma_sd', np.nan),
            'Cv_sma_se': self.Cv_data.get('sma_se', np.nan),
            'compress_T_sd': self.compress_T_data.get('sd', np.nan),
            'compress_T_se': self.compress_T_data.get('se', np.nan),
            'compress_T_sma_sd': self.compress_T_data.get('sma_sd', np.nan),
            'compress_T_sma_se': self.compress_T_data.get('sma_se', np.nan),
            'compress_S_sd': self.compress_S_data.get('sd', np.nan),
            'compress_S_se': self.compress_S_data.get('se', np.nan),
            'compress_S_sma_sd': self.compress_S_data.get('sma_sd', np.nan),
            'compress_S_sma_se': self.compress_S_data.get('sma_se', np.nan),
            'bulk_mod_T_sd': self.bulk_mod_T_data.get('sd', np.nan),
            'bulk_mod_T_se': self.bulk_mod_T_data.get('se', np.nan),
            'bulk_mod_T_sma_sd': self.bulk_mod_T_data.get('sma_sd', np.nan),
            'bulk_mod_T_sma_se': self.bulk_mod_T_data.get('sma_se', np.nan),
            'bulk_mod_S_sd': self.bulk_mod_S_data.get('sd', np.nan),
            'bulk_mod_S_se': self.bulk_mod_S_data.get('se', np.nan),
            'bulk_mod_S_sma_sd': self.bulk_mod_S_data.get('sma_sd', np.nan),
            'bulk_mod_S_sma_se': self.bulk_mod_S_data.get('sma_se', np.nan),
            'volume_exp_sd': self.volume_exp_data.get('sd', np.nan),
            'volume_exp_se': self.volume_exp_data.get('se', np.nan),
            'volume_exp_sma_sd': self.volume_exp_data.get('sma_sd', np.nan),
            'volume_exp_sma_se': self.volume_exp_data.get('sma_se', np.nan),
            'linear_exp_sd': self.linear_exp_data.get('sd', np.nan),
            'linear_exp_se': self.linear_exp_data.get('se', np.nan),
            'linear_exp_sma_sd': self.linear_exp_data.get('sma_sd', np.nan),
            'linear_exp_sma_se': self.linear_exp_data.get('sma_se', np.nan),
        }

        if do_traj and mdtraj_avail:
            self.read_traj()
            self.get_partial_charges()

            self.r2, self.r2_data = self.analyze_traj(self.traj, 'r2', ylabel='<R2> [nm^2]',
                                width=width, init=init, last=last, printout=printout, save=save_dir, temp=temp, charges=self.charges)

            self.diele, self.diele_data = self.analyze_traj(self.traj, 'dielectric', ylabel='Static dielectric constant',
                                width=width, init=init, last=last, printout=printout, save=save_dir, temp=temp, charges=self.charges)

            self.nop, self.nop_data = self.analyze_traj(self.traj, 'order_param', ylabel='Nematic order parameter',
                                width=width, init=init, last=last, printout=printout, save=save_dir)

            prop_data['r2'] = self.r2_data.get('mean', np.nan)
            conv_data['r2_sd'] = self.r2_data.get('sd', np.nan)
            conv_data['r2_se'] = self.r2_data.get('se', np.nan)
            conv_data['r2_sma_sd'] = self.r2_data.get('sma_sd', np.nan)
            conv_data['r2_sma_se'] = self.r2_data.get('sma_se', np.nan)

            prop_data['static_dielectric_const'] = self.diele_data.get('mean', np.nan)

            prop_data['nematic_order_parameter'] = self.nop_data.get('mean', np.nan)
            conv_data['nematic_order_parameter_sd'] = self.nop_data.get('sd', np.nan)
            conv_data['nematic_order_parameter_se'] = self.nop_data.get('se', np.nan)
            conv_data['nematic_order_parameter_sma_sd'] = self.nop_data.get('sma_sd', np.nan)
            conv_data['nematic_order_parameter_sma_se'] = self.nop_data.get('sma_se', np.nan)

        elif not mdtraj_avail:
            utils.radon_print('mdtraj is not available. You can use analyze_traj by "conda install -c conda-forge mdtraj"', level=3)

        self.prop_df = pd.DataFrame(prop_data, index=[0])
        self.conv_df = pd.DataFrame(conv_data, index=[0])
        if save:
            self.prop_df.to_csv(os.path.join(save_dir, 'eq_prop_data.csv'))
            self.conv_df.to_csv(os.path.join(save_dir, 'eq_conv_data.csv'))

        return prop_data


    def check_eq(self, do_analyze=False, temp=300.0, press=1.0, width=2000, init=2000, last=None, f_width=2000,
                printout=False, save=False, save_name='analyze'):

        if do_analyze:
            self.get_all_prop(temp=temp, press=press, printout=printout, width=width, init=init, last=last,
                f_width=f_width,save=save, save_name=save_name, do_traj=False)

        check = True
        conv_data = {}

        if self.totene_data.get('sma_sd') is not None and self.totene_data.get('mean') is not None and self.totene_sma_sd_crit is not None:
            if self.totene_data['sma_sd'] > abs(self.totene_data['mean']) * self.totene_sma_sd_crit:
                utils.radon_print('Total energy does not converge. mean = %f, sma_sd = %f'
                    % (self.totene_data['mean'], self.totene_data['sma_sd']), level=2)
                check = False
        elif self.totene_sma_sd_crit is None: pass
        else:
            utils.radon_print('Can not obtain the data of total energy.', level=2)
            return False

        if self.kinene_data.get('sma_sd') is not None and self.kinene_data.get('mean') and self.kinene_sma_sd_crit is not None:
            if self.kinene_data['sma_sd'] > abs(self.kinene_data['mean']) * self.kinene_sma_sd_crit:
                utils.radon_print('Kinetic energy does not converge. mean = %f, sma_sd = %f'
                    % (self.kinene_data['mean'], self.kinene_data['sma_sd']), level=2)
                check = False
        elif self.kinene_sma_sd_crit is None: pass
        else:
            utils.radon_print('Can not obtain the data of kinetic energy. Skip to check the kinetic energy convergence.', level=2)

        if self.ebond_data.get('sma_sd') is not None and self.ebond_data.get('mean') is not None and self.ebond_sma_sd_crit is not None:
            if self.ebond_data['sma_sd'] > abs(self.ebond_data['mean']) * self.ebond_sma_sd_crit:
                utils.radon_print('Potential energy of bonds does not converge. mean = %f, sma_sd = %f'
                    % (self.ebond_data['mean'], self.ebond_data['sma_sd']), level=2)
                check = False
        elif self.ebond_sma_sd_crit is None: pass
        else:
            utils.radon_print('Can not obtain the data of the potential energy of bonds.', level=2)
            return False

        if self.eangle_data.get('sma_sd') is not None and self.eangle_data.get('mean') is not None and self.eangle_sma_sd_crit is not None:
            if self.eangle_data['sma_sd'] > abs(self.eangle_data['mean']) * self.eangle_sma_sd_crit:
                utils.radon_print('Potential energy of angles does not converge. mean = %f, sma_sd = %f'
                    % (self.eangle_data['mean'], self.eangle_data['sma_sd']), level=2)
                check = False
        elif self.eangle_sma_sd_crit is None: pass
        else:
            utils.radon_print('Can not obtain the data of the potential energy of angles.', level=2)
            return False

        if self.edihed_data.get('sma_sd') is not None and self.edihed_data.get('mean') is not None and self.edihed_sma_sd_crit is not None:
            if self.edihed_data['sma_sd'] > abs(self.edihed_data['mean']) * self.edihed_sma_sd_crit:
                utils.radon_print('Potential energy of dihedrals does not converge. mean = %f, sma_sd = %f'
                    % (self.edihed_data['mean'], self.edihed_data['sma_sd']), level=2)
                check = False
        elif self.edihed_sma_sd_crit is None: pass
        else:
            utils.radon_print('Can not obtain the data of the potential energy of dihedrals.', level=2)
            return False

        if self.evdw_data.get('sma_sd') is not None and self.evdw_sma_sd_crit is not None:
            if self.evdw_data['sma_sd'] > self.evdw_sma_sd_crit:
                utils.radon_print('Potential energy of vdW interactions does not converge. mean = %f, sma_sd = %f'
                    % (self.evdw_data['mean'], self.evdw_data['sma_sd']), level=2)
                check = False
        elif self.evdw_sma_sd_crit is None: pass
        else:
            utils.radon_print('Can not obtain the data of the potential energy of vdW interactions.', level=2)
            return False

        if self.ecoul_data.get('sma_sd') is not None and self.ecoul_sma_sd_crit is not None:
            if self.ecoul_data['sma_sd'] > self.ecoul_sma_sd_crit:
                utils.radon_print('Potential energy of coulomb interactions does not converge. mean = %f, sma_sd = %f'
                    % (self.ecoul_data['mean'], self.ecoul_data['sma_sd']), level=2)
                check = False
        elif self.ecoul_sma_sd_crit is None: pass
        else:
            utils.radon_print('Can not obtain the data of the potential energy of coulomb interactions.', level=2)
            return False

        if self.elong_data.get('sma_sd') is not None and self.elong_data.get('mean') is not None and self.elong_sma_sd_crit is not None:
            if self.elong_data['sma_sd'] > abs(self.elong_data['mean']) * self.elong_sma_sd_crit:
                utils.radon_print('Potential energy of KSpace does not converge. mean = %f, sma_sd = %f'
                    % (self.elong_data['mean'], self.elong_data['sma_sd']), level=2)
                check = False
        elif self.elong_sma_sd_crit is None: pass
        else:
            utils.radon_print('Can not obtain the data of potential energy of KSpace. Skip to check the KSpace energy convergence.', level=2)

        if self.dens_data.get('sma_sd') is not None and self.dens_data.get('mean') is not None and self.dens_sma_sd_crit is not None:
            if self.dens_data['sma_sd'] > self.dens_data['mean'] * self.dens_sma_sd_crit:
                utils.radon_print('Density does not converge. mean = %f, sma_sd = %f'
                    % (self.dens_data['mean'], self.dens_data['sma_sd']), level=2)
                check = False
        elif self.dens_sma_sd_crit is None: pass
        else:
            utils.radon_print('Can not obtain the data of the density.', level=2)
            return False

        if self.rg_data.get('sd_max') is not None and self.rg_data.get('mean_mean') is not None and self.rg_sd_crit is not None:
            if self.rg_data['sd_max'] > self.rg_data['mean_mean'] * self.rg_sd_crit:
                utils.radon_print('Radius of gyration does not converge. mean = %f, sd = %f'
                    % (self.rg_data['mean_mean'], self.rg_data['sd_max']), level=2)
                check = False
        elif self.rg_sd_crit is None: pass
        else:
            utils.radon_print('Can not obtain the data of radius of gyration. Skip to check the Rg convergence.', level=2)

        if self.diffc_data.get('sma_sd') is not None and self.diffc_data.get('mean') is not None and self.diffc_sma_sd_crit is not None:
            if self.diffc_data['sma_sd'] > self.diffc_data['mean'] * self.diffc_sma_sd_crit:
                utils.radon_print('Self-diffusion coefficient does not converge. mean = %f, sma_sd = %f'
                    % (self.diffc_data['mean'], self.diffc_data['sma_sd']), level=2)
                check = False
        elif self.diffc_sma_sd_crit is None: pass
        else:
            utils.radon_print('Can not obtain the data of self-diffusion coefficient. Skip to check the self-diffusion coefficient convergence.', level=1)

        if self.Cp_data.get('sma_sd') is not None and self.Cp_data.get('mean') is not None and self.Cp_sma_sd_crit is not None:
            if self.Cp_data['sma_sd'] > self.Cp_data['mean'] * self.Cp_sma_sd_crit:
                utils.radon_print('Cp does not converge. mean = %f, sma_sd = %f'
                    % (self.Cp_data['mean'], self.Cp_data['sma_sd']), level=2)
                check = False
        elif self.Cp_sma_sd_crit is None: pass
        else:
            utils.radon_print('Can not obtain the data of Cp. Skip to check the Cp convergence.', level=2)

        if self.compress_T_data.get('sma_sd') is not None and self.compress_T_data.get('mean') is not None and self.compress_sma_sd_crit is not None:
            if self.compress_T_data['sma_sd'] > self.compress_T_data['mean'] * self.compress_sma_sd_crit:
                utils.radon_print('Compressibility does not converge. mean = %f, sma_sd = %f'
                    % (self.compress_T_data['mean'], self.compress_T_data['sma_sd']), level=2)
                check = False
        elif self.compress_sma_sd_crit is None: pass
        else:
            utils.radon_print('Can not obtain the data of compressibility. Skip to check the compressibility convergence.', level=2)

        if self.volume_exp_data.get('sma_sd') is not None and self.volume_exp_data.get('mean') is not None and self.volexp_sma_sd_crit is not None:
            if self.volume_exp_data['sma_sd'] > self.volume_exp_data['mean'] * self.volexp_sma_sd_crit:
                utils.radon_print('Volumetric thermal expantion coefficient does not converge. mean = %f, sma_sd = %f'
                    % (self.volume_exp_data['mean'], self.volume_exp_data['sma_sd']), level=2)
                check = False
        elif self.volexp_sma_sd_crit is None: pass
        else:
            utils.radon_print('Can not obtain the data of volumetric thermal expantion coefficient. Skip to check the volumetric thermal expantion coefficient convergence.', level=2)

        return check



###########################################
# IO functions
###########################################

def MolToLAMMPSdata(mol, file_name, confId=0, velocity=True, temp=300, drude=False):

    block = MolToLAMMPSdataBlock(mol, confId=confId, velocity=velocity, temp=temp, drude=drude)
    if block is None: return False

    with open(file_name, 'w') as fh:
        fh.write('\n'.join(block)+'\n')
        fh.flush()
        if hasattr(os, 'fdatasync'):
            os.fdatasync(fh.fileno())
        else:
            os.fsync(fh.fileno())

    return True


def MolToLAMMPSdataBlock(mol, confId=0, velocity=True, temp=300, drude=False):

    unique_ptype = []
    p_mass = []
    p_coeff = []
    i = 0

    if not mol.HasProp('pair_style'):
        utils.radon_print('pair_style is missing in MolToLAMMPSdataBlock. Assuming lj for pair_style.', level=2)
        mol.SetProp('pair_style', 'lj')
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'H' and atom.GetIsotope() == 3:
            ptype = '%s,0' % atom.GetProp('ff_type')
        else:
            ptype = '%s,%i' % (atom.GetProp('ff_type'), atom.GetIsotope())
        if ptype in unique_ptype:
            atom.SetIntProp('ff_type_num', unique_ptype.index(ptype)+1)
        else:
            i += 1
            unique_ptype.append(ptype)
            p_mass.append(atom.GetMass())
            if mol.GetProp('pair_style') == 'lj':
                p_coeff.append([atom.GetDoubleProp('ff_epsilon'), atom.GetDoubleProp('ff_sigma')])
            else:
                utils.radon_print('pair_style %s is not available.' % mol.GetProp('pair_style'), level=3)
            atom.SetIntProp('ff_type_num', i)

    unique_btype = []
    b_coeff = []
    i = 0
    if not mol.HasProp('bond_style'):
        utils.radon_print('bond_style is missing in MolToLAMMPSdataBlock. Assuming harmonic for bond_style.', level=2)
        mol.GetProp('bond_style', 'harmonic')
    for bond in mol.GetBonds():
        btype = bond.GetProp('ff_type')
        if btype in unique_btype:
            bond.SetIntProp('ff_type_num', unique_btype.index(btype)+1)
        else:
            i += 1
            unique_btype.append(btype)
            if mol.GetProp('bond_style') == 'harmonic':
                b_coeff.append([bond.GetDoubleProp('ff_k'), bond.GetDoubleProp('ff_r0')])
            else:
                utils.radon_print('bond_style %s is not available.' % mol.GetProp('bond_style'), level=3)
            bond.SetIntProp('ff_type_num', i)

    unique_atype = []
    a_coeff = []
    i = 0
    if not mol.HasProp('angle_style'):
        utils.radon_print('angle_style is missing in MolToLAMMPSdataBlock. Assuming harmonic for angle_style.', level=2)
        mol.SetProp('angle_style', 'harmonic')
    if hasattr(mol, 'angles'):
        for angle in mol.angles:
            if angle.ff.type in unique_atype:
                angle.ff.type_num = unique_atype.index(angle.ff.type)+1
            else:
                i += 1
                unique_atype.append(angle.ff.type)
                if mol.GetProp('angle_style') == 'harmonic':
                    a_coeff.append([angle.ff.k, angle.ff.theta0])
                else:
                    utils.radon_print('angle_style %s is not available.' % mol.GetProp('angle_style'), level=3)
                angle.ff.type_num = i

    unique_dtype = []
    d_coeff = []
    i = 0
    if not mol.HasProp('dihedral_style'):
        utils.radon_print('dihedral_style is missing in MolToLAMMPSdataBlock. Assuming fourier for dihedral_style.', level=2)
        mol.SetProp('dihedral_style', 'fourier')
    if hasattr(mol, 'dihedrals'):
        for dihedral in mol.dihedrals:
            if dihedral.ff.type in unique_dtype:
                dihedral.ff.type_num = unique_dtype.index(dihedral.ff.type)+1
            else:
                i += 1
                unique_dtype.append(dihedral.ff.type)
                if mol.GetProp('dihedral_style') == 'fourier':
                    coeff = [dihedral.ff.m]
                    for j in range(len(dihedral.ff.k)):
                        coeff.extend([dihedral.ff.k[j], dihedral.ff.n[j], dihedral.ff.d0[j]])
                    d_coeff.append(coeff)
                elif mol.GetProp('dihedral_style') == 'harmonic':
                    d_coeff.append([dihedral.ff.k, dihedral.ff.d0, dihedral.ff.n])
                else:
                    utils.radon_print('dihedral_style %s is not available.' % mol.GetProp('dihedral_style'), level=3)
                dihedral.ff.type_num = i

    unique_itype = []
    i_coeff = []
    i = 0
    if not mol.HasProp('improper_style'):
        utils.radon_print('improper_style is missing in MolToLAMMPSdataBlock. Assuming cvff for improper_style.', level=2)
        mol.SetProp('improper_style', 'cvff')
    if hasattr(mol, 'impropers'):
        for improper in mol.impropers:
            if improper.ff.type in unique_itype:
                improper.ff.type_num = unique_itype.index(improper.ff.type)+1
            else:
                i += 1
                unique_itype.append(improper.ff.type)
                if mol.GetProp('improper_style') == 'cvff':
                    i_coeff.append([improper.ff.k, improper.ff.d0, improper.ff.n])
                elif mol.GetProp('improper_style') == 'umbrella':
                    i_coeff.append([improper.ff.k, improper.ff.x0])
                else:
                    utils.radon_print('improper_style %s is not available.' % mol.GetProp('improper_style'), level=3)
                improper.ff.type_num = i

    utils.set_mol_id(mol)

    lines = []
    lines.append('Generated by RadonPy')
    lines.append('')
    lines.append('%i atoms' % (mol.GetNumAtoms()))
    if len(unique_ptype) > 0: lines.append('%i atom types' % (len(unique_ptype)))
    lines.append('%i bonds' % (mol.GetNumBonds()))
    if len(unique_btype) > 0: lines.append('%i bond types' % (len(unique_btype)))
    lines.append('%i angles' % (len(mol.angles)))
    if len(unique_atype) > 0: lines.append('%i angle types' % (len(unique_atype)))
    lines.append('%i dihedrals' % (len(mol.dihedrals)))
    if len(unique_dtype) > 0: lines.append('%i dihedral types' % (len(unique_dtype)))
    if len(unique_itype) > 0:
        lines.append('%i impropers' % (len(mol.impropers)))
        lines.append('%i improper types' % (len(unique_itype)))

    if hasattr(mol, 'cell'):
        lines.append('')
        lines.append('%.16e %.16e xlo xhi' % (mol.cell.xlo, mol.cell.xhi))
        lines.append('%.16e %.16e ylo yhi' % (mol.cell.ylo, mol.cell.yhi))
        lines.append('%.16e %.16e zlo zhi' % (mol.cell.zlo, mol.cell.zhi))
    else:
        xhi, xlo, yhi, ylo, zhi, zlo = poly.calc_cell_length([mol], [1], fit='max_cubic', margin=(5,5,5), confId=confId)
        lines.append('')
        lines.append('%.16e %.16e xlo xhi' % (xlo, xhi))
        lines.append('%.16e %.16e ylo yhi' % (ylo, yhi))
        lines.append('%.16e %.16e zlo zhi' % (zlo, zhi))

    if len(unique_ptype) > 0:
        lines.append('')
        lines.append('Masses')
        lines.append('')

        for i, ptype in enumerate(unique_ptype):
            lines.append('%5d\t%f\t# %s' % (i+1, p_mass[i], ptype))

        lines.append('')
        lines.append('Pair Coeffs')
        lines.append('')

        for i, ptype in enumerate(unique_ptype):
            c = '\t'.join([ '%f' % x for x in p_coeff[i] ])
            #lines.append('%5d\t%f\t%f\t# %s' % (i+1, p_coeff[i][0], p_coeff[i][1], ptype))
            lines.append('%5d\t%s\t# %s' % (i+1, c, ptype))


    if len(unique_btype) > 0:
        lines.append('')
        lines.append('Bond Coeffs')
        lines.append('')

        for i, btype in enumerate(unique_btype):
            c = '\t'.join([ '%f' % x for x in b_coeff[i] ])
            #lines.append('%5d\t%f\t%f\t# %s' % (i+1, b_coeff[i][0], b_coeff[i][1], btype))
            lines.append('%5d\t%s\t# %s' % (i+1, c, btype))


    if len(unique_atype) > 0:
        lines.append('')
        lines.append('Angle Coeffs')
        lines.append('')

        for i, atype in enumerate(unique_atype):
            c = '\t'.join([ '%f' % x for x in a_coeff[i] ])
            #lines.append('%5d\t%f\t%f\t# %s' % (i+1, a_coeff[i][0], a_coeff[i][1], atype))
            lines.append('%5d\t%s\t# %s' % (i+1, c, atype))


    if len(unique_dtype) > 0:
        lines.append('')
        lines.append('Dihedral Coeffs')
        lines.append('')

        for i, dtype in enumerate(unique_dtype):
            # string = '%5d\t%i' % (i+1, d_coeff[i][0])
            # for j in range(d_coeff[i][0]):
            #     string += '\t%f\t%i\t%f' % (d_coeff[i][j*3+1], d_coeff[i][j*3+2], d_coeff[i][j*3+3])
            # string += '\t# %s' % (dtype)
            # lines.append(string)
            c = '\t'.join([ '%f' % x if isinstance(x, float) else '%i' % x for x in d_coeff[i]])
            lines.append('%5d\t%s\t# %s' % (i+1, c, dtype))


    if len(unique_itype) > 0:
        lines.append('')
        lines.append('Improper Coeffs')
        lines.append('')

        for i, itype in enumerate(unique_itype):
            #lines.append('%5d\t%f\t%i\t%i\t# %s' % (i+1, i_coeff[i][0], i_coeff[i][1], i_coeff[i][2], itype))
            c = '\t'.join([ '%f' % x if isinstance(x, float) else '%i' % x for x in i_coeff[i] ])
            lines.append('%5d\t%s\t# %s' % (i+1, c, itype))


    if mol.GetNumAtoms() > 0:
        lines.append('')
        lines.append('Atoms')
        lines.append('')

        coord = mol.GetConformer(confId).GetPositions()
        for i, atom in enumerate(mol.GetAtoms()):
            lines.append('%5d\t%i\t%i\t% .16e\t% .16e\t% .16e\t% .16e' %
                (i+1, atom.GetIntProp('mol_id'), atom.GetIntProp('ff_type_num'), atom.GetDoubleProp('AtomicCharge'),
                    coord[i][0], coord[i][1], coord[i][2]))

        if velocity:
            lines.append('')
            lines.append('Velocities')
            lines.append('')

            if not mol.GetAtomWithIdx(0).HasProp('vx'):
                calc.set_velocity(mol, temp)

            for atom in mol.GetAtoms():
                vx = atom.GetDoubleProp('vx')
                vy = atom.GetDoubleProp('vy')
                vz = atom.GetDoubleProp('vz')
                lines.append('%5d\t% .16e\t% .16e\t% .16e' % (atom.GetIdx()+1, vx, vy, vz))


    if mol.GetNumBonds() > 0:
        lines.append('')
        lines.append('Bonds')
        lines.append('')

        for bond in mol.GetBonds():
            lines.append('%5d\t%i\t%i\t%i' %
                (bond.GetIdx()+1, bond.GetIntProp('ff_type_num'), bond.GetBeginAtom().GetIdx()+1, bond.GetEndAtom().GetIdx()+1))


    if len(mol.angles) > 0:
        lines.append('')
        lines.append('Angles')
        lines.append('')

        for i, angle in enumerate(mol.angles):
            lines.append('%5d\t%i\t%i\t%i\t%i' %
                (i+1, angle.ff.type_num, angle.a+1, angle.b+1, angle.c+1))

    if len(mol.dihedrals) > 0:
        lines.append('')
        lines.append('Dihedrals')
        lines.append('')

        for i, dihedral in enumerate(mol.dihedrals):
            lines.append('%5d\t%i\t%i\t%i\t%i\t%i' %
                (i+1, dihedral.ff.type_num, dihedral.a+1, dihedral.b+1, dihedral.c+1, dihedral.d+1))


    if len(mol.impropers) > 0:
        lines.append('')
        lines.append('Impropers')
        lines.append('')

        for i, improper in enumerate(mol.impropers):
            lines.append('%5d\t%i\t%i\t%i\t%i\t%i' %
                (i+1, improper.ff.type_num, improper.a+1, improper.b+1, improper.c+1, improper.d+1))

    return lines


def MolFromLAMMPSdata(file_name, bond_order=True,
    ff_style={'pair':'lj', 'bond':'harmonic', 'angle':'harmonic', 'dihedral':'fourier', 'improper':'cvff'}):

    flag = 'init'
    flag2 = False
    n_data = {}
    cell_data = {}
    mass = []
    pair = []
    epsilon = []
    sigma = []
    k_bond = []
    r0 = []
    k_angle = []
    theta0 = []
    m_dih = []
    k_dih = []
    n_dih = []
    d0_dih = []
    k_imp = []
    n_imp = []
    d0_imp = []

    with open(file_name, "r") as fh:
        for line in fh.readlines():
            line.replace('\n', '').replace('\r', '')

            if flag == 'init' and not line.strip():
                flag = 'n_atom'

            elif flag == 'n_atom':
                if not line.strip():
                    flag = 'cell'
                else:
                    vals = line.split()
                    key = vals[1] if len(vals) == 2 else '%s %s' % (vals[1], vals[2])
                    n_data[key] = int(vals[0])

            elif flag == 'cell':
                if not line.strip():
                    flag = 'none'
                else:
                    val1, val2, key1, key2 = line.split()
                    cell_data[key1] = float(val1)
                    cell_data[key2] = float(val2)

            elif flag == 'none':
                if line.find('Masses') >= 0:
                    flag = 'mass'
                elif line.find('Pair Coeffs') >= 0:
                    flag = 'pair_c'
                elif line.find('Bond Coeffs') >= 0:
                    flag = 'bond_c'
                elif line.find('Angle Coeffs') >= 0:
                    flag = 'angle_c'
                elif line.find('Dihedral Coeffs') >= 0:
                    flag = 'dihed_c'
                elif line.find('Improper Coeffs') >= 0:
                    flag = 'impro_c'
                elif line.find('Atoms') >= 0:
                    flag = 'atom'
                elif line.find('Velocities') >= 0:
                    flag = 'velocity'
                elif line.find('Bonds') >= 0:
                    flag = 'bond'
                elif line.find('Angles') >= 0:
                    flag = 'angle'
                elif line.find('Dihedrals') >= 0:
                    flag = 'dihed'
                elif line.find('Impropers') >= 0:
                    flag = 'impro'

            elif flag == 'mass':
                if not flag2:
                    flag2 = True
                elif flag2 and not line.strip():
                    flag = 'none'
                    flag2 = False
                elif flag2:
                    vals = line.split()
                    mass.append(float(vals[1]))

            elif flag == 'pair_c':
                if not flag2:
                    flag2 = True
                elif flag2 and not line.strip():
                    flag = 'none'
                    flag2 = False
                elif flag2:
                    vals = line.split()
                    if ff_style['pair'] == 'lj':
                        epsilon.append(float(vals[1]))
                        sigma.append(float(vals[2]))
                    else:
                        utils.radon_print('pair_style %s is not available.' % ff_style['pair'], level=3)

            elif flag == 'bond_c':
                if not flag2:
                    flag2 = True
                elif flag2 and not line.strip():
                    flag = 'none'
                    flag2 = False
                elif flag2:
                    vals = line.split()
                    if ff_style['bond'] == 'harmonic':
                        k_bond.append(float(vals[1]))
                        r0.append(float(vals[2]))
                    else:
                        utils.radon_print('bond_style %s is not available.' % ff_style['bond'], level=3)

            elif flag == 'angle_c':
                if not flag2:
                    flag2 = True
                elif flag2 and not line.strip():
                    flag = 'none'
                    flag2 = False
                elif flag2:
                    vals = line.split()
                    if ff_style['angle'] == 'harmonic':
                        k_angle.append(float(vals[1]))
                        theta0.append(float(vals[2]))
                    else:
                        utils.radon_print('angle_style %s is not available.' % ff_style['angle'], level=3)

            elif flag == 'dihed_c':
                if not flag2:
                    flag2 = True
                elif flag2 and not line.strip():
                    flag = 'none'
                    flag2 = False
                elif flag2:
                    k_dih_tmp = []
                    n_dih_tmp = []
                    d0_dih_tmp = []
                    vals = line.split()
                    if ff_style['dihedral'] == 'fourier':
                        m_dih.append(int(vals[1]))
                        for j in range(int(vals[1])):
                            k_dih_tmp.append(float(vals[j*3+2]))
                            n_dih_tmp.append(int(vals[j*3+3]))
                            d0_dih_tmp.append(float(vals[j*3+4]))
                        k_dih.append(k_dih_tmp)
                        n_dih.append(n_dih_tmp)
                        d0_dih.append(d0_dih_tmp)
                    elif ff_style['dihedral'] == 'harmonic':
                        k_dih.append(float(vals[1]))
                        n_dih.append(float(vals[2]))
                        d0_dih.append(float(vals[3]))
                    else:
                        utils.radon_print('dihedral_style %s is not available.' % ff_style['dihedral'], level=3)


            elif flag == 'impro_c':
                if not flag2:
                    flag2 = True
                elif flag2 and not line.strip():
                    flag = 'none'
                    flag2 = False
                elif flag2:
                    vals = line.split()
                    if ff_style['improper'] == 'cvff':
                        k_imp.append(float(vals[1]))
                        d0_imp.append(float(vals[2]))
                        n_imp.append(int(vals[3]))
                    elif ff_style['improper'] == 'umbrella':
                        k_imp.append(float(vals[1]))
                        d0_imp.append(float(vals[2]))
                    else:
                        utils.radon_print('improper_style %s is not available.' % ff_style['improper'], level=3)

            elif flag == 'atom':
                if not flag2:
                    mol_id = [0 for x in range(n_data['atoms'])]
                    atom_id = [0 for x in range(n_data['atoms'])]
                    charge = [0.0 for x in range(n_data['atoms'])]
                    coord = [[0.0, 0.0, 0.0] for x in range(n_data['atoms'])]
                    pbc = [[0, 0, 0] for x in range(n_data['atoms'])]
                    flag2 = True
                elif flag2 and not line.strip():
                    flag = 'none'
                    flag2 = False
                elif flag2:
                    vals = line.split()
                    i = int(vals[0])
                    mol_id[i-1] = int(vals[1])
                    atom_id[i-1] = int(vals[2])
                    charge[i-1] = float(vals[3])
                    coord[i-1] = [float(vals[4]), float(vals[5]), float(vals[6])]
                    if len(vals) == 10:
                        pbc[i-1] = [int(vals[7]), int(vals[8]), int(vals[9])]

            elif flag == 'velocity':
                if not flag2:
                    velocity = [[0.0, 0.0, 0.0] for x in range(n_data['atoms'])]
                    flag2 = True
                elif flag2 and not line.strip():
                    flag = 'none'
                    flag2 = False
                elif flag2:
                    vals = line.split()
                    i = int(vals[0])
                    velocity[i-1] = [float(vals[1]), float(vals[2]), float(vals[3])]

            elif flag == 'bond':
                if not flag2:
                    bond_id = [0 for x in range(n_data['bonds'])]
                    bond_atom = [[0, 0] for x in range(n_data['bonds'])]
                    flag2 = True
                elif flag2 and not line.strip():
                    flag = 'none'
                    flag2 = False
                elif flag2:
                    vals = line.split()
                    i = int(vals[0])
                    bond_id[i-1] = int(vals[1])
                    bond_atom[i-1] = [int(vals[2]), int(vals[3])]

            elif flag == 'angle':
                if not flag2:
                    angle_id = [0 for x in range(n_data['angles'])]
                    angle_atom = [[0, 0, 0] for x in range(n_data['angles'])]
                    flag2 = True
                elif flag2 and not line.strip():
                    flag = 'none'
                    flag2 = False
                elif flag2:
                    vals = line.split()
                    i = int(vals[0])
                    angle_id[i-1] = int(vals[1])
                    angle_atom[i-1] = [int(vals[2]), int(vals[3]), int(vals[4])]

            elif flag == 'dihed':
                if not flag2:
                    dihed_id = [0 for x in range(n_data['dihedrals'])]
                    dihed_atom = [[0, 0, 0, 0] for x in range(n_data['dihedrals'])]
                    flag2 = True
                elif flag2 and not line.strip():
                    flag = 'none'
                    flag2 = False
                elif flag2:
                    vals = line.split()
                    i = int(vals[0])
                    dihed_id[i-1] = int(vals[1])
                    dihed_atom[i-1] = [int(vals[2]), int(vals[3]), int(vals[4]), int(vals[5])]

            elif flag == 'impro':
                if not flag2:
                    impro_id = [0 for x in range(n_data['impropers'])]
                    impro_atom = [[0, 0, 0, 0] for x in range(n_data['impropers'])]
                    flag2 = True
                elif flag2 and not line.strip():
                    flag = 'none'
                    flag2 = False
                elif flag2:
                    vals = line.split()
                    i = int(vals[0])
                    impro_id[i-1] = int(vals[1])
                    impro_atom[i-1] = [int(vals[2]), int(vals[3]), int(vals[4]), int(vals[5])]

    rwmol = Chem.RWMol()
    w2ele = {1:'H', 2:'H', 3:'H', 12:'C', 14:'N', 16:'O', 19:'F', 28:'Si', 31:'P', 32:'S', 35:'Cl', 80:'Br', 127:'I'}

    for i in range(n_data['atoms']):
        atom_tmp = Chem.Atom(w2ele[round(mass[atom_id[i]-1])])
        atom_tmp.SetProp('ff_type', str(atom_id[i]))
        atom_tmp.SetIntProp('ff_type_num', int(atom_id[i]))

        if ff_style['pair'] == 'lj':
            atom_tmp.SetDoubleProp('ff_epsilon', float(epsilon[atom_id[i]-1]))
            atom_tmp.SetDoubleProp('ff_sigma', float(sigma[atom_id[i]-1]))
        else:
            utils.radon_print('pair_style %s is not available.' % ff_style['pair'], level=3)

        atom_tmp.SetDoubleProp('AtomicCharge', float(charge[i]))
        atom_tmp.SetIntProp('mol_id', int(mol_id[i]))
        if int(mass[atom_id[i]-1]) == 2: atom_tmp.SetIsotope(2)
        elif int(mass[atom_id[i]-1]) == 3: atom_tmp.SetIsotope(3)
        atom_tmp.SetDoubleProp('vx', float(velocity[i][0]))
        atom_tmp.SetDoubleProp('vy', float(velocity[i][1]))
        atom_tmp.SetDoubleProp('vz', float(velocity[i][2]))
        rwmol.AddAtom(atom_tmp)

    for i in range(n_data['bonds']):
        b_order = Chem.rdchem.BondType.UNSPECIFIED
        a_idx = bond_atom[i][0] - 1
        b_idx = bond_atom[i][1] - 1
        a_atom = rwmol.GetAtomWithIdx(a_idx)
        b_atom = rwmol.GetAtomWithIdx(b_idx)

        if bond_order:
            if a_atom.GetSymbol() in ['H', 'F', 'Cl', 'Br', 'I'] or b_atom.GetSymbol() in ['H', 'F', 'Cl', 'Br', 'I']:
                b_order = Chem.rdchem.BondType.SINGLE

            elif a_atom.GetSymbol() == 'C' and b_atom.GetSymbol() == 'C':
                if r0[bond_id[i]-1] >= 1.41:  # sp3-sp3, sp3-sp2, sp3-sp1
                    b_order = Chem.rdchem.BondType.SINGLE
                elif 1.39 <= r0[bond_id[i]-1] < 1.41:  # aromatic ca-ca, ca-cp, ca-cq
                    b_order = Chem.rdchem.BondType.AROMATIC
                elif 1.33 <= r0[bond_id[i]-1] < 1.39:
                    b_order = Chem.rdchem.BondType.DOUBLE
                elif 1.22 <= r0[bond_id[i]-1] < 1.33:  # sp2-sp1
                    b_order = Chem.rdchem.BondType.SINGLE
                elif r0[bond_id[i]-1] < 1.22:
                    b_order = Chem.rdchem.BondType.TRIPLE

            elif (a_atom.GetSymbol() == 'C' and b_atom.GetSymbol() == 'N') or (a_atom.GetSymbol() == 'N' and b_atom.GetSymbol() == 'C'):
                if r0[bond_id[i]-1] >= 1.34:  # sp3-sp3, sp3-sp2, sp3-sp1
                    b_order = Chem.rdchem.BondType.SINGLE
                elif 1.339 <= r0[bond_id[i]-1] < 1.340:  # aromatic ca-nb
                    b_order = Chem.rdchem.BondType.AROMATIC
                elif 1.29 <= r0[bond_id[i]-1] < 1.339:
                    b_order = Chem.rdchem.BondType.SINGLE
                elif 1.16 <= r0[bond_id[i]-1] < 1.29:
                    b_order = Chem.rdchem.BondType.DOUBLE
                elif r0[bond_id[i]-1] < 1.16:
                    b_order = Chem.rdchem.BondType.TRIPLE

            elif (a_atom.GetSymbol() == 'C' and b_atom.GetSymbol() == 'O') or (a_atom.GetSymbol() == 'O' and b_atom.GetSymbol() == 'C'):
                if r0[bond_id[i]-1] >= 1.22: #1.31:  # sp3-sp3, sp3-sp2, sp3-sp1
                    b_order = Chem.rdchem.BondType.SINGLE
                else:
                    b_order = Chem.rdchem.BondType.DOUBLE

            elif (a_atom.GetSymbol() == 'N' and b_atom.GetSymbol() == 'O') or (a_atom.GetSymbol() == 'O' and b_atom.GetSymbol() == 'N'):
                if r0[bond_id[i]-1] >= 1.25: # sp3-sp3, sp3-sp2, sp3-sp1
                    b_order = Chem.rdchem.BondType.SINGLE
                else:
                    b_order = Chem.rdchem.BondType.DOUBLE

            elif a_atom.GetSymbol() == 'N' and b_atom.GetSymbol() == 'N':
                if r0[bond_id[i]-1] >= 1.30: # sp3-sp3, sp3-sp2, sp3-sp1
                    b_order = Chem.rdchem.BondType.SINGLE
                elif 1.16 > r0[bond_id[i]-1] > 1.30:
                    b_order = Chem.rdchem.BondType.DOUBLE
                elif r0[bond_id[i]-1] > 1.16:
                    b_order = Chem.rdchem.BondType.TRIPLE

            elif a_atom.GetSymbol() == 'O' and b_atom.GetSymbol() == 'O':
                if r0[bond_id[i]-1] >= 1.44: # sp3-sp3, sp3-sp2, sp3-sp1
                    b_order = Chem.rdchem.BondType.SINGLE
                else:
                    b_order = Chem.rdchem.BondType.DOUBLE

            elif (a_atom.GetSymbol() == 'C' and b_atom.GetSymbol() == 'P') or (a_atom.GetSymbol() == 'P' and b_atom.GetSymbol() == 'C'):
                if r0[bond_id[i]-1] >= 1.70: # sp3-sp3, sp3-sp2, sp3-sp1
                    b_order = Chem.rdchem.BondType.SINGLE
                else:
                    b_order = Chem.rdchem.BondType.DOUBLE

            elif (a_atom.GetSymbol() == 'N' and b_atom.GetSymbol() == 'P') or (a_atom.GetSymbol() == 'P' and b_atom.GetSymbol() == 'N'):
                if r0[bond_id[i]-1] >= 1.65: # sp3-sp3, sp3-sp2, sp3-sp1
                    b_order = Chem.rdchem.BondType.SINGLE
                else:
                    b_order = Chem.rdchem.BondType.DOUBLE

            elif (a_atom.GetSymbol() == 'O' and b_atom.GetSymbol() == 'P') or (a_atom.GetSymbol() == 'P' and b_atom.GetSymbol() == 'O'):
                if r0[bond_id[i]-1] >= 1.53: # sp3-sp3, sp3-sp2, sp3-sp1
                    b_order = Chem.rdchem.BondType.SINGLE
                else:
                    b_order = Chem.rdchem.BondType.DOUBLE

            elif a_atom.GetSymbol() == 'P' and b_atom.GetSymbol() == 'P':
                if r0[bond_id[i]-1] >= 1.80: # sp3-sp3, sp3-sp2, sp3-sp1
                    b_order = Chem.rdchem.BondType.SINGLE
                else:
                    b_order = Chem.rdchem.BondType.DOUBLE

            elif (a_atom.GetSymbol() == 'C' and b_atom.GetSymbol() == 'S') or (a_atom.GetSymbol() == 'S' and b_atom.GetSymbol() == 'C'):
                if r0[bond_id[i]-1] >= 1.64: # sp3-sp3, sp3-sp2, sp3-sp1
                    b_order = Chem.rdchem.BondType.SINGLE
                else:
                    b_order = Chem.rdchem.BondType.DOUBLE

            elif (a_atom.GetSymbol() == 'N' and b_atom.GetSymbol() == 'S') or (a_atom.GetSymbol() == 'S' and b_atom.GetSymbol() == 'N'):
                if r0[bond_id[i]-1] >= 1.58: # sp3-sp3, sp3-sp2, sp3-sp1
                    b_order = Chem.rdchem.BondType.SINGLE
                else:
                    b_order = Chem.rdchem.BondType.DOUBLE

            elif (a_atom.GetSymbol() == 'O' and b_atom.GetSymbol() == 'S') or (a_atom.GetSymbol() == 'S' and b_atom.GetSymbol() == 'O'):
                if r0[bond_id[i]-1] >= 1.56: # sp3-sp3, sp3-sp2, sp3-sp1
                    b_order = Chem.rdchem.BondType.SINGLE
                else:
                    b_order = Chem.rdchem.BondType.DOUBLE

            elif a_atom.GetSymbol() == 'S' and b_atom.GetSymbol() == 'S':
                b_order = Chem.rdchem.BondType.SINGLE

            else:
                b_order = Chem.rdchem.BondType.SINGLE

        bond_idx = rwmol.AddBond(a_idx, b_idx, order=b_order)
        rwmol.GetBondWithIdx(bond_idx-1).SetProp('ff_type', str(bond_id[i]))
        rwmol.GetBondWithIdx(bond_idx-1).SetIntProp('ff_type_num', int(bond_id[i]))
        if ff_style['bond'] == 'harmonic':
            rwmol.GetBondWithIdx(bond_idx-1).SetDoubleProp('ff_k', float(k_bond[bond_id[i]-1]))
            rwmol.GetBondWithIdx(bond_idx-1).SetDoubleProp('ff_r0', float(r0[bond_id[i]-1]))
        else:
            utils.radon_print('bond_style %s is not available.' % ff_style['bond'], level=3)

    if bond_order: Chem.SanitizeMol(rwmol)
    mol = rwmol.GetMol()

    if 'angles' in n_data.keys():
        for i in range(n_data['angles']):
            if ff_style['angle'] == 'harmonic':
                angle_ff_tmp = ff_class.Angle_harmonic(ff_type=angle_id[i], k=k_angle[angle_id[i]-1], theta0=theta0[angle_id[i]-1])
            else:
                utils.radon_print('angle_style %s is not available.' % ff_style['angle'], level=3)
            utils.add_angle(mol, angle_atom[i][0]-1, angle_atom[i][1]-1, angle_atom[i][2]-1, ff=angle_ff_tmp)
    else:
        setattr(mol, 'angles', [])

    if 'dihedrals' in n_data.keys():
        for i in range(n_data['dihedrals']):
            if ff_style['dihedral'] == 'fourier':
                dihed_ff_tmp = ff_class.Dihedral_fourier(ff_type=dihed_id[i], k=k_dih[dihed_id[i]-1], d0=d0_dih[dihed_id[i]-1],
                                       m=m_dih[dihed_id[i]-1], n=n_dih[dihed_id[i]-1])
            elif ff_style['dihedral'] == 'harmonic':
                dihed_ff_tmp = ff_class.Dihedral_harmonic(ff_type=dihed_id[i], k=k_dih[dihed_id[i]-1], d0=d0_dih[dihed_id[i]-1],
                                       n=n_dih[dihed_id[i]-1])
            else:
                utils.radon_print('dihedral_style %s is not available.' % ff_style['dihedral'], level=3)
            utils.add_dihedral(mol, dihed_atom[i][0]-1, dihed_atom[i][1]-1, dihed_atom[i][2]-1, dihed_atom[i][3]-1, ff=dihed_ff_tmp)
    else:
        setattr(mol, 'dihedrals', [])

    if 'impropers' in n_data.keys():
        for i in range(n_data['impropers']):
            if ff_style['improper'] == 'cvff':
                impro_ff_tmp = ff_class.Improper_cvff(ff_type=impro_id[i], k=k_imp[impro_id[i]-1], d0=d0_imp[impro_id[i]-1], n=n_imp[impro_id[i]-1])
            elif ff_style['improper'] == 'umbrella':
                impro_ff_tmp = ff_class.Improper_umbrella(ff_type=impro_id[i], k=k_imp[impro_id[i]-1], x0=x0_imp[impro_id[i]-1])
            else:
                utils.radon_print('improper_style %s is not available.' % ff_style['improper'], level=3)                
            utils.add_improper(mol, impro_atom[i][0]-1, impro_atom[i][1]-1, impro_atom[i][2]-1, impro_atom[i][3]-1, ff=impro_ff_tmp)
    else:
        setattr(mol, 'impropers', [])

    mol.SetProp('pair_style', ff_style['pair']) 
    mol.SetProp('bond_style', ff_style['bond'])
    mol.SetProp('angle_style', ff_style['angle'])
    mol.SetProp('dihedral_style', ff_style['dihedral'])
    mol.SetProp('improper_style', ff_style['improper'])

    setattr(mol, 'cell', utils.Cell(cell_data['xhi'], cell_data['xlo'], cell_data['yhi'], cell_data['ylo'], cell_data['zhi'], cell_data['zlo']))

    conf = Chem.rdchem.Conformer(n_data['atoms'])
    conf.Set3D(True)
    for i in range(n_data['atoms']):
        x = coord[i][0] + mol.cell.dx * pbc[i][0]
        y = coord[i][1] + mol.cell.dy * pbc[i][1]
        z = coord[i][2] + mol.cell.dz * pbc[i][2]
        conf.SetAtomPosition(i, Geom.Point3D(x, y, z))
    conf_id = mol.AddConformer(conf, assignId=True)
    mol = calc.mol_trans_in_cell(mol, confId=conf_id)

    return mol

