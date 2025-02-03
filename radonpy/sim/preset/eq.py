#  Copyright (c) 2023. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# ******************************************************************************
# sim.preset.eq module
# ******************************************************************************

import os
import glob
import re
import datetime
import numpy as np
from ...core import calc, const, utils
from .. import lammps, preset
from ..md import MD

__version__ = '0.2.10'


class Equilibration(preset.Preset):
    def __init__(self, mol, prefix='', work_dir=None, save_dir=None, solver_path=None, **kwargs):
        """
        preset.eq.Equilibration

        Base class of equilibration preset

        Args:
            mol: RDKit Mol object
        """
        super().__init__(mol, prefix=prefix, work_dir=work_dir, save_dir=save_dir, solver_path=solver_path, **kwargs)

        self.in_file1 = kwargs.get('in_file1', '%seq1.in' % self.prefix)
        self.in_file2 = kwargs.get('in_file2', '%seq2.in' % self.prefix)
        self.in_file = kwargs.get('in_file', '%seq3.in' % self.prefix)
        self.dat_file1 = kwargs.get('dat_file1', '%seq1.data' % self.prefix)
        self.dat_file2 = kwargs.get('dat_file2', '%seq2.data' % self.prefix)
        self.dat_file = kwargs.get('dat_file', '%seq3.data' % self.prefix)
        self.pdb_file = kwargs.get('pdb_file', '%seq1.pdb' % self.prefix)
        self.log_file1 = kwargs.get('log_file1', '%seq1.log' % self.prefix)
        self.log_file2 = kwargs.get('log_file2', '%seq2.log' % self.prefix)
        self.log_file = kwargs.get('log_file', '%seq3.log' % self.prefix)
        self.dump_file1 = kwargs.get('dump_file1', '%seq1.dump' % self.prefix)
        self.dump_file2 = kwargs.get('dump_file2', '%seq2.dump' % self.prefix)
        self.dump_file = kwargs.get('dump_file', '%seq3.dump' % self.prefix)
        self.xtc_file1 = kwargs.get('xtc_file1', '%seq1.xtc' % self.prefix)
        self.xtc_file2 = kwargs.get('xtc_file2', '%seq2.xtc' % self.prefix)
        self.xtc_file = kwargs.get('xtc_file', '%seq3.xtc' % self.prefix)
        self.rg_file = kwargs.get('rg_file', '%srg3.profile' % self.prefix)
        self.last_str1 = kwargs.get('last_str1', '%seq1_last.dump' % self.prefix)
        self.last_str2 = kwargs.get('last_str2', '%seq2_last.dump' % self.prefix)
        self.last_str = kwargs.get('last_str', '%seq3_last.dump' % self.prefix)
        self.last_data1 = self.dat_file2
        self.last_data2 = self.dat_file
        self.last_data = kwargs.get('last_data', '%seq3_last.data' % self.prefix)
        self.pickle_file1 = kwargs.get('pickle_file1', '%seq1_last.pickle' % self.prefix)
        self.pickle_file2 = kwargs.get('pickle_file2', '%seq2_last.pickle' % self.prefix)
        self.pickle_file = kwargs.get('pickle_file', '%seq3_last.pickle' % self.prefix)
        self.json_file1 = kwargs.get('json_file1', '%seq1_last.json' % self.prefix)
        self.json_file2 = kwargs.get('json_file2', '%seq2_last.json' % self.prefix)
        self.json_file = kwargs.get('json_file', '%seq3_last.json' % self.prefix)


    def packing(self, f_density=0.8, max_temp=700, comm_cutoff=8.0, **kwargs):

        mass = calc.mol_mass(self.mol)
        f_length = np.cbrt( (mass / const.NA) / (f_density / const.cm2ang**3)) / 2

        # Initial relaxation and packing
        md = MD()
        md.pair_style = 'lj/cut'
        md.cutoff_in = 3.0
        md.cutoff_out = ''
        md.kspace_style = 'none'
        md.kspace_style_accuracy = ''
        md.bond_style = self.bond_style
        md.angle_style = self.angle_style
        md.dihedral_style = self.dihedral_style
        md.improper_style = self.improper_style
        md.log_file = kwargs.get('log_file', self.log_file1)
        md.dat_file = kwargs.get('dat_file', self.dat_file1)
        md.dump_file = kwargs.get('dump_file', self.dump_file1)
        md.xtc_file = kwargs.get('xtc_file', self.xtc_file1)
        md.rst = True
        md.outstr = kwargs.get('last_str', self.last_str1)
        md.write_data = kwargs.get('last_data', self.last_data1)
        md.add.append('comm_modify cutoff %f' % comm_cutoff)

        md.add_min(min_style='cg')
        md.add_md('nvt', 20000, time_step=0.1, shake=False, t_start=300.0, t_stop=300.0, **kwargs)
        md.add_md('nvt', 1000000, time_step=1.0, shake=True, t_start=300.0, t_stop=max_temp, **kwargs)
        md.add_md('nvt', 1000000, time_step=1.0, shake=True, t_start=max_temp, t_stop=max_temp, **kwargs)
        md.wf[-1].add_deform(dftype='final', deform_fin_lo=-f_length, deform_fin_hi=f_length, axis='xyz')

        return md


    def annealing(self, max_temp=700.0, temp=300.0, press=1.0, step=5000000, set_init_velocity=False, **kwargs):

        p_dump = 1000
        md = MD()
        md.pair_style = self.pair_style
        md.cutoff_in = self.cutoff_in
        md.cutoff_out = self.cutoff_out
        md.kspace_style = self.kspace_style
        md.kspace_style_accuracy = self.kspace_style_accuracy
        md.bond_style = self.bond_style
        md.angle_style = self.angle_style
        md.dihedral_style = self.dihedral_style
        md.improper_style = self.improper_style
        md.neighbor = '%s bin' % self.neighbor_dis
        md.log_file = kwargs.get('log_file', self.log_file2)
        md.dat_file = kwargs.get('dat_file', self.dat_file2)
        md.dump_file = kwargs.get('dump_file', self.dump_file2)
        md.xtc_file = kwargs.get('xtc_file', self.xtc_file2)
        md.rst = True
        md.outstr = kwargs.get('last_str', self.last_str2)
        md.write_data = kwargs.get('last_data', self.last_data2)
        if set_init_velocity:
            md.set_init_velocity = max_temp
        #if polarizable:
        #    md.add_drude()

        md.add_md('nvt', 20000, time_step=0.2, shake=False, t_start=max_temp, t_stop=max_temp, **kwargs)
        md.add_md('nvt', 100000, time_step=1.0, shake=True, t_start=max_temp, t_stop=max_temp, **kwargs)
        md.add_md('npt', 20000, time_step=1.0, shake=True, t_start=max_temp, t_stop=max_temp, p_start=press, p_stop=press, p_dump=p_dump, **kwargs)

        a_temp = np.linspace(max_temp, temp, int(step/100000)+1)
        for i in range(len(a_temp)-1):
            md.add_md('npt', 100000, time_step=1.0, shake=True, t_start=a_temp[i], t_stop=a_temp[i+1],
                      p_start=press, p_stop=press, p_dump=p_dump, **kwargs)

        return md


    def eq21step(self, temp=300, max_temp=600, press=1.0, max_press=50000, time_step=1.0,
                 step_list=None, press_ratio=None, set_init_velocity=False, **kwargs):

        if step_list is None:
            step_list = [
                [50000, 50000,  50000],
                [50000, 100000, 50000],
                [50000, 100000, 50000],
                [50000, 100000, 5000],
                [5000,  10000,  5000],
                [5000,  10000,  5000],
                [5000,  10000,  800000]
            ]

        if press_ratio is None:
            press_ratio = [0.02, 0.60, 1.00, 0.50, 0.10, 0.01]

        p_dump = 1000
        md = MD()
        md.pair_style = self.pair_style
        md.cutoff_in = self.cutoff_in
        md.cutoff_out = self.cutoff_out
        md.kspace_style = self.kspace_style
        md.kspace_style_accuracy = self.kspace_style_accuracy
        md.bond_style = self.bond_style
        md.angle_style = self.angle_style
        md.dihedral_style = self.dihedral_style
        md.improper_style = self.improper_style
        md.neighbor = '%s bin' % self.neighbor_dis
        md.log_file = kwargs.get('log_file', self.log_file2)
        md.dat_file = kwargs.get('dat_file', self.dat_file2)
        md.dump_file = kwargs.get('dump_file', self.dump_file2)
        md.xtc_file = kwargs.get('xtc_file', self.xtc_file2)
        md.rst = True
        md.outstr = kwargs.get('last_str', self.last_str2)
        md.write_data = kwargs.get('last_data', self.last_data2)
        if set_init_velocity:
            md.set_init_velocity = max_temp
        #if polarizable:
        #    md.add_drude()

        # 21 step compression/decompression equilibration protocol
        press_list = np.append(np.array(press_ratio) * max_press, press)
        for s, p in zip(step_list, press_list):
            md.add_md('nvt', int(s[0]), time_step=time_step, shake=True, t_start=max_temp, t_stop=max_temp,
                      add=['neigh_modify delay 0 every 1 check yes'], **kwargs)
            md.add_md('nvt', int(s[1]), time_step=time_step, shake=True, t_start=temp, t_stop=temp, **kwargs)
            md.add_md('npt', int(s[2]), time_step=time_step, shake=True, t_start=temp, t_stop=temp,
                      p_start=p, p_stop=p, p_dump=p_dump, add=['neigh_modify delay 0 every 1 check no'], **kwargs)

        return md


    def sampling(self, temp=300.0, press=1.0, step=5000000, **kwargs):

        p_dump = 1000
        md = MD()
        md.pair_style = self.pair_style
        md.cutoff_in = self.cutoff_in
        md.cutoff_out = self.cutoff_out
        md.kspace_style = self.kspace_style
        md.kspace_style_accuracy = self.kspace_style_accuracy
        md.bond_style = self.bond_style
        md.angle_style = self.angle_style
        md.dihedral_style = self.dihedral_style
        md.improper_style = self.improper_style
        md.neighbor = '%s bin' % self.neighbor_dis
        md.log_file = kwargs.get('log_file', self.log_file)
        md.dat_file = kwargs.get('dat_file', self.dat_file)
        md.dump_file = kwargs.get('dump_file', self.dump_file)
        md.xtc_file = kwargs.get('xtc_file', self.xtc_file)
        md.rst = True
        md.outstr = kwargs.get('last_str', self.last_str)
        md.write_data = kwargs.get('last_data', self.last_data)
        if kwargs.get('set_init_velocity', False):
            md.set_init_velocity = temp
        #if polarizable:
        #    md.add_drude()

        md.add_md('npt', step, time_step=1.0, shake=True, t_start=temp, t_stop=temp,
                   p_start=press, p_stop=press, p_dump=p_dump, **kwargs)
        md.wf[-1].add_rg(file=self.rg_file)
        md.wf[-1].add_msd()

        return md


    def analyze(self, ignore_log=[], **kwargs):

        analy = Equilibration_analyze(
            log_file  = os.path.join(self.work_dir, self.log_file),
            traj_file = os.path.join(self.work_dir, self.xtc_file),
            pdb_file  = os.path.join(self.work_dir, self.pdb_file),
            dat_file  = os.path.join(self.work_dir, self.dat_file),
            rg_file   = os.path.join(self.work_dir, self.rg_file),
            ignore_log = ignore_log,
            **kwargs
        )

        return analy


class Equilibration_analyze(lammps.Analyze):
    def __init__(self, log_file='eq3.log', ignore_log=[], **kwargs):
        super().__init__(log_file=log_file, ignore_log=ignore_log, **kwargs)
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




class Annealing(Equilibration):
    def exec(self, confId=0, f_density=0.8,
             max_temp=700.0, temp=300.0, press=1.0, ann_step=5, eq_step=8,
             omp=1, mpi=1, gpu=0, intel='auto', opt='auto', **kwargs):
        """
        preset.eq.Annealing.exec

        Execution of equilibration by annealing

        Args:
            mol: RDKit Mol object
        
        Optional args:
            confId: Target conformer ID (int)
            max_temp: Maximum temperature in annealing (float, K)
            temp: Finel temperature in annealing (float, K)
            press: Pressure (float, g/cm**3)
            polarizable: Use polarizable Drude model (boolean) not implemented
            solver: lammps (str) 
            omp: Number of threads of OpenMP (int)
            mpi: Number of MPI process (int)
            gpu: Number of GPU (int)

        Returns:
            RDKit Mol object
        """

        utils.MolToPDBFile(self.mol, os.path.join(self.work_dir, self.pdb_file))
        lmp = lammps.LAMMPS(work_dir=self.work_dir, solver_path=self.solver_path)
        lmp.make_dat(self.mol, file_name=self.dat_file1, confId=confId)

        dt1 = datetime.datetime.now()
        utils.radon_print('Packing simulation (eq1) by LAMMPS is running...', level=1)
        md1 = self.packing(f_density=f_density, max_temp=max_temp, comm_cutoff=kwargs.get('comm_cutoff', 8.0), **kwargs)
        self.mol = lmp.run(md1, mol=self.mol, confId=confId, input_file=self.in_file1, last_str=self.last_str1, last_data=self.last_data1,
                           omp=omp, mpi=mpi, gpu=gpu, intel=intel, opt=opt)
        utils.MolToJSON(self.mol, os.path.join(self.save_dir, self.json_file1))
        utils.pickle_dump(self.mol, os.path.join(self.save_dir, self.pickle_file1))
        dt2 = datetime.datetime.now()
        utils.radon_print('Complete packing simulation (eq1). Elapsed time = %s' % str(dt2-dt1), level=1)

        dt1 = datetime.datetime.now()
        utils.radon_print('Annealing simulation (eq2) by LAMMPS is running...', level=1)
        md2 = self.annealing(max_temp=max_temp, temp=temp, press=press, step=int(1000000*ann_step), set_init_velocity=True, **kwargs)
        self.mol = lmp.run(md2, mol=self.mol, confId=confId, input_file=self.in_file2, last_str=self.last_str2, last_data=self.last_data2,
                           omp=omp, mpi=mpi, gpu=gpu, intel=intel, opt=opt)
        utils.MolToJSON(self.mol, os.path.join(self.save_dir, self.json_file2))
        utils.pickle_dump(self.mol, os.path.join(self.save_dir, self.pickle_file2))
        dt2 = datetime.datetime.now()
        utils.radon_print('Complete annealing simulation (eq2). Elapsed time = %s' % str(dt2-dt1), level=1)

        dt1 = datetime.datetime.now()
        utils.radon_print('Sampling simulation (eq3) by LAMMPS is running...', level=1)
        md3 = self.sampling(temp=temp, press=press, step=int(1000000*eq_step), **kwargs)
        self.mol = lmp.run(md3, mol=self.mol, confId=confId, input_file=self.in_file, last_str=self.last_str, last_data=self.last_data,
                           omp=omp, mpi=mpi, gpu=gpu, intel=intel, opt=opt)
        utils.MolToJSON(self.mol, os.path.join(self.save_dir, self.json_file))
        utils.pickle_dump(self.mol, os.path.join(self.save_dir, self.pickle_file))
        dt2 = datetime.datetime.now()
        utils.radon_print('Complete sampling simulation (eq3). Elapsed time = %s' % str(dt2-dt1), level=1)

        return self.mol


    def make_lammps_input(self, confId=0, f_density=0.8, max_temp=700.0, temp=300.0, press=1.0, ann_step=5, eq_step=8, **kwargs):

        utils.MolToPDBFile(self.mol, os.path.join(self.work_dir, self.pdb_file))
        lmp = lammps.LAMMPS(work_dir=self.work_dir, solver_path=self.solver_path)
        lmp.make_dat(self.mol, file_name=self.dat_file1, confId=confId)

        md1 = self.packing(f_density=f_density, max_temp=max_temp, comm_cutoff=kwargs.get('comm_cutoff', 8.0), **kwargs)
        lmp.make_input(md1, file_name=self.in_file1)

        md2 = self.annealing(max_temp=max_temp, temp=temp, press=press, step=int(1000000*ann_step), set_init_velocity=True, **kwargs)
        lmp.make_input(md2, file_name=self.in_file2)

        md3 = self.sampling(temp=temp, press=press, step=int(1000000*eq_step), **kwargs)
        lmp.make_input(md3, file_name=self.in_file)

        return True




class EQ21step(Equilibration):
    def exec(self, confId=0, f_density=0.8,
             max_temp=600.0, temp=300.0, press=1.0, max_press=50000, step_list=None, press_ratio=None,
             time_step=1.0, eq_step=5, omp=1, mpi=1, gpu=0, intel='auto', opt='auto', **kwargs):
        """
        preset.eq.EQ21step.exec

        Execution of Larsen's 21 step compression/decompression equilibration protocol
        
        Optional args:
            confId: Target conformer ID (int)
            max_temp: Maximum temperature in the protocole (float, K)
            temp: Finel temperature in the protocole (float, K)
            max_press: Maximum pressure in the protocole (float)
            press: Finel pressure in the protocole (float)
            solver: lammps (str) 
            omp: Number of threads of OpenMP (int)
            mpi: Number of MPI process (int)
            gpu: Number of GPU (int)

        Returns:
            RDKit Mol object
        """

        utils.MolToPDBFile(self.mol, os.path.join(self.work_dir, self.pdb_file))
        lmp = lammps.LAMMPS(work_dir=self.work_dir, solver_path=self.solver_path)
        lmp.make_dat(self.mol, file_name=self.dat_file1, confId=confId)

        dt1 = datetime.datetime.now()
        utils.radon_print('Packing simulation (eq1) by LAMMPS is running...', level=1)
        md1 = self.packing(f_density=f_density, comm_cutoff=kwargs.get('comm_cutoff', 8.0), **kwargs)
        self.mol = lmp.run(md1, mol=self.mol, confId=confId, input_file=self.in_file1, last_str=self.last_str1, last_data=self.last_data1,
                           omp=omp, mpi=mpi, gpu=gpu, intel=intel, opt=opt)
        utils.MolToJSON(self.mol, os.path.join(self.save_dir, self.json_file1))
        utils.pickle_dump(self.mol, os.path.join(self.save_dir, self.pickle_file1))
        dt2 = datetime.datetime.now()
        utils.radon_print('Complete packing simulation (eq1). Elapsed time = %s' % str(dt2-dt1), level=1)

        dt1 = datetime.datetime.now()
        utils.radon_print('Larsen\'s 21 step compression/decompression equilibration (eq2) by LAMMPS is running...', level=1)
        md2 = self.eq21step(max_temp=max_temp, temp=temp, press=press, max_press=max_press,
                            step_list=step_list, press_ratio=press_ratio, time_step=time_step, set_init_velocity=True, **kwargs)
        self.mol = lmp.run(md2, mol=self.mol, confId=confId, input_file=self.in_file2, last_str=self.last_str2, last_data=self.last_data2,
                           omp=omp, mpi=mpi, gpu=gpu, intel=intel, opt=opt)
        utils.MolToJSON(self.mol, os.path.join(self.save_dir, self.json_file2))
        utils.pickle_dump(self.mol, os.path.join(self.save_dir, self.pickle_file2))
        dt2 = datetime.datetime.now()
        utils.radon_print('Complete Larsen 21 step compression/decompression equilibration (eq2). Elapsed time = %s' % str(dt2-dt1), level=1)

        dt1 = datetime.datetime.now()
        utils.radon_print('Sampling simulation (eq3) by LAMMPS is running...', level=1)
        md3 = self.sampling(temp=temp, press=press, step=int(1000000*eq_step), **kwargs)
        self.mol = lmp.run(md3, mol=self.mol, confId=confId, input_file=self.in_file, last_str=self.last_str, last_data=self.last_data,
                           omp=omp, mpi=mpi, gpu=gpu, intel=intel, opt=opt)
        utils.MolToJSON(self.mol, os.path.join(self.save_dir, self.json_file))
        utils.pickle_dump(self.mol, os.path.join(self.save_dir, self.pickle_file))
        dt2 = datetime.datetime.now()
        utils.radon_print('Complete sampling simulation (eq3). Elapsed time = %s' % str(dt2-dt1), level=1)

        return self.mol


    def make_lammps_input(self, confId=0, f_density=0.8, max_temp=600.0, temp=300.0, press=1.0, max_press=50000,
                          step_list=None, press_ratio=None, time_step=1.0, eq_step=5, **kwargs):

        utils.MolToPDBFile(self.mol, os.path.join(self.work_dir, self.pdb_file))
        lmp = lammps.LAMMPS(work_dir=self.work_dir, solver_path=self.solver_path)
        lmp.make_dat(self.mol, file_name=self.dat_file1, confId=confId)

        md1 = self.packing(f_density=f_density, comm_cutoff=kwargs.get('comm_cutoff', 8.0), **kwargs)
        lmp.make_input(md1, file_name=self.in_file1)

        md2 = self.eq21step(max_temp=max_temp, temp=temp, press=press, max_press=max_press,
                            step_list=step_list, press_ratio=press_ratio, time_step=time_step, set_init_velocity=True, **kwargs)
        lmp.make_input(md2, file_name=self.in_file2)

        md3 = self.sampling(temp=temp, press=press, step=int(1000000*eq_step), **kwargs)
        lmp.make_input(md3, file_name=self.in_file)

        return True




class Additional(Equilibration):
    def __init__(self, mol, prefix='', work_dir=None, solver_path=None, idx=0, **kwargs):
        """
        preset.eq.Additional

        Preset of simulated annealing for equilibration 

        Args:
            mol: RDKit Mol object
        """
        super().__init__(mol, prefix=prefix, work_dir=work_dir, solver_path=solver_path, **kwargs)

        self.idx = get_final_idx(self.work_dir) + 1 if idx == 0 else idx

        self.in_file = kwargs.get('in_file', '%seq%i.in' % (self.prefix, self.idx))
        self.dat_file = kwargs.get('dat_file', '%seq%i.data' % (self.prefix, self.idx))
        self.pdb_file = kwargs.get('pdb_file', '%seq%i.pdb' % (self.prefix, self.idx))
        self.log_file = kwargs.get('log_file', '%seq%i.log' % (self.prefix, self.idx))
        self.dump_file = kwargs.get('dump_file', '%seq%i.dump' % (self.prefix, self.idx))
        self.xtc_file = kwargs.get('xtc_file', '%seq%i.xtc' % (self.prefix, self.idx))
        self.rg_file = kwargs.get('rg_file', '%srg%i.profile' % (self.prefix, self.idx))
        self.last_str = kwargs.get('last_str', '%seq%i_last.dump' % (self.prefix, self.idx))
        self.last_data = kwargs.get('last_data', '%seq%i_last.data' % (self.prefix, self.idx))
        self.pickle_file = kwargs.get('pickle_file', '%seq%i_last.pickle' % (self.prefix, self.idx))
        self.json_file = kwargs.get('json_file', '%seq%i_last.json' % (self.prefix, self.idx))


    def exec(self, confId=0, temp=300.0, press=1.0, eq_step=5, omp=1, mpi=1, gpu=0, intel='auto', opt='auto', **kwargs):
        """
        preset.eq.Additional.exec

        Execution of additional equilibration 

        Args:
            mol: RDKit Mol object
        
        Optional args:
            confId: Target conformer ID (int)
            temp: Temperature(float, K)
            press: Pressure (float, g/cm**3)
            omp: Number of threads of OpenMP (int)
            mpi: Number of MPI process (int)
            gpu: Number of GPU (int)

        Returns:
            RDKit Mol object
        """

        utils.MolToPDBFile(self.mol, os.path.join(self.work_dir, self.pdb_file))
        lmp = lammps.LAMMPS(work_dir=self.work_dir, solver_path=self.solver_path)
        lmp.make_dat(self.mol, file_name=self.dat_file, confId=confId)

        dt1 = datetime.datetime.now()
        utils.radon_print('Additional equilibration (eq%i) by LAMMPS is running...' % self.idx, level=1)
        md = self.sampling(temp=temp, press=press, step=int(1000000*eq_step), **kwargs)
        self.mol = lmp.run(md, mol=self.mol, confId=confId, input_file=self.in_file, last_str=self.last_str, last_data=self.last_data,
                           omp=omp, mpi=mpi, gpu=gpu, intel=intel, opt=opt)
        utils.MolToJSON(self.mol, os.path.join(self.save_dir, self.json_file))
        utils.pickle_dump(self.mol, os.path.join(self.save_dir, self.pickle_file))
        dt2 = datetime.datetime.now()
        utils.radon_print('Complete additional equilibration (eq%i). Elapsed time = %s' % (self.idx, str(dt2-dt1)), level=1)

        return self.mol


    def make_lammps_input(self, confId=0, temp=300.0, press=1.0, eq_step=5, omp=1, mpi=1, gpu=0, **kwargs):

        utils.MolToPDBFile(self.mol, os.path.join(self.work_dir, self.pdb_file))
        lmp = lammps.LAMMPS(work_dir=self.work_dir, solver_path=self.solver_path)
        lmp.make_dat(self.mol, file_name=self.dat_file, confId=confId)

        md = self.sampling(temp=temp, press=press, step=int(1000000*eq_step), **kwargs)
        lmp.make_input(md, file_name=self.in_file)

        return True



def get_final_idx(work_dir):

    idx = 0

    last_list  = glob.glob(os.path.join(work_dir, '*eq[0-9]_last.data'))
    last_list2 = glob.glob(os.path.join(work_dir, '*eq[0-9][0-9]_last.data'))
    last_list3 = glob.glob(os.path.join(work_dir, '*eq[0-9][0-9][0-9]_last.data'))

    last_plist  = glob.glob(os.path.join(work_dir, '*eq[0-9]_last.pickle'))
    last_plist2 = glob.glob(os.path.join(work_dir, '*eq[0-9][0-9]_last.pickle'))
    last_plist3 = glob.glob(os.path.join(work_dir, '*eq[0-9][0-9][0-9]_last.pickle'))

    last_jlist  = glob.glob(os.path.join(work_dir, '*eq[0-9]_last.json'))
    last_jlist2 = glob.glob(os.path.join(work_dir, '*eq[0-9][0-9]_last.json'))
    last_jlist3 = glob.glob(os.path.join(work_dir, '*eq[0-9][0-9][0-9]_last.json'))


    if len(last_jlist) > 0:
        last_list = last_jlist
        if len(last_jlist2) > 0:
            last_list.extend(last_jlist2)
        if len(last_jlist3) > 0:
            last_list.extend(last_jlist3)

        for file in last_list:
            file = os.path.basename(file)

            m = re.search(r'eq[0-9]+_last\.json$', file)
            if m is not None:
                mi = re.search(r'[0-9]+', m.group())
                i = int(mi.group())
                if i > idx: idx = i

    elif len(last_plist) > 0:
        last_list = last_plist
        if len(last_plist2) > 0:
            last_list.extend(last_plist2)
        if len(last_plist3) > 0:
            last_list.extend(last_plist3)

        for file in last_list:
            file = os.path.basename(file)

            m = re.search(r'eq[0-9]+_last\.pickle$', file)
            if m is not None:
                mi = re.search(r'[0-9]+', m.group())
                i = int(mi.group())
                if i > idx: idx = i

    elif len(last_list) > 0:
        if len(last_list2) > 0:
            last_list.extend(last_list2)
        if len(last_list3) > 0:
            last_list.extend(last_list3)

        for file in last_list:
            file = os.path.basename(file)

            m = re.search(r'eq[0-9]+_last\.data$', file)
            if m is not None:
                mi = re.search(r'[0-9]+', m.group())
                i = int(mi.group())
                if i > idx: idx = i

    else:
        utils.radon_print('Cannot find any last lammps data files or pickle files of equilibration stages in %s' % (work_dir), level=2)

    return idx


def get_final_data(work_dir):
    if type(work_dir) is not list:
        work_dir = [work_dir]

    data_file = None
    for d in work_dir:
        idx = get_final_idx(d)
        data_files = glob.glob(os.path.join(d, '*eq%i_last.data' % idx))
        if len(data_files) > 0:
            data_file = data_files[0]
            break

    return data_file


def get_final_pickle(save_dir):
    if type(save_dir) is not list:
        save_dir = [save_dir]

    pickle_file = None
    for d in save_dir:
        idx = get_final_idx(d)
        pickle_files = glob.glob(os.path.join(d, '*eq%i_last.pickle' % idx))
        if len(pickle_files) > 0:
            pickle_file = pickle_files[0]
            break

    return pickle_file


def get_final_json(save_dir):
    if type(save_dir) is not list:
        save_dir = [save_dir]

    json_file = None
    for d in save_dir:
        idx = get_final_idx(d)
        json_files = glob.glob(os.path.join(d, '*eq%i_last.json' % idx))
        if len(json_files) > 0:
            json_file = json_files[0]
            break

    return json_file


def restore(save_dir, **kwargs):
    pkl = get_final_pickle(save_dir)
    mol = utils.pickle_load(pkl)
    return mol


def helper_options():
    op = {'check_eq': False}
    return op

