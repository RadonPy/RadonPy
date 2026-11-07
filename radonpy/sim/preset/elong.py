#!/usr/bin/env python3

#  Copyright (c) 2026. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.
#  Author: Ryohei Hosoya, Hidemine Furuya @ Institute of Science Tokyo
# ******************************************************************************
# sim.preset.elong module
# ******************************************************************************

__version__ = '0.1.2'

import os
import sys
import datetime
import glob
import numpy as np
import pandas as pd
import pickle
import platform
import shutil
import math
import re
from scipy import optimize

import radonpy
from ...core import utils, calc
import json
from .. import gromacs, lammps
from . import eq, tc
from ..md import MD

# from pkg_resources import parse_version
# utilbool = parse_version(utils.__version__) >= parse_version('0.2.8')
utilbool = True

ef_dp_avail = True
try:
    import ef_dp
except ImportError:
    ef_dp_avail = False

mdtraj_avail = True
try:
    import mdtraj
except ImportError:
    mdtraj_avail = False

from matplotlib import pyplot as plt
plt.rcParams['font.size'] = 11


class Deformation(eq.Equilibration):
    def __init__(self, mol, prefix='', work_dir=None, solver_path=None, idx=0, **kwargs):
        
        super().__init__(mol, prefix=prefix, work_dir=work_dir, solver_path=solver_path, **kwargs)

        self.idx = idx
        self.process = kwargs.get('process', None)
        self.deform_strain = kwargs.get('deform_strain', 0.5)
        self.deform_direction = kwargs.get('axis', None)

        self.in_file = kwargs.get('in_file', 'eq%i_deform_%s_%ipercent.in' % (self.idx, self.deform_direction, int(self.deform_strain*100)))
        self.dat_file = kwargs.get('dat_file', 'eq%i_deform_%s_%ipercent.data' % (self.idx, self.deform_direction, int(self.deform_strain*100)))
        self.pdb_file = kwargs.get('pdb_file', 'eq%i_deform_%s_%ipercent.pdb' % (self.idx, self.deform_direction, int(self.deform_strain*100)))
        self.log_file = kwargs.get('log_file', 'eq%i_deform_%s_%ipercent.log' % (self.idx, self.deform_direction, int(self.deform_strain*100)))
        self.dump_file = kwargs.get('dump_file', 'eq%i_deform_%s_%ipercent.dump' % (self.idx, self.deform_direction, int(self.deform_strain*100)))
        self.xtc_file = kwargs.get('xtc_file', 'eq%i_deform_%s_%ipercent.xtc' % (self.idx, self.deform_direction, int(self.deform_strain*100)))
        self.last_str = kwargs.get('last_str', 'eq%i_deform_%s_%ipercent_last.dump' % (self.idx, self.deform_direction, int(self.deform_strain*100)))
        self.last_data = kwargs.get('last_data', 'eq%i_deform_%s_%ipercent_last.data' % (self.idx, self.deform_direction, int(self.deform_strain*100)))
        self.json_file = kwargs.get('json_file', 'eq%i_deform_%s_%ipercent_last.json' % (self.idx, self.deform_direction, int(self.deform_strain*100)))
        self.pickle_file = kwargs.get('pickle_file', 'eq%i_deform_%s_%ipercent_last.pickle' % (self.idx, self.deform_direction, int(self.deform_strain*100)))
        self.rst1_file = 'eq%i_deform_%s_%ipercent_md_1.rst' % (self.idx, self.deform_direction, int(self.deform_strain*100))
        self.rst2_file = 'eq%i_deform_%s_%ipercent_md_2.rst' % (self.idx, self.deform_direction, int(self.deform_strain*100))

        self.save_dir = os.path.join(work_dir, 'analyze')


    def exec_deform(self, confId=0, press=1.0, time_step=1.0,
                    init_temp=300.0, max_temp=500.0, heating_step=1e5, # heating process
                    relax_step=500000, deform_rate=5e-6, deform_step=100000, # deformation process
                    cooling_step=500, interval_temp=10, # cooling process
                    omp=1, mpi=1, gpu=0, intel='auto', opt='auto', **kwargs):

        utils.MolToPDBFile(self.mol, os.path.join(self.work_dir, self.pdb_file))
        lmp = self.get_solver()
        lmp.make_dat(self.mol, file_name=self.dat_file, confId=confId)

        dt1 = datetime.datetime.now()
        utils.radon_print('Deformation MD (%s direction. Up to %i percent.) is running...' %(self.deform_direction, int(self.deform_strain*100)), level=1)
        md = self.sampling_deform(time_step=time_step, press=press,
                                  tstead=init_temp, tmax=max_temp, heating_step=heating_step, # heating process
                                  deform_rate=deform_rate, deform_step=deform_step, relax_step=relax_step, # deformation process
                                  cooling_step=cooling_step, interval_temp=interval_temp, # cooling process
                                  **kwargs)
        self.mol = lmp.run(md, mol=self.mol, confId=confId, input_file=self.in_file, last_str=self.last_str, last_data=self.last_data,
                           omp=omp, mpi=mpi, gpu=gpu, intel=intel, opt=opt)
        if utilbool:
            utils.MolToJSON(self.mol, os.path.join(self.save_dir, self.json_file))
        utils.pickle_dump(self.mol, os.path.join(self.save_dir, self.pickle_file))
        dt2 = datetime.datetime.now()
        utils.radon_print('Deformation MD (%s direction. Up to %i percent.); Elapsed time = %s' %(self.deform_direction, int(self.deform_strain*100), str(dt2-dt1)), level=1)

        return self.mol


    def sampling_deform(self, time_step=1.0, press=1.0,
                        tstead=300.0, tmax=500.0, heating_step=100000, # heating process
                        deform_rate=5e-6, deform_step=100000, relax_step=500000, # deformation process
                        cooling_step=500, interval_temp=10, # cooling process
                        **kwargs):
        if self.solver != 'lammps':
            return self.sampling_deform_options(time_step=time_step, press=press, tstead=tstead, tmax=tmax,
                                                heating_step=heating_step, deform_rate=deform_rate,
                                                deform_step=deform_step, relax_step=relax_step,
                                                cooling_step=cooling_step, interval_temp=interval_temp, **kwargs)
        p_dump = 1000
        md = MD(mol=self.mol, interaction_profile=self.interaction_profile)
        md.pair_style = self.pair_style
        md.cutoff_in = self.cutoff_in
        md.cutoff_out = self.cutoff_out
        md.kspace_style = self.kspace_style
        md.kspace_style_accuracy = self.kspace_style_accuracy
        md.neighbor = '%s bin' % self.neighbor_dis
        md.log_file = kwargs.get('log_file', self.log_file)
        md.dat_file = kwargs.get('dat_file', self.dat_file)
        md.rst = True
        md.rst1_file = kwargs.get('rst1_file', self.rst1_file)
        md.rst2_file = kwargs.get('rst2_file', self.rst2_file)
        md.outstr = kwargs.get('last_str', self.last_str)
        md.write_data = kwargs.get('last_data', self.last_data)
        md.dump_file = kwargs.get('dump_file', self.dump_file)
        md.xtc_file = kwargs.get('xtc_file', self.xtc_file)
        md.dump_freq = kwargs.get('dump_freq', 1000)
        md.dump_style = 'id type mol x y z ix iy iz vx vy vz'
        md.thermo_freq = kwargs.get('hthermo_freq', 100 if int(heating_step/1000)<100 else int('%s00' % str(heating_step/1000)[0]) if int(heating_step/1000)<1000 else 1000)
        dthermo_freq = kwargs.get('dthermo_freq', 100 if int(deform_step/1000)<100 else int('%s00' % str(deform_step/1000)[0]) if int(deform_step/1000)<1000 else 1000)
        drthermo_freq = kwargs.get('drthermo_freq', 100 if int(relax_step/1000)<100 else int('%s00' % str(relax_step/1000)[0]) if int(relax_step/1000)<1000 else 1000)
        cthermo_freq = kwargs.get('cthermo_freq', 100 if int(cooling_step/1000)<100 else int('%s00' % str(cooling_step/1000)[0]) if int(cooling_step/1000)<1000 else 1000)
        if kwargs.get('set_init_velocity', False):
            md.set_init_velocity = temp

        # heating process
        if tmax > tstead:
            add = []
            add.append('')
            md.add_md('npt', step=heating_step, time_step=time_step, shake=True, t_start=tstead, t_stop=tmax,
                       p_start=press, p_stop=press, p_dump=p_dump, add=add, **kwargs)
            md.add_md('npt', step=relax_step, time_step=time_step, shake=True, t_start=tmax, t_stop=tmax,
                      p_start=press, p_stop=press, p_dump=p_dump, add=add, **kwargs)

        # deformation process
        add = []
        add_f = []
        add.append('')
        add.append('# deformation process')
        add.append('fix shakeDEF1 all shake %.1e %i %i m %.1f' % (1e-4, 1000, 0, 1.0))
        add.append('')
        npt_conditions = 'fix md%i all npt temp %.6f %.6f %.6f tchain %i pchain %i' %(len(md.wf)+1, tmax, tmax, 100, 1, 1)
        px_start = kwargs.get('px_start') if self.deform_direction!='x' else None
        px_stop = kwargs.get('px_stop') if self.deform_direction!='x' else None
        px_dump = kwargs.get('px_dump') if self.deform_direction!='x' else None
        py_start = kwargs.get('py_start') if self.deform_direction!='y' else None
        py_stop = kwargs.get('py_stop') if self.deform_direction!='y' else None
        py_dump = kwargs.get('py_dump') if self.deform_direction!='y' else None
        pz_start = kwargs.get('pz_start') if self.deform_direction!='z' else None
        pz_stop = kwargs.get('pz_stop') if self.deform_direction!='z' else None
        pz_dump = kwargs.get('pz_dump') if self.deform_direction!='z' else None
        p_dict = {'x':[px_start, px_stop, px_dump], 'y':[py_start, py_stop, py_dump], 'z':[pz_start, pz_stop, pz_dump]}
        per_directions = []
        for keys, values in p_dict.items():
            if (values[0] and values[1] and values[2]) is not None:
                npt_conditions += (' %s %.1f %.1f %.1f' %(keys, values[0], values[1], values[2]))
                per_directions.append(keys)
        add.append(npt_conditions)
        add.append('fix DEF1 all deform %i %s erate %.1e units box remap x' %(1, self.deform_direction, deform_rate))
        add.append('')
        add.append('variable myL%s equal l%s' %(self.deform_direction, self.deform_direction))
        add.append('variable myL%s0 equal ${myL%s}' %(self.deform_direction, self.deform_direction))
        add.append('variable EngStrain equal (l%s-${myL%s0})/${myL%s0}' %(self.deform_direction, self.deform_direction, self.deform_direction))
        add.append('')
        add.append('fix momentum1 all momentum %i linear 1 1 1' % 50)
        add.append('variable GamNsurf equal l%s*(p%s%s-0.5*(p%s%s+p%s%s))*101325/1e10*1e3'
                   %(self.deform_direction, self.deform_direction, self.deform_direction, per_directions[0], per_directions[0], per_directions[1], per_directions[1]))
        add_f.append('unfix shakeDEF1')
        add_f.append('unfix DEF1')
        add_f.append('unfix momentum1')
        # Every run writes its thermo_style after the add lines, so the deformation variables are given as its own columns.
        md.add_md(None, step=deform_step, time_step=time_step, shake=False, t_start=tmax, t_stop=tmax,
                  thermo_style=['v_GamNsurf', 'v_EngStrain'], thermo_freq=dthermo_freq, add=add, add_f=add_f, **kwargs)

        # cooling process
        if tmax != tstead:
            add = []
            add.append('')
            add.append('# cooling process')
            add.append('thermo_style custom step time temp press enthalpy etotal ke pe ebond eangle edihed eimp evdwl ecoul elong etail vol lx ly lz density pxx pyy pzz pxy pxz pyz')
            add.append('thermo_modify flush yes')
            add.append('thermo %i' % cthermo_freq)
            add.append('')
            temp_range = np.arange(tmax, tstead, -interval_temp)
            for i, temp in enumerate(temp_range):
                if i == 0:
                    md.add_md('nvt', step=cooling_step, time_step=time_step, shake=True, t_start=temp, t_stop=temp, add=add, **kwargs)
                    md.add_md('nvt', step=cooling_step, time_step=time_step, shake=True, t_start=temp, t_stop=temp-interval_temp, **kwargs)
                else:
                    md.add_md('nvt', step=cooling_step, time_step=time_step, shake=True, t_start=temp, t_stop=temp, **kwargs)
                    md.add_md('nvt', step=cooling_step, time_step=time_step, shake=True, t_start=temp, t_stop=temp-interval_temp, **kwargs)
            md.add_md('nvt', step=relax_step, time_step=time_step, shake=True, t_start=tstead, t_stop=tstead, **kwargs)

        elif tmax == tstead:
            add = []
            add.append('')
            add.append('thermo_style custom step time temp press enthalpy etotal ke pe ebond eangle edihed eimp evdwl ecoul elong etail vol lx ly lz density pxx pyy pzz pxy pxz pyz')
            add.append('thermo_modify flush yes')
            add.append('thermo %i' % drthermo_freq)
            add.append('')
            md.add_md('nvt', step=relax_step, time_step=time_step, shake=True, t_start=tstead, t_stop=tstead, add=add, **kwargs)

        return md



    def sampling_deform_options(self, time_step=1.0, press=1.0, tstead=300.0, tmax=500.0, heating_step=100000,
                                deform_rate=5e-6, deform_step=100000, relax_step=500000, cooling_step=500,
                                interval_temp=10, **kwargs):
        """
        The protocol of sampling_deform as MD options instead of raw LAMMPS commands (for GROMACS)

        The deformation is a strain-rate deformation of the box with remap x, in NPT with the
        lateral pressures given as px/py/pz_start, _stop and _dump (Parrinello-Rahman), one
        thermo frequency for every stage (dthermo_freq), and no fix momentum (GROMACS removes the
        center-of-mass motion every 100 steps). Deformation_Analysis derives v_EngStrain and
        v_GamNsurf from the box lengths and pressures.
        """
        p_dump = 1000
        md = MD(mol=self.mol, interaction_profile=self.interaction_profile)
        md.pair_style = self.pair_style
        md.cutoff_in = self.cutoff_in
        md.cutoff_out = self.cutoff_out
        md.kspace_style = self.kspace_style
        md.kspace_style_accuracy = self.kspace_style_accuracy
        md.neighbor = '%s bin' % self.neighbor_dis
        md.log_file = kwargs.get('log_file', self.log_file)
        md.dat_file = kwargs.get('dat_file', self.dat_file)
        md.rst = True
        md.outstr = kwargs.get('last_str', self.last_str)
        md.write_data = kwargs.get('last_data', self.last_data)
        md.dump_file = kwargs.get('dump_file', self.dump_file)
        md.xtc_file = kwargs.get('xtc_file', self.xtc_file)
        md.dump_freq = kwargs.get('dump_freq', 1000)
        md.thermo_freq = kwargs.get('dthermo_freq', 100 if int(deform_step/1000)<100 else int('%s00' % str(deform_step/1000)[0]) if int(deform_step/1000)<1000 else 1000)
        lateral = {key: kwargs.pop(key) for key in list(kwargs)
                   if key[:2] in ('px', 'py', 'pz') and key[2:] in ('_start', '_stop', '_dump') and key[1] != self.deform_direction}
        for key in ('hthermo_freq', 'dthermo_freq', 'drthermo_freq', 'cthermo_freq'):
            kwargs.pop(key, None)
        if tmax > tstead:
            md.add_md('npt', step=heating_step, time_step=time_step, shake=True, t_start=tstead, t_stop=tmax,
                      p_start=press, p_stop=press, p_dump=p_dump, **kwargs)
            md.add_md('npt', step=relax_step, time_step=time_step, shake=True, t_start=tmax, t_stop=tmax,
                      p_start=press, p_stop=press, p_dump=p_dump, **kwargs)
        md.add_md('npt', step=deform_step, time_step=time_step, shake=True, t_start=tmax, t_stop=tmax,
                  p_start=press, p_stop=press, p_dump=p_dump, p_aniso=True, **lateral, **kwargs)
        md.wf[-1].add_deform(dftype='erate', axis=self.deform_direction, deform_rate=deform_rate, remap='x')
        if tmax != tstead:
            for temp in np.arange(tmax, tstead, -interval_temp):
                md.add_md('nvt', step=cooling_step, time_step=time_step, shake=True, t_start=temp, t_stop=temp, **kwargs)
                md.add_md('nvt', step=cooling_step, time_step=time_step, shake=True, t_start=temp, t_stop=temp-interval_temp, **kwargs)
        md.add_md('nvt', step=relax_step, time_step=time_step, shake=True, t_start=tstead, t_stop=tstead, **kwargs)
        return md



class further_Additional(eq.Equilibration):
    def __init__(self, mol, prefix='', work_dir=None, solver_path=None, idx=0, **kwargs):
        
        super().__init__(mol, prefix=prefix, work_dir=work_dir, solver_path=solver_path, **kwargs)

        self.evalue = kwargs.get('evalue', 1.0)
        self.axis = kwargs.get('axis', 'z')
        self.freq = kwargs.get('freq', 1.0)
        self.freq_name = str(round(self.freq/1e+9, 2))
        self.direction = kwargs.get('axis', None)
        self.process = kwargs.get('process', 'dp')
        self.tuning_rate = kwargs.get('tuning_rate', 2.0e-7)

        if self.process == 'tuning':
            self.in_file = kwargs.get('in_file', 'ef_%s_tuning.in' % self.direction)
            self.dat_file = kwargs.get('dat_file', 'ef_%s_tuning.data' % self.direction)
            self.pdb_file = kwargs.get('pdb_file', 'ef_%s_tuning.pdb' % self.direction)
            self.log_file = kwargs.get('log_file', 'ef_%s_tuning.log' % self.direction)
            self.dump_file = kwargs.get('dump_file', 'ef_%s_tuning.dump' % self.direction)
            self.xtc_file = kwargs.get('xtc_file', 'ef_%s_tuning.xtc' % self.direction)
            self.last_str = kwargs.get('last_str', 'ef_%s_tuning_last.dump' % self.direction)
            self.last_data = kwargs.get('last_data', 'ef_%s_tuning_last.data' % self.direction)
            self.json_file = kwargs.get('json_file', 'ef_%s_tuning_last.json' % self.direction)
            self.pickle_file = kwargs.get('pickle_file', 'ef_%s_tuning_last.pickle' % self.direction)

        elif self.process == 'dp':
            self.in_file = kwargs.get('in_file', 'ef_%sGHz_%sEF_%s.in' % (self.freq_name, self.evalue, self.direction))
            self.dat_file = kwargs.get('dat_file', 'ef_%sGHz_%sEF_%s.data' % (self.freq_name, self.evalue, self.direction))
            self.pdb_file = kwargs.get('pdb_file', 'ef_%sGHz_%sEF_%s.pdb' % (self.freq_name, self.evalue, self.direction))
            self.log_file = kwargs.get('log_file', 'ef_%sGHz_%sEF_%s.log' % (self.freq_name, self.evalue, self.direction))
            self.dump_file = kwargs.get('dump_file', 'ef_%sGHz_%sEF_%s.dump' % (self.freq_name, self.evalue, self.direction))
            self.xtc_file = kwargs.get('xtc_file', 'ef_%sGHz_%sEF_%s.xtc' % (self.freq_name, self.evalue, self.direction))
            self.last_str = kwargs.get('last_str', 'ef_%sGHz_%sEF_%s_last.dump' % (self.freq_name, self.evalue, self.direction))
            self.last_data = kwargs.get('last_data', 'ef_%sGHz_%sEF_%s_last.data' % (self.freq_name, self.evalue, self.direction))
            self.json_file = kwargs.get('json_file', 'ef_%sGHz_%sEF_%s_last.json' % (self.freq_name, self.evalue, self.direction))
            self.pickle_file = kwargs.get('pickle_file', 'ef_%sGHz_%sEF_%s_last.pickle' % (self.freq_name, self.evalue, self.direction))

        self.save_dir = os.path.join(work_dir, 'analyze')


    def exec_dp(self, confId=0, temp=300.0, press=1.0, eq_step=5, time_step=1.0,
                omp=1, mpi=1, gpu=0, intel='auto', opt='auto', **kwargs):
        
        utils.MolToPDBFile(self.mol, os.path.join(self.work_dir, self.pdb_file))
        lmp = self.get_solver()
        lmp.make_dat(self.mol, file_name=self.dat_file, confId=confId)

        dt1 = datetime.datetime.now()
        if self.process == 'tuning':
            utils.radon_print('Under tuning.', level=1)
        elif self.process == 'dp':
            utils.radon_print('MD: frequency=%sGHz, electric field=%s, efield axis=%s' % (self.freq_name, self.evalue, self.direction), level=1)
        md = self.sampling_dp(temp=temp, press=press, step=int(1000000*eq_step), time_step=time_step, **kwargs)
        self.mol = lmp.run(md, mol=self.mol, confId=confId, input_file=self.in_file, last_str=self.last_str, last_data=self.last_data,
                           omp=omp, mpi=mpi, gpu=gpu, intel=intel, opt=opt)
        if utilbool:
            utils.MolToJSON(self.mol, os.path.join(self.save_dir, self.json_file))
        utils.pickle_dump(self.mol, os.path.join(self.save_dir, self.pickle_file))
        dt2 = datetime.datetime.now()
        utils.radon_print('Elapsed time = %s' %str(dt2-dt1), level=1)


    def log_reader(self):
        return lammps.Analyze if self.solver == 'lammps' else gromacs.Analyze


    def sampling_dp(self, temp=300.0, press=1.0, step=5000000, time_step=1.0, **kwargs):

        p_dump = 1000
        md = MD(mol=self.mol, interaction_profile=self.interaction_profile)
        md.pair_style = self.pair_style
        md.cutoff_in = self.cutoff_in
        md.cutoff_out = self.cutoff_out
        md.kspace_style = self.kspace_style
        md.kspace_style_accuracy = self.kspace_style_accuracy
        md.neighbor = '%s bin' % self.neighbor_dis
        md.log_file = kwargs.get('log_file', self.log_file)
        md.dat_file = kwargs.get('dat_file', self.dat_file)
        md.dump_file = kwargs.get('dump_file', self.dump_file)
        md.xtc_file = kwargs.get('xtc_file', self.xtc_file)
        md.rst = True
        md.outstr = kwargs.get('last_str', self.last_str)
        md.write_data = kwargs.get('last_data', self.last_data)

        md.thermo_freq = kwargs.get('thermo_freq', 1000)
        ensemble = kwargs.get('ensemble_s', 'npt')

        add = []
        add_f = []

        if self.axis == 'x':
            ex = 'v_EField'
            ey = 0.0
            ez = 0.0
        elif self.axis == 'y':
            ex = 0.0
            ey = 'v_EField'
            ez = 0.0
        elif self.axis == 'z':
            ex = 0.0
            ey = 0.0
            ez = 'v_EField'
        elif self.axis == 'xy':
            ex = 'v_EField'
            ey = 'v_EField'
            ez = 0.0
        elif self.axis == 'xz':
            ex = 'v_EField'
            ey = 0.0
            ez = 'v_EField'
        elif self.axis == 'yz':
            ex = 0.0
            ey = 'v_EField'
            ez = 'v_EField'
        elif self.axis == 'xyz':
            ex = 'v_EField'
            ey = 'v_EField'
            ez = 'v_EField'

        add.append('')
        add.append('# efield')
        add.append('fix EF%i all efield %s %s %s' % (1, ex, ey, ez))
        if self.process == 'tuning':
            md.dump_freq = 1e10
            add.append('variable EField equal %s*time' % self.tuning_rate)
        elif self.process == 'dp':
            md.dump_freq = kwargs.get('dump_freq', 1000)
            add.append('variable EField equal %f*cos(2*PI*%f*time*1e-15)' % (self.evalue, self.freq))
        add_f.append('unfix EF%i' % 1)

        if kwargs.get('set_init_velocity', False):
            md.set_init_velocity = temp

        md.add_md(ensemble, step, time_step=time_step, shake=True, t_start=temp, t_stop=temp,
                   p_start=press, p_stop=press, p_dump=p_dump, add=add, add_f=add_f, **kwargs)

        md.wf[-1].add_dipole()

        return md


    def tuning(self, tuning_plot, **kwargs):

        echarge = 1.602176634E-19   # unit= e = 1.602176634E−19 C

        log_file = os.path.join(self.work_dir, self.log_file)
        lmpanal = self.log_reader()(log_file=log_file)
        logs = lmpanal.dfs[-1]
        times = logs['Time'].to_numpy()  # unit = fs
        dipoles = logs['v_mu%s' % self.axis].to_numpy() * 1e-10    # unit = charge(=e) * m
        cell_volume = logs['Volume'].to_numpy() * 1e-30 # angstrom^3 ---> m^3
        polarization = dipoles * echarge / cell_volume   # unit = charge(=C) / m^2
        tuning_efield = self.tuning_rate * times   # unit = volts / angstrom

        r2_thre = 0.90
        id_width = np.argmin(np.abs(tuning_efield - 0.01))
        fit_init = np.mean(polarization[: id_width+1])
        for i in range(id_width, len(tuning_efield)):
            polarization_min = np.min(polarization[: i])
            polarization_max = np.max(polarization[: i])
            polarization_normal = (polarization[: i] - polarization_min) / (polarization_max - polarization_min)
            fit_params_normal = self.fitting(tuning_efield[: i], polarization_normal[: i], fit_init)
            fit_line_normal = self.theoretic_curve(tuning_efield[: i], fit_params_normal[0], fit_init)
            r2, _ = self.calc_diff(polarization_normal[: i], fit_line_normal)
            if r2 > r2_thre or i == len(tuning_efield) - 1:
                target_id = i
                fit_line = self.theoretic_curve(tuning_efield, fit_params_normal[0], fit_init) * (polarization_max - polarization_min) + polarization_min
                break

        estimated_evalue = tuning_efield[target_id]

        if tuning_plot:
            fig, ax = plt.subplots()
            line1, = ax.plot(tuning_efield, polarization, linewidth=2, color='green', label='Polarization', zorder=1)
            line2, = ax.plot(tuning_efield[(fit_line<np.max(polarization))], fit_line[(fit_line<np.max(polarization))], linewidth=2, color='blue', label='Fitting curve', zorder=3)
            ax.axvline(estimated_evalue, linewidth=1, linestyle='--', color='black')
            ax.fill_between(tuning_efield[:target_id], np.min(polarization[:target_id+1]), np.max(polarization[:target_id+1]), alpha=0.5, color='cyan', zorder=0)
            ax.legend(handles=[line1, line2], loc='lower left', bbox_to_anchor=(0, 1.0), fontsize=14)
            ax.set_xlabel('Electric field / V$\mathrm{\AA}^{-1}$', fontsize=16)
            ax.set_ylabel(r'Polarization / Cm$^{-2}$', fontsize=16)
            fig.subplots_adjust(left=0.18, right=0.97, bottom=0.13, top=0.82)
            plt.show()
            plt.clf()
            plt.close()

        return estimated_evalue


    def dipole_anal(self, props=None, printout=True, save=True, **kwargs):

        epsilon0 = 8.854187817E-12    # unit = farad / meter = C / volts / meter
        echarge = 1.602176634E-19   # unit= e = 1.602176634E−19 C

        log_file = os.path.join(self.work_dir, self.log_file)
        lmpanal = self.log_reader()(log_file=log_file)
        logs = lmpanal.dfs[-1]
        logs = logs.iloc[int(logs.shape[0]/2):]
        times = logs['Time'].to_numpy()  # unit = fs
        dipoles = logs['v_mu%s' % self.axis].to_numpy() * 1e-10    # unit = charge(=e) * m
        cell_volume = logs['Volume'].to_numpy() * 1e-30 # angstrom^3 ---> m^3
        polarization = dipoles * echarge / cell_volume   # unit = charge(=C) / m^2
        E_field = self.evalue * 1e+10 * np.cos(2*np.pi*self.freq*times*1e-15) # unit = volts / m
        E_displace = epsilon0 * E_field + polarization   # unit = C / m^2

        fit_params = self.fitting(times, E_displace)
        fit_curve = self.theoretic_curve(times, fit_params[0], fit_params[1])
        polarization_re = fit_curve - epsilon0 * E_field
        r_perm, i_perm, l_tan = self.complex_permittivity(epsilon0, fit_params[0], fit_params[1])

        if props is not None and 'refractive_index' in props.keys():
            r_perm_corr = r_perm + props['refractive_index']**2
            remarks = None
        else:
            r_perm_corr = None
            remarks = 'no refractive index data'
            utils.radon_print('No refractive index data.', level=1)

        r2, rmspe = self.calc_diff(polarization, polarization_re)
        print('r2 = %f, rmspe = %f' % (r2, rmspe))

        results = np.append(fit_params, [r_perm, r_perm_corr, i_perm, l_tan, r2, rmspe, remarks])
        ef_dp_result = pd.DataFrame(results,
                                    index=['D0',
                                           'delta',
                                           'real part of permittivity',
                                           '(+ refractive_index^2)',
                                           'imaginary part of permittivity',
                                           'loss tangent',
                                           'r2',
                                           'rmspe',
                                           'remarks']).T
        if save:
            ef_dp_result.to_csv(os.path.join(self.save_dir, 'ef_dp_%sGHz_%sEF_%s_result.csv' %(self.freq_name, self.evalue, self.direction)))

        if printout:
            fig, ax = plt.subplots()
            ax2 = ax.twinx()
            line1, = ax.plot(times*1e-6, E_field*1e-10, linewidth=2, color='gray', label='Electric field', zorder=0)
            line2, = ax2.plot(times*1e-6, polarization, linewidth=2, color='green', label='Polarization', zorder=2)
            line3, = ax2.plot(times*1e-6, polarization_re, linewidth=2, color='blue', label='Theoretic curve', zorder=3)
            #ax.legend(handles=[line1, line2, line3], loc='upper left', fontsize=14)
            ax.set_xlabel('time / ns', fontsize=16)
            ax.set_ylabel(r'Electric field / V$\mathrm{\AA}^{-1}$', fontsize=16)
            ax2.set_ylabel(r'Polarization / Cm$^{-2}$', fontsize=16)
            fig.subplots_adjust(left=0.16, right=0.84, bottom=0.12, top=0.96)
            plt.savefig(os.path.join(self.save_dir, 'ef_dp_%sGHz_%sEF_%s.png' %(self.freq_name, self.evalue, self.direction)), dpi=300)
            plt.clf()
            plt.close()

        return ef_dp_result


    def fitting(self, xdata, ydata, delta=None):
        try:
            if self.process == 'dp':
                popt, _ = optimize.curve_fit(self.theoretic_curve,
                                             xdata, ydata,
                                             p0=[np.max(ydata), 0.3],
                                             #method='dogbox',
                                             #maxfev=500
                                             )
            elif self.process == 'tuning':
                popt, _ = optimize.curve_fit(lambda x, a: self.theoretic_curve(x, a, delta),
                                             xdata, ydata,
                                             p0=[(ydata[-1]-ydata[0])/(xdata[-1]-xdata[0])],
                                             )
        except RuntimeError:
            popt = [0, 0]

        return popt


    def theoretic_curve(self, x, a, delta):
        if self.process == 'dp':
            Y = a * np.cos((2*np.pi*self.freq*x*1e-15) - delta)   # unit = C / m^2
        elif self.process == 'tuning':
            Y = a * x + delta
        return Y


    def complex_permittivity(self, epsilon0, a, delta):
        # unit = (C/m^2) / (volts/m) / epsilon0 = C/volts/m/epsilon0 = F/m/epsilon0 = None
        r_perm = (a/(self.evalue*1e+10)) * np.cos(delta) / epsilon0
        i_perm = (a/(self.evalue*1e+10)) * np.sin(delta) / epsilon0
        return r_perm, i_perm, i_perm/r_perm


    def calc_diff(self, prop_data, fit_curve):
        prop_data_mean = np.mean(prop_data)
        if self.process == 'dp':
            prop_data -= prop_data_mean
        r2 = float(1 - sum((prop_data-fit_curve)**2) / sum((prop_data-prop_data.mean())**2))
        rmspe = np.sqrt(sum(((fit_curve-prop_data)/prop_data)**2) / len(prop_data))
        return r2, rmspe


class Deformation_Analysis():
    def __init__(self, work_dir=None, idx=0, **kwargs):

        self.idx = idx
        self.deform_strain = kwargs.get('deform_strain', 0.5)
        self.deform_direction = kwargs.get('axis', None)
        self.in_file = kwargs.get('in_file', 'eq%i_deform_%s_%ipercent.in' % (self.idx, self.deform_direction, int(self.deform_strain*100)))
        self.pdb_file = kwargs.get('pdb_file', 'eq%i_deform_%s_%ipercent.pdb' % (self.idx, self.deform_direction, int(self.deform_strain*100)))
        self.xtc_file = kwargs.get('xtc_file', 'eq%i_deform_%s_%ipercent.xtc' % (self.idx, self.deform_direction, int(self.deform_strain*100)))
        self.log_file = kwargs.get('log_file', 'eq%i_deform_%s_%ipercent.log' % (self.idx, self.deform_direction, int(self.deform_strain*100)))
        self.work_dir = work_dir
        self.save_dir = os.path.join(work_dir, 'analyze')
        self.solver = kwargs.get('solver', 'lammps').lower()


    def log_extraction_gromacs(self):
        """Stages of a GROMACS run, with the LAMMPS deformation variables computed from the box and pressures."""
        anal = gromacs.Analyze(log_file=os.path.join(self.work_dir, self.log_file))
        workflow = json.loads(open(anal.metadata['manifest']).read())['workflow']
        logs = [df.reset_index() for df in anal.dfs]
        index = [i for i, wf in enumerate(workflow) if wf.get('deform') == 'erate'][0]
        deform_log = self._deform_variables(logs[index])
        whole_log = pd.concat(logs, axis=0, ignore_index=True)
        return whole_log, deform_log


    def _deform_variables(self, deform_log):
        """The LAMMPS deformation variables v_EngStrain and v_GamNsurf from the box lengths and pressures."""
        axis = self.deform_direction
        lateral = [a for a in 'xyz' if a != axis]
        length = deform_log['L%s' % axis]
        deform_log['v_EngStrain'] = (length-length.iat[0])/length.iat[0]
        deform_log['v_GamNsurf'] = length*(deform_log['P%s%s' % (axis, axis)]
                                           - 0.5*(deform_log['P%s%s' % (lateral[0], lateral[0])]
                                                  + deform_log['P%s%s' % (lateral[1], lateral[1])]))*101325/1e10*1e3
        return deform_log


    def log_extraction(self,):

        if self.solver != 'lammps':
            return self.log_extraction_gromacs()
        nprocess = 0
        deform_process = None
        in_file = os.path.join(self.work_dir, self.in_file)
        with open(in_file, 'r') as f:
            lines = [s.replace(' \n', '').replace('\n', '').replace('\r', '').replace('*', ' ') for s in f.readlines()]
            for line in lines:
                if line.startswith('fix DEF') and deform_process is None:
                    deform_process = nprocess
                if 'run ' in line:
                    line_data = [f for f in line.split(' ')]
                    nprocess += 1

        log_file = os.path.join(self.work_dir, self.log_file)
        lmpanal = lammps.Analyze(log_file=log_file)
        logs = lmpanal.dfs
        nlogs = len(logs)
        init_id = nlogs - nprocess
        for i in range(init_id, nlogs):
            if 'v_EngStrain' in logs[i].columns:
                deform_log = logs[i]
            if i == init_id:
                whole_log = logs[i]
            else:
                whole_log = pd.concat([whole_log, logs[i]], axis=0, ignore_index=True)
        if 'deform_log' not in locals() and deform_process is not None:
            # The per-stage thermo_style written by MD replaces the deformation stage's
            # one, so v_EngStrain and v_GamNsurf are derived from the box as for GROMACS.
            deform_log = self._deform_variables(logs[init_id+deform_process])
            whole_log = pd.concat(logs[init_id:], axis=0, ignore_index=True)

        return whole_log, deform_log


    def density_plot(self,):

        whole_log, deform_log = self.log_extraction()
        time = whole_log['Time'].to_numpy() * 1e-6
        temp = whole_log['Temp'].to_numpy()
        wdensity = whole_log['Density'].to_numpy()
        ddensity = deform_log['Density'].to_numpy()
        strain = deform_log['v_EngStrain'].to_numpy()

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax12 = ax1.twinx()
        line1, = ax1.plot(time, wdensity, color='green', label='Density', linewidth=2)
        line2, = ax2.plot(strain, ddensity, color='green', label='Density', linewidth=2)
        line12, = ax12.plot(time, temp, color='orange', label='Temperature', linewidth=2)
        ax1.legend(handles=[line1, line12], loc='lower left', bbox_to_anchor=(0, 1.0), fontsize=14)
        ax1.set_xlabel('Time / ns', fontsize=14)
        ax1.set_ylabel('Density / $g$ $cm^{-3}$', fontsize=14)
        ax1.minorticks_on()
        ax2.legend(handles=[line2], loc='lower left', bbox_to_anchor=(0, 1.0), fontsize=14)
        ax2.set_xlabel('Strain', fontsize=14)
        ax2.set_ylabel('Density / $g$ $cm^{-3}$', fontsize=14)
        ax2.minorticks_on()
        ax12.set_ylabel('Temperature / K', fontsize=14)
        #ax1.set_ylim(0.0, 1.0)
        fig.subplots_adjust(left=0.13, right=0.97, bottom=0.23, top=0.72, wspace=0.80, hspace=0.0)
        plt.savefig(os.path.join(self.save_dir, 'density_plot.png'), dpi=300)
        plt.clf()
        plt.close()


    def nemorder_plot(self,):

        if mdtraj_avail:
            xtc_file = os.path.join(self.work_dir, self.xtc_file)
            pdb_file = os.path.join(self.work_dir, self.pdb_file)
            traj = mdtraj.load_xtc(xtc_file, top=pdb_file)
            S2 = mdtraj.compute_nematic_order(traj, indices='residues')
            dump_time = [i * 1e+3 for i in traj.time]    # unit: fs

            whole_log, _ = self.log_extraction()
            temp = whole_log['Temp'].to_numpy(dtype=float)
            log_time = whole_log['Time'].to_numpy(dtype=float)    # unit: fs
            strain = whole_log['v_EngStrain'].to_numpy(dtype=float)

            #log_data = {'Time': [], 'Temp': [], 'Strain': []}
            data = {'Time': [], 'Temp': [], 'Strain': [], 'Nematic order': S2}

            for i in range(len(dump_time)):
                if dump_time[i] in log_time:
                    log_data_id = np.where(log_time == dump_time[i])[0][0]
                    data['Time'].append(log_time[log_data_id] * 1e-6)    # unit: ns
                    data['Temp'].append(temp[log_data_id])
                    data['Strain'].append(strain[log_data_id])
                else:
                    data['Time'].append(None)
                    data['Temp'].append(None)
                    data['Strain'].append(None)

            df_all = pd.DataFrame(data)
            df_plot = df_all.dropna(thresh=3)
            df_all.to_csv(os.path.join(self.save_dir, 'nematic_order.csv'))

            fig = plt.figure()
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
            ax12 = ax1.twinx()
            line1, = ax1.plot(df_plot['Time'].values, df_plot['Nematic order'].values, color='green', label='Nematic order', linewidth=2)
            line2, = ax2.plot(df_plot['Strain'].values, df_plot['Nematic order'].values, color='green', label='Nematic order', linewidth=2)
            line12, = ax12.plot(df_plot['Time'].values, df_plot['Temp'].values, color='orange', label='Temperature', linewidth=2)
            ax1.legend(handles=[line1, line12], loc='lower left', bbox_to_anchor=(0, 1.0), fontsize=14)
            ax1.set_xlabel('Time / ns', fontsize=14)
            ax1.set_ylabel('Nematic order', fontsize=14)
            ax1.set_ylim(0.0, 1.0)
            ax1.minorticks_on()
            ax2.legend(handles=[line2], loc='lower left', bbox_to_anchor=(-0.1, 1.0), fontsize=14)
            ax2.set_xlabel('Strain', fontsize=14)
            ax2.set_ylabel('Nematic order', fontsize=14)
            ax2.set_ylim(0.0, 1.0)
            ax2.minorticks_on()
            ax12.set_ylabel('Temperature / K', fontsize=14)
            fig.subplots_adjust(left=0.13, right=0.97, bottom=0.23, top=0.72, wspace=0.80, hspace=0.0)
            plt.savefig(os.path.join(self.save_dir, 'nematic_order_plot.png'), dpi=300)
            plt.clf()
            plt.close()

        else:
            utils.radon_print('mdtraj does not exist. -> Skip nematic order plotting.', level=1)


if __name__ == '__main__':
    data = {
        'DBID': os.environ.get('RadonPy_DBID'),
        #'temp': float(os.environ.get('RadonPy_Temp', 300.0)),
        'press': float(os.environ.get('RadonPy_Press', 1.0)),
        'remarks': os.environ.get('RadonPy_Remarks', ''),
        'Python_ver': platform.python_version(),
        'RadonPy_ver': radonpy.__version__,
        'elong_ver': __version__,
    }

    print('DBID: %s' % data['DBID'])

    omp = int(os.environ.get('RadonPy_OMP', 0))
    mpi = int(os.environ.get('RadonPy_MPI', utils.cpu_count()))
    gpu = int(os.environ.get('RadonPy_GPU', 0))
    intel = 'auto'
    opt = 'auto'
    rst_pickle_file = os.environ.get('RadonPy_Pickle_File', None)
    #rst_data_file = os.environ.get('RadonPy_LAMMPS_Data_File', None)
    rst_json_file = os.environ.get('RadonPy_JSON_File', None)
    current_path = os.getcwd()
    work_dir = os.path.join(current_path, data['DBID'])
    save_dir = os.path.join(work_dir, 'analyze')

    # conditions for heating
    read_tg = os.environ.get('RadonPy_Read_Tg', 'True') == 'True'
    heating_rate = float(os.environ.get('RadonPy_Heating_Rate', 200.0))

    # conditions for deformation
    deform_axis = os.environ.get('RadonPy_Deformation_Direction', 'x')
    px_start = float(os.environ.get('RadonPy_Deformation_Start_External_Stress_x', 1.0))
    px_stop = float(os.environ.get('RadonPy_Deformation_Stop_External_Stress_x', 1.0))
    px_dump = float(os.environ.get('RadonPy_Deformation_Stress_Dumping_x', 350.0))
    py_start = float(os.environ.get('RadonPy_Deformation_Start_External_Stress_y', 1.0))
    py_stop = float(os.environ.get('RadonPy_Deformation_Stop_External_Stress_y', 1.0))
    py_dump = float(os.environ.get('RadonPy_Deformation_Stress_Dumping_y', 350.0))
    pz_start = float(os.environ.get('RadonPy_Deformation_Start_External_Stress_z', 1.0))
    pz_stop = float(os.environ.get('RadonPy_Deformation_Stop_External_Stress_z', 1.0))
    pz_dump = float(os.environ.get('RadonPy_Deformation_Stress_Dumping_z', 350.0))
    relax_step = int(os.environ.get('RadonPy_Deformation_Relaxation_Step', 500000))
    deform_rate = float(os.environ.get('RadonPy_Deformation_Rate', 5e-6))    # unit = epsilon / fs (i.e. 5e-6 = +0.0005% of unit cell / fs)
    deform_strains = [ds for ds in re.split(r'[:,]+', os.environ.get('RadonPy_Deformation_Strain', '0.5'))]
    deform_condition = {'strain':[], 'Tc': [], 'Dp': []}
    count = -1
    for ds in deform_strains:
        if ds[0].isdecimal():
            deform_condition['strain'].append(float(ds))
            deform_condition['Tc'].append(False)
            deform_condition['Dp'].append(False)
            count += 1
        elif ds.isalpha():
            if ds == 'Tc':
                deform_condition['Tc'][count] = True
            elif ds == 'Dp':
                deform_condition['Dp'][count] = True
    deform_force = os.environ.get('RadonPy_Deformation_Force', 'True') == 'True'
    input_df = pd.read_csv(os.path.join(save_dir, 'results.csv'), index_col=0)
    input_data = input_df.iloc[0].to_dict()
    data = {**input_data, **data}
    if not data['check_eq'] and not deform_force:
        print('check_eq: FALSE')
        sys.exit(0)

    # conditions for cooling
    cooling_rate = float(os.environ.get('RadonPy_Cooling_Rate', 8000.0))
    interval_temp = float(os.environ.get('RadonPy_Cooling_Interval_Temperature', 10.0))

    # conditions for tc
    exec_tc = int(os.environ.get('RadonPy_TryTC', 1))
    tc_axis = os.environ.get('RadonPy_TcAnalysis_Direction', 'x')
    tc_force = os.environ.get('RadonPy_TC_Force', 'True') == 'True'
    if not data['do_TC'] and not tc_force:
        sys.exit(0)

    # conditions for efdp
    dp_force = os.environ.get('RadonPy_DP_Force', 'True') == 'True'
    if not data['check_eq'] and not dp_force:
        print('check_eq: FALSE')
        sys.exit(0)
    tuning_maxe = float(os.environ.get('tuning_max_EField', 1.0))    # unit: V/angstrom
    tuning_rate = float(os.environ.get('tuning_rate', 2.0e-7))    # unit: V/angstrom/fs
    tuning_plot = os.environ.get('tuning_plot', 'False') == 'True'
    tuning_mag = float(os.environ.get('tuning_magnitude', 0.5))
    wave_num = int(os.environ.get('number_of_waves', 10))
    efield = os.environ.get('electric_field_value', 1.0)
    if efield == "auto":
        evalue = tuning_maxe
    else:
        evalue = [float(i) for i in efield.split(',')]
    ef_axis = os.environ.get('electric_field_axis', 'x')
    freq = float(os.environ.get('electric_field_freq', 1.0))
    ef_ensemble = os.environ.get('ef_ensemble', 'nvt')
    raw_ef_step = wave_num * 1e+9 / freq
    scale_ef_step = math.floor(math.log10(raw_ef_step))
    round_ef_step = round(raw_ef_step, 1-scale_ef_step)   # unit = ns
    if freq >= 1e+10:
        simulation_freq = round_ef_step * 1000
    else:
        simulation_freq = round_ef_step * 200

    idx = eq.get_final_idx(save_dir)

    # Elongation with NPT
    xyz_direction = [direction for direction in deform_axis.split(',')]
    for direction in xyz_direction:
        for deformid in range(len(deform_condition['strain'])):

            if rst_json_file and utilbool:
                mol = utils.JSONToMol(os.path.join(save_dir, rst_json_file))
            elif rst_pickle_file:
                mol = utils.pickle_load(os.path.join(save_dir, rst_pickle_file))
            else:
                if glob.glob(os.path.join(save_dir, 'eq*_last.json'), recursive=True) and utilbool:
                    mol = utils.JSONToMol(os.path.join(save_dir, 'eq%i_last.json' % idx))
                elif glob.glob(os.path.join(save_dir, 'eq*_last.pickle'), recursive=True):
                    mol = utils.pickle_load(eq.get_final_pickle([save_dir, work_dir]))

            if read_tg:
                if 'tg' in data.keys():
                    heating_maxtemp = data['tg'] + 200
                else:
                    heating_maxtemp = float(os.environ.get('RadonPy_Heating_Max_Temp', 500.0))
                    utils.radon_print('Tg data does not exist. -> Maximum temperature was set at %s.' % heating_maxtemp, level=1)
            else:
                heating_maxtemp = float(os.environ.get('RadonPy_Heating_Max_Temp', 500.0))

            deform_name = 'eq%i_deform_%s_%iK_%ipercent_rate%.0e' % (idx, direction, int(heating_maxtemp), int(deform_condition['strain'][deformid]*100), deform_rate)
            if not os.path.isdir(os.path.join(work_dir, deform_name)):
                os.mkdir(os.path.join(work_dir, deform_name))
                os.mkdir(os.path.join(work_dir, deform_name, 'analyze'))

            work_dir_rev = os.path.join(work_dir, deform_name)
            save_dir_rev = os.path.join(work_dir_rev, 'analyze')
            last_dfile = deform_name.replace('_%iK' % int(heating_maxtemp), '').replace('_rate%.0e' % deform_rate, '') + '_last'

            if os.path.isfile(os.path.join(save_dir_rev, last_dfile+'.json')) and utilbool:
                mol = utils.JSONToMol(os.path.join(save_dir_rev, last_dfile+'.json'))
            elif os.path.isfile(os.path.join(save_dir_rev, last_dfile+'.pickle')):
                mol = utils.pickle_load(os.path.join(save_dir_rev, last_dfile+'.pickle'))
            else:
                deform = Deformation(mol, idx=idx, work_dir=work_dir_rev, axis=direction,
                                     deform_strain=deform_condition['strain'][deformid])
                mol = deform.exec_deform(init_temp=data['temp'], max_temp=heating_maxtemp, heating_step=(heating_maxtemp-data['temp'])*heating_rate, # heating process
                                         px_start=px_start, px_stop=px_stop, px_dump=px_dump,
                                         py_start=py_start, py_stop=py_stop, py_dump=py_dump,
                                         pz_start=pz_start, pz_stop=pz_stop, pz_dump=pz_dump,
                                         deform_rate=deform_rate, deform_step=round(deform_condition['strain'][deformid]/deform_rate), relax_step=relax_step, # deformation process
                                         cooling_step=int(cooling_rate/2), interval_temp=interval_temp, # cooling process
                                         mpi=mpi, omp=omp, gpu=gpu, intel=intel, opt=opt)

                danal = Deformation_Analysis(idx=idx, work_dir=work_dir_rev, axis=direction, deform_strain=deform_condition['strain'][deformid])
                danal.density_plot()
                danal.nemorder_plot()

            if deform_condition['Tc'][deformid]:
                xyz_tc = [tcd for tcd in tc_axis.split(',')]
                for tcd in xyz_tc:
                    for i in range(exec_tc):
                        if not os.path.isfile(os.path.join(save_dir_rev, 'tc_prop_data_tc%s.csv' % tcd)):
                            # Non-equilibrium MD for thermal condultivity
                            nemd = tc.NEMD_MP(mol, axis=tcd, work_dir=work_dir_rev)
                            nemd.exec(decomp=True, temp=data['temp'], mpi=mpi, omp=omp, gpu=gpu)
                            nemd_analy = nemd.analyze()
                            TC = nemd_analy.calc_tc(decomp=True, save=True)

                            if not nemd_analy.Tgrad_data['Tgrad_check']:
                                data['check_tc'] = False
                                data['remarks'] = '[ERROR: Low linearity of temperature gradient.]'
                                utils.radon_print('Check Tc_%s: False' %tcd, level=1)
                            else:
                                data['check_tc'] = True
                                shutil.copyfile(os.path.join(save_dir_rev, 'tc_conv_data.csv'),
                                                os.path.join(save_dir_rev, 'tc_conv_data_tc%s.csv' % tcd))
                                shutil.copyfile(os.path.join(save_dir_rev, 'tc_prop_data.csv'),
                                                os.path.join(save_dir_rev, 'tc_prop_data_tc%s.csv' % tcd))
                                utils.radon_print('Check Tc_%s: True -> Complete' %tcd, level=1)

                            prop_data = {
                                'thermal_conductivity': TC,
                                'thermal_diffusivity': calc.thermal_diffusivity(TC, data['density'], data['Cp']),
                                **nemd_analy.TCdecomp_data
                                }
                            data.update(prop_data)

                            '''# Backup results.csv file
                            now = datetime.datetime.now()
                            if not os.path.isfile(os.path.join(save_dir_rev, 'results.csv')):
                                shutil.copyfile(os.path.join(save_dir, 'results.csv'),
                                                os.path.join(save_dir_rev, 'results.csv'))
                            shutil.copyfile(os.path.join(save_dir_rev, 'results.csv'),
                                            os.path.join(save_dir_rev, 'results_%i-%i-%i-%i-%i-%i.csv'
                                                         % (now.year, now.month, now.day, now.hour, now.minute, now.second)))'''

                            # Data output after NEMD
                            data_df = pd.DataFrame(data, index=[0]).set_index('DBID')
                            data_df.to_csv(os.path.join(save_dir_rev, 'results.csv'))
                            if data['check_tc']:
                                data_df.to_csv(os.path.join(save_dir_rev, 'results_tc%s.csv' % tcd))
                                break

                        else:
                             utils.radon_print('Completed Tc_%s data already exists. -> Skip' % tcd, level=1)


            if deform_condition['Dp'][deformid]:
                xyz_ef = [efd for efd in ef_axis.split(',')]
                for efd in xyz_ef:
                    if efield == "auto":
                        tuning_time = evalue / tuning_rate
                        round_tuning_time = round(tuning_time, 1-math.floor(math.log10(tuning_time)))
                        tuning_step = round_tuning_time / 1000000
                        tuning_freq = round_tuning_time / 5000

                        # Execute tuning
                        efeq = ef_dp.further_Additional(mol, work_dir=work_dir_rev, axis=efd, evalue=evalue, tuning_rate=tuning_rate, process='tuning') if ef_dp_avail \
                               else further_Additional(mol, work_dir=work_dir_rev, axis=efd, evalue=evalue, tuning_rate=tuning_rate, process='tuning')
                        if os.path.isfile(os.path.join(work_dir_rev, 'ef_%s_tuning_last.data' % efd)):
                            utils.radon_print('Tuning data exists.', level=1)
                        else:
                            efeq.exec_dp(temp=data['temp'], press=data['press'], eq_step=tuning_step,
                                             mpi=mpi, omp=omp, gpu=gpu,
                                             thermo_freq=tuning_freq, dump_freq=tuning_freq,
                                             ensemble_s=ef_ensemble, intel=intel, opt=opt)
                        estimated_evalue = round(efeq.tuning(tuning_plot), 2)

                        # Execute efield-MD
                        efeq = ef_dp.further_Additional(mol, work_dir=work_dir_rev, axis=efd, evalue=estimated_evalue, freq=freq, process='dp') if ef_dp_avail \
                               else further_Additional(mol, work_dir=work_dir_rev, axis=efd, evalue=estimated_evalue, freq=freq, process='dp')
                        if os.path.isfile(os.path.join(work_dir_rev, 'ef_%sGHz_%sEF_%s_last.data' % (round(freq/1e+9, 2), estimated_evalue, efd))):
                            utils.radon_print('Calculating permittivity (%s direction, %1.2f V/A)' %(efd, estimated_evalue), level=1)
                        else:
                            efeq.exec_dp(temp=data['temp'], press=data['press'], eq_step=round_ef_step,
                                         mpi=mpi, omp=omp, gpu=gpu,
                                         thermo_freq=simulation_freq, dump_freq=simulation_freq,
                                         ensemble_s=ef_ensemble, intel=intel, opt=opt)
                        results = efeq.dipole_anal(props=data, printout=True, save=True)

                    else:
                        for e in evalue:
                            efeq = ef_dp.further_Additional(mol, work_dir=work_dir_rev, axis=efd, evalue=e, freq=freq, process='dp') if ef_dp_avail \
                               else further_Additional(mol, work_dir=work_dir_rev, axis=efd, evalue=e, freq=freq, process='dp')
                            if os.path.isfile(os.path.join(work_dir_rev, 'ef_%sGHz_%sEF_%s_last.data' % (round(freq/1e+9, 2), e, direction))):
                                utils.radon_print('Calculating permittivity (%s direction, %1.2f V/A)' %(direction, e), level=1)
                            else:
                                # Execute efield-MD
                                efeq.exec_dp(temp=data['temp'], press=data['press'], eq_step=round_ef_step,
                                             mpi=mpi, omp=omp, gpu=gpu,
                                             thermo_freq=simulation_freq, dump_freq=simulation_freq,
                                             ensemble_s=ef_ensemble, intel=intel, opt=opt)
                            results = efeq.dipole_anal(props=data, printout=True, save=True)



