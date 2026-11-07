#!/usr/bin/env python3

#  Copyright (c) 2026. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.
#  Author: Ryohei Hosoya, Hidemine Furuya @ Institute of Science Tokyo
# ******************************************************************************
# sim.preset.ef_dp module
# ******************************************************************************

__version__ = '1.0b2'

import os
import sys
import datetime
import glob
import numpy as np
import math
import pandas as pd
import pickle
from scipy import optimize
from matplotlib import pyplot as plt
plt.rcParams['font.size'] = 11

import radonpy
from ...core import utils, const
from .. import gromacs, lammps
from . import eq
from ..md import MD

# from pkg_resources import parse_version
# utilbool = parse_version(utils.__version__) >= parse_version('0.2.8')
utilbool = True

import warnings
from scipy.optimize import OptimizeWarning
warnings.simplefilter('ignore', (RuntimeWarning, OptimizeWarning))

class further_Additional(eq.Equilibration):
    def __init__(self, mol, prefix='', work_dir=None, solver_path=None, idx=0, no_traj=False, **kwargs):
        
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
            if no_traj:
                self.dump_file = kwargs.get('dump_file')
                self.xtc_file = kwargs.get('xtc_file')
            else:
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
            if no_traj:
                self.dump_file = kwargs.get('dump_file')
                self.xtc_file = kwargs.get('xtc_file')
            else:
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
        md.dump_freq = kwargs.get('dump_freq', 1000)
        if kwargs.get('set_init_velocity', False):
            md.set_init_velocity = temp

        ensemble = kwargs.get('ensemble_s', 'npt')

        # add = []
        # add_f = []

        # if self.axis == 'x':
        #     ex = 'v_EField'
        #     ey = 0.0
        #     ez = 0.0
        # elif self.axis == 'y':
        #     ex = 0.0
        #     ey = 'v_EField'
        #     ez = 0.0
        # elif self.axis == 'z':
        #     ex = 0.0
        #     ey = 0.0
        #     ez = 'v_EField'
        # elif self.axis == 'xy':
        #     ex = 'v_EField'
        #     ey = 'v_EField'
        #     ez = 0.0
        # elif self.axis == 'xz':
        #     ex = 'v_EField'
        #     ey = 0.0
        #     ez = 'v_EField'
        # elif self.axis == 'yz':
        #     ex = 0.0
        #     ey = 'v_EField'
        #     ez = 'v_EField'
        # elif self.axis == 'xyz':
        #     ex = 'v_EField'
        #     ey = 'v_EField'
        #     ez = 'v_EField'

        # add.append('')
        # add.append('# efield')
        # add.append('fix EF%i all efield %s %s %s' % (1, ex, ey, ez))
        # if self.process == 'tuning':
        #     add.append('variable EField equal %s*time' % self.tuning_rate)
        # elif self.process == 'dp':
        #     add.append('variable EField equal %f*cos(2*PI*%f*time*1e-15)' % (self.evalue, self.freq))
        # add_f.append('unfix EF%i' % 1)

        md.add_md(ensemble, step, time_step=time_step, shake=True, t_start=temp, t_stop=temp,
                   p_start=press, p_stop=press, p_dump=p_dump, **kwargs)

        md.wf[-1].add_dipole()
        if self.process == 'tuning':
            md.wf[-1].add_efield(evalue=self.evalue, axis=self.axis, rate=self.tuning_rate)
        elif self.process == 'dp':
            md.wf[-1].add_efield(evalue=self.evalue, axis=self.axis, freq=self.freq)

        return md


    def tuning(self, tuning_plot, **kwargs):

        #echarge = 1.602176634E-19   # unit= e = 1.602176634E−19 C
        echarge = const.e   # unit C

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
            ax.set_xlabel(r'Electric field / V$\mathrm{\AA}^{-1}$', fontsize=16)
            ax.set_ylabel(r'Polarization / Cm$^{-2}$', fontsize=16)
            fig.subplots_adjust(left=0.18, right=0.97, bottom=0.13, top=0.82)
            plt.show()
            plt.clf()
            plt.close()

        return estimated_evalue


    def dipole_anal(self, props=None, printout=True, save=True, **kwargs):

        # epsilon0 = 8.854187817E-12    # unit = farad / meter = C / volts / meter
        # echarge = 1.602176634E-19   # unit= e = 1.602176634E−19 C
        epsilon0 = const.eps0   # unit: F/m
        echarge = const.e       # unit: C

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
            r_perm_corr = r_perm + props['refractive_index']**2 - 1
            l_tan = i_perm / r_perm_corr
            remarks = None
        else:
            r_perm_corr = None
            remarks = 'no refractive index data'
            utils.radon_print('No refractive index data.', level=1)

        r2, rmspe = self.calc_diff(polarization, polarization_re)
        print('r2 = %f, rmspe = %f' % (r2, rmspe))

        results = [fit_params[0], fit_params[1], r_perm, r_perm_corr, i_perm, l_tan, r2, rmspe, remarks]
        ef_dp_result = pd.DataFrame(results,
                                    index=['D0',
                                           'delta',
                                           'real part of permittivity',
                                           '(+ refractive_index^2 - 1)',
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



# if __name__ == '__main__':

#     data = {
#         'DBID': os.environ.get('RadonPy_DBID'),
#         'temp': float(os.environ.get('RadonPy_Temp', 300.0)),
#         'press': float(os.environ.get('RadonPy_Press', 1.0)),
#         'remarks': os.environ.get('RadonPy_Remarks', ''),
#         'efdp_ver': __version__,
#     }

#     print('DBID: %s' % data['DBID'])

#     omp = int(os.environ.get('RadonPy_OMP', 0))
#     mpi = int(os.environ.get('RadonPy_MPI', utils.cpu_count()))
#     gpu = int(os.environ.get('RadonPy_GPU', 0))
#     intel = 'auto'
#     opt = 'auto'
#     rst_pickle_file = os.environ.get('RadonPy_Pickle_File', None)
#     rst_json_file = os.environ.get('RadonPy_JSON_File', None)
#     work_dir = './%s' % data['DBID']
#     save_dir = os.path.join(work_dir, 'analyze')

#     dp_force = os.environ.get('RadonPy_DP_Force', 'False') == 'True'
#     tuning_maxe = float(os.environ.get('tuning_max_EField', 1.0))    # unit: V/angstrom
#     tuning_rate = float(os.environ.get('tuning_rate', 2.0e-7))    # unit: V/angstrom/fs
#     tuning_plot = os.environ.get('tuning_plot', 'False') == 'True'
#     wave_num = int(os.environ.get('number_of_waves', 10))
#     efield = os.environ.get('electric_field_value', 1.0)
#     if efield == "auto":
#         evalue = tuning_maxe
#     else:
#         evalue = [float(i) for i in efield.split(',')]
#     axis = os.environ.get('electric_field_axis', 'z')
#     freq = float(os.environ.get('electric_field_freq', 1.0))
#     ef_ensemble = os.environ.get('ef_ensemble', 'npt')

#     input_df = pd.read_csv(os.path.join(save_dir, 'results.csv'), index_col=0)
#     input_data = input_df.iloc[0].to_dict()
#     data = {**input_data, **data}
#     if not data['check_eq'] and not dp_force:
#         print('check_eq: FALSE')
#         sys.exit(0)

#     if rst_pickle_file:
#         mol = utils.pickle_load(os.path.join(save_dir, rst_pickle_file))
#     elif rst_json_file and utilbool:
#         mol = utils.JSONToMol(os.path.join(save_dir, rst_json_file))
#     else:
#         if glob.glob(os.path.join(save_dir, 'eq*_last.json'), recursive=True) and utilbool:
#             idx = eq.get_final_idx(work_dir)
#             if idx == 0:
#                 idx = eq.get_final_idx(save_dir)
#             mol = utils.JSONToMol(os.path.join(save_dir, 'eq%i_last.json' %idx))
#         elif glob.glob(os.path.join(save_dir, 'eq*_last.pickle'), recursive=True):
#             rst_pickle_file = eq.get_final_pickle([save_dir, work_dir])
#             mol = utils.pickle_load(rst_pickle_file)

#     raw_ef_step = wave_num * 1e+9 / freq
#     round_ef_step = round(raw_ef_step, 1-math.floor(math.log10(raw_ef_step)))   # unit = ns
#     if freq >= 1e+10:
#         simulation_freq = round_ef_step * 1000
#     else:
#         simulation_freq = round_ef_step * 200

#     xyz_direction = [direction for direction in axis.split(',')]
#     for direction in xyz_direction:
#         if efield == "auto":
#             tuning_time = evalue / tuning_rate
#             round_tuning_time = round(tuning_time, 1-math.floor(math.log10(tuning_time)))
#             tuning_step = round_tuning_time / 1000000
#             tuning_freq = round_tuning_time / 5000

#             # Execute tuning
#             efeq = further_Additional(mol, work_dir=work_dir, axis=direction, evalue=evalue, tuning_rate=tuning_rate, process='tuning')
#             if os.path.isfile(os.path.join(work_dir, 'ef_%s_tuning_last.data' % direction)):
#                 utils.radon_print('Tuning data exists.', level=1)
#             else:
#                 efeq.exec_dp(temp=data['temp'], press=data['press'], eq_step=tuning_step,
#                                  mpi=mpi, omp=omp, gpu=gpu,
#                                  thermo_freq=tuning_freq, dump_freq=tuning_freq,
#                                  ensemble_s=ef_ensemble, intel=intel, opt=opt)
#             estimated_evalue = round(efeq.tuning(tuning_plot), 2)

#             # Execute efield-MD
#             efeq = further_Additional(mol, work_dir=work_dir, axis=direction, evalue=estimated_evalue, freq=freq, process='dp')
#             if os.path.isfile(os.path.join(work_dir, 'ef_%sGHz_%sEF_%s_last.data' % (round(freq/1e+9, 2), estimated_evalue, direction))):
#                 utils.radon_print('Calculating permittivity (%s direction, %1.2f V/A)' %(direction, estimated_evalue), level=1)
#             else:
#                 efeq.exec_dp(temp=data['temp'], press=data['press'], eq_step=round_ef_step,
#                              mpi=mpi, omp=omp, gpu=gpu,
#                              thermo_freq=simulation_freq, dump_freq=simulation_freq,
#                              ensemble_s=ef_ensemble, intel=intel, opt=opt)
#             results = efeq.dipole_anal(props=data, printout=True, save=True)

#         else:
#             for e in evalue:
#                 efeq = further_Additional(mol, work_dir=work_dir, axis=direction, evalue=e, freq=freq, process='dp')
#                 if os.path.isfile(os.path.join(work_dir, 'ef_%sGHz_%sEF_%s_last.data' % (round(freq/1e+9, 2), e, direction))):
#                     utils.radon_print('Calculating permittivity (%s direction, %1.2f V/A)' %(direction, e), level=1)
#                 else:
#                     # Execute efield-MD
#                     efeq.exec_dp(temp=data['temp'], press=data['press'], eq_step=round_ef_step,
#                                  mpi=mpi, omp=omp, gpu=gpu,
#                                  thermo_freq=simulation_freq, dump_freq=simulation_freq,
#                                  ensemble_s=ef_ensemble, intel=intel, opt=opt)
#                 results = efeq.dipole_anal(props=data, printout=True, save=True)




