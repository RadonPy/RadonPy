#  Copyright (c) 2025. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.
#  Author: kazuyoshi Kaneko, Naoya Kowatari @ The Yokohama Rubber Co., Ltd.
# ******************************************************************************
# sim.preset.sp module
# ******************************************************************************

import os
import glob
import json
import re
import datetime
from pathlib import Path
import numpy as np
from rdkit import Chem
from ...core import calc, const, utils
from .. import gromacs, lammps, preset
from ..md import MD
from ..md_wrapper import MD_solver
from matplotlib import pyplot as plt
import pandas as pd
from . import eq

__version__ = '0.1.1'

# Lines that LAMMPS prints between the thermo rows of the rerun. With OPENMP styles (-sf omp), pe/mol/tally
# warns that the pair style did not tally, although the energies equal those of the plain styles.
IGNORE_LOG = ['WARNING: Compute pe/mol/tally only called from pair style',
              'WARNING: Energy was not tallied by pair style',
              'No /omp style for force computation currently active',
              'Last active /omp style is',
              'WARNING: Too many warnings:']


class SPMD(preset.Preset):
    def __init__(self, mol, prefix='', work_dir=None, save_dir=None, solver_path=None, idx=None, sp_dir='sp_calc',
                 solver='lammps', interaction_profile=None, **kwargs):
        """
        SPMD class create input file for sp value calculation

        Optional args:
            solver: lammps or gromacs (str)
            interaction_profile: None (RadonPy defaults) or 'portable' (str, for development and validation)
        """
        super().__init__(mol, prefix=prefix, work_dir=work_dir, save_dir=save_dir, solver_path=solver_path, **kwargs)
        self.solver = solver.lower()
        self.interaction_profile = interaction_profile
        os.makedirs(os.path.join(work_dir, sp_dir), exist_ok=True)
        self.data = {}
        self.idx = eq.get_final_idx(self.work_dir) if idx is None else idx

        self.in_file = kwargs.get('in_file', os.path.join(sp_dir, '%ssp%i.in' % (self.prefix, self.idx)))
        self.dat_file = kwargs.get('dat_file', os.path.join(sp_dir, '%ssp%i.data' % (self.prefix, self.idx)))
        # self.pdb_file = kwargs.get('pdb_file', os.path.join(sp_dir, '%ssp%i.pdb' % (self.prefix, self.idx)))
        self.log_file = kwargs.get('log_file', os.path.join(sp_dir, '%ssp%i.log' % (self.prefix, self.idx)))
        self.dump_file = kwargs.get('dump_file', os.path.join(sp_dir, '%ssp%i.dump' % (self.prefix, self.idx)))
        self.xtc_file = kwargs.get('xtc_file', os.path.join(sp_dir, '%ssp%i.xtc' % (self.prefix, self.idx)))
        self.rg_file = kwargs.get('rg_file', os.path.join(sp_dir, '%ssp%i.profile' % (self.prefix, self.idx)))
        self.last_str = kwargs.get('last_str', os.path.join(sp_dir, '%ssp%i_last.dump' % (self.prefix, self.idx)))
        self.last_data = kwargs.get('last_data', os.path.join(sp_dir, '%ssp%i_last.data' % (self.prefix, self.idx)))
        self.pickle_file = kwargs.get('pickle_file', os.path.join(sp_dir, '%ssp%i_last.pickle' % (self.prefix, self.idx)))
        self.json_file = kwargs.get('json_file', os.path.join(sp_dir, '%ssp%i_last.json' % (self.prefix, self.idx)))

    def rerun(self, temp=300.0, press=1.0, step=5000000, **kwargs):

        p_dump = 1000
        md = MD(mol=self.mol, interaction_profile=self.interaction_profile)
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
        md.rst = False
        md.outstr = kwargs.get('last_str', None)
        md.write_data = kwargs.get('last_data', None)
        md.add_md('npt', step, time_step=1.0, shake=True, t_start=temp, t_stop=temp,
                   p_start=press, p_stop=press, p_dump=p_dump, **kwargs)
        
        md.dump_file = None
        md.xtc_file = None
        for i_chain in range(min(utils.count_mols(self.mol), 31)):                  
            md.wf[-1].add.append(f'group mol{i_chain} molecule {i_chain+1}')
            md.wf[-1].add.append(f'compute mol{i_chain}_kspace mol{i_chain} group/group mol{i_chain} pair no kspace yes molecule intra')
            md.wf[-1].add.append(f'variable mol{i_chain}_elong_intra equal -c_mol{i_chain}_kspace')
            md.wf[-1].add.append(f'compute mol{i_chain}_pair mol{i_chain} pe/mol/tally mol{i_chain}')
            md.wf[-1].add.append(f'variable mol{i_chain}_evdw_intra equal -c_mol{i_chain}_pair[1]')
            md.wf[-1].add.append(f'variable mol{i_chain}_ecoul_intra equal -c_mol{i_chain}_pair[2]')         
            md.wf[-1].thermo_style += [f'v_mol{i_chain}_evdw_intra', f'v_mol{i_chain}_ecoul_intra', f'v_mol{i_chain}_elong_intra']   
        md.wf[-1].add_rerun(dump_file='eq%i.dump' % self.idx)
        
        return md


    def analyze(self, ignore_log=IGNORE_LOG, **kwargs):

        analyzer = SPMD_analyze if self.solver == 'lammps' else SPMD_analyze_gromacs
        analy = analyzer(
            log_file  = os.path.join(self.work_dir, self.log_file),
            ignore_log = ignore_log,
            **kwargs
        )

        return analy


class SPMD_analysis:
    '''
    Solubility parameter from the thermo data of either engine (see SPMD_analyze and SPMD_analyze_gromacs).
    '''
    def __init__(self, log_file=None, save_dir=None,
                 ignore_log=IGNORE_LOG, **kwargs):
        super().__init__(log_file=log_file, ignore_log=ignore_log, **kwargs)
        self.save_dir = save_dir
        # self.ignore_log = ignore_log


    def Solubility_Parameter(self, init=2000, last=None, width=2000, f_width=2000, conv_a=1.0, conv_b=0.0,printout=False, save=None):
        """
        Solubility_Paremeter class calculates SP_total, SP_vdw, SP_ele, SP_ele_short, SP_ele_long
        
        The number of molecules must always be less than 31.
        
        """
        
        #intramolecule_energy
        thermo_df = self.dfs[-1]  #self.read_log(self.log_file, self.ignore_log)[-1]

        tmp_col = [ i for i in thermo_df.columns if '_evdw_intra' in i ]
        evdw_intra = thermo_df[tmp_col].mean(axis=1)*-1*const.cal2j*1000   # kcal/mol -> J/mol
        tmp_col = [ i for i in thermo_df.columns if '_ecoul_intra' in i ]
        ecoul_short_intra = thermo_df[tmp_col].mean(axis=1)*-1*const.cal2j*1000
        tmp_col = [ i for i in thermo_df.columns if '_elong_intra' in i ]
        ecoul_long_intra = thermo_df[tmp_col].mean(axis=1)*-1*const.cal2j*1000       
        eele_intra = ecoul_short_intra + ecoul_long_intra
        etotal_intra = evdw_intra + eele_intra
        
        #intermolecule_energy
        evdw_inter =  thermo_df['E_vdwl'].to_numpy() *const.cal2j*1000/len(tmp_col)
        eele_short_inter = thermo_df['E_coul'].to_numpy() *const.cal2j*1000/len(tmp_col)
        eele_long_inter = thermo_df['E_long'].to_numpy() *const.cal2j*1000/len(tmp_col)
        eele_inter = eele_short_inter + eele_long_inter
        etotal_inter =  evdw_inter + eele_inter

        #molar volume
        if 'Volume' in thermo_df.columns:
            V = thermo_df['Volume'].to_numpy() * 1e-24 # Angstrom**3 -> cm**3
        else:
            V = thermo_df['Lx'].to_numpy() * thermo_df['Ly'].to_numpy() * thermo_df['Lz'].to_numpy() * 1e-24 # Angstrom**3 -> cm**3

        M = V * (thermo_df['Density'].to_numpy()) # cm**3 * (g/cm**3) -> g        
        Mw = M * const.NA / len(tmp_col) # g / mol 
        
        Vm = Mw * V / M # cm**3/mol

        #SP_value
        nCED = (etotal_intra - etotal_inter) / Vm
        nSP_total = ((etotal_intra - etotal_inter) / Vm)**0.5
        nSP_vdw = ((evdw_intra - evdw_inter) / Vm)**0.5
        nSP_ele = ((eele_intra - eele_inter) / Vm)**0.5
        nSP_ele_short = ((ecoul_short_intra - eele_short_inter) / Vm)**0.5
        nSP_ele_long = ((ecoul_long_intra - eele_long_inter) / Vm)**0.5

        mCED = nCED.to_numpy()
        mSP_total = nSP_total.to_numpy()
        mSP_vdw = nSP_vdw.to_numpy()
        mSP_ele = nSP_ele.to_numpy()
        mSP_ele_short = nSP_ele_short.to_numpy()
        mSP_ele_long = nSP_ele_long.to_numpy()
        init = 2000
        last = 5000
        CED = mCED[init:last].mean()
        SP_total = mSP_total[init:last].mean()
        SP_vdw = mSP_vdw[init:last].mean()
        SP_ele = mSP_ele[init:last].mean()
        SP_ele_short = mSP_ele_short[init:last].mean()
        SP_ele_long = mSP_ele_long[init:last].mean()

        time = thermo_df['Time'].index
        time = time*0.001#fs--ps
        
        data = {}
        data['sp_ced']      = CED
        data['sp_total']      = SP_total
        data['sp_vdw']      = SP_vdw
        data['sp_ele']      = SP_ele
        data['sp_ele_short']      = SP_ele_short
        data['sp_ele_long']      = SP_ele_long
         
        data_conv_total = mSP_total
        data_conv_total = pd.DataFrame(data_conv_total)
        data_sma_total = data_conv_total.rolling(width).mean()
        data_sd_total = data_conv_total.rolling(width).std()
        data_se_total = data_sd_total / np.sqrt(width)
        
        plt.xlabel("Time [ps]", fontsize=12)
        plt.ticklabel_format(style="sci",  axis="y", scilimits=(0,0))
        plt.ylabel("Solubility_parameter_total [MPa^0.5]", fontsize=12)
        x1 = time.values[init:last]
        y1 = mSP_total[init:last]
        plt.plot(x1, y1, linewidth=0.1)
        plt.errorbar(time, data_sma_total.values, linewidth=2.0)
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(self.save_dir, 'SP_total.png'))

        data_conv_vdw = mSP_vdw
        data_conv_vdw = pd.DataFrame(data_conv_vdw)
        data_sma_vdw = data_conv_vdw.rolling(width).mean()
        data_sd_vdw = data_conv_vdw.rolling(width).std()
        data_se_vdw = data_sd_vdw / np.sqrt(width)
        
        plt.figure()
        plt.xlabel("Time [ps]", fontsize=12)
        plt.ticklabel_format(style="sci",  axis="y", scilimits=(0,0))
        plt.ylabel("Solubility_parameter_vdw [MPa^0.5]", fontsize=12)              
        y2 = mSP_vdw[init:last]
        plt.plot(x1, y2, linewidth=0.1)
        plt.errorbar(time, data_sma_vdw.values, linewidth=2.0)
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(self.save_dir, 'SP_vdw.png'))   

        data_conv_ele = mSP_ele
        data_conv_ele = pd.DataFrame(data_conv_ele)
        data_sma_ele = data_conv_ele.rolling(width).mean()
        data_sd_ele = data_conv_ele.rolling(width).std()
        data_se_ele = data_sd_ele / np.sqrt(width)         

        plt.figure()
        plt.xlabel("Time [ps]", fontsize=12)
        plt.ticklabel_format(style="sci",  axis="y", scilimits=(0,0))
        plt.ylabel("Solubility_parameter_ele [MPa^0.5]", fontsize=12)        
        y3 = mSP_ele[init:last]
        plt.plot(x1, y3, linewidth=0.1)
        plt.errorbar(time, data_sma_ele.values, linewidth=2.0)
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(self.save_dir, 'SP_ele.png'))                    
       
        data_conv_ele_short = mSP_ele_short
        data_conv_ele_short = pd.DataFrame(data_conv_ele_short)
        data_sma_ele_short = data_conv_ele_short.rolling(width).mean()
        data_sd_ele_short = data_conv_ele_short.rolling(width).std()
        data_se_ele_short = data_sd_ele_short / np.sqrt(width)         

        plt.figure()
        plt.xlabel("Time [ps]", fontsize=12)
        plt.ticklabel_format(style="sci",  axis="y", scilimits=(0,0))
        plt.ylabel("Solubility_parameter_ele_short [MPa^0.5]", fontsize=12)        
        y4 = mSP_ele_short[init:last]
        plt.plot(x1, y4, linewidth=0.1)
        plt.errorbar(time, data_sma_ele_short.values, linewidth=2.0)
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(self.save_dir, 'SP_ele_short.png'))                    

        data_conv_ele_long = mSP_ele_long
        data_conv_ele_long = pd.DataFrame(data_conv_ele_long)
        data_sma_ele_long = data_conv_ele_long.rolling(width).mean()
        data_sd_ele_long = data_conv_ele_long.rolling(width).std()
        data_se_ele_long = data_sd_ele_long / np.sqrt(width)         

        plt.figure()
        plt.xlabel("Time [ps]", fontsize=12)
        plt.ticklabel_format(style="sci",  axis="y", scilimits=(0,0))
        plt.ylabel("Solubility_parameter_ele_long [MPa^0.5]", fontsize=12)        
        y5 = mSP_ele_long[init:last]
        plt.plot(x1, y5, linewidth=0.1)
        plt.errorbar(time, data_sma_ele_long.values, linewidth=2.0)
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(self.save_dir, 'SP_ele_long.png'))                    

        data_conv_ced = mCED
        data_conv_ced = pd.DataFrame(data_conv_ced)
        data_sma_ced = data_conv_ced.rolling(width).mean()
        data_sd_ced = data_conv_ced.rolling(width).std()
        data_se_ced = data_sd_ced / np.sqrt(width)         

        plt.figure()
        plt.xlabel("Time [ps]", fontsize=12)
        plt.ticklabel_format(style="sci",  axis="y", scilimits=(0,0))
        plt.ylabel("Cohesive Energy Density [J cm^-3]", fontsize=12)        
        y6 = mCED[init:last]
        plt.plot(x1, y6, linewidth=0.1)
        plt.errorbar(time, data_sma_ced.values, linewidth=2.0)
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(self.save_dir, 'SP_CED.png'))                    

        return data


class SPMD_analyze(SPMD_analysis, lammps.Analyze):
    '''
    SPMD_analyze class reads and analyzes log files. 
    '''


class SPMD_analyze_gromacs(SPMD_analysis, gromacs.Analyze):
    '''
    SPMD analysis of GROMACS reruns; same calculation as SPMD_analyze.
    '''


class Rerun(SPMD):
    def __init__(self, mol, prefix='', work_dir=None, solver_path=None, idx=None, sp_dir='sp_calc', **kwargs):
        """
        Rerun class
        """
        super().__init__(mol, prefix=prefix, work_dir=work_dir, solver_path=solver_path, idx=idx, sp_dir=sp_dir, **kwargs)

        self.in_file = kwargs.get('in_file', os.path.join(sp_dir, '%ssp_rerun_%i.in' % (self.prefix, self.idx)))
        self.dat_file = kwargs.get('dat_file', os.path.join(sp_dir, '%ssp_rerun_%i.data' % (self.prefix, self.idx)))
        # self.pdb_file = kwargs.get('pdb_file', os.path.join(sp_dir, '%ssp_rerun_%i.pdb' % (self.prefix, self.idx)))
        self.log_file = kwargs.get('log_file', os.path.join(sp_dir, '%ssp_rerun_%i.log' % (self.prefix, self.idx)))
        self.dump_file = kwargs.get('dump_file', None)
        self.xtc_file = kwargs.get('xtc_file', None)
        self.rg_file = kwargs.get('rg_file', None)
        self.last_str = kwargs.get('last_str', None)
        self.last_data = kwargs.get('last_data', None)

    def exec(self, confId=0, temp=300.0, press=1.0, eq_step=5, omp=1, mpi=1, gpu=0, intel=None, opt='auto', **kwargs): # intel needs to be None.
        """
        preset.eq.Additional.exec

        Execution of additional SPMD 

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

        if self.solver != 'lammps':
            return self.exec_gromacs(omp=omp, gpu=gpu)
        lmp = lammps.LAMMPS(work_dir=self.work_dir, solver_path=self.solver_path)
        lmp.make_dat(self.mol, file_name=self.dat_file, confId=confId)

        dt1 = datetime.datetime.now()
        utils.radon_print('sp_rerun by LAMMPS is running...', level=1)
        md = self.rerun(temp=temp, press=press, step=int(1000000*eq_step), **kwargs)
        self.mol = lmp.run(md, mol=self.mol, confId=confId, input_file=self.in_file, last_str=self.last_str, last_data=self.last_data,
                            omp=omp, mpi=mpi, gpu=gpu, intel=intel, opt=opt)#kaneko
        dt2 = datetime.datetime.now()
        utils.radon_print('Complete sp_rerun . Elapsed time = %s' % (str(dt2-dt1)), level=1)

        result = self.analyze(save_dir=self.save_dir).Solubility_Parameter()
        self.data.update(result)
        
        return self.mol, self.data


    def exec_gromacs(self, omp=1, gpu=0):
        """
        The energy decomposition of exec with GROMACS reruns of the eq trajectory (eq<idx>.xtc)

        A rerun of the whole system gives E_vdwl, E_coul and E_long; a rerun of each molecule alone in
        the same boxes (the other atoms without charges or LJ) gives its intramolecular vdW, short-range
        and reciprocal-space Coulomb energies, including its periodic images, as LAMMPS pe/mol/tally and
        group/group kspace do. Like LAMMPS, at most 31 molecules are decomposed. The frames are those of
        the XTC trajectory (0.001 nm precision).
        """
        from mdtraj.formats import XTCTrajectoryFile
        dt1 = datetime.datetime.now()
        utils.radon_print('sp_rerun by GROMACS is running...', level=1)
        with XTCTrajectoryFile(os.path.join(self.work_dir, 'eq%i.xtc' % self.idx)) as traj:
            xyz, time, _, box = traj.read()
        xyz, lengths = xyz*10, box[:, [0, 1, 2], [0, 1, 2]]*10
        md = MD(mol=self.mol, interaction_profile=self.interaction_profile)
        solver = self.get_solver()
        name = os.path.splitext(self.log_file)[0]
        frame = solver.rerun(md, self.mol, xyz, lengths, name=name+'_total', omp=omp, gpu=gpu)
        frame['Time'] = time*1000
        for i, atoms in enumerate(Chem.GetMolFrags(self.mol)[:31]):
            alone = utils.deepcopy_mol(self.mol)
            for atom in alone.GetAtoms():
                if atom.GetIdx() not in atoms:
                    atom.SetDoubleProp('AtomicCharge', 0.0)
                    atom.SetDoubleProp('ff_epsilon', 0.0)
            intra = solver.rerun(md, alone, xyz, lengths, name='%s_mol%i' % (name, i), omp=omp, gpu=gpu)
            # The signs of the LAMMPS thermo variables.
            frame['v_mol%i_evdw_intra' % i] = -intra['E_vdwl'].to_numpy()
            frame['v_mol%i_ecoul_intra' % i] = -intra['E_coul'].to_numpy()
            frame['v_mol%i_elong_intra' % i] = -intra['E_long'].to_numpy()
        Path(os.path.join(self.work_dir, self.log_file)+'.gmx.json').write_text(
            json.dumps({'frames': [frame.reset_index().to_dict('list')]}))
        dt2 = datetime.datetime.now()
        utils.radon_print('Complete sp_rerun . Elapsed time = %s' % (str(dt2-dt1)), level=1)
        result = self.analyze(save_dir=self.save_dir).Solubility_Parameter()
        self.data.update(result)
        return self.mol, self.data


    def get_solver(self):
        return MD_solver(md_solver=self.solver, work_dir=self.work_dir, solver_path=self.solver_path)


    def make_lammps_input(self, confId=0, temp=300.0, press=1.0, eq_step=5, omp=1, mpi=1, gpu=0, **kwargs):

        lmp = lammps.LAMMPS(work_dir=self.work_dir, solver_path=self.solver_path)
        lmp.make_dat(self.mol, file_name=self.dat_file, confId=confId)

        md = self.rerun(temp=temp, press=press, step=int(1000000*eq_step), **kwargs)
        lmp.make_input(md, file_name=self.in_file)

        return True

