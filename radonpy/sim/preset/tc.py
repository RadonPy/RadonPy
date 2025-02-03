#  Copyright (c) 2023. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# ******************************************************************************
# sim.preset.tc module
# ******************************************************************************

import os
import numpy as np
from scipy import stats
import pandas as pd
import datetime
from matplotlib import pyplot as pp
from rdkit import Geometry as Geom
from ...core import poly, utils, calc, const
from .. import lammps, preset

__version__ = '0.2.9'


class NEMD_MP(preset.Preset):
    def __init__(self, mol, axis='x', prefix='', work_dir=None, save_dir=None, solver_path=None, **kwargs):
        super().__init__(mol, prefix=prefix, work_dir=work_dir, save_dir=save_dir, solver_path=solver_path, **kwargs)
        self.axis = axis
        self.dat_file = kwargs.get('dat_file', '%snemd_TC-MP_%s.data' % (prefix, axis))
        self.pdb_file = kwargs.get('pdb_file', '%snemd_TC-MP_%s.pdb' % (prefix, axis))
        self.in_file = kwargs.get('in_file', '%snemd_TC-MP_%s.in' % (prefix, axis))
        self.log_file = kwargs.get('log_file', '%snemd_TC-MP_%s.log' % (prefix, axis))
        self.dump_file = kwargs.get('dump_file', '%snemd_TC-MP_%s.dump' % (prefix, axis))
        self.xtc_file = kwargs.get('xtc_file', '%snemd_TC-MP_%s.xtc' % (prefix, axis))
        self.rst1_file = kwargs.get('rst1_file', '%snemd_TC-MP_%s_1.rst' % (prefix, axis))
        self.rst2_file = kwargs.get('rst2_file', '%snemd_TC-MP_%s_2.rst' % (prefix, axis))
        self.tprof_file = kwargs.get('tprof_file', '%sslabtemp_%s.profile' % (prefix, axis))
        self.lJprof_file = kwargs.get('lJprof_file', '%sheatflux_left_%s.profile' % (prefix, axis))
        self.rJprof_file = kwargs.get('rJprof_file', '%sheatflux_right_%s.profile' % (prefix, axis))
        self.last_str = kwargs.get('last_str', '%snemd_TC-MP_%s_last.dump' % (prefix, axis))
        self.last_data = kwargs.get('last_data', '%snemd_TC-MP_%s_last.data' % (prefix, axis))
        self.pickle_file = kwargs.get('pickle_file', '%snemd_TC-MP_%s_last.pickle' % (prefix, axis))
        self.json_file = kwargs.get('json_file', '%snemd_TC-MP_%s_last.json' % (prefix, axis))


    def exec(self, confId=0, step=5000000, time_step=0.2, temp=300.0,
             decomp=False, step_decomp=500000, decomp_intermol=False,
             omp=1, mpi=1, gpu=0, intel='auto', opt='auto', **kwargs):
        """
        preset.tc.NEMD_MP.exec

        Preset of thermal conductivity calculation by kinetic energy exchanging NEMD, a.k.a. reverse NEMD (RNEMD).
        LAMMPS only

        Args:
            mol: RDKit Mol object
        
        Optional args:
            confId: Target conformer ID (int)
            step: Number of step (int)
            time_step: Timestep (float)
            axis: Target axis (str)
            temp: Avarage temperature (float, K)
            decomp: Do decomposition analysis of heat flux (boolean)
            step_decomp: Number of step in decomposition analysis (int)
            solver_path: File path of LAMMPS (str) 
            work_dir: Path of work directory (str)
            omp: Number of threads of OpenMP (int)
            mpi: Number of MPI process (int)
            gpu: Number of GPU (int)

        Returns:
            RDKit Mol object
        """
        rep = kwargs.get('rep', 3)
        repo = kwargs.get('rep_other', 1)
        lmp = lammps.LAMMPS(work_dir=self.work_dir, solver_path=self.solver_path)

        self.make_lammps_input(confId=confId, step=step, time_step=time_step, temp=temp, rep=rep, rep_other=repo,
            decomp=decomp, step_decomp=step_decomp, decomp_intermol=decomp_intermol)

        dt1 = datetime.datetime.now()
        utils.radon_print('Thermal conductive simulation (kinetic energy exchanging NEMD) by LAMMPS is running...', level=1)

        intel = 'off' if decomp else intel
        cp = lmp.exec(input_file=self.in_file, omp=omp, mpi=mpi, gpu=gpu, intel=intel, opt=opt)
        if cp.returncode != 0 and (
                    (self.last_str is not None and not os.path.exists(os.path.join(self.work_dir, self.last_str)))
                    or (self.last_data is not None and not os.path.exists(os.path.join(self.work_dir, self.last_data)))
                ):
            utils.radon_print('Error termination of %s' % (lmp.get_name), level=3)
            return None

        if self.axis == 'x':
            self.mol = poly.super_cell(self.mol, x=rep, y=repo, z=repo, confId=confId)
        elif self.axis == 'y':
            self.mol = poly.super_cell(self.mol, x=repo, y=rep, z=repo, confId=confId)
        elif self.axis == 'z':
            self.mol = poly.super_cell(self.mol, x=repo, y=repo, z=rep, confId=confId)

        self.uwstr, self.wstr, self.cell, self.vel, _ = lmp.read_traj_simple(os.path.join(self.work_dir, self.last_str))

        for i in range(self.mol.GetNumAtoms()):
            self.mol.GetConformer(0).SetAtomPosition(i, Geom.Point3D(self.uwstr[i, 0], self.uwstr[i, 1], self.uwstr[i, 2]))
            self.mol.GetAtomWithIdx(i).SetDoubleProp('vx', self.vel[i, 0])
            self.mol.GetAtomWithIdx(i).SetDoubleProp('vy', self.vel[i, 1])
            self.mol.GetAtomWithIdx(i).SetDoubleProp('vz', self.vel[i, 2])

        setattr(self.mol, 'cell', utils.Cell(self.cell[0, 1], self.cell[0, 0], self.cell[1, 1], self.cell[1, 0], self.cell[2, 1], self.cell[2, 0]))
        self.mol = calc.mol_trans_in_cell(self.mol)
        utils.MolToJSON(self.mol, os.path.join(self.save_dir, self.json_file))
        utils.pickle_dump(self.mol, os.path.join(self.save_dir, self.pickle_file))

        dt2 = datetime.datetime.now()
        utils.radon_print('Complete thermal conductive simulation (kinetic energy exchanging NEMD). Elapsed time = %s' % str(dt2-dt1), level=1)

        return self.mol


    def make_lammps_input(self, confId=0, step=5000000, time_step=0.2, temp=300.0, rep=3, rep_other=1,
                            decomp=False, step_decomp=500000, decomp_intermol=False, **kwargs):

        lmp = lammps.LAMMPS(work_dir=self.work_dir, solver_path=self.solver_path)
        lmp.make_dat(self.mol, file_name=self.dat_file, confId=confId)
        seed = np.random.randint(1000, 999999)

        # Make input file
        in_strings  = 'variable        axis    string %s\n' % (self.axis)
        in_strings += 'variable        rep     equal  %i\n' % (rep)
        in_strings += 'variable        repo    equal  %i\n' % (rep_other)
        in_strings += 'variable        slab    equal  %i\n' % (kwargs.get('slab', 20))
        in_strings += 'variable        exchg   equal  %i\n' % (kwargs.get('exchg', 1000))
        in_strings += 'variable        Nevery  equal  %i\n' % (kwargs.get('Nevery', 1))
        in_strings += 'variable        TimeSt  equal  %f\n' % (time_step)
        in_strings += 'variable        NStep   equal  %i\n' % (step)
        in_strings += 'variable        NStepd  equal  %i\n' % (step_decomp)
        in_strings += 'variable        Ttemp   equal  %f\n' % (temp)
        in_strings += 'variable        dataf   string %s\n' % (self.dat_file)
        in_strings += 'variable        seed    equal  %i\n' % (seed)
        in_strings += '##########################################################\n'
        in_strings += '## Setting variables\n'
        in_strings += '##########################################################\n'
        in_strings += 'variable        logf    string %s\n' % (self.log_file)
        in_strings += 'variable        dumpf   string %s\n' % (self.dump_file)
        in_strings += 'variable        xtcf    string %s\n' % (self.xtc_file)
        in_strings += 'variable        rstf1   string %s\n' % (self.rst1_file)
        in_strings += 'variable        rstf2   string %s\n' % (self.rst2_file)
        in_strings += 'variable        Tprof   string %s\n' % (self.tprof_file)
        in_strings += 'variable        lJprof  string %s\n' % (self.lJprof_file)
        in_strings += 'variable        rJprof  string %s\n' % (self.rJprof_file)
        in_strings += 'variable        ldumpf  string %s\n' % (self.last_str)
        in_strings += 'variable        ldataf  string %s\n' % (self.last_data)
        in_strings += 'variable        pairst  string %s\n' % (self.pair_style)
        in_strings += 'variable        cutoff1 string %s\n' % (self.cutoff_in)
        in_strings += 'variable        cutoff2 string %s\n' % (self.cutoff_out)
        in_strings += 'variable        bondst  string %s\n' % (self.bond_style)
        in_strings += 'variable        anglest string %s\n' % (self.angle_style)
        in_strings += 'variable        dihedst string %s\n' % (self.dihedral_style)
        in_strings += 'variable        improst string %s\n' % (self.improper_style)
        in_strings += '##########################################################\n'

        in_strings += """
log             ${logf} append

units           real
atom_style      full
boundary        p p p

bond_style      ${bondst}  
angle_style     ${anglest}
dihedral_style  ${dihedst}
improper_style  ${improst}

pair_style      ${pairst} ${cutoff1} ${cutoff2}
pair_modify     mix arithmetic
special_bonds   amber
neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes
kspace_style    pppm 1e-6

read_data       ${dataf}

thermo_modify   flush yes
thermo          1000



##########################################################
## Preparation
##########################################################
variable        NA     equal 6.02214076*1.0e23
variable        kcal2j equal 4.184*1000
variable        ang2m  equal 1.0e-10
variable        fs2s   equal 1.0e-15

if "${axis} == x" then &
  "replicate    ${rep} ${repo} ${repo}" &
  "variable     ahi   equal  xhi" &
  "variable     alo   equal  xlo" &
  "variable     Jarea equal  ly*lz" &
  "variable     idx   equal  1" &
elif "${axis} == y" &
  "replicate    ${repo} ${rep} ${repo}" &
  "variable     ahi   equal  yhi" &
  "variable     alo   equal  ylo" &
  "variable     Jarea equal  lx*lz" &
  "variable     idx   equal  2" &
elif "${axis} == z" &
  "replicate    ${repo} ${repo} ${rep}" &
  "variable     ahi   equal  zhi" &
  "variable     alo   equal  zlo" &
  "variable     Jarea equal  lx*ly" &
  "variable     idx   equal  3"

variable        Nfreq   equal  ${exchg}/${Nevery}       # Number of data points to compute temperature during exchange interval 
variable        invslab equal  1/${slab}
variable        width   equal  (${ahi}-${alo})/${slab}
variable        llo     equal  ${alo}+${width}*1.0
variable        lhi     equal  ${alo}+(${slab}/2)*${width}
variable        rlo     equal  ${alo}+(1+${slab}/2)*${width}
variable        rhi     equal  ${ahi}

if "${axis} == x" then &
  "region       lhalf   block    ${llo}  ${lhi}  INF  INF  INF  INF  units box" &
  "region       rhalf   block    ${rlo}  ${rhi}  INF  INF  INF  INF  units box" &
elif "${axis} == y" &
  "region       lhalf   block    INF  INF  ${llo}  ${lhi}  INF  INF  units box" &
  "region       rhalf   block    INF  INF  ${rlo}  ${rhi}  INF  INF  units box" &
elif "${axis} == z" &
  "region       lhalf   block    INF  INF  INF  INF  ${llo}  ${lhi}  units box" &
  "region       rhalf   block    INF  INF  INF  INF  ${rlo}  ${rhi}  units box"
##########################################################



##########################################################
## Initial equilibration to control temperature
##########################################################
velocity        all create ${Ttemp} ${seed} mom yes rot yes dist gaussian
timestep        ${TimeSt}
fix             NVT all nvt temp ${Ttemp} ${Ttemp} 100

thermo_style    custom step time temp press enthalpy etotal ke pe ebond eangle edihed eimp evdwl ecoul elong etail vol lx ly lz density pxx pyy pzz pxy pxz pyz
thermo_modify   flush yes
thermo          ${exchg}

run             10000
unfix           NVT

reset_timestep  0



##########################################################
## NEMD with kinetic energy exchange (RNEMD)
##########################################################
fix             NVE all nve
fix             mp  all thermal/conductivity ${exchg} ${axis} ${slab}

# Generate temperature profile of layers
compute         layers all chunk/atom bin/1d ${axis} lower ${invslab} units reduced
fix             2 all ave/chunk ${Nevery} ${Nfreq} ${exchg} layers temp density/mass file ${Tprof} norm sample

# Output
dump            1 all custom 1000 ${dumpf} id type mol xs ys zs ix iy iz
dump            2 all xtc 1000 ${xtcf}
dump_modify     2 unwrap yes
restart         100000 ${rstf1} ${rstf2}

variable        heatflux   equal   (f_mp*${kcal2j}/${NA})/(2*${Jarea}*${ang2m}*${ang2m})   # J/m^2 = Ws/m^2
thermo_style    custom step time temp press enthalpy etotal ke pe ebond eangle edihed eimp evdwl ecoul elong etail vol lx ly lz density pxx pyy pzz pxy pxz pyz f_mp v_heatflux
thermo_modify   flush yes
thermo          ${exchg}

run             ${NStep}


"""

        if decomp:
            in_strings += """
##########################################################
## Component decomposition of heat flux
##########################################################
# heat flux preparation
compute         KE        all   ke/atom
compute         PE        all   pe/atom

compute         Spair     all   stress/atom NULL pair
compute         Sbond     all   stress/atom NULL bond
compute         Sangle    all   centroid/stress/atom NULL angle
compute         Sdihed    all   centroid/stress/atom NULL dihedral
compute         Simpro    all   centroid/stress/atom NULL improper
compute         Skspac    all   stress/atom NULL kspace
compute         Sfix      all   stress/atom NULL fix
"""

            if decomp_intermol:
                in_strings += """
compute         Spairer   all   stress/atom NULL interpair
compute         Spairra   all   stress/atom NULL intrapair
"""

            in_strings += """
# Generate empty vector
group           empty     type      99999
compute         KENULL    empty ke/atom
compute         PENULL    empty pe/atom improper
compute         STNULL    empty stress/atom NULL improper

########################   Cell half-left   ########################
###  |//|  |  |  |  |**|  |  |  |  |  ###   |//| cold slab
###  |//|  |  |  |  |**|  |  |  |  |  ###   |**| hot slab
###  |//|  |  |  |  |**|  |  |  |  |  ###
###  |//|  |  |  |  |**|  |  |  |  |  ###
###  |//|  |  |  |  |**|  |  |  |  |  ###
###      <--------->                  ###
### heat flux decomposition of this reagion
#####################################################################

# left cell information
group           halfL     dynamic  all         region     lhalf     every  ${Nevery}  # Nevery=ave/time Nevery

#1st term   eivi
compute         lF1ke     halfL    heat/flux   KE         PENULL    STNULL
compute         lF1pe     halfL    heat/flux   KENULL     PE        STNULL

#2nd term   Sivi
compute         lFpair    halfL    heat/flux   KENULL     PENULL    Spair
compute         lFbond    halfL    heat/flux   KENULL     PENULL    Sbond
compute         lFangle   halfL    heat/flux   KENULL     PENULL    Sangle
compute         lFdihed   halfL    heat/flux   KENULL     PENULL    Sdihed
compute         lFimpro   halfL    heat/flux   KENULL     PENULL    Simpro
compute         lFkspac   halfL    heat/flux   KENULL     PENULL    Skspac
compute         lFfix     halfL    heat/flux   KENULL     PENULL    Sfix
"""

            if decomp_intermol:
                in_strings += """
compute         lFpairer  halfL    heat/flux   KENULL     PENULL    Spairer
compute         lFpairra  halfL    heat/flux   KENULL     PENULL    Spairra
fix             20 halfL ave/time ${Nevery} ${Nfreq} ${exchg}  c_lF1ke[${idx}] c_lF1pe[${idx}] c_lFpair[${idx}]  c_lFpairer[${idx}]  c_lFpairra[${idx}]  c_lFbond[${idx}]  c_lFangle[${idx}]  c_lFdihed[${idx}]  c_lFimpro[${idx}]  c_lFkspac[${idx}]  c_lFfix[${idx}]  file ${lJprof}
"""
            else:
                in_strings += """
fix             20 halfL ave/time ${Nevery} ${Nfreq} ${exchg}  c_lF1ke[${idx}] c_lF1pe[${idx}] c_lFpair[${idx}]  c_lFbond[${idx}]  c_lFangle[${idx}]  c_lFdihed[${idx}]  c_lFimpro[${idx}]  c_lFkspac[${idx}]  c_lFfix[${idx}]  file ${lJprof}
"""

            in_strings += """

########################   Cell half-right   #######################
###  |//|  |  |  |  |**|  |  |  |  |  ###   |//| cold slab
###  |//|  |  |  |  |**|  |  |  |  |  ###   |**| hot slab
###  |//|  |  |  |  |**|  |  |  |  |  ###
###  |//|  |  |  |  |**|  |  |  |  |  ###
###  |//|  |  |  |  |**|  |  |  |  |  ###
###                     <--------->   ###
###        heat flux decomposition of this reagion
#####################################################################

# right cell information
group           halfR     dynamic  all         region     rhalf     every  ${Nevery}

#1st term   eivi
compute         rF1ke     halfR    heat/flux   KE         PENULL    STNULL
compute         rF1pe     halfR    heat/flux   KENULL     PE        STNULL

#2nd term   Sivi
compute         rFpair    halfR    heat/flux   KENULL     PENULL    Spair
compute         rFbond    halfR    heat/flux   KENULL     PENULL    Sbond
compute         rFangle   halfR    heat/flux   KENULL     PENULL    Sangle
compute         rFdihed   halfR    heat/flux   KENULL     PENULL    Sdihed
compute         rFimpro   halfR    heat/flux   KENULL     PENULL    Simpro
compute         rFkspac   halfR    heat/flux   KENULL     PENULL    Skspac
compute         rFfix     halfR    heat/flux   KENULL     PENULL    Sfix
"""

            if decomp_intermol:
                in_strings += """
compute         rFpairer  halfR    heat/flux   KENULL     PENULL    Spairer
compute         rFpairra  halfR    heat/flux   KENULL     PENULL    Spairra
fix             30 halfR ave/time ${Nevery} ${Nfreq} ${exchg}  c_rF1ke[${idx}] c_rF1pe[${idx}] c_rFpair[${idx}]  c_rFpairer[${idx}]  c_rFpairra[${idx}]  c_rFbond[${idx}]  c_rFangle[${idx}]  c_rFdihed[${idx}]  c_rFimpro[${idx}]  c_rFkspac[${idx}]  c_rFfix[${idx}]  file ${rJprof}
"""
            else:
                in_strings += """
fix             30 halfR ave/time ${Nevery} ${Nfreq} ${exchg}  c_rF1ke[${idx}] c_rF1pe[${idx}] c_rFpair[${idx}]  c_rFbond[${idx}]  c_rFangle[${idx}]  c_rFdihed[${idx}]  c_rFimpro[${idx}]  c_rFkspac[${idx}]  c_rFfix[${idx}]  file ${rJprof}
"""

            in_strings += """

##########################################################
## RNEMD with kinetic energy exchange in decomposition
##########################################################
thermo_style    custom step time temp press enthalpy etotal ke pe ebond eangle edihed eimp evdwl ecoul elong etail vol lx ly lz density pxx pyy pzz pxy pxz pyz f_mp v_heatflux
thermo_modify   flush yes
thermo          ${exchg}

run             ${NStepd}


"""

        in_strings += """
write_dump      all custom ${ldumpf} id x y z xu yu zu vx vy vz fx fy fz modify sort id
write_data      ${ldataf}
quit
"""

        with open(os.path.join(self.work_dir, self.in_file), 'w') as fh:
            fh.write(in_strings)
            fh.flush()
            if hasattr(os, 'fdatasync'):
                os.fdatasync(fh.fileno())
            else:
                os.fsync(fh.fileno())

        mol_sc = utils.deepcopy_mol(self.mol)
        if self.axis == 'x':
            mol_sc = poly.super_cell(mol_sc, x=rep, y=rep_other, z=rep_other, confId=confId)
        elif self.axis == 'y':
            mol_sc = poly.super_cell(mol_sc, x=rep_other, y=rep, z=rep_other, confId=confId)
        elif self.axis == 'z':
            mol_sc = poly.super_cell(mol_sc, x=rep_other, y=rep_other, z=rep, confId=confId)

        utils.MolToPDBFile(mol_sc, os.path.join(self.work_dir, self.pdb_file))

        return True


    def analyze(self):

        anal = NEMD_MP_Analyze(
            axis = self.axis,
            log_file  = os.path.join(self.work_dir, self.log_file),
            tprof_file = os.path.join(self.work_dir, self.tprof_file),
            lJprof_file  = os.path.join(self.work_dir, self.lJprof_file),
            rJprof_file  = os.path.join(self.work_dir, self.rJprof_file),
            traj_file = os.path.join(self.work_dir, self.xtc_file),
            pdb_file  = os.path.join(self.work_dir, self.pdb_file),
            dat_file  = os.path.join(self.work_dir, self.dat_file)
        )

        return anal



class NEMD_MP_Analyze(lammps.Analyze):
    def __init__(self, axis='x', prefix='', **kwargs):
        kwargs['log_file'] = kwargs.get('log_file', '%snemd_TC-MP_%s.log' % (prefix, axis))
        super().__init__(**kwargs)
        self.axis = axis
        self.tprof_file = kwargs.get('tprof_file', '%sslabtemp_%s.profile' % (prefix, axis))
        self.lJprof_file = kwargs.get('lJprof_file', '%sheatflux_left_%s.profile' % (prefix, axis))
        self.rJprof_file = kwargs.get('rJprof_file', '%sheatflux_right_%s.profile' % (prefix, axis))

        self.TC = np.nan
        self.Tgrad_data = {}
        self.Qgrad_data = {}
        self.TCdecomp_data = {}
        self.Jdecomp_data = {}

        self.threshold_r2 = 0.98
        self.threshold_r2_i = 0.95
        self.threshold_rate = 0.667


    def calc_tc(self, init=4000, last=None, decomp=False, tschunk=1, printout=False, save=False, save_name='analyze'):

        if save:
            save_dir = os.path.join(os.path.dirname(self.log_file), save_name)
        else:
            save_dir = None

        if decomp:
            thermo_df = pd.concat((self.dfs[-2], self.dfs[-1]), sort=False)
        else:
            thermo_df = self.dfs[-1]

        if self.axis == 'x':
            length = thermo_df['Lx'].iloc[0]
        elif self.axis == 'y':
            length = thermo_df['Ly'].iloc[0]
        elif self.axis == 'z':
            length = thermo_df['Lz'].iloc[0]

        self.Tgrad_data = self.get_Tgrad_twoway(
                                self.tprof_file, length, init=init, last=last,
                                threshold_r2=self.threshold_r2, threshold_r2_i=self.threshold_r2_i, threshold_rate=self.threshold_rate,
                                tschunk=tschunk, printout=printout, save=save_dir
                            )
        self.Qgrad_data = self.calc_heatflux_mp(thermo_df, init=init, last=last, printout=printout, save=save_dir)
        self.TC = self.Qgrad_data['Qgrad']/self.Tgrad_data['Tgrad']

        prop_data = {'thermal_conductivity': self.TC}

        T_SD = self.Tgrad_data['T_SD']
        Tgrad_data = dict(**self.Tgrad_data)
        del Tgrad_data['T_SD']
        for i, sd in enumerate(T_SD):
            Tgrad_data['T_SD_%i' % i] = sd
        conv_data = dict(**Tgrad_data, **self.Qgrad_data)

        if decomp:
            self.TCdecomp_data, self.Jdecomp_data = self.analyze_decomp(tc=self.TC)
            prop_data.update(self.TCdecomp_data)

        self.prop_df = pd.DataFrame(prop_data, index=[0])
        self.conv_df = pd.DataFrame(conv_data, index=[0])

        if save:
            self.prop_df.to_csv(os.path.join(save_dir, 'tc_prop_data.csv'))
            self.conv_df.to_csv(os.path.join(save_dir, 'tc_conv_data.csv'))

        return self.TC


    def get_Tgrad_twoway(self, temp_file, length, threshold_r2=0.98, threshold_r2_i=0.95, threshold_rate=0.667,
                         printout=False, save=None, init=500, last=None, tschunk=1):
        """
        preset.tc.NEMD_MP.get_Tgrad_twoway

        Args:
            temp_file: Chunk averaged data of temperature
            length: Cell length along heat flux (float, angstrom)
        """   
        tgrads = []
        nchunk = 0
        density_flag = False

        df = self.read_ave(temp_file)
        if 'density/mass' in df.columns:
            density_flag = True
        
        for index1 in df.index.unique(level=0):
            data = df.loc[index1].to_numpy(dtype=np.float64)
            nchunk = len(data)
            coord = df.loc[index1].iloc[-1]['Coord1']*2-df.loc[index1].iloc[-2]['Coord1']
            Ncount = df.loc[index1].iloc[0]['Ncount']
            temp = df.loc[index1].iloc[0]['temp']
            if density_flag:
                density = df.loc[index1].iloc[0]['density/mass']
                data = np.vstack((data, [coord, Ncount, temp, density]))
            else:
                data = np.vstack((data, [coord, Ncount, temp]))
            tgrads.append(data)
        
        tgrads = np.array(tgrads)
        center = int(nchunk/2)
        grad_conv = length * 1e-10

        chunk_l_i = tschunk
        chunk_l_l = center-tschunk+1
        chunk_r_i = center+tschunk
        chunk_r_l = -tschunk if tschunk > 0 else None
        tgrads_mean = np.mean(tgrads[init:last, :, 2], axis=0)
        tgrads_sd = np.std(tgrads[init:last, :, 2], axis=0, ddof=1)

        OK = False

        tmax = np.max(tgrads_mean)
        tmin = np.min(tgrads_mean)
        coord_l = tgrads[0, chunk_l_i:chunk_l_l, 0]
        coord_r = tgrads[0, chunk_r_i:chunk_r_l, 0]
            
        res1=np.polyfit(coord_l, tgrads_mean[chunk_l_i:chunk_l_l], 1)
        res2=np.polyfit(coord_r, tgrads_mean[chunk_r_i:chunk_r_l], 1)
        y1 = np.poly1d(res1)(coord_l)
        y2 = np.poly1d(res2)(coord_r)
        grad1, k1, r1, p1, se1 = stats.linregress(coord_l, tgrads_mean[chunk_l_i:chunk_l_l])
        grad2, k2, r2, p2, se2 = stats.linregress(coord_r, tgrads_mean[chunk_r_i:chunk_r_l])
        grad1 = abs(grad1 / grad_conv)  # K/(coord1) -> k/m
        grad2 = abs(grad2 / grad_conv)  # K/(coord1) -> k/m
        grad_ave = (grad1 + grad2)/2
        r21 = r1**2
        r22 = r2**2
        se1 = se1 / grad_conv  # K/(coord1) -> k/m
        se2 = se2 / grad_conv  # K/(coord1) -> k/m
        se_ave = (se1 + se2)/2

        if r21 >= threshold_r2 and r22 >= threshold_r2:
            OK = True

        grad_data = {'Tgrad_check':OK, 'Tgrad':grad_ave, 'Tgrad_ave':grad_ave, 'Tgrad_SE_ave':se_ave,
                     'T_max':tmax, 'T_min':tmin, 'T_SD':tgrads_sd, 'T_SD_max':np.max(tgrads_sd),
                     'Tgrad1':grad1, 'Tgrad1_r2':r21, 'Tgrad1_p':p1, 'Tgrad1_SE':se1,
                     'Tgrad2':grad2, 'Tgrad2_r2':r22, 'Tgrad2_p':p2, 'Tgrad2_SE':se2}

        if printout or save:
            color = 'blue' if OK else 'red'

            fig, ax = pp.subplots(figsize=(6, 6))
            pp.scatter(tgrads[0, :, 0]*length, tgrads_mean, c=color)
            pp.plot(coord_l*length, y1, c=color)
            pp.plot(coord_r*length, y2, c=color)
            pp.xlim(0, tgrads[0, -1, 0]*length)
            pp.title('T grad mean')
            pp.xlabel('Length [Angstrom]')
            pp.ylabel('Temperature [K]')
            output = "T_max = %f    T_min = %f\n" % (tmax, tmin)
            if OK: output += 'OK: grad ave.(K/m) = %e,   se = %e\n' % (grad_ave, se_ave)
            else: output += 'NG: grad ave.(K/m) = %e,   se = %e\n' % (grad_ave, se_ave)
            output += "Left region: grad(K/m) = %e,   r2 = %f,   p = %e,   se = %e\n" %\
                        (grad1, r21, p1, se1)
            output += "Right region: grad(K/m) = %e,   r2 = %f,   p = %e,   se = %e\n" %\
                        (grad2, r22, p2, se2)
            output += 'Temp SD: ' + ','.join([str(x) for x in tgrads_sd]) + '\n'
            
            if printout:
                pp.show()
                print(output)

            if save:
                if not os.path.exists(save):
                    os.makedirs(save)
                fig.savefig(os.path.join(save, 'Tgrad_mean.png'))
                with open(os.path.join(save, 'Tgrad_mean.txt'), mode='w') as f:
                    f.write(output)
            
            pp.close(fig)


        grad_data_i = []
        n_data = len(tgrads[init:last, 0, 2])
        for i in range(n_data):
            grad1, k1, r1, p1, se1 = stats.linregress(coord_l, tgrads[init+i, chunk_l_i:chunk_l_l, 2])
            grad2, k2, r2, p2, se2 = stats.linregress(coord_r, tgrads[init+i, chunk_r_i:chunk_r_l, 2])
            grad1 = abs(grad1 / grad_conv)  # K/(coord1) -> k/m
            grad2 = abs(grad2 / grad_conv)  # K/(coord1) -> k/m
            grad_ave = (grad1 + grad2)/2
            r21 = r1**2
            r22 = r2**2
            if r21 >= threshold_r2_i and r22 >= threshold_r2_i:
                OK = True
            grad_data_i.append([OK, grad_ave, grad1, r21, p1, se1, grad2, r22, p2, se2])
        grad_data_i_df = pd.DataFrame(grad_data_i,
                            columns=['grad_check', 'grad_ave', 'grad1', 'r21', 'p1', 'se1', 'grad2', 'r22', 'p2', 'se2'])
        grad_data['Tgrad_rate'] = grad_data_i_df['grad_check'].sum() / n_data
        if grad_data['Tgrad_rate'] < threshold_rate:
            grad_data['Tgrad_check'] = False

        return grad_data


    def calc_heatflux_mp(self, thermo_df, init=0, last=None, heatflux='v_heatflux', printout=False, save=None):

        grad, k, r, p, se = stats.linregress(thermo_df['Time'].iloc[init:last]*1e-15, thermo_df[heatflux].iloc[init:last])
        r2 = r**2
        grad_data = {'Qgrad':grad, 'Qgrad_k':k, 'Qgrad_r2':r2, 'Qgrad_p':p, 'Qgrad_SE':se}

        if printout or save:
            res=np.polyfit(thermo_df['Time'].iloc[init:last]*1e-3, thermo_df[heatflux].iloc[init:last], 1)
            y = np.poly1d(res)(thermo_df['Time'].iloc[init:last]*1e-3)

            fig, ax = pp.subplots(figsize=(6, 6))
            pp.scatter(thermo_df['Time'].iloc[init:last]*1e-3, thermo_df[heatflux].iloc[init:last])
            pp.plot(thermo_df['Time'].iloc[init:last]*1e-3, y)
            pp.title('dQ/dT')
            pp.xlim(thermo_df['Time'].iloc[init:last].values[0]*1e-3, thermo_df['Time'].iloc[init:last].values[-1]*1e-3)
            pp.xlabel('Time [ps]')
            pp.ylabel('Q [Ws/m^2]')
            output = 'Q grad. [W/m^2] = %e,   se = %e,   r2 = %f,   p = %e\n' % (grad, se, r2, p)
            
            if printout:
                pp.show()
                print(output)

            if save:
                if not os.path.exists(save):
                    os.makedirs(save)
                fig.savefig(os.path.join(save, 'Qgrad.png'))
                with open(os.path.join(save, 'Qgrad.txt'), mode='w') as f:
                    f.write(output)
            
            pp.close(fig)

        return grad_data


    def analyze_decomp(self, tc=1.0, vol=None):

        df_l = self.read_ave(self.lJprof_file)
        df_r = self.read_ave(self.rJprof_file)

        if vol is None:
            df_T = self.read_ave(self.tprof_file)
            nslab = len(df_T.iloc[0].to_numpy(dtype=np.float64))
            vol = self.dfs[-1]['Volume'].to_numpy(dtype=np.float64)[-1] * ((nslab/2 - 1)/nslab) * const.ang2m**3

        conv_J = const.cal2j*1e3/const.NA * const.m2ang * 1e15  # [(kcal/mol) ang / fs] -> [J m/s] = [W m]

        if len(df_l.iloc[0, :]) == 9:
            all_l_tmp = df_l.sum(axis=1).to_numpy()
            all_r_tmp = df_r.sum(axis=1).to_numpy()
            TC_values = ((df_l.sum(axis=0)/all_l_tmp.sum(axis=0)).to_numpy() + (df_r.sum(axis=0)/all_r_tmp.sum(axis=0)).to_numpy())/2*tc
            TC_keys = ['TC_ke', 'TC_pe', 'TC_pair', 'TC_bond', 'TC_angle', 'TC_dihed', 'TC_improper', 'TC_kspace', 'TC_fix']
            J_values = (df_l.mean(axis=0).to_numpy() + df_r.mean(axis=0).to_numpy())*conv_J / 2 / vol
            J_keys = ['J_ke', 'J_pe', 'J_pair', 'J_bond', 'J_angle', 'J_dihed', 'J_improper', 'J_kspace', 'J_fix']

        elif len(df_l.iloc[0, :]) == 10:
            TC_values = ((df_l.sum(axis=0)/df_l.iloc[:, 0].sum(axis=0)).to_numpy() + (df_r.sum(axis=0)/df_r.iloc[:, 0].sum(axis=0)).to_numpy())/2*tc
            TC_keys = ['TC_all', 'TC_ke', 'TC_pe', 'TC_pair', 'TC_bond', 'TC_angle', 'TC_dihed', 'TC_improper', 'TC_kspace', 'TC_fix']
            J_values = (df_l.mean(axis=0).to_numpy() + df_r.mean(axis=0).to_numpy())*conv_J / 2 / vol
            J_keys = ['J_all', 'J_ke', 'J_pe', 'J_pair', 'J_bond', 'J_angle', 'J_dihed', 'J_improper', 'J_kspace', 'J_fix']

        elif len(df_l.iloc[0, :]) == 11:
            all_l_tmp = df_l.iloc[:, [0, 1, 2, 5, 6, 7, 8, 9, 10]].sum(axis=1).to_numpy()
            all_r_tmp = df_r.iloc[:, [0, 1, 2, 5, 6, 7, 8, 9, 10]].sum(axis=1).to_numpy()
            TC_values = ((df_l.sum(axis=0)/all_l_tmp.sum(axis=0)).to_numpy() + (df_r.sum(axis=0)/all_r_tmp.sum(axis=0)).to_numpy())/2*tc
            TC_keys = ['TC_ke', 'TC_pe', 'TC_pair', 'TC_pair_inter', 'TC_pair_intra',
                    'TC_bond', 'TC_angle', 'TC_dihed', 'TC_improper', 'TC_kspace', 'TC_fix']
            J_values = (df_l.mean(axis=0).to_numpy() + df_r.mean(axis=0).to_numpy())*conv_J / 2 / vol
            J_keys = ['J_ke', 'J_pe', 'J_pair', 'J_pair_inter', 'J_pair_intra',
                    'J_bond', 'J_angle', 'J_dihed', 'J_improper', 'J_kspace', 'J_fix']

        elif len(df_l.iloc[0, :]) == 12:
            TC_values = ((df_l.sum(axis=0)/df_l.iloc[:, 0].sum(axis=0)).to_numpy() + (df_r.sum(axis=0)/df_r.iloc[:, 0].sum(axis=0)).to_numpy())/2*tc
            TC_keys = ['TC_all', 'TC_ke', 'TC_pe', 'TC_pair', 'TC_pair_inter', 'TC_pair_intra',
                    'TC_bond', 'TC_angle', 'TC_dihed', 'TC_improper', 'TC_kspace', 'TC_fix']
            J_values = (df_l.mean(axis=0).to_numpy() + df_r.mean(axis=0).to_numpy())*conv_J / 2 / vol
            J_keys = ['J_all', 'J_ke', 'J_pe', 'J_pair', 'J_pair_inter', 'J_pair_intra',
                    'J_bond', 'J_angle', 'J_dihed', 'J_improper', 'J_kspace', 'J_fix']

        else:
            utils.radon_print('Can not read the format of decomposition analysis in thermal conductivity.', level=2)

        TCdecomp = dict(zip(TC_keys, TC_values))
        Jdecomp = dict(zip(J_keys, J_values))

        return TCdecomp, Jdecomp


class NEMD_MP_Additional(NEMD_MP):
    def exec(self, confId=0, step=5000000, time_step=0.2, temp=300.0,
             decomp=False, step_decomp=500000, decomp_intermol=False,
             omp=1, mpi=1, gpu=0, intel='auto', opt='auto', **kwargs):
        """
        preset.tc.NEMD_MP_Additional.exec

        Preset of thermal conductivity calculation by kinetic energy exchanging NEMD, a.k.a. reverse NEMD (RNEMD).
        LAMMPS only

        Args:
            mol: RDKit Mol object
        
        Optional args:
            confId: Target conformer ID (int)
            step: Number of step (int)
            time_step: Timestep (float)
            axis: Target axis (str)
            temp: Avarage temperature (float, K)
            decomp: Do decomposition analysis of heat flux (boolean)
            step_decomp: Number of step in decomposition analysis (int)
            solver_path: File path of LAMMPS (str) 
            work_dir: Path of work directory (str)
            omp: Number of threads of OpenMP (int)
            mpi: Number of MPI process (int)
            gpu: Number of GPU (int)

        Returns:
            RDKit Mol object
        """
        lmp = lammps.LAMMPS(work_dir=self.work_dir, solver_path=self.solver_path)
        self.make_lammps_input(confId=confId, step=step, time_step=time_step, temp=temp,
            decomp=decomp, step_decomp=step_decomp, decomp_intermol=decomp_intermol)

        dt1 = datetime.datetime.now()
        utils.radon_print('Additional thermal conductive simulation (kinetic energy exchanging NEMD) by LAMMPS is running...', level=1)

        intel = 'off' if decomp else intel
        cp = lmp.exec(input_file=self.in_file, omp=omp, mpi=mpi, gpu=gpu, intel=intel, opt=opt)
        if cp.returncode != 0 and (
                    (self.last_str is not None and not os.path.exists(os.path.join(self.work_dir, self.last_str)))
                    or (self.last_data is not None and not os.path.exists(os.path.join(self.work_dir, self.last_data)))
                ):
            utils.radon_print('Error termination of %s' % (lmp.get_name), level=3)
            return None, None

        self.uwstr, self.wstr, self.cell, self.vel, _ = lmp.read_traj_simple(os.path.join(self.work_dir, self.last_str))

        for i in range(self.mol.GetNumAtoms()):
            self.mol.GetConformer(0).SetAtomPosition(i, Geom.Point3D(self.uwstr[i, 0], self.uwstr[i, 1], self.uwstr[i, 2]))
            self.mol.GetAtomWithIdx(i).SetDoubleProp('vx', self.vel[i, 0])
            self.mol.GetAtomWithIdx(i).SetDoubleProp('vy', self.vel[i, 1])
            self.mol.GetAtomWithIdx(i).SetDoubleProp('vz', self.vel[i, 2])

        setattr(self.mol, 'cell', utils.Cell(self.cell[0, 1], self.cell[0, 0], self.cell[1, 1], self.cell[1, 0], self.cell[2, 1], self.cell[2, 0]))
        self.mol = calc.mol_trans_in_cell(self.mol)
        utils.MolToPDBFile(self.mol, os.path.join(self.work_dir, self.pdb_file))
        utils.MolToJSON(self.mol, os.path.join(self.save_dir, self.json_file))
        utils.pickle_dump(self.mol, os.path.join(self.save_dir, self.pickle_file))

        dt2 = datetime.datetime.now()
        utils.radon_print('Complete additional thermal conductive simulation (kinetic energy exchanging NEMD). Elapsed time = %s' % str(dt2-dt1), level=1)

        return self.mol


    def make_lammps_input(self, confId=0, step=5000000, time_step=0.2, temp=300.0,
                            decomp=False, step_decomp=500000, decomp_intermol=False, **kwargs):

        lmp = lammps.LAMMPS(work_dir=self.work_dir, solver_path=self.solver_path)
        lmp.make_dat(self.mol, file_name=self.dat_file, confId=confId)

        # Make input file
        in_strings  = 'variable        axis    string %s\n' % (self.axis)
        in_strings += 'variable        slab    equal  %i\n' % (kwargs.get('slab', 20))
        in_strings += 'variable        exchg   equal  %i\n' % (kwargs.get('exchg', 1000))
        in_strings += 'variable        Nevery  equal  %i\n' % (kwargs.get('Nevery', 1))
        in_strings += 'variable        TimeSt  equal  %f\n' % (time_step)
        in_strings += 'variable        NStep   equal  %i\n' % (step)
        in_strings += 'variable        NStepd  equal  %i\n' % (step_decomp)
        in_strings += 'variable        Ttemp   equal  %f\n' % (temp)
        in_strings += 'variable        dataf   string %s\n' % (self.dat_file)
        in_strings += '##########################################################\n'
        in_strings += '## Setting variables\n'
        in_strings += '##########################################################\n'
        in_strings += 'variable        logf    string %s\n' % (self.log_file)
        in_strings += 'variable        dumpf   string %s\n' % (self.dump_file)
        in_strings += 'variable        xtcf    string %s\n' % (self.xtc_file)
        in_strings += 'variable        rstf1   string %s\n' % (self.rst1_file)
        in_strings += 'variable        rstf2   string %s\n' % (self.rst2_file)
        in_strings += 'variable        Tprof   string %s\n' % (self.tprof_file)
        in_strings += 'variable        lJprof  string %s\n' % (self.lJprof_file)
        in_strings += 'variable        rJprof  string %s\n' % (self.rJprof_file)
        in_strings += 'variable        ldumpf  string %s\n' % (self.last_str)
        in_strings += 'variable        ldataf  string %s\n' % (self.last_data)
        in_strings += 'variable        pairst  string %s\n' % (self.pair_style)
        in_strings += 'variable        cutoff1 string %s\n' % (self.cutoff_in)
        in_strings += 'variable        cutoff2 string %s\n' % (self.cutoff_out)
        in_strings += 'variable        bondst  string %s\n' % (self.bond_style)
        in_strings += 'variable        anglest string %s\n' % (self.angle_style)
        in_strings += 'variable        dihedst string %s\n' % (self.dihedral_style)
        in_strings += 'variable        improst string %s\n' % (self.improper_style)
        in_strings += '##########################################################\n'

        in_strings += """
log             ${logf} append

units           real
atom_style      full
boundary        p p p

bond_style      ${bondst}  
angle_style     ${anglest}
dihedral_style  ${dihedst}
improper_style  ${improst}

pair_style      ${pairst} ${cutoff1} ${cutoff2}
pair_modify     mix arithmetic
special_bonds   amber
neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes
kspace_style    pppm 1e-6

read_data       ${dataf}

thermo_modify   flush yes
thermo          1000



##########################################################
## Preparation
##########################################################
variable        NA     equal 6.02214076*1.0e23
variable        kcal2j equal 4.184*1000
variable        ang2m  equal 1.0e-10
variable        fs2s   equal 1.0e-15

if "${axis} == x" then &
  "variable     ahi   equal  xhi" &
  "variable     alo   equal  xlo" &
  "variable     Jarea equal  ly*lz" &
  "variable     idx   equal  1" &
elif "${axis} == y" &
  "variable     ahi   equal  yhi" &
  "variable     alo   equal  ylo" &
  "variable     Jarea equal  lx*lz" &
  "variable     idx   equal  2" &
elif "${axis} == z" &
  "variable     ahi   equal  zhi" &
  "variable     alo   equal  zlo" &
  "variable     Jarea equal  lx*ly" &
  "variable     idx   equal  3"

variable        Nfreq   equal  ${exchg}/${Nevery}       # Number of data points to compute temperature during exchange interval 
variable        invslab equal  1/${slab}
variable        width   equal  (${ahi}-${alo})/${slab}
variable        llo     equal  ${alo}+${width}*1.0
variable        lhi     equal  ${alo}+(${slab}/2)*${width}
variable        rlo     equal  ${alo}+(1+${slab}/2)*${width}
variable        rhi     equal  ${ahi}

if "${axis} == x" then &
  "region       lhalf   block    ${llo}  ${lhi}  INF  INF  INF  INF  units box" &
  "region       rhalf   block    ${rlo}  ${rhi}  INF  INF  INF  INF  units box" &
elif "${axis} == y" &
  "region       lhalf   block    INF  INF  ${llo}  ${lhi}  INF  INF  units box" &
  "region       rhalf   block    INF  INF  ${rlo}  ${rhi}  INF  INF  units box" &
elif "${axis} == z" &
  "region       lhalf   block    INF  INF  INF  INF  ${llo}  ${lhi}  units box" &
  "region       rhalf   block    INF  INF  INF  INF  ${rlo}  ${rhi}  units box"
##########################################################



##########################################################
## NEMD with kinetic energy exchange (RNEMD)
##########################################################
timestep        ${TimeSt}
fix             NVE all nve
fix             mp  all thermal/conductivity ${exchg} ${axis} ${slab}

# Generate temperature profile of layers
compute         layers all chunk/atom bin/1d ${axis} lower ${invslab} units reduced
fix             2 all ave/chunk ${Nevery} ${Nfreq} ${exchg} layers temp file ${Tprof} norm sample

# Output
dump            1 all custom 1000 ${dumpf} id type mol xs ys zs ix iy iz
dump            2 all xtc 1000 ${xtcf}
dump_modify     2 unwrap yes
restart         100000 ${rstf1} ${rstf2}

variable        heatflux   equal   (f_mp*${kcal2j}/${NA})/(2*${Jarea}*${ang2m}*${ang2m})   # J/m^2 = Ws/m^2
thermo_style    custom step time temp press enthalpy etotal ke pe ebond eangle edihed eimp evdwl ecoul elong etail vol lx ly lz density pxx pyy pzz pxy pxz pyz f_mp v_heatflux
thermo_modify   flush yes
thermo          ${exchg}

run             ${NStep}


"""

        if decomp:
            in_strings += """
##########################################################
## Component decomposition of heat flux
##########################################################
# heat flux preparation
compute         KE        all   ke/atom
compute         PE        all   pe/atom

#compute         Stress    all   centroid/stress/atom NULL virial
compute         Spair     all   stress/atom NULL pair
compute         Sbond     all   stress/atom NULL bond
compute         Sangle    all   centroid/stress/atom NULL angle
compute         Sdihed    all   centroid/stress/atom NULL dihedral
compute         Simpro    all   centroid/stress/atom NULL improper
compute         Skspac    all   stress/atom NULL kspace
compute         Sfix      all   stress/atom NULL fix
"""

            if decomp_intermol:
                in_strings += """
compute         Spairer   all   stress/atom NULL interpair
compute         Spairra   all   stress/atom NULL intrapair
"""

            in_strings += """
# Generate empty vector
group           empty     type      99999
compute         KENULL    empty ke/atom
compute         PENULL    empty pe/atom improper
compute         STNULL    empty stress/atom NULL improper

########################   Cell half-left   ########################
###  |//|  |  |  |  |**|  |  |  |  |  ###   |//| cold slab
###  |//|  |  |  |  |**|  |  |  |  |  ###   |**| hot slab
###  |//|  |  |  |  |**|  |  |  |  |  ###
###  |//|  |  |  |  |**|  |  |  |  |  ###
###  |//|  |  |  |  |**|  |  |  |  |  ###
###      <--------->                  ###
### heat flux decomposition of this reagion
#####################################################################

# left cell information
group           halfL     dynamic  all         region     lhalf     every  ${Nevery}  # Nevery=ave/time Nevery

# left energy Flux  JE
#compute         lFlux     halfL    heat/flux   KE         PE        Stress

#1st term   eivi
compute         lF1ke     halfL    heat/flux   KE         PENULL    STNULL
compute         lF1pe     halfL    heat/flux   KENULL     PE        STNULL

#2nd term   Sivi
#compute         lSivi     halfL    heat/flux   KENULL     PENULL    Stress
compute         lFpair    halfL    heat/flux   KENULL     PENULL    Spair
compute         lFbond    halfL    heat/flux   KENULL     PENULL    Sbond
compute         lFangle   halfL    heat/flux   KENULL     PENULL    Sangle
compute         lFdihed   halfL    heat/flux   KENULL     PENULL    Sdihed
compute         lFimpro   halfL    heat/flux   KENULL     PENULL    Simpro
compute         lFkspac   halfL    heat/flux   KENULL     PENULL    Skspac
compute         lFfix     halfL    heat/flux   KENULL     PENULL    Sfix
"""

            if decomp_intermol:
                in_strings += """
compute         lFpairer  halfL    heat/flux   KENULL     PENULL    Spairer
compute         lFpairra  halfL    heat/flux   KENULL     PENULL    Spairra
fix             20 halfL ave/time ${Nevery} ${Nfreq} ${exchg}  c_lF1ke[${idx}] c_lF1pe[${idx}] c_lFpair[${idx}]  c_lFpairer[${idx}]  c_lFpairra[${idx}]  c_lFbond[${idx}]  c_lFangle[${idx}]  c_lFdihed[${idx}]  c_lFimpro[${idx}]  c_lFkspac[${idx}]  c_lFfix[${idx}]  file ${lJprof}
"""
            else:
                in_strings += """
fix             20 halfL ave/time ${Nevery} ${Nfreq} ${exchg}  c_lF1ke[${idx}] c_lF1pe[${idx}] c_lFpair[${idx}]  c_lFbond[${idx}]  c_lFangle[${idx}]  c_lFdihed[${idx}]  c_lFimpro[${idx}]  c_lFkspac[${idx}]  c_lFfix[${idx}]  file ${lJprof}
"""

            in_strings += """

########################   Cell half-right   #######################
###  |//|  |  |  |  |**|  |  |  |  |  ###   |//| cold slab
###  |//|  |  |  |  |**|  |  |  |  |  ###   |**| hot slab
###  |//|  |  |  |  |**|  |  |  |  |  ###
###  |//|  |  |  |  |**|  |  |  |  |  ###
###  |//|  |  |  |  |**|  |  |  |  |  ###
###                     <--------->   ###
###        heat flux decomposition of this reagion
#####################################################################

# right cell information
group           halfR     dynamic  all         region     rhalf     every  ${Nevery}

# right energy Flux  JE
#compute         rFlux     halfR    heat/flux   KE         PE        Stress

#1st term   eivi
compute         rF1ke     halfR    heat/flux   KE         PENULL    STNULL
compute         rF1pe     halfR    heat/flux   KENULL     PE        STNULL

#2nd term   Sivi
#compute         rSivi     halfR    heat/flux   KENULL     PENULL    Stress
compute         rFpair    halfR    heat/flux   KENULL     PENULL    Spair
compute         rFbond    halfR    heat/flux   KENULL     PENULL    Sbond
compute         rFangle   halfR    heat/flux   KENULL     PENULL    Sangle
compute         rFdihed   halfR    heat/flux   KENULL     PENULL    Sdihed
compute         rFimpro   halfR    heat/flux   KENULL     PENULL    Simpro
compute         rFkspac   halfR    heat/flux   KENULL     PENULL    Skspac
compute         rFfix     halfR    heat/flux   KENULL     PENULL    Sfix
"""

            if decomp_intermol:
                in_strings += """
compute         rFpairer  halfR    heat/flux   KENULL     PENULL    Spairer
compute         rFpairra  halfR    heat/flux   KENULL     PENULL    Spairra
fix             30 halfR ave/time ${Nevery} ${Nfreq} ${exchg}  c_rF1ke[${idx}] c_rF1pe[${idx}] c_rFpair[${idx}]  c_rFpairer[${idx}]  c_rFpairra[${idx}]  c_rFbond[${idx}]  c_rFangle[${idx}]  c_rFdihed[${idx}]  c_rFimpro[${idx}]  c_rFkspac[${idx}]  c_rFfix[${idx}]  file ${rJprof}
"""
            else:
                in_strings += """
fix             30 halfR ave/time ${Nevery} ${Nfreq} ${exchg}  c_rF1ke[${idx}] c_rF1pe[${idx}] c_rFpair[${idx}]  c_rFbond[${idx}]  c_rFangle[${idx}]  c_rFdihed[${idx}]  c_rFimpro[${idx}]  c_rFkspac[${idx}]  c_rFfix[${idx}]  file ${rJprof}
"""

            in_strings += """

##########################################################
## RNEMD with kinetic energy exchange in decomposition
##########################################################
thermo_style    custom step time temp press enthalpy etotal ke pe ebond eangle edihed eimp evdwl ecoul elong etail vol lx ly lz density pxx pyy pzz pxy pxz pyz f_mp v_heatflux
thermo_modify   flush yes
thermo          ${exchg}

run             ${NStepd}


"""

        in_strings += """
write_dump      all custom ${ldumpf} id x y z xu yu zu vx vy vz fx fy fz modify sort id
write_data      ${ldataf}
quit
"""

        with open(os.path.join(self.work_dir, self.in_file), 'w') as fh:
            fh.write(in_strings)
            fh.flush()
            if hasattr(os, 'fdatasync'):
                os.fdatasync(fh.fileno())
            else:
                os.fsync(fh.fileno())

        utils.MolToPDBFile(self.mol, os.path.join(self.work_dir, self.pdb_file))

        return True




class NEMD_Langevin(preset.Preset):
    def __init__(self, mol, axis='x', prefix='', work_dir=None, save_dir=None, solver_path=None, **kwargs):
        super().__init__(mol, prefix=prefix, work_dir=work_dir, save_dir=save_dir, solver_path=solver_path, **kwargs)
        self.axis = axis
        self.dat_file = kwargs.get('dat_file', '%snemd_TC-Langevin_%s.data' % (prefix, axis))
        self.pdb_file = kwargs.get('pdb_file', '%snemd_TC-Langevin_%s.pdb' % (prefix, axis))
        self.in_file = kwargs.get('in_file', '%snemd_TC-Langevin_%s.in' % (prefix, axis))
        self.log_file = kwargs.get('log_file', '%snemd_TC-Langevin_%s.log' % (prefix, axis))
        self.dump_file = kwargs.get('dump_file', '%snemd_TC-Langevin_%s.dump' % (prefix, axis))
        self.xtc_file = kwargs.get('xtc_file', '%snemd_TC-Langevin_%s.xtc' % (prefix, axis))
        self.rst1_file = kwargs.get('rst1_file', '%snemd_TC-Langevin_%s_1.rst' % (prefix, axis))
        self.rst2_file = kwargs.get('rst2_file', '%snemd_TC-Langevin_%s_2.rst' % (prefix, axis))
        self.tprof_file = kwargs.get('tprof_file', '%sslabtemp_%s.profile' % (prefix, axis))
        self.Jprof_file = kwargs.get('Jprof_file', '%sheatflux_%s.profile' % (prefix, axis))
        self.JDprof_file = kwargs.get('JDprof_file', '%sheatflux_decomp_%s.profile' % (prefix, axis))        
        self.last_str = kwargs.get('last_str', '%snemd_TC-Langevin_%s_last.dump' % (prefix, axis))
        self.last_data = kwargs.get('last_data', '%snemd_TC-Langevin_%s_last.data' % (prefix, axis))
        self.pickle_file = kwargs.get('pickle_file', '%snemd_TC-Langevin_%s_last.pickle' % (prefix, axis))
        self.json_file = kwargs.get('json_file', '%snemd_TC-Langevin_%s_last.json' % (prefix, axis))


    def exec(self, confId=0, step=10000000, time_step=0.2, h_temp=320.0, l_temp=280.0,
             decomp=False, step_decomp=500000, decomp_intermol=False,
             omp=1, mpi=1, gpu=0, intel='auto', opt='auto', **kwargs):
        """
        preset.tc.NEMD_Langevin.exec

        Preset of thermal conductivity calculation by Langevin thermostat NEMD.
        LAMMPS only

        Args:
            mol: RDKit Mol object
        
        Optional args:
            confId: Target conformer ID (int)
            step: Number of step (int)
            time_step: Timestep (float)
            axis: Target axis (str)
            h_temp: Higher temperature (float, K)
            l_temp: Lower temperature (float, K)
            decomp: Do decomposition analysis of heat flux (boolean)
            step_decomp: Number of step in decomposition analysis (int)
            solver_path: File path of LAMMPS (str) 
            work_dir: Path of work directory (str)
            omp: Number of threads of OpenMP (int)
            mpi: Number of MPI process (int)
            gpu: Number of GPU (int)

        Returns:
            RDKit Mol object
        """

        rep = kwargs.get('rep', 3)
        repo = kwargs.get('rep_other', 1)
        lmp = lammps.LAMMPS(work_dir=self.work_dir, solver_path=self.solver_path)

        self.make_lammps_input(confId=confId, step=step, time_step=time_step, h_temp=h_temp, l_temp=l_temp, rep=rep, rep_other=repo,
            decomp=decomp, step_decomp=step_decomp, decomp_intermol=decomp_intermol)

        dt1 = datetime.datetime.now()
        utils.radon_print('Thermal conductive simulation (Langevin thermostat NEMD) by LAMMPS is running...', level=1)

        intel = 'off' if decomp else intel
        cp = lmp.exec(input_file=self.in_file, omp=omp, mpi=mpi, gpu=gpu, intel=intel, opt=opt)
        if cp.returncode != 0 and (
                    (self.last_str is not None and not os.path.exists(os.path.join(self.work_dir, self.last_str)))
                    or (self.last_data is not None and not os.path.exists(os.path.join(self.work_dir, self.last_data)))
                ):
            utils.radon_print('Error termination of %s' % (lmp.get_name), level=3)
            return None

        if self.axis == 'x':
            self.mol = poly.super_cell(self.mol, x=rep, y=repo, z=repo, confId=confId)
        elif self.axis == 'y':
            self.mol = poly.super_cell(self.mol, x=repo, y=rep, z=repo, confId=confId)
        elif self.axis == 'z':
            self.mol = poly.super_cell(self.mol, x=repo, y=repo, z=rep, confId=confId)

        self.uwstr, self.wstr, self.cell, self.vel, _ = lmp.read_traj_simple(os.path.join(self.work_dir, self.last_str))

        for i in range(self.mol.GetNumAtoms()):
            self.mol.GetConformer(0).SetAtomPosition(i, Geom.Point3D(self.uwstr[i, 0], self.uwstr[i, 1], self.uwstr[i, 2]))
            self.mol.GetAtomWithIdx(i).SetDoubleProp('vx', self.vel[i, 0])
            self.mol.GetAtomWithIdx(i).SetDoubleProp('vy', self.vel[i, 1])
            self.mol.GetAtomWithIdx(i).SetDoubleProp('vz', self.vel[i, 2])

        setattr(self.mol, 'cell', utils.Cell(self.cell[0, 1], self.cell[0, 0], self.cell[1, 1], self.cell[1, 0], self.cell[2, 1], self.cell[2, 0]))
        self.mol = calc.mol_trans_in_cell(self.mol)
        utils.MolToPDBFile(self.mol, os.path.join(self.work_dir, self.pdb_file))
        utils.MolToJSON(self.mol, os.path.join(self.save_dir, self.json_file))
        utils.pickle_dump(self.mol, os.path.join(self.save_dir, self.pickle_file))

        dt2 = datetime.datetime.now()
        utils.radon_print('Complete thermal conductive simulation (Langevin thermostat NEMD). Elapsed time = %s' % str(dt2-dt1), level=1)

        return self.mol


    def make_lammps_input(self, confId=0, step=5000000, time_step=0.2, h_temp=320.0, l_temp=280.0, rep=3, rep_other=1,
                          decomp=False, step_decomp=500000, decomp_intermol=False, **kwargs):
        
        seed1 = np.random.randint(1000, 999999)
        seed2 = np.random.randint(1000, 999999)

        lmp = lammps.LAMMPS(work_dir=self.work_dir, solver_path=self.solver_path)
        lmp.make_dat(self.mol, file_name=self.dat_file, confId=confId)

        # Make input file
        in_strings  = 'variable        axis    string %s\n' % (self.axis)
        in_strings += 'variable        rep     equal  %i\n' % (rep)
        in_strings += 'variable        repo    equal  %i\n' % (rep_other)
        in_strings += 'variable        slab    equal  %i\n' % (kwargs.get('slab', 20))
        in_strings += 'variable        avetime equal  %i\n' % (kwargs.get('avetime', 1000))
        in_strings += 'variable        Nevery  equal  %i\n' % (kwargs.get('Nevery', 1))
        in_strings += 'variable        TimeSt  equal  %f\n' % (time_step)
        in_strings += 'variable        NStep   equal  %i\n' % (step)
        in_strings += 'variable        NStepd  equal  %i\n' % (step_decomp)
        in_strings += 'variable        Htemp   equal  %f\n' % (h_temp)
        in_strings += 'variable        Ltemp   equal  %f\n' % (l_temp)
        in_strings += 'variable        dataf   string %s\n' % (self.dat_file)
        in_strings += 'variable        seed1   equal  %i\n' % (seed1)
        in_strings += 'variable        seed2   equal  %i\n' % (seed2)
        in_strings += '##########################################################\n'
        in_strings += '## Setting variables\n'
        in_strings += '##########################################################\n'
        in_strings += 'variable        logf    string %s\n' % (self.log_file)
        in_strings += 'variable        dumpf   string %s\n' % (self.dump_file)
        in_strings += 'variable        xtcf    string %s\n' % (self.xtc_file)
        in_strings += 'variable        rstf1   string %s\n' % (self.rst1_file)
        in_strings += 'variable        rstf2   string %s\n' % (self.rst2_file)
        in_strings += 'variable        Tprof   string %s\n' % (self.tprof_file)
        in_strings += 'variable        Jprof   string %s\n' % (self.Jprof_file)
        in_strings += 'variable        JDprof  string %s\n' % (self.JDprof_file)
        in_strings += 'variable        ldumpf  string %s\n' % (self.last_str)
        in_strings += 'variable        ldataf  string %s\n' % (self.last_data)
        in_strings += 'variable        pairst  string %s\n' % (self.pair_style)
        in_strings += 'variable        cutoff1 string %s\n' % (self.cutoff_in)
        in_strings += 'variable        cutoff2 string %s\n' % (self.cutoff_out)
        in_strings += 'variable        bondst  string %s\n' % (self.bond_style)
        in_strings += 'variable        anglest string %s\n' % (self.angle_style)
        in_strings += 'variable        dihedst string %s\n' % (self.dihedral_style)
        in_strings += 'variable        improst string %s\n' % (self.improper_style)
        in_strings += '##########################################################\n'
        in_strings += """

log             ${logf} append

units           real
atom_style      full
boundary        p p p

bond_style      ${bondst}  
angle_style     ${anglest}
dihedral_style  ${dihedst}
improper_style  ${improst}

pair_style      ${pairst} ${cutoff1} ${cutoff2}
pair_modify     mix arithmetic
special_bonds   amber
neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes
kspace_style    pppm 1e-6

read_data       ${dataf}

thermo_modify   flush yes
thermo          1000



##########################################################
## Preparation
##########################################################
variable        NA     equal 6.02214076*1.0e23
variable        kcal2j equal 4.184*1000
variable        ang2m  equal 1.0e-10
variable        fs2s   equal 1.0e-15

if "${axis} == x" then &
  "replicate    ${rep} ${repo} ${repo}" &
  "variable     ahi   equal  xhi" &
  "variable     alo   equal  xlo" &
  "variable     Jarea equal  ly*lz" &
  "variable     idx   equal  1" &
elif "${axis} == y" &
  "replicate    ${repo} ${rep} ${repo}" &
  "variable     ahi   equal  yhi" &
  "variable     alo   equal  ylo" &
  "variable     Jarea equal  lx*lz" &
  "variable     idx   equal  2" &
elif "${axis} == z" &
  "replicate    ${repo} ${repo} ${rep}" &
  "variable     ahi   equal  zhi" &
  "variable     alo   equal  zlo" &
  "variable     Jarea equal  lx*ly" &
  "variable     idx   equal  3"

variable        Nfreq   equal  ${avetime}/${Nevery}       # Number of data points to compute temperature during exchange interval 
variable        invslab equal  1/${slab}
variable        width   equal  (${ahi}-${alo})/${slab}
variable        inlo    equal  ${alo}+${width}*1
variable        inhi    equal  ${alo}+${width}*2
variable        outlo   equal  ${ahi}-${width}*2
variable        outhi   equal  ${ahi}-${width}*1

if "${axis} == x" then &
  "region       rqin    block    ${inlo}  ${inhi}  INF  INF  INF  INF  units box" &
  "region       rqout   block    ${outlo}  ${outhi}  INF  INF  INF  INF  units box" &
  "region       rfree   block    ${inlo}  ${outhi}  INF  INF  INF  INF  units box" &
  "region       rflux   block    ${inhi}  ${outlo}  INF  INF  INF  INF  units box" &
elif "${axis} == y" &
  "region       rqin    block    INF  INF  ${inlo}  ${inhi}  INF  INF  units box" &
  "region       rqout   block    INF  INF  ${outlo}  ${outhi}  INF  INF  units box" &
  "region       rfree   block    INF  INF  ${inlo}  ${outhi}  INF  INF  units box" &
  "region       rflux   block    INF  INF  ${inhi}  ${outlo}  INF  INF  units box" &
elif "${axis} == z" &
  "region       rqin    block    INF  INF  INF  INF  ${inlo}  ${inhi}  units box" &
  "region       rqout   block    INF  INF  INF  INF  ${outlo}  ${outhi}  units box" &
  "region       rfree   block    INF  INF  INF  INF  ${inlo}  ${outhi}  units box" &
  "region       rflux   block    INF  INF  INF  INF  ${inhi}  ${outlo}  units box"

group           gin     dynamic all region rqin
group           gout    dynamic all region rqout
group           gfree               region rfree

reset_timestep  0
##########################################################


##########################################################
## NEMD with langevin thermostat
##########################################################
timestep        ${TimeSt}
fix             NVE gfree nve
fix             langin  gin  langevin ${Htemp} ${Htemp} 100.0 ${seed1} tally yes
fix             langout gout langevin ${Ltemp} ${Ltemp} 100.0 ${seed2} tally yes

compute         ke gfree ke/atom
variable        temp atom c_ke/0.003

# Generate temperature profile of layers
compute         layers all chunk/atom bin/1d ${axis} lower ${invslab} units reduced
fix             1 all ave/chunk ${Nevery} ${Nfreq} ${avetime} layers v_temp density/mass norm all ave one file ${Tprof}

# Output
dump            1 all custom 1000 ${dumpf} id type mol xs ys zs ix iy iz
dump            2 all xtc 1000 ${xtcf}
dump_modify     2 unwrap yes
restart         100000 ${rstf1} ${rstf2}

variable        heatfin   equal   (f_langin*${kcal2j}/${NA})/(${Jarea}*${ang2m}*${ang2m})   # J/m^2 = Ws/m^2
variable        heatfout  equal   (f_langout*${kcal2j}/${NA})/(${Jarea}*${ang2m}*${ang2m})   # J/m^2 = Ws/m^2
thermo_style    custom step time temp press enthalpy etotal ke pe ebond eangle edihed eimp evdwl ecoul elong etail vol lx ly lz density pxx pyy pzz pxy pxz pyz f_langin f_langout v_heatfin v_heatfout
thermo_modify   flush yes
thermo          ${avetime}

variable        Time equal step
variable        EL   equal f_langin
variable        ER   equal f_langout
fix             E_out all print ${avetime} "${Time} ${EL} ${ER}" file ${Jprof} screen no

run             ${NStep}


"""
        if decomp:
            in_strings += """
##########################################################
## Component decomposition of heat flux
##########################################################
# heat flux preparation
compute         KE        all   ke/atom
compute         PE        all   pe/atom

#compute         Stress    all   centroid/stress/atom NULL virial
compute         Spair     all   stress/atom NULL pair
compute         Sbond     all   stress/atom NULL bond
compute         Sangle    all   centroid/stress/atom NULL angle
compute         Sdihed    all   centroid/stress/atom NULL dihedral
compute         Simpro    all   centroid/stress/atom NULL improper
compute         Skspac    all   stress/atom NULL kspace
compute         Sfix      all   stress/atom NULL fix
"""

            if decomp_intermol:
                in_strings += """
compute         Spairer   all   stress/atom NULL interpair
compute         Spairra   all   stress/atom NULL intrapair
"""

            in_strings += """
# Generate empty vector
group           empty     type      99999
compute         KENULL    empty ke/atom
compute         PENULL    empty pe/atom improper
compute         STNULL    empty stress/atom NULL improper

########################   Cell flux   ########################
###  |##|//|  |  |  |  |  |  |**|##|  ###   |//| cold slab
###  |##|//|  |  |  |  |  |  |**|##|  ###   |**| hot slab
###  |##|//|  |  |  |  |  |  |**|##|  ###   |##| fixed slab
###  |##|//|  |  |  |  |  |  |**|##|  ###
###  |##|//|  |  |  |  |  |  |**|##|  ###
###         <--------------->         ###
### heat flux decomposition of this reagion
###############################################################

# cell information
group           gflux    dynamic  all         region     rflux     every  ${Nevery}

# energy Flux  JE
#compute         Flux     gflux    heat/flux   KE         PE        Stress

# 1st term   eivi
compute         F1ke     gflux    heat/flux   KE         PENULL    STNULL
compute         F1pe     gflux    heat/flux   KENULL     PE        STNULL

# 2nd term   Sivi
#compute         Sivi     gflux    heat/flux   KENULL     PENULL    Stress
compute         Fpair    gflux    heat/flux   KENULL     PENULL    Spair
compute         Fbond    gflux    heat/flux   KENULL     PENULL    Sbond
compute         Fangle   gflux    heat/flux   KENULL     PENULL    Sangle
compute         Fdihed   gflux    heat/flux   KENULL     PENULL    Sdihed
compute         Fimpro   gflux    heat/flux   KENULL     PENULL    Simpro
compute         Fkspac   gflux    heat/flux   KENULL     PENULL    Skspac
compute         Ffix     gflux    heat/flux   KENULL     PENULL    Sfix
"""

            if decomp_intermol:
                in_strings += """
compute         Fpairer  gflux    heat/flux   KENULL     PENULL    Spairer
compute         Fpairra  gflux    heat/flux   KENULL     PENULL    Spairra
fix             20 gflux ave/time ${Nevery} ${Nfreq} ${avetime}  c_F1ke[${idx}] c_F1pe[${idx}] c_Fpair[${idx}]  c_Fpairer[${idx}]  c_Fpairra[${idx}]  c_Fbond[${idx}]  c_Fangle[${idx}]  c_Fdihed[${idx}]  c_Fimpro[${idx}]  c_Fkspac[${idx}]  c_Ffix[${idx}]  file ${JDprof}
"""
            else:
                in_strings += """
fix             20 gflux ave/time ${Nevery} ${Nfreq} ${avetime}  c_F1ke[${idx}] c_F1pe[${idx}] c_Fpair[${idx}] c_Fbond[${idx}]  c_Fangle[${idx}]  c_Fdihed[${idx}]  c_Fimpro[${idx}]  c_Fkspac[${idx}]  c_Ffix[${idx}]  file ${JDprof}
"""

            in_strings += """

##########################################################
## RNEMD with langevin thermostat in decomposition
##########################################################
thermo_style    custom step time temp press enthalpy etotal ke pe ebond eangle edihed eimp evdwl ecoul elong etail vol lx ly lz density pxx pyy pzz pxy pxz pyz f_langin f_langout
thermo_modify   flush yes
thermo          ${avetime}

run             ${NStepd}


"""
        in_strings += """
write_dump      all custom ${ldumpf} id x y z xu yu zu vx vy vz fx fy fz modify sort id
write_data      ${ldataf}
quit
"""

        with open(os.path.join(self.work_dir, self.in_file), 'w') as fh:
            fh.write(in_strings)
            fh.flush()
            if hasattr(os, 'fdatasync'):
                os.fdatasync(fh.fileno())
            else:
                os.fsync(fh.fileno())

        mol_sc = utils.deepcopy_mol(self.mol)
        if self.axis == 'x':
            mol_sc = poly.super_cell(mol_sc, x=rep, y=rep_other, z=rep_other, confId=confId)
        elif self.axis == 'y':
            mol_sc = poly.super_cell(mol_sc, x=rep_other, y=rep, z=rep_other, confId=confId)
        elif self.axis == 'z':
            mol_sc = poly.super_cell(mol_sc, x=rep_other, y=rep_other, z=rep, confId=confId)

        utils.MolToPDBFile(mol_sc, os.path.join(self.work_dir, self.pdb_file))

        return True


    def analyze(self):

        anal = NEMD_Langevin_Analyze(
            axis = self.axis,
            log_file  = os.path.join(self.work_dir, self.log_file),
            tprof_file = os.path.join(self.work_dir, self.tprof_file),
            JDprof_file  = os.path.join(self.work_dir, self.JDprof_file),
            traj_file = os.path.join(self.work_dir, self.xtc_file),
            pdb_file  = os.path.join(self.work_dir, self.pdb_file),
            dat_file  = os.path.join(self.work_dir, self.dat_file)
        )

        return anal



class NEMD_Langevin_Analyze(lammps.Analyze):
    def __init__(self, axis='x', prefix='', **kwargs):
        kwargs['log_file'] = kwargs.get('log_file', '%snemd_TC-MP_%s.log' % (prefix, axis))
        super().__init__(**kwargs)
        self.axis = axis
        self.tprof_file = kwargs.get('tprof_file', '%sslabtemp_%s.profile' % (prefix, axis))
        self.JDprof_file = kwargs.get('JDprof_file', '%sheatflux_decomp_%s.profile' % (prefix, axis))

        self.TC = np.nan
        self.Tgrad_data = {}
        self.Qgrad_data = {}
        self.TCdecomp_data = {}


    def calc_tc(self, init=4000, last=None, decomp=False, tschunk=5, printout=False, save=None, save_name='analyze'):

        if save:
            save_dir = os.path.join(os.path.dirname(self.log_file), save_name)
        else:
            save_dir = None

        if decomp:
            thermo_df = pd.concat((self.dfs[-2], self.dfs[-1]), sort=False)
        else:
            thermo_df = self.dfs[-1]

        if self.axis == 'x':
            length = thermo_df['Lx'].iloc[0]
        elif self.axis == 'y':
            length = thermo_df['Ly'].iloc[0]
        elif self.axis == 'z':
            length = thermo_df['Lz'].iloc[0]

        self.Tgrad_data = self.get_Tgrad_oneway(self.tprof_file, length, init=init, last=last,
                                      tschunk=tschunk, printout=printout, save=save)
        self.Qgrad_data = self.calc_heatflux_langevin(thermo_df, init=init, last=last, printout=printout, save=save)
        self.TC = self.Qgrad_data['Qgrad']/self.Tgrad_data['Tgrad']

        prop_data = {'thermal_conductivity': self.TC}
        conv_data = dict(**self.Tgrad_data, **self.Qgrad_data)

        if decomp:
            self.TCdecomp_data = self.analyze_decomp(tc=self.TC)
            prop_data.update(self.TCdecomp_data)

        self.prop_df = pd.DataFrame(prop_data, index=[0])
        self.conv_df = pd.DataFrame(conv_data, index=[0])

        if save:
            self.prop_df.to_csv(os.path.join(save_dir, 'tc_prop_data.csv'))
            self.conv_df.to_csv(os.path.join(save_dir, 'tc_conv_data.csv'))

        return self.TC


    def get_Tgrad_oneway(self, temp_file, length, threshold_r2=0.99, threshold_p=1e-7, target_temp=200,
                         printout=True, save=False, init=100, last=None, tschunk=5):
        """
        preset.tc.NEMD_Langevin.get_Tgrad_oneway

        Args:
            temp_data: Chunk averaged data of temperature
            length: Cell length along heat flux (float, angstrom)
        """   
        tgrads = []
        nchunk = 0

        df = self.read_ave(temp_file)
        
        for index1 in df.index.unique(level=0):
            data = df.loc[index1].to_numpy(dtype=np.float64)
            nchunk = len(data)
            tgrads.append(data)
        
        tgrads = np.array(tgrads)
        grad_conv = length * 1e-10

        chunk_free = np.where(tgrads[0, :, 2] > target_temp)[0]
        chunk_i = chunk_free[0]+tschunk
        chunk_l = chunk_free[-1]-tschunk+1
        tgrads_mean = np.mean(tgrads[init:last, chunk_i:chunk_l, 2], axis=0)
        tgrads_sd = np.std(tgrads[init:last, chunk_i:chunk_l, 2], axis=0, ddof=1)

        OK = False

        tmax = np.max(tgrads_mean)
        tmin = np.min(tgrads_mean)
            
        res=np.polyfit(tgrads[0, chunk_i:chunk_l, 0], tgrads_mean, 1)
        y = np.poly1d(res)(tgrads[0, chunk_i:chunk_l, 0])
        grad, k, r, p, se = stats.linregress(tgrads[0, chunk_i:chunk_l, 0], tgrads_mean)
        grad = abs(grad / grad_conv)  # K/(coord1) -> k/m
        grad_ave = grad
        r2 = r**2
        se = se / grad_conv  # K/(coord1) -> k/m
        se_ave = se

        if r2 >= threshold_r2 and p <= threshold_p:
            OK = True
            
        grad_data = {'Tgrad_check':OK, 'T_max':tmax, 'T_min':tmin, 'T_SD':tgrads_sd, 'T_SD_max':np.max(tgrads_sd),
                     'Tgrad_ave':grad, 'Tgrad':grad, 'Tgrad_r2':r2, 'Tgrad_p':p, 'Tgrad_SE':se}
            
        if printout or save:
            color = 'blue' if OK else 'red'

            fig, ax = pp.subplots(figsize=(6, 6))
            pp.scatter(tgrads[0, chunk_i:chunk_l, 0]*length, tgrads_mean, c=color)
            pp.plot(tgrads[0, chunk_i:chunk_l, 0]*length, y, c=color)
            pp.xlim(0, tgrads[0, -1, 0]*length)
            pp.title('T grad mean')
            pp.xlabel('Length [Angstrom]')
            pp.ylabel('Temperature [K]')
            output = "T_max = %f    T_min = %f\n" % (tmax, tmin)
            if OK: output += 'OK: grad ave.(K/m) = %e,   se = %e\n' % (grad_ave, se_ave)
            else: output += 'NG: grad ave.(K/m) = %e,   se = %e\n' % (grad_ave, se_ave)
            output += "grad(K/m) = %e,   r2 = %f,   p = %e,   se = %e\n" % (grad, r2, p, se)
            
            if printout:
                pp.show()
                print(output)

            if save:
                if not os.path.exists(save):
                    os.makedirs(save)
                fig.savefig(os.path.join(save, 'Tgrad_mean.png'))
                with open(os.path.join(save, 'Tgrad_mean.txt'), mode='w') as f:
                    f.write(output)
            
            pp.close(fig)

        return grad_data


    def calc_heatflux_langevin(self, thermo_df, init=0, last=None, langin='v_heatfin', langout='v_heatfout', printout=True, save=False):

        grad1, k1, r1, p1, se1 = stats.linregress(thermo_df['Time'].iloc[init:last]*1e-15, thermo_df[langin].iloc[init:last]*-1)
        r2_1 = r1**2

        grad2, k2, r2, p2, se2 = stats.linregress(thermo_df['Time'].iloc[init:last]*1e-15, thermo_df[langout].iloc[init:last])
        r2_2 = r2**2
        grad_data = {'Qgrad':(grad1+grad2)/2, 'Qgrad_ave':(grad1+grad2)/2,
                     'Qgrad_in':grad1, 'Qgrad_in_k':k1, 'Qgrad_in_r2':r2_1, 'Qgrad_in_p':p1, 'Qgrad_in_SE':se1,
                     'Qgrad_out':grad2, 'Qgrad_out_k':k2, 'Qgrad_out_r2':r2_2, 'Qgrad_out_p':p2, 'Qgrad_out_SE':se2}

        if printout or save:
            res1=np.polyfit(thermo_df['Time'].iloc[init:last]*1e-3, thermo_df[langin].iloc[init:last]*-1, 1)
            res2=np.polyfit(thermo_df['Time'].iloc[init:last]*1e-3, thermo_df[langout].iloc[init:last], 1)
            y1 = np.poly1d(res1)(thermo_df['Time'].iloc[init:last]*1e-3)
            y2 = np.poly1d(res2)(thermo_df['Time'].iloc[init:last]*1e-3)

            fig, ax = pp.subplots(figsize=(6, 6))
            pp.scatter(thermo_df['Time'].iloc[init:last]*1e-3, thermo_df[langin].iloc[init:last]*-1)
            pp.plot(thermo_df['Time'].iloc[init:last]*1e-3, y1)
            pp.scatter(thermo_df['Time'].iloc[init:last]*1e-3, thermo_df[langout].iloc[init:last])
            pp.plot(thermo_df['Time'].iloc[init:last]*1e-3, y2)
            pp.title('dQ/dT')
            pp.xlim(thermo_df['Time'].iloc[init:last].values[0]*1e-3, thermo_df['Time'].iloc[init:last].values[-1]*1e-3)
            pp.xlabel('Time [ps]')
            pp.ylabel('Q [Ws/m^2]')
            output = 'Heat source: Q grad. [W/m^2] = %e,   se = %e,   r2 = %f,   p = %e\n' % (grad1, se1, r2_1, p1)
            output += "Heat sink: Q grad. [W/m^2] = %e,   se = %e,   r2 = %f,   p = %e\n" % (grad2, se2, r2_2, p2)
            
            if printout:
                pp.show()
                print(output)

            if save:
                if not os.path.exists(save):
                    os.makedirs(save)
                fig.savefig(os.path.join(save, 'Qgrad.png'))
                with open(os.path.join(save, 'Qgrad.txt'), mode='w') as f:
                    f.write(output)
            
            pp.close(fig)

        return grad_data


    def analyze_decomp(self, tc=1.0):

        df = self.read_ave(self.JDprof_file)
        values = (df.sum(axis=0)/df.iloc[:, 0].sum(axis=0)).to_numpy()*tc

        if len(df.iloc[0, :]) == 9:
            all_tmp = df.sum(axis=1).to_numpy()
            values = ((df.sum(axis=0)/all_tmp.sum(axis=0)).to_numpy())*tc
            keys=['TC_ke', 'TC_pe', 'TC_pair', 'TC_bond', 'TC_angle', 'TC_dihed', 'TC_improper', 'TC_kspace', 'TC_fix']
        elif len(df.iloc[0, :]) == 10:
            values = ((df.sum(axis=0)/df.iloc[:, 0].sum(axis=0)).to_numpy())*tc
            keys=['TC_all', 'TC_ke', 'TC_pe', 'TC_pair', 'TC_bond', 'TC_angle', 'TC_dihed', 'TC_improper', 'TC_kspace', 'TC_fix']
        elif len(df.iloc[0, :]) == 11:
            all_tmp = df.iloc[:, [0, 1, 2, 5, 6, 7, 8, 9, 10]].sum(axis=1).to_numpy()
            values = ((df.sum(axis=0)/all_tmp.sum(axis=0)).to_numpy())*tc
            keys=['TC_ke', 'TC_pe', 'TC_pair', 'TC_pair_inter', 'TC_pair_intra',
                  'TC_bond', 'TC_angle', 'TC_dihed', 'TC_improper', 'TC_kspace', 'TC_fix']
        elif len(df.iloc[0, :]) == 12:
            values = ((df.sum(axis=0)/df.iloc[:, 0].sum(axis=0)).to_numpy())*tc
            keys=['TC_all', 'TC_ke', 'TC_pe', 'TC_pair', 'TC_pair_inter', 'TC_pair_intra',
                  'TC_bond', 'TC_angle', 'TC_dihed', 'TC_improper', 'TC_kspace', 'TC_fix']
        else:
            utils.radon_print('Can not read the format of decomposition analysis in thermal conductivity.', level=2)

        TCdecomp = dict(zip(keys, values))

        return TCdecomp




class EMD_GK(preset.Preset):
    def __init__(self, mol, prefix='', work_dir=None, save_dir=None, solver_path=None, **kwargs):
        super().__init__(mol, prefix=prefix, work_dir=work_dir, save_dir=save_dir, solver_path=solver_path, **kwargs)

        self.dat_file = kwargs.get('dat_file', 'emd_TC-GK.data')
        self.pdb_file = kwargs.get('pdb_file', 'emd_TC-GK.pdb')
        self.in_file = kwargs.get('in_file', 'emd_TC-GK.in')
        self.log_file = kwargs.get('log_file', 'emd_TC-GK.log')
        self.dump_file = kwargs.get('dump_file', 'emd_TC-GK.dump')
        self.xtc_file = kwargs.get('xtc_file', 'emd_TC-GK.xtc')
        self.rst1_file = kwargs.get('rst1_file', 'emd_TC-GK_1.rst')
        self.rst2_file = kwargs.get('rst2_file', 'emd_TC-GK_2.rst')
        self.kappa_file = kwargs.get('kappa_file', 'emd_TC-GK_kappa.profile')
        self.autocorr_file = kwargs.get('autocorr_file', 'autocorr_heatflux.profile')
        self.last_str = kwargs.get('last_str', 'emd_TC-GK_last.dump')
        self.last_data = kwargs.get('last_data', 'emd_TC-GK_last.data')
        self.pickle_file = kwargs.get('pickle_file', 'emd_TC-GK_last.pickle')
        self.json_file = kwargs.get('json_file', 'emd_TC-GK_last.json')


    def exec(self, confId=0, step=10000000, time_step=0.2, temp=300.0, hfsample=5, hfcorrlen=5000,
             omp=1, mpi=1, gpu=0, intel='auto', opt='auto', **kwargs):
        """
        preset.tc.EMD_GK.exec

        Preset of thermal conductivity calculation by Green-Kubo method.
        LAMMPS only

        Args:
            mol: RDKit Mol object
        
        Optional args:
            confId: Target conformer ID (int)
            step: Number of step (int)
            time_step: Timestep (float)
            temp: Temperature (float, K)
            hfsample: Sample interval of heat flux (int)
            hfcorrlen: Correlation length of heat flux (int)
            solver_path: File path of LAMMPS (str) 
            work_dir: Path of work directory (str)
            omp: Number of threads of OpenMP (int)
            mpi: Number of MPI process (int)
            gpu: Number of GPU (int)

        Returns:
            RDKit Mol object
        """
        lmp = lammps.LAMMPS(work_dir=self.work_dir, solver_path=self.solver_path)
        self.make_lammps_input(confId=confId, step=step, time_step=time_step, temp=temp, hfsample=hfsample, hfcorrlen=hfcorrlen, **kwargs)

        dt1 = datetime.datetime.now()
        utils.radon_print('Thermal conductive simulation (Green-Kubo EMD) by LAMMPS is running...', level=1)

        cp = lmp.exec(input_file=self.in_file, omp=omp, mpi=mpi, gpu=gpu, intel=intel, opt=opt)
        if cp.returncode != 0 and (
                    (self.last_str is not None and not os.path.exists(os.path.join(self.work_dir, self.last_str)))
                    or (self.last_data is not None and not os.path.exists(os.path.join(self.work_dir, self.last_data)))
                ):
            utils.radon_print('Error termination of %s' % (lmp.get_name), level=3)
            return None

        self.uwstr, self.wstr, _, self.vel, _ = lmp.read_traj_simple(os.path.join(self.work_dir, self.last_str))

        for i in range(self.mol.GetNumAtoms()):
            self.mol.GetConformer(confId).SetAtomPosition(i, Geom.Point3D(self.uwstr[i, 0], self.uwstr[i, 1], self.uwstr[i, 2]))
            self.mol.GetAtomWithIdx(i).SetDoubleProp('vx', self.vel[i, 0])
            self.mol.GetAtomWithIdx(i).SetDoubleProp('vy', self.vel[i, 1])
            self.mol.GetAtomWithIdx(i).SetDoubleProp('vz', self.vel[i, 2])

        self.mol = calc.mol_trans_in_cell(self.mol, confId=confId)
        utils.MolToJSON(self.mol, os.path.join(self.save_dir, self.json_file))
        utils.pickle_dump(self.mol, os.path.join(self.save_dir, self.pickle_file))

        dt2 = datetime.datetime.now()
        utils.radon_print('Complete thermal conductive simulation (Green-Kubo EMD). Elapsed time = %s' % str(dt2-dt1), level=1)

        return self.mol


    def make_lammps_input(self, confId=0, step=10000000, time_step=0.2, temp=300.0, hfsample=5, hfcorrlen=5000, **kwargs):

        utils.MolToPDBFile(self.mol, os.path.join(self.work_dir, self.pdb_file))
        lmp = lammps.LAMMPS(work_dir=self.work_dir, solver_path=self.solver_path)
        lmp.make_dat(self.mol, file_name=self.dat_file, confId=confId)
        seed = np.random.randint(1000, 999999)

        # Make input file
        in_strings  = 'variable        TimeSt    equal  %f\n' % (time_step)
        in_strings += 'variable        NStep     equal  %i\n' % (step)
        in_strings += 'variable        Ttemp     equal  %f\n' % (temp)
        in_strings += 'variable        dataf     string %s\n' % (self.dat_file)
        in_strings += 'variable        kpsample  equal  %i\n' % (hfsample)   # sample interval dt = kpsample * timestep
        in_strings += 'variable        kpcorrlen equal  %i\n' % (hfcorrlen)  # correlation length [0, kpcorrlen*dt]
        in_strings += 'variable        seed      equal  %i\n' % (seed)
        in_strings += '##########################################################\n'
        in_strings += '## Setting variables\n'
        in_strings += '##########################################################\n'
        in_strings += 'variable        kpdump    equal  ${kpcorrlen}*${kpsample}          # dump interval\n'
        in_strings += 'variable        logf      string %s\n' % (self.log_file)
        in_strings += 'variable        dumpf     string %s\n' % (self.dump_file)
        in_strings += 'variable        xtcf      string %s\n' % (self.xtc_file)
        in_strings += 'variable        rstf1     string %s\n' % (self.rst1_file)
        in_strings += 'variable        rstf2     string %s\n' % (self.rst2_file)
        in_strings += 'variable        kappaf    string %s\n' % (self.kappa_file)
        in_strings += 'variable        autocorrf string %s\n' % (self.autocorr_file)
        in_strings += 'variable        ldumpf    string %s\n' % (self.last_str)
        in_strings += 'variable        ldataf    string %s\n' % (self.last_data)
        in_strings += 'variable        pairst    string %s\n' % (self.pair_style)
        in_strings += 'variable        cutoff1   string %s\n' % (self.cutoff_in)
        in_strings += 'variable        cutoff2   string %s\n' % (self.cutoff_out)
        in_strings += 'variable        bondst    string %s\n' % (self.bond_style)
        in_strings += 'variable        anglest   string %s\n' % (self.angle_style)
        in_strings += 'variable        dihedst   string %s\n' % (self.dihedral_style)
        in_strings += 'variable        improst   string %s\n' % (self.improper_style)
        in_strings += """

variable        NA     equal 6.02214076*1.0e23
variable        kB     equal 1.380649*1.0e-23
variable        kcal2j equal 4.184*1000
variable        ang2m  equal 1.0e-10
variable        fs2s   equal 1.0e-15
variable        conv   equal (${kcal2j}/${NA})*(${kcal2j}/${NA})/${fs2s}/${ang2m}
##########################################################

log             ${logf} append

units           real
atom_style      full
boundary        p p p

bond_style      ${bondst}  
angle_style     ${anglest}
dihedral_style  ${dihedst}
improper_style  ${improst}

pair_style      ${pairst} ${cutoff1} ${cutoff2}
pair_modify     mix arithmetic
special_bonds   amber
neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes
kspace_style    pppm 1e-6

read_data       ${dataf}

velocity        all create ${Ttemp} ${seed} mom yes rot yes dist gaussian

##########################################################
## Thermal conductivity calculation by Green-Kubo method
##########################################################
timestep        ${TimeSt}

compute         kpKE      all ke/atom    # KE_i
compute         kpPE      all pe/atom    # PE_i
compute         kpStress  all centroid/stress/atom NULL virial  # S_i
compute         kpflux    all heat/flux kpKE kpPE kpStress

# x, y, z components of JE
variable        kpJx      equal c_kpflux[1]/vol
variable        kpJy      equal c_kpflux[2]/vol
variable        kpJz      equal c_kpflux[3]/vol

# Compute the autocorrelation function
fix             JJ all ave/correlate ${kpsample} ${kpcorrlen} ${kpdump} c_kpflux[1] c_kpflux[2] c_kpflux[3] type auto file ${autocorrf} overwrite ave running

variable        kpscale   equal ${conv}*(${kpsample}*dt)/${Ttemp}/${Ttemp}/vol/${kB}
variable        kappaxx   equal trap(f_JJ[3])*${kpscale}
variable        kappayy   equal trap(f_JJ[4])*${kpscale}
variable        kappazz   equal trap(f_JJ[5])*${kpscale}
variable        kappa     equal (v_kappaxx+v_kappayy+v_kappazz)/3.0   # in isotropic system, getting the average
fix             kappa     all ave/time ${kpdump} 1 ${kpdump} v_kappaxx v_kappayy v_kappazz v_kappa ave one file ${kappaf}

fix             NVT1 all nvt temp ${Ttemp} ${Ttemp} 100

# Output
dump            1 all custom 1000 ${dumpf} id type mol x y z vx vy vz
dump            2 all xtc 1000 ${xtcf}
dump_modify     2 unwrap yes
restart         100000 ${rstf1} ${rstf2}

thermo_style    custom step time temp press enthalpy etotal ke pe ebond eangle edihed eimp evdwl ecoul elong etail vol lx ly lz density pxx pyy pzz pxy pxz pyz v_kpJx v_kpJy v_kpJz
thermo          1000

run             ${NStep}

write_dump      all custom ${ldumpf} id x y z xu yu zu vx vy vz fx fy fz modify sort id
write_data      ${ldataf}
quit
"""

        with open(os.path.join(self.work_dir, self.in_file), 'w') as fh:
            fh.write(in_strings)
            fh.flush()
            if hasattr(os, 'fdatasync'):
                os.fdatasync(fh.fileno())
            else:
                os.fsync(fh.fileno())

        return True


    def analyze(self):

        anal = lammps.Analyze(
            log_file  = os.path.join(self.work_dir, self.log_file),
            traj_file = os.path.join(self.work_dir, self.xtc_file),
            pdb_file  = os.path.join(self.work_dir, self.pdb_file),
            dat_file  = os.path.join(self.work_dir, self.dat_file)
        )

        return anal



def restore(save_dir, **kwargs):
    method = kwargs.get('method', 'TC-MP')
    axis = kwargs.get('axis', 'x')
    if method == 'TC-GK':
        jsn = 'emd_TC-GK_last.json'
        pkl = 'emd_TC-GK_last.pickle'
    else:
        jsn = 'nemd_%s_%s_last.json' % (method, axis)
        pkl = 'nemd_%s_%s_last.pickle' % (method, axis)

    if os.path.isfile(os.path.join(save_dir, jsn)):
        mol = utils.JSONToMol(os.path.join(save_dir, jsn))
    else:
        mol = utils.pickle_load(os.path.join(save_dir, pkl))
    return mol


def helper_options():
    op = {
        'do_TC': False,
        'check_tc': False
    }
    return op

