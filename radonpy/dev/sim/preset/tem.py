#  Copyright (c) 2026. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# CAUTION!!
# This preset module is currently under development.
# Validity of calculation results has not been verified.

# ******************************************************************************
# sim.preset.dev.tem module
# ******************************************************************************

import os
import numpy as np
from scipy import stats
import pandas as pd
from matplotlib import pyplot as pp
from rdkit import Geometry as Geom
from ....sim import lammps, preset
from ....sim.md import MD
from ....core import utils, calc, const

__version__ = '1.0b2'

utils.radon_print('The preset module preset.tem is currently under development. Validity of calculation results has not been verified.', level=2)


class NEMD_tensile(preset.Preset):
    def __init__(self, mol, axis='x', prefix='', work_dir=None, save_dir=None, solver_path=None, idx=0, no_traj=False, **kwargs):
        """
        preset.tem.NEMD_tensile

        Preset of uniaxial tension by NEMD 

        Args:
            mol: RDKit Mol object
        """
        super().__init__(mol, prefix=prefix, work_dir=work_dir, save_dir=save_dir, solver_path=solver_path, **kwargs)
        self.axis = axis
        self.dat_file = {}
        self.pdb_file = {}
        self.in_file = {}
        self.log_file = {}
        self.dump_file = {}
        self.xtc_file = {}
        self.last_str = {}
        self.last_data = {}
        self.json_file = {}

        self.dat_file = kwargs.get('dat_file', '%snemd_tensile_%i.data' % (self.prefix, idx))
        self.pdb_file = kwargs.get('pdb_file', '%snemd_tensile_%i.pdb' % (self.prefix, idx))

        for ax in ['x', 'y', 'z']:
            self.in_file[ax] = kwargs.get('in_file', '%snemd_tensile_%s_%i.in' % (self.prefix, ax, idx))
            self.log_file[ax] = kwargs.get('log_file', '%snemd_tensile_%s_%i.log' % (self.prefix, ax, idx))
            if no_traj:
                self.dump_file[ax] = kwargs.get('dump_file')
                self.xtc_file[ax] = kwargs.get('xtc_file')
            else:
                self.dump_file[ax] = kwargs.get('dump_file', '%snemd_tensile_%s_%i.dump' % (self.prefix, ax, idx))
                self.xtc_file[ax] = kwargs.get('xtc_file', '%snemd_tensile_%s_%i.xtc' % (self.prefix, ax, idx))
            self.last_str[ax] = kwargs.get('last_str', '%snemd_tensile_%s_%i_last.dump' % (self.prefix, ax, idx))
            self.last_data[ax] = kwargs.get('last_data', '%snemd_tensile_%s_%i_last.data' % (self.prefix, ax, idx))
            self.json_file[ax] = kwargs.get('json_file', '%snemd_tensile_%s_%i_last.json' % (self.prefix, ax, idx))


    def uniaxial(self, step=2000000, time_step=1.0, scale=1.10, axis='x', n_split=1,
                temp=300.0, f_temp=None, press=1.0, f_press=None, **kwargs):

        self.axis = axis
        md = MD()
        md.pair_style = self.pair_style
        md.cutoff_in = self.cutoff_in
        md.cutoff_out = self.cutoff_out
        md.kspace_style = 'pppm'
        md.kspace_style_accuracy = '1e-6'
        md.log_file = self.log_file[axis]
        md.dat_file = self.dat_file
        md.dump_file = self.dump_file[axis]
        md.xtc_file = self.xtc_file[axis]
        md.rst = True
        md.outstr = self.last_str[axis]
        md.write_data = self.last_data[axis]

        if f_temp is None: f_temp = temp
        if f_press is None: f_press = press
        md.add_md('npt', 10000, time_step=0.2, shake=False, t_start=temp, t_stop=f_temp, p_start=press, p_stop=f_press, **kwargs)

        if n_split > 1:
            s_scale = (scale - 1) / n_split
            r_scale = [(1.0+s_scale*(i+1)) for i in range(n_split)]
            c_scale = [(r_scale[i]/r_scale[i-1]) if i > 0 else r_scale[0] for i in range(n_split)]
            for s in c_scale:
                md.add_md('npt', step, time_step=time_step, shake=True, t_start=temp, t_stop=f_temp, p_start=press, p_stop=f_press, **kwargs)
                md.wf[-1].add_deform(dftype='scale', deform_scale=s, axis=axis)
                md.wf[-1].add_variable([
                    ['p',   'equal', 'press'],
                    ['pxx', 'equal', 'pxx'],
                    ['pyy', 'equal', 'pyy'],
                    ['pzz', 'equal', 'pzz'],
                    ['pxy', 'equal', 'pxy'],
                    ['pxz', 'equal', 'pxz'],
                    ['pyz', 'equal', 'pyz']
                ])
                md.wf[-1].add_timeave(name='pave', var=['v_p', 'v_pxx', 'v_pyy', 'v_pzz', 'v_pxy', 'v_pxz', 'v_pyz'], nounfix=True)

        else:
            md.add_md('npt', step, time_step=time_step, shake=True, t_start=temp, t_stop=f_temp, p_start=press, p_stop=f_press, **kwargs)
            md.wf[-1].add_deform(dftype='scale', deform_scale=scale, axis=axis)
            md.wf[-1].add_variable([
                ['p',   'equal', 'press'],
                ['pxx', 'equal', 'pxx'],
                ['pyy', 'equal', 'pyy'],
                ['pzz', 'equal', 'pzz'],
                ['pxy', 'equal', 'pxy'],
                ['pxz', 'equal', 'pxz'],
                ['pyz', 'equal', 'pyz']
            ])
            md.wf[-1].add_timeave(name='pave', var=['v_p', 'v_pxx', 'v_pyy', 'v_pzz', 'v_pxy', 'v_pxz', 'v_pyz'], nounfix=True)

        return md


    def analyze(self):

        analy = lammps.Analyze(
            log_file  = os.path.join(self.work_dir, self.log_file['x']),
            pdb_file  = os.path.join(self.work_dir, self.pdb_file),
            dat_file  = os.path.join(self.work_dir, self.dat_file),

            work_dir = self.work_dir,
            save_dir = self.save_dir,
            log_file_list  = self.log_file,
            traj_file_list = self.xtc_file
        )

        return analy


class NEMD_tensile_modulus(NEMD_tensile):
    def exec(self, confId=0, step=1000000, time_step=1.0, scale=0.90, temp=300.0, f_temp=None, press=1.0, f_press=None,
                omp=1, mpi=1, gpu=0, intel='auto', opt='auto', **kwargs):
        """
        preset.tem.NEMD_tensile_modulus

        NEMD simulation of tensile modulus and poisson ratio by uniaxial stretching along each x, y, and z axis with NPT ensemble

        Optional args:
            confId: Target conformer ID (int)
            step: Number of MD steps (int)
            time_step: Set timestep of MD (float or None, fs)
            scale: Scale of uniaxial stretching (float)
            temp: Initial temperature (float, K)
            f_temp: Final temperature (float or None, K)
            press: Initial pressure (float, atm)
            f_press: Final pressure (float or None, atm)
            polarizable: Use polarizable Drude model (boolean)
            solver: lammps (str)
            thermostat: Nose-Hoover, Langevin, Berendsen, csvr, or csld (str, default:Nose-Hoover)
            barostat: Nose-Hoover, or Berendsen (str, default:Nose-Hoover)

        Returns:
            Unwrapped coordinates (float, numpy.ndarray, angstrom)
            Cell length (float, numpy.ndarray, angstrom)
        """

        utils.MolToPDBFile(self.mol, os.path.join(self.work_dir, self.pdb_file))
        lmp = lammps.LAMMPS(work_dir=self.work_dir, solver_path=self.solver_path)
        lmp.make_dat(self.mol, file_name=self.dat_file, confId=confId)

        for axis in ['x', 'y', 'z']:
            mol = utils.deepcopy_mol(self.mol)
            md = self.uniaxial(step=step, time_step=time_step, scale=scale, axis=axis, n_split=1,
                            temp=temp, f_temp=f_temp, press=press, f_press=f_press, **kwargs)
            mol = lmp.run(md, mol=mol, confId=confId, input_file=self.in_file[axis], last_str=self.last_str[axis],
                            last_data=self.last_data[axis], omp=omp, mpi=mpi, gpu=gpu, intel=intel, opt=opt)
            utils.MolToJSON(mol, os.path.join(self.save_dir, self.json_file[axis]))

        return mol


    def make_lammps_input(self, confId=0, step=1000000, time_step=1.0, scale=0.90,
                temp=300.0, f_temp=None, press=1.0, f_press=None, **kwargs):

        utils.MolToPDBFile(self.mol, os.path.join(self.work_dir, self.pdb_file))
        lmp = lammps.LAMMPS(work_dir=self.work_dir, solver_path=self.solver_path, check_lammps_package=False)
        lmp.make_dat(self.mol, file_name=self.dat_file, confId=confId)

        for axis in ['x', 'y', 'z']:
            md = self.uniaxial(step=step, time_step=time_step, scale=scale, axis=axis,
                            temp=temp, f_temp=f_temp, press=press, f_press=f_press, **kwargs)
            lmp.make_input(md, file_name=self.in_file[axis])

        return True


    def analyze(self, ignore_log=[], **kwargs):

        analy = NEMD_tensile_modulus_analyze(
            log_file  = os.path.join(self.work_dir, self.log_file['x']),
            traj_file = os.path.join(self.work_dir, self.xtc_file['x']) if self.xtc_file['x'] is not None else None,
            pdb_file  = os.path.join(self.work_dir, self.pdb_file),
            dat_file  = os.path.join(self.work_dir, self.dat_file),

            work_dir = self.work_dir,
            save_dir = self.save_dir,
            log_file_list  = self.log_file,
            traj_file_list = self.xtc_file,
            ignore_log = ignore_log,
            **kwargs
        )

        return analy


class NEMD_tensile_modulus_analyze(lammps.Analyze):
    def __init__(self, prefix='', target=-1, ignore_log=[], **kwargs):
        super().__init__(ignore_log=ignore_log, **kwargs)
        self.work_dir = kwargs.get('work_dir', './')
        self.save_dir = kwargs.get('save_dir', self.work_dir)

        self.log_file_list = kwargs.get('log_file_list',
            {
                'x': 'nemd_tensile_x_0.log',
                'y': 'nemd_tensile_y_0.log',
                'z': 'nemd_tensile_z_0.log'
            })

        self.traj_file_list = kwargs.get('traj_file_list',
            {
                'x': 'nemd_tensile_x_0.xtc',
                'y': 'nemd_tensile_y_0.xtc',
                'z': 'nemd_tensile_z_0.xtc'
            })

        self.df_axis = {
            'x': self.read_log(os.path.join(self.work_dir, self.log_file_list['x'])),
            'y': self.read_log(os.path.join(self.work_dir, self.log_file_list['y'])),
            'z': self.read_log(os.path.join(self.work_dir, self.log_file_list['z']))
        }

        self.tensile_mod_data = {}
        self.poisson_ratio_data = {}
        self.bulk_mod_data = {}
        self.shear_mod_data = {}
        self.lame_const_data = {}
        self.speed_sound_data = {}
        self.speed_sound_s_data = {}


    def calc_tensile_prop(self, n_split=1):
        young_m = {}
        poisson = {}

        for i, axis in enumerate(['x', 'y', 'z']):
            strain = {}
            true_strain = {}
            df = pd.concat([self.df_axis[axis][i] for i in range(-n_split, 0, 1)])
            
            lx0 = df['Lx'].iloc[0]
            ly0 = df['Ly'].iloc[0]
            lz0 = df['Lz'].iloc[0]
            strain['x'] = (df['Lx'].to_numpy() - lx0) / lx0
            strain['y'] = (df['Ly'].to_numpy() - ly0) / ly0
            strain['z'] = (df['Lz'].to_numpy() - lz0) / lz0
            true_strain['x'] = np.log(1.0 + strain['x'])
            true_strain['y'] = np.log(1.0 + strain['y'])
            true_strain['z'] = np.log(1.0 + strain['z'])
            stress = df['f_pave[%i]' % (i+2)].to_numpy() * -1 * const.atm2pa * 1e-6  # atm -> MPa
            time = df['Time'].to_numpy()

            young_m[axis] = self.calc_tensile_modulus(stress, true_strain[axis], time, axis=axis, save=self.save_dir)
            poisson[axis] = self.calc_poisson_ratio(true_strain, time, axis=axis, save=self.save_dir)

        prop_data = {
            'tem_tensile_modulus': np.mean(list(young_m.values())),
            'tem_poisson_ratio': np.mean(list(poisson.values())),
 
            'tem_tensile_modulus_x': young_m['x'],
            'tem_poisson_ratio_x': poisson['x'],

            'tem_tensile_modulus_y': young_m['y'],
            'tem_poisson_ratio_y': poisson['y'],

            'tem_tensile_modulus_z': young_m['z'],
            'tem_poisson_ratio_z': poisson['z']
        }

        prop_data['tem_bulk_modulus'] = self.calc_bulk_mod(prop_data['tem_tensile_modulus'], prop_data['tem_poisson_ratio'])
        prop_data['tem_shear_modulus'] = self.calc_shear_mod(prop_data['tem_tensile_modulus'], prop_data['tem_poisson_ratio'])
        prop_data['tem_lame_constant'] = self.calc_lame_const(prop_data['tem_tensile_modulus'], prop_data['tem_poisson_ratio'])

        for ax in ['x', 'y', 'z']:
            prop_data['tem_bulk_modulus_%s' % ax] = self.calc_bulk_mod(prop_data['tem_tensile_modulus_%s' % ax], prop_data['tem_poisson_ratio_%s' % ax])
            prop_data['tem_shear_modulus_%s' % ax] = self.calc_shear_mod(prop_data['tem_tensile_modulus_%s' % ax], prop_data['tem_poisson_ratio_%s' % ax])
            prop_data['tem_lame_constant_%s' % ax] = self.calc_lame_const(prop_data['tem_tensile_modulus_%s' % ax], prop_data['tem_poisson_ratio_%s' % ax])
    
        return prop_data


    def calc_tensile_modulus(self, stress, strain, time, init_strain=0.01, last_strain=0.03, init2_strain=0.02, last2_strain=0.04,
                            axis='x', printout=False, save=None):

        if not (0 <= init_strain < last_strain):
            raise ValueError("Invalid strain range")

        if np.max(np.abs(strain)) < last_strain:
            raise ValueError("Insufficient maximum strain")

        init = np.where(abs(strain)>=init_strain)[0][0]
        last = np.where(abs(strain)<=last_strain)[0][-1] + 1
        # init2 = np.where(abs(strain)>=init2_strain)[0][0]
        # last2 = np.where(abs(strain)<=last2_strain)[0][-1] + 1

        res = np.polyfit(strain[init:last], stress[init:last], 1)
        y = np.poly1d(res)(strain[init:last])
        grad, k, r, p, std = stats.linregress(strain[init:last], stress[init:last])
        young = grad * 1e-3  # MPa -> GPa

        # res2 = np.polyfit(strain[init2:last2], stress[init2:last2], 1)
        # y2 = np.poly1d(res2)(strain[init2:last2])
        # grad2, k2, r2, p2, std2 = stats.linregress(strain[init2:last2], stress[init2:last2])
        # young2 = grad2 * 1e-3  # MPa -> GPa

        young_m = young # if young >= young2 else young2

        fig, ax = pp.subplots(figsize=(6, 6))
        ax.ticklabel_format(style="sci",  axis="y", scilimits=(0,0))
        ax.plot(strain, stress, linewidth=1.0)
        ax.plot(strain[init:last], y, linewidth=3.0)
        # ax.plot(strain[init2:last2], y2, linewidth=3.0)
        ax.set_xlabel('Strain', fontsize=12)
        ax.set_ylabel('Stress [MPa]', fontsize=12)

        output = 'Young\'s modulus (%s axis) = %f [GPa]\n' % (axis, young_m)

        if printout:
            pp.show()
            utils.radon_print(output, level=1)

        if save:
            if not os.path.exists(save):
                os.makedirs(save)
            fig.savefig(os.path.join(save, 'stress_strain_%s.png' % axis))
            with open(os.path.join(save, 'stress_strain_%s.txt' % axis), mode='w') as f:
                f.write(output)
            
        pp.close(fig)
        
        return young_m


    def calc_poisson_ratio(self, strain, time, axis='x', init_strain=0.01, last_strain=0.03, printout=False, save=None):
        if not (0 <= init_strain < last_strain):
            raise ValueError("Invalid strain range")

        if np.max(np.abs(strain[axis])) < last_strain:
            raise ValueError("Insufficient maximum strain")

        if axis == 'x':
            axis1 = 'y'
            axis2 = 'z'
        elif axis == 'y':
            axis1 = 'x'
            axis2 = 'z'
        elif axis == 'z':
            axis1 = 'x'
            axis2 = 'y'

        init = np.where(abs(strain[axis])>=init_strain)[0][0]
        last = np.where(abs(strain[axis])<=last_strain)[0][-1] + 1

        poisson_ratio1 = -stats.linregress(
            strain[axis][init:last],
            strain[axis1][init:last]
        ).slope

        poisson_ratio2 = -stats.linregress(
            strain[axis][init:last],
            strain[axis2][init:last]
        ).slope

        poisson_ratio = (poisson_ratio1 + poisson_ratio2) / 2

        # with np.errstate(divide='ignore', invalid='ignore'):
        #     poisson_axis1 = -1 * strain[axis1] / strain[axis]
        #     poisson_axis2 = -1 * strain[axis2] / strain[axis]
        #     poisson = -1 * ((strain[axis1]+strain[axis2])/2) / strain[axis]
        
        # time = time * 1e-3  # fs -> ps
        # poisson_ratio = np.nanmean(poisson[init:last])
        # poisson_ratio1 = np.nanmean(poisson_axis1[init:last])
        # poisson_ratio2 = np.nanmean(poisson_axis2[init:last])

        fig, ax = pp.subplots(figsize=(6, 6))
        ax.ticklabel_format(style="sci",  axis="y", scilimits=(0,0))
        ax.scatter(
            strain[axis][init:last],
            strain[axis1][init:last],
            s=5,
            label=axis1
        )
        ax.scatter(
            strain[axis][init:last],
            strain[axis2][init:last],
            s=5,
            label=axis2
        )       
        ax.set_xlabel('Axial strain', fontsize=12)
        ax.set_ylabel('Transverse strain', fontsize=12)
        ax.legend()

        output = 'Poisson\'s ratio (tensile in %s axis) = %f\n' % (axis, poisson_ratio)
        output += 'Poisson\'s ratio %s (tensile in %s axis) = %f\n' % (axis1, axis, poisson_ratio1)
        output += 'Poisson\'s ratio %s (tensile in %s axis) = %f' % (axis2, axis, poisson_ratio2)

        if printout:
            pp.show()
            utils.radon_print(output, level=1)

        if save:
            if not os.path.exists(save):
                os.makedirs(save)
            fig.savefig(os.path.join(save, 'poisson_ratio_%s.png' % axis))
            with open(os.path.join(save, 'poisson_ratio_%s.txt' % axis), mode='w') as f:
                f.write(output)
            
        pp.close(fig)

        return poisson_ratio


    def calc_bulk_mod(self, young, poisson):
        bulk = young / (3*(1-2*poisson))
        return bulk


    def calc_shear_mod(self, young, poisson):
        shear = young / (2*(1+poisson))
        return shear


    def calc_lame_const(self, young, poisson):
        lame = young*poisson / ((1+poisson)*(1-2*poisson))
        return lame



# EXPERIMENTAL
class NEMD_uniaxial_tensile(NEMD_tensile):
    """
    preset.tem.NEMD_uniaxial_tensile

    Preset of uniaxial tension by NEMD 

    Args:
        mol: RDKit Mol object
    """
    def exec(self, confId=0, step=50000000, time_step=1.0, scale=5.0, axis='x', n_split=1, temp=300.0, f_temp=None, press=1.0, f_press=None,
                omp=1, mpi=1, gpu=0, intel='auto', opt='auto', **kwargs):
        """
        preset.tensile.NEMD_uniaxial_tensile

        NEMD simulation of uniaxial extension with NPT ensemble

        Optional args:
            confId: Target conformer ID (int)
            step: Number of MD steps (int)
            time_step: Set timestep of MD (float or None, fs)
            scale: Scale of uniaxial extension (float)
            axis: Axis of uniaxial extension (str, x, y, or z)
            temp: Initial temperature (float, K)
            f_temp: Final temperature (float or None, K)
            press: Initial pressure (float, atm)
            f_press: Final pressure (float or None, atm)
            polarizable: Use polarizable Drude model (boolean)
            solver: lammps (str)
            thermostat: Nose-Hoover, Langevin, Berendsen, csvr, or csld (str, default:Nose-Hoover)
            barostat: Nose-Hoover, or Berendsen (str, default:Nose-Hoover)

        Returns:
            Unwrapped coordinates (float, numpy.ndarray, angstrom)
            Cell length (float, numpy.ndarray, angstrom)
        """
        utils.MolToPDBFile(self.mol, os.path.join(self.work_dir, self.pdb_file))
        lmp = lammps.LAMMPS(work_dir=self.work_dir, solver_path=self.solver_path)
        lmp.make_dat(self.mol, file_name=self.dat_file, confId=confId)

        md = self.uniaxial(step=step, time_step=time_step, scale=scale, axis=axis, n_split=n_split,
                        temp=temp, f_temp=f_temp, press=press, f_press=f_press, **kwargs)

        self.mol = lmp.run(md, mol=self.mol, confId=confId, input_file=self.in_file[axis], last_str=self.last_str[axis],
                            last_data=self.last_data[axis], omp=omp, mpi=mpi, gpu=gpu, intel=intel)
        utils.MolToJSON(self.mol, os.path.join(self.save_dir, self.json_file[axis]))

        return self.mol


    def make_lammps_input(self, confId=0, step=50000000, time_step=1.0, scale=5.0, axis='x', n_split=1,
                temp=300.0, f_temp=None, press=1.0, f_press=None, omp=1, mpi=1, gpu=0, intel='auto', **kwargs):

        utils.MolToPDBFile(self.mol, os.path.join(self.work_dir, self.pdb_file))
        lmp = lammps.LAMMPS(work_dir=self.work_dir, solver_path=self.solver_path)
        lmp.make_dat(self.mol, file_name=self.dat_file, confId=confId)

        md = self.uniaxial(step=step, time_step=time_step, scale=scale, axis=axis, n_split=n_split,
                        temp=temp, f_temp=f_temp, press=press, f_press=f_press, **kwargs)
        lmp.make_input(md, file_name=self.in_file[axis])

        return True


# EXPERIMENTAL
class NEMD_uniaxial_tensile_rough(NEMD_tensile):
    """
    preset.tem.NEMD_uniaxial_tensile_rough

    Preset of uniaxial tension by NEMD 

    Args:
        mol: RDKit Mol object
    """
    def exec(self, confId=0, step=50000000, time_step=1.0, scale=5.0, axis='x', n_split=1, temp=300.0, f_temp=None, press=1.0, f_press=None,
                omp=1, mpi=1, gpu=0, intel='auto', opt='auto', **kwargs):
        """
        preset.tensile.NEMD_uniaxial_tensile_rough

        NEMD simulation of uniaxial extension with NPT ensemble

        Optional args:
            confId: Target conformer ID (int)
            step: Number of MD steps (int)
            time_step: Set timestep of MD (float or None, fs)
            scale: Scale of uniaxial extension (float)
            axis: Axis of uniaxial extension (str, x, y, or z)
            temp: Initial temperature (float, K)
            f_temp: Final temperature (float or None, K)
            press: Initial pressure (float, atm)
            f_press: Final pressure (float or None, atm)
            polarizable: Use polarizable Drude model (boolean)
            solver: lammps (str)
            thermostat: Nose-Hoover, Langevin, Berendsen, csvr, or csld (str, default:Nose-Hoover)
            barostat: Nose-Hoover, or Berendsen (str, default:Nose-Hoover)

        Returns:
            Unwrapped coordinates (float, numpy.ndarray, angstrom)
            Cell length (float, numpy.ndarray, angstrom)
        """
        utils.MolToPDBFile(self.mol, os.path.join(self.work_dir, self.pdb_file))
        lmp = lammps.LAMMPS(work_dir=self.work_dir, solver_path=self.solver_path)
        lmp.make_dat(self.mol, file_name=self.dat_file, confId=confId)

        md = self.uniaxial(step=step, time_step=time_step, scale=scale, axis=axis, n_split=n_split,
                        temp=temp, f_temp=f_temp, press=press, f_press=f_press, **kwargs)
        md.pair_style = 'lj/charmm/coul/charmm'
        md.kspace_style = 'none'
        md.kspace_style_accuracy = ''

        self.mol = lmp.run(md, mol=self.mol, confId=confId, input_file=self.in_file[axis], last_str=self.last_str[axis],
                            last_data=self.last_data[axis], omp=omp, mpi=mpi, gpu=gpu, intel=intel)
        utils.MolToJSON(self.mol, os.path.join(self.save_dir, self.json_file[axis]))

        return self.mol


    def make_lammps_input(self, confId=0, step=50000000, time_step=1.0, scale=5.0, axis='x', n_split=1,
                temp=300.0, f_temp=None, press=1.0, f_press=None, omp=1, mpi=1, gpu=0, intel='auto', **kwargs):

        utils.MolToPDBFile(self.mol, os.path.join(self.work_dir, self.pdb_file))
        lmp = lammps.LAMMPS(work_dir=self.work_dir, solver_path=self.solver_path)
        lmp.make_dat(self.mol, file_name=self.dat_file, confId=confId)

        md = self.uniaxial(step=step, time_step=time_step, scale=scale, axis=axis, n_split=n_split,
                        temp=temp, f_temp=f_temp, press=press, f_press=f_press, **kwargs)
        md.pair_style = 'lj/charmm/coul/charmm'
        md.kspace_style = 'none'
        md.kspace_style_accuracy = ''
        lmp.make_input(md, file_name=self.in_file[axis])

        return True


def restore(save_dir, axis='x', idx=0, **kwargs):
    js = os.path.join(save_dir, 'nemd_tensile_%s_%i_last.json' % (axis, idx))
    mol = utils.JSONToMol(js)
    return mol


def helper_options():
    op = {'check_tensile': False}
    return op

