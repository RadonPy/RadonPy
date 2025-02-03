#  Copyright (c) 2022. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# ******************************************************************************
# sim.psi4_wrapper module
# ******************************************************************************

import numpy as np
import os
import gc
import datetime
import socket
from distutils.version import LooseVersion
import multiprocessing as MP
import concurrent.futures as confu
from rdkit import Chem
from rdkit import Geometry as Geom
import psi4
import resp

from ..core import const, calc, utils

__version__ = '0.2.9'

if LooseVersion(psi4.__version__) >= LooseVersion('1.4'):
    import qcengine

if const.mpi4py_avail:
    try:
        from mpi4py.futures import MPIPoolExecutor
    except ImportError as e:
        utils.radon_print('Cannot import mpi4py. Change to const.mpi4py_avail = False. %s' % e, level=2)
        const.mpi4py_avail = False


class Psi4w():
    def __init__(self, mol, confId=0, work_dir=None, tmp_dir=None, name=None, **kwargs):
        self.work_dir = work_dir if work_dir is not None else './'
        self.tmp_dir = tmp_dir if tmp_dir is not None else self.work_dir
        self.num_threads = kwargs.get('num_threads', kwargs.get('omp', utils.cpu_count()))
        self.memory = kwargs.get('memory', 1000) # MByte

        self.name = name if name else 'radonpy'
        self.mol = utils.deepcopy_mol(mol)
        self.confId = confId
        self.wfn = kwargs.get('wfn', None)
        self.charge = kwargs.get('charge', Chem.rdmolops.GetFormalCharge(self.mol))
        nr = calc.get_num_radicals(self.mol)
        self.multiplicity = kwargs.get('multiplicity', 1 if nr == 0 else 2 if nr%2 == 1 else 3)

        self.method = kwargs.get('method', 'wb97m-d3bj')
        self.basis = kwargs.get('basis', '6-31G(d,p)')
        basis_Br = kwargs.get('basis_Br', self.basis)
        basis_I = kwargs.get('basis_I', 'lanl2dz')
        self.basis_gen = {'Br': basis_Br, 'I': basis_I, **kwargs.get('basis_gen', {})}

        self.scf_type = kwargs.get('scf_type', 'df')
        self.scf_maxiter = kwargs.get('scf_maxiter', 128)
        self.scf_fail_on_maxiter = kwargs.get('scf_fail_on_maxiter', True)
        self.cc2wfn = kwargs.get('cc2wfn', None)
        self.cache_level = kwargs.get('cache_level', 2)
        self.cwd = os.getcwd()
        self.error_flag = False

        if LooseVersion(psi4.__version__) >= LooseVersion('1.4'):
            self.get_global_org = qcengine.config.get_global

        # Corresponds to Gaussian keyword
        if kwargs.get('dft_integral', None) == 'fine':
            self.dft_spherical_points = 302
            self.dft_radial_points = 75
        elif kwargs.get('dft_integral', None) == 'ultrafine':
            self.dft_spherical_points = 590
            self.dft_radial_points = 99
        elif kwargs.get('dft_integral', None) == 'superfine':
            self.dft_spherical_points = 974
            self.dft_radial_points = 175
        elif kwargs.get('dft_integral', None) == 'coarse':
            self.dft_spherical_points = 110
            self.dft_radial_points = 35
        elif kwargs.get('dft_integral', None) == 'SG1':
            self.dft_spherical_points = 194
            self.dft_radial_points = 50
        else:
            self.dft_spherical_points = kwargs.get('dft_spherical_points', 590)
            self.dft_radial_points = kwargs.get('dft_radial_points', 99)


    def __del__(self):
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.core.clean_variables()
        del self.wfn, self.cc2wfn
        gc.collect()


    @property
    def get_name(self):
        return 'Psi4'


    @property
    def psi4_version(self):
        return psi4.__version__


    def _init_psi4(self, *args, output=None):

        # Avoiding errors on Fugaku and in mpi4py
        if LooseVersion(psi4.__version__) >= LooseVersion('1.4'):
            qcengine.config.get_global = _override_get_global

        psi4.core.clean_options()
        os.environ['PSI_SCRATCH'] = os.path.abspath(self.tmp_dir)
        psi4.core.IOManager.shared_object().set_default_path(os.path.abspath(self.tmp_dir))

        pmol = self._mol2psi4(*args)
        self.error_flag = False
        self._basis_set_construction()
        psi4.set_num_threads(self.num_threads)
        psi4.set_memory('%i MB' % (self.memory))
        psi4.set_options({
            'dft_spherical_points': self.dft_spherical_points,
            'dft_radial_points': self.dft_radial_points,
            'scf_type': self.scf_type,
            'maxiter': self.scf_maxiter,
            'fail_on_maxiter': self.scf_fail_on_maxiter,
            'cachelevel': self.cache_level,
            'CC_NUM_THREADS': self.num_threads,
            'basis': 'radonpy_basis'
            })

        # Avoiding the bug due to MKL (https://github.com/psi4/psi4/issues/2279)
        if '1.4rc' in str(psi4.__version__):
            psi4.set_options({'wcombine': False})
        elif LooseVersion('1.4.1') > LooseVersion(psi4.__version__) >= LooseVersion('1.4'):
            psi4.set_options({'wcombine': False})

        self.cwd = os.getcwd()
        os.chdir(self.work_dir)
        if output is not None:
            psi4.core.set_output_file(output, False)

        return pmol

    def _fin_psi4(self):
        psi4.core.clean()
        psi4.core.clean_options()
        if LooseVersion(psi4.__version__) >= LooseVersion('1.4'):
            qcengine.config.get_global = self.get_global_org

        # Avoiding the bug that optimization is failed due to the optimization binary file remaining.
        opt_bin_file = os.path.join(self.tmp_dir, 'psi.%i.1' % os.getpid())
        if os.path.isfile(opt_bin_file):
            os.remove(opt_bin_file)

        os.chdir(self.cwd)
        gc.collect()


    def _mol2psi4(self, *args):
        """
        Psi4w._mol2psi4

        Convert RDkit Mol object to Psi4 Mol object

        Returns:
            Psi4 Mol object
        """

        geom = '%i   %i\n' % (self.charge, self.multiplicity)
        for arg in args:
            geom += '%s\n' % (arg)

        coord = np.array(self.mol.GetConformer(int(self.confId)).GetPositions())
        for i in range(self.mol.GetNumAtoms()):
            geom += '%2s  % .8f  % .8f  % .8f\n' % (self.mol.GetAtomWithIdx(i).GetSymbol(), coord[i, 0], coord[i, 1], coord[i, 2])

        pmol = psi4.geometry(geom)
        pmol.update_geometry()
        pmol.set_name(self.name)

        return pmol


    def _basis_set_construction(self):
        basis = self.basis.replace('(', '_').replace(')', '_').replace(',', '_').replace('+', 'p').replace('*', 's')
        bs = 'assign %s\n' % basis

        for element, basis in self.basis_gen.items():
            basis = basis.replace('(', '_').replace(')', '_').replace(',', '_').replace('+', 'p').replace('*', 's')
            bs += 'assign %s %s\n' % (element, basis)

        psi4.basis_helper(bs, name='radonpy_basis', set_option=True)


    def energy(self, wfn=True, **kwargs):
        """
        Psi4w.energy

        Single point energy calculation by Psi4

        Optional args:
            wfn: Store the wfn object of Psi4 (boolean)

        Returns:
            energy (float, kJ/mol)
        """

        pmol = self._init_psi4(output='./%s_psi4.log' % self.name)
        dt1 = datetime.datetime.now()
        utils.radon_print('Psi4 single point calculation is running...', level=1)

        try:
            if wfn:
                energy, self.wfn = psi4.energy(self.method, molecule=pmol, return_wfn=True, **kwargs)
            else:
                energy = psi4.energy(self.method, molecule=pmol, return_wfn=False, **kwargs)
            dt2 = datetime.datetime.now()
            utils.radon_print('Normal termination of psi4 single point calculation. Elapsed time = %s' % str(dt2-dt1), level=1)

        except psi4.SCFConvergenceError as e:
            utils.radon_print('Psi4 SCF convergence error. %s' % e, level=2)
            energy = e.wfn.energy()
            self.error_flag = True

        self._fin_psi4()

        return energy*const.au2kj # Hartree -> kJ/mol


    def optimize(self, wfn=True, freeze=[], ignore_conv_error=False, opt_type='min', geom_iter=50, geom_conv='QCHEM', geom_algorithm='RFO', dynamic_level=0, **kwargs):
        """
        Psi4w.optimize

        Structure optimization calculation by Psi4

        Optional args:
            wfn: Return the wfn object of Psi4 (boolean)
            freeze: Specify bond length, bond angle, or dihedral angles between atoms to be frozen
            ignore_conv_error: If optimization has an convergence error,
                                False: return np.nan, True: return energy without converging (boolean)

        Returns:
            energy (float, kJ/mol)
            coord (ndarray(float), angstrom)
        """

        pmol = self._init_psi4(output='./%s_psi4_opt.log' % self.name)

        if dynamic_level == 0 and calc.find_liner_angle(self.mol) and LooseVersion(psi4.__version__) < LooseVersion('1.8'):
            utils.radon_print('Found a linear angle in the molecule. Psi4 optimization setting \'dynamic_level\' was changed to 2.')
            dynamic_level = 2
            geom_iter = int(2*geom_iter)

        opt_dict = {
            'OPT_TYPE': opt_type,
            'GEOM_MAXITER': geom_iter,
            'G_CONVERGENCE': geom_conv,
            'STEP_TYPE': geom_algorithm,
            'OPTKING__ENSURE_BT_CONVERGENCE': True,
            'DYNAMIC_LEVEL': dynamic_level,
            'PRINT_OPT_PARAMS': True,
        }

        # Frozen coordinates
        frozen_bond = []
        frozen_angle = []
        frozen_dihedral = []
        for atoms in freeze:
            if len(atoms) == 2:
                frozen_bond.append('%i %i' % (atoms[0]+1, atoms[1]+1))
            elif len(atoms) == 3:
                frozen_angle.append('%i %i %i' % (atoms[0]+1, atoms[1]+1, atoms[2]+1))
            elif len(atoms) == 4:
                frozen_dihedral.append('%i %i %i %i' % (atoms[0]+1, atoms[1]+1, atoms[2]+1, atoms[3]+1))
            else:
                utils.radon_print('Illegal length of array for input atoms. (2, 3, or 4)', level=3)
        if len(frozen_bond) > 0:
            opt_dict['OPTKING__FROZEN_DISTANCE'] = ' '.join(frozen_bond)
        if len(frozen_angle) > 0:
            opt_dict['OPTKING__FROZEN_BEND'] = ' '.join(frozen_angle)
        if len(frozen_dihedral) > 0:
            opt_dict['OPTKING__FROZEN_DIHEDRAL'] = ' '.join(frozen_dihedral)

        psi4.set_options(opt_dict)
        dt1 = datetime.datetime.now()
        utils.radon_print('Psi4 optimization is running...', level=1)

        try:
            if wfn:
                energy, self.wfn = psi4.optimize(self.method, molecule=pmol, return_wfn=True, **kwargs)
            else:
                energy = psi4.optimize(self.method, molecule=pmol, return_wfn=False, **kwargs)
            dt2 = datetime.datetime.now()
            utils.radon_print('Normal termination of psi4 optimization. Elapsed time = %s' % str(dt2-dt1), level=1)
            coord = pmol.geometry().to_array() * const.bohr2ang

        except psi4.OptimizationConvergenceError as e:
            utils.radon_print('Psi4 optimization convergence error. %s' % e, level=2)
            if ignore_conv_error:
                energy = e.wfn.energy()
            else:
                energy = np.nan
            coord = np.array(e.wfn.molecule().geometry()) * const.bohr2ang
            self.error_flag = True

        except psi4.SCFConvergenceError as e:
            utils.radon_print('Psi4 SCF convergence error. %s' % e, level=2)
            if ignore_conv_error:
                energy = e.wfn.energy()
            else:
                energy = np.nan
            coord = np.array(e.wfn.molecule().geometry()) * const.bohr2ang
            self.error_flag = True

        except BaseException as e:
            self._fin_psi4()
            self.error_flag = True
            utils.radon_print('Error termination of psi4 optimization. %s' % e, level=3)

        self._fin_psi4()

        for i, atom in enumerate(self.mol.GetAtoms()):
            self.mol.GetConformer(int(self.confId)).SetAtomPosition(i, Geom.Point3D(coord[i, 0], coord[i, 1], coord[i, 2]))

        return energy*const.au2kj, coord # Hartree -> kJ/mol


    def scan(self, atoms, values=[], opt=True, ignore_conv_error=False, geom_iter=50, geom_conv='QCHEM', geom_algorithm='RFO', dynamic_level=0, **kwargs):
        """
        Psi4w.scan

        Scanning potential energy surface by Psi4

        Args:
            atoms: Array of index number of atoms in a scanning bond length, bond angle, or dihedral angle (list(int))

        Optional args:
            values: Array of bond length (angstrom), bond angle (degree), or dihedral angle (degree) values
                    to be calculated potential energies (list(float))
            opt: Perform optimization (boolean)
            ignore_conv_error: If optimization has an convergence error,
                                False: return np.nan, True: return energy without converging (boolean)

        Returns:
            energy (ndarray(float), kJ/mol)
            coord (ndarray(float), angstrom)
        """
        energies = np.array([])
        coords = []

        if dynamic_level == 0 and calc.find_liner_angle(self.mol) and LooseVersion(psi4.__version__) < LooseVersion('1.8'):
            utils.radon_print('Found a linear angle in the molecule. Psi4 optimization setting \'dynamic_level\' was changed to 2.')
            dynamic_level = 2
            geom_iter = int(2*geom_iter)

        opt_dict = {
            'GEOM_MAXITER': geom_iter,
            'G_CONVERGENCE': geom_conv,
            'STEP_TYPE': geom_algorithm,
            'OPTKING__ENSURE_BT_CONVERGENCE': True,
            'DYNAMIC_LEVEL': dynamic_level,
            'PRINT_OPT_PARAMS': True,
        }

        if len(atoms) == 2:
            opt_dict['OPTKING__FROZEN_DISTANCE'] = '%i %i' % (atoms[0]+1, atoms[1]+1)
            scan_type = 'bond length'
        elif len(atoms) == 3:
            opt_dict['OPTKING__FROZEN_BEND'] = '%i %i %i' % (atoms[0]+1, atoms[1]+1, atoms[2]+1)
            scan_type = 'bond angle'
        elif len(atoms) == 4:
            opt_dict['OPTKING__FROZEN_DIHEDRAL'] = '%i %i %i %i' % (atoms[0]+1, atoms[1]+1, atoms[2]+1, atoms[3]+1)
            scan_type = 'dihedral angle'
        else:
            utils.radon_print('Illegal length of array for input atoms. (2, 3, or 4)', level=3)

        dt1 = datetime.datetime.now()
        utils.radon_print('Psi4 scan (%s) is running...' % scan_type, level=1)

        for v in values:
            log_name = None
            conf = self.mol.GetConformer(self.confId)

            if len(atoms) == 2:
                Chem.rdMolTransforms.SetBondLength(conf, atoms[0], atoms[1], float(v))
                log_name = './%s_psi4_scan%i-%i_%f.log' % (self.name, atoms[0], atoms[1], float(v))
            elif len(atoms) == 3:
                Chem.rdMolTransforms.SetAngleDeg(conf, atoms[0], atoms[1], atoms[2], float(v))
                log_name = './%s_psi4_scan%i-%i-%i_%i.log' % (self.name, atoms[0], atoms[1], atoms[2], int(v))
            elif len(atoms) == 4:
                Chem.rdMolTransforms.SetDihedralDeg(conf, atoms[0], atoms[1], atoms[2], atoms[3], float(v))
                log_name = './%s_psi4_scan%i-%i-%i-%i_%i.log' % (self.name, atoms[0], atoms[1], atoms[2], atoms[3], int(v))
            
            pmol = self._init_psi4('symmetry c1', output=log_name)
            psi4.set_options(opt_dict)

            try:
                if opt:
                    utils.radon_print('Psi4 optimization (%s = %f) is running...' % (scan_type, float(v)), level=1)
                    dt3 = datetime.datetime.now()
                    energy = psi4.optimize(self.method, molecule=pmol, return_wfn=False, **kwargs)
                    dt4 = datetime.datetime.now()
                    utils.radon_print('Normal termination of psi4 optimization. Elapsed time = %s' % str(dt4-dt3), level=1)
                    coord = pmol.geometry().to_array() * const.bohr2ang
                else:
                    energy = psi4.energy(self.method, molecule=pmol, return_wfn=False, **kwargs)
                    coord = pmol.geometry().to_array() * const.bohr2ang

            except psi4.OptimizationConvergenceError as e:
                utils.radon_print('Psi4 optimization convergence error. %s' % e, level=2)
                if ignore_conv_error:
                    energy = e.wfn.energy()
                else:
                    energy = np.nan
                coord = np.array(e.wfn.molecule().geometry()) * const.bohr2ang
                self.error_flag = True

            except psi4.SCFConvergenceError as e:
                utils.radon_print('Psi4 SCF convergence error. %s' % e, level=2)
                if ignore_conv_error:
                    energy = e.wfn.energy()
                else:
                    energy = np.nan
                coord = np.array(e.wfn.molecule().geometry()) * const.bohr2ang
                self.error_flag = True

            except BaseException as e:
                self._fin_psi4()
                self.error_flag = True
                utils.radon_print('Error termination of psi4 optimization. %s' % e, level=3)

            energies = np.append(energies, energy)
            coords.append(coord)
            if opt:
                conf = Chem.rdchem.Conformer(self.mol.GetNumAtoms())
                conf.Set3D(True)
                for i in range(self.mol.GetNumAtoms()):
                    self.mol.GetConformer(int(self.confId)).SetAtomPosition(i, Geom.Point3D(coord[i, 0], coord[i, 1], coord[i, 2]))
                    conf.SetAtomPosition(i, Geom.Point3D(coord[i, 0], coord[i, 1], coord[i, 2]))
                conf_id = self.mol.AddConformer(conf, assignId=True)

            self._fin_psi4()

        dt2 = datetime.datetime.now()
        utils.radon_print('Normal termination of psi4 scan. Elapsed time = %s' % str(dt2-dt1), level=1)

        return energies*const.au2kj, np.array(coords) # Hartree -> kJ/mol


    def force(self, wfn=True, **kwargs):
        """
        Psi4w.force

        Force calculation by Psi4

        Optional args:
            wfn: Return the wfn object of Psi4 (boolean)

        Returns:
            force (float, kJ/(mol angstrom))
        """

        pmol = self._init_psi4(output='./%s_psi4_force.log' % self.name)
        dt1 = datetime.datetime.now()
        utils.radon_print('Psi4 force calculation is running...', level=1)

        try:
            if wfn:
                grad, self.wfn = psi4.gradient(self.method, molecule=pmol, return_wfn=True, **kwargs)
            else:
                grad = psi4.gradient(self.method, molecule=pmol, return_wfn=False, **kwargs)
            dt2 = datetime.datetime.now()
            utils.radon_print('Normal termination of psi4 force calculation. Elapsed time = %s' % str(dt2-dt1), level=1)

        except psi4.SCFConvergenceError as e:
            utils.radon_print('Psi4 SCF convergence error. %s' % e, level=2)
            grad = e.wfn.gradient()
            self.error_flag = True

        except BaseException as e:
            self._fin_psi4()
            self.error_flag = True
            utils.radon_print('Error termination of psi4 force calculation. %s' % e, level=3)

        self._fin_psi4()

        return grad.to_array()*const.au2kj/const.bohr2ang # Hartree/bohr -> kJ/(mol angstrom)


    def frequency(self, wfn=True, **kwargs):
        """
        Psi4w.frequency

        Frequency calculation by Psi4

        Optional args:
            wfn: Return the wfn object of Psi4 (boolean)

        Returns:
            energy (float, kJ/mol)
        """

        pmol = self._init_psi4(output='./%s_psi4_freq.log' % self.name)
        dt1 = datetime.datetime.now()
        utils.radon_print('Psi4 frequency calculation is running...')

        try:
            if wfn:
                energy, self.wfn = psi4.frequency(self.method, molecule=pmol, return_wfn=True, **kwargs)
            else:
                energy = psi4.frequency(self.method, molecule=pmol, return_wfn=False, **kwargs)
            dt2 = datetime.datetime.now()
            utils.radon_print('Normal termination of psi4 frequency calculation. Elapsed time = %s' % str(dt2-dt1), level=1)

        except psi4.SCFConvergenceError as e:
            utils.radon_print('Psi4 SCF convergence error. %s' % e, level=2)
            energy = e.wfn.energy()
            self.error_flag = True

        except BaseException as e:
            self._fin_psi4()
            self.error_flag = True
            utils.radon_print('Error termination of psi4 frequency calculation. %s' % e, level=3)

        self._fin_psi4()

        return energy*const.au2kj # Hartree -> kJ/mol


    def hessian(self, wfn=True, **kwargs):
        """
        Psi4w.hessian

        Hessian calculation by Psi4

        Optional args:
            wfn: Return the wfn object of Psi4 (boolean)

        Returns:
            hessian (float, kJ/(mol angstrom**2))
        """

        pmol = self._init_psi4(output='./%s_psi4_hessian.log' % self.name)
        dt1 = datetime.datetime.now()
        utils.radon_print('Psi4 hessian calculation is running...', level=1)

        try:
            if wfn:
                hessian, self.wfn = psi4.hessian(self.method, molecule=pmol, return_wfn=True, **kwargs)
            else:
                hessian = psi4.hessian(self.method, molecule=pmol, return_wfn=False, **kwargs)
            dt2 = datetime.datetime.now()
            utils.radon_print('Normal termination of psi4 hessian calculation. Elapsed time = %s' % str(dt2-dt1), level=1)

        except psi4.SCFConvergenceError as e:
            utils.radon_print('Psi4 SCF convergence error. %s' % e, level=2)
            hessian = e.wfn.hessian()
            self.error_flag = True

        except BaseException as e:
            self._fin_psi4()
            self.error_flag = True
            utils.radon_print('Error termination of psi4 hessian calculation. %s' % e, level=3)

        self._fin_psi4()

        return hessian.to_array()*const.au2kj/const.bohr2ang/const.bohr2ang # Hartree/bohr^2 -> kJ/(mol angstrom^2)


    def tddft(self, n_state=6, p_state=None, triplet='NONE', tda=False, tdscf_maxiter=60, **kwargs):
        """
        Psi4w.tddft

        TD-DFT calculation by Psi4

        Optional args:
            wfn: Store the wfn object of Psi4 (boolean)
            n_state: Number of states (int). If n_state < 0, all excitation states are calculated.
            p_state: Number of states, which is determined by [Num. of all excitation states] * p_state (float, 0.0 < p_state <= 1.0).
                     p_state is given priority over n_state.
            triplet: NONE, ALSO, or ONLY
            tda: Run with Tamm-Dancoff approximation (TDA), uses random-phase approximation (RPA) when false (boolean)
            tdscf_maxiter: Maximum number of TDSCF solver iterations (int)

        Returns:
            TD-DFT result
        """
        if LooseVersion(psi4.__version__) < LooseVersion('1.3.100'):
            utils.radon_print('TD-DFT calclation is not implemented in Psi4 of this version (%s).' % str(psi4.__version__), level=3)
            return []

        pmol = self._init_psi4(output='./%s_psi4_tddft.log' % self.name)
        psi4.set_options({
            'wcombine': False,
            'save_jk': True,
            'TDSCF_TRIPLETS': triplet,
            'TDSCF_TDA': tda,
            'TDSCF_MAXITER': tdscf_maxiter
            })
        dt1 = datetime.datetime.now()
        utils.radon_print('Psi4 TD-DFT calculation is running...', level=1)

        try:
            energy, self.wfn = psi4.energy(self.method, molecule=pmol, return_wfn=True, **kwargs)
            max_n_states = int((self.wfn.nmo() - self.wfn.nalpha()) * self.wfn.nalpha())
            if p_state is not None:
                if 0.0 < p_state <= 1.0:
                    n_state = int(max_n_states * p_state)
                    utils.radon_print('n_state of Psi4 TD-DFT calculation set to %i.' % n_state, level=1)
                else:
                    utils.radon_print('p_state=%f of Psi4 TD-DFT calculation is out of range (0.0 < p_state <= 1.0).' % float(p_state), level=3)
            elif n_state > max_n_states or n_state < 0:
                utils.radon_print('n_state of Psi4 TD-DFT calculation set to %i.' % max_n_states, level=1)
                n_state = max_n_states
            res = psi4.procrouting.response.scf_response.tdscf_excitations(self.wfn, states=n_state)

            dt2 = datetime.datetime.now()
            utils.radon_print('Normal termination of psi4 TD-DFT calculation. Elapsed time = %s' % str(dt2-dt1), level=1)

        except psi4.SCFConvergenceError as e:
            utils.radon_print('Psi4 SCF convergence error. %s' % e, level=2)
            res = []
            self.error_flag = True

        except BaseException as e:
            self._fin_psi4()
            self.error_flag = True
            utils.radon_print('Error termination of psi4 TD-DFT calculation. %s' % e, level=3)

        self._fin_psi4()

        return res


    def resp(self, **kwargs):
        """
        Psi4w.resp

        RESP charge calculation by Psi4

        Returns:
            RESP charge (float, array)
        """

        def _new_esp_solve(A, B):
            """
            Override function of resp.espfit.esp_solve to avoid LinAlgError
            Solves for point charges: A*q = B

            Parameters
            ----------
            A : ndarray
                array of matrix A
            B : ndarray
                array of matrix B

            Return
            ------
            q : ndarray
                array of charges

            """

            try:
                q = np.linalg.solve(A, B)
            except np.linalg.LinAlgError:
                q = np.linalg.lstsq(A, B, rcond=None)[0]

            if np.linalg.cond(A) > 1/np.finfo(A.dtype).eps:
                q = np.linalg.lstsq(A, B, rcond=None)[0]

            if LooseVersion(resp.__version__) >= LooseVersion('0.8'):
                return q
            else:
                return q, ''

        # Function override to avoid LinAlgError in calculation of inverse matrix
        resp.espfit.esp_solve = _new_esp_solve

        ptab = Chem.GetPeriodicTable()

        # Avoid a bug in RESP 0.8
        if LooseVersion(resp.__version__) >= LooseVersion('0.8'):
            # Make a dict of expected bonds
            defv_dict = {ptab.GetElementSymbol(n).upper(): ptab.GetDefaultValence(n) if ptab.GetDefaultValence(n) >= 0 else 0 for n in range(1, 119)}
            psi4.qcdb.parker._expected_bonds = {**defv_dict, **psi4.qcdb.parker._expected_bonds} 

        pmol = self._init_psi4(output='./%s_psi4_resp.log' % self.name)

        # Make a dict of vdW radii
        except_list = set([1, 6, 7, 8, 9, 15, 16, 17])  # H, C, N, O, F, P, S, Cl
        rvdw_dict = {ptab.GetElementSymbol(n): ptab.GetRvdw(n) for n in range(1, 119) if n not in except_list}
        # https://www.cgl.ucsf.edu/chimerax/docs/user/radii.html
        rvdw_dict['Br'] = 1.978
        rvdw_dict['I'] = 2.094

        options = {'VDW_SCALE_FACTORS'  : [1.4, 1.6, 1.8, 2.0],
                   'VDW_POINT_DENSITY'  : 20.0,
                   'RESP_A'             : 0.0005,
                   'RESP_B'             : 0.1,
                   'RADIUS'             : rvdw_dict,
                   'VDW_RADII'          : rvdw_dict,
                   'METHOD_ESP'         : self.method,
                   'BASIS_ESP'          : 'radonpy_basis'
                   }
        dt1 = datetime.datetime.now()
        utils.radon_print('Psi4 RESP charge calculation is running...', level=1)

        try:
            # First stage
            utils.radon_print('First stage of RESP charge calculation', level=0)
            if resp.resp.__code__.co_argcount == 2:
                charges1 = resp.resp([pmol], options)
            else:
                charges1 = resp.resp([pmol], [options])[0]

            # Change the value of the RESP parameter A
            options['RESP_A'] = 0.001

            # Reset radius parameters to avoid a bug of the RESP plugin
            options['RADIUS'] = {}

            # Add c for atoms fixed in second stage
            if LooseVersion(resp.__version__) >= LooseVersion('0.8'):
                resp.stage2_helper.set_stage2_constraint(pmol, charges1[1], options)
                options['grid'] = ['./1_%s_grid.dat' % pmol.name()]
                options['esp'] = ['./1_%s_grid_esp.dat' % pmol.name()]
            else:
                helper = resp.stage2_helper()
                helper.set_stage2_constraint(pmol, charges1[1], options)
                options['grid'] = './1_%s_grid.dat' % pmol.name()
                options['esp'] = './1_%s_grid_esp.dat' % pmol.name()
        
            # Second stage
            utils.radon_print('Second stage of RESP charge calculation', level=0)
            psi4.core.set_output_file('./%s_psi4_resp.log' % self.name, True)
            if resp.resp.__code__.co_argcount == 2:
                charges2 = resp.resp([pmol], options)
            else:
                charges2 = resp.resp([pmol], [options])[0]

            charges2 = np.array(charges2)
            dt2 = datetime.datetime.now()
            utils.radon_print('Normal termination of psi4 RESP charge calculation. Elapsed time = %s' % str(dt2-dt1), level=1)

        except psi4.SCFConvergenceError as e:
            utils.radon_print('Psi4 SCF convergence error. %s' % e, level=2)
            charges2 = np.array([np.nan for x in range(self.mol.GetNumAtoms())])
            self.error_flag = True

        except BaseException as e:
            self._fin_psi4()
            self.error_flag = True
            utils.radon_print('Error termination of psi4 RESP charge calculation. %s' % e, level=3)

        self._fin_psi4()

        # Get RESP charges
        for i, atom in enumerate(self.mol.GetAtoms()):
            atom.SetDoubleProp('ESP', charges2[0, i])
            atom.SetDoubleProp('RESP', charges2[1, i])

        return charges2[1]


    def mulliken_charge(self, recalc=False, **kwargs):
        """
        Psi4w.mulliken_charge

        Mulliken charge calculation by Psi4

        Optional args:
            recalc: Recalculation of wavefunction (boolean)

        Returns:
            Mulliken charge (float, ndarray)
        """

        if self.wfn is None or recalc:
            pmol = self._init_psi4(output='./%s_psi4.log' % self.name)
            energy, self.wfn = psi4.energy(self.method, molecule=pmol, return_wfn=True, **kwargs)
            self._fin_psi4()

        psi4.oeprop(self.wfn, 'MULLIKEN_CHARGES')
        mulliken = self.wfn.atomic_point_charges().np

        for i, atom in enumerate(self.mol.GetAtoms()):
            atom.SetDoubleProp('MullikenCharge', mulliken[i])

        return mulliken


    def lowdin_charge(self, recalc=False, **kwargs):
        """
        psi4w.lowdin_charge

        Lowdin charge calculation by Psi4

        Optional args:
            recalc: Recalculation of wavefunction (boolean)

        Returns:
            Lowdin charge (float, ndarray)
        """

        if self.wfn is None or recalc:
            pmol = self._init_psi4(output='./%s_psi4.log' % self.name)
            energy, self.wfn = psi4.energy(self.method, molecule=pmol, return_wfn=True, **kwargs)
            self._fin_psi4()

        psi4.oeprop(self.wfn, 'LOWDIN_CHARGES')
        lowdin = self.wfn.atomic_point_charges().np

        for i, atom in enumerate(self.mol.GetAtoms()):
            atom.SetDoubleProp('LowdinCharge', lowdin[i])

        return lowdin


    def polar(self, eps=1e-4, mp=0, **kwargs):
        """
        psi4w.polar

        Computation of dipole polarizability by finite field

        Optional args:
            eps: Epsilon of finite field
            mp: Number of multiprocessing

        Return:
            Dipole polarizability (float, angstrom^3)
            Polarizability tensor (ndarray, angstrom^3)
        """
        self.error_flag = False

        # Finit different of d(mu)/dE
        d_mu = np.zeros((3, 3))
        p_mu = np.zeros((2, 3, 3))

        dt1 = datetime.datetime.now()
        utils.radon_print('Psi4 polarizability calculation (finite field) is running...', level=1)

        # Dipole moment calculation by perturbed SCF
        # Multiprocessing
        if mp > 0 or const.mpi4py_avail:
            args = []
            utils.picklable(self.mol)
            wfn_copy = self.wfn
            self.wfn = None
            cc2wfn_copy = self.cc2wfn
            self.cc2wfn = None

            c = utils.picklable_const()
            for e in [eps, -eps]:
                for ax in ['x', 'y', 'z']:
                    args.append([e, ax, self, c])

            # mpi4py
            if const.mpi4py_avail:
                utils.radon_print('Parallel method: mpi4py')
                with MPIPoolExecutor(max_workers=mp) as executor:
                    results = executor.map(_polar_mp_worker, args)
                    for i, res in enumerate(results):
                        p_mu[(i // 3), (i % 3)] = res[0]
                        if res[1]: self.error_flag = True

            # concurrent.futures
            else:
                utils.radon_print('Parallel method: concurrent.futures.ProcessPoolExecutor')
                if mp == 1:
                    for i, arg in enumerate(args):
                        with confu.ProcessPoolExecutor(max_workers=1, mp_context=MP.get_context('spawn')) as executor:
                            results = executor.map(_polar_mp_worker, [arg])
                            for res in results:
                                p_mu[(i // 3), (i % 3)] = res[0]
                                if res[1]: self.error_flag = True
                else:
                    with confu.ProcessPoolExecutor(max_workers=mp, mp_context=MP.get_context('spawn')) as executor:
                        results = executor.map(_polar_mp_worker, args)
                        for i, res in enumerate(results):
                            p_mu[(i // 3), (i % 3)] = res[0]
                            if res[1]: self.error_flag = True

            utils.restore_picklable(self.mol)
            self.wfn = wfn_copy
            self.cc2wfn = cc2wfn_copy

        # Sequential
        else:
            pmol = self._init_psi4('symmetry c1')
            psi4.set_options({
                'perturb_h': True,
                'perturb_with': 'dipole'
                })
            for i, e in enumerate([eps, -eps]):
                for j, ax in enumerate(['x', 'y', 'z']):
                    try:
                        psi4.core.set_output_file('./%s_psi4_polar_%s%i.log' % (self.name, ax, i), False)
                        divec = [0.0, 0.0, 0.0]
                        divec[j] = e
                        psi4.set_options({'perturb_dipole': divec})
                        energy_x, wfn = psi4.energy(self.method, molecule=pmol, return_wfn=True, **kwargs)
                        psi4.oeprop(wfn, 'DIPOLE')
                        if LooseVersion(psi4.__version__) < LooseVersion('1.3.100'):
                            p_mu[i, j, 0] = psi4.variable('SCF DIPOLE X') / const.au2debye
                            p_mu[i, j, 1] = psi4.variable('SCF DIPOLE Y') / const.au2debye
                            p_mu[i, j, 2] = psi4.variable('SCF DIPOLE Z') / const.au2debye
                        else:
                            p_mu[i, j] = np.array(psi4.variable('SCF DIPOLE'))

                    except psi4.SCFConvergenceError as e:
                        utils.radon_print('Psi4 SCF convergence error. %s' % e, level=2)
                        p_mu[i, j] = np.array([np.nan, np.nan, np.nan])
                        self.error_flag = True

                    except BaseException as e:
                        self._fin_psi4()
                        self.error_flag = True
                        utils.radon_print('Error termination of psi4 polarizability calculation (finite field). %s' % e, level=3)
            self._fin_psi4()

        a_conv = 1.648777e-41    # a.u. -> C^2 m^2 J^-1
        pv = (a_conv*const.m2ang**3)/(4*np.pi*const.eps0)    # C^2 m^2 J^-1 -> angstrom^3 (polarizability volume)

        d_mu = -(p_mu[0] - p_mu[1]) / (2*eps) * pv
        alpha = np.mean(np.diag(d_mu))

        if self.error_flag:
            utils.radon_print('Psi4 polarizability calculation (finite field) failure.', level=2)
        else:
            dt2 = datetime.datetime.now()
            utils.radon_print('Normal termination of psi4 polarizability calculation (finite field). Elapsed time = %s' % str(dt2-dt1), level=1)

        return alpha, d_mu


    def cc2_polar(self, omega=[], unit='nm', method='cc2', **kwargs):
        """
        psi4w.cc2_polar

        Computation of dipole polarizability by coupled cluster linear response calculation

        Optional args:
            omega: Computation of dynamic polarizability at the wave lengths (float, list)
            unit: Unit of omega (str; nm, au, ev, or hz)
            method: Coupled cluster method (cc2 | ccsd)
            cache_level: 

        Returns:
            Static or dynamic dipole polarizability (ndarray, angstrom^3)
        """
            
        pmol = self._init_psi4(output='./%s_psi4_cc2polar.log' % self.name)
        if len(omega) > 0:
            omega.append(unit)
            psi4.set_options({'omega': omega})

        dt1 = datetime.datetime.now()
        utils.radon_print('Psi4 polarizability calculation (CC linear response) is running...', level=1)

        try:
            energy, self.cc2wfn = psi4.properties(method, properties=['polarizability'], molecule=pmol, return_wfn=True, **kwargs)
            dt2 = datetime.datetime.now()
            utils.radon_print('Normal termination of psi4 polarizability calculation (CC linear response). Elapsed time = %s' % str(dt2-dt1), level=1)

        except psi4.SCFConvergenceError as e:
            utils.radon_print('Psi4 SCF convergence error. %s' % e, level=2)
            self.error_flag = True
            return [np.nan]

        except BaseException as e:
            self._fin_psi4()
            self.error_flag = True
            utils.radon_print('Error termination of psi4 polarizability calculation (CC linear response). %s' % e, level=3)

        self._fin_psi4()

        a_conv = 1.648777e-41 # a.u. -> C^2 m^2 J^-1
        pv = (a_conv*const.m2ang**3)/(4*np.pi*const.eps0) # C^2 m^2 J^-1 -> angstrom^3 (polarizability volume)

        alpha = []
        if len(omega) == 0:
            if method == 'cc2' or method == 'CC2':
                alpha.append( psi4.variable('CC2 DIPOLE POLARIZABILITY @ INF NM') * pv )
            elif method == 'ccsd' or method == 'CCSD':
                alpha.append( psi4.variable('CCSD DIPOLE POLARIZABILITY @ INF NM') * pv )
        elif len(omega) > 0:
            for i in range(len(omega)-1):
                if unit == 'NM' or unit == 'nm':
                    lamda = round(omega[i])
                elif unit == 'AU' or unit == 'au':
                    lamda = round( (const.h * const.c) / (omega[i] * const.au2ev * const.e) * 1e9 )
                elif unit == 'EV' or unit == 'ev':
                    lamda = round( (const.h * const.c) / (omega[i] * const.au2kj) * 1e6 )
                elif unit == 'HZ' or unit == 'hz':
                    lamda = round( const.c / omega[i] * 1e9 )
                else:
                    utils.radon_print('Illeagal input of unit = %s in cc2_polar.' % str(unit), level=3)

                if method == 'cc2' or method == 'CC2':
                    alpha.append( psi4.variable('CC2 DIPOLE POLARIZABILITY @ %iNM' % lamda) * pv )
                elif method == 'ccsd' or method == 'CCSD':
                    alpha.append( psi4.variable('CCSD DIPOLE POLARIZABILITY @ %iNM' % lamda) * pv )

        return np.array(alpha)


    def cphf_polar(self, **kwargs):
        """
        psi4w.cphf_polar

        Computation of dipole polarizability by linear response CPHF/CPKS calculation

        Returns:
            Static dipole polarizability (ndarray, angstrom^3)
        """            
        pmol = self._init_psi4(output='./%s_psi4_cphfpolar.log' % self.name)

        dt1 = datetime.datetime.now()
        utils.radon_print('Psi4 polarizability calculation (CPHF/CPKS) is running...', level=1)

        try:
            energy, self.wfn = psi4.properties(self.method, properties=['DIPOLE_POLARIZABILITIES'], molecule=pmol, return_wfn=True, **kwargs)
            dt2 = datetime.datetime.now()
            utils.radon_print('Normal termination of psi4 polarizability calculation (CPHF/CPKS). Elapsed time = %s' % str(dt2-dt1), level=1)

        except psi4.SCFConvergenceError as e:
            utils.radon_print('Psi4 SCF convergence error. %s' % e, level=2)
            pol = np.full((6), np.nan)
            self.error_flag = True

        except BaseException as e:
            self._fin_psi4()
            self.error_flag = True
            utils.radon_print('Error termination of psi4 polarizability calculation (CPHF/CPKS). %s' % e, level=3)

        self._fin_psi4()

        a_conv = 1.648777e-41 # a.u. -> C^2 m^2 J^-1
        pv = (a_conv*const.m2ang**3)/(4*np.pi*const.eps0) # C^2 m^2 J^-1 -> angstrom^3 (polarizability volume)

        pol = np.array([psi4.variable('DIPOLE POLARIZABILITY %s' % ax) * pv for ax in ['XX', 'YY', 'ZZ', 'XY', 'XZ', 'YZ']])
        alpha = np.mean(pol[:3])
        tensor = np.array([[pol[0], pol[3], pol[4]], [pol[3], pol[1], pol[5]], [pol[4], pol[5], pol[2]]])

        return alpha, tensor


    # Experimental
    def cphf_hyperpolar(self, eps=1e-4, mp=0, **kwargs):
        """
        psi4w.polar

        Computation of first dipole hyperpolarizability by finite field and CPHF/CPKS hybrid

        Optional args:
            eps: Epsilon of finite field
            mp: Number of multiprocessing

        Return:
            First dipole hyperpolarizability (float, a.u.)
            First hyperpolarizability tensor (ndarray, a.u.)
        """
        self.error_flag = False
        pmol = self._init_psi4('symmetry c1')

        dt1 = datetime.datetime.now()
        utils.radon_print('Psi4 first hyperpolarizability calculation (CPHF/CPKS & finite field) is running...', level=1)

        # Calculate Non-perturbed dipole moment
        np_mu = np.zeros((3))
        try:
            psi4.core.set_output_file('./%s_psi4_hyperpolar.log' % (self.name), False)
            energy_x, wfn = psi4.energy(self.method, molecule=pmol, return_wfn=True, **kwargs)
            psi4.oeprop(wfn, 'DIPOLE')
            if LooseVersion(psi4.__version__) < LooseVersion('1.3.100'):
                np_mu[0] = psi4.variable('SCF DIPOLE X') / const.au2debye
                np_mu[1] = psi4.variable('SCF DIPOLE Y') / const.au2debye
                np_mu[2] = psi4.variable('SCF DIPOLE Z') / const.au2debye
            else:
                np_mu = np.array(psi4.variable('SCF DIPOLE'))

        except psi4.SCFConvergenceError as e:
            utils.radon_print('Psi4 SCF convergence error. %s' % e, level=2)
            self.error_flag = True
            np_mu = np.full((3), np.nan)

        except BaseException as e:
            self._fin_psi4()
            self.error_flag = True
            utils.radon_print('Error termination of psi4 first hyperpolarizability calculation (CPHF/CPKS & finite field). %s' % e, level=3)

        self._fin_psi4()


        # Finit different of polarizability d(alpha)/dE
        tensor = np.zeros((3, 3, 3))
        p_alpha = np.zeros((2, 3, 3, 3))

        # Polarizability calculation by perturbed SCF
        # Multiprocessing
        if mp > 0 or const.mpi4py_avail:
            args = []
            utils.picklable(self.mol)
            wfn_copy = self.wfn
            self.wfn = None
            cc2wfn_copy = self.cc2wfn
            self.cc2wfn = None

            c = utils.picklable_const()
            for e in [eps, -eps]:
                for ax in ['x', 'y', 'z']:
                    args.append([e, ax, self, c])

            # mpi4py
            if const.mpi4py_avail:
                utils.radon_print('Parallel method: mpi4py')
                with MPIPoolExecutor(max_workers=mp) as executor:
                    results = executor.map(_cphf_hyperpolar_mp_worker, args)
                    for i, res in enumerate(results):
                        p_alpha[(i // 3), (i % 3)] = res[0]
                        if res[1]: self.error_flag = True

            # concurrent.futures
            else:
                utils.radon_print('Parallel method: concurrent.futures.ProcessPoolExecutor')
                with confu.ProcessPoolExecutor(max_workers=mp, mp_context=MP.get_context('spawn')) as executor:
                    results = executor.map(_cphf_hyperpolar_mp_worker, args)
                    for i, res in enumerate(results):
                        p_alpha[(i // 3), (i % 3)] = res[0]
                        if res[1]: self.error_flag = True

            utils.restore_picklable(self.mol)
            self.wfn = wfn_copy
            self.cc2wfn = cc2wfn_copy

        # Sequential
        else:
            pmol = self._init_psi4('symmetry c1')
            psi4.set_options({
                'perturb_h': True,
                'perturb_with': 'dipole'
                })

            for i, e in enumerate([eps, -eps]):
                for j, ax in enumerate(['x', 'y', 'z']):
                    try:
                        psi4.core.set_output_file('./%s_psi4_hyperpolar_%s%i.log' % (self.name, ax, i), False)
                        divec = [0.0, 0.0, 0.0]
                        divec[j] = e
                        psi4.set_options({'perturb_dipole': divec})
                        energy_x, wfn = psi4.properties(self.method, properties=['DIPOLE_POLARIZABILITIES'], molecule=pmol, return_wfn=True)
                        p_alpha[i, j, 0, 0] = psi4.variable('DIPOLE POLARIZABILITY XX')
                        p_alpha[i, j, 1, 1] = psi4.variable('DIPOLE POLARIZABILITY YY')
                        p_alpha[i, j, 2, 2] = psi4.variable('DIPOLE POLARIZABILITY ZZ')
                        p_alpha[i, j, 0, 1] = p_alpha[i, j, 1, 0] = psi4.variable('DIPOLE POLARIZABILITY XY')
                        p_alpha[i, j, 0, 2] = p_alpha[i, j, 2, 0] = psi4.variable('DIPOLE POLARIZABILITY XZ')
                        p_alpha[i, j, 1, 2] = p_alpha[i, j, 2, 1] = psi4.variable('DIPOLE POLARIZABILITY YZ')

                    except psi4.SCFConvergenceError as e:
                        utils.radon_print('Psi4 SCF convergence error. %s' % e, level=2)
                        p_alpha[i, j] = np.full((3,3), np.nan)
                        self.error_flag = True

                    except BaseException as e:
                        self._fin_psi4()
                        self.error_flag = True
                        utils.radon_print('Error termination of psi4 first hyperpolarizability calculation (CPHF/CPKS & finite field). %s' % e, level=3)
            self._fin_psi4()

        tensor = -(p_alpha[0] - p_alpha[1]) / (2*eps)
        b_x = tensor[0,0,0] + tensor[0,1,1] + tensor[0,2,2] + tensor[1,0,1] + tensor[2,0,2] + tensor[1,1,0] + tensor[2,2,0]
        b_y = tensor[1,1,1] + tensor[1,0,0] + tensor[1,2,2] + tensor[0,1,0] + tensor[2,1,2] + tensor[0,0,1] + tensor[2,2,1]
        b_z = tensor[2,2,2] + tensor[2,0,0] + tensor[2,1,1] + tensor[0,2,0] + tensor[1,2,1] + tensor[0,0,2] + tensor[1,1,2]
        beta = np.sqrt(b_x**2 + b_y**2 + b_z**2)

        if self.error_flag:
            utils.radon_print('Psi4 first hyperpolarizability calculation (CPHF/CPKS & finite field) failure.', level=2)
        else:
            dt2 = datetime.datetime.now()
            utils.radon_print(
                'Normal termination of psi4 first hyperpolarizability calculation (CPHF/CPKS & finite field). Elapsed time = %s' % str(dt2-dt1),
                level=1)

        return beta, tensor


    @property
    def homo(self):
        """
        Psi4w.homo

        Returns:
            HOMO (float, eV)
        """
        if self.wfn is None: return np.nan
        else: return self.wfn.epsilon_a_subset('AO', 'ALL').np[self.wfn.nalpha() - 1] * const.au2ev # Hartree -> eV


    @property
    def lumo(self):
        """
        Psi4w.lumo

        Returns:
            LUMO (float, eV)
        """
        if self.wfn is None: return np.nan
        return self.wfn.epsilon_a_subset('AO', 'ALL').np[self.wfn.nalpha()] * const.au2ev # Hartree -> eV


    @property
    def dipole(self):
        """
        Psi4w.dipole

        Returns:
            dipole vector (float, ndarray, debye)
        """
        if self.wfn is None: return np.nan
        psi4.oeprop(self.wfn, 'DIPOLE')
        if LooseVersion(psi4.__version__) < LooseVersion('1.3.100'):
            x = psi4.variable('SCF DIPOLE X')
            y = psi4.variable('SCF DIPOLE Y')
            z = psi4.variable('SCF DIPOLE Z')
            mu = np.array([x, y, z])
        else:
            mu = np.array(psi4.variable('SCF DIPOLE')) * const.au2debye

        return mu


    @property
    def quadrupole(self):
        """
        Psi4w.quadrupole

        Returns:
            quadrupole xx, yy, zz, xy, xz, yz (float, ndarray, debye ang)
        """
        if self.wfn is None: return np.nan
        psi4.oeprop(self.wfn, 'QUADRUPOLE')
        if LooseVersion(psi4.__version__) < LooseVersion('1.3.100'):
            xx = psi4.variable('SCF QUADRUPOLE XX')
            yy = psi4.variable('SCF QUADRUPOLE YY')
            zz = psi4.variable('SCF QUADRUPOLE ZZ')
            xy = psi4.variable('SCF QUADRUPOLE XY')
            xz = psi4.variable('SCF QUADRUPOLE XZ')
            yz = psi4.variable('SCF QUADRUPOLE YZ')
            quad = np.array([xx, yy, zz, xy, xz, yz])
        else:
            quad = psi4.variable('SCF QUADRUPOLE')

        return quad 


    @property
    def wiberg_bond_index(self):
        """
        Psi4w.wiberg_bond_index

        Returns:
            wiberg bond index (float, ndarray)
        """
        if self.wfn is None: return np.nan
        psi4.oeprop(self.wfn, 'WIBERG_LOWDIN_INDICES')
        return None


    @property
    def mayer_bond_index(self):
        """
        Psi4w.mayer_bond_index

        Returns:
            mayer bond index (float, ndarray)
        """
        if self.wfn is None: return np.nan
        psi4.oeprop(self.wfn, 'MAYER_INDICES')
        return None
        

    @property
    def natural_orbital_occ(self):
        """
        Psi4w.mayer_bond_index

        Returns:
            mayer bond index (float, ndarray)
        """
        if self.wfn is None: return np.nan
        psi4.oeprop(self.wfn, 'NO_OCCUPATIONS')
        no = self.wfn.no_occupations()
        return np.array(no)
        

    @property
    def total_energy(self):
        """
        Psi4w.total_energy

        Returns:
            DFT total energy (float)
        """
        return float(psi4.variable('DFT TOTAL ENERGY'))


    @property
    def scf_energy(self):
        """
        Psi4w.scf_energy

        Returns:
            DFT scf energy (float)
        """
        return float(psi4.variable('SCF TOTAL ENERGY'))


    @property
    def xc_energy(self):
        """
        Psi4w.xc_energy

        Returns:
            DFT xc energy (float)
        """
        return float(psi4.variable('DFT XC ENERGY'))
        

    @property
    def dispersion_energy(self):
        """
        Psi4w.dispersion_energy

        Returns:
            Dispersion correction energy (float)
        """
        return float(psi4.variable('DISPERSION CORRECTION ENERGY'))
        

    @property
    def dh_energy(self):
        """
        Psi4w.dh_energy

        Returns:
            Double hybrid correction energy (float)
        """
        return float(psi4.variable('DOUBLE-HYBRID CORRECTION ENERGY'))
        

    @property
    def NN_energy(self):
        """
        Psi4w.NN_energy

        Returns:
            Nuclear repulsion energy (float)
        """
        return float(psi4.variable('NUCLEAR REPULSION ENERGY'))
        

    @property
    def e1_energy(self):
        """
        Psi4w.e1_energy

        Returns:
            One-electron energy (float)
        """
        return float(psi4.variable('ONE-ELECTRON ENERGY'))
        

    @property
    def e2_energy(self):
        """
        Psi4w.e2_energy

        Returns:
            Two-electron energy (float)
        """
        return float(psi4.variable('TWO-ELECTRON ENERGY'))
        

    @property
    def cc2_energy(self):
        """
        Psi4w.cc2_energy

        Returns:
            CC2 total energy (float)
        """
        return float(psi4.variable('CC2 TOTAL ENERGY'))
        

    @property
    def cc2_corr_energy(self):
        """
        Psi4w.cc2_corr_energy

        Returns:
            CC2 correlation energy (float)
        """
        return float(psi4.variable('CC2 CORRELATION ENERGY'))
        

    @property
    def ccsd_energy(self):
        """
        Psi4w.ccsd_energy

        Returns:
            CCSD total energy (float)
        """
        return float(psi4.variable('CCSD TOTAL ENERGY'))
        

    @property
    def ccsd_corr_energy(self):
        """
        Psi4w.ccsd_corr_energy

        Returns:
            CCSD correlation energy (float)
        """
        return float(psi4.variable('CCSD CORRELATION ENERGY'))
        

    @property
    def ccsd_t_energy(self):
        """
        Psi4w.ccsd_t_energy

        Returns:
            CCSD(T) total energy (float)
        """
        return float(psi4.variable('CCSD(T) TOTAL ENERGY'))
        

    @property
    def ccsd_t_corr_energy(self):
        """
        Psi4w.ccsd_t_corr_energy

        Returns:
            CCSD(T) correlation energy (float)
        """
        return float(psi4.variable('CCSD(T) CORRELATION ENERGY'))
        

def _polar_mp_worker(args):
    eps, ax, psi4obj, c = args
    utils.restore_const(c)
    
    i = 0 if eps > 0 else 1
    j = 0 if ax == 'x' else 1 if ax == 'y' else 2 if ax == 'z' else np.nan
    error_flag = False

    utils.radon_print('Worker process %s%i start on %s. PID: %i' % (ax, i, socket.gethostname(), os.getpid()))

    utils.restore_picklable(psi4obj.mol)
    pmol = psi4obj._init_psi4('symmetry c1', output='./%s_psi4_polar_%s%i.log' % (psi4obj.name, ax, i))
    divec = [0.0, 0.0, 0.0]
    divec[j] = eps
    psi4.set_options({
        'perturb_h': True,
        'perturb_with': 'dipole',
        'perturb_dipole': divec
        })

    dipole = np.zeros((3))

    try:
        energy_x, wfn = psi4.energy(psi4obj.method, molecule=pmol, return_wfn=True)
        psi4.oeprop(wfn, 'DIPOLE')
        if LooseVersion(psi4.__version__) < LooseVersion('1.3.100'):
            dipole[0] = psi4.variable('SCF DIPOLE X') / const.au2debye
            dipole[1] = psi4.variable('SCF DIPOLE Y') / const.au2debye
            dipole[2] = psi4.variable('SCF DIPOLE Z') / const.au2debye
        else:
            dipole = np.array(psi4.variable('SCF DIPOLE'))

    except psi4.SCFConvergenceError as e:
        utils.radon_print('Psi4 SCF convergence error. %s' % e, level=2)
        dipole = np.full((3), np.nan)
        error_flag = True

    except BaseException as e:
        psi4obj._fin_psi4()
        error_flag = True
        utils.radon_print('Error termination of psi4 polarizability calculation (finite field). %s' % e, level=3)

    psi4obj._fin_psi4()

    return dipole, error_flag


def _cphf_hyperpolar_mp_worker(args):
    eps, ax, psi4obj, c = args
    utils.restore_const(c)

    i = 0 if eps > 0 else 1
    j = 0 if ax == 'x' else 1 if ax == 'y' else 2 if ax == 'z' else np.nan
    error_flag = False

    utils.radon_print('Worker process %s%i start on %s. PID: %i' % (ax, i, socket.gethostname(), os.getpid()))

    utils.restore_picklable(psi4obj.mol)
    pmol = psi4obj._init_psi4('symmetry c1', output='./%s_psi4_hyperpolar_%s%i.log' % (psi4obj.name, ax, i))
    divec = [0.0, 0.0, 0.0]
    divec[j] = eps
    psi4.set_options({
        'perturb_h': True,
        'perturb_with': 'dipole',
        'perturb_dipole': divec
        })

    alpha = np.zeros((3,3))

    try:
        energy_x, wfn = psi4.properties(psi4obj.method, properties=['DIPOLE_POLARIZABILITIES'], molecule=pmol, return_wfn=True)
        alpha[0, 0] = psi4.variable('DIPOLE POLARIZABILITY XX')
        alpha[1, 1] = psi4.variable('DIPOLE POLARIZABILITY YY')
        alpha[2, 2] = psi4.variable('DIPOLE POLARIZABILITY ZZ')
        alpha[0, 1] = alpha[1, 0] = psi4.variable('DIPOLE POLARIZABILITY XY')
        alpha[0, 2] = alpha[2, 0] = psi4.variable('DIPOLE POLARIZABILITY XZ')
        alpha[1, 2] = alpha[2, 1] = psi4.variable('DIPOLE POLARIZABILITY YZ')

    except psi4.SCFConvergenceError as e:
        utils.radon_print('Psi4 SCF convergence error. %s' % e, level=2)
        alpha = np.full((3,3), np.nan)
        error_flag = True

    except BaseException as e:
        psi4obj._fin_psi4()
        error_flag = True
        utils.radon_print('Error termination of psi4 first hyperpolarizability calculation (CPHF/CPKS & finite field). %s' % e, level=3)

    psi4obj._fin_psi4()

    return alpha, error_flag


# Override function for qcengine.config.get_global
from typing import Any, Dict, Optional, Union
_global_values = None
def _override_get_global(key: Optional[str] = None) -> Union[str, Dict[str, Any]]:
    import psutil
    import getpass
    import cpuinfo

    # TODO (wardlt): Implement a means of getting CPU information from compute nodes on clusters for MPI tasks
    #  The QC code runs on a different node than the node running this Python function, which may have different info

    global _global_values
    if _global_values is None:
        _global_values = {}
        _global_values["hostname"] = socket.gethostname()
        _global_values["memory"] = round(psutil.virtual_memory().available / (1024 ** 3), 3)
        _global_values["username"] = getpass.getuser()

        # Work through VMs and logical cores.
        if hasattr(psutil.Process(), "cpu_affinity"):
            cpu_cnt = len(psutil.Process().cpu_affinity())

            # For mpi4py
            if const.mpi4py_avail and cpu_cnt == 1:
                cpu_cnt = psutil.cpu_count(logical=False)
                if cpu_cnt is None:
                   cpu_cnt = psutil.cpu_count(logical=True)

        else:
            cpu_cnt = psutil.cpu_count(logical=False)
            if cpu_cnt is None:
                cpu_cnt = psutil.cpu_count(logical=True)

        _global_values["ncores"] = cpu_cnt
        _global_values["nnodes"] = 1

        _global_values["cpuinfo"] = cpuinfo.get_cpu_info()
        try:
            _global_values["cpu_brand"] = _global_values["cpuinfo"]["brand_raw"]
        except KeyError:
            try:
                # Remove this if py-cpuinfo is pinned to >=6.0.0
                _global_values["cpu_brand"] = _global_values["cpuinfo"]["brand"]
            except KeyError:
                # Assuming Fugaku
                _global_values["cpu_brand"] = 'A64FX'

    if key is None:
        return _global_values.copy()
    else:
        return _global_values[key]

