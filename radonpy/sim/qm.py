#  Copyright (c) 2022. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# ******************************************************************************
# sim.qm module
# ******************************************************************************

import numpy as np
import os
import json
import gc
from rdkit import Chem
from rdkit import Geometry as Geom
from ..core import utils, const, calc
from .qm_wrapper import QMw

__version__ = '0.2.9'


def assign_charges(mol, charge='RESP', confId=0, opt=True, work_dir=None, tmp_dir=None, log_name='charge', qm_solver='psi4',
    opt_method='wb97m-d3bj', opt_basis='6-31G(d,p)', geom_iter=50, geom_conv='QCHEM', geom_algorithm='RFO',
    charge_method='HF', charge_basis='6-31G(d)', charge_basis_gen={'Br':'6-31G(d)', 'I': 'lanl2dz'},
    total_charge=None, total_multiplicity=None, **kwargs):
    """
    sim.qm.assign_charges

    Assignment atomic charge for RDKit Mol object
    This is wrapper function of core.calc.assign_charges

    Args:
        mol: RDKit Mol object

    Optional args:
        charge: Select charge type of gasteiger, RESP, ESP, Mulliken, Lowdin, or zero (str, default:RESP)
        confID: Target conformer ID (int)
        opt: Do optimization (boolean)
        work_dir: Work directory path (str)
        omp: Num of threads of OpenMP in the quantum chemical calculation (int)
        memory: Using memory in the quantum chemical calculation (int, MB)
        opt_method: Using method in the optimize calculation (str, default:wb97m-d3bj)
        opt_basis: Using basis set in the optimize calculation (str, default:6-31G(d,p))
        opt_basis_gen: Using basis set in the optimize calculation for each element
        charge_method: Using method in the charge calculation (str, default:HF)
        charge_basis: Using basis set in the charge calculation (str, default:6-31G(d))
        charge_basis_gen: Using basis set in the charge calculation for each element

    Returns:
        boolean
    """
    flag = calc.assign_charges(mol, charge=charge, confId=confId, opt=opt, work_dir=work_dir, tmp_dir=tmp_dir, log_name=log_name, qm_solver=qm_solver,
            opt_method=opt_method, opt_basis=opt_basis, geom_iter=geom_iter, geom_conv=geom_conv, geom_algorithm=geom_algorithm,
            charge_method=charge_method, charge_basis=charge_basis, charge_basis_gen=charge_basis_gen,
            total_charge=total_charge, total_multiplicity=total_multiplicity, **kwargs)

    return flag
        

def conformation_search(mol, ff=None, nconf=1000, dft_nconf=4, etkdg_ver=2, rmsthresh=0.5, tfdthresh=0.02, clustering='TFD', qm_solver='psi4',
    opt_method='wb97m-d3bj', opt_basis='6-31G(d,p)', opt_basis_gen={'Br': '6-31G(d,p)', 'I': 'lanl2dz'},
    geom_iter=50, geom_conv='QCHEM', geom_algorithm='RFO', log_name='mol', solver='lammps', solver_path=None, work_dir=None, tmp_dir=None,
    etkdg_omp=-1, psi4_omp=-1, psi4_mp=0, omp=1, mpi=-1, gpu=0, mm_mp=0, memory=1000, total_charge=None, total_multiplicity=None, **kwargs):
    """
    sim.qm.conformation_search

    Conformation search
    This is wrapper function of core.calc.conformation_search

    Args:
        mol: RDKit Mol object
    
    Optional args:
        ff: Force field instance. If None, MMFF94 optimization is carried out by RDKit
        nconf: Number of generating conformations (int)
        dft_nconf: Number of conformations for DFT optimization (int, default:4)
        solver: lammps (str)
        solver_path: File path of solver (str)
        work_dir: Path of work directory (str)
        etkdg_omp: Number of threads of OpenMP in ETKDG of RDkit (int)
        omp: Number of threads of OpenMP in LAMMPS (int)
        mpi: Number of MPI process in LAMMPS (int)
        gpu: Number of GPU in LAMMPS (int)
        mm_mp: Number of parallel execution of LAMMPS or RDKit in MM optimization (int)
        psi4_omp: Number of threads of OpenMP in Psi4 (int)
        psi4_mp: Number of parallel execution of Psi4 in DFT optimization (int)
        opt_method: Using method in the optimize calculation (str, default:wb97m-d3bj)
        opt_basis: Using basis set in the optimize calculation (str, default:6-31G(d,p))
        opt_basis_gen: Using basis set in the optimize calculation for each element

    Returns:
        RDKit Mol object
        DFT and MM energy (ndarray, kcal/mol)
    """
    mol, energy = calc.conformation_search(mol, ff=ff, nconf=nconf, dft_nconf=dft_nconf, etkdg_ver=etkdg_ver, rmsthresh=rmsthresh, qm_solver=qm_solver,
                tfdthresh=tfdthresh, clustering=clustering, opt_method=opt_method, opt_basis=opt_basis,
                opt_basis_gen=opt_basis_gen, geom_iter=geom_iter, geom_conv=geom_conv, geom_algorithm=geom_algorithm, log_name=log_name,
                solver=solver, solver_path=solver_path, work_dir=work_dir, tmp_dir=tmp_dir,
                etkdg_omp=etkdg_omp, psi4_omp=psi4_omp, psi4_mp=psi4_mp, omp=omp, mpi=mpi, gpu=gpu, mm_mp=mm_mp, memory=memory,
                total_charge=total_charge, total_multiplicity=total_multiplicity, **kwargs)

    return mol, energy


def sp_prop(mol, confId=0, opt=True, work_dir=None, tmp_dir=None, log_name='sp_prop', qm_solver='psi4',
    opt_method='wb97m-d3bj', opt_basis='6-31G(d,p)', opt_basis_gen={'Br': '6-31G(d,p)', 'I': 'lanl2dz'}, 
    geom_iter=50, geom_conv='QCHEM', geom_algorithm='RFO',
    sp_method='wb97m-d3bj', sp_basis='6-311G(d,p)', sp_basis_gen={'Br': '6-311G(d,p)', 'I': 'lanl2dz'},
    total_charge=None, total_multiplicity=None, **kwargs):
    """
    sim.qm.sp_prop

    Calculation of total energy, HOMO, LUMO, dipole moment by QM calculation

    Args:
        mol: RDKit Mol object
    
    Optional args:
        confID: Target conformer ID (int)
        opt: Do optimization (boolean)
        work_dir: Work directory path (str)
        omp: Num of threads of OpenMP in the quantum chemical calculation (int)
        memory: Using memory in the quantum chemical calculation (int, MB)
        opt_method: Using method in the optimize calculation (str, default:wb97m-d3bj)
        opt_basis: Using basis set in the optimize calculation (str, default:6-31G(d,p))
        opt_basis_gen: Using basis set in the optimize calculation for each element
        sp_method: Using method in the single point calculation (str, default:wb97m-d3bj)
        sp_basis: Using basis set in the single point calculation (str, default:6-311G(2d,p))
        opt_basis_gen: Using basis set in the single point calculation for each element

    return
        dict
            qm_total_energy (float, kJ/mol)
            qm_homo (float, eV)
            qm_lumo (float, eV)
            qm_dipole (x, y, z) (float, Debye)
    """
    e_prop = {}
    
    if type(total_charge) is int:
        kwargs['charge'] = total_charge
    if type(total_multiplicity) is int:
        kwargs['multiplicity'] = total_multiplicity

    psi4mol = QMw(mol, confId=confId, work_dir=work_dir, tmp_dir=tmp_dir, method=opt_method, basis=opt_basis, basis_gen=opt_basis_gen, qm_solver=qm_solver,
                    name=log_name, **kwargs)
    if opt:
        psi4mol.optimize(geom_iter=geom_iter, geom_conv=geom_conv, geom_algorithm=geom_algorithm)
        if psi4mol.error_flag:
            utils.radon_print('Psi4 optimization error in sim.qm.sp_prop.', level=2)
            return e_prop

        coord = psi4mol.mol.GetConformer(confId).GetPositions()
        for i, atom in enumerate(psi4mol.mol.GetAtoms()):
            mol.GetConformer(confId).SetAtomPosition(i, Geom.Point3D(coord[i, 0], coord[i, 1], coord[i, 2]))

    psi4mol.method = sp_method
    psi4mol.basis = sp_basis
    psi4mol.basis_gen = sp_basis_gen

    e_prop['qm_total_energy'] = psi4mol.energy()
    e_prop['qm_homo'] = psi4mol.homo
    e_prop['qm_lumo'] = psi4mol.lumo
    e_prop['qm_dipole_x'], e_prop['qm_dipole_y'], e_prop['qm_dipole_z'] = psi4mol.dipole

    del psi4mol
    gc.collect()

    return e_prop


def polarizability(mol, confId=0, opt=True, work_dir=None, tmp_dir=None, log_name='polarizability', qm_solver='psi4', mp=0,
    opt_method='wb97m-d3bj', opt_basis='6-31G(d,p)', opt_basis_gen={'Br': '6-31G(d,p)', 'I': 'lanl2dz'}, 
    geom_iter=50, geom_conv='QCHEM', geom_algorithm='RFO',
    polar_method='wb97m-d3bj', polar_basis='6-311+G(2d,p)', polar_basis_gen={'Br': '6-311G(d,p)', 'I': 'lanl2dz'},
    total_charge=None, total_multiplicity=None, **kwargs):
    """
    sim.qm.polarizability

    Calculation of dipole polarizability by QM calculation

    Args:
        mol: RDKit Mol object
    
    Optional args:
        confID: Target conformer ID (int)
        opt: Do optimization (boolean)
        work_dir: Work directory path (str)
        omp: Num of threads of OpenMP in the quantum chemical calculation (int)
        memory: Using memory in the quantum chemical calculation (int, MB)
        opt_method: Using method in the optimize calculation (str, default:wb97m-d3bj)
        opt_basis: Using basis set in the optimize calculation (str, default:6-31G(d,p))
        opt_basis_gen: Using basis set in the optimize calculation for each element
        polar_method: Using method in the polarizability calculation (str, default:wb97m-d3bj)
        polar_basis: Using basis set in the polarizability calculation (str, default:6-311+G(2d,p))
        polar_basis_gen: Using basis set in the polarizability calculation for each element

    return
        dict
            Dipole polarizability (float, angstrom^3)
            Polarizability tensor (xx, yy, zz, xy, xz, yz) (float, angstrom^3)
    """
    polar_data = {}

    if type(total_charge) is int:
        kwargs['charge'] = total_charge
    if type(total_multiplicity) is int:
        kwargs['multiplicity'] = total_multiplicity

    psi4mol = QMw(mol, confId=confId, work_dir=work_dir, tmp_dir=tmp_dir, method=opt_method, basis=opt_basis, basis_gen=opt_basis_gen, qm_solver=qm_solver,
                    name=log_name, **kwargs)
    if opt:
        psi4mol.optimize(geom_iter=geom_iter, geom_conv=geom_conv, geom_algorithm=geom_algorithm)
        if psi4mol.error_flag:
            utils.radon_print('Psi4 optimization error in calc.polarizability.', level=2)
            return polar_data

        coord = psi4mol.mol.GetConformer(confId).GetPositions()
        for i, atom in enumerate(psi4mol.mol.GetAtoms()):
            mol.GetConformer(confId).SetAtomPosition(i, Geom.Point3D(coord[i, 0], coord[i, 1], coord[i, 2]))

    psi4mol.method = polar_method
    psi4mol.basis = polar_basis
    psi4mol.basis_gen = polar_basis_gen

    alpha, d_mu = psi4mol.polar(mp=mp)
    if psi4mol.error_flag:
        utils.radon_print('Psi4 polarizability calculation error in sim.qm.polarizability.', level=2)

    polar_data = {
        'qm_polarizability': alpha,
        'qm_polarizability_xx': d_mu[0, 0],
        'qm_polarizability_yy': d_mu[1, 1],
        'qm_polarizability_zz': d_mu[2, 2],
        'qm_polarizability_xy': (d_mu[0, 1]+d_mu[1, 0])/2,
        'qm_polarizability_xz': (d_mu[0, 2]+d_mu[2, 0])/2,
        'qm_polarizability_yz': (d_mu[1, 2]+d_mu[2, 1])/2,
    }

    del psi4mol
    gc.collect()

    return polar_data


def refractive_index(mols, density, ratio=None, confId=0, opt=True, work_dir=None, tmp_dir=None, log_name='refractive_index', qm_solver='psi4', mp=0,
        opt_method='wb97m-d3bj', opt_basis='6-31G(d,p)', opt_basis_gen={'Br': '6-31G(d,p)', 'I': 'lanl2dz'}, 
        geom_iter=50, geom_conv='QCHEM', geom_algorithm='RFO',
        polar_method='wb97m-d3bj', polar_basis='6-311+G(2d,p)', polar_basis_gen={'Br': '6-311G(d,p)', 'I': 'lanl2dz'},
        total_charge=None, total_multiplicity=None, **kwargs):
    """
    sim.qm.refractive_index

    Calculation of refractive index by QM calculation

    Args:
        mols: List of RDKit Mol object
        density: [g/cm^3]
    
    Optional args:
        ratio: ratio of repeating units in a copolymer
        confID: Target conformer ID (int)
        opt: Do optimization (boolean)
        work_dir: Work directory path (str)
        omp: Num of threads of OpenMP in the quantum chemical calculation (int)
        memory: Using memory in the quantum chemical calculation (int, MB)
        opt_method: Using method in the optimize calculation (str, default:wb97m-d3bj)
        opt_basis: Using basis set in the optimize calculation (str, default:6-31G(d,p))
        opt_basis_gen: Using basis set in the optimize calculation for each element
        polar_method: Using method in the polarizability calculation (str, default:wb97m-d3bj)
        polar_basis: Using basis set in the polarizability calculation (str, default:6-311+G(2d,p))
        polar_basis_gen: Using basis set in the polarizability calculation for each element

    return
        Refractive index data (dict)
            refractive_index (float)
            polarizability of repeating units (float, angstrom^3)
            polarizability tensor of repeating units (float, angstrom^3)
    """
    ri_data = {}

    if type(mols) is Chem.Mol: mols = [mols]
    mol_weight = [calc.molecular_weight(mol) for mol in mols]
    a_list = []
    
    for i, mol in enumerate(mols):
        polar_data = polarizability(mol, confId=confId, opt=opt, work_dir=work_dir, tmp_dir=tmp_dir, log_name='%s_%i' % (log_name, i), qm_solver=qm_solver, mp=mp,
                            opt_method=opt_method, opt_basis=opt_basis, opt_basis_gen=opt_basis_gen, 
                            geom_iter=geom_iter, geom_conv=geom_conv, geom_algorithm=geom_algorithm,
                            polar_method=polar_method, polar_basis=polar_basis, polar_basis_gen=polar_basis_gen,
                            total_charge=total_charge, total_multiplicity=total_multiplicity, **kwargs)

        a_list.append(polar_data['qm_polarizability'])
        for k in polar_data.keys(): ri_data['%s_monomer%i' % (k, i+1)] = polar_data[k]

    ri_data['refractive_index'] = calc.refractive_index(a_list, density, mol_weight, ratio=ratio)

    return ri_data


# Experimental
def abbe_number_cc2(mol, density, confId=0, opt=True, work_dir=None, tmp_dir=None, log_name='abbe_number_cc2',
        opt_method='wb97m-d3bj', opt_basis='6-31G(d,p)', opt_basis_gen={'Br': '6-31G(d,p)', 'I': 'lanl2dz'}, 
        geom_iter=50, geom_conv='QCHEM', geom_algorithm='RFO',
        polar_basis='6-311+G(2d,p)', polar_basis_gen={'Br': '6-311G(d,p)', 'I': 'lanl2dz'},
        total_charge=None, total_multiplicity=None, **kwargs):
    """
    sim.qm.abbe_number_cc2

    Calculation of abbe's number by CC2 calculation

    Args:
        mol: RDKit Mol object
        density: [g/cm^3]
    
    Optional args:
        confID: Target conformer ID (int)
        opt: Do optimization (boolean)
        work_dir: Work directory path (str)
        omp: Num of threads of OpenMP in the quantum chemical calculation (int)
        memory: Using memory in the quantum chemical calculation (int, MB)
        opt_method: Using method in the optimize calculation (str, defaultcam-b3lyp-d3bj)
        opt_basis: Using basis set in the optimize calculation (str, default:6-31G(d,p))
        opt_basis_gen: Using basis set in the optimize calculation for each element
        polar_basis: Using basis set in the dynamic polarizability calculation (str, default:6-311+G(2d,p))
        polar_basis_gen: Using basis set in the polarizability calculation for each element

    return
        Abbe's number data (dict)
            abbe_number (float)
            refractive_index_656 (float)
            refractive_index_589 (float)
            refractive_index_486 (float)
            polarizability_656 (float, angstrom^3)
            polarizability_589 (float, angstrom^3)
            polarizability_486 (float, angstrom^3)
    """
    abbe_data = {}

    if type(total_charge) is int:
        kwargs['charge'] = total_charge
    if type(total_multiplicity) is int:
        kwargs['multiplicity'] = total_multiplicity

    mol_weight = calc.molecular_weight(mol)

    psi4mol = QMw(mol, confId=confId, work_dir=work_dir, tmp_dir=tmp_dir, method=opt_method, basis=opt_basis, basis_gen=opt_basis_gen,
                    name=log_name, **kwargs)
    if opt:
        psi4mol.optimize(geom_iter=geom_iter, geom_conv=geom_conv, geom_algorithm=geom_algorithm)
        if psi4mol.error_flag:
            utils.radon_print('Psi4 optimization error in sim.qm.abbe_number_cc2.', level=2)
            return abbe_data

        coord = psi4mol.mol.GetConformer(confId).GetPositions()
        for j, atom in enumerate(psi4mol.mol.GetAtoms()):
            mol.GetConformer(confId).SetAtomPosition(j, Geom.Point3D(coord[j, 0], coord[j, 1], coord[j, 2]))

    psi4mol.basis = polar_basis
    psi4mol.basis_gen = polar_basis_gen

    alpha = psi4mol.cc2_polar(omega=[656, 589, 486])

    n_656 = calc.refractive_index(alpha[0], density, mol_weight)
    n_589 = calc.refractive_index(alpha[1], density, mol_weight)
    n_486 = calc.refractive_index(alpha[2], density, mol_weight)

    abbe_data = {
        'abbe_number_cc2': (n_589 - 1)/(n_486 - n_656),
        'refractive_index_cc2_656': n_656,
        'refractive_index_cc2_589': n_589,
        'refractive_index_cc2_486': n_486,
        'qm_polarizability_cc2_656': alpha[0],
        'qm_polarizability_cc2_589': alpha[1],
        'qm_polarizability_cc2_486': alpha[2],
    }

    del psi4mol
    gc.collect()

    return abbe_data


# Experimental
def polar_sos(res, wavelength=None):
    """
    sim.qm.polar_sos

    Calculation of static/dynamic electric dipole polarizability by sum-over-states approach using TD-DFT results
    J. Phys. Chem. A 2004, 108, 11063-11072

    Args:
        res: Results of TD-DFT calculation
    
    Optional args:
        wavelength: wavelength [nm]. If None, static dipole polarizability is computed. (float)

    return
        Polarizability (float, angstrom^3)
        Polarizability tensor (ndarray, angstrom^3)
    """
    a_conv = 1.648777e-41    # a.u. -> C^2 m^2 J^-1
    pv = (a_conv*const.m2ang**3)/(4*np.pi*const.eps0)    # C^2 m^2 J^-1 -> angstrom^3 (polarizability volume)

    E = np.array([r['EXCITATION ENERGY'] for r in res])
    mu = np.array([r['ELECTRIC DIPOLE TRANSITION MOMENT (LEN)'] for r in res])
    
    if wavelength is None:
        tensor = 2*np.sum( (mu[:, np.newaxis, :] * mu[:, :, np.newaxis]) / E.reshape((-1,1,1)), axis=0 ) * pv
    else:
        Ep = const.h*const.c/(wavelength*1e-9) / 4.3597447222071e-18    # (J s) * (m/s) / (nm->m) = J -> hartree
        tensor = 2*np.sum( (mu[:, np.newaxis, :] * mu[:, :, np.newaxis]) / (E - (Ep**2)/E).reshape((-1,1,1)), axis=0 ) * pv
        
    alpha = np.mean(np.diag(tensor))
    
    return alpha, tensor


def polarizability_sos(mol, wavelength=None, confId=0, opt=True, work_dir=None, tmp_dir=None, log_name='polarizability_sos',
        opt_method='wb97m-d3bj', opt_basis='6-31G(d,p)', opt_basis_gen={'Br': '6-31G(d,p)', 'I': 'lanl2dz'},
        geom_iter=50, geom_conv='QCHEM', geom_algorithm='RFO',
        td_method='cam-b3lyp-d3bj', td_basis='6-311+G(2d,p)', td_basis_gen={'Br': '6-311G(d,p)', 'I': 'lanl2dz'},
        n_state=1000, p_state=None, tda=False, tdscf_maxiter=60, td_output='polarizability_sos_tddft.json',
        total_charge=None, total_multiplicity=None, **kwargs):
    """
    sim.qm.polarizability_sos

    Calculation of static/dynemic electric dipole polarizability by using TD-DFT calculation

    Args:
        mol: RDKit Mol object
    
    Optional args:
        wavelength: wavelength [nm]. If None, static dipole polarizability is computed.
        confID: Target conformer ID (int)
        opt: Do optimization (boolean)
        work_dir: Work directory path (str)
        omp: Num of threads of OpenMP in the quantum chemical calculation (int)
        memory: Using memory in the quantum chemical calculation (int, MB)
        opt_method: Using method in the optimize calculation (str, default:wb97m-d3bj)
        opt_basis: Using basis set in the optimize calculation (str, default:6-31G(d,p))
        opt_basis_gen: Using basis set in the optimize calculation for each element
        td_method: Using method in the TD-DFT calculation (str, default:wb97m-d3bj)
        td_basis: Using basis set in the TD-DFT calculation (str, default:6-311+G(2d,p))
        td_basis_gen: Using basis set in the TD-DFT calculation for each element
        n_state: Number of state in the TD-DFT calculation
        p_state: Number of states, which is determined by [Num. of all excitation states] * p_state (float, 0.0 < p_state <= 1.0).
                 p_state is given priority over n_state.
        tda: Run with Tamm-Dancoff approximation (TDA), uses random-phase approximation (RPA) when false (boolean)
        tdscf_maxiter: Maximum number of TDSCF solver iterations (int)

    return
        list of dict
            Frequency dependent dipole polarizability (float, angstrom^3)
            Frequency dependent dipole polarizability tensor (xx, yy, zz, xy, xz, yz) (float, angstrom^3)
    """
    polar_data = {}
    if wavelength is None:
        wavelength = [None]
    elif type(wavelength) is float or type(wavelength) is int:
        wavelength = [wavelength]

    if type(total_charge) is int:
        kwargs['charge'] = total_charge
    if type(total_multiplicity) is int:
        kwargs['multiplicity'] = total_multiplicity

    psi4mol = QMw(mol, confId=confId, work_dir=work_dir, tmp_dir=tmp_dir, method=opt_method, basis=opt_basis, basis_gen=opt_basis_gen,
                    name=log_name, **kwargs)
    if opt:
        psi4mol.optimize(geom_iter=geom_iter, geom_conv=geom_conv, geom_algorithm=geom_algorithm)
        if psi4mol.error_flag:
            utils.radon_print('Psi4 optimization error in calc.polarizability_sos.', level=2)
            return polar_data

        coord = psi4mol.mol.GetConformer(confId).GetPositions()
        for i, atom in enumerate(psi4mol.mol.GetAtoms()):
            mol.GetConformer(confId).SetAtomPosition(i, Geom.Point3D(coord[i, 0], coord[i, 1], coord[i, 2]))

    psi4mol.method = td_method
    psi4mol.basis = td_basis
    psi4mol.basis_gen = td_basis_gen

    res = psi4mol.tddft(n_state=n_state, p_state=p_state, tda=tda, tdscf_maxiter=tdscf_maxiter)

    if psi4mol.error_flag:
        utils.radon_print('Psi4 TD-DFT calculation error in sim.qm.polarizability_sos.', level=2)

    for l in wavelength:
        alpha, tensor = polar_sos(res, wavelength=l)

        if l is None:
            p_data = {
                'qm_polarizability_sos': alpha,
                'qm_polarizability_sos_xx': tensor[0, 0],
                'qm_polarizability_sos_yy': tensor[1, 1],
                'qm_polarizability_sos_zz': tensor[2, 2],
                'qm_polarizability_sos_xy': (tensor[0, 1]+tensor[1, 0])/2,
                'qm_polarizability_sos_xz': (tensor[0, 2]+tensor[2, 0])/2,
                'qm_polarizability_sos_yz': (tensor[1, 2]+tensor[2, 1])/2,
            }
            polar_data.update(p_data)

        else:
            p_data = {
                'qm_polarizability_sos_%i' % int(l): alpha,
                'qm_polarizability_sos_%i_xx' % int(l): tensor[0, 0],
                'qm_polarizability_sos_%i_yy' % int(l): tensor[1, 1],
                'qm_polarizability_sos_%i_zz' % int(l): tensor[2, 2],
                'qm_polarizability_sos_%i_xy' % int(l): (tensor[0, 1]+tensor[1, 0])/2,
                'qm_polarizability_sos_%i_xz' % int(l): (tensor[0, 2]+tensor[2, 0])/2,
                'qm_polarizability_sos_%i_yz' % int(l): (tensor[1, 2]+tensor[2, 1])/2,
            }
            polar_data.update(p_data)

    if td_output:
        json_data = {}
        for i, r in enumerate(res):
            r['ELECTRIC DIPOLE TRANSITION MOMENT (LEN)'] = ','.join(str(x) for x in r['ELECTRIC DIPOLE TRANSITION MOMENT (LEN)'])
            r['ELECTRIC DIPOLE TRANSITION MOMENT (VEL)'] = ','.join(str(x) for x in r['ELECTRIC DIPOLE TRANSITION MOMENT (VEL)'])
            r['MAGNETIC DIPOLE TRANSITION MOMENT'] = ','.join(str(x) for x in r['MAGNETIC DIPOLE TRANSITION MOMENT'])
            del r['RIGHT EIGENVECTOR ALPHA']
            del r['LEFT EIGENVECTOR ALPHA']
            del r['RIGHT EIGENVECTOR BETA']
            del r['LEFT EIGENVECTOR BETA']
            json_data['Excitation state %i' % (i+1)] = r

        with open(os.path.join(work_dir, td_output), 'w') as fh:
            json.dump(json_data, fh, ensure_ascii=False, indent=4, separators=(',', ': '))

    del psi4mol
    gc.collect()

    return polar_data


def refractive_index_sos(mols, density, ratio=None, wavelength=None, confId=0, opt=True, work_dir=None, tmp_dir=None, log_name='refractive_index_sos',
        opt_method='wb97m-d3bj', opt_basis='6-31G(d,p)', opt_basis_gen={'Br': '6-31G(d,p)', 'I': 'lanl2dz'},
        geom_iter=50, geom_conv='QCHEM', geom_algorithm='RFO',
        td_method='cam-b3lyp-d3bj', td_basis='6-311+G(2d,p)', td_basis_gen={'Br': '6-311G(d,p)', 'I': 'lanl2dz'},
        n_state=1000, p_state=None, tda=False, tdscf_maxiter=60, td_output='refractive_index_sos_tddft.json',
        total_charge=None, total_multiplicity=None, **kwargs):
    """
    sim.qm.refractive_index_sos

    Calculation of refractive index by sum-over-states approach using TD-DFT calculation

    Args:
        mols: List of RDKit Mol object
        density: [g/cm^3]
    
    Optional args:
        ratio: ratio of repeating units in a copolymer
        wavelength: wavelength [nm]. If None, static dipole polarizability is computed.
        confID: Target conformer ID (int)
        opt: Do optimization (boolean)
        work_dir: Work directory path (str)
        omp: Num of threads of OpenMP in the quantum chemical calculation (int)
        memory: Using memory in the quantum chemical calculation (int, MB)
        opt_method: Using method in the optimize calculation (str, defaultcam-b3lyp-d3bj)
        opt_basis_gen: Using basis set in the optimize calculation for each element
        opt_basis: Using basis set in the optimize calculation (str, default:6-31G(d,p))
        td_method: Using method in the TD-DFT calculation (str, default:cam-b3lyp-d3bj)
        td_basis: Using basis set in the TD-DFT calculation (str, default:6-311+G(2d,p))
        td_basis_gen: Using basis set in the TD-DFT calculation for each element
        n_state: Number of state in the TD-DFT calculation
        p_state: Number of states, which is determined by [Num. of all excitation states] * p_state (float, 0.0 < p_state <= 1.0).
                 p_state is given priority over n_state.
        tda: Run with Tamm-Dancoff approximation (TDA), uses random-phase approximation (RPA) when false (boolean)
        tdscf_maxiter: Maximum number of TDSCF solver iterations (int)

    return
        Refractive index data (list of dict)
            frequency dependent refractive index (float)
            frequency dependent dipole polarizability of repeating units (float, angstrom^3)
            frequency dependent dipole polarizability tensor of repeating units (float, angstrom^3)
    """
    if wavelength is None:
        wavelength = [None]
    elif type(wavelength) is float or type(wavelength) is int:
        wavelength = [wavelength]

    if type(mols) is Chem.Mol: mols = [mols]
    mol_weight = [calc.molecular_weight(mol) for mol in mols]

    p_data = {}
    a_list = []
    for i, mol in enumerate(mols):
        polar_data = polarizability_sos(mol, wavelength=wavelength, confId=confId, opt=opt, work_dir=work_dir, tmp_dir=tmp_dir,
                                log_name='%s_%i' % (log_name, i), opt_method=opt_method, opt_basis=opt_basis, opt_basis_gen=opt_basis_gen,
                                geom_iter=geom_iter, geom_conv=geom_conv, geom_algorithm=geom_algorithm,
                                td_method=td_method, td_basis=td_basis, td_basis_gen=td_basis_gen,
                                n_state=n_state, p_state=p_state, tda=tda, tdscf_maxiter=tdscf_maxiter, td_output=td_output,
                                total_charge=total_charge, total_multiplicity=total_multiplicity, **kwargs)

        for k, v in polar_data.items():
            p_data['%s_monomer%i' % (k, i+1)] = v

        a_tmp = []
        for l in wavelength:
            if l is None:
                a_tmp.append(polar_data['qm_polarizability_sos'])
            else:
                a_tmp.append(polar_data['qm_polarizability_sos_%i' % int(l)])
        a_list.append(a_tmp)
    a_list = np.array(a_list)

    ri_data = {}
    for i, l in enumerate(wavelength):
        if i is None:
            ri_data['refractive_index_sos'] = calc.refractive_index(a_list[:, i], density, mol_weight, ratio=ratio)
        else:
            ri_data['refractive_index_sos_%i' % int(l)] = calc.refractive_index(a_list[:, i], density, mol_weight, ratio=ratio)

    ri_data.update(p_data)

    return ri_data


def abbe_number_sos(mols, density, ratio=None, confId=0, opt=True, work_dir=None, tmp_dir=None, log_name='abbe_number_sos',
        opt_method='wb97m-d3bj', opt_basis='6-31G(d,p)', opt_basis_gen={'Br': '6-31G(d,p)', 'I': 'lanl2dz'},
        geom_iter=50, geom_conv='QCHEM', geom_algorithm='RFO',
        td_method='cam-b3lyp-d3bj', td_basis='6-311+G(2d,p)', td_basis_gen={'Br': '6-311G(d,p)', 'I': 'lanl2dz'},
        n_state=1000, p_state=0.003, tda=False, tdscf_maxiter=60, td_output='abbe_number_sos_tddft.json',
        total_charge=None, total_multiplicity=None, **kwargs):
    """
    sim.qm.abbe_number_sos

    Calculation of abbe's number by sum-over-states approach using TD-DFT calculation

    Args:
        mols: List of RDKit Mol object
        density: [g/cm^3]
    
    Optional args:
        ratio: ratio of repeating units in a copolymer
        confID: Target conformer ID (int)
        opt: Do optimization (boolean)
        work_dir: Work directory path (str)
        omp: Num of threads of OpenMP in the quantum chemical calculation (int)
        memory: Using memory in the quantum chemical calculation (int, MB)
        opt_method: Using method in the optimize calculation (str, defaultcam-b3lyp-d3bj)
        opt_basis_gen: Using basis set in the optimize calculation for each element
        opt_basis: Using basis set in the optimize calculation (str, default:6-31G(d,p))
        td_method: Using method in the TD-DFT calculation (str, default:cam-b3lyp-d3bj)
        td_basis: Using basis set in the TD-DFT calculation (str, default:6-311+G(2d,p))
        td_basis_gen: Using basis set in the TD-DFT calculation for each element
        n_state: Number of state in the TD-DFT calculation
        p_state: Number of states, which is determined by [Num. of all excitation states] * p_state (float, 0.0 < p_state <= 1.0).
                 p_state is given priority over n_state.
        tda: Run with Tamm-Dancoff approximation (TDA), uses random-phase approximation (RPA) when false (boolean)
        tdscf_maxiter: Maximum number of TDSCF solver iterations (int)

    return
        Abbe's number data (dict)
            abbe_number (float)
            refractive_index_656 (float)
            refractive_index_589 (float)
            refractive_index_486 (float)
            frequency dependent dipole polarizability of repeating units (float, angstrom^3)
            frequency dependent dipole polarizability tensor of repeating units (float, angstrom^3)
    """
    if type(mols) is Chem.Mol: mols = [mols]
    mol_weight = [calc.molecular_weight(mol) for mol in mols]

    ri_data = refractive_index_sos(mols, density=density, ratio=ratio, wavelength=[656, 589, 486], confId=confId,
                            opt=opt, work_dir=work_dir, tmp_dir=tmp_dir, log_name=log_name,
                            opt_method=opt_method, opt_basis=opt_basis, opt_basis_gen=opt_basis_gen,
                            geom_iter=geom_iter, geom_conv=geom_conv, geom_algorithm=geom_algorithm,
                            td_method=td_method, td_basis=td_basis, td_basis_gen=td_basis_gen,
                            n_state=n_state, p_state=p_state, tda=tda, tdscf_maxiter=tdscf_maxiter, td_output=td_output,
                            total_charge=total_charge, total_multiplicity=total_multiplicity, **kwargs)

    n_656 = ri_data['refractive_index_sos_656']
    n_589 = ri_data['refractive_index_sos_589']
    n_486 = ri_data['refractive_index_sos_486']

    abbe_data = {
        'abbe_number_sos': (n_589 - 1)/(n_486 - n_656),
        **ri_data
    }

    return abbe_data

