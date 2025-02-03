#  Copyright (c) 2022. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# ******************************************************************************
# core.calc module
# ******************************************************************************

import numpy as np
import math
import socket
import os
from math import sqrt, sin, cos
from copy import deepcopy
from scipy.stats import maxwell
import multiprocessing as MP
import concurrent.futures as confu
import gc
from decimal import Decimal
from fractions import Fraction
from rdkit import Chem
from rdkit.Chem import AllChem, TorsionFingerprints
from rdkit import Geometry as Geom
from rdkit.ML.Cluster import Butina
from . import utils, const

__version__ = '0.2.10'

MD_avail = True
try:
    from ..sim import md
except ImportError:
    MD_avail = False

qm_avail = True
try:
    from ..sim.qm_wrapper import QMw
except ImportError:
    qm_avail = False

if const.tqdm_disable:
    tqdm = utils.tqdm_stub
else:
    try:
        from tqdm.autonotebook import tqdm
    except ImportError:
        tqdm = utils.tqdm_stub

if const.mpi4py_avail:
    try:
        from mpi4py.futures import MPIPoolExecutor
    except ImportError as e:
        utils.radon_print('Cannot import mpi4py. Change to const.mpi4py_avail = False. %s' % e, level=2)
        const.mpi4py_avail = False


def fix_trans(coord):
    # Centering
    m = np.mean(coord, axis=0)
    tx = coord - m
    return tx


def fix_rot(coord, ref):
    # Kabsch algorithm
    H = np.dot(coord.T, ref)
    U, s, Vt = np.linalg.svd(H, full_matrices=True)
    d = (np.linalg.det(Vt) * np.linalg.det(U)) < 0.0
    if d:
        Vt[:, -1] = -Vt[:, -1]
    R = np.dot(U, Vt)
    
    # Rotate
    tx = np.dot(coord, R)
    return tx


def rmsd(coord, ref):
    x = fix_trans(coord)
    x = fix_rot(x, ref)
    diff = np.array(ref) - np.array(x)
    N = len(x)
    return np.sqrt(np.sum(diff**2, axis=1) / N)


def set_coord(atom):
    atom.x = atom.GetDoubleProp('x')
    atom.y = atom.GetDoubleProp('y')
    atom.z = atom.GetDoubleProp('z')
    return atom
    
    
def centering_mol(mol, confId=0):
    coord = mol.GetConformer(confId).GetPositions()
    coord = fix_trans(coord)

    for i in range(mol.GetNumAtoms()):
        mol.GetConformer(confId).SetAtomPosition(i, Geom.Point3D(coord[i, 0], coord[i, 1], coord[i, 2]))

    return coord


def fix_rot_mol(mol, ref, confId=0):
    coord = mol.GetConformer(confId).GetPositions()
    coord = fix_rot(coord, ref)
    
    for i in range(mol.GetNumAtoms()):
        mol.GetConformer(confId).SetAtomPosition(i, Geom.Point3D(coord[i, 0], coord[i, 1], coord[i, 2]))

    return coord


def distance(a, b):
    a = set_coord(a)
    b = set_coord(b)
    dis = sqrt((b.x - a.x)**2 + (b.y - a.y)**2 + (b.z - a.z)**2)
    
    return dis
    
    
def distance_matrix(coord1, coord2=None):
    coord1 = np.array(coord1)
    coord2 = np.array(coord2) if coord2 is not None else coord1
    return np.sqrt(np.sum((coord1[:, np.newaxis, :] - coord2[np.newaxis, :, :])**2, axis=-1))


def distance_matrix_mol(mol1, mol2=None):
    coord1 = np.array(mol1.GetConformer(0).GetPositions())
    coord2 = np.array(mol2.GetConformer(0).GetPositions()) if mol2 is not None else coord1
    return np.sqrt(np.sum((coord1[:, np.newaxis, :] - coord2[np.newaxis, :, :])**2, axis=-1))


def angle(a, b, c, rad=False):
    a = set_coord(a)
    b = set_coord(b)
    c = set_coord(c)

    va = np.array([a.x-b.x, a.y-b.y, a.z-b.z])
    vc = np.array([c.x-b.x, c.y-b.y, c.z-b.z])

    return angle_vec(va, vc, rad=rad)


def angle_coord(a, b, c, rad=False):
    va = np.array([a[0]-b[0], a[1]-b[1], a[2]-b[2]])
    vc = np.array([c[0]-b[0], c[1]-b[1], c[2]-b[2]])

    return angle_vec(va, vc, rad=rad)


def angle_vec(va, vc, rad=False):
    vcos = np.dot(va, vc) / (np.linalg.norm(va) * np.linalg.norm(vc))
    if vcos > 1: vcos = 1
    elif vcos < -1: vcos = -1

    angle = np.arccos(vcos)
    angle = angle if rad else math.degrees(angle)

    return angle


def angle_vec_matrix(vec1, vec2, rad=False):
    vcos = np.matmul(vec1, vec2.T) / (np.linalg.norm(vec1, axis=1) * np.linalg.norm(vec2, axis=1))
    vcos = np.where(vcos > 1, 1, vcos)
    vcos = np.where(vcos < -1, -1, vcos)
    angle = np.arccos(vcos)

    return angle if rad else angle*(180/np.pi)


def find_liner_angle(mol, confId=0):
    coord = mol.GetConformer(confId).GetPositions()

    for p in mol.GetAtoms():
        b = p.GetIdx()

        for p1 in p.GetNeighbors():
            a = p1.GetIdx()

            for p2 in p.GetNeighbors():
                c = p2.GetIdx()

                if a == c:
                    continue

                ang = angle_coord(coord[a], coord[b], coord[c])
                if 175 < np.abs(ang) <= 180:
                    return True
    return False


def dihedral(a, b, c, d, rad=False):
    a = set_coord(a)
    b = set_coord(b)
    c = set_coord(c)
    d = set_coord(d)

    va = np.array([a.x-b.x, a.y-b.y, a.z-b.z])
    vb = np.array([b.x-c.x, b.y-c.y, b.z-c.z])
    vc = np.array([c.x-b.x, c.y-b.y, c.z-b.z])
    vd = np.array([d.x-c.x, d.y-c.y, d.z-c.z])
        
    return dihedral_vec(va, vc, vb, vd, rad=rad)


def dihedral_coord(a, b, c, d, rad=False):
    va = np.array([a[0]-b[0], a[1]-b[1], a[2]-b[2]])
    vb = np.array([b[0]-c[0], b[1]-c[1], b[2]-c[2]])
    vc = np.array([c[0]-b[0], c[1]-b[1], c[2]-b[2]])
    vd = np.array([d[0]-c[0], d[1]-c[1], d[2]-c[2]])

    return dihedral_vec(va, vc, vb, vd, rad=rad)


def dihedral_vec(va, vc, vb, vd, rad=False):
    vcross1 = np.cross(va, vc)
    vcross2 = np.cross(vb, vd)
    vcos = np.dot(vcross1, vcross2) / (np.linalg.norm(vcross1) * np.linalg.norm(vcross2))

    if vcos > 1: vcos = 1
    elif vcos < -1: vcos = -1

    vcross3 = np.cross(vcross1, vcross2)
    rot = 1 if np.dot(vc, vcross3) >= 0 else -1

    dih = np.arccos(vcos) * rot
    dih = dih if rad else math.degrees(dih)
    
    return dih


def rotate_rod(coord, vec, theta, center=np.array([0., 0., 0.])):
    coord_tr = np.array(coord - center)
    vec = vec / np.linalg.norm(vec)

    rot = np.array([
        [cos(theta)+(1-cos(theta))*vec[0]**2, vec[0]*vec[1]*(1-cos(theta))-vec[2]*sin(theta), vec[0]*vec[2]*(1-cos(theta))+vec[1]*sin(theta)],
        [vec[1]*vec[0]*(1-cos(theta))+vec[2]*sin(theta), cos(theta)+(1-cos(theta))*vec[1]**2, vec[1]*vec[2]*(1-cos(theta))-vec[0]*sin(theta)],
        [vec[2]*vec[0]*(1-cos(theta))-vec[1]*sin(theta), vec[2]*vec[1]*(1-cos(theta))+vec[0]*sin(theta), cos(theta)+(1-cos(theta))*vec[2]**2],
    ])

    coord_rot = np.dot(rot, coord_tr.T)

    return np.array(coord_rot.T) + center


def grad_lj_rigid_tr(coord1, coord2, sigma=2.7, epsilon=0.1):
    delta1 = (coord1[:, np.newaxis, :] - coord2[np.newaxis, :, :])
    dist_matrix = distance_matrix(coord1, coord2)
    lj_c = -24 * epsilon * ( (2 * sigma**12 * dist_matrix**(-14)) - (sigma**6 * dist_matrix**(-8)) )
    lj_grad_tr = np.sum(lj_c[:, :, np.newaxis] * delta1, axis=1)
    return lj_grad_tr


def wrap(coord, xhi, xlo, yhi, ylo, zhi, zlo):
    wcoord = deepcopy(coord)
    if xhi is not None: wcoord[:, 0] = np.where(wcoord[:, 0] > xhi, wcoord[:, 0] - (xhi - xlo), wcoord[:, 0])
    if xlo is not None: wcoord[:, 0] = np.where(wcoord[:, 0] < xlo, wcoord[:, 0] + (xhi - xlo), wcoord[:, 0])
    if yhi is not None: wcoord[:, 1] = np.where(wcoord[:, 1] > yhi, wcoord[:, 1] - (yhi - ylo), wcoord[:, 1])
    if ylo is not None: wcoord[:, 1] = np.where(wcoord[:, 1] < ylo, wcoord[:, 1] + (yhi - ylo), wcoord[:, 1])
    if zhi is not None: wcoord[:, 2] = np.where(wcoord[:, 2] > zhi, wcoord[:, 2] - (zhi - zlo), wcoord[:, 2])
    if zlo is not None: wcoord[:, 2] = np.where(wcoord[:, 2] < zlo, wcoord[:, 2] + (zhi - zlo), wcoord[:, 2])

    return wcoord


def wrap_mol(mol, confId=0):
    mol = mol_trans_in_cell(mol)
    coord = np.array(mol.GetConformer(confId).GetPositions())
    return wrap(coord, mol.cell.xhi, mol.cell.xlo, mol.cell.yhi, mol.cell.ylo, mol.cell.zhi, mol.cell.zlo)


def wrap_cubic(coord, length):
    wcoord = deepcopy(coord)
    wcoord = np.where(wcoord > length, wcoord - length*2, wcoord)
    wcoord = np.where(wcoord < -length, wcoord + length*2, wcoord)
    return wcoord


def wrap_cubic_mol(mol, confId=0):
    coord = np.array(mol.GetConformer(confId).GetPositions())
    length = mol.cell.xhi
    return wrap_cubic(coord, length)


def mol_trans_in_cell(mol, confId=0):

    mol_c = utils.deepcopy_mol(mol)
    mol_c = utils.set_mol_id(mol_c)
    n_mol = utils.count_mols(mol_c)
    coord = np.array(mol_c.GetConformer(confId).GetPositions())
    cell_center = [(mol_c.cell.xhi+mol_c.cell.xlo)/2, (mol_c.cell.yhi+mol_c.cell.ylo)/2, (mol_c.cell.zhi+mol_c.cell.zlo)/2]
    mol_coord_list = []

    for i in range(1, n_mol+1):
        mol_coord = []

        for j, atom in enumerate(mol_c.GetAtoms()):
            if i == atom.GetIntProp('mol_id'):
                mol_coord.append(coord[j])

        mol_coord = np.array(mol_coord)
        mol_coord_center = np.mean(mol_coord, axis=0)
        x_shift = round((mol_coord_center[0] - cell_center[0]) / mol_c.cell.dx) * mol_c.cell.dx
        y_shift = round((mol_coord_center[1] - cell_center[1]) / mol_c.cell.dy) * mol_c.cell.dy
        z_shift = round((mol_coord_center[2] - cell_center[2]) / mol_c.cell.dz) * mol_c.cell.dz

        for j, atom in enumerate(mol_c.GetAtoms()):
            if i == atom.GetIntProp('mol_id'):
                coord[j] -= np.array([x_shift, y_shift, z_shift])

    for i in range(mol_c.GetNumAtoms()):
        mol_c.GetConformer(confId).SetAtomPosition(i, Geom.Point3D(coord[i, 0], coord[i, 1], coord[i, 2]))

    return mol_c


def mirror_inversion_mol(mol, confId=0):
    mol_c = utils.deepcopy_mol(mol)
    coord = np.array(mol_c.GetConformer(confId).GetPositions())
    coord[:, 2] = coord[:, 2] * -1.0
    for i in range(mol_c.GetNumAtoms()):
        mol_c.GetConformer(confId).SetAtomPosition(i, Geom.Point3D(coord[i, 0], coord[i, 1], coord[i, 2]))

    return mol_c


def molecular_weight(mol, ignore_linker=True):
    mol_weight = 0.0
    for atom in mol.GetAtoms():
        if ignore_linker and atom.GetSymbol() == "H" and atom.GetIsotope() == 3:
            pass
        else:
            mol_weight += atom.GetMass()

    return mol_weight


def get_num_radicals(mol):
    nr = 0
    for atom in mol.GetAtoms():
        nr += atom.GetNumRadicalElectrons()
    return nr


def assign_charges(mol, charge='gasteiger', confId=0, opt=True, work_dir=None, tmp_dir=None, log_name='charge', qm_solver='psi4',
    opt_method='wb97m-d3bj', opt_basis='6-31G(d,p)', opt_basis_gen={'Br':'6-31G(d)', 'I': 'lanl2dz'}, 
    geom_iter=50, geom_conv='QCHEM', geom_algorithm='RFO',
    charge_method='HF', charge_basis='6-31G(d)', charge_basis_gen={'Br':'6-31G(d)', 'I': 'lanl2dz'},
    total_charge=None, total_multiplicity=None, **kwargs):
    """
    calc.assign_charges

    Assignment atomic charge for RDKit Mol object

    Args:
        mol: RDKit Mol object

    Optional args:
        charge: Select charge type of gasteiger, RESP, ESP, Mulliken, Lowdin, or zero (str, deffault:gasteiger)
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

    if charge == 'zero':
        for p in mol.GetAtoms():
            p.SetDoubleProp('AtomicCharge', 0.0)

    elif charge == 'gasteiger':
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        for p in mol.GetAtoms():
            p.SetDoubleProp('AtomicCharge', float(p.GetProp('_GasteigerCharge')))

    elif charge in ['RESP', 'ESP', 'Mulliken', 'Lowdin']:
        if not qm_avail:
            utils.radon_print('Cannot import psi4_wrapper. You can use psi4_wrapper by "conda install -c psi4 psi4 resp dftd3"', level=3)
            return False

        if type(total_charge) is int:
            kwargs['charge'] = total_charge
        if type(total_multiplicity) is int:
            kwargs['multiplicity'] = total_multiplicity

        psi4mol = QMw(mol, confId=confId, work_dir=work_dir, tmp_dir=tmp_dir, qm_solver=qm_solver, method=opt_method, basis=opt_basis, basis_gen=opt_basis_gen,
                      name=log_name, **kwargs)
        if opt:
            psi4mol.optimize(geom_iter=geom_iter, geom_conv=geom_conv, geom_algorithm=geom_algorithm)
            if psi4mol.error_flag: return False
            coord = psi4mol.mol.GetConformer(confId).GetPositions()
            for i, atom in enumerate(psi4mol.mol.GetAtoms()):
                mol.GetConformer(confId).SetAtomPosition(i, Geom.Point3D(coord[i, 0], coord[i, 1], coord[i, 2]))

        psi4mol.method = charge_method
        psi4mol.basis = charge_basis
        psi4mol.basis_gen = charge_basis_gen

        if charge == 'RESP':
            psi4mol.resp()
            if psi4mol.error_flag: return False
            for i, atom in enumerate(psi4mol.mol.GetAtoms()):
                mol.GetAtomWithIdx(i).SetDoubleProp('RESP', atom.GetDoubleProp('RESP'))
                mol.GetAtomWithIdx(i).SetDoubleProp('ESP', atom.GetDoubleProp('ESP'))
                mol.GetAtomWithIdx(i).SetDoubleProp('AtomicCharge', atom.GetDoubleProp('RESP'))

        elif charge == 'ESP':
            psi4mol.resp()
            if psi4mol.error_flag: return False
            for i, atom in enumerate(psi4mol.mol.GetAtoms()):
                mol.GetAtomWithIdx(i).SetDoubleProp('RESP', atom.GetDoubleProp('RESP'))
                mol.GetAtomWithIdx(i).SetDoubleProp('ESP', atom.GetDoubleProp('ESP'))
                mol.GetAtomWithIdx(i).SetDoubleProp('AtomicCharge', atom.GetDoubleProp('ESP'))

        elif charge == 'Mulliken':
            psi4mol.mulliken_charge(recalc=True)
            if psi4mol.error_flag: return False
            for i, atom in enumerate(psi4mol.mol.GetAtoms()):
                mol.GetAtomWithIdx(i).SetDoubleProp('MullikenCharge', atom.GetDoubleProp('MullikenCharge'))
                mol.GetAtomWithIdx(i).SetDoubleProp('AtomicCharge', atom.GetDoubleProp('MullikenCharge'))

        elif charge == 'Lowdin':
            psi4mol.lowdin_charge(recalc=True)
            if psi4mol.error_flag: return False
            for i, atom in enumerate(psi4mol.mol.GetAtoms()):
                mol.GetAtomWithIdx(i).SetDoubleProp('LowdinCharge', atom.GetDoubleProp('LowdinCharge'))
                mol.GetAtomWithIdx(i).SetDoubleProp('AtomicCharge', atom.GetDoubleProp('LowdinCharge'))

        del psi4mol
        gc.collect()

    else:
        utils.radon_print('%s is not implemented in RadonPy.' % (charge), level=3)
        return False

    return True


def charge_neutralize(mol, tol=1e-5):
    charge = 0.0
    for p in mol.GetAtoms():
        charge += p.GetDoubleProp('AtomicCharge')

    if abs(charge) > tol:
        corr = charge/mol.GetNumAtoms()
        for p in mol.GetAtoms():
            p.SetDoubleProp('AtomicCharge', p.GetDoubleProp('AtomicCharge')-corr)

    return mol


def charge_neutralize2(mol, tol=1e-12, retry=100):
    """
    calc.charge_neutralize

    Charge neutralization

    Args:
        mol: RDKit Mol object

    Optional args:
        tol:
        retry:

    Returns:
        RDKit Mol object
    """

    charge = Decimal('%.16e' % 0.0)
    for p in mol.GetAtoms():
        charge += Decimal('%.16e' % (p.GetDoubleProp('AtomicCharge')))

    if abs(charge) > tol:
        for i in range(retry):
            corr = Fraction(charge) / mol.GetNumAtoms()
            for p in mol.GetAtoms():
                p.SetDoubleProp(
                    'AtomicCharge',
                    float(Decimal('%.16e' % p.GetDoubleProp('AtomicCharge'))-Decimal('%.16e' % float(corr)))
                )

            charge_new = Decimal('%.16e' % 0.0)
            for p in mol.GetAtoms():
                charge_new += Decimal('%.16e' % (p.GetDoubleProp('AtomicCharge')))

            if abs(charge_new) < tol:
                break
            elif charge == charge_new:
                break
            else:
                charge = charge_new

        c_sign = charge.as_tuple()['sign']
        c_digit = list(charge.as_tuple()['digits'])
        c_exp = charge.as_tuple()['exponent']

        if c_digit[-1] == 0:
            del c_digit[-1]
            c_exp += 1

        num = int(''.join(c_digit))


    return mol


def set_velocity(mol, temp, d_type='maxwell'):
    """
    calc.set_velocity

    Randomly generate verosity vector of atoms

    Args:
        mol: RDKit Mol object
        temp: temperature (float, K)

    Optional args:
        d_type: Distribution type (gaussian or maxwell) (str)

    Returns:
        RDKit Mol object

    """

    vel = np.zeros((mol.GetNumAtoms(), 3))

    if d_type == 'gaussian':
        for i, atom in enumerate(mol.GetAtoms()):
            vel[i, 0], vel[i, 1], vel[i, 2] = gaussian_velocity(temp, atom.GetMass())
    elif d_type == 'maxwell':
        for i, atom in enumerate(mol.GetAtoms()):
            vel[i, 0], vel[i, 1], vel[i, 2] = maxwell_velocity(temp, atom.GetMass())

    # Removing translational move
    vel_mean = np.mean(vel, axis=0)
    vel -= vel_mean

    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetDoubleProp('vx', vel[i, 0])
        atom.SetDoubleProp('vy', vel[i, 1])
        atom.SetDoubleProp('vz', vel[i, 2])

    return mol


def gaussian_velocity(temp, mass):
    """
    calc.gaussian_velocity

    Randomly generate verosity vector with gaussian distribution for a particle

    Args:
        temp: temperature (float, K)
        mass: (float, atomic mass unit)

    Returns:
        ndarray of verosity vector (float, angstrom / fs)

    """

    kbT = const.kB*temp*const.NA * 10**(-7)   # Kg m2 / s2  ->  AMU angstrom2 / fs2

    v = np.abs(np.random.normal(loc=sqrt(3*kbT/mass), scale=sqrt(3*kbT/mass)/10))

    theta, phi = np.random.uniform(-np.pi, np.pi, 2)
    vx = v*sin(theta)*cos(phi)
    vy = v*sin(theta)*sin(phi)
    vz = v*cos(theta)

    return np.array([vx, vy, vz])


def maxwell_velocity(temp, mass):
    """
    calc.maxwell_velocity

    Randomly generate verosity vector with maxwell distribution for a particle

    Args:
        temp: temperature (float, K)
        mass: (float, atomic mass unit)

    Returns:
        ndarray of verosity vector (float, angstrom / fs)

    """

    kbT = const.kB*temp*const.NA * 10**(-7)   # Kg m2 / s2  ->  AMU angstrom2 / fs2

    v = maxwell.rvs(scale=sqrt(kbT/mass))
    theta, phi = np.random.uniform(-np.pi, np.pi, 2)
    vx = v*sin(theta)*cos(phi)
    vy = v*sin(theta)*sin(phi)
    vz = v*cos(theta)

    return np.array([vx, vy, vz])


def mol_mass(mol):
    mass = 0
    for atom in mol.GetAtoms():
        mass += atom.GetMass()
    return mass


def mol_density(mol):
    if not hasattr(mol, 'cell'):
        return np.nan
    
    vol = mol.cell.volume * const.ang2cm**3
    mass = mol_mass(mol) / const.NA
    density = mass/vol

    return density


# Experimental
def vdw_volume(mol, confId=0, method='grid', radii='rdkit', gridSpacing=0.2):
    """
    calc.vdw_volume

    method = 'grid'
        Computation of van der Waals volume by grid-based method implemented in RDKit

    method = 'bondi'
        Computation of van der Waals volume by accumulation of spherical segments
        Bondi, A. "Van der Waals volumes and radii". J. Phys. Chem. 68(3):441–451 (1964)

        V_vdw = sum( (4/3)*pi*(R_i**3) - sum((1/3)*pi*(h_ij**2)*(3R_i - h_ij)) )
        h_ij = (R_j**2 - (R_i - d_ij)**2/(2*d_ij)

        R_i: van der Waals radii of atom i
        R_j: van der Waals radii of neighbor atom j
        d_ij: Distance between atom i and neighbor atom j

    Args:
        mol: RDKit mol object

    Optional args:
        confId: Target conformer ID

    Return:
        vdw volume (angstrome**3, float)
    """
    if method == 'grid':
        V_vdw = AllChem.ComputeMolVolume(mol, confId=confId, gridSpacing=gridSpacing)
        return V_vdw

    elif method == 'bondi':
        coord = np.array(mol.GetConformer(confId).GetPositions())
        r = []
        for atom in mol.GetAtoms():
            if radii == 'ff':
                r.append(atom.GetDoubleProp('ff_sigma') * 2**(1/6) / 2)
            elif radii == 'rdkit':
                r.append(Chem.PeriodicTable.GetRvdw(Chem.GetPeriodicTable(), atom.GetAtomicNum()))
        r = np.array(r)

        V_vdw = (4/3)*np.pi*(r**3)

        for atom in mol.GetAtoms():
            r1_vdw = r[atom.GetIdx()]

            for na in atom.GetNeighbors():
                r2_vdw = r[na.GetIdx()]
                d = np.sqrt(np.sum((coord[atom.GetIdx()] - coord[na.GetIdx()])**2))
                h = (r2_vdw**2 - (r1_vdw - d)**2)/(2*d)
                if h <= 0: continue
                v_tmp = (1/3)*np.pi*(h**2)*(3*r1_vdw - h)
                V_vdw[atom.GetIdx()] -= v_tmp
        
        return np.sum(V_vdw)
    
    else:
        utils.radon_print('Illegal input of method = %s' % method, level=3)
        return np.nan


# Experimental
# def vdw_volume2(mol, confId=0, radii='rdkit'):
#     """
#     calc.vdw_volume2

#     Computation of van der Waals volume by accumulation of spherical segments
#     Bondi, A. "Van der Waals volumes and radii". J. Phys. Chem. 68(3):441–451 (1964)

#     V_vdw = sum( (4/3)*pi*(R_i**3) - sum((1/3)*pi*(h_ij**2)*(3R_i - h_ij)) )
#     h_ij = (R_j**2 - (R_i - d_ij)**2/(2*d_ij)

#     R_i: van der Waals radii of atom i
#     R_j: van der Waals radii of neighbor atom j
#     d_ij: Distance between atom i and neighbor atom j

#     Args:
#         mol: RDKit mol object

#     Optional args:
#         confId: Target conformer ID

#     Return:
#         vdw volume (angstrome**3, float)
#     """

#     coord = np.array(mol.GetConformer(confId).GetPositions())
#     d_ij = distance_matrix(coord)
#     bmat = Chem.GetAdjacencyMatrix(mol)
#     r = []
#     for atom in mol.GetAtoms():
#         if radii == 'ff':
#             r.append(atom.GetDoubleProp('ff_sigma') * 2**(1/6) / 2)
#         elif radii == 'rdkit':
#             r.append(Chem.PeriodicTable.GetRvdw(Chem.GetPeriodicTable(), atom.GetAtomicNum()))
#     r = np.array(r)

#     V_vdw = (4/3)*np.pi*np.sum(r**3)
#     with np.errstate(divide='ignore', invalid='ignore'):
#         h_ij = np.where(d_ij > 0.0, (r.reshape(-1,1)**2 - (r - d_ij)**2)/(2*d_ij), 0.0)
#     V_cap = (np.pi/3)*np.where((h_ij > 0.0) & (bmat == 1), (h_ij**2)*(3*r - h_ij), 0.0)
#     V_vdw -= np.sum(V_cap)

#     return V_vdw


# Experimental
def fractional_free_volume(mol, confId=0, gridSpacing=0.2, method='grid'):
    """
    calc.fractional_free_volume
    
    Computation of fractional free volume
    ffv = (V_cell - 1.3*V_vdw) / V_cell

    Args:
        mol: RDKit mol object

    Optional args:
        confId: Target conformer ID
        gridSpacing: Grid spacing of molecular volume computation
        method: bondi or grid

    Return:
        Fractional free volume (float)
    """

    if not hasattr(mol, 'cell'):
        utils.radon_print('The cell attribute of the input Mol object is undefined', level=2)
        return np.nan

    #coord = wrap_mol(mol, confId=confId)

    #conf = Chem.rdchem.Conformer(mol.GetNumAtoms())
    #conf.Set3D(True)
    #for i in range(mol.GetNumAtoms()):
    #    conf.SetAtomPosition(i, Geom.Point3D(coord[i, 0], coord[i, 1], coord[i, 2]))
    #conf_id = mol.AddConformer(conf, assignId=True)

    V_vdw = vdw_volume(mol, confId=confId, method=method, gridSpacing=gridSpacing)
    ffv = (mol.cell.volume - 1.3*V_vdw) / mol.cell.volume

    return ffv


def conformation_search(mol, ff=None, nconf=1000, dft_nconf=0, etkdg_ver=2, rmsthresh=0.5, tfdthresh=0.02, clustering='TFD',
        qm_solver='psi4', opt_method='wb97m-d3bj', opt_basis='6-31G(d,p)', opt_basis_gen={'Br': '6-31G(d,p)', 'I': 'lanl2dz'},
        geom_iter=50, geom_conv='QCHEM', geom_algorithm='RFO', log_name='mol', solver='lammps', solver_path=None, work_dir=None, tmp_dir=None,
        etkdg_omp=-1, psi4_omp=-1, psi4_mp=0, omp=1, mpi=-1, gpu=0, mm_mp=0, memory=1000,
        total_charge=None, total_multiplicity=None, **kwargs):
    """
    calc.conformation_search

    Conformation search

    Args:
        mol: RDKit Mol object
    
    Optional args:
        ff: Force field instance. If None, MMFF94 optimization is carried out by RDKit
        nconf: Number of generating conformations (int)
        dft_nconf: Number of conformations for DFT optimization (int, default:0)
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

    mol_c = utils.deepcopy_mol(mol)

    if etkdg_ver == 3:
        etkdg = AllChem.ETKDGv3()
    elif etkdg_ver == 2:
        etkdg = AllChem.ETKDGv2()
    elif etkdg_ver == 1:
        etkdg = AllChem.ETKDG()
    else:
        utils.radon_print('Illegal input of etkdg_ver = %s' % clustering, level=3)
        return mol_c, np.array([])

    if etkdg_omp < 0:
        etkdg_omp = utils.cpu_count()
    if psi4_omp < 0:
        psi4_omp = utils.cpu_count()
    if mpi < 0 and omp == 1:
        mpi = utils.cpu_count()
    if mpi == 1 and omp < 0:
        omp = utils.cpu_count()

    if tmp_dir is None: tmp_dir = work_dir

    etkdg.pruneRmsThresh = rmsthresh
    etkdg.numThreads = etkdg_omp
    AllChem.EmbedMultipleConfs(mol_c, nconf, etkdg)
    nconf = mol_c.GetNumConformers()
    utils.radon_print('%i conformers were generated.' % nconf)

    energies = []
    dft_energies = []
    e_res = []
    e_dft_res = []

    # Optimization conformers by MM
    utils.radon_print('Start optimization of %i conformers by MM level.' % nconf, level=1)
    if ff and MD_avail:
        # Using LAMMPS
        ff.ff_assign(mol_c, charge='gasteiger')

        # Parallel execution of psi4 optimizations
        #if mm_mp > 0 and mpi > 0 and nconf > 1:
        #    utils.radon_print('Parallel method: MPMD')
        #    mol_c, energy, _ = md.quick_min_all(mol_c, min_style='cg', tmp_clear=False, solver=solver, solver_path=solver_path,
        #                                        work_dir=tmp_dir, omp=omp, mpi=mpi, gpu=gpu, mp=mm_mp)
        #    for i, e in enumerate(energy):
        #        energies.append((e, i))

        if (mm_mp > 0 or const.mpi4py_avail) and nconf > 1:
            utils.picklable(mol_c)
            c = utils.picklable_const()
            args = [(mol_c, i, solver, solver_path, tmp_dir, omp, mpi, gpu, c) for i in range(nconf)]

            # mpi4py
            if const.mpi4py_avail:
                utils.radon_print('Parallel method: mpi4py')
                with MPIPoolExecutor(max_workers=mm_mp) as executor:
                    results = executor.map(_conf_search_lammps_worker, args)

                    for i, res in enumerate(results):
                        energies.append((res[0], i))
                        coord = res[1]
                        for j, atom in enumerate(mol_c.GetAtoms()):
                            mol_c.GetConformer(i).SetAtomPosition(j, Geom.Point3D(coord[j, 0], coord[j, 1], coord[j, 2]))

            # concurrent.futures
            else:
                utils.radon_print('Parallel method: concurrent.futures.ProcessPoolExecutor')
                with confu.ProcessPoolExecutor(max_workers=mm_mp) as executor:
                    results = executor.map(_conf_search_lammps_worker, args)

                    for i, res in enumerate(results):
                        energies.append((res[0], i))
                        coord = res[1]
                        for j, atom in enumerate(mol_c.GetAtoms()):
                            mol_c.GetConformer(i).SetAtomPosition(j, Geom.Point3D(coord[j, 0], coord[j, 1], coord[j, 2]))

        # Sequential execution of MM optimizations
        else:
            for i in tqdm(range(nconf), desc='[Conformation search]', disable=const.tqdm_disable):
                mol_c, energy, _ = md.quick_min(mol_c, confId=i, min_style='cg', solver=solver,
                                                solver_path=solver_path, work_dir=tmp_dir, omp=omp, mpi=mpi, gpu=gpu)
                energies.append((energy, i))

    else:
        # Using RDKit
        prop = AllChem.MMFFGetMoleculeProperties(mol_c)

        # Parallel execution of RDKit optimizations
        if (mm_mp > 0 or const.mpi4py_avail) and nconf > 1:
            utils.picklable(mol_c)
            c = utils.picklable_const()
            args = [(mol_c, prop, i, c) for i in range(nconf)]

            # mpi4py
            if const.mpi4py_avail:
                utils.radon_print('Parallel method: mpi4py')
                with MPIPoolExecutor(max_workers=mm_mp) as executor:
                    results = executor.map(_conf_search_rdkit_worker, args)

                    for i, res in enumerate(results):
                        energies.append((res[0], i))
                        coord = res[1]
                        for j, atom in enumerate(mol_c.GetAtoms()):
                            mol_c.GetConformer(i).SetAtomPosition(j, Geom.Point3D(coord[j, 0], coord[j, 1], coord[j, 2]))

            # concurrent.futures
            else:
                utils.radon_print('Parallel method: concurrent.futures.ProcessPoolExecutor')
                with confu.ProcessPoolExecutor(max_workers=mm_mp) as executor:
                    results = executor.map(_conf_search_rdkit_worker, args)

                    for i, res in enumerate(results):
                        energies.append((res[0], i))
                        coord = res[1]
                        for j, atom in enumerate(mol_c.GetAtoms()):
                            mol_c.GetConformer(i).SetAtomPosition(j, Geom.Point3D(coord[j, 0], coord[j, 1], coord[j, 2]))

        # Sequential execution of RDKit optimizations
        else:
            for i in tqdm(range(nconf), desc='[Conformation search]', disable=const.tqdm_disable):
                mmff = AllChem.MMFFGetMoleculeForceField(mol_c, prop, confId=i)
                mmff.Minimize()
                energies.append((mmff.CalcEnergy(), i))

    # Sort by MM energy
    energies = np.array(energies)
    energies = energies[np.argsort(energies[:, 0])]
    e_res = energies[:, 0]
    for i in range(nconf):
        mol_c.GetConformer(i).SetId(i+nconf)
    for i, cid in enumerate(energies):
        mol_c.GetConformer(int(cid[1]+nconf)).SetId(i)

    # Clustering by RMS or TFD matrix
    if clustering is not None:
        skip_flag = False
        if clustering == 'RMS':
            rmsmat = AllChem.GetConformerRMSMatrix(mol_c, prealigned=False)
            clusters = Butina.ClusterData(rmsmat, nconf, rmsthresh, isDistData=True, reordering=True)
        elif clustering == 'TFD':
            try:
                tfdmat = TorsionFingerprints.GetTFDMatrix(mol_c)
                clusters = Butina.ClusterData(tfdmat, nconf, tfdthresh, isDistData=True, reordering=True)
            except IndexError:
                utils.radon_print('Skip the clustering.')
                skip_flag = True
        else:
            utils.radon_print('Illegal input of clustering = %s' % clustering, level=3)
            Chem.SanitizeMol(mol_c)
            return mol_c, e_res

        if not skip_flag:
            utils.radon_print('Clusters:%s' % str(clusters))
            idx_list = []
            for cl in clusters:
                cl = list(cl)
                if len(cl) > 1:
                    idx = min(cl)
                    idx_list.append(idx)
                    cl.remove(idx)
                    for conf_id in cl:
                        mol_c.RemoveConformer(conf_id)
                else:
                    idx_list.append(cl[0])
            idx_list.sort()

            e_res_new = np.zeros_like(idx_list, dtype=float)
            utils.radon_print('%i conformers were reduced to %i conformers by clustering.' % (nconf, len(idx_list)))
        
            for cid in idx_list:
                mol_c.GetConformer(cid).SetId(cid+nconf)
            for i, cid in enumerate(idx_list):
                mol_c.GetConformer(cid+nconf).SetId(i)
                e_res_new[i] = e_res[cid]
                utils.radon_print('Changing conformer ID %i -> %i' % (cid, i))
        else:
            e_res_new = e_res[:]
    else:
        e_res_new = e_res[:]

    Chem.SanitizeMol(mol_c)
    nconf_new = mol_c.GetNumConformers()
    re_energy = np.array(e_res_new)

    # DFT optimization of conformers
    if dft_nconf > 0:
        opt_success = 0
        if not qm_avail:
            utils.radon_print('Cannot import psi4_wrapper. You can use psi4_wrapper by "conda install -c psi4 psi4 resp dftd3"', level=3)
            Chem.SanitizeMol(mol_c)
            return mol_c, re_energy

        if type(total_charge) is int:
            kwargs['charge'] = total_charge
        if type(total_multiplicity) is int:
            kwargs['multiplicity'] = total_multiplicity

        if dft_nconf > nconf_new: dft_nconf = nconf_new
        psi4mol = QMw(mol_c, work_dir=work_dir, tmp_dir=tmp_dir, omp=psi4_omp, qm_solver=qm_solver, method=opt_method, basis=opt_basis, basis_gen=opt_basis_gen,
                      memory=memory, **kwargs)

        utils.radon_print('Start optimization of %i conformers by DFT level.' % dft_nconf, level=1)

        # Parallel execution of psi4 optimizations
        if (psi4_mp > 0 or const.mpi4py_avail) and dft_nconf > 1:
            utils.picklable(mol_c)
            c = utils.picklable_const()
            args = [(mol_c, i, work_dir, tmp_dir, psi4_omp, qm_solver, opt_method, opt_basis, opt_basis_gen, log_name,
                    geom_iter, geom_conv, geom_algorithm, memory, kwargs, c) for i in range(dft_nconf)]

            # mpi4py
            if const.mpi4py_avail:
                utils.radon_print('Parallel method: mpi4py')
                with MPIPoolExecutor(max_workers=psi4_mp) as executor:
                    results = executor.map(_conf_search_psi4_worker, args)

                    for i, res in enumerate(results):
                        dft_energies.append((res[0], i))
                        coord = res[1]
                        for j, atom in enumerate(psi4mol.mol.GetAtoms()):
                            psi4mol.mol.GetConformer(i).SetAtomPosition(j, Geom.Point3D(coord[j, 0], coord[j, 1], coord[j, 2]))
                        opt_success += 1 if not res[2] else 0

            # concurrent.futures
            else:
                utils.radon_print('Parallel method: concurrent.futures.ProcessPoolExecutor')
                if psi4_mp == 1:
                    for i, arg in enumerate(args):
                        with confu.ProcessPoolExecutor(max_workers=psi4_mp, mp_context=MP.get_context('spawn')) as executor:
                            results = executor.map(_conf_search_psi4_worker, [arg])

                        for res in results:
                            dft_energies.append((res[0], i))
                            coord = res[1]
                            for j, atom in enumerate(psi4mol.mol.GetAtoms()):
                                psi4mol.mol.GetConformer(i).SetAtomPosition(j, Geom.Point3D(coord[j, 0], coord[j, 1], coord[j, 2]))
                            opt_success += 1 if not res[2] else 0

                else:
                    with confu.ProcessPoolExecutor(max_workers=psi4_mp, mp_context=MP.get_context('spawn')) as executor:
                        results = executor.map(_conf_search_psi4_worker, args)

                        for i, res in enumerate(results):
                            dft_energies.append((res[0], i))
                            coord = res[1]
                            for j, atom in enumerate(psi4mol.mol.GetAtoms()):
                                psi4mol.mol.GetConformer(i).SetAtomPosition(j, Geom.Point3D(coord[j, 0], coord[j, 1], coord[j, 2]))
                            opt_success += 1 if not res[2] else 0

            utils.restore_picklable(mol_c)

        # Sequential execution of psi4 optimizations
        else:
            for i in tqdm(range(dft_nconf), desc='[DFT optimization]', disable=const.tqdm_disable):
                utils.radon_print('DFT optimization of conformer %i' % i)
                psi4mol.confId = i
                psi4mol.name = '%s_conf_search_%i' % (log_name, i)
                energy, _ = psi4mol.optimize(ignore_conv_error=True, geom_iter=geom_iter, geom_conv=geom_conv, geom_algorithm=geom_algorithm)
                if not psi4mol.error_flag:
                    opt_success += 1
                dft_energies.append((energy, i))

        if opt_success == 0:
            utils.radon_print('All conformers were failed to optimize geometry. Return the results of the MM optimization.', level=2)
            Chem.SanitizeMol(mol_c)
            return mol_c, re_energy

        # Sort by DFT energy
        utils.radon_print('Conformer IDs were sorted by DFT energy.')
 
        dft_energies = np.array(dft_energies)
        e_dft_res = dft_energies[:, 0]
        dft_energies = dft_energies[np.argsort(dft_energies[:, 0])]
        re_energy = np.full((len(e_res_new), 2), np.nan, dtype=float)

        for i in range(dft_nconf):
            psi4mol.mol.GetConformer(i).SetId(i+nconf_new)
        for i, cid in enumerate(dft_energies):
            psi4mol.mol.GetConformer(int(cid[1]+nconf_new)).SetId(i)
            re_energy[i, 0] = e_dft_res[int(cid[1])]
            re_energy[i, 1] = e_res_new[int(cid[1])]
            utils.radon_print('Changing conformer ID %i -> %i' % (int(cid[1]), i))
        if nconf_new > dft_nconf:
            re_energy[dft_nconf:, 1] = e_res_new[dft_nconf:]

        return_mol = utils.deepcopy_mol(psi4mol.mol)
        del psi4mol
        gc.collect()
        Chem.SanitizeMol(return_mol)
        return return_mol, re_energy

    else:
        Chem.SanitizeMol(mol_c)
        return mol_c, re_energy


def _conf_search_lammps_worker(args):
    mol, confId, solver, solver_path, work_dir, omp, mpi, gpu, c = args
    utils.restore_const(c)

    utils.radon_print('Worker process %i start on %s. PID: %i' % (confId, socket.gethostname(), os.getpid()))

    utils.restore_picklable(mol)
    mol, energy, coord = md.quick_min(mol, confId=confId, min_style='cg', idx=confId, tmp_clear=True,
                        solver=solver, solver_path=solver_path, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu)

    gc.collect()

    return energy, coord


def _conf_search_rdkit_worker(args):
    mol, prop, confId, c = args
    utils.restore_const(c)

    utils.radon_print('Worker process %i start on %s. PID: %i' % (confId, socket.gethostname(), os.getpid()))
    
    utils.restore_picklable(mol)
    mmff = AllChem.MMFFGetMoleculeForceField(mol, prop, confId=confId)
    mmff.Minimize()
    energy = mmff.CalcEnergy()
    coord = np.array(mmff.Positions())

    gc.collect()

    return energy, coord


def _conf_search_psi4_worker(args):
    mol, confId, work_dir, tmp_dir, psi4_omp, qm_solver, opt_method, opt_basis, opt_basis_gen, log_name, geom_iter, geom_conv, geom_algorithm, memory, kwargs, c = args
    utils.restore_const(c)

    utils.radon_print('Worker process %i start on %s. PID: %i' % (confId, socket.gethostname(), os.getpid()))

    error_flag = False
    utils.restore_picklable(mol)
    psi4mol = QMw(mol, work_dir=work_dir, tmp_dir=tmp_dir, omp=psi4_omp, qm_solver=qm_solver, method=opt_method, basis=opt_basis, basis_gen=opt_basis_gen,
                    memory=memory, **kwargs)
    psi4mol.confId = confId
    psi4mol.name = '%s_conf_search_%i' % (log_name, confId)
    utils.radon_print('DFT optimization of conformer %i' % psi4mol.confId)

    energy, coord = psi4mol.optimize(ignore_conv_error=True, geom_iter=geom_iter, geom_conv=geom_conv, geom_algorithm=geom_algorithm)
    error_flag = psi4mol.error_flag

    del psi4mol
    gc.collect()

    return energy, coord, error_flag


def refractive_index(polarizability, density, mol_weight, ratio=None):
    """
    calc.refractive_index

    Calculation of refractive index by Lorentz–Lorenz equation

    Args:
        polarizability: list of polarizability of repeating units (float, angstrom^3)
        density: (float, g/cm^3)
        mol_weight: list of molecular weight of repeating units (float, g mol^-1)
        ratio: ratio of repeating units in a copolymer

    Return:
        Refractive index (float)
    """

    if type(polarizability) is float or type(polarizability) is int:
        polarizability = np.array([polarizability])
    elif type(polarizability) is list:
        polarizability = np.array(polarizability)

    if type(mol_weight) is float or type(mol_weight) is int:
        mol_weight = np.array([mol_weight])
    elif type(mol_weight) is list:
        mol_weight = np.array(mol_weight)

    if ratio is None:
        ratio = np.array([1])
    elif type(ratio) is list:
        ratio = np.array(ratio / np.sum(ratio))

    alpha = polarizability * const.ang2cm**3    # angstrom^3 -> cm^3
    phi = (4*np.pi*density*const.NA/3) * np.sum(alpha*ratio)/np.sum(mol_weight*ratio)   # (g cm^-3) * (mol^-1) * (cm^3) / (g mol^-1)
    ri = np.sqrt((1+2*phi)/(1-phi))

    return ri


def thermal_diffusivity(tc, density, Cp):
    """
    calc.thermal_diffusivity

    Calculation of thermal diffusivity

    Args:
        tc: thermal conductivity (float, W/(m K))
        density: (float, g/cm^3)
        Cp: isobaric heat capacity (float, J/(kg K))

    Return:
        Thermal diffusivity (float, m^2/s)
    """
    td = tc / (density * 1e3 * Cp)
    return td


def set_charge_matrix(mol, ff):
    if not hasattr(mol, 'atoms12'): set_special_bonds(mol)
    charge_matrix = np.zeros( (len(mol.GetAtoms()), len(mol.GetAtoms())) )

    for i, p1 in enumerate(mol.GetAtoms()):
        for j, p2 in enumerate(mol.GetAtoms()):
            charge_matrix[i,j] = p1.GetDoubleProp('AtomicCharge') * p2.GetDoubleProp('AtomicCharge')
            
            if p2.GetIdx() in mol.atoms12[p1.GetIdx()]:
                charge_matrix[i,j] *= ff.param.c_c12
            elif p2.GetIdx() in mol.atoms13[p1.GetIdx()]:
                charge_matrix[i,j] *= ff.param.c_c13
            elif p2.GetIdx() in mol.atoms14[p1.GetIdx()]:
                charge_matrix[i,j] *= ff.param.c_c14

    setattr(mol, 'charge_matrix', charge_matrix)

    return charge_matrix


def set_lj_matrix(mol, ff):
    if not hasattr(mol, 'atoms12'): set_special_bonds(mol)
    lj_epsilon_matrix = np.zeros( (len(mol.GetAtoms()), len(mol.GetAtoms())) )
    lj_sigma_matrix = np.zeros( (len(mol.GetAtoms()), len(mol.GetAtoms())) )

    for i, p1 in enumerate(mol.GetAtoms()):
        for j, p2 in enumerate(mol.GetAtoms()):
            lj_sigma_matrix[i,j] = (p1.GetDoubleProp('ff_sigma') + p2.GetDoubleProp('ff_sigma')) / 2
            lj_epsilon_matrix[i,j] = sqrt(p1.GetDoubleProp('ff_epsilon') * p2.GetDoubleProp('ff_epsilon'))

            if p2.GetIdx() in mol.atoms12[p1.GetIdx()]:
                lj_epsilon_matrix[i,j] *= ff.param.lj_c12
            elif p2.GetIdx() in mol.atoms13[p1.GetIdx()]:
                lj_epsilon_matrix[i,j] *= ff.param.lj_c13
            elif p2.GetIdx() in mol.atoms14[p1.GetIdx()]:
                lj_epsilon_matrix[i,j] *= ff.param.lj_c14

    setattr(mol, 'lj_epsilon_matrix', lj_epsilon_matrix)
    setattr(mol, 'lj_sigma_matrix', lj_sigma_matrix)

    return lj_epsilon_matrix, lj_sigma_matrix


def set_charge_lj_matrix(mol, ff):
    if not hasattr(mol, 'atoms12'): set_special_bonds(mol)
    charge_matrix = np.zeros( (len(mol.GetAtoms()), len(mol.GetAtoms())) )
    lj_epsilon_matrix = np.zeros( (len(mol.GetAtoms()), len(mol.GetAtoms())) )
    lj_sigma_matrix = np.zeros( (len(mol.GetAtoms()), len(mol.GetAtoms())) )

    for i, p1 in enumerate(mol.GetAtoms()):
        for j, p2 in enumerate(mol.GetAtoms()):
            charge_matrix[i,j] = p1.GetDoubleProp('AtomicCharge') * p2.GetDoubleProp('AtomicCharge')
            lj_sigma_matrix[i,j] = (p1.GetDoubleProp('ff_sigma') + p2.GetDoubleProp('ff_sigma')) / 2
            lj_epsilon_matrix[i,j] = sqrt(p1.GetDoubleProp('ff_epsilon') * p2.GetDoubleProp('ff_epsilon'))

            if p2.GetIdx() in mol.atoms12[p1.GetIdx()]:
                charge_matrix[i,j] *= ff.param.c_c12
                lj_epsilon_matrix[i,j] *= ff.param.lj_c12
            elif p2.GetIdx() in mol.atoms13[p1.GetIdx()]:
                charge_matrix[i,j] *= ff.param.c_c13
                lj_epsilon_matrix[i,j] *= ff.param.lj_c13
            elif p2.GetIdx() in mol.atoms14[p1.GetIdx()]:
                charge_matrix[i,j] *= ff.param.c_c14
                lj_epsilon_matrix[i,j] *= ff.param.lj_c14

    setattr(mol, 'charge_matrix', charge_matrix)
    setattr(mol, 'lj_epsilon_matrix', lj_epsilon_matrix)
    setattr(mol, 'lj_sigma_matrix', lj_sigma_matrix)

    return charge_matrix, lj_epsilon_matrix, lj_sigma_matrix


def set_special_bonds(mol):
    setattr(mol, 'atoms12', [])
    setattr(mol, 'atoms13', [])
    setattr(mol, 'atoms14', [])
        
    for p in mol.GetAtoms():
        atoms12 = []
        atoms13 = []
        atoms14 = []
            
        for pb in p.GetNeighbors():
            atoms12.append(pb.GetIdx())
            
        for pb in p.GetNeighbors():
            for pbb in pb.GetNeighbors():
                if pbb.GetIdx() == p.GetIdx(): continue
                if pbb.GetIdx() in atoms12: continue
                atoms13.append(pbb.GetIdx())
                    
        for pb in p.GetNeighbors():
            for pbb in pb.GetNeighbors():
                for pbbb in pbb.GetNeighbors():
                    if pbbb.GetIdx() == pb.GetIdx() and pbbb.GetIdx() == p.GetIdx(): continue
                    if pbbb.GetIdx() in atoms12 or pbbb.GetIdx() in atoms13: continue
                    atoms14.append(pbb.GetIdx())
                        
        mol.atoms12.append(atoms12)
        mol.atoms13.append(atoms13)
        mol.atoms14.append(atoms14)
            
    return True


def energy_mm(mol, diele=1.0, coord=None, confId=0, **kwargs):
    """
    calc.energy_mm

    Calculate energy by molecular mechanics

    Args:
        mol: RDKit Mol object (requiring force field assignment and 3D position)

    Optional args:
        diele: Dielectric constants (float)
        coord: Atomic coordinates (ndarray (size(n, 3)), angstrom)
        confId: Target conformer ID
        bond, angle, dihedral, improper, coulomb, lj:
            Switcing turn on/off for each energy term (boolean)

    Returns:
        energies (float, kcal/mol)
    """

    calc_bond = kwargs.get('bond', True)
    calc_angle = kwargs.get('angle', True)
    calc_dihedral = kwargs.get('dihedral', True)
    calc_improper = kwargs.get('improper', True)
    calc_coulomb = kwargs.get('coulomb', True)
    calc_lj = kwargs.get('lj', True)

    energy_bond = 0.0
    energy_angle = 0.0
    energy_dihedral = 0.0
    energy_improper = 0.0
    energy_coulomb = 0.0
    energy_lj = 0.0
        
    if coord is None: coord = mol.GetConformer(confId).GetPositions()
    dist_matrix = distance_matrix(coord)

    for i, p in enumerate(mol.GetAtoms()):
        p.SetDoubleProp('x', coord[i, 0])
        p.SetDoubleProp('y', coord[i, 1])
        p.SetDoubleProp('z', coord[i, 2])
    
    if calc_bond:
        for b in mol.GetBonds():
            length = dist_matrix[b.GetBeginAtom().GetIdx(), b.GetEndAtom().GetIdx()]
            energy_bond += b.GetDoubleProp('ff_k')*( (length - b.GetDoubleProp('ff_r0'))**2 )
            
    if calc_angle:
        for ang in mol.angles:
            theta = angle(mol.GetAtomWithIdx(ang.a), mol.GetAtomWithIdx(ang.b), mol.GetAtomWithIdx(ang.c), rad=True)
            energy_angle += ang.ff.k*( (theta - ang.ff.theta0_rad)**2 )
            
    if calc_dihedral:
        for dih in mol.dihedrals:
            phi = dihedral(mol.GetAtomWithIdx(dih.a), mol.GetAtomWithIdx(dih.b), mol.GetAtomWithIdx(dih.c), mol.GetAtomWithIdx(dih.d), rad=True)
            energy_dihedral += np.sum(dih.ff.k*(1.0 + np.cos(dih.ff.n*phi - dih.ff.d0_rad)))
                
    if calc_improper:
        for imp in mol.impropers:
            phi = dihedral(mol.GetAtomWithIdx(imp.a), mol.GetAtomWithIdx(imp.b), mol.GetAtomWithIdx(imp.c), mol.GetAtomWithIdx(imp.d), rad=True)
            energy_improper += imp.ff.k*(1.0 + imp.ff.d0*cos(imp.ff.n*phi))
    
    if calc_coulomb:
        energy_coulomb = np.sum( np.divide(mol.charge_matrix, dist_matrix, out=np.zeros_like(mol.charge_matrix), where=dist_matrix!=0) ) / 2
        energy_coulomb = energy_coulomb / diele * const.bohr2ang * const.au2kcal

    if calc_lj:
        sigma_div = np.divide(mol.lj_sigma_matrix, dist_matrix, out=np.zeros_like(mol.lj_sigma_matrix), where=dist_matrix!=0)
        energy_lj = np.sum( 4*mol.lj_epsilon_matrix*(sigma_div**12 - sigma_div**6) ) / 2
    
    energy = energy_bond + energy_angle + energy_dihedral + energy_improper + energy_coulomb + energy_lj
    
    return energy, energy_bond, energy_angle, energy_dihedral, energy_improper, energy_coulomb, energy_lj

