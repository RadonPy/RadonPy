#  Copyright (c) 2022. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# ******************************************************************************
# core.poly module
# ******************************************************************************

import numpy as np
import pandas as pd
import re
from copy import copy
import itertools
import datetime
import multiprocessing as MP
import concurrent.futures as confu
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import Geometry as Geom
from rdkit import RDLogger
from . import calc, const, utils

__version__ = '0.3.0b1'

MD_avail = True
try:
    from ..sim import md
except ImportError:
    MD_avail = False

if const.tqdm_disable:
    tqdm = utils.tqdm_stub
else:
    try:
        from tqdm.autonotebook import tqdm
    except ImportError:
        tqdm = utils.tqdm_stub


def connect_mols(mol1, mol2, bond_length=1.5, dihedral=np.pi, random_rot=False, dih_type='monomer', set_linker=True, label1=1, label2=1,
                headhead=False, tailtail=False, confId1=0, confId2=0, res_name_1='RU0', res_name_2='RU0'):
    """
    poly.connect_mols

    Connect tail atom in mol1 to head atom in mol2

    Args:
        mol1, mol2: RDkit Mol object (requiring AddHs and 3D position)

    Optional args:
        bond_length: Bond length of connecting bond (float, angstrome)
        random_rot: Dihedral angle around connecting bond is rotated randomly (boolean)
        dihedral: Dihedral angle around connecting bond (float, radian)
        dih_type: Definition type of dihedral angle (str; monomer, or bond)
        headhead: Connect with head-to-head
        tailtail: Connect with tail-to-tail
        confId1, confId2: Target conformer ID of mol1 and mol2
        res_name_1, res_name_2: Set residue name of PDB

    Returns:
        RDkit Mol object
    """

    if mol1 is None: return mol2
    if mol2 is None: return mol1

    # Initialize
    if headhead: set_linker_flag(mol1, reverse=True, label=label1)
    elif set_linker: set_linker_flag(mol1, label=label1)
    if tailtail and not headhead: set_linker_flag(mol2, reverse=True, label=label2)
    elif set_linker: set_linker_flag(mol2, label=label2)

    if mol1.GetIntProp('tail_idx') < 0 or mol1.GetIntProp('tail_ne_idx') < 0:
        utils.radon_print('Cannot connect_mols because mol1 does not have a tail linker atom.', level=2)
        return mol1
    elif mol2.GetIntProp('head_idx') < 0 or mol2.GetIntProp('head_ne_idx') < 0:
        utils.radon_print('Cannot connect_mols because mol2 does not have a head linker atom.', level=2)
        return mol1

    mol1_n = mol1.GetNumAtoms()
    mol1_coord = mol1.GetConformer(confId1).GetPositions()
    mol2_coord = mol2.GetConformer(confId2).GetPositions()
    mol1_tail_vec = mol1_coord[mol1.GetIntProp('tail_ne_idx')] - mol1_coord[mol1.GetIntProp('tail_idx')]
    mol2_head_vec = mol2_coord[mol2.GetIntProp('head_ne_idx')] - mol2_coord[mol2.GetIntProp('head_idx')]
    charge_list = ['AtomicCharge', '_GasteigerCharge', 'RESP', 'ESP', 'MullikenCharge', 'LowdinCharge']
    #bd1type = mol1.GetBondBetweenAtoms(mol1.GetIntProp('tail_idx'), mol1.GetIntProp('tail_ne_idx')).GetBondTypeAsDouble()
    #bd2type = mol2.GetBondBetweenAtoms(mol2.GetIntProp('head_idx'), mol2.GetIntProp('head_ne_idx')).GetBondTypeAsDouble()

    # Rotation mol2 to align bond vectors of head and tail
    angle = calc.angle_vec(mol1_tail_vec, mol2_head_vec, rad=True)
    center = mol2_coord[mol2.GetIntProp('head_ne_idx')]
    if angle == 0:
        mol2_coord_rot = (mol2_coord - center) * -1 + center
    elif angle == np.pi:
        mol2_coord_rot = mol2_coord
    else:
        vcross = np.cross(mol1_tail_vec, mol2_head_vec)
        mol2_coord_rot = calc.rotate_rod(mol2_coord, vcross, (np.pi-angle), center=center)

    # Translation mol2
    trans = mol1_coord[mol1.GetIntProp('tail_ne_idx')] - ( bond_length * mol1_tail_vec / np.linalg.norm(mol1_tail_vec) )
    mol2_coord_rot = mol2_coord_rot + trans - mol2_coord_rot[mol2.GetIntProp('head_ne_idx')]

    # Rotation mol2 around new bond
    if random_rot == True:
        dih = np.random.uniform(-np.pi, np.pi)
    else:
        if dih_type == 'monomer':
            dih = calc.dihedral_coord(mol1_coord[mol1.GetIntProp('head_idx')], mol1_coord[mol1.GetIntProp('tail_ne_idx')],
                                      mol2_coord_rot[mol2.GetIntProp('head_ne_idx')], mol2_coord_rot[mol2.GetIntProp('tail_idx')], rad=True)
        elif dih_type == 'bond':
            path1 = Chem.GetShortestPath(mol1, mol1.GetIntProp('head_idx'), mol1.GetIntProp('tail_idx'))
            path2 = Chem.GetShortestPath(mol2, mol2.GetIntProp('head_idx'), mol2.GetIntProp('tail_idx'))
            dih = calc.dihedral_coord(mol1_coord[path1[-3]], mol1_coord[path1[-2]],
                                      mol2_coord_rot[path2[1]], mol2_coord_rot[path2[2]], rad=True)
        else:
            utils.radon_print('Illegal option of dih_type=%s.' % str(dih_type), level=3)
    mol2_coord_rot = calc.rotate_rod(mol2_coord_rot, -mol1_tail_vec, (dihedral-dih), center=mol2_coord_rot[mol2.GetIntProp('head_ne_idx')])

    # Combining mol1 and mol2
    mol = combine_mols(mol1, mol2, res_name_1=res_name_1, res_name_2=res_name_2)

    # Set atomic coordinate
    for i in range(mol2.GetNumAtoms()):
        mol.GetConformer(0).SetAtomPosition(i+mol1_n, Geom.Point3D(mol2_coord_rot[i, 0], mol2_coord_rot[i, 1], mol2_coord_rot[i, 2]))

    # Set atomic charge
    for charge in charge_list:
        if not mol.GetAtomWithIdx(0).HasProp(charge): continue
        head_charge = mol.GetAtomWithIdx(mol2.GetIntProp('head_idx') + mol1_n).GetDoubleProp(charge)
        head_ne_charge = mol.GetAtomWithIdx(mol2.GetIntProp('head_ne_idx') + mol1_n).GetDoubleProp(charge)
        tail_charge = mol.GetAtomWithIdx(mol1.GetIntProp('tail_idx')).GetDoubleProp(charge)
        tail_ne_charge = mol.GetAtomWithIdx(mol1.GetIntProp('tail_ne_idx')).GetDoubleProp(charge)
        mol.GetAtomWithIdx(mol2.GetIntProp('head_ne_idx') + mol1_n).SetDoubleProp(charge, head_charge+head_ne_charge)
        mol.GetAtomWithIdx(mol1.GetIntProp('tail_ne_idx')).SetDoubleProp(charge, tail_charge+tail_ne_charge)

    # Delete linker atoms and bonds
    del_idx1 = mol1.GetIntProp('tail_idx')
    del_idx2 = mol2.GetIntProp('head_idx') + mol1_n - 1
    mol = utils.remove_atom(mol, del_idx1)
    mol = utils.remove_atom(mol, del_idx2)

    # Add a new bond
    tail_ne_idx = mol1.GetIntProp('tail_ne_idx')
    head_ne_idx = mol2.GetIntProp('head_ne_idx') + mol1_n - 1
    if del_idx1 < tail_ne_idx: tail_ne_idx -= 1
    if del_idx2 < head_ne_idx: head_ne_idx -= 1
    mol = utils.add_bond(mol, tail_ne_idx, head_ne_idx, order=Chem.rdchem.BondType.SINGLE)
    #if bd1type == 2.0 or bd2type == 2.0:
    #    mol = utils.add_bond(mol, tail_ne_idx, head_ne_idx, order=Chem.rdchem.BondType.DOUBLE)
    #elif bd1type == 3.0 or bd2type == 3.0:
    #    mol = utils.add_bond(mol, tail_ne_idx, head_ne_idx, order=Chem.rdchem.BondType.TRIPLE)
    #else:
    #    mol = utils.add_bond(mol, tail_ne_idx, head_ne_idx, order=Chem.rdchem.BondType.SINGLE)

    # Finalize
    Chem.SanitizeMol(mol)
    #set_linker_flag(mol)

    # Update linker_flag
    mol.GetAtomWithIdx(tail_ne_idx).SetBoolProp('tail_neighbor', False)
    mol.GetAtomWithIdx(head_ne_idx).SetBoolProp('head_neighbor', False)

    head_idx = mol1.GetIntProp('head_idx')
    if del_idx1 < head_idx: head_idx -= 1

    head_ne_idx = mol1.GetIntProp('head_ne_idx')
    if del_idx1 < head_ne_idx: head_ne_idx -= 1

    tail_idx = mol2.GetIntProp('tail_idx') + mol1_n - 1
    if del_idx2 < tail_idx: tail_idx -= 1

    tail_ne_idx = mol2.GetIntProp('tail_ne_idx') + mol1_n - 1
    if del_idx2 < tail_ne_idx: tail_ne_idx -= 1

    mol.SetIntProp('head_idx', head_idx)
    mol.SetIntProp('head_ne_idx', head_ne_idx)
    mol.SetIntProp('tail_idx', tail_idx)
    mol.SetIntProp('tail_ne_idx', tail_ne_idx)

    return mol


def combine_mols(mol1, mol2, res_name_1='RU0', res_name_2='RU0'):
    """
    poly.combine_mols

    Combining mol1 and mol2 taking over the angles, dihedrals, impropers, and cell data

    Args:
        mol1, mol2: RDkit Mol object

    Optional args:
        res_name_1, res_name_2: Set residue name of PDB

    Returns:
        RDkit Mol object
    """

    mol = Chem.rdmolops.CombineMols(mol1, mol2)

    mol1_n = mol1.GetNumAtoms()
    mol2_n = mol2.GetNumAtoms()
    angles = {}
    dihedrals = {}
    impropers = {}
    cell = None

    if hasattr(mol1, 'angles'):
        angles = mol1.angles.copy()
    if hasattr(mol2, 'angles'):
        for angle in mol2.angles.values():
            key = '%i,%i,%i' % (angle.a+mol1_n, angle.b+mol1_n, angle.c+mol1_n)
            angles[key] = utils.Angle(
                                a=angle.a+mol1_n,
                                b=angle.b+mol1_n,
                                c=angle.c+mol1_n,
                                ff=angle.ff
                            )

    if hasattr(mol1, 'dihedrals'):
        dihedrals = mol1.dihedrals.copy()
    if hasattr(mol2, 'dihedrals'):
        for dihedral in mol2.dihedrals.values():
            key = '%i,%i,%i,%i' % (dihedral.a+mol1_n, dihedral.b+mol1_n, dihedral.c+mol1_n, dihedral.d+mol1_n)
            dihedrals[key] = utils.Dihedral(
                                    a=dihedral.a+mol1_n,
                                    b=dihedral.b+mol1_n,
                                    c=dihedral.c+mol1_n,
                                    d=dihedral.d+mol1_n,
                                    ff=dihedral.ff
                                )

    if hasattr(mol1, 'impropers'):
        impropers = mol1.impropers.copy()
    if hasattr(mol2, 'impropers'):
        for improper in mol2.impropers.values():
            key = '%i,%i,%i,%i' % (improper.a+mol1_n, improper.b+mol1_n, improper.c+mol1_n, improper.d+mol1_n)
            impropers[key] = utils.Improper(
                                    a=improper.a+mol1_n,
                                    b=improper.b+mol1_n,
                                    c=improper.c+mol1_n,
                                    d=improper.d+mol1_n,
                                    ff=improper.ff
                                )
    
    if hasattr(mol1, 'cell'):
        cell = copy(mol1.cell)
    elif hasattr(mol2, 'cell'):
        cell = copy(mol2.cell)

    # Generate PDB information and repeating unit information
    resid = []
    if mol1.HasProp('num_units'):
        max_resid = mol1.GetIntProp('num_units')
    else:
        for i in range(mol1_n):
            atom = mol.GetAtomWithIdx(i)
            atom_name = atom.GetProp('ff_type') if atom.HasProp('ff_type') else atom.GetSymbol()
            if atom.GetPDBResidueInfo() is None:
                atom.SetMonomerInfo(
                    Chem.AtomPDBResidueInfo(
                        atom_name,
                        residueName=res_name_1,
                        residueNumber=1,
                        isHeteroAtom=False
                    )
                )
                resid.append(1)
            else:
                atom.GetPDBResidueInfo().SetName(atom_name)
                resid1 = atom.GetPDBResidueInfo().GetResidueNumber()
                resid.append(resid1)
        max_resid = max(resid) if len(resid) > 0 else 0

    for i in range(mol2_n):
        atom = mol.GetAtomWithIdx(i+mol1_n)
        atom_name = atom.GetProp('ff_type') if atom.HasProp('ff_type') else atom.GetSymbol()
        if atom.GetPDBResidueInfo() is None:
            atom.SetMonomerInfo(
                Chem.AtomPDBResidueInfo(
                    atom_name,
                    residueName=res_name_2,
                    residueNumber=1+max_resid,
                    isHeteroAtom=False
                )
            )
            resid.append(1+max_resid)
        else:
            atom.GetPDBResidueInfo().SetName(atom_name)
            resid2 = atom.GetPDBResidueInfo().GetResidueNumber()
            atom.GetPDBResidueInfo().SetResidueNumber(resid2+max_resid)
            resid.append(resid2+max_resid)

    max_resid = max(resid) if len(resid) > 0 else 0
    mol.SetIntProp('num_units', max_resid)

    setattr(mol, 'angles', angles)
    setattr(mol, 'dihedrals', dihedrals)
    setattr(mol, 'impropers', impropers)
    if cell is not None: setattr(mol, 'cell', cell)

    return mol


##########################################################
# Polymer chain generator (non random walk)
##########################################################
def simple_polymerization(mols, m_idx, chi_inv, start_num=0, init_poly=None, headhead=False, confId=0,
                bond_length=1.5, dihedral=np.pi, random_rot=False, dih_type='monomer', res_name_1='RU0', res_name_2=None):
    mols_copy = []
    mols_inv = []
    poly = None

    if type(init_poly) == Chem.Mol:
        poly = utils.deepcopy_mol(init_poly)
        set_linker_flag(poly)

    for mol in mols:
        set_linker_flag(mol)
        mols_copy.append(utils.deepcopy_mol(mol))
        mols_inv.append(calc.mirror_inversion_mol(mol))

    if res_name_2 is None:
        res_name_2 = ['RU%s' % const.pdb_id[i] for i in range(len(mols))]

    utils.radon_print('Start poly.simple_polymerization.')

    for i in tqdm(range(start_num, len(m_idx)), desc='[Polymerization]', disable=const.tqdm_disable):
        if chi_inv[i]:
            mol_c = mols_inv[m_idx[i]]
        else:
            mol_c = mols_copy[m_idx[i]]

        if headhead and i % 2 == 0:
            poly = connect_mols(poly, mol_c, tailtail=True, set_linker=False,
                    bond_length=bond_length, dihedral=dihedral, random_rot=random_rot, dih_type=dih_type,
                    confId1=confId, confId2=confId, res_name_1=res_name_1, res_name_2=res_name_2[m_idx[i]])
        else:
            poly = connect_mols(poly, mol_c, set_linker=False,
                    bond_length=bond_length, dihedral=dihedral, random_rot=random_rot, dih_type=dih_type,
                    confId1=confId, confId2=confId, res_name_1=res_name_1, res_name_2=res_name_2[m_idx[i]])

    return poly


def polymerize_mols(mol, n, init_poly=None, headhead=False, confId=0, tacticity='atactic', atac_ratio=0.5,
                        bond_length=1.5, dihedral=np.pi, random_rot=False, dih_type='monomer'):
    dt1 = datetime.datetime.now()
    utils.radon_print('Start poly.polymerize_mols.', level=1)

    m_idx = gen_monomer_array(1, n)
    chi_inv, _ = gen_chiral_inv_array([mol], m_idx, init_poly=init_poly, tacticity=tacticity, atac_ratio=atac_ratio)
    poly = simple_polymerization(
        [mol], m_idx, chi_inv, start_num=0, init_poly=init_poly, headhead=headhead, confId=confId,
        bond_length=bond_length, dihedral=dihedral, random_rot=random_rot, dih_type=dih_type
    )

    dt2 = datetime.datetime.now()
    utils.radon_print('Normal termination of poly.polymerize_mols. Elapsed time = %s' % str(dt2-dt1), level=1)

    return poly


def copolymerize_mols(mols, n, init_poly=None, headhead=False, confId=0, tacticity='atactic', atac_ratio=0.5,
                        bond_length=1.5, dihedral=np.pi, random_rot=False, dih_type='monomer'):
    dt1 = datetime.datetime.now()
    utils.radon_print('Start poly.copolymerize_mols.', level=1)

    m_idx = gen_monomer_array(len(mols), n, copoly='alt')
    chi_inv, _ = gen_chiral_inv_array(mols, m_idx, init_poly=init_poly, tacticity=tacticity, atac_ratio=atac_ratio)
    poly = simple_polymerization(
        mols, m_idx, chi_inv, start_num=0, init_poly=init_poly, headhead=headhead, confId=confId,
        bond_length=bond_length, dihedral=dihedral, random_rot=random_rot, dih_type=dih_type
    )

    dt2 = datetime.datetime.now()
    utils.radon_print('Normal termination of poly.copolymerize_mols. Elapsed time = %s' % str(dt2-dt1), level=1)

    return poly


def random_copolymerize_mols(mols, n, ratio=None, reac_ratio=[], init_poly=None, headhead=False, confId=0,
                                tacticity='atactic', atac_ratio=0.5, bond_length=1.5, dihedral=np.pi, random_rot=False, dih_type='monomer'):
    dt1 = datetime.datetime.now()
    utils.radon_print('Start poly.random_copolymerize_mols.', level=1)

    m_idx = gen_monomer_array(len(mols), n, copoly='random', ratio=ratio, reac_ratio=reac_ratio)
    chi_inv, _ = gen_chiral_inv_array(mols, m_idx, init_poly=init_poly, tacticity=tacticity, atac_ratio=atac_ratio)
    poly = simple_polymerization(
        mols, m_idx, chi_inv, start_num=0, init_poly=init_poly, headhead=headhead, confId=confId,
        bond_length=bond_length, dihedral=dihedral, random_rot=random_rot, dih_type=dih_type
    )
    
    dt2 = datetime.datetime.now()
    utils.radon_print('Normal termination of poly.random_copolymerize_mols. Elapsed time = %s' % str(dt2-dt1), level=1)

    return poly


def block_copolymerize_mols(mols, n, init_poly=None, headhead=False, confId=0, tacticity='atactic', atac_ratio=0.5,
                        bond_length=1.5, dihedral=np.pi, random_rot=False, dih_type='monomer'):
    dt1 = datetime.datetime.now()
    utils.radon_print('Start poly.block_copolymerize_mols.', level=1)

    m_idx = gen_monomer_array(len(mols), n, copoly='block')
    chi_inv, _ = gen_chiral_inv_array(mols, m_idx, init_poly=init_poly, tacticity=tacticity, atac_ratio=atac_ratio)
    poly = simple_polymerization(
        mols, m_idx, chi_inv, start_num=0, init_poly=init_poly, headhead=headhead, confId=confId,
        bond_length=bond_length, dihedral=dihedral, random_rot=random_rot, dih_type=dih_type
    )
    
    dt2 = datetime.datetime.now()
    utils.radon_print('Normal termination of poly.block_copolymerize_mols. Elapsed time = %s' % str(dt2-dt1), level=1)

    return poly


def terminate_mols(poly, mol1, mol2=None, confId=0, bond_length=1.5, dihedral=np.pi, random_rot=False, dih_type='monomer'):
    """
    poly.terminate_mols

    Simple termination function of polymer of RDkit Mol object

    Args:
        poly: polymer (RDkit Mol object)
        mol1: terminated substitute at head (and tail) (RDkit Mol object)

    Optional args:
        mol2: terminated substitute at tail (RDkit Mol object)
        bond_length: Bond length of connecting bond (float, angstrome)
        random_rot: Dihedral angle around connecting bond is rotated randomly (boolean)
        dihedral: Dihedral angle around connecting bond (float, radian)
        dih_type: Definition type of dihedral angle (str; monomer, or bond)

    Returns:
        Rdkit Mol object
    """
    if mol2 is None: mol2 = mol1
    poly_c = utils.deepcopy_mol(poly)
    mol1_c = utils.deepcopy_mol(mol1)
    mol2_c = utils.deepcopy_mol(mol2)
    res_name_1 = 'TU0'
    res_name_2 = 'TU1'
        
    if Chem.MolToSmiles(mol1_c) == '[H][3H]' or Chem.MolToSmiles(mol1_c) == '[3H][H]':
        poly_c.GetAtomWithIdx(poly_c.GetIntProp('head_idx')).SetIsotope(1)
        poly_c.GetAtomWithIdx(poly_c.GetIntProp('head_idx')).GetPDBResidueInfo().SetResidueName(res_name_1)
        poly_c.GetAtomWithIdx(poly_c.GetIntProp('head_idx')).GetPDBResidueInfo().SetResidueNumber(1+poly_c.GetIntProp('num_units'))
        poly_c.SetIntProp('num_units', 1+poly_c.GetIntProp('num_units'))
    else:
        poly_c = connect_mols(mol1_c, poly_c, bond_length=bond_length, dihedral=dihedral, random_rot=random_rot, dih_type=dih_type,
                            res_name_1=res_name_1)

    if Chem.MolToSmiles(mol2_c) == '[H][3H]' or Chem.MolToSmiles(mol2_c) == '[3H][H]':
        poly_c.GetAtomWithIdx(poly_c.GetIntProp('tail_idx')).SetIsotope(1)
        poly_c.GetAtomWithIdx(poly_c.GetIntProp('tail_idx')).GetPDBResidueInfo().SetResidueName(res_name_2)
        poly_c.GetAtomWithIdx(poly_c.GetIntProp('tail_idx')).GetPDBResidueInfo().SetResidueNumber(1+poly_c.GetIntProp('num_units'))
        poly_c.SetIntProp('num_units', 1+poly_c.GetIntProp('num_units'))
    else:
        poly_c = connect_mols(poly_c, mol2_c, bond_length=bond_length, dihedral=dihedral, random_rot=random_rot, dih_type=dih_type,
                            res_name_2=res_name_2)

    set_terminal_idx(poly_c)

    return poly_c


##########################################################
# Polymer chain generator with self-avoiding random walk
##########################################################
def random_walk_polymerization(mols, m_idx, chi_inv, start_num=0, init_poly=None, headhead=False, confId=0,
            dist_min=0.7, retry=200, rollback=5, retry_step=1000, retry_opt_step=0, tacticity=None,
            res_name_init='INI', res_name=None, label=None, label_init=1, opt='rdkit', ff=None, work_dir=None, omp=0, mpi=1, gpu=0):
    """
    poly.random_walk_polymerization

    Polymerization of RDkit Mol object by self-avoiding random walk

    Args:
        mols: list of RDkit Mol object
        m_idx: Input array of repeating units by index number of mols
        chi_inv: Input boolean array of chiral inversion

    Optional args:
        start_num: Index number of m_idx of starting point
        init_poly: Perform additional polymerization for init_poly (RDkit Mol object)
        headhead: Connect monomer unit by head-to-head
        confId: Target conformer ID
        dist_min: (float, angstrom)
        retry: Number of retry for this function when generating unsuitable structure (int)
        rollback: Number of rollback step when retry random_walk_polymerization (int)
        retry_step: Number of retry for a random-walk step when generating unsuitable structure (int)
        retry_opt_step: Number of retry for a random-walk step with optimization when generating unsuitable structure (int)
        opt:
            lammps: Using short time MD of lammps
            rdkit: MMFF94 optimization by RDKit
        work_dir: Work directory path of LAMMPS (str, requiring when opt is LAMMPS)
        ff: Force field object (requiring when opt is LAMMPS)
        omp: Number of threads of OpenMP in LAMMPS (int)
        mpi: Number of MPI process in LAMMPS (int)
        gpu: Number of GPU in LAMMPS (int)

    Returns:
        Rdkit Mol object
    """
    utils.radon_print('Start poly.random_walk_polymerization.')

    if len(m_idx) != len(chi_inv):
        utils.radon_print('Inconsistency length of m_idx and chi_inv', level=3)
    if len(mols) <= max(m_idx):
        utils.radon_print('Illegal index number was found in m_idx', level=3)

    mols_copy = []
    mols_inv = []
    has_ring = False
    retry_flag = False
    tri_coord = None
    bond_coord = None    
    poly = None
    poly_copy = [None]

    if res_name is None:
        res_name = ['RU%s' % const.pdb_id[i] for i in range(len(mols))]
    else:
        if len(mols) != len(res_name):
            utils.radon_print('Inconsistency length of mols and res_name', level=3)

    if label is None:
        label = [[1, 1] for x in range(len(mols))]
    else:
        if len(mols) != len(label):
            utils.radon_print('Inconsistency length of mols and label', level=3)

    if type(init_poly) == Chem.Mol:
        poly = utils.deepcopy_mol(init_poly)
        set_linker_flag(poly, label=label_init)
        poly_copy = []
        if Chem.GetSSSR(poly) > 0:
            has_ring = True

    for i, mol in enumerate(mols):
        set_linker_flag(mol, label=label[i][0])
        mols_copy.append(utils.deepcopy_mol(mol))
        mols_inv.append(calc.mirror_inversion_mol(mol))
        if Chem.GetSSSR(mol) > 0:
            has_ring = True

    for i in tqdm(range(start_num, len(m_idx)), desc='[Polymerization]', disable=const.tqdm_disable):
        dmat = None
    
        if type(poly) is Chem.Mol:
            poly_copy.append(utils.deepcopy_mol(poly))
        if len(poly_copy) > rollback:
            del poly_copy[0]

        if chi_inv[i]:
            mol_c = mols_inv[m_idx[i]]
        else:
            mol_c = mols_copy[m_idx[i]]

        if i == 0:
            res_name_1 = res_name_init
        else:
            res_name_1 = res_name[m_idx[i-1]]

        for r in range(retry_step+retry_opt_step):
            check_3d = False
            label1 = label[m_idx[i-1]][1] if i > 0 else 1

            if headhead and i % 2 == 0:
                poly = connect_mols(poly, mol_c, tailtail=True, random_rot=True, set_linker=True,
                            confId2=confId, res_name_1=res_name_1, res_name_2=res_name[m_idx[i]],
                            label1=label1, label2=label[m_idx[i]][1])
            else:
                poly = connect_mols(poly, mol_c, random_rot=True, set_linker=True,
                            confId2=confId, res_name_1=res_name_1, res_name_2=res_name[m_idx[i]],
                            label1=label1, label2=label[m_idx[i]][0])

            if i == 0 and init_poly is None:
                break

            if dmat is None and dist_min > 1.0:
                # This deepcopy avoids a bug of RDKit
                dmat = Chem.GetDistanceMatrix(utils.deepcopy_mol(poly))

            if r >= retry_step:
                if opt == 'lammps' and MD_avail:
                    ff.ff_assign(poly)
                    poly, _ = md.quick_rw(poly, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu)
                elif opt == 'rdkit':
                    AllChem.MMFFOptimizeMolecule(poly, maxIters=50, mmffVariant='MMFF94s', nonBondedThresh=3.0, confId=0)
                check_3d = check_3d_structure_poly(poly, mol_c, dmat, dist_min=dist_min, check_bond_length=True, tacticity=tacticity)
                tri_coord = None
                bond_coord = None
            else:
                check_3d = check_3d_structure_poly(poly, mol_c, dmat, dist_min=dist_min, check_bond_length=False)

            if check_3d and has_ring:
                check_3d, tri_coord_new, bond_coord_new = check_3d_bond_ring_intersection(poly, mon=mol_c,
                                                                    tri_coord=tri_coord, bond_coord=bond_coord)

            if check_3d:
                if has_ring:
                    tri_coord = tri_coord_new
                    bond_coord = bond_coord_new
                break
            elif r < retry_step + retry_opt_step - 1:
                poly = utils.deepcopy_mol(poly_copy[-1]) if type(poly_copy[-1]) is Chem.Mol else None
                if r == 0 or (r+1) % 100 == 0:
                    utils.radon_print('Retry random walk step %i, %i/%i' % (i+1, r+1, retry_step))
                if r == retry_step - 1:
                    utils.radon_print('Switch to algorithm with optimization.', level=1)
            else:
                retry_flag = True
                utils.radon_print(
                    'Reached maximum number of retrying step in random walk step %i of poly.random_walk_polymerization.' % (i+1),
                    level=1)

        if retry_flag: break

    if retry_flag:
        if retry <= 0:
            utils.radon_print(
                'poly.random_walk_polymerization is failure because reached maximum number of rollback times in random walk step %i.' % (i+1),
                level=3)
        else:
            utils.radon_print(
                'Retry poly.random_walk_polymerization and rollback %i steps. Remaining %i times.' % (len(poly_copy), retry),
                level=1)
            retry -= 1
            start_num = i-len(poly_copy)+1
            label_init = label[m_idx[start_num-1]][1]
            poly = random_walk_polymerization(
                mols, m_idx, chi_inv, start_num=start_num, init_poly=poly_copy[0], headhead=headhead, confId=confId,
                dist_min=dist_min, retry=retry, rollback=rollback, retry_step=retry_step, retry_opt_step=retry_opt_step, tacticity=tacticity,
                res_name_init=res_name_init, res_name=res_name, label=label, label_init=label_init,
                opt=opt, ff=ff, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu
            )

    return poly


def polymerize_rw(mol, n, init_poly=None, headhead=False, confId=0, tacticity='atactic', atac_ratio=0.5,
            dist_min=0.7, retry=200, rollback=5, retry_step=1000, retry_opt_step=0, ter1=None, ter2=None,
            label=None, label_ter1=1, label_ter2=1, res_name='RU0', opt='rdkit', ff=None, work_dir=None, omp=0, mpi=1, gpu=0):
    """
    poly.polymerize_rw

    Homo-polymerization of RDkit Mol object by self-avoiding random walk

    Args:
        mol: RDkit Mol object
        n: Polymerization degree (int)

    Optional args:
        init_poly: polymerize_rw perform additional polymerization for init_poly (RDkit Mol object)
        headhead: Connect monomer unit by head-to-head
        confId: Target conformer ID
        tacticity: isotactic, syndiotactic, or atactic
        atac_ratio: Chiral inversion ration for atactic polymer
        dist_min: (float, angstrom)
        retry: Number of retry for this function when generating unsuitable structure (int)
        rollback: Number of rollback step when retry polymerize_rw (int)
        retry_step: Number of retry for a random-walk step when generating unsuitable structure (int)
        retry_opt_step: Number of retry for a random-walk step with optimization when generating unsuitable structure (int)
        opt:
            lammps: Using short time MD of lammps
            rdkit: MMFF94 optimization by RDKit
        work_dir: Work directory path of LAMMPS (str, requiring when opt is LAMMPS)
        ff: Force field object (requiring when opt is LAMMPS)
        omp: Number of threads of OpenMP in LAMMPS (int)
        mpi: Number of MPI process in LAMMPS (int)
        gpu: Number of GPU in LAMMPS (int)

    Returns:
        Rdkit Mol object
    """
    dt1 = datetime.datetime.now()
    utils.radon_print('Start poly.polymerize_rw.', level=1)

    m_idx = gen_monomer_array(1, n)
    chi_inv, check_chi = gen_chiral_inv_array([mol], m_idx, init_poly=init_poly, tacticity=tacticity, atac_ratio=atac_ratio)
    if not check_chi:
        tacticity = None

    if type(ter1) is Chem.Mol:
        if ter2 is None:
            ter2 = ter1
        mols = [mol, ter1, ter2]
        res_name = [res_name, 'TU0', 'TU1']
        m_idx = [1, *m_idx, 2]
        chi_inv = [False, *chi_inv, False]
        if label is None:
            label = [1, 1]
        label = [label, [label_ter1, label_ter1], [label_ter2, label_ter2]]
    else:
        mols = [mol]
        res_name = [res_name]
        if label is not None:
            label = [label]

    poly = random_walk_polymerization(
        mols, m_idx, chi_inv, start_num=0, init_poly=init_poly, headhead=headhead, confId=confId,
        dist_min=dist_min, retry=retry, rollback=rollback, retry_step=retry_step, retry_opt_step=retry_opt_step,
        tacticity=tacticity, res_name=res_name, label=label, opt=opt, ff=ff, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu
    )

    if type(ter1) is Chem.Mol:
        set_terminal_idx(poly)
    dt2 = datetime.datetime.now()
    utils.radon_print('Normal termination of poly.polymerize_rw. Elapsed time = %s' % str(dt2-dt1), level=1)

    return poly


def polymerize_rw_old(mol, n, init_poly=None, headhead=False, confId=0, tacticity='atactic', atac_ratio=0.5,
            dist_min=0.7, retry=10, rollback=5, retry_step=0, retry_opt_step=50, ter1=None, ter2=None,
            label=None, label_ter1=1, label_ter2=1, res_name='RU0', opt='rdkit', ff=None, work_dir=None, omp=0, mpi=1, gpu=0):
    # Backward campatibility
    return polymerize_rw(mol, n, init_poly=init_poly, headhead=headhead, confId=confId, tacticity=tacticity, atac_ratio=atac_ratio,
            dist_min=dist_min, retry=retry, rollback=rollback, retry_step=retry_step, retry_opt_step=retry_opt_step, ter1=ter1, ter2=ter2,
            label=label, label_ter1=label_ter1, label_ter2=label_ter2, res_name=res_name, opt=opt, ff=ff,
            work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu)


def polymerize_rw_mp(mol, n, init_poly=None, headhead=False, confId=0, tacticity='atactic', atac_ratio=0.5,
            dist_min=0.7, retry=100, rollback=5, retry_step=1000, retry_opt_step=0, ter1=None, ter2=None,
            label=None, label_ter1=1, label_ter2=1, res_name='RU0',
            opt='rdkit', ff=None, work_dir=None, omp=0, mpi=1, gpu=0, nchain=1, mp=None, fail_copy=True):

    utils.picklable(mol)
    if type(ter1) is Chem.Mol:
        utils.picklable(ter1)
    if type(ter2) is Chem.Mol:
        utils.picklable(ter2)

    if mp is None:
        mp = utils.cpu_count()
    np = max([nchain, mp])

    c = utils.picklable_const()
    args = [(mol, n, init_poly, headhead, confId, tacticity, atac_ratio, dist_min, retry, rollback, retry_step, retry_opt_step,
            ter1, ter2, label, label_ter1, label_ter2, res_name, opt, ff, work_dir, omp, mpi, gpu, c) for i in range(np)]

    polys = polymerize_mp_exec(_polymerize_rw_mp_worker, args, mp, nchain=nchain, fail_copy=fail_copy)

    return polys


def _polymerize_rw_mp_worker(args):
    (mol, n, init_poly, headhead, confId, tacticity, atac_ratio, dist_min, retry, rollback, retry_step, retry_opt_step,
        ter1, ter2, label, label_ter1, label_ter2, res_name, opt, ff, work_dir, omp, mpi, gpu, c) = args
    utils.restore_const(c)

    try:
        poly = polymerize_rw(mol, n, init_poly=init_poly, headhead=headhead, confId=confId,
                    tacticity=tacticity, atac_ratio=atac_ratio,
                    dist_min=dist_min, retry=retry, rollback=rollback, retry_step=retry_step, retry_opt_step=retry_opt_step,
                    ter1=ter1, ter2=ter2, label=label, label_ter1=label_ter1, label_ter2=label_ter2, res_name=res_name,
                    opt=opt, ff=ff, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu)
        utils.picklable(poly)
    except BaseException as e:
        utils.radon_print('%s' % e)
        poly = None

    return poly


def copolymerize_rw(mols, n, init_poly=None, headhead=False, confId=0, tacticity='atactic', atac_ratio=0.5,
            dist_min=0.7, retry=200, rollback=5, retry_step=1000, retry_opt_step=0, ter1=None, ter2=None,
            label=None, label_ter1=1, label_ter2=1, res_name=None, opt='rdkit', ff=None, work_dir=None, omp=0, mpi=1, gpu=0):
    """
    poly.copolymerize_rw

    Alternating copolymerization of RDkit Mol object by self-avoiding random walk

    Args:
        mols: list of RDkit Mol object
        n: Polymerization degree (int)

    Optional args:
        init_poly: polymerize_rw perform additional polymerization for init_poly (RDkit Mol object)
        headhead: Connect monomer unit by head-to-head
        confId: Target conformer ID
        tacticity: isotactic, syndiotactic, or atactic
        atac_ratio: Chiral inversion ration for atactic polymer
        dist_min: (float, angstrom)
        retry: Number of retry for this function when generating unsuitable structure (int)
        rollback: Number of rollback step when retry polymerize_rw (int)
        retry_step: Number of retry for a random-walk step when generating unsuitable structure (int)
        retry_opt_step: Number of retry for a random-walk step with optimization when generating unsuitable structure (int)
        opt:
            lammps: Using short time MD of lammps
            rdkit: MMFF94 optimization by RDKit
        work_dir: Work directory path of LAMMPS (str, requiring when opt is LAMMPS)
        ff: Force field object (requiring when opt is LAMMPS)
        omp: Number of threads of OpenMP in LAMMPS (int)
        mpi: Number of MPI process in LAMMPS (int)
        gpu: Number of GPU in LAMMPS (int)

    Returns:
        Rdkit Mol object
    """
    dt1 = datetime.datetime.now()
    utils.radon_print('Start poly.copolymerize_rw.', level=1)

    m_idx = gen_monomer_array(len(mols), n, copoly='alt')
    chi_inv, check_chi = gen_chiral_inv_array(mols, m_idx, init_poly=init_poly, tacticity=tacticity, atac_ratio=atac_ratio)
    if not check_chi:
        tacticity = None

    if res_name is None:
        res_name = ['RU%s' % const.pdb_id[i] for i in range(len(mols))]

    if type(ter1) is Chem.Mol:
        if ter2 is None:
            ter2 = ter1
        n = len(mols)
        mols = [*mols, ter1, ter2]
        res_name = [*res_name, 'TU0', 'TU1']
        m_idx = [n, *m_idx, n+1]
        chi_inv = [False, *chi_inv, False]
        if label is not None:
            label = [*label, [label_ter1, label_ter1], [label_ter2, label_ter2]]

    poly = random_walk_polymerization(
        mols, m_idx, chi_inv, start_num=0, init_poly=init_poly, headhead=headhead, confId=confId,
        dist_min=dist_min, retry=retry, rollback=rollback, retry_step=retry_step, retry_opt_step=retry_opt_step,
        tacticity=tacticity, res_name=res_name, label=label, opt=opt, ff=ff, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu
    )

    if type(ter1) is Chem.Mol:
        set_terminal_idx(poly)
    dt2 = datetime.datetime.now()
    utils.radon_print('Normal termination of poly.copolymerize_rw. Elapsed time = %s' % str(dt2-dt1), level=1)

    return poly


def copolymerize_rw_old(mols, n, init_poly=None, headhead=False, confId=0, tacticity='atactic', atac_ratio=0.5,
            dist_min=0.7, retry=10, rollback=5, retry_step=0, retry_opt_step=50, ter1=None, ter2=None,
            label=None, label_ter1=1, label_ter2=1, res_name=None, opt='rdkit', ff=None, work_dir=None, omp=0, mpi=1, gpu=0):
    # Backward campatibility
    return copolymerize_rw(mols, n, init_poly=init_poly, headhead=headhead, confId=confId, tacticity=tacticity, atac_ratio=atac_ratio,
            dist_min=dist_min, retry=retry, rollback=rollback, retry_step=retry_step, retry_opt_step=retry_opt_step, ter1=ter1, ter2=ter2,
            label=label, label_ter1=label_ter1, label_ter2=label_ter2, res_name=res_name, opt=opt, ff=ff,
            work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu)


def copolymerize_rw_mp(mols, n, init_poly=None, tacticity='atactic', atac_ratio=0.5,
            dist_min=0.7, retry=100, rollback=5, retry_step=1000, retry_opt_step=0, ter1=None, ter2=None,
            label=None, label_ter1=1, label_ter2=1, res_name=None, 
            opt='rdkit', ff=None, work_dir=None, omp=0, mpi=1, gpu=0, nchain=1, mp=None, fail_copy=True):

    for i in range(len(mols)):
        utils.picklable(mols[i])
    if type(ter1) is Chem.Mol:
        utils.picklable(ter1)
        if type(ter2) is Chem.Mol:
            utils.picklable(ter2)
        else:
            ter2 = ter1

    if mp is None:
        mp = utils.cpu_count()
    np = max([nchain, mp])

    c = utils.picklable_const()
    args = [(mols, n, init_poly, tacticity, atac_ratio, dist_min, retry, rollback, retry_step, retry_opt_step,
            ter1, ter2, label, label_ter1, label_ter2, res_name, opt, ff, work_dir, omp, mpi, gpu, c) for i in range(np)]

    polys = polymerize_mp_exec(_copolymerize_rw_mp_worker, args, mp, nchain=nchain, fail_copy=fail_copy)

    return polys


def _copolymerize_rw_mp_worker(args):
    (mols, n, init_poly, tacticity, atac_ratio, dist_min, retry, rollback, retry_step, retry_opt_step,
        ter1, ter2, label, label_ter1, label_ter2, res_name, opt, ff, work_dir, omp, mpi, gpu, c) = args
    utils.restore_const(c)

    try:
        poly = copolymerize_rw(mols, n, init_poly=init_poly, tacticity=tacticity, atac_ratio=atac_ratio,
                    dist_min=dist_min, retry=retry, rollback=rollback, retry_step=retry_step, retry_opt_step=retry_opt_step,
                    ter1=ter1, ter2=ter2, label=label, label_ter1=label_ter1, label_ter2=label_ter2, res_name=res_name,
                    opt=opt, ff=ff, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu)
        utils.picklable(poly)
    except BaseException as e:
        utils.radon_print('%s' % e)
        poly = None

    return poly


def random_copolymerize_rw(mols, n, ratio=None, reac_ratio=[], init_poly=None, headhead=False, confId=0, tacticity='atactic', atac_ratio=0.5,
            dist_min=0.7, retry=200, rollback=5, retry_step=1000, retry_opt_step=0, ter1=None, ter2=None,
            label=None, label_ter1=1, label_ter2=1, res_name=None, opt='rdkit', ff=None, work_dir=None, omp=0, mpi=1, gpu=0):
    """
    poly.random_copolymerize_rw

    Random copolymerization of RDkit Mol object by self-avoiding random walk

    Args:
        mols: List of RDkit Mol object
        n: Polymerization degree (int)

    Optional args:
        ratio: List of monomer composition ratio (float)
        reac_ratio: List of monomer reactivity ratio (float)
        init_poly: polymerize_rw perform additional polymerization for init_poly (RDkit Mol object)
        headhead: Connect monomer unit by head-to-head
        confId: Target conformer ID
        tacticity: isotactic, syndiotactic, or atactic
        atac_ratio: Chiral inversion ration for atactic polymer
        dist_min: (float, angstrom)
        retry: Number of retry for this function when generating unsuitable structure (int)
        rollback: Number of rollback step when retry polymerize_rw (int)
        retry_step: Number of retry for a random-walk step when generating unsuitable structure (int)
        retry_opt_step: Number of retry for a random-walk step with optimization when generating unsuitable structure (int)
        opt:
            lammps: Using short time MD of lammps
            rdkit: MMFF94 optimization by RDKit
        work_dir: Work directory path of LAMMPS (str, requiring when opt is LAMMPS)
        ff: Force field object (requiring when opt is LAMMPS)
        omp: Number of threads of OpenMP in LAMMPS (int)
        mpi: Number of MPI process in LAMMPS (int)
        gpu: Number of GPU in LAMMPS (int)

    Returns:
        Rdkit Mol object
    """
    dt1 = datetime.datetime.now()
    utils.radon_print('Start poly.random_copolymerize_rw.', level=1)

    if ratio is None:
        ratio = np.full(len(mols), 1/len(mols))

    if len(mols) != len(ratio):
        utils.radon_print('Inconsistency length of mols and ratio', level=3)

    m_idx = gen_monomer_array(len(mols), n, copoly='random', ratio=ratio, reac_ratio=reac_ratio)
    chi_inv, check_chi = gen_chiral_inv_array(mols, m_idx, init_poly=init_poly, tacticity=tacticity, atac_ratio=atac_ratio)
    if not check_chi:
        tacticity = None

    if res_name is None:
        res_name = ['RU%s' % const.pdb_id[i] for i in range(len(mols))]

    if type(ter1) is Chem.Mol:
        if ter2 is None:
            ter2 = ter1
        n = len(mols)
        mols = [*mols, ter1, ter2]
        res_name = [*res_name, 'TU0', 'TU1']
        m_idx = [n, *m_idx, n+1]
        chi_inv = [False, *chi_inv, False]
        if label is not None:
            label = [*label, [label_ter1, label_ter1], [label_ter2, label_ter2]]

    poly = random_walk_polymerization(
        mols, m_idx, chi_inv, start_num=0, init_poly=init_poly, headhead=headhead, confId=confId,
        dist_min=dist_min, retry=retry, rollback=rollback, retry_step=retry_step, retry_opt_step=retry_opt_step,
        tacticity=tacticity, res_name=res_name, label=label, opt=opt, ff=ff, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu
    )

    if type(ter1) is Chem.Mol:
        set_terminal_idx(poly)
    dt2 = datetime.datetime.now()
    utils.radon_print('Normal termination of poly.random_copolymerize_rw. Elapsed time = %s' % str(dt2-dt1), level=1)

    return poly


def random_copolymerize_rw_old(mols, n, ratio=None, reac_ratio=[], init_poly=None, headhead=False, confId=0, tacticity='atactic', atac_ratio=0.5,
            dist_min=0.7, retry=10, rollback=5, retry_step=0, retry_opt_step=50, ter1=None, ter2=None,
            label=None, label_ter1=1, label_ter2=1, res_name=None, opt='rdkit', ff=None, work_dir=None, omp=0, mpi=1, gpu=0):
    # Backward campatibility
    return random_copolymerize_rw(mols, n, ratio=ratio, reac_ratio=reac_ratio, init_poly=init_poly, headhead=headhead, confId=confId,
            tacticity=tacticity, atac_ratio=atac_ratio, dist_min=dist_min, retry=retry, rollback=rollback, retry_step=retry_step,
            retry_opt_step=retry_opt_step, ter1=ter1, ter2=ter2, label=label, label_ter1=label_ter1, label_ter2=label_ter2,
            res_name=res_name, opt=opt, ff=ff, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu)


def random_copolymerize_rw_mp(mols, n, ratio=None, reac_ratio=[], init_poly=None, tacticity='atactic', atac_ratio=0.5,
                dist_min=0.7, retry=100, rollback=5, retry_step=1000, retry_opt_step=0, ter1=None, ter2=None,
                label=None, label_ter1=1, label_ter2=1, res_name=None,
                opt='rdkit', ff=None, work_dir=None, omp=1, mpi=1, gpu=0, nchain=1, mp=None, fail_copy=True):

    for i in range(len(mols)):
        utils.picklable(mols[i])
    if type(ter1) is Chem.Mol:
        utils.picklable(ter1)
        if type(ter2) is Chem.Mol:
            utils.picklable(ter2)
        else:
            ter2 = ter1

    if mp is None:
        mp = utils.cpu_count()
    np = max([nchain, mp])

    c = utils.picklable_const()
    args = [(mols, n, ratio, reac_ratio, init_poly, tacticity, atac_ratio, dist_min, retry, rollback, retry_step, retry_opt_step,
            ter1, ter2, label, label_ter1, label_ter2, res_name, opt, ff, work_dir, omp, mpi, gpu, c) for i in range(np)]

    polys = polymerize_mp_exec(_random_copolymerize_rw_mp_worker, args, mp, nchain=nchain, fail_copy=fail_copy)

    return polys


def _random_copolymerize_rw_mp_worker(args):
    (mols, n, ratio, reac_ratio, init_poly, tacticity, atac_ratio, dist_min, retry, rollback, retry_step, retry_opt_step,
        ter1, ter2, label, label_ter1, label_ter2, res_name, opt, ff, work_dir, omp, mpi, gpu, c) = args
    utils.restore_const(c)

    try:
        poly = random_copolymerize_rw(mols, n, ratio=ratio, reac_ratio=reac_ratio, init_poly=init_poly, tacticity=tacticity, atac_ratio=atac_ratio,
                    dist_min=dist_min, retry=retry, rollback=rollback, retry_step=retry_step, retry_opt_step=retry_opt_step,
                    ter1=ter1, ter2=ter2, label=label, label_ter1=label_ter1, label_ter2=label_ter2, res_name=res_name,
                    opt=opt, ff=ff, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu)
        utils.picklable(poly)
    except BaseException as e:
        utils.radon_print('%s' % e)
        poly = None

    return poly


def block_copolymerize_rw(mols, n, init_poly=None, headhead=False, confId=0, tacticity='atactic', atac_ratio=0.5,
            dist_min=0.7, retry=200, rollback=5, retry_step=1000, retry_opt_step=0, ter1=None, ter2=None,
            label=None, label_ter1=1, label_ter2=1, res_name=None, opt='rdkit', ff=None, work_dir=None, omp=0, mpi=1, gpu=0):
    """
    poly.block_copolymerize_rw

    Block copolymerization of RDkit Mol object by self-avoiding random walk

    Args:
        mols: List of RDkit Mol object
        n: List of polymerization degree (list, int)

    Optional args:
        init_poly: polymerize_rw perform additional polymerization for init_poly (RDkit Mol object)
        headhead: Connect monomer unit by head-to-head
        confId: Target conformer ID
        tacticity: isotactic, syndiotactic, or atactic
        atac_ratio: Chiral inversion ration for atactic polymer
        dist_min: (float, angstrom)
        retry: Number of retry for this function when generating unsuitable structure (int)
        rollback: Number of rollback step when retry polymerize_rw (int)
        retry_step: Number of retry for a random-walk step when generating unsuitable structure (int)
        retry_opt_step: Number of retry for a random-walk step with optimization when generating unsuitable structure (int)
        opt:
            lammps: Using short time MD of lammps
            rdkit: MMFF94 optimization by RDKit
        work_dir: Work directory path of LAMMPS (str, requiring when opt is LAMMPS)
        ff: Force field object (requiring when opt is LAMMPS)
        omp: Number of threads of OpenMP in LAMMPS (int)
        mpi: Number of MPI process in LAMMPS (int)
        gpu: Number of GPU in LAMMPS (int)

    Returns:
        Rdkit Mol object
    """
    dt1 = datetime.datetime.now()
    utils.radon_print('Start poly.block_copolymerize_rw.', level=1)

    if len(mols) != len(n):
        utils.radon_print('Inconsistency length of mols and n', level=3)

    m_idx = gen_monomer_array(len(mols), n, copoly='block')
    chi_inv, check_chi = gen_chiral_inv_array(mols, m_idx, init_poly=init_poly, tacticity=tacticity, atac_ratio=atac_ratio)
    if not check_chi:
        tacticity = None

    if res_name is None:
        res_name = ['RU%s' % const.pdb_id[i] for i in range(len(mols))]
        
    if type(ter1) is Chem.Mol:
        if ter2 is None:
            ter2 = ter1
        n = len(mols)
        mols = [*mols, ter1, ter2]
        res_name = [*res_name, 'TU0', 'TU1']
        m_idx = [n, *m_idx, n+1]
        chi_inv = [False, *chi_inv, False]
        if label is not None:
            label = [*label, [label_ter1, label_ter1], [label_ter2, label_ter2]]

    poly = random_walk_polymerization(
        mols, m_idx, chi_inv, start_num=0, init_poly=init_poly, headhead=headhead, confId=confId,
        dist_min=dist_min, retry=retry, rollback=rollback, retry_step=retry_step, retry_opt_step=retry_opt_step,
        tacticity=tacticity, res_name=res_name, label=label, opt=opt, ff=ff, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu
    )

    if type(ter1) is Chem.Mol:
        set_terminal_idx(poly)
    dt2 = datetime.datetime.now()
    utils.radon_print('Normal termination of poly.block_copolymerize_rw. Elapsed time = %s' % str(dt2-dt1), level=1)

    return poly


def block_copolymerize_rw_old(mols, n, init_poly=None, headhead=False, confId=0, tacticity='atactic', atac_ratio=0.5,
            dist_min=0.7, retry=10, rollback=5, retry_step=0, retry_opt_step=50, ter1=None, ter2=None,
            label=None, label_ter1=1, label_ter2=1, res_name=None, opt='rdkit', ff=None, work_dir=None, omp=0, mpi=1, gpu=0):
    # Backward campatibility
    return block_copolymerize_rw(mols, n, init_poly=init_poly, headhead=headhead, confId=confId,
            tacticity=tacticity, atac_ratio=atac_ratio, dist_min=dist_min, retry=retry, rollback=rollback, retry_step=retry_step,
            retry_opt_step=retry_opt_step, ter1=ter1, ter2=ter2, label=label, label_ter1=label_ter1, label_ter2=label_ter2,
            res_name=res_name, opt=opt, ff=ff, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu)


def block_copolymerize_rw_mp(mols, n, init_poly=None, tacticity='atactic', atac_ratio=0.5,
            dist_min=0.7, retry=100, rollback=5, retry_step=1000, retry_opt_step=0, ter1=None, ter2=None,
            label=None, label_ter1=1, label_ter2=1, res_name=None, 
            opt='rdkit', ff=None, work_dir=None, omp=0, mpi=1, gpu=0, nchain=10, mp=10, fail_copy=True):

    for i in range(len(mols)):
        utils.picklable(mols[i])
    if type(ter1) is Chem.Mol:
        utils.picklable(ter1)
        if type(ter2) is Chem.Mol:
            utils.picklable(ter2)
        else:
            ter2 = ter1

    if mp is None:
        mp = utils.cpu_count()
    np = max([nchain, mp])

    c = utils.picklable_const()
    args = [(mols, n, init_poly, tacticity, atac_ratio, dist_min, retry, rollback, retry_step, retry_opt_step,
            ter1, ter2, label, label_ter1, label_ter2, res_name, opt, ff, work_dir, omp, mpi, gpu, c) for i in range(np)]

    polys = polymerize_mp_exec(_block_copolymerize_rw_mp_worker, args, mp, nchain=nchain, fail_copy=fail_copy)

    return polys


def _block_copolymerize_rw_mp_worker(args):
    (mols, n, init_poly, tacticity, atac_ratio, dist_min, retry, rollback, retry_step, retry_opt_step,
        ter1, ter2, label, label_ter1, label_ter2, res_name, opt, ff, work_dir, omp, mpi, gpu, c) = args
    utils.restore_const(c)

    try:
        poly = block_copolymerize_rw(mols, n, init_poly=init_poly, tacticity=tacticity, atac_ratio=atac_ratio,
                    dist_min=dist_min, retry=retry, rollback=rollback, retry_step=retry_step, retry_opt_step=retry_opt_step,
                    ter1=ter1, ter2=ter2, label=label, label_ter1=label_ter1, label_ter2=label_ter2, res_name=res_name,
                    opt=opt, ff=ff, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu)
        utils.picklable(poly)
    except BaseException as e:
        utils.radon_print('%s' % e)
        poly = None

    return poly


def polymerize_mp_exec(func, args, mp, nchain=1, fail_copy=True):
    executor = confu.ProcessPoolExecutor(max_workers=mp, mp_context=MP.get_context('spawn'))
    futures = [executor.submit(func, arg) for arg in args]

    results = []
    success = 0
    for future in confu.as_completed(futures):
        res = future.result()
        results.append(res)
        if type(res) is Chem.Mol:
            success += 1
        if success >= nchain:
            break

    for future in futures:
        if not future.running():
            future.cancel()

    for process in executor._processes.values():
        process.kill()

    executor.shutdown(wait=False)

    polys = [res for res in results if type(res) is Chem.Mol]

    if len(polys) == 0:
        utils.radon_print('Generation of polymer chain by random walk was failure.', level=3)
    elif len(polys) < nchain:
        if fail_copy:
            for i in range(nchain-len(polys)):
                r_idx = np.random.choice(range(len(polys)))
                polys.append(utils.deepcopy_mol(polys[r_idx]))
            utils.radon_print('%i success, %i copy' % (len(polys), (nchain-len(polys))))
        else:
            for i in range(nchain-len(polys)):
                polys.append(None)
            utils.radon_print('%i success, %i fail' % (len(polys), (nchain-len(polys))))
    elif len(polys) > nchain:
        polys = polys[:nchain]

    return polys


def terminate_rw(poly, mol1, mol2=None, confId=0, dist_min=1.0, retry_step=1000, retry_opt_step=0,
            res_name='RU0', label=None, opt='rdkit', ff=None, work_dir=None, omp=0, mpi=1, gpu=0):
    """
    poly.terminate_rw

    Termination of polymer of RDkit Mol object by random walk

    Args:
        poly: RDkit Mol object of a polymer
        mol1: RDkit Mol object of a terminal unit (head side or both sides)

    Optional args:
        mol2: RDkit Mol object of a terminal unit (tail side)
        confId: Target conformer ID
        dist_min: (float, angstrom)
        retry_step: Number of retry for a random-walk step when generating unsuitable structure (int)
        retry_opt_step: Number of retry for a random-walk step with optimization when generating unsuitable structure (int)
        opt:
            lammps: Using short time MD of lammps
            rdkit: MMFF94 optimization by RDKit
        work_dir: Work directory path of LAMMPS (str, requiring when opt is LAMMPS)
        ff: Force field object (requiring when opt is LAMMPS)
        omp: Number of threads of OpenMP in LAMMPS (int)
        mpi: Number of MPI process in LAMMPS (int)
        gpu: Number of GPU in LAMMPS (int)

    Returns:
        Rdkit Mol object
    """
    dt1 = datetime.datetime.now()
    utils.radon_print('Start poly.terminate_rw.', level=1)

    if mol2 is None:
        mol2 = mol1
    H2_flag1 = False
    H2_flag2 = False
    res_name_1 = 'TU0'
    res_name_2 = 'TU1'
    poly_c = utils.deepcopy_mol(poly)

    if Chem.MolToSmiles(mol1) == '[H][3H]' or Chem.MolToSmiles(mol1) == '[3H][H]':
        poly_c.GetAtomWithIdx(poly_c.GetIntProp('head_idx')).SetIsotope(1)
        poly_c.GetAtomWithIdx(poly_c.GetIntProp('head_idx')).GetPDBResidueInfo().SetResidueName(res_name_1)
        poly_c.GetAtomWithIdx(poly_c.GetIntProp('head_idx')).GetPDBResidueInfo().SetResidueNumber(1+poly_c.GetIntProp('num_units'))
        poly_c.SetIntProp('num_units', 1+poly_c.GetIntProp('num_units'))
        H2_flag1 = True
    if Chem.MolToSmiles(mol2) == '[H][3H]' or Chem.MolToSmiles(mol2) == '[3H][H]':
        poly_c.GetAtomWithIdx(poly_c.GetIntProp('tail_idx')).SetIsotope(1)
        poly_c.GetAtomWithIdx(poly_c.GetIntProp('tail_idx')).GetPDBResidueInfo().SetResidueName(res_name_2)
        poly_c.GetAtomWithIdx(poly_c.GetIntProp('tail_idx')).GetPDBResidueInfo().SetResidueNumber(1+poly_c.GetIntProp('num_units'))
        poly_c.SetIntProp('num_units', 1+poly_c.GetIntProp('num_units'))
        H2_flag2 = True

    mols = [poly_c, mol1, mol2]
    res_name = [res_name, res_name_1, res_name_2]

    if H2_flag1 and H2_flag2:
        pass
    elif H2_flag2:
        mon_idx = [1, 0]
        chi_inv = [False, False]
    elif H2_flag1:
        mon_idx = [0, 1]
        chi_inv = [False, False]
    else:
        mon_idx = [1, 0, 2]
        chi_inv = [False, False, False]

    if not H2_flag1 or not H2_flag2:
        poly_c = random_walk_polymerization(
            mols, mon_idx, chi_inv, confId=confId,
            dist_min=dist_min, retry=0, retry_step=retry_step, retry_opt_step=retry_opt_step,
            res_name=res_name, label=label, opt=opt, ff=ff, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu
        )

    set_terminal_idx(poly_c)
    dt2 = datetime.datetime.now()
    utils.radon_print('Normal termination of poly.terminate_rw. Elapsed time = %s' % str(dt2-dt1), level=1)

    return poly_c


def gen_monomer_array(n_mon, n, copoly='alt', ratio=[], reac_ratio=[]):
    """
    poly.gen_monomer_array

    Monomer array generator for the polymer chain generators
    """
    mon_array = []
    if type(n) is list and len(n) == 1:
        n = int(n[0])

    # Homopolymer
    if n_mon == 1:
        mon_array = np.zeros(n)

    # Alternating copolymer
    elif n_mon > 1 and copoly == 'alt':
        mon_array = np.tile(list(range(n_mon)), n)

    # Random copolymer
    elif n_mon > 1 and copoly == 'random':
        # Monomer reactivity ratio
        if len(reac_ratio) == 2 and n_mon == 2:
            p11 = reac_ratio[0]/(1+reac_ratio[0])
            p22 = reac_ratio[1]/(1+reac_ratio[1])

            if np.random.rand(1)[0] >= 0.5:
                pre = 0
                mon_array.append(0)
            else:
                pre = 1
                mon_array.append(1)

            for i in range(n-1):
                rand = np.random.rand(1)[0]
                if pre == 0:
                    if rand <= p11:
                        pre = 0
                        mon_array.append(0)
                    else:
                        pre = 1
                        mon_array.append(1)
                else:
                    if rand <= p22:
                        pre = 1
                        mon_array.append(1)
                    else:
                        pre = 0
                        mon_array.append(0)
            mon_array = np.array(mon_array)

        # Monomer composition ratio
        else:
            if len(ratio) != n_mon:
                utils.radon_print('Inconsistency length of mols and ratio', level=3)

            for i, r in enumerate(ratio):
                tmp = [i]*int(n*r+0.5)
                mon_array.extend(tmp)
            if len(mon_array) < n:
                d = int(n-len(mon_array))
                imax = np.argmax(ratio)
                tmp = [imax]*d
                mon_array.extend(tmp)
            elif len(mon_array) > n:
                d = int(len(mon_array)-n)
                imax = np.argmax(ratio)
                for i in range(d):
                    mon_array.remove(imax)
            mon_array = np.array(mon_array[:])
            np.random.shuffle(mon_array)

    # Block copolymer
    elif n_mon > 1 and copoly == 'block':
        if type(n) is int:
            n = np.full(n_mon, n)

        if len(n) != n_mon:
            utils.radon_print('Inconsistency length of mols and n', level=3)

        for i in range(n_mon):
            tmp = [i]*n[i]
            mon_array.extend(tmp)
        mon_array = np.array(mon_array[:])

    else:
        utils.radon_print('Illegal input for copoly=%s in poly.gen_monomer_array.' % copoly, level=3)

    mon_array = [int(x) for x in list(mon_array)]

    return mon_array


def gen_chiral_inv_array(mols, mon_array, init_poly=None, tacticity='atactic', atac_ratio=0.5):
    """
    poly.gen_chiral_inv_array

    Chiral inversion array generator for the polymer chain generators
    """
    mon_array = np.array(mon_array)
    chi_inv = np.full(len(mon_array), False)
    chiral_poly = []
    chiral_poly_last = 0
    chiral_mon = []
    n_mon = []
    check_chi = True

    if type(init_poly) is Chem.Mol:
        chiral_poly = get_chiral_list(init_poly)
        chiral_poly_last = chiral_poly[-1] if len(chiral_poly) > 0 else 0

    for i, mol in enumerate(mols):
        if i == 0:
            n_mon.append(len(mon_array) - np.count_nonzero(mon_array))
        else:
            n_mon.append(np.count_nonzero(mon_array == i))
        num_chiral = check_chiral_monomer(mol)
        if num_chiral == 0:
            chiral_mon.append(0)
        elif num_chiral == 1:
            chiral_mon.append(get_chiral_list(mol)[0])
        else:
            chiral_mon.append(0)
            if check_chi:
                utils.radon_print(
                    'Found multiple chiral center in the mainchain of polymer repeating unit. The chirality control is turned off.',
                    level=1)
                check_chi = False

    if tacticity == 'isotactic':
        flag = chiral_poly_last
        for i, mon_idx in enumerate(mon_array):
            if flag == 0 and chiral_mon[mon_idx] != 0:
                flag = chiral_mon[mon_idx]
            if chiral_mon[mon_idx] == 0:
                chi_inv[i] = False
            elif flag == chiral_mon[mon_idx]:
                chi_inv[i] = False
            else:
                chi_inv[i] = True

    elif tacticity == 'syndiotactic':
        flag = chiral_poly_last
        for i, mon_idx in enumerate(mon_array):
            if chiral_mon[mon_idx] == 0:
                chi_inv[i] = False
            elif flag == chiral_mon[mon_idx]:
                chi_inv[i] = True
                flag = 2 if chiral_mon[mon_idx] == 1 else 1
            else:
                chi_inv[i] = False
                flag = 1 if chiral_mon[mon_idx] == 1 else 2

    elif tacticity == 'atactic':
        chi_inv_list = []
        for n, c in zip(n_mon, chiral_mon):
            chi_mon_inv = np.full(n, False)
            if c == 0:
                chi_inv_list.append(list(chi_mon_inv))
            elif c == 1:
                n_inv = n*atac_ratio
                chi_mon_inv[int(n_inv):] = True
                np.random.shuffle(chi_mon_inv)
                chi_inv_list.append(list(chi_mon_inv))
            else:
                n_inv = n*(1-atac_ratio)
                chi_mon_inv[int(n_inv):] = True
                np.random.shuffle(chi_mon_inv)
                chi_inv_list.append(list(chi_mon_inv))

        for i, mon_idx in enumerate(mon_array):
            chi_inv[i] = chi_inv_list[mon_idx].pop()

    else:
        utils.radon_print('%s is illegal input for tacticity.' % str(tacticity), level=3)

    return chi_inv, check_chi


######################################################
# Unit cell generators
######################################################
def amorphous_cell(mols, n, cell=None, density=0.1, retry=20, retry_step=1000, threshold=2.0, dec_rate=0.8,
        check_bond_ring_intersection=False, mp=0, restart_flag=False):
    """
    poly.amorphous_cell

    Simple unit cell generator for amorphous system

    Args:
        mols: RDkit Mol object or its list
        n: Number of molecules in the unit cell (int) or its list

    Optional args:
        cell: Initial structure of unit cell (RDkit Mol object)
        density: (float, g/cm3)
        retry: Number of retry for this function when inter-molecular atom-atom distance is below threshold (int)
        retry_step: Number of retry for a random placement step when inter-molecular atom-atom distance is below threshold (int)
        threshold: Threshold of inter-molecular atom-atom distance (float, angstrom)
        dec_rate: Decrease rate of density when retry for this function (float)

    Returns:
        Rdkit Mol object
    """
    if not restart_flag:
        dt1 = datetime.datetime.now()
        utils.radon_print('Start amorphous cell generation by poly.amorphous_cell.', level=1)

    if type(mols) is Chem.Mol:
        mols = [mols]
    if type(n) is int:
        n = [int(n)]

    mols_c = [utils.deepcopy_mol(mol) for mol in mols]
    mol_coord = [np.array(mol.GetConformer(0).GetPositions()) for mol in mols_c]
    has_ring = False
    tri_coord = None
    bond_coord = None

    if cell is None:
        cell_c = Chem.Mol()
    else:
        cell_c = utils.deepcopy_mol(cell)
        if Chem.GetSSSR(cell_c) > 0:
            has_ring = True

    for mol_c in mols_c:
        if Chem.GetSSSR(mol_c) > 0:
            has_ring = True

    if density is None and hasattr(cell_c, 'cell'):
        xhi = cell_c.cell.xhi
        xlo = cell_c.cell.xlo
        yhi = cell_c.cell.yhi
        ylo = cell_c.cell.ylo
        zhi = cell_c.cell.zhi
        zlo = cell_c.cell.zlo
    else:
        xhi, xlo, yhi, ylo, zhi, zlo = calc_cell_length([*mols_c, cell_c], [*n, 1], density=density)
        setattr(cell_c, 'cell', utils.Cell(xhi, xlo, yhi, ylo, zhi, zlo))

    retry_flag = False
    for m, mol in enumerate(mols_c):
        for i in tqdm(range(n[m]), desc='[Unit cell generation %i/%i]' % (m+1, len(mols_c)), disable=const.tqdm_disable):
            for r in range(retry_step):
                # Random rotation and translation
                trans = np.array([
                    np.random.uniform(xlo, xhi),
                    np.random.uniform(ylo, yhi),
                    np.random.uniform(zlo, zhi)
                ])
                rot = np.random.uniform(-np.pi, np.pi, 3)

                mol_coord_c = mol_coord[m]
                mol_coord_c = calc.rotate_rod(mol_coord_c, np.array([1, 0, 0]), rot[0])
                mol_coord_c = calc.rotate_rod(mol_coord_c, np.array([0, 1, 0]), rot[1])
                mol_coord_c = calc.rotate_rod(mol_coord_c, np.array([0, 0, 1]), rot[2])
                mol_coord_c += trans - np.mean(mol_coord_c, axis=0)

                if cell_c.GetNumConformers() == 0: break

                check_3d = check_3d_structure_cell(cell_c, mol_coord_c, dist_min=threshold)
                if check_3d and check_bond_ring_intersection and has_ring:
                    check_3d, tri_coord_new, bond_coord_new = check_3d_bond_ring_intersection(cell_c, mon=mol_c, mon_coord=mol_coord_c,
                                                                    tri_coord=tri_coord, bond_coord=bond_coord, mp=mp)
                if check_3d:
                    if check_bond_ring_intersection and has_ring:
                        tri_coord = tri_coord_new
                        bond_coord = bond_coord_new
                    break
                elif r < retry_step-1:
                    if r == 0 or (r+1) % 100 == 0:
                        step_n = sum(n[:m])+i+1 if m > 0 else i+1
                        utils.radon_print('Retry placing a molecule in cell. Step=%i, %i/%i' % (step_n, r+1, retry_step))
                else:
                    retry_flag = True
                    step_n = sum(n[:m])+i+1 if m > 0 else i+1
                    utils.radon_print('Reached maximum number of retrying in the step %i of poly.amorphous_cell.' % step_n, level=1)

            if retry_flag and retry > 0: break

            cell_n = cell_c.GetNumAtoms()

            # Add Mol to cell
            cell_c = combine_mols(cell_c, mol)

            # Set atomic coordinate
            for j in range(mol.GetNumAtoms()):
                cell_c.GetConformer(0).SetAtomPosition(
                    cell_n+j,
                    Geom.Point3D(mol_coord_c[j, 0], mol_coord_c[j, 1], mol_coord_c[j, 2])
                )
            
        if retry_flag and retry > 0: break

    if retry_flag:
        if retry <= 0:
            utils.radon_print('Reached maximum number of retrying poly.amorphous_cell.', level=3)
        else:
            density *= dec_rate
            utils.radon_print('Retry poly.amorphous_cell. Remainig %i times. The density is reduced to %f.' % (retry, density), level=1)
            retry -= 1
            cell_c = amorphous_cell(mols, n, cell=cell, density=density, retry=retry,
                    retry_step=retry_step, threshold=threshold, dec_rate=dec_rate,
                    check_bond_ring_intersection=check_bond_ring_intersection, mp=mp, restart_flag=True)

    if not restart_flag:
        dt2 = datetime.datetime.now()
        utils.radon_print('Normal termination of poly.amorphous_cell. Elapsed time = %s' % str(dt2-dt1), level=1)

    return cell_c


def amorphous_mixture_cell(mols, n, cell=None, density=0.1, retry=20, retry_step=1000, threshold=2.0, dec_rate=0.8,
                            check_bond_ring_intersection=False, mp=0):
    """
    poly.amorphous_mixture_cell

    This function is alias of poly.amorphous_cell to maintain backward compatibility
    """
    return amorphous_cell(mols, n, cell=cell, density=density, retry=retry, retry_step=retry_step, threshold=threshold, dec_rate=dec_rate,
                            check_bond_ring_intersection=check_bond_ring_intersection, mp=mp, restart_flag=False)


def nematic_cell(mols, n, cell=None, density=0.1, retry=20, retry_step=1000, threshold=2.0, dec_rate=0.8,
        check_bond_ring_intersection=False, mp=0, restart_flag=False):
    """
    poly.nematic_cell

    Simple unit cell generator with nematic-like ordered structure for x axis

    Args:
        mols: Array of RDkit Mol object
        n: Array of number of molecules in the unit cell (int)

    Optional args:
        cell: Initial structure of unit cell (RDkit Mol object)
        density: (float, g/cm3)
        retry: Number of retry for this function when inter-molecular atom-atom distance is below threshold (int)
        retry_step: Number of retry for a random placement step when inter-molecular atom-atom distance is below threshold (int)
        threshold: Threshold of inter-molecular atom-atom distance (float, angstrom)
        dec_rate: Decrease rate of density when retry for this function (float)

    Returns:
        Rdkit Mol object
    """
    if not restart_flag:
        dt1 = datetime.datetime.now()
        utils.radon_print('Start amorphous cell generation by poly.nematic_cell.', level=1)

    if type(mols) is Chem.Mol:
        mols = [mols]
    if type(n) is int:
        n = [int(n)]

    has_ring = False
    tri_coord = None
    bond_coord = None

    if cell is None:
        cell_c = Chem.Mol()
    else:
        cell_c = utils.deepcopy_mol(cell)
        if Chem.GetSSSR(cell_c) > 0:
            has_ring = True

    mols_c = [utils.deepcopy_mol(mol) for mol in mols]
    mol_coord = [np.array(mol.GetConformer(0).GetPositions()) for mol in mols_c]
    # Alignment molecules
    for mol in mols_c:
        Chem.rdMolTransforms.CanonicalizeConformer(mol.GetConformer(0), ignoreHs=False)
        if Chem.GetSSSR(mol_c) > 0:
            has_ring = True

    if density is None and hasattr(cell_c, 'cell'):
        xhi = cell_c.cell.xhi
        xlo = cell_c.cell.xlo
        yhi = cell_c.cell.yhi
        ylo = cell_c.cell.ylo
        zhi = cell_c.cell.zhi
        zlo = cell_c.cell.zlo
    else:
        xhi, xlo, yhi, ylo, zhi, zlo = calc_cell_length([*mols_c, cell_c], [*n, 1], density=density)
        setattr(cell_c, 'cell', utils.Cell(xhi, xlo, yhi, ylo, zhi, zlo))

    utils.radon_print('Start nematic-like cell generation.', level=1)
    retry_flag = False
    for m, mol in enumerate(mols_c):
        for i in tqdm(range(n[m]), desc='[Unit cell generation %i/%i]' % (m+1, len(mols_c)), disable=const.tqdm_disable):
            for r in range(retry_step):
                # Random translation
                trans = np.array([
                    np.random.uniform(xlo, xhi),
                    np.random.uniform(ylo, yhi),
                    np.random.uniform(zlo, zhi)
                ])

                #Random rotation around x axis
                rot = np.random.uniform(-np.pi, np.pi)

                mol_coord_c = mol_coord[m]
                if i % 2 == 0:
                    mol_coord_c = calc.rotate_rod(mol_coord_c, np.array([1, 0, 0]), rot) + trans
                else:
                    mol_coord_c = calc.rotate_rod(mol_coord_c, np.array([1, 0, 0]), rot)
                    mol_coord_c = calc.rotate_rod(mol_coord_c, np.array([0, 1, 0]), np.pi)
                    mol_coord_c += trans

                if cell_c.GetNumConformers() == 0: break

                check_3d = check_3d_structure_cell(cell_c, mol_coord_c, dist_min=threshold)
                if check_3d and check_bond_ring_intersection and has_ring:
                    check_3d, tri_coord_new, bond_coord_new = check_3d_bond_ring_intersection(cell_c, mon=mol_c, mon_coord=mol_coord_c,
                                                                    tri_coord=tri_coord, bond_coord=bond_coord, mp=mp)
                if check_3d:
                    if check_bond_ring_intersection and has_ring:
                        tri_coord = tri_coord_new
                        bond_coord = bond_coord_new
                    break
                elif r < retry_step-1:
                    if r == 0 or (r+1) % 100 == 0:
                        step_n = sum(n[:m])+i+1 if m > 0 else i+1
                        utils.radon_print('Retry placing a molecule in cell. Step=%i, %i/%i' % (step_n, r+1, retry_step))
                else:
                    retry_flag = True
                    step_n = sum(n[:m])+i+1 if m > 0 else i+1
                    utils.radon_print('Reached maximum number of retrying in the step %i of poly.nematic_cell.' % step_n, level=1)

            if retry_flag and retry > 0: break

            cell_n = cell_c.GetNumAtoms()

            # Add Mol to cell
            cell_c = combine_mols(cell_c, mol)

            # Set atomic coordinate
            for j in range(mol.GetNumAtoms()):
                cell_c.GetConformer(0).SetAtomPosition(
                    cell_n+j,
                    Geom.Point3D(mol_coord_c[j, 0], mol_coord_c[j, 1], mol_coord_c[j, 2])
                )

        if retry_flag and retry > 0: break

    if retry_flag:
        if retry <= 0:
            utils.radon_print('Reached maximum number of retrying poly.nematic_cell.', level=3)
        else:
            density *= dec_rate
            utils.radon_print('Retry poly.nematic_cell. Remainig %i times. The density is reduced to %f.' % (retry, density), level=1)
            retry -= 1
            cell_c = nematic_cell(mols, n, cell=cell, density=density, retry=retry,
                    retry_step=retry_step, threshold=threshold, dec_rate=dec_rate,
                    check_bond_ring_intersection=check_bond_ring_intersection, mp=mp, restart_flag=True)

    if not restart_flag:
        dt2 = datetime.datetime.now()
        utils.radon_print('Normal termination of poly.nematic_cell. Elapsed time = %s' % str(dt2-dt1), level=1)

    return cell_c


def nematic_mixture_cell(mols, n, cell=None, density=0.1, retry=20, retry_step=1000, threshold=2.0, dec_rate=0.8,
                            check_bond_ring_intersection=False, mp=0):
    """
    poly.nematic_mixture_cell

    This function is alias of poly.nematic_cell to maintain backward compatibility
    """
    return nematic_cell(mols, n, cell=cell, density=density, retry=retry, retry_step=retry_step, threshold=threshold, dec_rate=dec_rate,
                            check_bond_ring_intersection=check_bond_ring_intersection, mp=mp, restart_flag=False)


# DEPRECATION
def polymerize_cell(mol, n, m, terminate=None, terminate2=None, cell=None, density=0.1,
                    ff=None, opt='rdkit', retry=50, dist_min=0.7, threshold=2.0, work_dir=None, omp=1, mpi=1, gpu=0):
    """
    poly.polymerize_cell

    *** DEPRECATION ***
    Unit cell generator of a homopolymer by random walk

    Args:
        mol: RDkit Mol object
        n: Polymerization degree (int)
        m: Number of polymer chains (int)

    Optional args:
        terminate: terminated substitute at head (and tail) (RDkit Mol object)
        terminate2: terminated substitute at tail (RDkit Mol object)
        cell: Initial structure of unit cell (RDkit Mol object)
        density: (float, g/cm3)
        ff: Force field object (requiring when use_lammps is True)
        opt:
            True: Using short time MD of lammps
            False: Dihedral angle around connecting bond is rotated randomly
        retry: Number of retry when generating unsuitable structure (int)
        dist_min: Threshold of intra-molecular atom-atom distance(float, angstrom)
        threshold: Threshold of inter-molecular atom-atom distance (float, angstrom)

    Returns:
        Rdkit Mol object
    """

    mol_n = mol.GetNumAtoms()
    if terminate2 is None: terminate2 = terminate

    if cell is None:
        cell = Chem.Mol()
    else:
        cell = utils.deepcopy_mol(cell)
 
    if density is None and hasattr(cell, 'cell'):
        xhi = cell.cell.xhi
        xlo = cell.cell.xlo
        yhi = cell.cell.yhi
        ylo = cell.cell.ylo
        zhi = cell.cell.zhi
        zlo = cell.cell.zlo
    else:
        if terminate is None:
            xhi, xlo, yhi, ylo, zhi, zlo = calc_cell_length([mol, cell], [n*m, 1], density=density)
        else:
            xhi, xlo, yhi, ylo, zhi, zlo = calc_cell_length([mol, terminate, terminate2, cell], [n*m, m, m, 1], density=density)
        setattr(cell, 'cell', utils.Cell(xhi, xlo, yhi, ylo, zhi, zlo))

    for j in range(m):
        cell_n = cell.GetNumAtoms()
        cell = replicate_cell(mol, 1, cell=cell, density=None, retry=retry, threshold=threshold)

        for i in tqdm(range(n-1), desc='[Unit cell generation %i/%i]' % (j+1, m), disable=const.tqdm_disable):
            cell_copy = utils.deepcopy_mol(cell)

            for r in range(retry):
                cell = connect_mols(cell, mol, random_rot=True)

                if opt == 'lammps' and MD_avail:
                    ff.ff_assign(cell)
                    cell, _ = md.quick_rw(cell, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu)
                elif opt == 'rdkit':
                    AllChem.MMFFOptimizeMolecule(cell, maxIters=50, confId=0)

                cell_coord = np.array(cell.GetConformer(0).GetPositions())
                cell_wcoord = calc.wrap(cell_coord, xhi, xlo, yhi, ylo, zhi, zlo)
                dist_matrix1 = calc.distance_matrix(cell_wcoord[:-mol_n], cell_wcoord[-mol_n:])

                if cell_n > 0:
                    dist_matrix2 = calc.distance_matrix(cell_wcoord[:cell_n], cell_wcoord[-mol_n:])
                    dist_min2 = dist_matrix2.min()
                else:
                    dist_min2 = threshold + 1

                if dist_matrix1.min() > dist_min and dist_min2 > threshold:
                    break
                elif r < retry-1:
                    cell = utils.deepcopy_mol(cell_copy)
                    utils.radon_print('Retry random walk step %03d' % (i+1))
                else:
                    utils.radon_print('Reached maximum number of retrying random walk step.', level=2)

        if terminate is not None:
            cell = terminate_rw(cell, terminate, terminate2)

    return cell


# DEPRECATION
def copolymerize_cell(mols, n, m, terminate=None, terminate2=None, cell=None, density=0.1,
                    ff=None, opt='rdkit', retry=50, dist_min=0.7, threshold=2.0, work_dir=None, omp=1, mpi=1, gpu=0):
    """
    poly.copolymerize_cell

    *** DEPRECATION ***
    Unit cell generator of a copolymer by random walk

    Args:
        mols: Array of RDkit Mol object
        n: Polymerization degree (int)
        m: Number of polymer chains (int)

    Optional args:
        terminate: terminated substitute at head (and tail) (RDkit Mol object)
        terminate2: terminated substitute at tail (RDkit Mol object)
        cell: Initial structure of unit cell (RDkit Mol object)
        density: (float, g/cm3)
        ff: Force field object (requiring when use_lammps is True)
        opt:
            True: Using short time MD of lammps
            False: Dihedral angle around connecting bond is rotated randomly
        retry: Number of retry when generating unsuitable structure (int)
        dist_min: Threshold of intra-molecular atom-atom distance(float, angstrom)
        threshold: Threshold of inter-molecular atom-atom distance (float, angstrom)

    Returns:
        Rdkit Mol object
    """
    if terminate2 is None: terminate2 = terminate

    if cell is None:
        cell = Chem.Mol()
    else:
        cell = utils.deepcopy_mol(cell)
 
    if density is None and hasattr(cell, 'cell'):
        xhi = cell.cell.xhi
        xlo = cell.cell.xlo
        yhi = cell.cell.yhi
        ylo = cell.cell.ylo
        zhi = cell.cell.zhi
        zlo = cell.cell.zlo
    else:
        nl = [n*m] * len(mols)
        if terminate is None:
            xhi, xlo, yhi, ylo, zhi, zlo = calc_cell_length([*mols, cell], [*nl, 1], density=density)
        else:
            xhi, xlo, yhi, ylo, zhi, zlo = calc_cell_length([*mols, terminate, terminate2, cell], [*nl, m, m, 1], density=density)
        setattr(cell, 'cell', utils.Cell(xhi, xlo, yhi, ylo, zhi, zlo))

    for k in range(m):
        cell_n = cell.GetNumAtoms()
        cell = replicate_cell(mols[0], 1, cell=cell, density=None, retry=retry, threshold=threshold)

        for i in tqdm(range(n), desc='[Unit cell generation %i/%i]' % (k+1, m), disable=const.tqdm_disable):
            for j, mol in enumerate(mols):
                if i == 0 and j==0: continue
                mol_n = mol.GetNumAtoms()
                cell_copy = utils.deepcopy_mol(cell)

                for r in range(retry):
                    cell = connect_mols(cell, mol, random_rot=True)

                    if opt == 'lammps' and MD_avail:
                        ff.ff_assign(cell)
                        cell, _ = md.quick_rw(cell, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu)
                    elif opt == 'rdkit':
                        AllChem.MMFFOptimizeMolecule(cell, maxIters=50, confId=0)

                    cell_coord = np.array(cell.GetConformer(0).GetPositions())
                    cell_wcoord = calc.wrap(cell_coord, xhi, xlo, yhi, ylo, zhi, zlo)
                    dist_matrix1 = calc.distance_matrix(cell_wcoord[:-mol_n], cell_wcoord[-mol_n:])

                    if cell_n > 0:
                        dist_matrix2 = calc.distance_matrix(cell_wcoord[:cell_n], cell_wcoord[-mol_n:])
                        dist_min2 = dist_matrix2.min()
                    else:
                        dist_min2 = threshold + 1

                    if dist_matrix1.min() > dist_min and dist_min2 > threshold:
                        break
                    elif r < retry-1:
                        cell = utils.deepcopy_mol(cell_copy)
                        utils.radon_print('Retry random walk step %03d' % (i+1))
                    else:
                        utils.radon_print('Reached maximum number of retrying random walk step.', level=2)

        if terminate is not None:
            cell = terminate_rw(cell, terminate, terminate2)

    return cell


# DEPRECATION
def random_copolymerize_cell(mols, n, ratio, m, terminate=None, terminate2=None, cell=None, density=0.1,
                    ff=None, opt='rdkit', retry=50, dist_min=0.7, threshold=2.0, work_dir=None, omp=1, mpi=1, gpu=0):
    """
    poly.random_copolymerize_cell

    *** DEPRECATION ***
    Unit cell generator of a random copolymer by random walk

    Args:
        mols: Array of RDkit Mol object
        n: Polymerization degree (int)
        ratio: Array of monomer ratio (float, sum=1.0)
        m: Number of polymer chains (int)

    Optional args:
        terminate: terminated substitute at head (and tail) (RDkit Mol object)
        terminate2: terminated substitute at tail (RDkit Mol object)
        cell: Initial structure of unit cell (RDkit Mol object)
        density: (float, g/cm3)
        ff: Force field object (requiring when use_lammps is True)
        opt:
            True: Using short time MD of lammps
            False: Dihedral angle around connecting bond is rotated randomly
        retry: Number of retry when generating unsuitable structure (int)
        dist_min: Threshold of intra-molecular atom-atom distance(float, angstrom)
        threshold: Threshold of inter-molecular atom-atom distance (float, angstrom)

    Returns:
        Rdkit Mol object
    """

    if len(mols) != len(ratio):
        utils.radon_print('Inconsistency length of mols and ratio', level=3)
        return None

    ratio = np.array(ratio) / np.sum(ratio)

    mol_index = np.random.choice(a=list(range(len(mols))), size=(m, n), p=ratio)

    if terminate2 is None: terminate2 = terminate

    if cell is None:
        cell = Chem.Mol()
    else:
        cell = utils.deepcopy_mol(cell)
 
    if density is None and hasattr(cell, 'cell'):
        xhi = cell.cell.xhi
        xlo = cell.cell.xlo
        yhi = cell.cell.yhi
        ylo = cell.cell.ylo
        zhi = cell.cell.zhi
        zlo = cell.cell.zlo
    else:
        nl = [n*m*r for r in ratio]
        if terminate is None:
            xhi, xlo, yhi, ylo, zhi, zlo = calc_cell_length([*mols, cell], [*nl, 1], density=density)
        else:
            xhi, xlo, yhi, ylo, zhi, zlo = calc_cell_length([*mols, terminate, terminate2, cell], [*nl, m, m, 1], density=density)
        setattr(cell, 'cell', utils.Cell(xhi, xlo, yhi, ylo, zhi, zlo))

    for j in range(m):
        cell_n = cell.GetNumAtoms()
        cell = replicate_cell(mols[mol_index[j, 0]], 1, cell=cell, density=None, retry=retry, threshold=threshold)

        for i in tqdm(range(n-1), desc='[Unit cell generation %i/%i]' % (j+1, m), disable=const.tqdm_disable):
            cell_copy = utils.deepcopy_mol(cell)
            mol = mols[mol_index[j, i+1]]
            mol_n = mol.GetNumAtoms()

            for r in range(retry):
                cell = connect_mols(cell, mol, random_rot=True)

                if opt == 'lammps' and MD_avail:
                    ff.ff_assign(cell)
                    cell, _ = md.quick_rw(cell, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu)
                elif opt == 'rdkit':
                    AllChem.MMFFOptimizeMolecule(cell, maxIters=50, confId=0)

                cell_coord = np.array(cell.GetConformer(0).GetPositions())
                cell_wcoord = calc.wrap(cell_coord, xhi, xlo, yhi, ylo, zhi, zlo)
                dist_matrix1 = calc.distance_matrix(cell_wcoord[:-mol_n], cell_wcoord[-mol_n:])

                if cell_n > 0:
                    dist_matrix2 = calc.distance_matrix(cell_wcoord[:cell_n], cell_wcoord[-mol_n:])
                    dist_min2 = dist_matrix2.min()
                else:
                    dist_min2 = threshold + 1

                if dist_matrix1.min() > dist_min and dist_min2 > threshold:
                    break
                elif r < retry-1:
                    cell = utils.deepcopy_mol(cell_copy)
                    utils.radon_print('Retry random walk step %03d' % (i+1))
                else:
                    utils.radon_print('Reached maximum number of retrying random walk step.', level=2)

        if terminate is not None:
            cell = terminate_rw(cell, terminate, terminate2)

    return cell


def crystal_cell(mol, density=0.7, margin=(0.75, 1.0, 1.0), alpha=1, theta=1, d=1,
                dist_min=2.5, dist_max=3.0, step=0.1, max_iter=100, wrap=True, confId=0,
                chain1rot=False, FixSelfRot=True):
    """
    poly.crystal_cell

    Unit cell generator of crystalline polymer
    J. Phys. Chem. Lett. 2020, 11, 15, 5823–5829

    Args:
        mol: RDkit Mol object

    Optional args:
        density: (float, g/cm3)
        margin: Tuple of Margin of cell along x,y,z axis (float, angstrom)
        alpha: Rotation parameter of chain 2 around chain 1 in the confomation generator (int)
        theta: Rotation parameter of chain 2 around itself in the confomation generator (int)
        d: Translation parameter of chain 2 in the confomation generator (int)
        dist_min: Minimum threshold of inter-molecular atom-atom distance (float, angstrom)
        dist_max: Maximum threshold of inter-molecular atom-atom distance (float, angstrom)
        step: Step size of an iteration in adjusting interchain distance (float, angstrom)
        max_iter: Number of retry when adjusting interchain distance (int)
        wrap: Output coordinates are wrapped in cell against only x-axis (boolean)
        confId: Target conformer ID

    Debug option:
        chain1rot: Rotation parameter alpha works as rotation parameter of chain 1 around itself in the confomation generator (boolean)
        FixSelfRot: (boolean)

    Returns:
        Rdkit Mol object
    """

    # Alignment molecules
    mol_c = utils.deepcopy_mol(mol)
    Chem.rdMolTransforms.CanonicalizeConformer(mol_c.GetConformer(confId), ignoreHs=False)
    mol1_coord = np.array(mol_c.GetConformer(confId).GetPositions())

    # Rotation mol2 to align link vectors and x-axis
    set_linker_flag(mol_c)
    center = mol1_coord[mol_c.GetIntProp('head_idx')]
    link_vec = mol1_coord[mol_c.GetIntProp('tail_idx')] - mol1_coord[mol_c.GetIntProp('head_idx')]
    angle = calc.angle_vec(link_vec, np.array([1.0, 0.0, 0.0]), rad=True)
    if angle == 0 or angle == np.pi:
        pass
    else:
        vcross = np.cross(link_vec, np.array([1.0, 0.0, 0.0]))
        mol1_coord = calc.rotate_rod(mol1_coord, vcross, angle, center=center)

    # Adjusting interchain distance
    mol2_coord = mol1_coord + np.array([0.0, 0.0, dist_min])
    for r in range(max_iter):
        dist_matrix = calc.distance_matrix(mol1_coord, mol2_coord)
        if dist_matrix.min() >= dist_min and dist_matrix.min() <= dist_max:
            break
        elif dist_matrix.min() > dist_max:
            mol2_coord -= np.array([0.0, 0.0, step])
        elif dist_matrix.min() < dist_min:
            mol2_coord += np.array([0.0, 0.0, step])

    # Copy a polymer chain
    cell = combine_mols(mol_c, mol_c)
    cell_n = cell.GetNumAtoms()

    # Set atomic coordinate of chain2 in initial structure
    new_coord = np.vstack((mol1_coord, mol2_coord))
    for i in range(cell_n):
        cell.GetConformer(0).SetAtomPosition(i, Geom.Point3D(new_coord[i, 0], new_coord[i, 1], new_coord[i, 2]))

    # Set cell information
    xhi, xlo, yhi, ylo, zhi, zlo = calc_cell_length([cell], [1], density=density, margin=margin, fit='x')
    setattr(cell, 'cell', utils.Cell(xhi, xlo, yhi, ylo, zhi, zlo))

    mol1_center = np.mean(mol1_coord, axis=0)
    mol2_center = np.mean(mol2_coord, axis=0)

    # Generate conformers for grobal minimum search
    utils.radon_print('Start crystal cell generation.', level=1)
    if not const.tqdm_disable:
        bar = tqdm(total=alpha*theta*d)
        bar.set_description('Generate confomers')
    for a in range(alpha):
        a_rot = 2*np.pi/alpha * a

        for t in range(theta):
            t_rot = 2*np.pi/theta * t

            for r in range(d):
                if a==0 and t==0 and r==0:
                    _, _, yhi, ylo, zhi, zlo = calc_cell_length([cell], [1], margin=margin, fit='xyz')
                    set_cell_param_conf(cell, 0, xhi, xlo, yhi, ylo, zhi, zlo)
                    if not const.tqdm_disable: bar.update(1)
                    continue

                r_tr = (xhi-xlo)/d * r

                # Translate chain2 along x-axis
                mol2_tr = mol2_coord + np.array([r_tr, 0.0, 0.0])

                # Rotate chain2 around itself
                mol2_rot = calc.rotate_rod(mol2_tr, np.array([1.0, 0.0, 0.0]), t_rot, center=mol2_center)

                if chain1rot:
                    # Rotate chain1 around itself
                    mol1_new = calc.rotate_rod(mol1_coord, np.array([1.0, 0.0, 0.0]), -a_rot, center=mol1_center)
                    mol2_new = mol2_rot
                    if FixSelfRot: mol2_new = calc.rotate_rod(mol2_new, np.array([1.0, 0.0, 0.0]), -a_rot, center=mol2_center)
                else:
                    # Rotate chain2 around chain1
                    mol1_new = mol1_coord
                    mol2_new = calc.rotate_rod(mol2_rot, np.array([1.0, 0.0, 0.0]), -a_rot, center=mol2_center)
                    if FixSelfRot: mol2_new = calc.rotate_rod(mol2_new, np.array([1.0, 0.0, 0.0]), a_rot, center=mol1_center)

                # Adjusting interchain distance
                for i in range(max_iter):
                    if wrap: mol2_new_w = calc.wrap(mol2_new, xhi, xlo, None, None, None, None)
                    else: mol2_new_w = mol2_new
                    dist_matrix = calc.distance_matrix(mol1_new, mol2_new_w)

                    if dist_matrix.min() >= dist_min and dist_matrix.min() <= dist_max:
                        break

                    mol2_new_center = np.mean(mol2_new, axis=0)
                    vec = np.array([0.0, mol2_new_center[1]-mol1_center[1], mol2_new_center[2]-mol1_center[2]])
                    vec = vec/np.linalg.norm(vec)

                    if dist_matrix.min() > dist_max:
                        mol2_new -= vec*step
                    elif dist_matrix.min() < dist_min:
                        mol2_new += vec*step

                # Add new conformer
                new_coord = np.vstack((mol1_new, mol2_new))
                if wrap: new_coord = calc.wrap(new_coord, xhi, xlo, None, None, None, None)
                conf = Chem.rdchem.Conformer(cell_n)
                conf.Set3D(True)
                for i in range(cell_n):
                    conf.SetAtomPosition(i, Geom.Point3D(new_coord[i, 0], new_coord[i, 1], new_coord[i, 2]))
                conf_id = cell.AddConformer(conf, assignId=True)

                # Set cell parameters
                _, _, yhi, ylo, zhi, zlo = calc_cell_length([cell], [1], confId=conf_id, margin=margin, fit='xyz')
                set_cell_param_conf(cell, conf_id, xhi, xlo, yhi, ylo, zhi, zlo)

                if not const.tqdm_disable: bar.update(1)
    if not const.tqdm_disable: bar.close()

    return cell


def single_chain_cell(mol, density=0.8, margin=0.75, confId=0):
    """
    poly.single_chain_cell

    Unit cell generator of single chain polymer (for 1D PBC calculation)

    Args:
        mol: RDkit Mol object

    Optional args:
        density: (float, g/cm3)
        margin: Margin of cell along x-axis (float, angstrom)
        confId: Target conformer ID

    Returns:
        Rdkit Mol object
    """

    cell = utils.deepcopy_mol(mol)

    # Alignment molecules
    Chem.rdMolTransforms.CanonicalizeConformer(cell.GetConformer(confId), ignoreHs=False)
    cell_coord = np.array(cell.GetConformer(confId).GetPositions())

    # Rotation mol to align link vectors and x-axis
    set_linker_flag(cell)
    center = cell_coord[cell.GetIntProp('head_idx')]
    link_vec = cell_coord[cell.GetIntProp('tail_idx')] - cell_coord[cell.GetIntProp('head_idx')]
    angle = calc.angle_vec(link_vec, np.array([1.0, 0.0, 0.0]), rad=True)
    if angle == 0 or angle == np.pi:
        pass
    else:
        vcross = np.cross(link_vec, np.array([1.0, 0.0, 0.0]))
        cell_coord = calc.rotate_rod(cell_coord, vcross, angle, center=center)

    # Set atomic coordinate
    for i in range(cell.GetNumAtoms()):
        cell.GetConformer(confId).SetAtomPosition(i, Geom.Point3D(cell_coord[i, 0], cell_coord[i, 1], cell_coord[i, 2]))

    # Set cell information
    xhi, xlo, yhi, ylo, zhi, zlo = calc_cell_length([cell], [1], density=density, margin=(margin, 1.5, 1.5), fit='x')
    setattr(cell, 'cell', utils.Cell(xhi, xlo, yhi, ylo, zhi, zlo))

    return cell


def super_cell(cell, x=1, y=1, z=1, confId=0):
    """
    poly.super_cell

    Super cell generator of RDkit Mol object

    Args:
        cell: RDkit Mol object

    Optional args:
        x, y, z: Number of replicating super cell (int)
        confId: Target conformer ID

    Returns:
        Rdkit Mol object
    """

    if cell.GetConformer(confId).HasProp('cell_dx'):
        lx = cell.GetConformer(confId).GetDoubleProp('cell_dx')
        ly = cell.GetConformer(confId).GetDoubleProp('cell_dy')
        lz = cell.GetConformer(confId).GetDoubleProp('cell_dz')
    else:
        lx = cell.cell.dx
        ly = cell.cell.dy
        lz = cell.cell.dz

    xcell = utils.deepcopy_mol(cell)
    cell_n = cell.GetNumAtoms()
    cell_coord = np.array(cell.GetConformer(confId).GetPositions())

    xcell.RemoveAllConformers()
    conf = Chem.rdchem.Conformer(cell_n)
    conf.Set3D(True)
    for i in range(cell_n):
        conf.SetAtomPosition(i, Geom.Point3D(cell_coord[i, 0], cell_coord[i, 1], cell_coord[i, 2]))
    xcell.AddConformer(conf, assignId=True)

    for ix in range(x-1):
        xcell_n = xcell.GetNumAtoms()
        xcell = combine_mols(xcell, xcell)
        new_coord = cell_coord + np.array([lx*(ix+1), 0.0, 0.0])
        for i in range(cell_n):
            xcell.GetConformer(0).SetAtomPosition(
                xcell_n+i,
                Geom.Point3D(new_coord[i, 0], new_coord[i, 1], new_coord[i, 2])
            )

    ycell = utils.deepcopy_mol(xcell)
    xcell_n = xcell.GetNumAtoms()
    xcell_coord = np.array(xcell.GetConformer(0).GetPositions())
    for iy in range(y-1):
        ycell_n = ycell.GetNumAtoms()
        ycell = combine_mols(ycell, xcell)
        new_coord = xcell_coord + np.array([0.0, ly*(iy+1), 0.0])
        for i in range(xcell_n):
            ycell.GetConformer(0).SetAtomPosition(
                ycell_n+i,
                Geom.Point3D(new_coord[i, 0], new_coord[i, 1], new_coord[i, 2])
            )

    zcell = utils.deepcopy_mol(ycell)
    ycell_n = ycell.GetNumAtoms()
    ycell_coord = np.array(ycell.GetConformer(0).GetPositions())
    for iz in range(z-1):
        zcell_n = zcell.GetNumAtoms()
        zcell = combine_mols(zcell, ycell)
        new_coord = ycell_coord + np.array([0.0, 0.0, lz*(iz+1)])
        for i in range(ycell_n):
            zcell.GetConformer(0).SetAtomPosition(
                zcell_n+i,
                Geom.Point3D(new_coord[i, 0], new_coord[i, 1], new_coord[i, 2])
            )

    zcell.cell = utils.Cell(lx*x, zcell.cell.xlo, ly*y, zcell.cell.ylo, lz*z, zcell.cell.zlo)

    Chem.SanitizeMol(zcell)

    return zcell


def calc_cell_length(mols, n, density=1.0, confId=0, ignoreLinkers=True, fit=None, margin=(1.2, 1.2, 1.2)):
    """
    poly.calc_cell_length

    Calculate cell length

    Args:
        mols: Array of RDkit Mol object
        n: Array of number of molecules into cell (float)
            i.e. (polymerization degree) * (monomer ratio) * (number of polymer chains)
        density: (float, g/cm3)

    Optional args:
        confId: Target conformer ID (int)
        ignoreLinkers: Ignoring linker atoms in the mass calculation (boolean)
        fit: Length of specific axis is fitted for molecular coordinate (str, (x, y, z, xy, xz, yz, xyz, or max_cubic))
             If fit is xyz, density is ignored.
        margin: Margin of cell for fitting axis (float, angstrom)

    Returns:
        xhi, xlo: Higher and lower edges of x axis (float, angstrom)
        yhi, ylo: Higher and lower edges of y axis (float, angstrom)
        zhi, zlo: Higher and lower edges of z axis (float, angstrom)
    """

    # Calculate mass
    mass = 0
    for i, mol in enumerate(mols):
        if ignoreLinkers: set_linker_flag(mol)
        for atom in mol.GetAtoms():
            if ignoreLinkers and atom.GetBoolProp('linker'):
                pass
            else:
                mass += atom.GetMass()*n[i]

    # Check molecular length
    x_min = x_max = y_min = y_max = z_min = z_max = 0
    for mol in mols:
        if mol.GetNumConformers() == 0: continue
        mol_coord = np.array(mol.GetConformer(confId).GetPositions())

        if ignoreLinkers:
            del_coord = []
            for i, atom in enumerate(mol.GetAtoms()):
                if not atom.GetBoolProp('linker'):
                    del_coord.append(mol_coord[i])
            mol_coord = np.array(del_coord)

        mol_coord = calc.fix_trans(mol_coord)
        x_min = mol_coord.min(axis=0)[0] - margin[0] if mol_coord.min(axis=0)[0] - margin[0] < x_min else x_min
        x_max = mol_coord.max(axis=0)[0] + margin[0] if mol_coord.max(axis=0)[0] + margin[0] > x_max else x_max
        y_min = mol_coord.min(axis=0)[1] - margin[1] if mol_coord.min(axis=0)[1] - margin[1] < y_min else y_min
        y_max = mol_coord.max(axis=0)[1] + margin[1] if mol_coord.max(axis=0)[1] + margin[1] > y_max else y_max
        z_min = mol_coord.min(axis=0)[2] - margin[2] if mol_coord.min(axis=0)[2] - margin[2] < z_min else z_min
        z_max = mol_coord.max(axis=0)[2] + margin[2] if mol_coord.max(axis=0)[2] + margin[2] > z_max else z_max

    x_length = x_max - x_min
    y_length = y_max - y_min
    z_length = z_max - z_min

    # Determining cell length (angstrom)
    length = np.cbrt( (mass / const.NA) / (density / const.cm2ang**3)) / 2

    if fit is None:
        xhi = yhi = zhi = length
        xlo = ylo = zlo = -length
    elif fit == 'auto':
        if x_length > length*2:
            xhi = x_length/2
            xlo = -xhi
            yhi = zhi = np.sqrt( ((mass / const.NA) / (density / const.cm2ang**3)) / x_length ) / 2
            ylo = zlo = -yhi
        elif y_length > length*2:
            yhi = y_length/2
            ylo = -yhi
            xhi = zhi = np.sqrt( ((mass / const.NA) / (density / const.cm2ang**3)) / y_length ) / 2
            xlo = zlo = -xhi
        elif z_length > length*2:
            zhi = z_length/2
            zlo = -zhi
            xhi = yhi = np.sqrt( ((mass / const.NA) / (density / const.cm2ang**3)) / z_length ) / 2
            xlo = ylo = -xhi
        else:
            xhi = yhi = zhi = length
            xlo = ylo = zlo = -length
    elif fit == 'x':
        xhi = x_length/2
        xlo = -xhi
        yhi = zhi = np.sqrt( ((mass / const.NA) / (density / const.cm2ang**3)) / x_length ) / 2
        ylo = zlo = -yhi
    elif fit == 'y':
        yhi = y_length/2
        ylo = -yhi
        xhi = zhi = np.sqrt( ((mass / const.NA) / (density / const.cm2ang**3)) / y_length ) / 2
        xlo = zlo = -xhi
    elif fit == 'z':
        zhi = z_length/2
        zlo = -zhi
        xhi = yhi = np.sqrt( ((mass / const.NA) / (density / const.cm2ang**3)) / z_length ) / 2
        xlo = ylo = -xhi
    elif fit == 'xy':
        xhi = x_length/2
        xlo = -xhi
        yhi = y_length/2
        ylo = -yhi
        zhi = (((mass / const.NA) / (density / const.cm2ang**3)) / x_length / y_length) / 2
        zlo = -zhi
    elif fit == 'xz':
        xhi = x_length/2
        xlo = -xhi
        zhi = z_length/2
        zlo = -zhi
        yhi = (((mass / const.NA) / (density / const.cm2ang**3)) / x_length / z_length) / 2
        ylo = -yhi
    elif fit == 'yz':
        yhi = y_length/2
        ylo = -yhi
        zhi = z_length/2
        zlo = -zhi
        xhi = (((mass / const.NA) / (density / const.cm2ang**3)) / y_length / z_length) / 2
        xlo = -xhi
    elif fit == 'xyz':
        xhi = x_length/2
        xlo = -xhi
        yhi = y_length/2
        ylo = -yhi
        zhi = z_length/2
        zlo = -zhi
    elif fit == 'max_cubic':
        cmax = max([x_max, y_max, z_max])
        cmin = min([x_min, y_min, z_min])
        cell_l = cmax if cmax > abs(cmin) else abs(cmin)
        xhi = yhi = zhi = cell_l
        xlo = ylo = zlo = -cell_l

    return xhi, xlo, yhi, ylo, zhi, zlo


def set_cell_param_conf(mol, confId, xhi, xlo, yhi, ylo, zhi, zlo):
    """
    poly.set_cell_param_conf

    Set cell parameters for RDKit Conformer object

    Args:
        mol: RDkit Mol object
        confId: Target conformer ID (int)
        xhi, xlo: Higher and lower edges of x axis (float, angstrom)
        yhi, ylo: Higher and lower edges of y axis (float, angstrom)
        zhi, zlo: Higher and lower edges of z axis (float, angstrom)

    Returns:
        Rdkit Mol object
    """

    conf = mol.GetConformer(confId)
    conf.SetDoubleProp('cell_xhi', xhi)
    conf.SetDoubleProp('cell_xlo', xlo)
    conf.SetDoubleProp('cell_yhi', yhi)
    conf.SetDoubleProp('cell_ylo', ylo)
    conf.SetDoubleProp('cell_zhi', zhi)
    conf.SetDoubleProp('cell_zlo', zlo)
    conf.SetDoubleProp('cell_dx', xhi-xlo)
    conf.SetDoubleProp('cell_dy', yhi-ylo)
    conf.SetDoubleProp('cell_dz', zhi-zlo)

    return mol


##########################################################
# Utility functions for check of 3D structure
##########################################################
def check_3d_proximity(coord1, coord2=None, dist_min=1.5, wrap=None, ignore_rad=3, dmat=None):
    """
    poly.check_3d_proximity

    Checking proximity between atoms

    Args:
        mol1: RDKit Mol object

    Optional args:
        mol2: RDKit Mol object of a flagment unit
        dist_min: Threshold of the minimum atom-atom distance (float, angstrom)
        confId: Target conformer ID of the polymer (int)
        wrap: Input cell object (mol.cell) if use wrapped coordinates

    Returns:
        boolean
            True: Without proximity atoms
            False: Found proximity atoms
    """
    if wrap is not None:
        coord1 = calc.wrap(coord1, wrap.xhi, wrap.xlo,
                        wrap.yhi, wrap.ylo, wrap.zhi, wrap.zlo)
        if coord2 is not None:
            coord2 = calc.wrap(coord2, wrap.xhi, wrap.xlo,
                            wrap.yhi, wrap.ylo, wrap.zhi, wrap.zlo)

    if coord2 is not None:
        dist_matrix = calc.distance_matrix(coord1, coord2)
    else:
        dist_matrix = calc.distance_matrix(coord1)
        np.fill_diagonal(dist_matrix, np.nan)

    if dmat is not None:
        imat = np.where(dmat <= ignore_rad, np.nan, 1)
        dist_matrix = dist_matrix * imat

    if np.nanmin(dist_matrix) > dist_min:
        return True
    else:
        return False


def check_3d_bond_length(mol, confId=0, bond_s=2.7, bond_a=1.9, bond_d=1.8, bond_t=1.4):
    """
    poly.check_3d_bond_length

    Args:
        mol: RDkit Mol object

    Optional args:
        bond_s: Threshold of the maximum single bond length (float, angstrom)
        bond_a: Threshold of the maximum aromatic bond length (float, angstrom)
        bond_d: Threshold of the maximum double bond length (float, angstrom)
        bond_t: Threshold of the maximum triple bond length (float, angstrom)

    Returns:
        boolean
    """
    coord = np.array(mol.GetConformer(confId).GetPositions())
    dist_matrix = calc.distance_matrix(coord)
    check = True

    # Cheking bond length
    for b in mol.GetBonds():
        bond_l = dist_matrix[b.GetBeginAtom().GetIdx(), b.GetEndAtom().GetIdx()]
        if b.GetBondTypeAsDouble() == 1.0 and bond_l > bond_s:
            check = False
            break
        elif b.GetBondTypeAsDouble() == 1.5 and bond_l > bond_a:
            check = False
            break
        elif b.GetBondTypeAsDouble() == 2.0 and bond_l > bond_d:
            check = False
            break
        elif b.GetBondTypeAsDouble() == 3.0 and bond_l > bond_t:
            check = False
            break

    return check


def check_3d_bond_ring_intersection(poly, mon=None, poly_coord=None, mon_coord=None, tri_coord=None, bond_coord=None, confId=0, wrap=None, mp=0):
    """
    poly.check_3d_bond_ring_intersection

    Checking bond-ring intersection using the Möller–Trumbore ray-triangle intersection algorithm

    Args:
        poly: RDKit Mol object of a polymer

    Optional args:
        mon: RDKit Mol object of a repeating unit
        tri_coord: Constructed atomic coordinates of triangles in the polymer (ndarray)
        bond_coord: Constructed atomic coordinates of bonds in the polymer (ndarray)
        confId: Target conformer ID of the polymer (int)
        wrap: Use wrapped coordinates if the poly having cell information
        mp: Parallel number of multiprocessing

    Return:
        check: (boolean)
            True: Without a penetration structure
            False: Found a penetration structure
        tri_coord: Constructed atomic coordinates of triangles in the polymer (ndarray)
        bond_coord: Constructed atomic coordinates of bonds in the polymer (ndarray)
    """
    check = True
    poly_n = poly.GetNumAtoms()
    mon_idx = poly_n
    poly_bond_n = poly.GetNumBonds()
    ring = Chem.GetSymmSSSR(poly)
    if poly_coord is None:
        poly_coord = np.array(poly.GetConformer(confId).GetPositions())

    if len(ring) == 0:
        return True, None, None

    if type(mon) is Chem.Mol:
        mon_n = mon.GetNumAtoms()
        if mon_coord is None:
            mon_idx = int(poly_n - mon_n)

    # Construction of bond and ring surface coordinates in the growing chain part
    if tri_coord is None:
        if wrap is not None:
            poly_coord = calc.wrap(poly_coord, wrap.xhi, wrap.xlo,
                            wrap.yhi, wrap.ylo, wrap.zhi, wrap.zlo)
        p_ring_coord = [[poly_coord[j] for j in list(ring[i])] for i in range(len(ring)) if list(ring[i])[0] < mon_idx]
        p_tri_coord = np.array([np.array([r[0], r[x+1], r[x+2]]) for r in p_ring_coord for x in range(len(r)-2)])
    else:
        p_tri_coord = tri_coord

    if bond_coord is None:
        p_bond_coord = np.array([
                            np.array([
                                poly_coord[b.GetBeginAtomIdx()],
                                poly_coord[b.GetEndAtomIdx()]
                            ]) for b in poly.GetBonds() if b.GetBeginAtomIdx() < mon_idx and b.GetEndAtomIdx() < mon_idx
                        ])
    else:
        p_bond_coord = bond_coord


    if type(mon) is Chem.Mol:
        # Vector construction of bond and ring surface coordinates in the additional monomer part
        if mon_coord is None:
            m_ring_coord = []
            for i in range(len(ring), 0, -1):
                tmp_list = []
                if list(ring[i-1])[0] < mon_idx:
                    break
                for j in list(ring[i-1]):
                    tmp_list.append(poly_coord[j])
                m_ring_coord.append(tmp_list)
            m_tri_coord = np.array([np.array([r[0], r[x+1], r[x+2]]) for r in m_ring_coord for x in range(len(r)-2)])

            m_bond_coord = np.array([
                                np.array([
                                    poly_coord[b.GetBeginAtomIdx()],
                                    poly_coord[b.GetEndAtomIdx()]
                                ]) for b in poly.GetBonds() if b.GetBeginAtomIdx() >= mon_idx or b.GetEndAtomIdx() >= mon_idx
                            ])
        else:
            if wrap is not None:
                mon_coord = calc.wrap(mon_coord, wrap.xhi, wrap.xlo,
                                wrap.yhi, wrap.ylo, wrap.zhi, wrap.zlo)

                m_ring = Chem.GetSymmSSSR(mon)
                m_ring_coord = [[mon_coord[j] for j in list(m_ring[i])] for i in range(len(m_ring))]
                m_tri_coord = np.array([np.array([r[0], r[x+1], r[x+2]]) for r in m_ring_coord for x in range(len(r)-2)])

                m_bond_coord = np.array([
                                    np.array([
                                        mon_coord[b.GetBeginAtomIdx()],
                                        mon_coord[b.GetEndAtomIdx()]
                                    ]) for b in mon.GetBonds()
                                ])

        if  len(p_tri_coord) > 0 and len(m_tri_coord) > 0:
            r_tri_coord = np.vstack([p_tri_coord, m_tri_coord])
        elif len(p_tri_coord) > 0:
            r_tri_coord = p_tri_coord
        else:
            r_tri_coord = m_tri_coord
        r_bond_coord = np.vstack([p_bond_coord, m_bond_coord])


        # Ray-triangle intersection to check bond-ring intersection
        # Step 1: rings in poly vs. bonds in monomer
        if mp == 0:
            for bond, tri in itertools.product(m_bond_coord, p_tri_coord):
                if MollerTrumbore(bond, tri):
                    check = False
                    break

            # Step 2: rings in monomer vs. bonds in poly
            if check:
                for bond, tri in itertools.product(p_bond_coord, m_tri_coord):
                    if MollerTrumbore(bond, tri):
                        check = False
                        break
        else:
            args = list(itertools.product(m_bond_coord, p_tri_coord)) + list(itertools.product(p_bond_coord, m_tri_coord))
            args = list(np.array_split(np.array(args, dtype=object), mp))
            with confu.ProcessPoolExecutor(max_workers=mp, mp_context=MP.get_context('spawn')) as executor:
                results = executor.map(_MollerTrumbore, args)
            check = not np.any(np.array(list(results)))


    else:
        r_tri_coord = p_tri_coord
        r_bond_coord = p_bond_coord

        # Ray-triangle intersection to judge penetration into a ring by a bond
        if mp == 0:
            for bond, tri in itertools.product(p_bond_coord, p_tri_coord):
                if MollerTrumbore(bond, tri):
                    check = False
                    break
        else:
            args = list(itertools.product(p_bond_coord, p_tri_coord))
            args = list(np.array_split(np.array(args, dtype=object), mp))
            with confu.ProcessPoolExecutor(max_workers=mp, mp_context=MP.get_context('spawn')) as executor:
                results = executor.map(_MollerTrumbore, args)
            check = not np.any(np.array(list(results)))

    if not check:
        utils.radon_print('A bond-ring intersection was found.')

    return check, r_tri_coord, r_bond_coord


def MollerTrumbore(bond, tri):
    """
    poly.MollerTrumbore

    The Möller–Trumbore ray-triangle intersection algorithm

    Args:
        bond: Atomic coordinates of a bond (2*3 ndarray)
        tri: Atomic coordinates of a triangle (3*3 ndarray)

    Return:
        boolean
    """
    eps = 1e-4
    R   = bond[1, :] - bond[0, :]
    T   = bond[0, :] - tri[0, :]
    E1  = tri[1, :]  - tri[0, :]
    E2  = tri[2, :]  - tri[0, :]
    P   = np.cross(R, E2)
    S   = np.dot(P, E1)
    if -eps < S < eps:
        return False
    
    u   = np.dot(P, T)  / S
    if u < 0:
        return False

    Q   = np.cross(T, E1)
    v   = np.dot(Q, R)  / S
    t   = np.dot(Q, E2) / S

    if ((-eps < u < eps and -eps < v < eps) or
        (-eps < u < eps and 1-eps < v < 1+eps) or
        (1-eps < u < 1+eps and -eps < v < eps)
       ) and (-eps < t < eps or 1-eps < t < 1+eps):
        return False 
    elif v >= 0 and u+v <= 1 and 0 <= t <= 1:
        return True
    else:
        return False


def _MollerTrumbore(args):
    for arg in args:
        bond, tri = arg
        if MollerTrumbore(bond, tri):
            return True
    return False


def check_3d_structure_poly(poly, mon, poly_dmat=None, dist_min=1.0, ignore_rad=3, check_bond_length=False, tacticity=None):
    """
    poly.check_3d_structure_poly

    Checking proximity between atoms for polymer chain generators

    Args:
        poly: RDKit Mol object of a polymer
        mon: RDKit Mol object of an additional molecular
        poly_dmat: Topological distance matrix of a polymer

    Optional args:
        dist_min: Threshold of the minimum atom-atom distance (float, angstrom)
        ignore_rad: Radius of topological distance is ignored in distance check
        check_bond_length: Perform bond length check
        tacticity: Perform tacticity check

    Returns:
        boolean
            True: Without proximity atoms
            False: Found proximity atoms
    """
    n_mon = mon.GetNumAtoms()
    if poly_dmat is not None:
        poly_dmat = poly_dmat[:-n_mon, -n_mon:]
    coord = np.array(poly.GetConformer(0).GetPositions())
    p_coord = coord[:-n_mon]
    m_coord = coord[-n_mon:]
    check = check_3d_proximity(p_coord, coord2=m_coord, dist_min=dist_min, dmat=poly_dmat, ignore_rad=ignore_rad)

    if check and check_bond_length:
        check = check_3d_bond_length(poly)

    if check and tacticity:
        check = check_tacticity(poly, tacticity=tacticity)

    return check


def check_3d_structure_cell(cell, mol_coord, dist_min=2.0):
    """
    poly.check_3d_structure_cell

    Checking proximity between atoms for unit cell generators

    Args:
        cell: RDKit Mol object of a cell
        mol_coord: Atomic coordinates of an additional molecule

    Optional args:
        dist_min: Threshold of the minimum atom-atom distance (float, angstrom)

    Returns:
        boolean
            True: Without proximity atoms
            False: Found proximity atoms
    """
    check = True

    # Step 1. Self proximity check of an additional molecule (mol_coord) in the PBC cell
    wflag = np.where(
        (mol_coord[:, 0] > cell.cell.xhi) | (mol_coord[:, 0] < cell.cell.xlo) |
        (mol_coord[:, 1] > cell.cell.yhi) | (mol_coord[:, 1] < cell.cell.ylo) |
        (mol_coord[:, 2] > cell.cell.zhi) | (mol_coord[:, 2] < cell.cell.zlo),
        True, False
    )
    wcoord = mol_coord[wflag]
    uwcoord = mol_coord[np.logical_not(wflag)]
    if len(wcoord) > 0 and len(uwcoord) > 0:
        check = check_3d_proximity(uwcoord, coord2=wcoord, wrap=cell.cell, dist_min=dist_min)

    # Step 2. Proximity check between atoms in the cell (cell_coord) and in an addional molecule (mol_coord)
    if check:
        cell_coord = np.array(cell.GetConformer(0).GetPositions())
        check = check_3d_proximity(cell_coord, coord2=mol_coord, wrap=cell.cell, dist_min=dist_min)

    return check


##########################################################
# Utility functions for calculate polymerization degree 
##########################################################
def calc_n_from_num_atoms(mols, natom, ratio=[1.0], terminal1=None, terminal2=None):
    """
    poly.calc_n_from_num_atoms

    Calculate polymerization degree from target number of atoms

    Args:
        mols: List of RDkit Mol object
        natom: Target of number of atoms

    Optional args:
        terminal1, terminal2: Terminal substruct of RDkit Mol object
        ratio: List of monomer ratio

    Returns:
        int
    """

    if type(mols) is Chem.Mol:
        mols = [mols]
    elif type(mols) is not list:
        utils.radon_print('Input should be an RDKit Mol object or its List', level=3)
        return None

    if len(mols) != len(ratio):
        utils.radon_print('Inconsistency length of mols and ratio', level=3)
        return None

    ratio = np.array(ratio) / np.sum(ratio)

    mol_n = 0.0
    for i, mol in enumerate(mols):
        new_mol = remove_linker_atoms(mol)
        mol_n += new_mol.GetNumAtoms() * ratio[i]
    
    if terminal1 is not None:
        new_ter1 = remove_linker_atoms(terminal1)
        ter1_n = new_ter1.GetNumAtoms()
    else:
        ter1_n = 0

    if terminal2 is not None:
        new_ter2 = remove_linker_atoms(terminal2)
        ter2_n = new_ter2.GetNumAtoms()
    elif terminal1 is not None:
        ter2_n = ter1_n
    else:
        ter2_n = 0

    n = int((natom - ter1_n - ter2_n) / mol_n + 0.5)
    
    return n


def calc_n_from_mol_weight(mols, mw, ratio=[1.0], terminal1=None, terminal2=None):
    """
    poly.calc_n_from_mol_weight

    Calculate polymerization degree from target molecular weight

    Args:
        mols: List of RDkit Mol object
        mw: Target of molecular weight

    Optional args:
        terminal1, terminal2: Terminal substruct of RDkit Mol object
        ratio: List of monomer ratio

    Returns:
        int
    """
    if type(mols) is Chem.Mol:
        mols = [mols]
    elif type(mols) is not list:
        utils.radon_print('Input should be an RDKit Mol object or its list', level=3)
        return None

    if len(mols) != len(ratio):
        utils.radon_print('Inconsistency length of mols and ratio', level=3)
        return None

    ratio = np.array(ratio) / np.sum(ratio)

    mol_mw = 0.0
    for i, mol in enumerate(mols):
        mol_mw += calc.molecular_weight(mol) * ratio[i]
    
    ter1_mw = 0.0
    if terminal1 is not None:
        ter1_mw = calc.molecular_weight(terminal1)

    ter2_mw = 0.0
    if terminal2 is not None:
        ter2_mw = calc.molecular_weight(terminal2)
    elif terminal1 is not None:
        ter2_mw = ter1_mw

    n = int((mw - ter1_mw - ter2_mw) / mol_mw + 0.5)

    return n


##########################################################
# Utility function for RDKit Mol object of polymers
##########################################################
def set_linker_flag(mol, reverse=False, label=1):
    """
    poly.set_linker_flag

    Args:
        mol: RDkit Mol object
        reverse: Reversing head and tail (boolean)

    Returns:
        boolean
    """

    flag = False
    mol.SetIntProp('head_idx', -1)
    mol.SetIntProp('tail_idx', -1)
    mol.SetIntProp('head_ne_idx', -1)
    mol.SetIntProp('tail_ne_idx', -1)

    for atom in mol.GetAtoms():
        atom.SetBoolProp('linker', False)
        atom.SetBoolProp('head', False)
        atom.SetBoolProp('tail', False)
        atom.SetBoolProp('head_neighbor', False)
        atom.SetBoolProp('tail_neighbor', False)
        if (atom.GetSymbol() == "H" and atom.GetIsotope() == label+2) or atom.GetSymbol() == "*":
            atom.SetBoolProp('linker', True)
            if not flag:
                mol.SetIntProp('head_idx', atom.GetIdx())
                mol.SetIntProp('tail_idx', atom.GetIdx())
                flag = True
            else:
                if reverse:
                    mol.SetIntProp('head_idx', atom.GetIdx())
                else:
                    mol.SetIntProp('tail_idx', atom.GetIdx())

    if not flag: return False

    mol_head_idx = mol.GetIntProp('head_idx')
    mol.GetAtomWithIdx(mol_head_idx).SetBoolProp('head', True)

    mol_tail_idx = mol.GetIntProp('tail_idx')
    mol.GetAtomWithIdx(mol_tail_idx).SetBoolProp('tail', True)

    head_ne_idx = mol.GetAtomWithIdx(mol_head_idx).GetNeighbors()[0].GetIdx()
    mol.SetIntProp('head_ne_idx', head_ne_idx)
    mol.GetAtomWithIdx(head_ne_idx).SetBoolProp('head_neighbor', True)

    tail_ne_idx = mol.GetAtomWithIdx(mol_tail_idx).GetNeighbors()[0].GetIdx()
    mol.SetIntProp('tail_ne_idx', tail_ne_idx)
    mol.GetAtomWithIdx(tail_ne_idx).SetBoolProp('tail_neighbor', True)

    return True


def remove_linker_atoms(mol, label=1):
    """
    poly.remove_linker_atoms

    Args:
        mol: RDkit Mol object

    Returns:
        RDkit Mol object
    """

    new_mol = utils.deepcopy_mol(mol)

    def recursive_remove_linker_atoms(mol):
        for atom in mol.GetAtoms():
            if (atom.GetSymbol() == "H" and atom.GetIsotope() == label+2) or atom.GetSymbol() == "*":
                mol = utils.remove_atom(mol, atom.GetIdx())
                mol = recursive_remove_linker_atoms(mol)
                break
        return mol

    new_mol = recursive_remove_linker_atoms(new_mol)

    return new_mol


def set_terminal_idx(mol):

    count = 0
    for atom in mol.GetAtoms():
        resinfo = atom.GetPDBResidueInfo()
        if resinfo is None: continue
        resname = resinfo.GetResidueName()

        if resname == 'TU0':
            for na in atom.GetNeighbors():
                if na.GetPDBResidueInfo() is None: continue
                elif na.GetPDBResidueInfo().GetResidueName() != 'TU0':
                    mol.SetIntProp('terminal_idx1', atom.GetIdx())
                    mol.SetIntProp('terminal_ne_idx1', na.GetIdx())
                    count += 1

        elif resname == 'TU1':
            for na in atom.GetNeighbors():
                if na.GetPDBResidueInfo() is None: continue
                elif na.GetPDBResidueInfo().GetResidueName() != 'TU1':
                    mol.SetIntProp('terminal_idx2', atom.GetIdx())
                    mol.SetIntProp('terminal_ne_idx2', na.GetIdx())
                    count += 1
    
    return count


def set_mainchain_flag(mol):

    for atom in mol.GetAtoms():
        atom.SetBoolProp('main_chain', False)
    
    linker_result = set_linker_flag(mol)
    terminal_result = set_terminal_idx(mol)
    if not linker_result and terminal_result == 0:
        return False

    if terminal_result > 0:
        if mol.GetIntProp('terminal_ne_idx1') == mol.GetIntProp('terminal_ne_idx2'):
            path = [mol.GetIntProp('terminal_ne_idx1')]
        else:
            path = Chem.GetShortestPath(mol, mol.GetIntProp('terminal_ne_idx1'), mol.GetIntProp('terminal_ne_idx2'))
    elif mol.GetIntProp('head_idx') == mol.GetIntProp('tail_idx'):
        path = Chem.GetShortestPath(mol, mol.GetIntProp('head_idx'), mol.GetIntProp('head_ne_idx'))
    else:
        path = Chem.GetShortestPath(mol, mol.GetIntProp('head_idx'), mol.GetIntProp('tail_idx'))

    for idx in path:
        atom = mol.GetAtomWithIdx(idx)
        atom.SetBoolProp('main_chain', True)
        for batom in atom.GetNeighbors():
            if batom.GetTotalDegree() == 1:  # Expect -H, =O, =S, -F, -Cl, -Br, -I
                batom.SetBoolProp('main_chain', True)
    
    rings = Chem.GetSymmSSSR(mol)
    m_rings = []
    for ring in rings:
        dup = list(set(path) & set(ring))
        if len(dup) > 0:
            m_rings.append(ring)
            for idx in ring:
                atom = mol.GetAtomWithIdx(idx)
                atom.SetBoolProp('main_chain', True)
                for batom in atom.GetNeighbors():
                    if batom.GetTotalDegree() == 1:  # Expect -H, =O, =S, -F, -Cl, -Br, -I
                        batom.SetBoolProp('main_chain', True)

    for m_ring in m_rings:
        for ring in rings:
            dup = list(set(m_ring) & set(ring))
            if len(dup) > 0:
                for idx in ring:
                    atom = mol.GetAtomWithIdx(idx)
                    atom.SetBoolProp('main_chain', True)
                    for batom in atom.GetNeighbors():
                        if batom.GetTotalDegree() == 1:  # Expect -H, =O, =S, -F, -Cl, -Br, -I
                            batom.SetBoolProp('main_chain', True)

    return True


def check_chiral_monomer(mol):
    n_chiral = 0
    ter = utils.mol_from_smiles('*[C]')
    mol_c = utils.deepcopy_mol(mol)
    mol_c = terminate_mols(mol_c, ter, random_rot=True)
    set_mainchain_flag(mol_c)
    for atom in mol_c.GetAtoms():
        if (int(atom.GetChiralTag()) == 1 or int(atom.GetChiralTag()) == 2) and atom.GetBoolProp('main_chain'):
            n_chiral += 1

    return n_chiral


def get_chiral_list(mol, confId=0):
    mol_c = utils.deepcopy_mol(mol)
    set_linker_flag(mol_c)
    if mol_c.GetIntProp('head_idx') >= 0 and mol_c.GetIntProp('tail_idx') >= 0:
        ter = utils.mol_from_smiles('*[C]')
        mol_c = terminate_mols(mol_c, ter, random_rot=True)
    set_mainchain_flag(mol_c)

    Chem.AssignStereochemistryFrom3D(mol_c, confId=confId)
    chiral_centers = np.array(Chem.FindMolChiralCenters(mol_c))

    if len(chiral_centers) == 0:
        return []

    chiral_centers = [int(x) for x in chiral_centers[:, 0]]
    chiral_list = []
    for atom in mol_c.GetAtoms():
        if atom.GetBoolProp('main_chain') and atom.GetIdx() in chiral_centers:
            chiral_list.append(int(atom.GetChiralTag()))
    chiral_list = np.array(chiral_list)

    return chiral_list


def get_tacticity(mol, confId=0):
    """
    poly.get_tacticity

    Get tacticity of polymer

    Args:
        mol: RDkit Mol object

    Optional args:
        confId: Target conformer ID (int)

    Returns:
        tacticity (str; isotactic, syndiotactic, atactic, or none)
    """

    tac = 'none'
    chiral_list = get_chiral_list(mol, confId=confId)

    chiral_cw = np.count_nonzero(chiral_list == 1)
    chiral_ccw = np.count_nonzero(chiral_list == 2)

    if chiral_cw == len(chiral_list) or chiral_ccw == len(chiral_list):
        tac = 'isotactic'

    else:
        chiral_even_s = np.count_nonzero(chiral_list[0::2] == 1)
        chiral_even_r = np.count_nonzero(chiral_list[0::2] == 2)
        chiral_odd_s = np.count_nonzero(chiral_list[1::2] == 1)
        chiral_odd_r = np.count_nonzero(chiral_list[1::2] == 2)
        if ((chiral_even_s == len(chiral_list[0::2]) and chiral_odd_r == len(chiral_list[1::2]))
                or (chiral_even_r == len(chiral_list[0::2]) and chiral_odd_s == len(chiral_list[1::2]))):
            tac = 'syndiotactic'
        else:
            tac = 'atactic'

    return tac


def check_tacticity(mol, tacticity, tac_array=None, confId=0):

    if tacticity == 'atactic' and tac_array is None:
        return True

    tac = get_tacticity(mol, confId=confId)

    check = False
    if tac == 'none':
        check = True

    elif tacticity == 'atactic' and tac_array is not None:
        set_mainchain_flag(mol)
        Chem.AssignStereochemistryFrom3D(mol, confId=confId)
        chiral_list = np.array(Chem.FindMolChiralCenters(mol))
        chiral_centers = [int(x) for x in chiral_list[:, 0]]
        chiral_idx = []

        for atom in mol.GetAtoms():
            if atom.GetBoolProp('main_chain') and atom.GetIdx() in chiral_centers:
                chiral_idx.append(int(atom.GetChiralTag()))
        chiral_idx = np.array(chiral_idx)

        tac_list1 = np.where(np.array(tac_array)[:len(chiral_idx)], 1, 2)
        tac_list2 = np.where(np.array(tac_array)[:len(chiral_idx)], 2, 1)

        if (chiral_idx == tac_list1).all() or (chiral_idx == tac_list2).all():
            check = True

    elif tac == 'isotactic' and tacticity == 'isotactic':
        check = True

    elif tac == 'syndiotactic' and tacticity == 'syndiotactic':
        check = True


    return check


def polymer_stats(mol, df=False):
    """
    poly.polymer_stats

    Calculate statistics of polymers

    Args:
        mol: RDkit Mol object

    Optional args:
        df: Data output type, True: pandas.DataFrame, False: dict  (boolean)

    Returns:
        dict or pandas.DataFrame
    """

    molcount = utils.count_mols(mol)
    natom = [0 for i in range(molcount)]
    molweight = [0.0 for i in range(molcount)]

    for atom in mol.GetAtoms():
        molid = atom.GetIntProp('mol_id')
        natom[molid-1] += 1
        molweight[molid-1] += atom.GetMass()

    natom = np.array(natom)
    molweight = np.array(molweight)

    poly_stats = {
        'n_mol': molcount,
        'n_atom': natom if not df else '/'.join([str(n) for n in natom]),
        'n_atom_mean': np.mean(natom),
        'n_atom_var': np.var(natom),
        'mol_weight': molweight if not df else '/'.join([str(n) for n in molweight]),
        'Mn': np.mean(molweight),
        'Mw': np.sum(molweight**2)/np.sum(molweight),
        'Mw/Mn': np.sum(molweight**2)/np.sum(molweight)/np.mean(molweight)
    }

    mol.SetIntProp('n_mol', molcount)
    mol.SetDoubleProp('n_atom_mean', poly_stats['n_atom_mean'])
    mol.SetDoubleProp('n_atom_var', poly_stats['n_atom_var'])
    mol.SetDoubleProp('Mn', poly_stats['Mn'])
    mol.SetDoubleProp('Mw', poly_stats['Mw'])
    mol.SetDoubleProp('Mw/Mn', poly_stats['Mw/Mn'])

    return poly_stats if not df else pd.DataFrame(poly_stats, index=[0])


##########################################################
# Utility function for SMILES of polymers
##########################################################
def polymerize_MolFromSmiles(smiles, n=2, terminal='C'):
    """
    poly.polymerize_MolFromSmiles

    Generate polimerized RDkit Mol object from SMILES

    Args:
        smiles: SMILES (str)
        n: Polimerization degree (int)
        terminal: SMILES of terminal substruct (str)

    Returns:
        RDkit Mol object
    """

    poly_smiles = make_linearpolymer(smiles, n=n, terminal=terminal)

    try:
        mol = Chem.MolFromSmiles(poly_smiles)
        mol = Chem.AddHs(mol)
    except Exception as e:
        mol = None
    
    return mol


def make_linearpolymer(smiles, n=2, terminal='C'):
    """
    poly.make_linearpolymer

    Generate linearpolymer SMILES from monomer SMILES

    Args:
        smiles: SMILES (str)
        n: Polimerization degree (int)
        terminal: SMILES of terminal substruct (str)

    Returns:
        SMILES
    """

    dummy = '[*]'
    dummy_head = '[Nh]'
    dummy_tail = '[Og]'

    smiles_in = smiles
    smiles = smiles.replace('[*]', '*')
    smiles = smiles.replace('[3H]', '*')

    if smiles.count('*') != 2:
        utils.radon_print('Illegal number of connecting points in SMILES. %s' % smiles_in, level=2)
        return None

    smiles = smiles.replace('\\', '\\\\')
    smiles_head = re.sub(r'\*', dummy_head, smiles, 1)
    smiles_tail = re.sub(r'\*', dummy_tail, smiles, 1)
    #smiles_tail = re.sub(r'%s\\\\' % dummy_tail, '%s/' % dummy_tail, smiles_tail, 1)
    #smiles_tail = re.sub(r'%s/' % dummy_tail, '%s\\\\' % dummy_tail, smiles_tail, 1)

    try:
        mol_head = Chem.MolFromSmiles(smiles_head)
        mol_tail = Chem.MolFromSmiles(smiles_tail)
        mol_terminal = Chem.MolFromSmiles(terminal)
        mol_dummy = Chem.MolFromSmiles(dummy)
        mol_dummy_head = Chem.MolFromSmiles(dummy_head)
        mol_dummy_tail = Chem.MolFromSmiles(dummy_tail)

        con_point = 1
        bdtype = 1.0
        for atom in mol_tail.GetAtoms():
            if atom.GetSymbol() == mol_dummy_tail.GetAtomWithIdx(0).GetSymbol():
                con_point = atom.GetNeighbors()[0].GetIdx()
                bdtype = mol_tail.GetBondBetweenAtoms(atom.GetIdx(), con_point).GetBondTypeAsDouble()
                break
        
        for poly in range(n-1):
            mol_head = Chem.rdmolops.ReplaceSubstructs(mol_head, mol_dummy, mol_tail, replacementConnectionPoint=con_point)[0]
            mol_head = Chem.RWMol(mol_head)
            for atom in mol_head.GetAtoms():
                if atom.GetSymbol() == mol_dummy_tail.GetAtomWithIdx(0).GetSymbol():
                    idx = atom.GetIdx()
                    break
            mol_head.RemoveAtom(idx)
            Chem.SanitizeMol(mol_head)
        
        mol = mol_head.GetMol()
        mol = Chem.rdmolops.ReplaceSubstructs(mol, mol_dummy, mol_terminal, replacementConnectionPoint=0)[0]

        poly_smiles = Chem.MolToSmiles(mol)
        poly_smiles = poly_smiles.replace(dummy_head, terminal)

    except Exception as e:
        utils.radon_print('Cannot transform to polymer from monomer SMILES. %s' % smiles_in, level=2)
        return None

    return poly_smiles


def make_cyclicpolymer(smiles, n=2, return_mol=False, removeHs=False):
    """
    poly.make_cyclicpolymer

    Generate cyclicpolymer SMILES from monomer SMILES

    Args:
        smiles: SMILES (str)
        n: Polimerization degree (int)
        return_mol: Return Mol object (True) or SMILES strings (False)

    Returns:
        SMILES or RDKit Mol object
    """

    mol = polymerize_MolFromSmiles(smiles, n=n, terminal='*')
    # mol = utils.mol_from_smiles(smiles, stereochemistry_control=False)
    # mol = polymerize_mols(mol, n)
    if mol is None:
        return None
        
    set_mainchain_flag(mol)
    set_linker_flag(mol)
    head_idx = mol.GetIntProp('head_idx')
    tail_idx = mol.GetIntProp('tail_idx')
    head_ne_idx = mol.GetIntProp('head_ne_idx')
    tail_ne_idx = mol.GetIntProp('tail_ne_idx')

    # Get bond type of deleting bonds
    bd1type = mol.GetBondBetweenAtoms(head_idx, head_ne_idx).GetBondTypeAsDouble()
    bd2type = mol.GetBondBetweenAtoms(tail_idx, tail_ne_idx).GetBondTypeAsDouble()
    
    # Delete linker atoms and bonds
    mol = utils.remove_atom(mol, head_idx)
    if tail_idx > head_idx: tail_idx -= 1
    mol = utils.remove_atom(mol, tail_idx)

    # Add a new bond
    if mol.GetIntProp('head_ne_idx') > mol.GetIntProp('tail_idx'): head_ne_idx -= 1
    if mol.GetIntProp('head_ne_idx') > mol.GetIntProp('head_idx'): head_ne_idx -= 1
    if mol.GetIntProp('tail_ne_idx') > mol.GetIntProp('tail_idx'): tail_ne_idx -= 1
    if mol.GetIntProp('tail_ne_idx') > mol.GetIntProp('head_idx'): tail_ne_idx -= 1
    if bd1type == 2.0 or bd2type == 2.0:
        mol = utils.add_bond(mol, tail_ne_idx, head_ne_idx, order=Chem.rdchem.BondType.DOUBLE)
    elif bd1type == 3.0 or bd2type == 3.0:
        mol = utils.add_bond(mol, tail_ne_idx, head_ne_idx, order=Chem.rdchem.BondType.TRIPLE)
    else:
        mol = utils.add_bond(mol, tail_ne_idx, head_ne_idx, order=Chem.rdchem.BondType.SINGLE)

    Chem.SanitizeMol(mol)

    if removeHs:
        try:
            mol = Chem.RemoveHs(mol)
        except:
            return None

    if return_mol:
        return mol
    else:
        poly_smiles = Chem.MolToSmiles(mol)
        return poly_smiles


def make_cyclicpolymer_mp(smiles, n=2, return_mol=False, removeHs=False, mp=None):
    """
    poly.make_cyclicpolymer_mp

    Multiprocessing version of make_cyclicpolymer

    Args:
        smiles: SMILES (list, str)
        n: Polimerization degree (int)
        return_mol: Return Mol object (True) or SMILES strings (False)
        mp: Number of process (int)

    Returns:
        List of SMILES or RDKit Mol object
    """
    if mp is None:
        mp = utils.cpu_count()
    
    c = utils.picklable_const()
    args = [[smi, n, return_mol, removeHs, c] for smi in smiles]

    with confu.ProcessPoolExecutor(max_workers=mp) as executor:
        results = executor.map(_make_cyclicpolymer_worker, args)
        res = [r for r in results]

    return res


def _make_cyclicpolymer_worker(args):
    smi, n, return_mol, removeHs, c = args
    utils.restore_const(c)
    res = make_cyclicpolymer(smi, n=n, return_mol=return_mol, removeHs=removeHs)
    if return_mol:
        utils.picklable()
    return res


def substruct_match_mol(pmol, smol, useChirality=False):

    psmi = Chem.MolToSmiles(pmol)
    pmol = make_cyclicpolymer(psmi, 3, return_mol=True)
    if pmol is None:
        return False

    return pmol.HasSubstructMatch(smol, useChirality=useChirality)


def substruct_match_smiles(poly_smiles, sub_smiles, useChirality=False):
    """
    poly.substruct_match_smiles

    Substruct matching of smiles2 in smiles1 as a polymer structure

    Args:
        poly_smiles: polymer SMILES (str)
        sub_smiles: substruct SMILES (str)

    Optional args:
        useChirality: enables the use of stereochemistry in the matching (boolean)

    Returns:
        RDkit Mol object
    """

    pmol = make_cyclicpolymer(poly_smiles, 3, return_mol=True)
    if pmol is None:
        return False
    smol = Chem.MolFromSmarts(sub_smiles)

    return pmol.HasSubstructMatch(smol, useChirality=useChirality)


def substruct_match_smiles_list(smiles, smi_series, mp=None):
    
    if mp is None:
        mp = utils.cpu_count()

    c = utils.picklable_const()
    args = [[smi, smiles, c] for smi in smi_series]

    with confu.ProcessPoolExecutor(max_workers=mp) as executor:
        results = executor.map(_substruct_match_smiles_worker, args)
        res = [r for r in results]
    
    smi_list = smi_series[res].index.values.tolist()

    return smi_list


def _substruct_match_smiles_worker(args):
    smi, smiles, c = args
    utils.restore_const(c)
    return substruct_match_smiles(smi, smiles)


def full_match_mol(mol1, mol2, monomerize=True):
    smiles1 = Chem.MolToSmiles(mol1)
    smiles2 = Chem.MolToSmiles(mol2)

    return full_match_smiles(smiles1, smiles2, monomerize=monomerize)


def full_match_smiles(smiles1, smiles2, monomerize=True):
    """
    poly.full_match_smiles

    Full matching of smiles1 and smiles2 as a polymer structure

    Args:
        smiles1, smiles2: polymer SMILES (str)

    Returns:
        RDkit Mol object
    """
    if monomerize:
        smiles1 = monomerization_smiles(smiles1)
        smiles2 = monomerization_smiles(smiles2)

    try:
        if Chem.MolFromSmiles(smiles1).GetNumAtoms() != Chem.MolFromSmiles(smiles2).GetNumAtoms():
            return False
    except:
        return False

    smi1 = make_cyclicpolymer(smiles1, n=3)
    smi2 = make_cyclicpolymer(smiles2, n=3)
    if smi1 is None or smi2 is None:
        return False

    # Canonicalize
    smi1 = Chem.MolToSmiles(Chem.MolFromSmiles(smi1))
    smi2 = Chem.MolToSmiles(Chem.MolFromSmiles(smi2))

    if smi1 == smi2:
        return True
    else:
        return False


def full_match_smiles_list(smiles, smi_series, mp=None, monomerize=True):
    
    if mp is None:
        mp = utils.cpu_count()

    c = utils.picklable_const()
    args = [[smiles, smi, monomerize, c] for smi in smi_series]
    with confu.ProcessPoolExecutor(max_workers=mp) as executor:
        results = executor.map(_full_match_smiles_worker, args)
        res = [r for r in results]
    
    smi_list = smi_series[res].index.values.tolist()

    return smi_list


def _full_match_smiles_worker(args):
    smiles, smi, monomerize, c = args
    utils.restore_const(c)
    return full_match_smiles(smiles, smi, monomerize=monomerize)


def full_match_smiles_listself(smi_series, mp=None, monomerize=True):
    
    if mp is None:
        mp = utils.cpu_count()

    idx_list = smi_series.index.tolist()
    c = utils.picklable_const()
    args = [[smi, monomerize, c] for smi in smi_series]
    with confu.ProcessPoolExecutor(max_workers=mp) as executor:
        results = executor.map(_make_full_match_smiles, args)
        smi_list = [r for r in results]

    result = []
    i = 1
    for idx1, smi1 in tqdm(zip(idx_list, smi_list), total=len(smi_series), desc='[Full match smiles]', disable=const.tqdm_disable):
        match_list = []
        for idx2, smi2 in zip(idx_list[i:], smi_list[i:]):
            if smi1 == smi2:
                match_list.append(idx2)
        if len(match_list) > 0:
            result.append((idx1, match_list))
        i += 1

    return result


def full_match_smiles_listlist(smi_series1, smi_series2, mp=None, monomerize=True):
    
    if mp is None:
        mp = utils.cpu_count()

    idx_list1 = smi_series1.index.tolist()
    idx_list2 = smi_series2.index.tolist()
    c = utils.picklable_const()

    args = [[smi, monomerize, c] for smi in smi_series1]
    with confu.ProcessPoolExecutor(max_workers=mp) as executor:
        results = executor.map(_make_full_match_smiles, args)
        smi_list1 = [r for r in results]

    args = [[smi, monomerize, c] for smi in smi_series2]
    with confu.ProcessPoolExecutor(max_workers=mp) as executor:
        results = executor.map(_make_full_match_smiles, args)
        smi_list2 = [r for r in results]

    result = []
    for idx1, smi1 in tqdm(zip(idx_list1, smi_list1), total=len(smi_series1), desc='[Full match smiles]', disable=const.tqdm_disable):
        match_list = []
        for idx2, smi2 in zip(idx_list2, smi_list2):
            if smi1 == smi2:
                match_list.append(idx2)
        if len(match_list) > 0:
            result.append((idx1, match_list))

    return result


def _make_full_match_smiles(args):
    smi, monomerize, c = args
    utils.restore_const(c)

    if monomerize:
        smi = monomerization_smiles(smi)
    smi = make_cyclicpolymer(smi, n=3)
    if smi is None:
        return args[0]

    try:
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    except Exception as e:
        utils.radon_print('Cannot convert to canonical SMILES from %s' % smi, level=2)
        return args[0]
        
    return smi


def ff_test_mp(smi_list, ff, mp=None):
    if mp is None:
        mp = utils.cpu_count()

    c = utils.picklable_const()
    args = [[smi, ff, c] for smi in smi_list]

    with confu.ProcessPoolExecutor(max_workers=mp) as executor:
        results = executor.map(_ff_test_mp_worker, args)
        res = [r for r in results]

    return res


def _ff_test_mp_worker(args):
    smi, ff, c = args
    utils.restore_const(c)

    try:
        mol = polymerize_MolFromSmiles(smi, n=2)
        mol = Chem.AddHs(mol)
        result = ff.ff_assign(mol)
    except:
        result = False
        
    return result


def monomerization_smiles(smiles, min_length=1):
    
    if smiles.count('*') != 2:
        utils.radon_print('Illegal number of connecting points in SMILES. %s' % smiles_in, level=2)
        return smiles

    smi = smiles.replace('[*]', '[3H]')
    smi = smi.replace('*', '[3H]')

    try:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
    except:
        utils.radon_print('Cannot convert to Mol object from %s' % smiles, level=2)
        return smiles

    set_linker_flag(mol)
    path = list(Chem.GetShortestPath(mol, mol.GetIntProp('head_idx'), mol.GetIntProp('tail_idx')))
    del path[0], path[-1]

    length = len(path)

    for l in range(min_length, int(length/2+1)):
        if length % l == 0:
            bidx = [mol.GetBondBetweenAtoms(path[l*i-1], path[l*i]).GetIdx() for i in range(1, int(length/l))]
            fmol = Chem.FragmentOnBonds(mol, bidx, addDummies=True)
            try:
                RDLogger.DisableLog('rdApp.*')
                fsmi = [Chem.MolToSmiles(Chem.MolFromSmiles(re.sub('\[[0-9]+\*\]', '[*]', x).replace('[3H]', '[*]'))) for x in Chem.MolToSmiles(fmol).split('.')]
            except:
                RDLogger.EnableLog('rdApp.*')
                continue

            RDLogger.EnableLog('rdApp.*')
            if len(list(set(fsmi))) == 1 and fsmi[0].count('*') == 2:
                csmi = make_linearpolymer(fsmi[0], n=len(fsmi), terminal='*')
                if csmi is not None:
                    try:
                        if Chem.MolToSmiles(Chem.MolFromSmiles(csmi)) == Chem.MolToSmiles(Chem.MolFromSmiles(smiles)):
                            return fsmi[0]
                    except Exception as e:
                        utils.radon_print('Cannot convert to canonical SMILES from %s' % smi, level=2)

    return smiles


def extract_mainchain(smiles):

    main_smi = None
    if smiles.count('*') != 2:
        utils.radon_print('Illegal number of connecting points in SMILES. %s' % smiles, level=2)
        return main_smi

    smi = smiles.replace('[*]', '[3H]')
    smi = smi.replace('*', '[3H]')

    try:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
    except:
        utils.radon_print('Cannot convert to Mol object from %s' % smiles, level=2)
        return main_smi

    set_mainchain_flag(mol)

    for atom in mol.GetAtoms():
        if atom.GetBoolProp('main_chain'):
            for na in atom.GetNeighbors():
                if not na.GetBoolProp('main_chain'):
                    bidx = mol.GetBondBetweenAtoms(atom.GetIdx(), na.GetIdx()).GetIdx()
                    mol = Chem.FragmentOnBonds(mol, [bidx], addDummies=False)

    RDLogger.DisableLog('rdApp.*')

    try:
        fsmi = [Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x in Chem.MolToSmiles(mol).split('.')]
    except:
        utils.radon_print('Cannot convert to fragmented Mol', level=2)
        RDLogger.EnableLog('rdApp.*')
        return main_smi

    RDLogger.EnableLog('rdApp.*')

    for s in fsmi:
        if '[3H]' in s:
            try:
                main_smi = Chem.MolToSmiles(Chem.MolFromSmiles(s.replace('[3H]', '*')))
            except:
                utils.radon_print('Cannot convert to canonical SMILES from %s' % smi, level=2)

    return main_smi


def polyinfo_classifier(smi, return_flag=False):
    """
    poly.polyinfo_classifier

    Classifier of polymer structure to 21 class IDs of PoLyInfo

    Args:
        smi: polymer SMILES (str)

    Returns:
        class ID (int)
    """
    class_id = 0
    
    # Definition of SMARTS of each class. [14C] means a carbon atom in main chain
    styrene = ['*-[14C]([14C]-*)c1[c,n][c,n][c,n][c,n][c,n]1']        # P02
    acryl = ['*-[14C]([14C]-*)C(=[O,S])-[O,S,N,n]']                   # P04
    ether = ['[!$(C(=[O,S]))&!$([X4&S](=O)(=O))&!Si][X2&O,X2&o][!$(C(=[O,S]))&!$([X4&S](=O)(=O))&!Si]'] # P07
    thioether = ['[!$(C(=[O,S]))&!Si][X2&S,X2&s][!$(C(=[O,S]))&!Si]'] # P08
    ester = ['[!$(C(=[O,S]))]-[O,S]C(=[O,S])-[!N&!O&!S]']             # P09
    amide = ['[!$(C(=[O,S]))]-NC(=[O,S])-[!N&!O&!S]']                 # P10
    urethane = ['*-NC(=[O,S])[O,S]-*']                                # P11
    urea = ['*-NC(=[O,S])N-*']                                        # P12
    imide = ['[X3&C,X3&c](=[O,S])[X3&N,X3&n][X3&C,X3&c](=[O,S])']     # P13
    anhyd = ['*-C(=[O,S])[O,S]C(=[O,S])-*']                           # P14
    carbonate = ['*-[O,S]C(=[O,S])[O,S]-*']                           # P15
    amine = ['[X3&N,X2&N,X4&N,X3&n,X2&n;!$([N,n][C,c](=[O,S]));!$(N=P)]'] # P16
    silane = ['*-[X4&Si]-*']                                          # P17
    phosphazene = ['*-N=P-*']                                         # P18
    ketone = ['[!N&!O&!S]-C(=[O,S])-[!N&!O&!S]']                      # P19
    sulfon = ['*-[X4&S](=O)(=O)-*', '*-[X3&S](=O)-*']                 # P20
    phenylene = ['*-c1[c,n][c,n][c,n]([c,n]c1)-*', '*-c1[c,n][c,n]([c,n][c,n]c1)-*', '*-c1[c,n]([c,n][c,n][c,n]c1)-*'] # P21

    m_smi = extract_mainchain(smi)
    if m_smi is None:
        m_smi = smi

    mr_mol = make_cyclicpolymer(m_smi, 4, return_mol=True)
    if mr_mol is None:
        if return_flag:
            return class_id, {}
        else:
            return class_id
    
    flag = {
        'PHYC': False,
        'PSTR': False,
        'PVNL': False,
        'PACR': False,
        'PHAL': False,
        'PDIE': False,
        'POXI': [mr_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in ether].count(True) > 0,
        'PSUL': [mr_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in thioether].count(True) > 0,
        'PEST': [mr_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in ester].count(True) > 0,
        'PAMD': [mr_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in amide].count(True) > 0,
        'PURT': [mr_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in urethane].count(True) > 0,
        'PURA': [mr_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in urea].count(True) > 0,
        'PIMD': [mr_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in imide].count(True) > 0,
        'PANH': [mr_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in anhyd].count(True) > 0,
        'PCBN': [mr_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in carbonate].count(True) > 0,
        'PIMN': [mr_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in amine].count(True) > 0,
        'PSIL': [mr_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in silane].count(True) > 0,
        'PPHS': [mr_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in phosphazene].count(True) > 0,
        'PKTN': [mr_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in ketone].count(True) > 0,
        'PSFO': [mr_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in sulfon].count(True) > 0,
        'PPNL': [mr_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in phenylene].count(True) > 0,
    }
    
    if list(flag.values()).count(True) == 0:
        try:
            m_mol = Chem.MolFromSmiles(m_smi)
            m_mol = Chem.AddHs(m_mol)
        except:
            utils.radon_print('Cannot convert to Mol object from %s' % m_smi, level=2)
            if return_flag:
                return class_id, {}
            else:
                return class_id

        m_nelem = {'H':0, 'C':0, 'hetero':0, 'halogen':0}
        for atom in m_mol.GetAtoms():
            elem = atom.GetSymbol()
            if elem == '*':
                continue
            elif elem in ['H', 'C']:
                m_nelem[elem] += 1
            else:
                m_nelem['hetero'] += 1
                if elem in ['F', 'Cl', 'Br', 'I']:
                    m_nelem['halogen'] += 1

        try:
            mol = Chem.MolFromSmiles(smi)
            mol = Chem.AddHs(mol)
        except:
            utils.radon_print('Cannot convert to Mol object from %s' % smi, level=2)
            if return_flag:
                return class_id, {}
            else:
                return class_id

        nelem = {'H':0, 'C':0, 'hetero':0, 'halogen':0}
        for atom in mol.GetAtoms():
            elem = atom.GetSymbol()
            if elem == '*':
                continue
            elif elem in ['H', 'C']:
                nelem[elem] += 1
            else:
                nelem['hetero'] += 1
                if elem in ['F', 'Cl', 'Br', 'I']:
                    nelem['halogen'] += 1

        ndbond = 0
        nabond = 0
        ntbond = 0
        for bond in mol.GetBonds():
            if bond.GetBondTypeAsDouble() == 2.0:
                ndbond += 1
            elif bond.GetBondTypeAsDouble() == 1.5:
                nabond += 1
            elif bond.GetBondTypeAsDouble() == 3.0:
                ntbond += 1
        nubond = ndbond + nabond + ntbond

        r_mol = make_cyclicpolymer(smi, 4, return_mol=True)
        if r_mol is None:
            if return_flag:
                return class_id, {}
            else:
                return class_id

        for atom in r_mol.GetAtoms():
            if atom.GetSymbol() == 'C' and atom.GetBoolProp('main_chain'):
                atom.SetIsotope(14)
        
        flag['PSTR'] = [r_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in styrene].count(True) > 0
        flag['PACR'] = [r_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in acryl].count(True) > 0
        flag['PHAL'] = m_nelem['halogen'] > 0 #(m_nelem['halogen'] > 1 or (nelem['halogen'] - m_nelem['halogen']) > 0)
        flag['PDIE'] = nelem['hetero'] == 0 and (ndbond > 0 or ntbond > 0) and nabond == 0
        flag['PHYC'] = nelem['hetero'] == 0 and nubond == 0
        flag['PVNL'] = m_nelem['halogen'] == 0 and (nelem['hetero'] > 0 or nabond > 0) and not flag['PHYC'] and not flag['PDIE']
    
    if flag['PURT']:
        class_id = 11
    elif flag['PURA']:
        class_id = 12
    elif flag['PIMD']:
        class_id = 13
    elif flag['PANH']:
        class_id = 14
    elif flag['PCBN']:
        class_id = 15

    elif flag['PEST']:
        class_id = 9
    elif flag['PAMD']:
        class_id = 10

    elif flag['PSIL']:
        class_id = 17
    elif flag['PPHS']:
        class_id = 18
    elif flag['PKTN']:
        class_id = 19
    elif flag['PSFO']:
        class_id = 20

    elif flag['POXI']:
        class_id = 7
    elif flag['PSUL']:
        class_id = 8
    elif flag['PIMN']:
        class_id = 16        
    elif flag['PPNL']:
        class_id = 21

    elif flag['PSTR']:
        class_id = 2
    elif flag['PACR']:
        class_id = 4
    elif flag['PHAL']:
        class_id = 5
    elif flag['PDIE']:
        class_id = 6
    elif flag['PHYC']:
        class_id = 1
    elif flag['PVNL']:
        class_id = 3
        
    if return_flag:
        return class_id, flag
    else:
        return class_id


def polyinfo_classifier_list(smi_series, return_flag=False, mp=None):

    if mp is None:
        mp = utils.cpu_count()

    c = utils.picklable_const()
    args = [[smi, return_flag, c] for smi in smi_series]
    with confu.ProcessPoolExecutor(max_workers=mp) as executor:
        results = executor.map(_polyinfo_classifier_worker, args)
        res = [r for r in results]

    return res


def _polyinfo_classifier_worker(args):
    smi, return_flag, c = args
    utils.restore_const(c)
    return polyinfo_classifier(smi, return_flag=return_flag)

