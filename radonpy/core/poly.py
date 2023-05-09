#  Copyright (c) 2022. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# ******************************************************************************
# core.poly module
# ******************************************************************************

import numpy as np
import pandas as pd
from copy import deepcopy
import re
import random
import concurrent.futures as confu
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import Geometry as Geom
from rdkit import RDLogger
from . import calc, const, utils

__version__ = '0.2.6'

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


def connect_mols(mol1, mol2, bond_length=1.5, dihedral=np.pi, random_rot=False, dih_type='monomer',
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
    if headhead: set_linker_flag(mol1, reverse=True)
    else: set_linker_flag(mol1)
    if tailtail and not headhead: set_linker_flag(mol2, reverse=True)
    else: set_linker_flag(mol2)

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
    mol = utils.remove_atom(mol, mol2.GetIntProp('head_idx') + mol1_n)
    mol = utils.remove_atom(mol, mol1.GetIntProp('tail_idx'))

    # Add a new bond
    tail_ne_idx = mol1.GetIntProp('tail_ne_idx')
    head_ne_idx = mol1_n - 1 + mol2.GetIntProp('head_ne_idx')
    if mol1.GetIntProp('tail_ne_idx') > mol1.GetIntProp('tail_idx'): tail_ne_idx -= 1
    if mol2.GetIntProp('head_ne_idx') > mol2.GetIntProp('head_idx'): head_ne_idx -= 1
    mol = utils.add_bond(mol, tail_ne_idx, head_ne_idx, order=Chem.rdchem.BondType.SINGLE)
    #if bd1type == 2.0 or bd2type == 2.0:
    #    mol = utils.add_bond(mol, tail_ne_idx, head_ne_idx, order=Chem.rdchem.BondType.DOUBLE)
    #elif bd1type == 3.0 or bd2type == 3.0:
    #    mol = utils.add_bond(mol, tail_ne_idx, head_ne_idx, order=Chem.rdchem.BondType.TRIPLE)
    #else:
    #    mol = utils.add_bond(mol, tail_ne_idx, head_ne_idx, order=Chem.rdchem.BondType.SINGLE)

    # Finalize
    Chem.SanitizeMol(mol)
    set_linker_flag(mol)

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
    angles = []
    dihedrals = []
    impropers = []
    cell = None

    if hasattr(mol1, 'angles'):
        angles = deepcopy(mol1.angles)
    if hasattr(mol2, 'angles'):
        for angle in mol2.angles:
            angles.append(
                utils.Angle(
                    a=angle.a+mol1_n,
                    b=angle.b+mol1_n,
                    c=angle.c+mol1_n,
                    ff=deepcopy(angle.ff)
                )
            )

    if hasattr(mol1, 'dihedrals'):
        dihedrals = deepcopy(mol1.dihedrals)
    if hasattr(mol2, 'dihedrals'):
        for dihedral in mol2.dihedrals:
            dihedrals.append(
                utils.Dihedral(
                    a=dihedral.a+mol1_n,
                    b=dihedral.b+mol1_n,
                    c=dihedral.c+mol1_n,
                    d=dihedral.d+mol1_n,
                    ff=deepcopy(dihedral.ff)
                )
            )

    if hasattr(mol1, 'impropers'):
        impropers = deepcopy(mol1.impropers)
    if hasattr(mol2, 'impropers'):
        for improper in mol2.impropers:
            impropers.append(
                utils.Improper(
                    a=improper.a+mol1_n,
                    b=improper.b+mol1_n,
                    c=improper.c+mol1_n,
                    d=improper.d+mol1_n,
                    ff=deepcopy(improper.ff)
                )
            )
    
    if hasattr(mol1, 'cell'):
        cell = deepcopy(mol1.cell)
    elif hasattr(mol2, 'cell'):
        cell = deepcopy(mol2.cell)

    # Generate PDB information and repeating unit information
    resid = []
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


def polymerize_mols(mol, n, bond_length=1.5, dihedral=np.pi, random_rot=False, dih_type='monomer',
                    confId=0, tacticity='atactic', atac_ratio=0.5, tac_array=None):
    """
    poly.polymerize_mols

    Simple polymerization function of RDkit Mol object

    Args:
        mol: RDkit Mol object
        n: Polymerization degree (int)

    Optional args:
        bond_length: Bond length of connecting bond (float, angstrome)
        random_rot: Dihedral angle around connecting bond is rotated randomly (boolean)
        dihedral: Dihedral angle around connecting bond (float, radian)
        dih_type: Definition type of dihedral angle (str; monomer, or bond)
        tacticity: atactic, syndiotactic, or isotactic
        atac_ratio: Chiral inversion ration for atactic polymer
        tac_array: Array of chiral inversion for atactic polymer (list of boolean)
        confId: Target conformer ID

    Returns:
        Rdkit Mol object
    """

    chi = gen_chi_array(tacticity, n, atac_ratio=atac_ratio, tac_array=tac_array)

    if n < 2: return mol

    if chi[0]:
        mol1 = calc.mirror_inversion_mol(mol, confId=confId)
    else:
        mol1 = utils.deepcopy_mol(mol)

    if chi[1]:
        mol2 = calc.mirror_inversion_mol(mol, confId=confId)
    else:
        mol2 = utils.deepcopy_mol(mol)

    poly = connect_mols(mol1, mol2, bond_length=bond_length, dihedral=dihedral, random_rot=random_rot, dih_type=dih_type, confId1=confId, confId2=confId)
    for i in range(n-2):
        if chi[i+2]:
            mol_c = calc.mirror_inversion_mol(mol, confId=confId)
        else:
            mol_c = utils.deepcopy_mol(mol)

        poly = connect_mols(poly, mol_c, bond_length=bond_length, dihedral=dihedral, random_rot=random_rot, dih_type=dih_type, confId2=confId)

    return poly


def copolymerize_mols(mols, n, bond_length=1.5, dihedral=np.pi, random_rot=False, dih_type='monomer',
                    confId=0, tacticity='atactic', atac_ratio=0.5, tac_array=None):
    """
    poly.copolymerize_mols

    Simple co-polymerization function of RDkit Mol object

    Args:
        mols: Array of RDkit Mol object
        n: Polymerization degree (int)

    Optional args:
        bond_length: Bond length of connecting bond (float, angstrome)
        random_rot: Dihedral angle around connecting bond is rotated randomly (boolean)
        dihedral: Dihedral angle around connecting bond (float, radian)
        dih_type: Definition type of dihedral angle (str; monomer, or bond)
        tacticity: atactic, syndiotactic, or isotactic
        atac_ratio: Chiral inversion ration for atactic polymer
        tac_array: Array of chiral inversion for atactic polymer (list of boolean)

    Returns:
        Rdkit Mol object
    """

    poly = None

    tac_list = []
    n_chi = 0
    for mol in mols:
        c = True if check_chiral_monomer(mol) == 1 else False
        tac_list.append(c)
        if c: n_chi += 1

    chi = gen_chi_array(tacticity, n*len(mols), atac_ratio=atac_ratio, tac_array=tac_array)

    k = -1
    for i in range(n):
        for j, mol in enumerate(mols):
            if tac_list[j]: k += 1
            if tac_list[j] and chi[k]:
                mol_c = calc.mirror_inversion_mol(mol, confId=confId)
            else:
                mol_c = utils.deepcopy_mol(mol)

            poly = connect_mols(poly, mol_c, bond_length=bond_length, dihedral=dihedral, random_rot=random_rot, dih_type=dih_type,
                                confId1=confId, res_name_2='RU%s' % const.pdb_id[j])

    return poly


def random_copolymerize_mols(mols, n, ratio, bond_length=1.5, dihedral=np.pi, random_rot=False, dih_type='monomer',
                            ratio_type='exact', tacticity='atactic', atac_ratio=0.5, confId=0):
    """
    poly.random_copolymerize_mols

    Simple random co-polymerization function of RDkit Mol object

    Args:
        mols: Array of RDkit Mol object
        n: Polymerization degree (int)
        ratio: Array of monomer ratio (float, sum=1.0)

    Optional args:
        ratio_type: exact or choice
        bond_length: Bond length of connecting bond (float, angstrome)
        random_rot: Dihedral angle around connecting bond is rotated randomly (boolean)
        dihedral: Dihedral angle around connecting bond (float, radian)
        dih_type: Definition type of dihedral angle (str; monomer, or bond)
        tacticity: atactic, syndiotactic, or isotactic
        atac_ratio: Chiral inversion ration for atactic polymer
        tac_array: Array of chiral inversion for atactic polymer (list of boolean)

    Returns:
        Rdkit Mol object
    """

    if len(mols) != len(ratio):
        utils.radon_print('Inconsistency length of mols and ratio', level=3)
        return None

    ratio = np.array(ratio) / np.sum(ratio)

    poly = None

    chi = gen_chi_array(tacticity, n, atac_ratio=atac_ratio)

    if ratio_type == 'choice':
        mol_index = np.random.choice(a=list(range(len(mols))), size=n, p=ratio)
    elif ratio_type == 'exact':
        mol_index = []
        for i, r in enumerate(ratio):
            tmp = [i]*int(n*r+0.5)
            mol_index.extend(tmp)
        if len(mol_index) < n:
            d = int(n-len(mol_index))
            imax = np.argmax(ratio)
            tmp = [imax]*d
            mol_index.extend(tmp)
        elif len(mol_index) > n:
            d = int(len(mol_index)-n)
            imax = np.argmax(ratio)
            for i in range(d):
                mol_index.remove(imax)
        random.shuffle(mol_index)

    for i in range(n):
        if chi[i]:
            mol_c = calc.mirror_inversion_mol(mols[mol_index[i]], confId=confId)
        else:
            mol_c = utils.deepcopy_mol(mols[mol_index[i]])

        poly = connect_mols(poly, mol_c, bond_length=bond_length, dihedral=dihedral, random_rot=random_rot, dih_type=dih_type,
                            confId1=confId, res_name_2='RU%s' % const.pdb_id[mol_index[i]])

    return poly


def block_copolymerize_mols(mols, n, bond_length=1.5, dihedral=np.pi, random_rot=False, dih_type='monomer',
                            tacticity='atactic', atac_ratio=0.5, tac_array=None, confId=0):
    """
    poly.block_copolymerize_mols

    Simple block co-polymerization function of RDkit Mol object

    Args:
        mols: Array of RDkit Mol object
        n: Array of polymerization degree (int)

    Optional args:
        bond_length: Bond length of connecting bond (float, angstrome)
        random_rot: Dihedral angle around connecting bond is rotated randomly (boolean)
        dihedral: Dihedral angle around connecting bond (float, radian)
        dih_type: Definition type of dihedral angle (str; monomer, or bond)
        tacticity: atactic, syndiotactic, or isotactic
        atac_ratio: Chiral inversion ration for atactic polymer
        tac_array: Array of chiral inversion for atactic polymer (list of boolean)

    Returns:
        Rdkit Mol object
    """

    poly = None

    tac_list = []
    n_chi = 0
    for i, mol in enumerate(mols):
        c = True if check_chiral_monomer(mol) == 1 else False
        tac_list.append(c)
        if c: n_chi += n[i]

    chi = gen_chi_array(tacticity, n_chi, atac_ratio=atac_ratio, tac_array=tac_array)

    k = -1
    for i, mol in enumerate(mols):
        for j in range(n[i]):
            if tac_list[i]: k += 1
            if tac_list[i] and chi[k]:
                mol_c = calc.mirror_inversion_mol(mol, confId=confId)
            else:
                mol_c = utils.deepcopy_mol(mol)

            poly = connect_mols(poly, mol_c, bond_length=bond_length, dihedral=dihedral, random_rot=random_rot, dih_type=dih_type,
                                confId1=confId, res_name_2='RU%s' % const.pdb_id[i])

    return poly


def terminate_mols(poly, mol1, mol2=None, bond_length=1.5, dihedral=np.pi, random_rot=False, dih_type='monomer'):
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
    res_name_1 = 'TU0'
    res_name_2 = 'TU1'
        
    if Chem.MolToSmiles(mol1) == '[H][3H]' or Chem.MolToSmiles(mol1) == '[3H][H]':
        poly.GetAtomWithIdx(poly.GetIntProp('head_idx')).SetIsotope(1)
        poly.GetAtomWithIdx(poly.GetIntProp('head_idx')).GetPDBResidueInfo().SetResidueName(res_name_1)
        poly.GetAtomWithIdx(poly.GetIntProp('head_idx')).GetPDBResidueInfo().SetResidueNumber(1+poly.GetIntProp('num_units'))
        poly.SetIntProp('num_units', 1+poly.GetIntProp('num_units'))
    else:
        poly = connect_mols(mol1, poly, bond_length=bond_length, dihedral=dihedral, random_rot=random_rot, dih_type=dih_type,
                            res_name_1=res_name_1)

    if Chem.MolToSmiles(mol2) == '[H][3H]' or Chem.MolToSmiles(mol2) == '[3H][H]':
        poly.GetAtomWithIdx(poly.GetIntProp('tail_idx')).SetIsotope(1)
        poly.GetAtomWithIdx(poly.GetIntProp('tail_idx')).GetPDBResidueInfo().SetResidueName(res_name_2)
        poly.GetAtomWithIdx(poly.GetIntProp('tail_idx')).GetPDBResidueInfo().SetResidueNumber(1+poly.GetIntProp('num_units'))
        poly.SetIntProp('num_units', 1+poly.GetIntProp('num_units'))
    else:
        poly = connect_mols(poly, mol2, bond_length=bond_length, dihedral=dihedral, random_rot=random_rot, dih_type=dih_type,
                            res_name_2=res_name_2)

    set_terminal_idx(poly)
    
    return poly


def polymerize_rw(mol, n, headhead=False, confId=0, tacticity='atactic', atac_ratio=0.5, tac_array=None,
            retry=10, retry_step=100, dist_min=0.7, opt='rdkit', ff=None, work_dir=None, omp=1, mpi=1, gpu=0):
    """
    poly.polymerize_rw

    Polymerization of RDkit Mol object by random walk

    Args:
        mol: RDkit Mol object
        n: Polymerization degree (int)

    Optional args:
        headhead: Connect monomer unit by head-to-head
        tacticity: isotactic, syndiotactic, or atactic
        atac_ratio: Chiral inversion ration for atactic polymer
        tac_array: Array of chiral inversion for atactic polymer (list of boolean)
        retry: Number of retry for this function when generating unsuitable structure (int)
        retry_step: Number of retry for a random-walk step when generating unsuitable structure (int)
        dist_min: (float, angstrom)
        opt:
            lammps: Using short time MD of lammps
            rdkit: MMFF94 optimization by RDKit
            None: Dihedral angle around connecting bond is rotated randomly
        work_dir: Work directory path of LAMMPS (str, requiring when opt is LAMMPS)
        ff: Force field object (requiring when opt is LAMMPS)
        omp: Number of threads of OpenMP in LAMMPS (int)
        mpi: Number of MPI process in LAMMPS (int)
        gpu: Number of GPU in LAMMPS (int)

    Returns:
        Rdkit Mol object
    """

    mol_n = mol.GetNumAtoms()
    poly = None
    result = None

    check_chi = True
    if check_chiral_monomer(mol) > 1:
        check_chi = False
        utils.radon_print('Find multiple chiral centers in the mainchain of polymer repeating unit. The chirality control is turned off.', level=1)
    chi = gen_chi_array(tacticity, n, atac_ratio=atac_ratio, tac_array=tac_array)

    utils.radon_print('Start polymerize_rw.', level=1)
    retry_flag = False
    for i in tqdm(range(n), desc='[Polymerization]', disable=const.tqdm_disable):
        poly_copy = utils.deepcopy_mol(poly) if poly is not None else None

        if chi[i]:
            mol_c = calc.mirror_inversion_mol(mol, confId=confId)
        else:
            mol_c = utils.deepcopy_mol(mol)

        for r in range(retry_step):
            if headhead and i % 2 == 0:
                poly = connect_mols(poly, mol_c, tailtail=True, random_rot=True, confId1=confId, confId2=confId)
            else:
                poly = connect_mols(poly, mol_c, random_rot=True, confId1=confId, confId2=confId)

            if opt == 'lammps' and MD_avail:
                ff.ff_assign(poly)
                poly, _ = md.quick_rw(poly, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu)
            elif opt == 'rdkit':
                AllChem.MMFFOptimizeMolecule(poly, maxIters=50, confId=0)

            if i == 0: break

            if check_3d_structure(poly, dist_min=dist_min) and (not check_chi or check_tacticity(poly, tacticity, tac_array=tac_array)):
                break
            elif r < retry_step-1:
                poly = utils.deepcopy_mol(poly_copy) if poly_copy is not None else None
                utils.radon_print('Retry random walk step %03d' % (i+1))
            else:
                retry_flag = True
                utils.radon_print('Reached maximum number of retrying step in polymerize_rw.', level=1)

        if retry_flag and retry > 0: break

    if not check_3d_structure(poly, dist_min=dist_min) or (check_chi and not check_tacticity(poly, tacticity, tac_array=tac_array)):
        if retry <= 0:
            utils.radon_print('Reached maximum number of retrying polymerize_rw.', level=3)
        else:
            utils.radon_print('Retry polymerize_rw.', level=1)
            retry -= 1
            poly = polymerize_rw(mol, n, headhead=headhead, confId=confId, tacticity=tacticity,
                    atac_ratio=atac_ratio, tac_array=tac_array, retry=retry, retry_step=retry_step,
                    dist_min=dist_min, opt=opt, ff=ff, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu)

    return poly


def copolymerize_rw(mols, n, tacticity='atactic', atac_ratio=0.5, tac_array=None, retry=10, retry_step=100,
                dist_min=0.7, opt='rdkit', ff=None, work_dir=None, omp=1, mpi=1, gpu=0):
    """
    poly.copolymerize_rw

    Alternating co-polymerization of RDkit Mol object by random walk

    Args:
        mols: Array of RDkit Mol object
        n: Polymerization degree (int)

    Optional args:
        tacticity: isotactic, syndiotactic, or atactic
        atac_ratio: Chiral inversion ration for atactic polymer
        tac_array: Array of chiral inversion for atactic polymer (list of boolean)
        retry: Number of retry for this function when generating unsuitable structure (int)
        retry_step: Number of retry for a random-walk step when generating unsuitable structure (int)
        dist_min: (float, angstrom)
        opt:
            lammps: Using short time MD of lammps
            rdkit: MMFF94 optimization by RDKit
            None: Dihedral angle around connecting bond is rotated randomly
        work_dir: Work directory path of LAMMPS (str, requiring when opt is LAMMPS)
        ff: Force field object (requiring when opt is LAMMPS)
        omp: Number of threads of OpenMP in LAMMPS (int)
        mpi: Number of MPI process in LAMMPS (int)
        gpu: Number of GPU in LAMMPS (int)

    Returns:
        Rdkit Mol object
    """

    poly = None

    tac_list = []
    n_chi = 0
    check_chi = True
    for mol in mols:
        nc = check_chiral_monomer(mol)
        if nc > 1 and check_chi:
            check_chi = False
            utils.radon_print('Find multiple chiral centers in the mainchain of polymer repeating unit. The chirality control is turned off.', level=1)
        c = True if nc == 1 else False
        tac_list.append(c)
        if c: n_chi += 1

    chi = gen_chi_array(tacticity, n*n_chi, atac_ratio=atac_ratio, tac_array=tac_array)

    utils.radon_print('Start copolymerize_rw.', level=1)
    k = -1
    retry_flag = False
    for i in tqdm(range(n), desc='[Polymerization]', disable=const.tqdm_disable):
        for j, mol in enumerate(mols):
            mol_n = mol.GetNumAtoms()
            poly_copy = utils.deepcopy_mol(poly) if poly is not None else None

            if tac_list[j]: k += 1
            if tac_list[j] and chi[k]:
                mol_c = calc.mirror_inversion_mol(mol)
            else:
                mol_c = utils.deepcopy_mol(mol)

            for r in range(retry_step):
                poly = connect_mols(poly, mol_c, random_rot=True, res_name_2='RU%s' % const.pdb_id[j])

                if opt == 'lammps' and MD_avail:
                    ff.ff_assign(poly)
                    poly, _ = md.quick_rw(poly, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu)
                elif opt == 'rdkit':
                    AllChem.MMFFOptimizeMolecule(poly, maxIters=50, confId=0)
                
                if i == 0: break
            
                if check_3d_structure(poly, dist_min=dist_min) and (not check_chi or check_tacticity(poly, tacticity, tac_array=tac_array)):
                    break
                elif r < retry_step-1:
                    poly = utils.deepcopy_mol(poly_copy) if poly_copy is not None else None
                    utils.radon_print('Retry random walk step %03d' % (i+1))
                else:
                    retry_flag = True
                    utils.radon_print('Reached maximum number of retrying copolymerize_rw.', level=1)

            if retry_flag and retry > 0: break
        if retry_flag and retry > 0: break

    if not check_3d_structure(poly, dist_min=dist_min) or (check_chi and not check_tacticity(poly, tacticity, tac_array=tac_array)):
        if retry <= 0:
            utils.radon_print('Reached maximum number of retrying copolymerize_rw.', level=3)
        else:
            utils.radon_print('Retry copolymerize_rw.', level=1)
            retry -= 1
            poly = copolymerize_rw(mols, n, tacticity=tacticity, atac_ratio=atac_ratio,
                    tac_array=tac_array, retry=retry, retry_step=retry_step, dist_min=dist_min,
                    opt=opt, ff=ff, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu)

    return poly


def random_copolymerize_rw(mols, n, ratio, ratio_type='exact', tacticity='atactic', atac_ratio=0.5,
                retry=10, retry_step=100, dist_min=0.7, opt='rdkit', ff=None, work_dir=None, omp=1, mpi=1, gpu=0):
    """
    poly.random_copolymerize_rw

    Random co-polymerization of RDkit Mol object by random walk

    Args:
        mols: Array of RDkit Mol object
        n: Polymerization degree (int)
        ratio: Array of monomer ratio (float, sum=1.0)

    Optional args:
        ratio_type: exact or choice
        tacticity: isotactic, syndiotactic, or atactic
        atac_ratio: Chiral inversion ration for atactic polymer
        tac_array: Array of chiral inversion for atactic polymer (list of boolean)
        retry: Number of retry for this function when generating unsuitable structure (int)
        retry_step: Number of retry for a random-walk step when generating unsuitable structure (int)
        dist_min: (float, angstrom)
        opt:
            lammps: Using short time MD of lammps
            rdkit: MMFF94 optimization by RDKit
            None: Dihedral angle around connecting bond is rotated randomly
        work_dir: Work directory path of LAMMPS (str, requiring when opt is LAMMPS)
        ff: Force field object (requiring when opt is LAMMPS)
        omp: Number of threads of OpenMP in LAMMPS (int)
        mpi: Number of MPI process in LAMMPS (int)
        gpu: Number of GPU in LAMMPS (int)

    Returns:
        Rdkit Mol object
    """

    if len(mols) != len(ratio):
        utils.radon_print('Inconsistency length of mols and ratio', level=3)
        return None

    ratio = np.array(ratio) / np.sum(ratio)

    poly = None

    tac_list = []
    n_chi = 0
    check_chi = True
    for mol in mols:
        nc = check_chiral_monomer(mol)
        if nc > 1 and check_chi:
            check_chi = False
            utils.radon_print('Find multiple chiral centers in the mainchain of polymer repeating unit. The chirality control is turned off.', level=1)
        c = True if nc == 1 else False
        tac_list.append(c)
        if c: n_chi += 1

    chi = gen_chi_array(tacticity, n, atac_ratio=atac_ratio)

    if ratio_type == 'choice':
        mol_index = np.random.choice(a=list(range(len(mols))), size=n, p=ratio)
    elif ratio_type == 'exact':
        mol_index = []
        for i, r in enumerate(ratio):
            tmp = [i]*int(n*r+0.5)
            mol_index.extend(tmp)
        if len(mol_index) < n:
            d = int(n-len(mol_index))
            imax = np.argmax(ratio)
            tmp = [imax]*d
            mol_index.extend(tmp)
        elif len(mol_index) > n:
            d = int(len(mol_index)-n)
            imax = np.argmax(ratio)
            for i in range(d):
                mol_index.remove(imax)
        random.shuffle(mol_index)

    utils.radon_print('Start random_copolymerize_rw.', level=1)
    k = -1
    retry_flag =  False
    for i in tqdm(range(n), desc='[Polymerization]', disable=const.tqdm_disable):
        poly_copy = utils.deepcopy_mol(poly) if poly is not None else None
        mol = mols[mol_index[i]]

        if tac_list[mol_index[i]]: k += 1
        if tac_list[mol_index[i]] and chi[k]:
            mol_c = calc.mirror_inversion_mol(mol)
        else:
            mol_c = utils.deepcopy_mol(mol)

        for r in range(retry_step):
            poly = connect_mols(poly, mol_c, random_rot=True, res_name_2='RU%s' % const.pdb_id[mol_index[i]])

            if opt == 'lammps' and MD_avail:
                ff.ff_assign(poly)
                poly, _ = md.quick_rw(poly, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu)
            elif opt == 'rdkit':
                AllChem.MMFFOptimizeMolecule(poly, maxIters=50, confId=0)

            if i == 0: break
            
            if check_3d_structure(poly, dist_min=dist_min) and (not check_chi or check_tacticity(poly, tacticity)):
                break
            elif r < retry_step-1:
                poly = utils.deepcopy_mol(poly_copy) if poly_copy is not None else None
                utils.radon_print('Retry random walk step %03d' % (i+1))
            else:
                retry_flag = True
                utils.radon_print('Reached maximum number of retrying random_copolymerize_rw.', level=1)

        if retry_flag and retry > 0: break

    if not check_3d_structure(poly, dist_min=dist_min) or (check_chi and not check_tacticity(poly, tacticity)):
        if retry <= 0:
            utils.radon_print('Reached maximum number of retrying random_copolymerize_rw.', level=3)
        else:
            utils.radon_print('Retry random_copolymerize_rw.', level=1)
            retry -= 1
            poly = random_copolymerize_rw(mols, n, ratio, ratio_type=ratio_type, tacticity=tacticity,
                    atac_ratio=atac_ratio, retry=retry, retry_step=retry_step,
                    dist_min=dist_min, opt=opt, ff=ff, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu)

    return poly


def random_copolymerize_rw_mp(mols, n, ratio, ratio_type='exact', tacticity='atactic', atac_ratio=0.5,
                retry=10, retry_step=100, dist_min=0.7, opt='rdkit', ff=None, work_dir=None, omp=1, mpi=1, gpu=0, nchain=10, mp=10):

    for i in range(len(mols)): utils.picklable(mols[i])
    c = utils.picklable_const()
    args = [(mols, n, ratio, ratio_type, tacticity, atac_ratio, retry, retry_step, dist_min, opt, ff, work_dir, omp, mpi, gpu, c) for i in range(nchain)]

    with confu.ProcessPoolExecutor(max_workers=mp) as executor:
        results = executor.map(_random_copolymerize_rw_mp_worker, args)

    polys = [res for res in results]
    for i in range(len(polys)): utils.restore_picklable(polys[i])
    for i in range(len(mols)): utils.restore_picklable(mols[i])

    return polys


def _random_copolymerize_rw_mp_worker(args):
    (mols, n, ratio, ratio_type, tacticity, atac_ratio, retry, retry_step, dist_min, opt, ff, work_dir, omp, mpi, gpu, c) = args
    utils.restore_const(c)
    for i in range(len(mols)): utils.restore_picklable(mols[i])
    poly = random_copolymerize_rw(mols, n, ratio, ratio_type=ratio_type, tacticity=tacticity, atac_ratio=atac_ratio,
                retry=retry, retry_step=retry_step, dist_min=dist_min, opt=opt, ff=ff, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu)
    utils.picklable(poly)

    return poly


def block_copolymerize_rw(mols, n, tacticity='atactic', atac_ratio=0.5, tac_array=None,
            retry=10, retry_step=100, dist_min=0.7, opt='rdkit', ff=None, work_dir=None, omp=1, mpi=1, gpu=0):
    """
    poly.block_copolymerize_rw

    Block co-polymerization of RDkit Mol object by random walk

    Args:
        mols: Array of RDkit Mol object
        n: Array of polymerization degree (int)

    Optional args:
        tacticity: isotactic, syndiotactic, or atactic
        atac_ratio: Chiral inversion ration for atactic polymer
        tac_array: Array of chiral inversion for atactic polymer (list of boolean)
        retry: Number of retry for this function when generating unsuitable structure (int)
        retry_step: Number of retry for a random-walk step when generating unsuitable structure (int)
        dist_min: (float, angstrom)
        opt:
            lammps: Using short time MD of lammps
            rdkit: MMFF94 optimization by RDKit
            None: Dihedral angle around connecting bond is rotated randomly
        work_dir: Work directory path of LAMMPS (str, requiring when opt is LAMMPS)
        ff: Force field object (requiring when opt is LAMMPS)
        omp: Number of threads of OpenMP in LAMMPS (int)
        mpi: Number of MPI process in LAMMPS (int)
        gpu: Number of GPU in LAMMPS (int)

    Returns:
        Rdkit Mol object
    """

    poly = None

    tac_list = []
    n_chi = 0
    check_chi = True
    for i, mol in enumerate(mols):
        nc = check_chiral_monomer(mol)
        if nc > 1 and check_chi:
            check_chi = False
            utils.radon_print('Find multiple chiral centers in the mainchain of polymer repeating unit. The chirality control is turned off.', level=1)
        c = True if nc == 1 else False
        tac_list.append(c)
        if c: n_chi += n[i]

    chi = gen_chi_array(tacticity, n_chi, atac_ratio=atac_ratio, tac_array=tac_array)

    utils.radon_print('Start block_copolymerize_rw.', level=1)
    k = -1
    retry_flag = False
    for i, mol in enumerate(mols):
        for j in range(n[i]):
            poly_copy = utils.deepcopy_mol(poly) if poly is not None else None

            if tac_list[i]: k += 1
            if tac_list[i] and chi[k]:
                mol_c = calc.mirror_inversion_mol(mol)
            else:
                mol_c = utils.deepcopy_mol(mol)

            for r in range(retry_step):
                poly = connect_mols(poly, mol_c, random_rot=True, res_name_2='RU%s' % const.pdb_id[i])

                if opt == 'lammps' and MD_avail:
                    ff.ff_assign(poly)
                    poly, _ = md.quick_rw(poly, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu)
                elif opt == 'rdkit':
                    AllChem.MMFFOptimizeMolecule(poly, maxIters=50, confId=0)
                
                if i == 0 and j == 0: break
            
                if check_3d_structure(poly, dist_min=dist_min) and (not check_chi or check_tacticity(poly, tacticity, tac_array=tac_array)):
                    break
                elif r < retry_step-1:
                    poly = utils.deepcopy_mol(poly_copy) if poly_copy is not None else None
                    utils.radon_print('Retry random walk step %03d' % (i+1))
                else:
                    retry_flag = True
                    utils.radon_print('Reached maximum number of retrying block_copolymerize_rw.', level=1)

            if retry_flag and retry > 0: break
        if retry_flag and retry > 0: break

    if not check_3d_structure(poly, dist_min=dist_min) or (check_chi and not check_tacticity(poly, tacticity, tac_array=tac_array)):
        if retry <= 0:
            utils.radon_print('Reached maximum number of retrying block_copolymerize_rw.', level=3)
        else:
            utils.radon_print('Retry block_copolymerize_rw.', level=1)
            retry -= 1
            poly = block_copolymerize_rw(mols, n, tacticity=tacticity, atac_ratio=atac_ratio,
                    tac_array=tac_array, retry=retry, retry_step=retry_step, dist_min=dist_min,
                    opt=opt, ff=ff, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu)

    return poly


def terminate_rw(poly, mol1, mol2=None, retry_step=100, dist_min=0.7,
            opt='rdkit', ff=None, work_dir=None, omp=1, mpi=1, gpu=0):
    """
    poly.terminate_rw

    Termination of polymer of RDkit Mol object by random walk

    Args:
        poly: polymer (RDkit Mol object)
        mol1: terminated substitute at head (and tail) (RDkit Mol object)

    Optional args:
        mol2: terminated substitute at tail (RDkit Mol object)
        retry_step: Number of retry for a random-walk step when generating unsuitable structure (int)
        dist_min: (float, angstrom)
        opt:
            lammps: Using short time MD of lammps
            rdkit: MMFF94 optimization by RDKit
            None: Dihedral angle around connecting bond is rotated randomly
        work_dir: Work directory path of LAMMPS (str, requiring when opt is LAMMPS)
        ff: Force field object (requiring when opt is LAMMPS)
        omp: Number of threads of OpenMP in LAMMPS (int)
        mpi: Number of MPI process in LAMMPS (int)
        gpu: Number of GPU in LAMMPS (int)

    Returns:
        Rdkit Mol object
    """

    H2_flag = False

    if mol2 is None: mol2 = mol1
    res_name_1 = 'TU0'
    res_name_2 = 'TU1'

    mols = [mol1, mol2]

    utils.radon_print('Start terminate_rw.', level=1)
    for i, mol in enumerate(mols):
        poly_copy = utils.deepcopy_mol(poly)
        if Chem.MolToSmiles(mol) == '[H][3H]' or Chem.MolToSmiles(mol) == '[3H][H]':
            H2_flag = True
        else:
            H2_flag = False
            
        for r in range(retry_step):
            if i == 0:
                if H2_flag:
                    poly.GetAtomWithIdx(poly.GetIntProp('head_idx')).SetIsotope(1)
                    poly.GetAtomWithIdx(poly.GetIntProp('head_idx')).GetPDBResidueInfo().SetResidueName(res_name_1)
                    poly.GetAtomWithIdx(poly.GetIntProp('head_idx')).GetPDBResidueInfo().SetResidueNumber(1+poly.GetIntProp('num_units'))
                    poly.SetIntProp('num_units', 1+poly.GetIntProp('num_units'))
                else:
                    poly = connect_mols(mol, poly, random_rot=True, res_name_1=res_name_1)
            else:
                if H2_flag:
                    poly.GetAtomWithIdx(poly.GetIntProp('tail_idx')).SetIsotope(1)
                    poly.GetAtomWithIdx(poly.GetIntProp('tail_idx')).GetPDBResidueInfo().SetResidueName(res_name_2)
                    poly.GetAtomWithIdx(poly.GetIntProp('tail_idx')).GetPDBResidueInfo().SetResidueNumber(1+poly.GetIntProp('num_units'))
                    poly.SetIntProp('num_units', 1+poly.GetIntProp('num_units'))
                else:
                    poly = connect_mols(poly, mol, random_rot=True, res_name_2=res_name_2)

            if opt == 'lammps' and MD_avail:
                ff.ff_assign(poly)
                poly, _ = md.quick_rw(poly, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu)
            elif opt == 'rdkit':
                AllChem.MMFFOptimizeMolecule(poly, maxIters=50, confId=0)

            if check_3d_structure(poly, dist_min=dist_min):
                break
            elif r < retry_step-1:
                poly = utils.deepcopy_mol(poly_copy)
                utils.radon_print('Retry random walk step %03d' % (i+1))
            else:
                utils.radon_print('Reached maximum number of retrying termination.', level=2)

    set_terminal_idx(poly)

    return poly


def gen_chi_array(tacticity, n, atac_ratio=0.5, tac_array=None):

    chi = np.full(n, False)
    if tacticity == 'isotactic':
        pass

    elif tacticity == 'syndiotactic':
        chi[1::2] = True

    elif tacticity == 'atactic':
        chi[int(n*atac_ratio):] = True
        random.shuffle(chi)

    elif tacticity == 'manual':
        if tac_array is None:
            chi[int(n*atac_ratio):] = True
            random.shuffle(chi)
        elif len(tac_array) == n:
            chi = tac_array
        else:
            utils.radon_print('Length of tac_array is not equal to n.', level=3)

    else:
        utils.radon_print('%s is illegal input for tacticity.' % str(tacticity), level=3)

    return chi


def amorphous_cell(mol, n, cell=None, density=0.03, retry=10, retry_step=100, threshold=2.0, dec_rate=0.8):
    """
    poly.amorphous_cell

    Simple unit cell generator for single component system

    Args:
        mol: RDkit Mol object
        n: Number of molecules in the unit cell (int)

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

    if cell is None:
        cell_c = Chem.Mol()
    else:
        cell_c = utils.deepcopy_mol(cell)

    if density is None and hasattr(cell_c, 'cell'):
        xhi = cell_c.cell.xhi
        xlo = cell_c.cell.xlo
        yhi = cell_c.cell.yhi
        ylo = cell_c.cell.ylo
        zhi = cell_c.cell.zhi
        zlo = cell_c.cell.zlo
    else:
        xhi, xlo, yhi, ylo, zhi, zlo = calc_cell_length([mol, cell_c], [n, 1], density=density)
        setattr(cell_c, 'cell', utils.Cell(xhi, xlo, yhi, ylo, zhi, zlo))

    utils.radon_print('Start amorphous cell generation.', level=1)
    retry_flag = False
    for i in tqdm(range(n), desc='[Unit cell generation]', disable=const.tqdm_disable):
        for r in range(retry_step):
            # Random rotation and translation
            trans = np.array([
                np.random.uniform(xlo, xhi),
                np.random.uniform(ylo, yhi),
                np.random.uniform(zlo, zhi)
            ])
            rot = np.random.uniform(-np.pi, np.pi, 3)

            mol_coord = np.array(mol.GetConformer(0).GetPositions())
            mol_coord = calc.rotate_rod(mol_coord, np.array([1, 0, 0]), rot[0])
            mol_coord = calc.rotate_rod(mol_coord, np.array([0, 1, 0]), rot[1])
            mol_coord = calc.rotate_rod(mol_coord, np.array([0, 0, 1]), rot[2])
            mol_coord += trans - np.mean(mol_coord, axis=0)

            if cell_c.GetNumConformers() == 0: break

            if check_3d_structure_cell(cell_c, mol_coord, dist_min=threshold):
                break
            elif r < retry_step-1:
                utils.radon_print('Retry placing replicate in cell')
            else:
                retry_flag = True
                utils.radon_print('Reached maximum number of retrying step in amorphous_cell.', level=1)

        if retry_flag and retry > 0: break

        cell_n = cell_c.GetNumAtoms()

        # Add Mol to cell
        cell_c = combine_mols(cell_c, mol)

        # Set atomic coordinate
        for j in range(mol.GetNumAtoms()):
            cell_c.GetConformer(0).SetAtomPosition(
                cell_n+j,
                Geom.Point3D(mol_coord[j, 0], mol_coord[j, 1], mol_coord[j, 2])
            )

    if retry_flag:
        if retry <= 0:
            utils.radon_print('Reached maximum number of retrying amorphous_cell.', level=3)
        else:
            utils.radon_print('Retry amorphous_cell.', level=1)
            retry -= 1
            density *= dec_rate
            cell_c = amorphous_cell(mol, n, cell=cell, density=density, retry=retry,
                    retry_step=retry_step, threshold=threshold)

    return cell_c


def amorphous_mixture_cell(mols, n, cell=None, density=0.03, retry=10, retry_step=100, threshold=2.0, dec_rate=0.8):
    """
    poly.amorphous_mixture_cell

    Simple unit cell generator for mixture system

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

    if cell is None:
        cell_c = Chem.Mol()
    else:
        cell_c = utils.deepcopy_mol(cell)

    if density is None and hasattr(cell_c, 'cell'):
        xhi = cell_c.cell.xhi
        xlo = cell_c.cell.xlo
        yhi = cell_c.cell.yhi
        ylo = cell_c.cell.ylo
        zhi = cell_c.cell.zhi
        zlo = cell_c.cell.zlo
    else:
        xhi, xlo, yhi, ylo, zhi, zlo = calc_cell_length([*mols, cell_c], [*n, 1], density=density)
        setattr(cell_c, 'cell', utils.Cell(xhi, xlo, yhi, ylo, zhi, zlo))

    utils.radon_print('Start amorphous cell generation.', level=1)
    retry_flag = False
    for m, mol in enumerate(mols):
        for i in tqdm(range(n[m]), desc='[Unit cell generation %i/%i]' % (m+1, len(mols)), disable=const.tqdm_disable):
            for r in range(retry_step):
                # Random rotation and translation
                trans = np.array([
                    np.random.uniform(xlo, xhi),
                    np.random.uniform(ylo, yhi),
                    np.random.uniform(zlo, zhi)
                ])
                rot = np.random.uniform(-np.pi, np.pi, 3)

                mol_coord = np.array(mol.GetConformer(0).GetPositions())
                mol_coord = calc.rotate_rod(mol_coord, np.array([1, 0, 0]), rot[0])
                mol_coord = calc.rotate_rod(mol_coord, np.array([0, 1, 0]), rot[1])
                mol_coord = calc.rotate_rod(mol_coord, np.array([0, 0, 1]), rot[2])
                mol_coord += trans - np.mean(mol_coord, axis=0)

                if cell_c.GetNumConformers() == 0: break

                if check_3d_structure_cell(cell_c, mol_coord, dist_min=threshold):
                    break
                elif r < retry_step-1:
                    utils.radon_print('Retry placing replicate in cell')
                else:
                    retry_flag = True
                    utils.radon_print('Reached maximum number of retrying a step in amorphous_mixture_cell.', level=1)

            if retry_flag and retry > 0: break

            cell_n = cell_c.GetNumAtoms()

            # Add Mol to cell
            cell_c = combine_mols(cell_c, mol)

            # Set atomic coordinate
            for j in range(mol.GetNumAtoms()):
                cell_c.GetConformer(0).SetAtomPosition(
                    cell_n+j,
                    Geom.Point3D(mol_coord[j, 0], mol_coord[j, 1], mol_coord[j, 2])
                )
            
        if retry_flag and retry > 0: break

    if retry_flag:
        if retry <= 0:
            utils.radon_print('Reached maximum number of retrying amorphous_mixture_cell.', level=3)
        else:
            utils.radon_print('Retry amorphous_mixture_cell.', level=1)
            retry -= 1
            density *= dec_rate
            cell_c = amorphous_mixture_cell(mols, n, cell=cell, density=density, retry=retry,
                    retry_step=retry_step, threshold=threshold)

    return cell_c


def nematic_cell(mol, n, cell=None, density=0.1, retry=10, retry_step=100, threshold=2.0, dec_rate=0.8):
    """
    poly.nematic_cell

    Simple unit cell generator for single component system with nematic-like ordered structure for x axis

    Args:
        mol: RDkit Mol object
        n: Number of molecules in the unit cell (int)

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

    if cell is None:
        cell_c = Chem.Mol()
    else:
        cell_c = utils.deepcopy_mol(cell)

    # Alignment molecules
    Chem.rdMolTransforms.CanonicalizeConformer(mol.GetConformer(0), ignoreHs=False)
    mol_coord = np.array(mol.GetConformer(0).GetPositions())
 
    if density is None and hasattr(cell_c, 'cell'):
        xhi = cell_c.cell.xhi
        xlo = cell_c.cell.xlo
        yhi = cell_c.cell.yhi
        ylo = cell_c.cell.ylo
        zhi = cell_c.cell.zhi
        zlo = cell_c.cell.zlo
    else:
        xhi, xlo, yhi, ylo, zhi, zlo = calc_cell_length([mol, cell_c], [n, 1], density=density)
        setattr(cell_c, 'cell', utils.Cell(xhi, xlo, yhi, ylo, zhi, zlo))

    utils.radon_print('Start nematic-like cell generation.', level=1)
    retry_flag = False
    for i in tqdm(range(n), desc='[Unit cell generation]', disable=const.tqdm_disable):
        for r in range(retry_step):
            # Random translation
            trans = np.array([
                np.random.uniform(xlo, xhi),
                np.random.uniform(ylo, yhi),
                np.random.uniform(zlo, zhi)
            ])

            #Random rotation around x axis
            rot = np.random.uniform(-np.pi, np.pi)

            if i % 2 == 0:
                mol_coord = calc.rotate_rod(mol_coord, np.array([1, 0, 0]), rot) + trans
            else:
                mol_coord = calc.rotate_rod(mol_coord, np.array([1, 0, 0]), rot)
                mol_coord = calc.rotate_rod(mol_coord, np.array([0, 1, 0]), np.pi)
                mol_coord += trans

            if cell_c.GetNumConformers() == 0: break

            if check_3d_structure_cell(cell_c, mol_coord, dist_min=threshold):
                break
            elif r < retry_step-1:
                utils.radon_print('Retry placing replicate in cell')
            else:
                retry_flag = True
                utils.radon_print('Reached maximum number of retrying a step in nematic_cell.', level=1)

        if retry_flag and retry > 0: break

        cell_n = cell_c.GetNumAtoms()

        # Add Mol to cell
        cell_c = combine_mols(cell_c, mol)

        # Set atomic coordinate
        for j in range(mol.GetNumAtoms()):
            cell_c.GetConformer(0).SetAtomPosition(
                cell_n+j,
                Geom.Point3D(mol_coord[j, 0], mol_coord[j, 1], mol_coord[j, 2])
            )

    if retry_flag:
        if retry <= 0:
            utils.radon_print('Reached maximum number of retrying nematic_cell.', level=3)
        else:
            utils.radon_print('Retry nematic_cell.', level=1)
            retry -= 1
            density *= dec_rate
            cell_c = nematic_cell(mol, n, cell=cell, density=density, retry=retry,
                    retry_step=retry_step, threshold=threshold)

    return cell_c


def nematic_mixture_cell(mols, n, cell=None, density=0.1, retry=10, retry_step=100, threshold=2.0, dec_rate=0.8):
    """
    poly.nematic_mixture_cell

    Simple unit cell generator for mixture system with nematic-like ordered structure for x axis

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

    if cell is None:
        cell_c = Chem.Mol()
    else:
        cell_c = utils.deepcopy_mol(cell)

    # Alignment molecules
    for mol in mols:
        Chem.rdMolTransforms.CanonicalizeConformer(mol.GetConformer(0), ignoreHs=False)

    if density is None and hasattr(cell_c, 'cell'):
        xhi = cell_c.cell.xhi
        xlo = cell_c.cell.xlo
        yhi = cell_c.cell.yhi
        ylo = cell_c.cell.ylo
        zhi = cell_c.cell.zhi
        zlo = cell_c.cell.zlo
    else:
        xhi, xlo, yhi, ylo, zhi, zlo = calc_cell_length([*mols, cell_c], [*n, 1], density=density)
        setattr(cell_c, 'cell', utils.Cell(xhi, xlo, yhi, ylo, zhi, zlo))

    utils.radon_print('Start nematic-like cell generation.', level=1)
    retry_flag = False
    for m, mol in enumerate(mols):
        for i in tqdm(range(n[m]), desc='[Unit cell generation %i/%i]' % (m+1, len(mols)), disable=const.tqdm_disable):
            for r in range(retry_step):
                # Random translation
                trans = np.array([
                    np.random.uniform(xlo, xhi),
                    np.random.uniform(ylo, yhi),
                    np.random.uniform(zlo, zhi)
                ])

                #Random rotation around x axis
                rot = np.random.uniform(-np.pi, np.pi)

                mol_coord = np.array(mol.GetConformer(0).GetPositions())
                if i % 2 == 0:
                    mol_coord = calc.rotate_rod(mol_coord, np.array([1, 0, 0]), rot) + trans
                else:
                    mol_coord = calc.rotate_rod(mol_coord, np.array([1, 0, 0]), rot)
                    mol_coord = calc.rotate_rod(mol_coord, np.array([0, 1, 0]), np.pi)
                    mol_coord += trans

                if cell_c.GetNumConformers() == 0: break

                if check_3d_structure_cell(cell_c, mol_coord, dist_min=threshold):
                    break
                elif r < retry_step-1:
                    utils.radon_print('Retry placing replicate in cell')
                else:
                    retry_flag = True
                    utils.radon_print('Reached maximum number of retrying a step in nematic_mixture_cell.', level=1)

            if retry_flag and retry > 0: break

            cell_n = cell_c.GetNumAtoms()

            # Add Mol to cell
            cell_c = combine_mols(cell_c, mol)

            # Set atomic coordinate
            for j in range(mol.GetNumAtoms()):
                cell_c.GetConformer(0).SetAtomPosition(
                    cell_n+j,
                    Geom.Point3D(mol_coord[j, 0], mol_coord[j, 1], mol_coord[j, 2])
                )

        if retry_flag and retry > 0: break

    if retry_flag:
        if retry <= 0:
            utils.radon_print('Reached maximum number of retrying nematic_mixture_cell.', level=3)
        else:
            utils.radon_print('Retry nematic_mixture_cell.', level=1)
            retry -= 1
            density *= dec_rate
            cell_c = nematic_mixture_cell(mols, n, cell=cell, density=density, retry=retry,
                    retry_step=retry_step, threshold=threshold)

    return cell_c


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
        cell = amorphous_cell(mol, 1, cell=cell, density=None, retry=retry, threshold=threshold)

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
        cell = amorphous_cell(mols[0], 1, cell=cell, density=None, retry=retry, threshold=threshold)

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
        cell = amorphous_cell(mols[mol_index[j, 0]], 1, cell=cell, density=None, retry=retry, threshold=threshold)

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
    Chem.rdMolTransforms.CanonicalizeConformer(mol.GetConformer(confId), ignoreHs=False)
    mol1_coord = np.array(mol.GetConformer(confId).GetPositions())

    # Rotation mol2 to align link vectors and x-axis
    set_linker_flag(mol)
    center = mol1_coord[mol.GetIntProp('head_idx')]
    link_vec = mol1_coord[mol.GetIntProp('tail_idx')] - mol1_coord[mol.GetIntProp('head_idx')]
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
    cell = combine_mols(mol, mol)
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
        xcell = combine_mols(xcell, cell)
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


def check_3d_structure(mol, confId=0, dist_min=0.7, bond_s=2.7, bond_a=1.9, bond_d=1.8, bond_t=1.4, wrap=True):
    """
    poly.check_3d_structure

    Args:
        mol: RDkit Mol object

    Optional args:
        dist_min: Threshold of the minimum atom-atom distance (float, angstrom)
        bond_s: Threshold of the maximum single bond length (float, angstrom)
        bond_a: Threshold of the maximum aromatic bond length (float, angstrom)
        bond_d: Threshold of the maximum double bond length (float, angstrom)
        bond_t: Threshold of the maximum triple bond length (float, angstrom)
        wrap: Use wrapped coordinates (boolean)

    Returns:
        boolean
    """

    coord = np.array(mol.GetConformer(confId).GetPositions())
    if wrap and hasattr(mol, 'cell'):
        coord = calc.wrap(coord, mol.cell.xhi, mol.cell.xlo, mol.cell.yhi, mol.cell.ylo, mol.cell.zhi, mol.cell.zlo)

    dist_matrix = calc.distance_matrix(coord)
    dist_matrix = np.where(dist_matrix == 0, dist_min, dist_matrix)

    # Cheking bond length
    bond_l_c = True
    for b in mol.GetBonds():
        bond_l = dist_matrix[b.GetBeginAtom().GetIdx(), b.GetEndAtom().GetIdx()]
        if b.GetBondTypeAsDouble() == 1.0 and bond_l > bond_s:
            bond_l_c = False
            break
        elif b.GetBondTypeAsDouble() == 1.5 and bond_l > bond_a:
            bond_l_c = False
            break
        elif b.GetBondTypeAsDouble() == 2.0 and bond_l > bond_d:
            bond_l_c = False
            break
        elif b.GetBondTypeAsDouble() == 3.0 and bond_l > bond_t:
            bond_l_c = False
            break

    if dist_matrix.min() >= dist_min and bond_l_c:
        check = True
    else:
        check = False

    return check


def check_3d_structure_cell(cell, mol_coord, dist_min=2.0):
    cell_coord = np.array(cell.GetConformer(0).GetPositions())
    cell_wcoord = calc.wrap(cell_coord, cell.cell.xhi, cell.cell.xlo,
                    cell.cell.yhi, cell.cell.ylo, cell.cell.zhi, cell.cell.zlo)
    mol_wcoord = calc.wrap(mol_coord, cell.cell.xhi, cell.cell.xlo,
                    cell.cell.yhi, cell.cell.ylo, cell.cell.zhi, cell.cell.zlo)
    dist_matrix = calc.distance_matrix(cell_wcoord, mol_wcoord)
    if dist_matrix.min() > dist_min:
        return True
    else:
        return False


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


def set_linker_flag(mol, reverse=False):
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
        if (atom.GetSymbol() == "H" and atom.GetIsotope() == 3) or atom.GetSymbol() == "*":
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


def remove_linker_atoms(mol):
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
            if (atom.GetSymbol() == "H" and atom.GetIsotope() == 3) or atom.GetSymbol() == "*":
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
    ter = utils.mol_from_smiles('*C')
    mol_c = utils.deepcopy_mol(mol)
    mol_c = terminate_mols(mol_c, ter, random_rot=True)
    set_mainchain_flag(mol_c)
    for atom in mol_c.GetAtoms():
        if (int(atom.GetChiralTag()) == 1 or int(atom.GetChiralTag()) == 2) and atom.GetBoolProp('main_chain'):
            n_chiral += 1

    return n_chiral


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

    mol_c = utils.deepcopy_mol(mol)
    set_linker_flag(mol_c)
    if mol_c.GetIntProp('head_idx') >= 0 and mol_c.GetIntProp('tail_idx') >= 0:
        ter = utils.mol_from_smiles('*C')
        mol_c = terminate_mols(mol_c, ter, random_rot=True)
    set_mainchain_flag(mol_c)

    Chem.AssignStereochemistryFrom3D(mol_c, confId=confId)
    chiral_centers = np.array(Chem.FindMolChiralCenters(mol_c))

    if len(chiral_centers) == 0:
        return tac

    chiral_centers = [int(x) for x in chiral_centers[:, 0]]
    chiral_list = []
    for atom in mol_c.GetAtoms():
        if atom.GetBoolProp('main_chain') and atom.GetIdx() in chiral_centers:
            chiral_list.append(int(atom.GetChiralTag()))
    chiral_list = np.array(chiral_list)

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

    check = False

    tac = get_tacticity(mol, confId=confId)
    set_mainchain_flag(mol)

    if tac == 'none':
        check = True

    elif tac == 'isotactic' and tacticity == 'isotactic':
        check = True

    elif tac == 'syndiotactic' and tacticity == 'syndiotactic':
        check = True

    elif tacticity == 'atactic':
        if tac_array is None:
            check = True

        else:
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

    return check


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


def polymer_stats(mol, df=False, join=False):
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
    polymer_chains = Chem.GetMolFrags(mol, asMols=True)
    natom = np.array([chain.GetNumAtoms() for chain in polymer_chains])
    molweight = np.array([Descriptors.MolWt(chain) for chain in polymer_chains])

    poly_stats = {
        'n_mol': molcount,
        'n_atom': natom if not df and not join else '/'.join([str(n) for n in natom]),
        'n_atom_mean': np.mean(natom),
        'n_atom_var': np.var(natom),
        'mol_weight': molweight if not df and not join else '/'.join([str(n) for n in molweight]),
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

    Returns:
        SMILES or RDKit Mol object
    """

    mol = polymerize_MolFromSmiles(smiles, n=n, terminal='*')
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


def substruct_match_smiles_list(smiles, smi_series, mp=None, boolean=False):
    
    if mp is None:
        mp = utils.cpu_count()

    c = utils.picklable_const()
    args = [[smi, smiles, c] for smi in smi_series]

    with confu.ProcessPoolExecutor(max_workers=mp) as executor:
        results = executor.map(_substruct_match_smiles_worker, args)
        res = [r for r in results]
    
    if boolean:
        return res
    else:
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
        utils.radon_print('Illegal number of connecting points in SMILES. %s' % smiles, level=2)
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

