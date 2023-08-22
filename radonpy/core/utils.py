#  Copyright (c) 2023. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# ******************************************************************************
# core.utils module
# ******************************************************************************

import os
import re
import psutil
from copy import deepcopy
from itertools import permutations
import numpy as np
import pickle
import json
from rdkit import Chem
from rdkit.Chem import AllChem
from . import const
from ..ff import ff_class

__version__ = '0.3.0b3'


class Angle():
    """
        utils.Angle() object
    """
    def __init__(self, a, b, c, ff):
        self.a = a
        self.b = b
        self.c = c
        self.ff = ff
    
    def to_dict(self):
        dic = {
            'a': int(self.a),
            'b': int(self.b),
            'c': int(self.c),
            'ff': self.ff.to_dict()
        }
        return dic


class Dihedral():
    """
        utils.Dihedral() object
    """
    def __init__(self, a, b, c, d, ff):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.ff = ff

    def to_dict(self):
        dic = {
            'a': int(self.a),
            'b': int(self.b),
            'c': int(self.c),
            'd': int(self.d),
            'ff': self.ff.to_dict()
        }
        return dic


class Improper():
    """
        utils.Improper() object
    """
    def __init__(self, a, b, c, d, ff):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.ff = ff

    def to_dict(self):
        dic = {
            'a': int(self.a),
            'b': int(self.b),
            'c': int(self.c),
            'd': int(self.d),
            'ff': self.ff.to_dict()
        }
        return dic

        
class Cell():
    def __init__(self, xhi, xlo, yhi, ylo, zhi, zlo):
        self.xhi = xhi
        self.xlo = xlo
        self.yhi = yhi
        self.ylo = ylo
        self.zhi = zhi
        self.zlo = zlo
        self.dx = xhi-xlo
        self.dy = yhi-ylo
        self.dz = zhi-zlo
        self.volume = self.dx * self.dy * self.dz

    def to_dict(self):
        dic = {
            'xhi': float(self.xhi),
            'xlo': float(self.xlo),
            'yhi': float(self.yhi),
            'ylo': float(self.ylo),
            'zhi': float(self.zhi),
            'zlo': float(self.zlo),
        }
        return dic


class RadonPyError(Exception):
    pass


def radon_print(text, level=0):
    if level == 0:
        text = 'RadonPy debug info: '+text
    elif level == 1:
        text = 'RadonPy info: '+text
    elif level == 2:
        text = 'RadonPy warning: '+text
    elif level == 3:
        raise RadonPyError(text)

    if level >= const.print_level or const.debug:
        print(text, flush=True)


def set_mol_id(mol, pdb=True):
    """
    utils.set_mol_id

    Set molecular ID

    Args:
        mol: RDkit Mol object

    Optional args:
        pdb: Update the ChainId of PDB (boolean)

    Returns:
        Rdkit Mol object
    """

    molid = 1

    # Clear mol_id
    for atom in mol.GetAtoms():
        atom.SetIntProp('mol_id', 0)

    def recursive_set_mol_id(atom, molid):
        for na in atom.GetNeighbors():
            if na.GetIntProp('mol_id') == 0:
                na.SetIntProp('mol_id', molid)
                if pdb and na.GetPDBResidueInfo() is not None:
                    if molid <= len(const.pdb_id):
                        na.GetPDBResidueInfo().SetChainId(const.pdb_id[molid-1])
                recursive_set_mol_id(na, molid)

    for atom in mol.GetAtoms():
        if atom.GetIntProp('mol_id') == 0:
            atom.SetIntProp('mol_id', molid)
            if pdb and atom.GetPDBResidueInfo() is not None:
                if molid <= len(const.pdb_id):
                    atom.GetPDBResidueInfo().SetChainId(const.pdb_id[molid-1])
            recursive_set_mol_id(atom, molid)
            molid += 1

    return mol


def count_mols(mol):
    """
    utils.count_mols

    Count number of molecules

    Args:
        mol: RDkit Mol object

    Returns:
        Number of molecules (int)
    """

    fragments = Chem.GetMolFrags(mol, asMols=True)
    return len(fragments)


def remove_atom(mol, idx, angle_fix=False):
    """
    utils.remove_atom

    Remove a specific atom from RDkit Mol object

    Args:
        mol: RDkit Mol object
        idx: Atom index of removing atom in RDkit Mol object

    Options:
        angle_fix: Fix information of bond angles, dihedral angles, and improper angles. (boolean)

    Returns:
        RDkit Mol object
    """

    angles_copy = {}
    dihedrals_copy = {}
    impropers_copy = {}
    cell_copy = mol.cell if hasattr(mol, 'cell') else None

    if angle_fix:
        if hasattr(mol, 'impropers'):
            for imp in mol.impropers.values():
                imp_idx = {imp.a, imp.b, imp.c, imp.d}
                if idx in imp_idx:
                    continue

                if max(imp_idx) < idx:
                    key = '%i,%i,%i,%i' % (imp.a, imp.b, imp.c, imp.d)
                    impropers_copy[key] = imp
                else:
                    idx_a = imp.a if imp.a < idx else imp.a-1
                    idx_b = imp.b if imp.b < idx else imp.b-1
                    idx_c = imp.c if imp.c < idx else imp.c-1
                    idx_d = imp.d if imp.d < idx else imp.d-1
                    key = '%i,%i,%i,%i' % (idx_a, idx_b, idx_c, idx_d)
                    impropers_copy[key] = Improper(
                                                a=idx_a,
                                                b=idx_b,
                                                c=idx_c,
                                                d=idx_d,
                                                ff=imp.ff
                                            )

        if hasattr(mol, 'dihedrals'):
            for dih in mol.dihedrals.values():
                dih_idx = {dih.a, dih.b, dih.c, dih.d}
                if idx in dih_idx:
                    continue

                if max(dih_idx) < idx:
                    key = '%i,%i,%i,%i' % (dih.a, dih.b, dih.c, dih.d)
                    dihedrals_copy.append(dih)
                else:
                    idx_a = dih.a if dih.a < idx else dih.a-1
                    idx_b = dih.b if dih.b < idx else dih.b-1
                    idx_c = dih.c if dih.c < idx else dih.c-1
                    idx_d = dih.d if dih.d < idx else dih.d-1
                    key = '%i,%i,%i,%i' % (idx_a, idx_b, idx_c, idx_d)
                    dihedrals_copy[key] = Dihedral(
                                                a=idx_a,
                                                b=idx_b,
                                                c=idx_c,
                                                d=idx_d,
                                                ff=dih.ff
                                            )

        if hasattr(mol, 'angles'):
            for angle in mol.angles.values():
                ang_idx = {angle.a, angle.b, angle.c}
                if idx in ang_idx:
                    continue

                if max(ang_idx) < idx:
                    key = '%i,%i,%i,%i' % (ang.a, ang.b, ang.c, ang.d)
                    angles_copy[key] = angle
                else:
                    idx_a = angle.a if angle.a < idx else angle.a-1
                    idx_b = angle.b if angle.b < idx else angle.b-1
                    idx_c = angle.c if angle.c < idx else angle.c-1
                    key = '%i,%i,%i,%i' % (idx_a, idx_b, idx_c)
                    angles_copy[key] = Angle(
                                            a=idx_a,
                                            b=idx_b,
                                            c=idx_c,
                                            ff=angle.ff
                                        )

    rwmol = Chem.RWMol(mol)
    for pb in mol.GetAtomWithIdx(idx).GetNeighbors():
        rwmol.RemoveBond(idx, pb.GetIdx())

    rwmol.RemoveAtom(idx)

    mol = rwmol.GetMol()
    setattr(mol, 'angles', angles_copy)
    setattr(mol, 'dihedrals', dihedrals_copy)
    setattr(mol, 'impropers', impropers_copy)
    if cell_copy is not None: setattr(mol, 'cell', cell_copy)

    return mol


def add_bond(mol, idx1, idx2, order=Chem.rdchem.BondType.SINGLE):
    """
    utils.add_bond

    Add a new bond in RDkit Mol object

    Args:
        mol: RDkit Mol object
        idx1, idx2: Atom index adding a new bond (int)
        order: bond order (RDkit BondType object, ex. Chem.rdchem.BondType.SINGLE)

    Returns:
        RDkit Mol object
    """

    # Copy the extended attributes
    #angles_copy = mol.angles.copy() if hasattr(mol, 'angles') else {}
    #dihedrals_copy = mol.dihedrals.copy() if hasattr(mol, 'dihedrals') else {}
    #impropers_copy = mol.impropers.copy() if hasattr(mol, 'impropers') else {}
    cell_copy = mol.cell if hasattr(mol, 'cell') else None

    rwmol = Chem.RWMol(mol)
    rwmol.AddBond(idx1, idx2, order=order)
    mol = rwmol.GetMol()

    setattr(mol, 'angles', {})
    setattr(mol, 'dihedrals', {})
    setattr(mol, 'impropers', {})
    if cell_copy is not None: setattr(mol, 'cell', cell_copy)

    return mol


def remove_bond(mol, idx1, idx2):
    """
    utils.remove_bond

    Remove a specific bond in RDkit Mol object

    Args:
        mol: RDkit Mol object
        idx1, idx2: Atom index removing a specific bond (int)

    Returns:
        RDkit Mol object
    """

    # Copy the extended attributes
    #angles_copy = mol.angles.copy() if hasattr(mol, 'angles') else {}
    #dihedrals_copy = mol.dihedrals.copy() if hasattr(mol, 'dihedrals') else {}
    #impropers_copy = mol.impropers.copy() if hasattr(mol, 'impropers') else {}
    cell_copy = mol.cell if hasattr(mol, 'cell') else None

    rwmol = Chem.RWMol(mol)
    rwmol.RemoveBond(idx1, idx2)
    mol = rwmol.GetMol()

    setattr(mol, 'angles', {})
    setattr(mol, 'dihedrals', {})
    setattr(mol, 'impropers', {})
    if cell_copy is not None: setattr(mol, 'cell', cell_copy)

    return mol


def add_angle(mol, a, b, c, ff=None):
    """
    utils.add_angle

    Add a new angle in RDkit Mol object

    Args:
        mol: RDkit Mol object
        a, b, c: Atom index adding a new angle (int)

    Returns:
        boolean
    """

    if not hasattr(mol, 'angles'):
        setattr(mol, 'angles', {})

    key = '%i,%i,%i' % (a, b, c)
    mol.angles[key] = Angle(a=a, b=b, c=c, ff=ff)

    return True


def remove_angle(mol, a, b, c):
    """
    utils.remove_angle

    Remove a specific angle in RDkit Mol object

    Args:
        mol: RDkit Mol object
        a, b, c: Atom index removing a specific angle (int)

    Returns:
        boolean
    """

    if not hasattr(mol, 'angles'):
        return False

    # for i, angle in enumerate(mol.angles:
    #     if ((angle.a == a and angle.b == b and angle.c == c) or
    #         (angle.c == a and angle.b == b and angle.a == c)):
    #         del mol.angles[i]
    #         break

    key1 = '%i,%i,%i' % (a, b, c)
    key2 = '%i,%i,%i' % (c, b, a)
    if key1 in mol.angles:
        del mol.angles[key1]
    elif key2 in mol.angles:
        del mol.angles[key2]

    return True


def add_dihedral(mol, a, b, c, d, ff=None):
    """
    utils.add_dihedral

    Add a new dihedral in RDkit Mol object

    Args:
        mol: RDkit Mol object
        a, b, c, d: Atom index adding a new dihedral (int)

    Returns:
        boolean
    """

    if not hasattr(mol, 'dihedrals'):
        setattr(mol, 'dihedrals', {})

    key = '%i,%i,%i,%i' % (a, b, c, d)
    mol.dihedrals[key] = Dihedral(a=a, b=b, c=c, d=d, ff=ff)

    return True
    

def remove_dihedral(mol, a, b, c, d):
    """
    utils.remove_dihedral

    Remove a specific dihedral in RDkit Mol object

    Args:
        mol: RDkit Mol object
        a, b, c: Atom index removing a specific dihedral (int)

    Returns:
        boolean
    """

    if not hasattr(mol, 'dihedrals'):
        return False

    # for i, dihedral in enumerate(mol.dihedrals):
    #     if ((dihedral.a == a and dihedral.b == b and dihedral.c == c and dihedral.d == d) or
    #         (dihedral.d == a and dihedral.c == b and dihedral.b == c and dihedral.a == d)):
    #         del mol.dihedrals[i]
    #         break

    key1 = '%i,%i,%i,%i' % (a, b, c, d)
    key2 = '%i,%i,%i,%i' % (d, c, b, a)
    if key1 in mol.dihedrals:
        del mol.dihedrals[key1]
    elif key2 in mol.dihedrals:
        del mol.dihedrals[key2]

    return True


def add_improper(mol, a, b, c, d, ff=None):
    """
    utils.add_improper

    Add a new imploper in RDkit Mol object

    Args:
        mol: RDkit Mol object
        a, b, c, d: Atom index adding a new imploper (int)

    Returns:
        boolean
    """

    if not hasattr(mol, 'impropers'):
        setattr(mol, 'impropers', {})

    key = '%i,%i,%i,%i' % (a, b, c, d)
    mol.impropers[key] = Improper(a=a, b=b, c=c, d=d, ff=ff)

    return True
    

def remove_improper(mol, a, b, c, d):
    """
    utils.remove_improper

    Remove a specific improper in RDkit Mol object

    Args:
        mol: RDkit Mol object
        a, b, c: Atom index removing a specific improper (int)

    Returns:
        boolean
    """

    if not hasattr(mol, 'impropers'):
        return False

    # match = False
    # for i, improper in enumerate(mol.impropers):
    #     if improper.a == a:
    #         for perm in permutations([b, c, d], 3):
    #             if improper.b == perm[0] and improper.c == perm[1] and improper.d == perm[2]:
    #                 del mol.impropers[i]
    #                 match = True
    #                 break
    #         if match: break

    for perm in permutations([b, c, d], 3):
        key = '%i,%i,%i,%i' % (a, perm[0], perm[1], perm[2])
        if key in mol.impropers:
            del mol.impropers[key]
            break

    return True


def MolToPDBBlock(mol, confId=0):
    """
    utils.MolToPDBBlock

    Convert RDKit Mol object to PDB block

    Args:
        mol: RDkit Mol object

    Optional args:
        confId: Target conformer ID (int)

    Returns:
        PDB block (str, array)
    """

    coord = mol.GetConformer(confId).GetPositions()
    PDBBlock = ['TITLE    pdb written using RadonPy']
    conect = []
    serial = 0
    ter = 0
    chainid_pre = 1
    chainid_pdb_pre = ''

    for i, atom in enumerate(mol.GetAtoms()):
        serial += 1
        resinfo = atom.GetPDBResidueInfo()
        if resinfo is None: return None

        chainid_pdb = resinfo.GetChainId()
        chainid = atom.GetIntProp('mol_id')
        if chainid != chainid_pre:
            if chainid_pdb_pre:
                PDBBlock.append('TER   %5i      %3s %1s%4i%1s' % (serial, resname, chainid_pdb_pre, resnum, icode))
            else:
                PDBBlock.append('TER   %5i      %3s %1s%4i%1s' % (serial, resname, '*', resnum, icode))
            ter += 1
            serial += 1

        record = 'HETATM' if resinfo.GetIsHeteroAtom() else 'ATOM  '
        name = atom.GetProp('ff_type') if atom.HasProp('ff_type') else atom.GetSymbol()
        altLoc = resinfo.GetAltLoc()
        resname = resinfo.GetResidueName()
        resnum = resinfo.GetResidueNumber()
        icode = resinfo.GetInsertionCode()
        x = coord[i][0]
        y = coord[i][1]
        z = coord[i][2]
        occ = resinfo.GetOccupancy() if resinfo.GetOccupancy() else 1.0
        tempf = resinfo.GetTempFactor() if resinfo.GetTempFactor() else 0.0

        if chainid_pdb:
            line = '%-6s%5i %4s%1s%3s %1s%4i%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s' % (
                    record, serial, name, altLoc, resname, chainid_pdb, resnum, icode, x, y, z, occ, tempf, atom.GetSymbol())
        else:
            line = '%-6s%5i %4s%1s%3s %1s%4i%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s' % (
                    record, serial, name, altLoc, resname, '*', resnum, icode, x, y, z, occ, tempf, atom.GetSymbol())

        PDBBlock.append(line)

        chainid_pre = chainid
        chainid_pdb_pre = chainid_pdb

        if len(atom.GetNeighbors()) > 0:
            flag = False
            conect_line = 'CONECT%5i' % (serial)
            for na in atom.GetNeighbors():
                if atom.GetIdx() < na.GetIdx():
                    conect_line += '%5i' % (na.GetIdx()+1+ter)
                    flag = True
            if flag:
                conect.append(conect_line)

    PDBBlock.append('TER   %5i      %3s %1s%4i%1s' % (serial+1, resname, chainid_pre, resnum, icode))
    PDBBlock.extend(conect)
    PDBBlock.append('END')

    return PDBBlock


def MolToPDBFile(mol, filename, confId=0):
    """
    utils.MolToPDBFile

    Convert RDKit Mol object to PDB file

    Args:
        mol: RDkit Mol object
        filename: Output pdb filename (str)

    Optional args:
        confId: Target conformer ID (int)

    Returns:
        Success or fail (boolean)
    """

    mol = set_mol_id(mol)
    PDBBlock = MolToPDBBlock(mol, confId=confId)
    if PDBBlock is None: return False

    with open(filename, 'w') as fh:
        fh.write('\n'.join(PDBBlock)+'\n')
        fh.flush()
        if hasattr(os, 'fdatasync'):
            os.fdatasync(fh.fileno())
        else:
            os.fsync(fh.fileno())

    return True


def StructureFromXYZFile(filename):
    with open(filename, 'r') as fh:
        lines = [s.strip() for s in fh.readlines()]

    strucs = []
    struc = []
    t_flag = False
    n_flag = False
    n_atom = 0
    c_atom = 0
    for line in lines:
        if not n_flag:
            if line.isdecimal():
                n_atom = line
                n_flag = True
        elif not t_flag:
            t_flag = True
            continue
        else:
            c_atom += 1
            element, x, y, z = line.split()
            struc.append([element, x, y, z])
            if c_atom >= n_atom:
                strucs.append(struc)
                t_flag = False
                n_flag = False
                n_atom = 0
                c_atom = 0

    return strucs


def MolToExXYZBlock(mol, confId=0):

    XYZBlock = Chem.MolToXYZBlock(mol, confId=confId)
    XYZBlock = XYZBlock.split('\n')
    if mol.GetConformer(confId).HasProp('cell_xhi'):
        conf = mol.GetConformer(confId)
        cell_line = 'Lattice=\"%.4f 0.0 0.0 0.0 %.4f 0.0 0.0 0.0 %.4f\"' % (
            conf.GetDoubleProp('cell_dx'), conf.GetDoubleProp('cell_dy'), conf.GetDoubleProp('cell_dz'))
        XYZBlock[1] = cell_line
    elif hasattr(mol, 'cell'):
        cell_line = 'Lattice=\"%.4f 0.0 0.0 0.0 %.4f 0.0 0.0 0.0 %.4f\"' % (mol.cell.dx, mol.cell.dy, mol.cell.dz)
        XYZBlock[1] = cell_line

    return XYZBlock


def MolToExXYZFile(mol, filename, confId=0):

    XYZBlock = MolToExXYZBlock(mol, confId=confId)
    if XYZBlock is None: return False
    with open(filename, 'w') as fh:
        fh.write('\n'.join(XYZBlock)+'\n')
        fh.flush()
        if hasattr(os, 'fdatasync'):
            os.fdatasync(fh.fileno())
        else:
            os.fsync(fh.fileno())

    return True


def MolToJSON(mol, file, useRDKitExtensions=False):
    json_dict = MolToJSON_dict(mol, useRDKitExtensions=useRDKitExtensions)
    with open(file, mode='w') as f:
        json.dump(json_dict, f, indent=2)


def MolToJSON_dict(mol, useRDKitExtensions=False):
    Chem.SanitizeMol(mol)
    if hasattr(Chem.rdMolInterchange, 'JSONWriteParameters'):
        params = Chem.rdMolInterchange.JSONWriteParameters()
        params.useRDKitExtensions = useRDKitExtensions
        json_str = Chem.rdMolInterchange.MolToJSON(mol, params=params)
    else:
        json_str = Chem.rdMolInterchange.MolToJSON(mol)
    json_dict = json.loads(json_str)
    
    radonpy_ext = {
        'name': 'radonpy_extention',
        "formatVersion": 1,
        'lib_version': __version__,
    }

    atom_prop = []
    for a in mol.GetAtoms():
        atom_data = {}
        # FF on atoms
        if a.HasProp('ff_type'):
            atom_data['ff_type'] = a.GetProp('ff_type')
        if a.HasProp('ff_epsilon'):
            atom_data['ff_epsilon'] = a.GetDoubleProp('ff_epsilon')
        if a.HasProp('ff_sigma'):
            atom_data['ff_sigma'] = a.GetDoubleProp('ff_sigma')

        # charge
        if a.HasProp('AtomicCharge'):
            atom_data['AtomicCharge'] = a.GetDoubleProp('AtomicCharge')
        if a.HasProp('RESP'):
            atom_data['RESP'] = a.GetDoubleProp('RESP')
        if a.HasProp('ESP'):
            atom_data['ESP'] = a.GetDoubleProp('ESP')
        if a.HasProp('Mulliken'):
            atom_data['Mulliken'] = a.GetDoubleProp('Mulliken')
        if a.HasProp('Lowdin'):
            atom_data['Lowdin'] = a.GetDoubleProp('Lowdin')
        if a.HasProp('_GasteigerCharge'):
            atom_data['_GasteigerCharge'] = a.GetProp('_GasteigerCharge')
    
        # velocity
        if a.HasProp('vx'):
            atom_data['vx'] = a.GetDoubleProp('vx')
            atom_data['vy'] = a.GetDoubleProp('vy')
            atom_data['vz'] = a.GetDoubleProp('vz')

        # others
        atom_data['isotope'] = a.GetIsotope()
        if a.HasProp('mol_id'):
            atom_data['mol_id'] = a.GetIntProp('mol_id')

        # PDB
        resinfo = a.GetPDBResidueInfo()
        if resinfo is not None:
            atom_data['ResidueName'] = resinfo.GetResidueName()
            atom_data['ResidueNumber'] = resinfo.GetResidueNumber()

        atom_prop.append(atom_data)
    radonpy_ext['atoms'] = atom_prop


    bond_prop = []
    for b in mol.GetBonds():
        bond_data = {}
        # FF on bonds
        if b.HasProp('ff_type'):
            bond_data['ff_type'] = b.GetProp('ff_type')
        if b.HasProp('ff_k'):
            bond_data['ff_k'] = b.GetDoubleProp('ff_k')
        if b.HasProp('ff_r0'):
            bond_data['ff_r0'] = b.GetDoubleProp('ff_r0')

        bond_prop.append(bond_data)
    radonpy_ext['bonds'] = bond_prop

 
    # angle
    if hasattr(mol, 'angles'):
        # if len(mol.angles) > 0 and hasattr(mol.angles[list(mol.angles.keys())[0]], 'to_dict'):
            angle_prop = [ang.to_dict() for key, ang in mol.angles.items()]
            radonpy_ext['angles'] = angle_prop
        # else:
        #     angle_prop = []
        #     for key, ang in mol.angles.items():
        #         dic = {
        #             'a': ang.a,
        #             'b': ang.b,
        #             'c': ang.c,
        #             'ff': {
        #                 'ff_type': ang.ff.type,
        #                 'k': ang.ff.k,
        #                 'theta0': ang.ff.theta0,
        #             }
        #         }
        #         angle_prop.append(dic)
    else:
        angle_prop = []
    
    # dihedral
    if hasattr(mol, 'dihedrals'):
        # if len(mol.dihedrals) > 0 and hasattr(mol.dihedrals[list(mol.dihedrals.keys())[0]], 'to_dict'):
            dihedral_prop = [dih.to_dict() for key, dih in mol.dihedrals.items()]
            radonpy_ext['dihedrals'] = dihedral_prop
        # else:
        #     dihedral_prop = []
        #     for key, dih in mol.dihedrals.items():
        #         dic = {
        #             'a': dih.a,
        #             'b': dih.b,
        #             'c': dih.c,
        #             'd': dih.d,
        #             'ff': {
        #                 'ff_type': dih.ff.type,
        #                 'k': dih.ff.k,
        #                 'd0': dih.ff.d0,
        #                 'm': dih.ff.m,
        #                 'n': dih.ff.n,
        #             }
        #         }
        #         dihedral_prop.append(dic)
    else:
        dihedral_prop = []

    # improper
    if hasattr(mol, 'impropers'):
        # if len(mol.impropers) > 0 and hasattr(mol.impropers[list(mol.impropers.keys())[0]], 'to_dict'):
            improper_prop = [imp.to_dict() for key, imp in mol.impropers.items()]
            radonpy_ext['impropers'] = improper_prop
        # else:
        #     improper_prop = []
        #     for key, imp in mol.impropers.items():
        #         dic = {
        #             'a': imp.a,
        #             'b': imp.b,
        #             'c': imp.c,
        #             'd': imp.d,
        #             'ff': {
        #                 'ff_type': imp.ff.type,
        #                 'k': imp.ff.k,
        #                 'd0': imp.ff.d0,
        #                 'n': imp.ff.n,
        #             }
        #         }
        #         improper_prop.append(dic)
    else:
        improper_prop = []

    # cell
    if hasattr(mol, 'cell'):
        # if hasattr(mol.cell, 'to_dict'):
            cell_prop = mol.cell.to_dict()
            radonpy_ext['cell'] = cell_prop
        # else:
        #     cell_prop = {
        #         'xhi': mol.cell.xhi,
        #         'xlo': mol.cell.xlo,
        #         'yhi': mol.cell.yhi,
        #         'ylo': mol.cell.ylo,
        #         'zhi': mol.cell.zhi,
        #         'zlo': mol.cell.zlo,
        #     }

    json_dict['molecules'][0]['extensions'].append(radonpy_ext)

    return json_dict


def JSONToMol(file):
    with open(file, mode='r') as f:
        json_dict = json.load(f)

    radonpy_ext = None
    for ext in json_dict['molecules'][0]['extensions']:
        if 'name' in ext and ext['name'] == 'radonpy_extention':
            radonpy_ext = ext
    if radonpy_ext is None:
        radon_print('RadonPy extention data was not found in JSON file.', level=3)

    mol = Chem.rdMolInterchange.JSONToMols(json.dumps(json_dict))[0]
    Chem.SanitizeMol(mol)

    for i, a in enumerate(mol.GetAtoms()):
        atom_data = radonpy_ext['atoms'][i]

        # FF on atoms
        if 'ff_type' in atom_data:
            a.SetProp('ff_type', str(atom_data['ff_type']))
        if 'ff_epsilon' in atom_data:
            a.SetDoubleProp('ff_epsilon', float(atom_data['ff_epsilon']))
        if 'ff_sigma' in atom_data:
            a.SetDoubleProp('ff_sigma', float(atom_data['ff_sigma']))

        # charge
        if 'AtomicCharge' in atom_data:
            a.SetDoubleProp('AtomicCharge', float(atom_data['AtomicCharge']))
        if 'RESP' in atom_data:
            a.SetDoubleProp('RESP', float(atom_data['RESP']))
        if 'ESP' in atom_data:
            a.SetDoubleProp('ESP', float(atom_data['ESP']))
        if 'Mulliken' in atom_data:
            a.SetDoubleProp('Mulliken', float(atom_data['Mulliken']))
        if 'Lowdin' in atom_data:
            a.SetDoubleProp('Lowdin', float(atom_data['Lowdin']))
        if '_GasteigerCharge' in atom_data:
            a.SetProp('_GasteigerCharge', str(atom_data['_GasteigerCharge']))
    
        # velocity
        if 'vx' in atom_data:
            a.SetDoubleProp('vx', float(atom_data['vx']))
            a.SetDoubleProp('vy', float(atom_data['vy']))
            a.SetDoubleProp('vz', float(atom_data['vz']))

        # others
        a.SetIsotope(int(atom_data['isotope']))
        if 'mol_id' in atom_data:
            a.SetIntProp('mol_id', int(atom_data['mol_id']))

        # PDB
        atom_name = str(atom_data['ff_type']) if 'ff_type' in atom_data else a.GetSymbol()
        if 'ResidueName' in atom_data and 'ResidueNumber' in atom_data:
            a.SetMonomerInfo(
                Chem.AtomPDBResidueInfo(
                    atom_name,
                    residueName=atom_data['ResidueName'],
                    residueNumber=atom_data['ResidueNumber'],
                    isHeteroAtom=False
                )
            )


    for i, b in enumerate(mol.GetBonds()):
        bond_data = radonpy_ext['bonds'][i]

        # FF on bonds
        if 'ff_type' in bond_data:
            b.SetProp('ff_type', str(bond_data['ff_type']))
        if 'ff_k' in bond_data:
            b.SetDoubleProp('ff_k', float(bond_data['ff_k']))
        if 'ff_r0' in bond_data:
            b.SetDoubleProp('ff_r0', float(bond_data['ff_r0']))


    if 'angles' in radonpy_ext:
        if hasattr(mol, 'angle_style') and mol.angle_style == 'harmonic':
            angle_class = ff_class.GAFF_Angle
        else:
            angle_class = ff_class.GAFF_Angle

        angle_prop = {}
        for ang in radonpy_ext['angles']:
            key = '%i,%i,%i' % (int(ang['a']), int(ang['b']), int(ang['c']))
            angle_prop[key] = Angle(
                a = int(ang['a']),
                b = int(ang['b']),
                c = int(ang['c']),
                ff = angle_class(**ang['ff'])
            )
        setattr(mol, 'angles', angle_prop)


    if 'dihedrals' in radonpy_ext:
        if hasattr(mol, 'dihedral_style') and mol.dihedral_style == 'fourier':
            dihedral_class = ff_class.GAFF_Dihedral
        else:
            dihedral_class = ff_class.GAFF_Dihedral

        dihedral_prop = {}
        for dih in radonpy_ext['dihedrals']:
            key = '%i,%i,%i,%i' % (int(dih['a']), int(dih['b']), int(dih['c']), int(dih['d']))
            dihedral_prop[key] = Dihedral(
                a = int(dih['a']),
                b = int(dih['b']),
                c = int(dih['c']),
                d = int(dih['d']),
                ff = dihedral_class(**dih['ff'])
            )
        setattr(mol, 'dihedrals', dihedral_prop)


    if 'impropers' in radonpy_ext:
        if hasattr(mol, 'improper_style') and mol.improper_style == 'cvff':
            improper_class = ff_class.GAFF_Improper
        else:
            improper_class = ff_class.GAFF_Improper

        improper_prop = {}
        for imp in radonpy_ext['impropers']:
            key = '%i,%i,%i,%i' % (int(imp['a']), int(imp['b']), int(imp['c']), int(imp['d']))
            improper_prop[key] = Improper(
                a = int(imp['a']),
                b = int(imp['b']),
                c = int(imp['c']),
                d = int(imp['d']),
                ff = improper_class(**imp['ff'])
            )
        setattr(mol, 'impropers', improper_prop)


    # cell
    if 'cell' in radonpy_ext and 'xhi' in radonpy_ext['cell']:
        cell_prop = Cell(**radonpy_ext['cell'])
        setattr(mol, 'cell', cell_prop)

    return mol


def picklable(mol=None):
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    return mol


def restore_picklable(mol=None):
    # Backward campatibility
    return mol


def pickle_dump(mol, path):
    Chem.SanitizeMol(mol)
    mol = picklable(mol)
    with open(path, mode='wb') as f:
        pickle.dump(mol, f)


def pickle_load(path):
    try:
        with open(path, mode='rb') as f:
            mol = pickle.load(f)
    except BaseException as e:
        radon_print('%s' % e, level=2)
        return None

    # Backward campatibility from version 0.2 to 0.3
    if hasattr(mol, 'angles'):
        if type(mol.angles) == list:
            mol.angles = {'%i,%i,%i' % (ang.a, ang.b, ang.c): ang for ang in mol.angles}

    if hasattr(mol, 'dihedrals'):
        if type(mol.dihedrals) == list:
            mol.dihedrals = {'%i,%i,%i,%i' % (dih.a, dih.b, dih.c, dih.d): dih for dih in mol.dihedrals}

    if hasattr(mol, 'impropers'):
        if type(mol.impropers) == list:
            mol.impropers = {'%i,%i,%i,%i' % (imp.a, imp.b, imp.c, imp.d): imp for imp in mol.impropers}

    return mol


def deepcopy_mol(mol):
    mol = picklable(mol)
    copy_mol = deepcopy(mol)

    return copy_mol


def cpu_count():
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError as e:
        cpu_count = psutil.cpu_count(logical=False)
        if cpu_count is None:
            cpu_count = psutil.cpu_count(logical=True)

    return cpu_count


def tqdm_stub(it, **kwargs):
    return it


def mol_from_smiles(smiles, coord=True, version=3, ez='E', chiral='S', stereochemistry_control=True):

    smi = smiles.replace('[*]', '[3H]')
    smi = re.sub('\[([0-9]+)\*\]', lambda m: '[%iH]' % int(int(m.groups()[0])+2), smi)
    smi = smi.replace('*', '[3H]')

    l = re.findall(r'\[([0-9]+)H\]', smi)
    labels = [int(x) for x in l if int(x) >= 3]
    if len(labels) > 0:
        n_conn = smi.count('[%iH]' % min(labels))
    else:
        n_conn = 0

    if version == 3:
        etkdg = AllChem.ETKDGv3()
    elif version == 2:
        etkdg = AllChem.ETKDGv2()
    else:
        etkdg = AllChem.ETKDG()
    etkdg.enforceChirality=True
    etkdg.useRandomCoords = False
    etkdg.maxAttempts = 100

    try:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
    except Exception as e:
        radon_print('Cannot transform to RDKit Mol object from %s' % smiles, level=3)
        return None

    if stereochemistry_control:
        ### cis/trans and chirality control
        Chem.AssignStereochemistry(mol)

        # Get polymer backbone
        backbone_atoms = []
        backbone_bonds = []
        backbone_dih = []

        if n_conn == 2:
            link_idx = []
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == "H" and atom.GetIsotope() >= 3:
                    link_idx.append(atom.GetIdx())
            backbone_atoms = Chem.GetShortestPath(mol, link_idx[0], link_idx[1])

            for i in range(len(backbone_atoms)-1):
                bond = mol.GetBondBetweenAtoms(backbone_atoms[i], backbone_atoms[i+1])
                backbone_bonds.append(bond.GetIdx())
                if bond.GetBondTypeAsDouble() == 2 and str(bond.GetStereo()) == 'STEREONONE' and not bond.IsInRing():
                    backbone_dih.append((backbone_atoms[i-1], backbone_atoms[i], backbone_atoms[i+1], backbone_atoms[i+2]))

        # List of unspecified double bonds (except for bonds in polymer backbone and a ring structure)
        db_list = []
        for bond in mol.GetBonds():
            if bond.GetBondTypeAsDouble() == 2 and str(bond.GetStereo()) == 'STEREONONE' and not bond.IsInRing():
                if n_conn == 2 and bond.GetIdx() in backbone_bonds:
                    continue
                else:
                    db_list.append(bond.GetIdx())

        # Enumerate stereo isomers
        opts = Chem.EnumerateStereoisomers.StereoEnumerationOptions(unique=True, tryEmbedding=True)
        isomers = tuple(Chem.EnumerateStereoisomers.EnumerateStereoisomers(mol, options=opts))

        if len(isomers) > 1:
            radon_print('%i candidates of stereoisomers were generated.' % len(isomers))
            chiral_num_max = 0
            
            for isomer in isomers:
                ez_flag = False
                chiral_flag = 0

                Chem.AssignStereochemistry(isomer)

                # Contorol unspecified double bonds (except for bonds in polymer backbone and a ring structure)
                ez_list = []
                for idx in db_list:
                    bond = isomer.GetBondWithIdx(idx)
                    if str(bond.GetStereo()) == 'STEREOANY' or str(bond.GetStereo()) == 'STEREONONE':
                        continue
                    elif ez == 'E' and (str(bond.GetStereo()) == 'STEREOE' or str(bond.GetStereo()) == 'STEREOTRANS'):
                        ez_list.append(True)
                    elif ez == 'Z' and (str(bond.GetStereo()) == 'STEREOZ' or str(bond.GetStereo()) == 'STEREOCIS'):
                        ez_list.append(True)
                    else:
                        ez_list.append(False)

                if len(ez_list) > 0:
                    ez_flag = np.all(np.array(ez_list))
                else:
                    ez_flag = True

                # Contorol unspecified chirality
                chiral_list = np.array(Chem.FindMolChiralCenters(isomer))
                if len(chiral_list) > 0:
                    chiral_centers = chiral_list[:, 0]

                    chirality = chiral_list[:, 1]
                    chiral_num = np.count_nonzero(chirality == chiral)
                    if chiral_num == len(chiral_list):
                        chiral_num_max = chiral_num
                        chiral_flag = 2
                    elif chiral_num > chiral_num_max:
                        chiral_num_max = chiral_num
                        chiral_flag = 1
                else:
                    chiral_flag = 2

                if ez_flag and chiral_flag:
                    mol = isomer
                    if chiral_flag == 2:
                        break

    # Generate 3D coordinates
    if coord:
        try:
            enbed_res = AllChem.EmbedMolecule(mol, etkdg)
        except Exception as e:
            radon_print('Cannot generate 3D coordinate of %s' % smiles, level=3)
            return None
        if enbed_res == -1:
            etkdg.useRandomCoords = True
            try:
                enbed_res = AllChem.EmbedMolecule(mol, etkdg)
            except Exception as e:
                radon_print('Cannot generate 3D coordinate of %s' % smiles, level=3)
                return None
            if enbed_res == -1:
                radon_print('Cannot generate 3D coordinate of %s' % smiles, level=3)
                return None

    # Dihedral angles of unspecified double bonds in a polymer backbone are modified to 180 degree.
    if stereochemistry_control:
        if len(backbone_dih) > 0:
            for dih_idx in backbone_dih:
                Chem.rdMolTransforms.SetDihedralDeg(mol.GetConformer(0), dih_idx[0], dih_idx[1], dih_idx[2], dih_idx[3], 180.0)

                for na in mol.GetAtomWithIdx(dih_idx[2]).GetNeighbors():
                    na_idx = na.GetIdx()
                    if na_idx != dih_idx[1] and na_idx != dih_idx[3]:
                        break
                Chem.rdMolTransforms.SetDihedralDeg(mol.GetConformer(0), dih_idx[0], dih_idx[1], dih_idx[2], na_idx, 0.0)

    return mol


def is_in_ring(ab, max_size=10):

    for i in range(3, max_size+1):
        if ab.IsInRingSize(int(i)):
            return True
    return False


def picklable_const():
    c = {}
    for v in dir(const):
        if v.count('__') != 2 and v != 'os':
            c[v] = getattr(const, v)
    return c


def restore_const(c):
    for k, v in c.items():
        setattr(const, k, v)
    return True

