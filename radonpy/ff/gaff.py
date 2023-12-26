#  Copyright (c) 2023. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# ******************************************************************************
# ff.gaff module
# ******************************************************************************

import numpy as np
import os
import json
from itertools import permutations
from rdkit import Chem
from ..core import calc, utils
from . import ff_class

__version__ = '0.2.9'


class GAFF():
    """
    gaff.GAFF() class

    Forcefield object with typing rules for Gaff model.
    By default reads data file in forcefields subdirectory.

    Attributes:
        ff_name: gaff
        pair_style: lj
        bond_style: harmonic
        angle_style: harmonic
        dihedral_style: fourier
        improper_style: cvff
        ff_class: 1
    """
    def __init__(self, db_file=None):
        if db_file is None:
            db_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ff_dat', 'gaff.json')
        self.param = self.load_ff_json(db_file)
        self.name = 'gaff'
        self.pair_style = 'lj'
        self.bond_style = 'harmonic'
        self.angle_style = 'harmonic'
        self.dihedral_style = 'fourier'
        self.improper_style = 'cvff'
        self.ff_class = '1'
        self.param.c_c12 = 0.0
        self.param.c_c13 = 0.0
        self.param.c_c14 = 5/6
        self.param.lj_c12 = 0.0
        self.param.lj_c13 = 0.0
        self.param.lj_c14 = 0.5
        self.max_ring_size = 14
        self.alt_ptype = {
            'cc': 'c2', 'cd': 'c2', 'ce': 'c2', 'cf': 'c2', 'cg': 'c1', 'ch': 'c1',
            'cp': 'ca', 'cq': 'ca', 'cu': 'c2', 'cv': 'c2', 'cx': 'c3', 'cy': 'c3',
            'h1': 'hc', 'h2': 'hc', 'h3': 'hc', 'h4': 'ha', 'h5': 'ha',
            'nb': 'nc', 'nc': 'n2', 'nd': 'n2', 'ne': 'n2', 'nf': 'n2',
            'pb': 'pc', 'pc': 'p2', 'pd': 'p2', 'pe': 'p2', 'pf': 'p2',
            'p4': 'px', 'px': 'p4', 'p5': 'py', 'py': 'p5',
            's4': 'sx', 'sx': 's4', 's6': 'sy', 'sy': 's6'
        }


    def ff_assign(self, mol, charge=None, retryMDL=True, useMDL=True):
        """
        GAFF.ff_assign

        GAFF force field assignment for RDkit Mol object

        Args:
            mol: rdkit mol object

        Optional args:
            charge: Method of charge assignment. If None, charge assignment is skipped. 
            retryMDL: Retry assignment using MDL aromaticity model if default aromaticity model is failure (boolean)
            useMDL: Assignment using MDL aromaticity model (boolean)

        Returns: (boolean)
            True: Success assignment
            False: Failure assignment
        """

        if useMDL:
            Chem.rdmolops.Kekulize(mol, clearAromaticFlags=True)
            Chem.rdmolops.SetAromaticity(mol, model=Chem.rdmolops.AromaticityModel.AROMATICITY_MDL)

        mol.SetProp('ff_name', str(self.name))
        mol.SetProp('ff_class', str(self.ff_class))
        result = self.assign_ptypes(mol)
        if result: result = self.assign_btypes(mol)
        if result: result = self.assign_atypes(mol)
        if result: result = self.assign_dtypes(mol)
        if result: result = self.assign_itypes(mol)
        if result and charge is not None: result = calc.assign_charges(mol, charge=charge)
        
        if not result and retryMDL and not useMDL:
            utils.radon_print('Retry to assign with MDL aromaticity model', level=1)
            Chem.rdmolops.Kekulize(mol, clearAromaticFlags=True)
            Chem.rdmolops.SetAromaticity(mol, model=Chem.rdmolops.AromaticityModel.AROMATICITY_MDL)

            result = self.assign_ptypes(mol)
            if result: result = self.assign_btypes(mol)
            if result: result = self.assign_atypes(mol)
            if result: result = self.assign_dtypes(mol)
            if result: result = self.assign_itypes(mol)
            if result and charge is not None: result = calc.assign_charges(mol, charge=charge)
            if result: utils.radon_print('Success to assign with MDL aromaticity model', level=1)

        return result


    def assign_ptypes(self, mol):
        """
        GAFF.assign_ptypes

        GAFF specific particle typing rules.

        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        result_flag = True
        mol.SetProp('pair_style', self.pair_style)
        
        for p in mol.GetAtoms():
            ######################################
            # Assignment routine of H
            ######################################
            if p.GetSymbol() == 'H':
                if p.GetNeighbors()[0].GetSymbol() == 'O':
                    water = False
                    for pb in p.GetNeighbors():
                        if pb.GetSymbol() == 'O' and pb.GetTotalNumHs(includeNeighbors=True) == 2:
                            water = True
                    if water:
                        self.set_ptype(p, 'hw')
                    else:
                        self.set_ptype(p, 'ho')
                        
                elif p.GetNeighbors()[0].GetSymbol() == 'N':
                    self.set_ptype(p, 'hn')
                    
                elif p.GetNeighbors()[0].GetSymbol() == 'P':
                    self.set_ptype(p, 'hp')
                    
                elif p.GetNeighbors()[0].GetSymbol() == 'S':
                    self.set_ptype(p, 'hs')
                    
                elif p.GetNeighbors()[0].GetSymbol() == 'C':
                    for pb in p.GetNeighbors():
                        if pb.GetSymbol() == 'C':
                            elctrwd = 0
                            for pbb in pb.GetNeighbors():
                                if pbb.GetSymbol() in ['N', 'O', 'F', 'Cl', 'Br', 'I']: 
                                    elctrwd += 1
                            if elctrwd == 0:
                                if str(pb.GetHybridization()) == 'SP2' or str(pb.GetHybridization()) == 'SP':
                                    self.set_ptype(p, 'ha')
                                else:
                                    self.set_ptype(p, 'hc')
                            elif pb.GetTotalDegree() == 4 and elctrwd == 1:
                                self.set_ptype(p, 'h1')
                            elif pb.GetTotalDegree() == 4 and elctrwd == 2:
                                self.set_ptype(p, 'h2')
                            elif pb.GetTotalDegree() == 4 and elctrwd == 3:
                                self.set_ptype(p, 'h3')
                            elif pb.GetTotalDegree() == 3 and elctrwd == 1:
                                self.set_ptype(p, 'h4')
                            elif pb.GetTotalDegree() == 3 and elctrwd == 2:
                                self.set_ptype(p, 'h5')
                            else:
                                utils.radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
                                            % (p.GetIdx(), p.GetSymbol(), p.GetTotalDegree(), str(p.GetHybridization())), level=2)
                                result_flag = False
                                
                else:
                    utils.radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
                                % (p.GetIdx(), p.GetSymbol(), p.GetTotalDegree(), str(p.GetHybridization())), level=2 )
                    result_flag = False
                                
                                
                                
            ######################################
            # Assignment routine of C
            ######################################
            elif p.GetSymbol() == 'C':
                if str(p.GetHybridization()) == 'SP3':
                    self.set_ptype(p, 'c3')
                    
                elif str(p.GetHybridization()) == 'SP2':
                    carbonyl = False
                    cs = False
                    for pb in p.GetNeighbors():
                        if pb.GetSymbol() == 'O':
                            for b in pb.GetBonds():
                                if (
                                    (b.GetBeginAtom().GetIdx() == p.GetIdx() and b.GetEndAtom().GetIdx() == pb.GetIdx()) or
                                    (b.GetBeginAtom().GetIdx() == pb.GetIdx() and b.GetEndAtom().GetIdx() == p.GetIdx())
                                ):
                                    if b.GetBondTypeAsDouble() == 2 and pb.GetTotalDegree() == 1:
                                        carbonyl = True
                    if carbonyl:
                        self.set_ptype(p, 'c')  # Carbonyl carbon
                    elif p.GetIsAromatic():
                        self.set_ptype(p, 'ca')
                    else:
                        self.set_ptype(p, 'c2')  # Other sp2 carbon
                        
                elif str(p.GetHybridization()) == 'SP':
                    self.set_ptype(p, 'c1')
                    
                else:
                    utils.radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
                                % (p.GetIdx(), p.GetSymbol(), p.GetTotalDegree(), str(p.GetHybridization())), level=2 )
                    result_flag = False
                    
                    
                    
            ######################################
            # Assignment routine of N
            ######################################
            elif p.GetSymbol() == 'N':
                if str(p.GetHybridization()) == 'SP':
                    self.set_ptype(p, 'n1')
                    
                elif p.GetTotalDegree() == 2:
                    bond_orders = [x.GetBondTypeAsDouble() for x in p.GetBonds()]
                    if p.GetIsAromatic():
                        self.set_ptype(p, 'nb')
                    elif 2 in bond_orders:
                        self.set_ptype(p, 'n2')
                    else:
                        utils.radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
                                    % (p.GetIdx(), p.GetSymbol(), p.GetTotalDegree(), str(p.GetHybridization())), level=2 )
                        result_flag = False
                        
                elif p.GetTotalDegree() == 3:
                    amide = False
                    aromatic_ring = False
                    no2 = 0
                    sp2 = 0
                    for pb in p.GetNeighbors():
                        if pb.GetSymbol() == 'C':
                            if pb.GetIsAromatic():
                                aromatic_ring = True
                            for b in pb.GetBonds():
                                bp = b.GetBeginAtom() if pb.GetIdx() == b.GetEndAtom().GetIdx() else b.GetEndAtom()
                                if (bp.GetSymbol() == 'O' or bp.GetSymbol() == 'S') and b.GetBondTypeAsDouble() == 2:
                                    amide = True
                        elif pb.GetSymbol() == 'O':
                            no2 += 1
                        if str(pb.GetHybridization()) == 'SP2' or str(pb.GetHybridization()) == 'SP':
                            sp2 += 1
                    if no2 >= 2:
                        self.set_ptype(p, 'no')
                    elif amide:
                        self.set_ptype(p, 'n')
                    elif p.GetIsAromatic():
                        self.set_ptype(p, 'na')
                    elif sp2 >= 2:
                        self.set_ptype(p, 'na')
                    elif aromatic_ring:
                        self.set_ptype(p, 'nh')
                    else:
                        self.set_ptype(p, 'n3')
                        
                elif p.GetTotalDegree() == 4:
                    self.set_ptype(p, 'n4')
                    
                else:
                    utils.radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
                                % (p.GetIdx(), p.GetSymbol(), p.GetTotalDegree(), str(p.GetHybridization())), level=2 )
                    result_flag = False



            ######################################
            # Assignment routine of O
            ######################################
            elif p.GetSymbol() == 'O':
                if p.GetTotalDegree() == 1:
                    self.set_ptype(p, 'o')
                elif p.GetTotalNumHs(includeNeighbors=True) == 2:
                    self.set_ptype(p, 'ow')
                elif p.GetTotalNumHs(includeNeighbors=True) == 1:
                    self.set_ptype(p, 'oh')
                else:
                    self.set_ptype(p, 'os')



            ######################################
            # Assignment routine of F, Cl, Br, I
            ######################################
            elif p.GetSymbol() == 'F':
                self.set_ptype(p, 'f')
            elif p.GetSymbol() == 'Cl':
                self.set_ptype(p, 'cl')
            elif p.GetSymbol() == 'Br':
                self.set_ptype(p, 'br')
            elif p.GetSymbol() == 'I':
                self.set_ptype(p, 'i')
                
                
                
            ######################################
            # Assignment routine of P
            ######################################
            elif p.GetSymbol() == 'P':
                if p.GetIsAromatic():
                    self.set_ptype(p, 'pb')
                    
                elif p.GetTotalDegree() == 2:
                    self.set_ptype(p, 'p2')
                    
                elif p.GetTotalDegree() == 3:
                    bond_orders = [x.GetBondTypeAsDouble() for x in p.GetBonds()]
                    if 2 in bond_orders:
                        conj = False
                        for pb in p.GetNeighbors():
                            for b in pb.GetBonds():
                                if b.GetBeginAtom().GetIdx() != p.GetIdx() and b.GetEndAtom().GetIdx() != p.GetIdx():
                                    if b.GetBondTypeAsDouble() >= 1.5:
                                        conj = True
                        if conj:
                            self.set_ptype(p, 'px')
                        else:
                            self.set_ptype(p, 'p4')
                    else:
                        self.set_ptype(p, 'p3')
                        
                elif p.GetTotalDegree() == 4:
                    conj = False
                    for pb in p.GetNeighbors():
                        for b in pb.GetBonds():
                            if b.GetBeginAtom().GetIdx() != p.GetIdx() and b.GetEndAtom().GetIdx() != p.GetIdx():
                                if b.GetBondTypeAsDouble() >= 1.5:
                                    conj = True
                    if conj:
                        self.set_ptype(p, 'py')
                    else:
                        self.set_ptype(p, 'p5')

                elif p.GetTotalDegree() == 5 or p.GetTotalDegree() == 6:
                    self.set_ptype(p, 'p5')
                        
                else:
                    utils.radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
                                % (p.GetIdx(), p.GetSymbol(), p.GetTotalDegree(), str(p.GetHybridization())), level=2 )
                    result_flag = False



            ######################################
            # Assignment routine of S
            ######################################
            elif p.GetSymbol() == 'S':
                if p.GetTotalDegree() == 1:
                    self.set_ptype(p, 's')
                    
                elif p.GetTotalDegree() == 2:
                    bond_orders = [x.GetBondTypeAsDouble() for x in p.GetBonds()]
                    if p.GetIsAromatic():
                        self.set_ptype(p, 'ss')
                    elif p.GetTotalNumHs(includeNeighbors=True) == 1:
                        self.set_ptype(p, 'sh')
                    elif 2 in bond_orders:
                        self.set_ptype(p, 's2')
                    else:
                        self.set_ptype(p, 'ss')
                    
                elif p.GetTotalDegree() == 3:
                    conj = False
                    for pb in p.GetNeighbors():
                        for b in pb.GetBonds():
                            if b.GetBeginAtom().GetIdx() != p.GetIdx() and b.GetEndAtom().GetIdx() != p.GetIdx():
                                if b.GetBondTypeAsDouble() >= 1.5:
                                    conj = True
                    if conj:
                        self.set_ptype(p, 'sx')
                    else:
                        self.set_ptype(p, 's4')
                        
                elif p.GetTotalDegree() == 4:
                    conj = False
                    for pb in p.GetNeighbors():
                        for b in pb.GetBonds():
                            if b.GetBeginAtom().GetIdx() != p.GetIdx() and b.GetEndAtom().GetIdx() != p.GetIdx():
                                if b.GetBondTypeAsDouble() >= 1.5:
                                    conj = True
                    if conj:
                        self.set_ptype(p, 'sy')
                    else:
                        self.set_ptype(p, 's6')
                        
                elif p.GetTotalDegree() == 5 or p.GetTotalDegree() == 6:
                    self.set_ptype(p, 's6')

                else:
                    utils.radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
                                % (p.GetIdx(), p.GetSymbol(), p.GetTotalDegree(), str(p.GetHybridization())), level=2 )
                    result_flag = False


            elif p.GetSymbol() == '*':
                p.SetProp('ff_type', '*')
                p.SetDoubleProp('ff_epsilon', 0.0)
                p.SetDoubleProp('ff_sigma', 0.0)

            ######################################
            # Assignment error
            ######################################
            else:
                utils.radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
                            % (p.GetIdx(), p.GetSymbol(), p.GetTotalDegree(), str(p.GetHybridization())), level=2 )
                result_flag = False


        ###########################################
        # Assignment of special atom type in GAFF
        ###########################################
        if result_flag: self.assign_special_ptype(mol)
        
        
        return result_flag



    def assign_special_ptype(self, mol):
        """
        GAFF.assign_special_ptype
        
            Assignment of special particle type in GAFF
            C: cc, cd, ce, cf, cg, ch, cp, cq, cx, cy cu, cv
            N: nc, nd, ne, nf
            P: pc, pd, pe, pf
        """

        # For 3 or 4-membered ring
        for p in mol.GetAtoms():
            if p.GetProp('ff_type') == 'c2': # cu, cv
                if p.IsInRingSize(3):
                    self.set_ptype(p, 'cu')
                elif p.IsInRingSize(4):
                    self.set_ptype(p, 'cv')
            elif p.GetProp('ff_type') == 'c3':  # cx, cy
                if p.IsInRingSize(3):
                    self.set_ptype(p, 'cx')
                elif p.IsInRingSize(4):
                    self.set_ptype(p, 'cy')

        conj_c = ['c', 'c1', 'c2', 'ca', 'cc', 'cd', 'ce', 'cf', 'cg', 'ch', 'cp', 'cq', 'cu', 'cv',
                'n1', 'n2', 'na', 'nb', 'nc', 'nd', 'ne', 'nf',
                'p2', 'pb', 'pc', 'pd', 'pe', 'pf', 'px', 'py',
                'sx', 'sy']
        conj_r = ['c', 'c1', 'c2', 'ca', 'cc', 'cd', 'ce', 'cf', 'cg', 'ch', 'cp', 'cq', 'cu', 'cv',
                'n1', 'n2', 'na', 'nb', 'nc', 'nd', 'ne', 'nf',
                'os',
                'p2', 'pb', 'pc', 'pd', 'pe', 'pf', 'px', 'py',
                'ss', 'sx', 'sy']

        # For inner sp2 or sp atoms
        for p in mol.GetAtoms():
            count = 0
            if not utils.is_in_ring(p, max_size=self.max_ring_size): # Chain
                for pb in p.GetNeighbors():
                    if pb.GetProp('ff_type') in conj_c:
                        count += 1
                if count >= 2:
                    if p.GetProp('ff_type') == 'c2':
                        self.set_ptype(p, 'ce')
                    elif p.GetProp('ff_type') == 'c1':
                        self.set_ptype(p, 'cg')
                    elif p.GetProp('ff_type') == 'n2':
                        self.set_ptype(p, 'ne')
                    elif p.GetProp('ff_type') == 'p2':
                        self.set_ptype(p, 'pe')
            else: # Ring
                for pb in p.GetNeighbors():
                    if pb.GetProp('ff_type') in conj_r:
                        count += 1
                if count >= 2:
                    if p.GetProp('ff_type') == 'c2':
                        self.set_ptype(p, 'cc')
                    elif p.GetProp('ff_type') == 'n2':
                        self.set_ptype(p, 'nc')
                    elif p.GetProp('ff_type') == 'p2':
                        self.set_ptype(p, 'pc')

        # Replacement of ce to cf
        target_c = ['ce', 'cg', 'ne', 'pe']
        target_r = ['cc', 'nc', 'pc']
        rep_r = ['cd', 'nd', 'pd']
        rep = {'ce': 'cf', 'cg': 'ch', 'ne': 'nf', 'pe': 'pf',
                'cc': 'cd', 'nc': 'nd', 'pc': 'pd'}
        for p in mol.GetAtoms():
            if p.GetProp('ff_type') in target_c: # Chain
                for b in p.GetBonds():
                    if b.GetBondTypeAsDouble() == 2 or b.GetBondTypeAsDouble() == 3:
                        bp = b.GetBeginAtom() if p.GetIdx() == b.GetEndAtom().GetIdx() else b.GetEndAtom()
                        if bp.GetProp('ff_type') in target_c:
                            self.set_ptype(bp, rep[bp.GetProp('ff_type')])
                            for bpb in bp.GetBonds():
                                if bpb.GetBondTypeAsDouble() == 1:
                                    bpbp = bpb.GetBeginAtom() if bp.GetIdx() == bpb.GetEndAtom().GetIdx() else bpb.GetEndAtom()
                                    if bpbp.GetProp('ff_type') in target_c:
                                        self.set_ptype(bpbp, rep[bpbp.GetProp('ff_type')])
                                        
            elif p.GetProp('ff_type') in target_r: # Kekulized Ring
                for b in p.GetBonds():
                    if b.GetBondTypeAsDouble() == 2:
                        bp = b.GetBeginAtom() if p.GetIdx() == b.GetEndAtom().GetIdx() else b.GetEndAtom()
                        if bp.GetProp('ff_type') in target_r:
                            self.set_ptype(bp, rep[bp.GetProp('ff_type')])
                            for bpb in bp.GetBonds():
                                if bpb.GetBondTypeAsDouble() == 1:
                                    bpbp = bpb.GetBeginAtom() if bp.GetIdx() == bpb.GetEndAtom().GetIdx() else bpb.GetEndAtom()
                                    if bpbp.GetProp('ff_type') in target_r:
                                        self.set_ptype(bpbp, rep[bpbp.GetProp('ff_type')])

        # For biphenyl head atom
        target = ['ca', 'nb', 'pb']
        for p in mol.GetAtoms():
            if p.GetProp('ff_type') == 'ca':
                for b in p.GetBonds():
                    if b.GetBondTypeAsDouble() == 1:
                        bp = b.GetBeginAtom() if p.GetIdx() == b.GetEndAtom().GetIdx() else b.GetEndAtom()
                        if bp.GetProp('ff_type') in target:
                            self.set_ptype(p, 'cp')
                            if bp.GetProp('ff_type') == 'ca':
                                self.set_ptype(bp, 'cp')

        # Replacement of cp to cq
        for p in mol.GetAtoms():
            if p.GetProp('ff_type') == 'cp':
                for b in p.GetBonds():
                    if b.GetBondTypeAsDouble() == 1.5:
                        bp = b.GetBeginAtom() if p.GetIdx() == b.GetEndAtom().GetIdx() else b.GetEndAtom()
                        if bp.GetProp('ff_type') == 'cp':
                            self.set_ptype(bp, 'cq')
                            for cqb in bp.GetBonds():
                                if cqb.GetBondTypeAsDouble() == 1:
                                    cqbp = cqb.GetBeginAtom() if bp.GetIdx() == cqb.GetEndAtom().GetIdx() else cqb.GetEndAtom()
                                    if cqbp.GetProp('ff_type') == 'cp':
                                        self.set_ptype(cqbp, 'cq')

        
        return True
        
        
    def set_ptype(self, p, pt):
        p.SetProp('ff_type', pt)
        p.SetDoubleProp('ff_epsilon', self.param.pt[pt].epsilon)
        p.SetDoubleProp('ff_sigma', self.param.pt[pt].sigma)
        
        return p
        
        
    def assign_btypes(self, mol):
        """
        GAFF.assign_btypes

        GAFF specific bond typing rules.

        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        result_flag = True
        mol.SetProp('bond_style', self.bond_style)
        alt_ptype = self.alt_ptype
        for b in mol.GetBonds():
            ba = b.GetBeginAtom().GetProp('ff_type')
            bb = b.GetEndAtom().GetProp('ff_type')
            bt = '%s,%s' % (ba, bb)
            
            result = self.set_btype(b, bt)
            if not result:
                alt1 = alt_ptype[ba] if ba in alt_ptype.keys() else None
                alt2 = alt_ptype[bb] if bb in alt_ptype.keys() else None
                if alt1 is None and alt2 is None:
                    utils.radon_print(('Can not assign this bond %s,%s' % (ba, bb)), level=2)
                    result_flag = False
                    continue
                
                bt_alt = []
                if alt1: bt_alt.append('%s,%s' % (alt1, bb))
                if alt2: bt_alt.append('%s,%s' % (ba, alt2))
                if alt1 and alt2: bt_alt.append('%s,%s' % (alt1, alt2))

                for bt in bt_alt:
                    result = self.set_btype(b, bt)
                    if result:
                        utils.radon_print('Using alternate bond type %s instead of %s,%s' % (bt, ba, bb))
                        break
                        
                if not b.HasProp('ff_type'):
                    utils.radon_print(('Can not assign this bond %s,%s' % (ba, bb)), level=2)
                    result_flag = False
                    
        return result_flag
    
    
    def set_btype(self, b, bt):
        if bt not in self.param.bt:
            return False
            
        b.SetProp('ff_type', self.param.bt[bt].tag)
        b.SetDoubleProp('ff_k', self.param.bt[bt].k)
        b.SetDoubleProp('ff_r0', self.param.bt[bt].r0)
        
        return True
        

    def assign_atypes(self, mol):
        """
        GAFF.assign_atypes

        GAFF specific angle typing rules.

        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        result_flag = True
        mol.SetProp('angle_style', self.angle_style)
        alt_ptype = self.alt_ptype
        setattr(mol, 'angles', [])
        
        for p in mol.GetAtoms():
            for p1 in p.GetNeighbors():
                for p2 in p.GetNeighbors():
                    if p1.GetIdx() == p2.GetIdx(): continue
                    unique = True
                    atoms = [p1, p, p2]
                    for ang in mol.angles:
                        if ((ang.a == p1.GetIdx() and ang.b == p.GetIdx() and ang.c == p2.GetIdx()) or
                            (ang.c == p1.GetIdx() and ang.b == p.GetIdx() and ang.a == p2.GetIdx())):
                            unique = False
                    if unique:
                        pt1 = p1.GetProp('ff_type')
                        pt = p.GetProp('ff_type')
                        pt2 = p2.GetProp('ff_type')
                        at = '%s,%s,%s' % (pt1, pt, pt2)
                        
                        result = self.set_atype(mol, a=p1.GetIdx(), b=p.GetIdx(), c=p2.GetIdx(), at=at)
                        
                        if not result:
                            alt1 = alt_ptype[pt1] if pt1 in alt_ptype.keys() else None
                            alt2 = alt_ptype[pt] if pt in alt_ptype.keys() else None
                            alt3 = alt_ptype[pt2] if pt2 in alt_ptype.keys() else None
                            if alt1 is None and alt2 is None and alt3 is None:
                                emp_result = self.empirical_angle_param(mol, p1, p, p2)
                                if not emp_result:
                                    utils.radon_print(('Can not assign this angle %s,%s,%s' % (pt1, pt, pt2)), level=2)
                                    result_flag = False
                                continue

                            at_alt = []
                            if alt1: at_alt.append('%s,%s,%s' % (alt1, pt, pt2))
                            if alt2: at_alt.append('%s,%s,%s' % (pt1, alt2, pt2))
                            if alt3: at_alt.append('%s,%s,%s' % (pt1, pt, alt3))
                            if alt1 and alt2: at_alt.append('%s,%s,%s' % (alt1, alt2, pt2))
                            if alt1 and alt3: at_alt.append('%s,%s,%s' % (alt1, pt, alt3))
                            if alt2 and alt3: at_alt.append('%s,%s,%s' % (pt1, alt2, alt3))
                            if alt1 and alt2 and alt3: at_alt.append('%s,%s,%s' % (alt1, alt2, alt3))
                            
                            for at in at_alt:
                                result = self.set_atype(mol, a=p1.GetIdx(), b=p.GetIdx(), c=p2.GetIdx(), at=at)
                                if result:
                                    utils.radon_print('Using alternate angle type %s instead of %s,%s,%s' % (at, pt1, pt, pt2))
                                    break
                                    
                            if not result:
                                emp_result = self.empirical_angle_param(mol, p1, p, p2)
                                if not emp_result:
                                    utils.radon_print(('Can not assign this angle %s,%s,%s' % (pt1, pt, pt2)), level=2)
                                    result_flag = False

        return result_flag
        

    def empirical_angle_param(self, mol, a, b, c):

        param_C = {'H':0.000, 'C':1.339, 'N':1.300, 'O':1.249, 'F':0.000, 'Cl':0.000, 'Br':0.000, 'I':0.000, 'P':0.906, 'S':1.448}
        param_Z = {'H':0.784, 'C':1.183, 'N':1.212, 'O':1.219, 'F':1.166, 'Cl':1.272, 'Br':1.378, 'I':1.398, 'P':1.620, 'S':1.280}

        emp_theta = None
        emp_k_ang = None

        pt1 = a.GetProp('ff_type')
        pt = b.GetProp('ff_type')
        pt2 = c.GetProp('ff_type')

        at1 = '%s,%s,%s' % (pt1, pt, pt1)
        at2 = '%s,%s,%s' % (pt2, pt, pt2)

        bt1 = '%s,%s' % (pt1, pt)
        bt2 = '%s,%s' % (pt, pt2)

        if b.GetSymbol() in ['H', 'F', 'Cl', 'Br', 'I']:
            utils.radon_print(('Can not estimate parameters of this angle %s,%s,%s' % (pt1, pt, pt2)), level=2)
            return False

        # Estimate theta0
        if at1 not in self.param.at or at2 not in self.param.at or bt1 not in self.param.bt or bt2 not in self.param.bt:
            alt1 = self.alt_ptype[pt1] if pt1 in self.alt_ptype.keys() else None
            alt2 = self.alt_ptype[pt] if pt in self.alt_ptype.keys() else None
            alt3 = self.alt_ptype[pt2] if pt2 in self.alt_ptype.keys() else None
            if alt1 is None and alt2 is None and alt3 is None:
                utils.radon_print(('Can not estimate parameters of this angle %s,%s,%s' % (pt1, pt, pt2)), level=2)
                return False

            if at1 not in self.param.at:
                if alt1 and '%s,%s,%s' % (alt1, pt, alt1) in self.param.at:
                    at1 = '%s,%s,%s' % (alt1, pt, alt1)
                elif alt2 and '%s,%s,%s' % (pt1, alt2, pt1) in self.param.at:
                    at1 = '%s,%s,%s' % (pt1, alt2, pt1)
                elif alt1 and alt2 and '%s,%s,%s' % (alt1, alt2, alt1) in self.param.at:
                    at1 = '%s,%s,%s' % (alt1, alt2, alt1)
                else:
                    utils.radon_print(('Can not estimate parameters of this angle %s,%s,%s' % (pt1, pt, pt2)), level=2)
                    return False

            if at2 not in self.param.at:
                if alt3 and '%s,%s,%s' % (alt3, pt, alt3) in self.param.at:
                    at2 = '%s,%s,%s' % (alt3, pt, alt3)
                elif alt2 and '%s,%s,%s' % (pt2, alt2, pt2) in self.param.at:
                    at2 = '%s,%s,%s' % (pt2, alt2, pt2)
                elif alt3 and alt2 and '%s,%s,%s' % (alt3, alt2, alt3) in self.param.at:
                    at2 = '%s,%s,%s' % (alt3, alt2, alt3)
                else:
                    utils.radon_print(('Can not estimate parameters of this angle %s,%s,%s' % (pt1, pt, pt2)), level=2)
                    return False

            if bt1 not in self.param.bt:
                if alt1 and '%s,%s' % (alt1, pt) in self.param.bt:
                    bt1 = '%s,%s' % (alt1, pt)
                elif alt2 and '%s,%s' % (pt1, alt2) in self.param.bt:
                    bt1 = '%s,%s' % (pt1, alt2)
                elif alt1 and alt2 and '%s,%s' % (alt1, alt2) in self.param.bt:
                    bt1 = '%s,%s' % (alt1, alt2)
                else:
                    utils.radon_print(('Can not estimate parameters of this angle %s,%s,%s' % (pt1, pt, pt2)), level=2)
                    return False

            if bt2 not in self.param.bt:
                if alt2 and '%s,%s' % (alt2, pt2) in self.param.bt:
                    bt2 = '%s,%s' % (alt2, pt2)
                elif alt3 and '%s,%s' % (pt, alt3) in self.param.bt:
                    bt2 = '%s,%s' % (pt, alt3)
                elif alt2 and alt3 and '%s,%s' % (alt2, alt3) in self.param.bt:
                    bt2 = '%s,%s' % (alt2, alt3)
                else:
                    utils.radon_print(('Can not estimate parameters of this angle %s,%s,%s' % (pt1, pt, pt2)), level=2)
                    return False

        emp_theta = (self.param.at[at1].theta0 + self.param.at[at2].theta0)/2
        emp_k_ang = (143.9*param_Z[a.GetSymbol()]*param_C[b.GetSymbol()]*param_Z[c.GetSymbol()]
                    / (self.param.bt[bt1].r0 + self.param.bt[bt2].r0) / np.sqrt(emp_theta*np.pi/180)
                    * np.exp(-2*(self.param.bt[bt1].r0 - self.param.bt[bt2].r0)**2/(self.param.bt[bt1].r0 + self.param.bt[bt2].r0)**2) )

        angle = utils.Angle(
            a=a.GetIdx(), b=b.GetIdx(), c=c.GetIdx(),
            ff=self.Angle_ff(
                ff_type = '%s,%s,%s' % (pt1, pt, pt2),
                k = emp_k_ang,
                theta0 = emp_theta
            )
        )
        
        mol.angles.append(angle)

        utils.radon_print('Using empirical angle parameters theta0 = %f, k_angle = %f for %s,%s,%s'
                    % (emp_theta, emp_k_ang, pt1, pt, pt2), level=1)

        return True
        

    def set_atype(self, mol, a, b, c, at):
        if at not in self.param.at:
            return False
    
        angle = utils.Angle(
            a=a, b=b, c=c,
            ff=ff_class.Angle_harmonic(
                ff_type=self.param.at[at].tag,
                k=self.param.at[at].k,
                theta0=self.param.at[at].theta0
            )
        )
        
        mol.angles.append(angle)
        
        return True


    def assign_dtypes(self, mol):
        """
        GAFF.assign_dtypes

        GAFF specific dihedral typing rules.
        
        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        result_flag = True
        mol.SetProp('dihedral_style', self.dihedral_style)
        alt_ptype = self.alt_ptype
        setattr(mol, 'dihedrals', [])
        
        for b in mol.GetBonds():
            p1 = b.GetBeginAtom()
            p2 = b.GetEndAtom()
            for p1b in p1.GetNeighbors():
                for p2b in p2.GetNeighbors():
                    if p1.GetIdx() == p2b.GetIdx() or p2.GetIdx() == p1b.GetIdx() or p1b.GetIdx() == p2b.GetIdx(): continue
                    unique = True
                    atoms = [p1b, p1, p2, p2b]
                    for dih in mol.dihedrals:
                        if ((dih.a == p1b.GetIdx() and dih.b == p1.GetIdx() and
                             dih.c == p2.GetIdx() and dih.d == p2b.GetIdx()) or
                            (dih.d == p1b.GetIdx() and dih.c == p1.GetIdx() and
                             dih.b == p2.GetIdx() and dih.a == p2b.GetIdx())):
                            unique = False
                    if unique:
                        p1bt = p1b.GetProp('ff_type')
                        p1t = p1.GetProp('ff_type')
                        p2t = p2.GetProp('ff_type')
                        p2bt = p2b.GetProp('ff_type')
                        dt = '%s,%s,%s,%s' % (p1bt, p1t, p2t, p2bt)
                        
                        result = self.set_dtype(mol, a=p1b.GetIdx(), b=p1.GetIdx(), c=p2.GetIdx(), d=p2b.GetIdx(), dt=dt)
                        
                        if not result:
                            alt1 = alt_ptype[p1t] if p1t in alt_ptype.keys() else None
                            alt2 = alt_ptype[p2t] if p2t in alt_ptype.keys() else None
                            if alt1 is None and alt2 is None:
                                utils.radon_print('Can not assign this dihedral %s,%s,%s,%s' % (p1bt, p1t, p2t, p2bt), level=2)
                                result_flag = False
                                continue
                            
                            dt_alt = []
                            if alt1: dt_alt.append('%s,%s,%s,%s' % (p1bt, alt1, p2t, p2bt))
                            if alt2: dt_alt.append('%s,%s,%s,%s' % (p1bt, p1t, alt2, p2bt))
                            if alt1 and alt2: dt_alt.append('%s,%s,%s,%s' % (p1bt, alt1, alt2, p2bt))
                            
                            for dt in dt_alt:
                                result = self.set_dtype(mol, a=p1b.GetIdx(), b=p1.GetIdx(), c=p2.GetIdx(), d=p2b.GetIdx(), dt=dt)
                                if result:
                                    utils.radon_print('Using alternate dihedral type %s instead of %s,%s,%s,%s' % (dt, p1bt, p1t, p2t, p2bt))
                                    break
                                    
                            if not result:
                                utils.radon_print(('Can not assign this dihedral %s,%s,%s,%s' % (p1bt, p1t, p2t, p2bt)), level=2)
                                result_flag = False
        
        return result_flag


    def set_dtype(self, mol, a, b, c, d, dt):
        if dt not in self.param.dt:
            pt = dt.split(',')
            dt1 = 'X,%s,%s,X' % (pt[1], pt[2])
            dt2 = 'X,%s,%s,%s' % (pt[1], pt[2], pt[3])
            dt3 = '%s,%s,%s,X' % (pt[0], pt[1], pt[2])
            if dt1 in self.param.dt:
                dt = dt1
            elif dt2 in self.param.dt:
                dt = dt2
            elif dt3 in self.param.dt:
                dt = dt3
            else:
                return False

        dihedral = utils.Dihedral(
            a=a, b=b, c=c, d=d,
            ff=ff_class.Dihedral_fourier(
                ff_type=self.param.dt[dt].tag,
                k=self.param.dt[dt].k,
                d0=self.param.dt[dt].d,
                m=self.param.dt[dt].m,
                n=self.param.dt[dt].n
            )
        )
        
        mol.dihedrals.append(dihedral)
        
        return True


    def assign_itypes(self, mol):
        """
        GAFF.assign_itypes

        GAFF specific improper typing rules.

        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        mol.SetProp('improper_style', self.improper_style)
        alt_ptype = self.alt_ptype
        setattr(mol, 'impropers', [])
        
        for p in mol.GetAtoms():
            if len(p.GetNeighbors()) == 3:
                for perm in permutations(p.GetNeighbors(), 3):
                    pt = p.GetProp('ff_type')
                    p1t = perm[0].GetProp('ff_type')
                    p2t = perm[1].GetProp('ff_type')
                    p3t = perm[2].GetProp('ff_type')
                    it = '%s,%s,%s,%s' % (pt, p1t, p2t, p3t)
                    
                    result = self.set_itype(mol, a=p.GetIdx(), b=perm[0].GetIdx(), c=perm[1].GetIdx(), d=perm[2].GetIdx(), it=it)
                    
                    if not result:
                        alt1 = alt_ptype[pt] if pt in alt_ptype.keys() else None
                        alt2 = alt_ptype[p1t] if p1t in alt_ptype.keys() else None
                        alt3 = alt_ptype[p2t] if p2t in alt_ptype.keys() else None
                        alt4 = alt_ptype[p3t] if p3t in alt_ptype.keys() else None
                        if alt1 is None and alt2 is None and alt3 is None and alt4 is None:
                            break
                        
                        it_alt = []
                        if alt1: it_alt.append('%s,%s,%s,%s' % (alt1, p1t, p2t, p3t))
                        if alt2: it_alt.append('%s,%s,%s,%s' % (pt, alt2, p2t, p3t))
                        if alt3: it_alt.append('%s,%s,%s,%s' % (pt, p1t, alt3, p3t))
                        if alt4: it_alt.append('%s,%s,%s,%s' % (pt, p1t, p2t, alt4))

                        if alt1 and alt2: it_alt.append('%s,%s,%s,%s' % (alt1, alt2, p2t, p3t))
                        if alt1 and alt3: it_alt.append('%s,%s,%s,%s' % (alt1, p1t, alt3, p3t))
                        if alt1 and alt4: it_alt.append('%s,%s,%s,%s' % (alt1, p1t, p2t, alt4))
                        if alt2 and alt3: it_alt.append('%s,%s,%s,%s' % (pt, alt2, alt3, p3t))
                        if alt2 and alt4: it_alt.append('%s,%s,%s,%s' % (pt, alt2, p2t, alt4))
                        if alt3 and alt4: it_alt.append('%s,%s,%s,%s' % (pt, p1t, alt3, alt4))
                        
                        if alt1 and alt2 and alt3: it_alt.append('%s,%s,%s,%s' % (alt1, alt2, alt3, p3t))
                        if alt1 and alt2 and alt4: it_alt.append('%s,%s,%s,%s' % (alt1, alt2, p2t, alt4))
                        if alt1 and alt3 and alt4: it_alt.append('%s,%s,%s,%s' % (alt1, p1t, alt3, alt4))
                        if alt2 and alt3 and alt4: it_alt.append('%s,%s,%s,%s' % (pt, alt2, alt3, alt4))

                        if alt1 and alt2 and alt3 and alt4: it_alt.append('%s,%s,%s,%s' % (alt1, alt2, alt3, alt4))
                        
                        for it in it_alt:
                            result = self.set_itype(mol, a=p.GetIdx(), b=perm[0].GetIdx(), c=perm[1].GetIdx(), d=perm[2].GetIdx(), it=it)
                            if result:
                                utils.radon_print('Using alternate improper type %s instead of %s,%s,%s,%s' % (it, pt, p1t, p2t, p3t))
                                break
                    if result:
                        break
        
        return True            


    def set_itype(self, mol, a, b, c, d, it):
        if it not in self.param.it:
            pt = it.split(',')
            it1 = '%s,X,%s,%s' % (pt[0], pt[2], pt[3])
            it2 = '%s,X,X,%s' % (pt[0], pt[3])
            if it1 in self.param.it:
                it = it1
            elif it2 in self.param.it:
                it = it2
            else:
                return False
            
        improper = utils.Improper(
            a=a, b=b, c=c, d=d,
            ff=ff_class.Improper_cvff(
                ff_type=self.param.it[it].tag,
                k=self.param.it[it].k,
                d0=self.param.it[it].d,
                n=self.param.it[it].n
            )
        )
        
        mol.impropers.append(improper)
        
        return True


    def load_ff_json(self, json_file):
        with open(json_file) as f:
            j = json.loads(f.read())

        ff = self.Container()
        ff.pt = {}
        ff.bt = {}
        ff.at = {}
        ff.dt = {}
        ff.it = {}

        ff.ff_name = j.get('ff_name')
        ff.ff_class = j.get('ff_class')
        ff.pair_style = j.get('pair_style')
        ff.bond_style = j.get('bond_style')
        ff.angle_style = j.get('angle_style')
        ff.dihedral_style = j.get('dihedral_style')
        ff.improper_style = j.get('improper_style')
        
        for pt in j.get('particle_types'):
            pt_obj = self.Container()
            for key in pt.keys():
                setattr(pt_obj, key, pt[key])
            ff.pt[pt['name']] = pt_obj
        
        for bt in j.get('bond_types'):
            bt_obj = self.Container()
            for key in bt.keys():
                setattr(bt_obj, key, bt[key])
            ff.bt[bt['name']] = bt_obj
            ff.bt[bt['rname']] = bt_obj
        
        for at in j.get('angle_types'):
            at_obj = self.Container()
            for key in at.keys():
                setattr(at_obj, key, at[key])
            ff.at[at['name']] = at_obj
            ff.at[at['rname']] = at_obj
        
        for dt in j.get('dihedral_types'):
            dt_obj = self.Container()
            for key in dt.keys():
                setattr(dt_obj, key, dt[key])
            ff.dt[dt['name']] = dt_obj
            ff.dt[dt['rname']] = dt_obj
        
        for it in j.get('improper_types'):
            it_obj = self.Container()
            for key in it.keys():
                setattr(it_obj, key, it[key])
            ff.it[it['name']] = it_obj
        
        return ff
            
    
    class Container(object):
        pass


    ## Backward compatibility
    class Angle_ff():
        """
            GAFF.Angle_ff() object
        """
        def __init__(self, ff_type=None, k=None, theta0=None):
            self.type = ff_type
            self.k = k
            self.theta0 = theta0
            self.theta0_rad = theta0*(np.pi/180)

        def to_dict(self):
            dic = {
                'ff_type': str(self.type),
                'k': float(self.k),
                'theta0': float(self.theta0),
            }
            return dic
        
        
    class Dihedral_ff():
        """
            GAFF.Dihedral_ff() object
        """
        def __init__(self, ff_type=None, k=[], d0=[], m=None, n=[]):
            self.type = ff_type
            self.k = np.array(k)
            self.d0 = np.array(d0)
            self.d0_rad = np.array(d0)*(np.pi/180)
            self.m = m
            self.n = np.array(n)
        
        def to_dict(self):
            dic = {
                'ff_type': str(self.type),
                'k': [float(x) for x in self.k],
                'd0': [float(x) for x in self.d0],
                'm': int(self.m),
                'n': [int(x) for x in self.n],
            }
            return dic

        
    class Improper_ff():
        """
            GAFF.Improper_ff() object
        """
        def __init__(self, ff_type=None, k=None, d0=-1, n=None):
            self.type = ff_type
            self.k = k
            self.d0 = d0
            self.n = n
        
        def to_dict(self):
            dic = {
                'ff_type': str(self.type),
                'k': float(self.k),
                'd0': float(self.d0),
                'n': int(self.n),
            }
            return dic

