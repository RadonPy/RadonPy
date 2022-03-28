#  Copyright (c) 2022. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# ******************************************************************************
# ff.gaff2_mod module
# ******************************************************************************

import os
from ..core.utils import radon_print
from .gaff2 import GAFF2

__version__ = '0.2.0'


class GAFF2_mod(GAFF2):
    """
    gaff2_mod.GAFF2_mod() class (base class: gaff2.GAFF2)

    Forcefield object with typing rules for modified Gaff2 model.
    By default reads data file in forcefields subdirectory.

    Added atomic type are follows:
        c3f: SP3 carbon bonded to F atom, J. Mol. Model. (2019) 25, 39

    Attributes:
        ff_name: gaff2_mod
        pair_style: lj
        bond_style: harmonic
        angle_style: harmonic
        dihedral_style: fourier
        improper_style: cvff
        ff_class: 1
    """
    def __init__(self, db_file=None):
        if db_file is None:
            db_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ff_dat', 'gaff2_mod.json')
        super().__init__(db_file)
        self.name = 'gaff2_mod'

        # Added atomic types in GAFF2_mod
        alt_ptype_gaff2_mod = {'c3f': 'c3'}
        self.alt_ptype.update(alt_ptype_gaff2_mod)


    def assign_ptypes(self, mol):
        """
        GAFF2_mod.assign_ptypes

        Modified GAFF2 specific particle typing rules.

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
                            cation = False
                            for pbb in pb.GetNeighbors():
                                if pbb.GetSymbol() in ['N', 'O', 'F', 'Cl', 'Br', 'I']: 
                                    elctrwd += 1
                                if pbb.GetSymbol() in ['N', 'P'] and pbb.GetTotalDegree() == 4:
                                    cation = True
                            if cation:
                                self.set_ptype(p, 'hx')
                            elif elctrwd == 0:
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
                                radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
                                            % (p.GetIdx(), p.GetSymbol(), p.GetTotalDegree(), str(p.GetHybridization())), level=2)
                                result_flag = False

                else:
                    radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
                                % (p.GetIdx(), p.GetSymbol(), p.GetTotalDegree(), str(p.GetHybridization())), level=2)
                    result_flag = False
                                
                                
                                
            ######################################
            # Assignment routine of C
            ######################################
            elif p.GetSymbol() == 'C':
                if str(p.GetHybridization()) == 'SP3':
                    fluoro = False
                    for pb in p.GetNeighbors():
                        if pb.GetSymbol() == 'F': fluoro = True
                    if fluoro:
                        self.set_ptype(p, 'c3f')
                    else:
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
                        elif pb.GetSymbol() == 'S':
                            for b in pb.GetBonds():
                                if (
                                    (b.GetBeginAtom().GetIdx() == p.GetIdx() and b.GetEndAtom().GetIdx() == pb.GetIdx()) or
                                    (b.GetBeginAtom().GetIdx() == pb.GetIdx() and b.GetEndAtom().GetIdx() == p.GetIdx())
                                ):
                                    if b.GetBondTypeAsDouble() == 2 and pb.GetTotalDegree() == 1:
                                        cs = True
                    if carbonyl:
                        self.set_ptype(p, 'c')  # Carbonyl carbon
                    elif cs:
                        self.set_ptype(p, 'cs')  # C=S group
                    elif p.GetIsAromatic():
                        self.set_ptype(p, 'ca')
                    else:
                        self.set_ptype(p, 'c2')  # Other sp2 carbon
                        
                elif str(p.GetHybridization()) == 'SP':
                    self.set_ptype(p, 'c1')
                    
                else:
                    radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
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
                        radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
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
                        if p.GetTotalNumHs(includeNeighbors=True) == 1:
                            self.set_ptype(p, 'ns')
                        elif p.GetTotalNumHs(includeNeighbors=True) == 2:
                            self.set_ptype(p, 'nt')
                        else:
                            self.set_ptype(p, 'n')
                        
                    elif p.GetIsAromatic():
                        self.set_ptype(p, 'na')
                    elif sp2 >= 2:
                        self.set_ptype(p, 'na')
                    elif aromatic_ring:
                        if p.GetTotalNumHs(includeNeighbors=True) == 1:
                            self.set_ptype(p, 'nu')
                        elif p.GetTotalNumHs(includeNeighbors=True) == 2:
                            self.set_ptype(p, 'nv')
                        else:
                            self.set_ptype(p, 'nh')
                        
                    else:
                        if p.GetTotalNumHs(includeNeighbors=True) == 1:
                            self.set_ptype(p, 'n7')
                        elif p.GetTotalNumHs(includeNeighbors=True) == 2:
                            self.set_ptype(p, 'n8')
                        elif p.GetTotalNumHs(includeNeighbors=True) == 3:
                            self.set_ptype(p, 'n9') # NH3
                        else:
                            self.set_ptype(p, 'n3')
                        
                        
                elif p.GetTotalDegree() == 4:
                    if p.GetTotalNumHs(includeNeighbors=True) == 1:
                        self.set_ptype(p, 'nx')
                    elif p.GetTotalNumHs(includeNeighbors=True) == 2:
                        self.set_ptype(p, 'ny')
                    elif p.GetTotalNumHs(includeNeighbors=True) == 3:
                        self.set_ptype(p, 'nz')
                    elif p.GetTotalNumHs(includeNeighbors=True) == 4:
                        self.set_ptype(p, 'n+') # NH4+
                    else:
                        self.set_ptype(p, 'n4')
                    
                else:
                    radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
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
                    radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
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
                    radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
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
                radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
                            % (p.GetIdx(), p.GetSymbol(), p.GetTotalDegree(), str(p.GetHybridization())), level=2 )
                result_flag = False


        ###########################################
        # Assignment of special atom type in GAFF2
        ###########################################
        if result_flag: self.assign_special_ptype(mol)
        
        
        return result_flag

