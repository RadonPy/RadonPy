#  Copyright (c) 2022. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# ******************************************************************************
# ff.gaff2_mod module
# ******************************************************************************

import os
from ..core import utils
from .gaff2 import GAFF2

__version__ = '0.3.0b1'


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


    def assign_ptypes_atom(self, p):
        """
        GAFF2_mod.assign_ptypes_atom

        Modified GAFF2 specific particle typing rules for atom.

        Args:
            p: rdkit atom object

        Returns:
            boolean
        """
        result_flag = True

        ######################################
        # Assignment routine of H
        ######################################
        if p.GetSymbol() == 'H':
            nb_sym = p.GetNeighbors()[0].GetSymbol()

            if nb_sym == 'O':
                water = False
                for pb in p.GetNeighbors():
                    if pb.GetSymbol() == 'O' and pb.GetTotalNumHs(includeNeighbors=True) == 2:
                        water = True
                if water:
                    self.set_ptype(p, 'hw')
                else:
                    self.set_ptype(p, 'ho')
                    
            elif nb_sym == 'N':
                self.set_ptype(p, 'hn')
                
            elif nb_sym == 'P':
                self.set_ptype(p, 'hp')
                
            elif nb_sym == 'S':
                self.set_ptype(p, 'hs')
                
            elif nb_sym == 'C':
                for pb in p.GetNeighbors():
                    if pb.GetSymbol() == 'C':
                        elctrwd = 0
                        cation = False
                        degree = pb.GetTotalDegree()

                        for pbb in pb.GetNeighbors():
                            pbb_degree = pbb.GetTotalDegree()
                            pbb_sym = pbb.GetSymbol()
                            if pbb_sym in self.elctrwd_elements and pbb_degree < 4: 
                                elctrwd += 1
                            if pbb_sym in self.elctropositive_elements and pbb_degree == 4:
                                cation = True
                        if cation:
                            self.set_ptype(p, 'hx')
                        elif elctrwd == 0:
                            if str(pb.GetHybridization()) == 'SP2' or str(pb.GetHybridization()) == 'SP':
                                self.set_ptype(p, 'ha')
                            else:
                                self.set_ptype(p, 'hc')
                        elif degree == 4 and elctrwd == 1:
                            self.set_ptype(p, 'h1')
                        elif degree == 4 and elctrwd == 2:
                            self.set_ptype(p, 'h2')
                        elif degree == 4 and elctrwd == 3:
                            self.set_ptype(p, 'h3')
                        elif degree == 3 and elctrwd == 1:
                            self.set_ptype(p, 'h4')
                        elif degree == 3 and elctrwd == 2:
                            self.set_ptype(p, 'h5')
                        else:
                            utils.radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
                                        % (p.GetIdx(), p.GetSymbol(), p.GetTotalDegree(), str(p.GetHybridization())), level=2)
                            result_flag = False

            else:
                utils.radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
                            % (p.GetIdx(), p.GetSymbol(), p.GetTotalDegree(), str(p.GetHybridization())), level=2)
                result_flag = False
                            
                            
                            
        ######################################
        # Assignment routine of C
        ######################################
        elif p.GetSymbol() == 'C':
            hyb = str(p.GetHybridization())

            if hyb == 'SP3':
                fluoro = False
                for pb in p.GetNeighbors():
                    if pb.GetSymbol() == 'F':
                        fluoro = True
                        break

                if p.IsInRingSize(3):
                    self.set_ptype(p, 'cx')
                elif p.IsInRingSize(4):
                    self.set_ptype(p, 'cy')
                elif fluoro:
                    self.set_ptype(p, 'c3f')
                else:
                    self.set_ptype(p, 'c3')
                
            elif hyb == 'SP2':
                p_idx = p.GetIdx()
                carbonyl = False
                cs = False
                conj = 0
                for pb in p.GetNeighbors():
                    pb_sym = pb.GetSymbol()
                    if pb_sym == 'O':
                        pb_idx = pb.GetIdx()
                        pb_degree = pb.GetTotalDegree()
                        for b in pb.GetBonds():
                            if (
                                (b.GetBeginAtom().GetIdx() == p_idx and b.GetEndAtom().GetIdx() == pb_idx) or
                                (b.GetBeginAtom().GetIdx() == pb_idx and b.GetEndAtom().GetIdx() == p_idx)
                            ):
                                if b.GetBondTypeAsDouble() == 2 and pb_degree == 1:
                                    carbonyl = True
                    elif pb_sym == 'S':
                        pb_idx = pb.GetIdx()
                        pb_degree = pb.GetTotalDegree()
                        for b in pb.GetBonds():
                            if (
                                (b.GetBeginAtom().GetIdx() == p_idx and b.GetEndAtom().GetIdx() == pb_idx) or
                                (b.GetBeginAtom().GetIdx() == pb_idx and b.GetEndAtom().GetIdx() == p_idx)
                            ):
                                if b.GetBondTypeAsDouble() == 2 and pb_degree == 1:
                                    cs = True

                for b in p.GetBonds():
                    if b.GetIsConjugated():
                        conj += 1

                if carbonyl:
                    self.set_ptype(p, 'c')  # Carbonyl carbon
                elif cs:
                    self.set_ptype(p, 'cs')  # C=S group
                elif p.GetIsAromatic():
                    self.set_ptype(p, 'ca')

                    # For biphenyl head atom
                    for b in p.GetBonds():
                        if b.GetBondTypeAsDouble() == 1:
                            bp = b.GetBeginAtom() if p_idx == b.GetEndAtom().GetIdx() else b.GetEndAtom()
                            if bp.GetIsAromatic():
                                self.set_ptype(p, 'cp')
                                if bp.GetSymbol() == 'C':
                                    self.set_ptype(bp, 'cp')

                elif p.IsInRingSize(3):
                    self.set_ptype(p, 'cu')
                elif p.IsInRingSize(4):
                    self.set_ptype(p, 'cv')
                elif conj >= 2:
                    if utils.is_in_ring(p, max_size=self.max_ring_size):
                        self.set_ptype(p, 'cc')
                    else:
                        self.set_ptype(p, 'ce')
                else:
                    self.set_ptype(p, 'c2')  # Other sp2 carbon
                    
            elif hyb == 'SP':
                conj = 0
                for b in p.GetBonds():
                    if b.GetIsConjugated():
                        conj += 1

                if conj >= 2:
                    self.set_ptype(p, 'cg')
                else:
                    self.set_ptype(p, 'c1')
                
            else:
                utils.radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
                            % (p.GetIdx(), p.GetSymbol(), p.GetTotalDegree(), str(p.GetHybridization())), level=2 )
                result_flag = False
                
                
                
        ######################################
        # Assignment routine of N
        ######################################
        elif p.GetSymbol() == 'N':
            hyb = str(p.GetHybridization())
            degree = p.GetTotalDegree()

            if hyb == 'SP':
                self.set_ptype(p, 'n1')
                
            elif degree == 2:
                bond_orders = []
                conj = 0
                for b in p.GetBonds():
                    bond_orders.append(b.GetBondTypeAsDouble())
                    if b.GetIsConjugated():
                        conj += 1

                if p.GetIsAromatic():
                    self.set_ptype(p, 'nb')
                elif 2 in bond_orders:
                    if conj >= 2:
                        if utils.is_in_ring(p, max_size=self.max_ring_size):
                            self.set_ptype(p, 'nc')
                        else:
                            self.set_ptype(p, 'ne')
                    else:
                        self.set_ptype(p, 'n2')
                else:
                    utils.radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
                                % (p.GetIdx(), p.GetSymbol(), p.GetTotalDegree(), str(p.GetHybridization())), level=2 )
                    result_flag = False
                    
            elif degree == 3:
                amide = False
                aromatic_ring = False
                no2 = 0
                sp2 = 0
                for pb in p.GetNeighbors():
                    pb_sym = pb.GetSymbol()
                    pb_hyb = str(pb.GetHybridization())

                    if pb_sym == 'C':
                        if pb.GetIsAromatic():
                            aromatic_ring = True
                        for b in pb.GetBonds():
                            bp = b.GetBeginAtom() if pb.GetIdx() == b.GetEndAtom().GetIdx() else b.GetEndAtom()
                            if (bp.GetSymbol() == 'O' or bp.GetSymbol() == 'S') and b.GetBondTypeAsDouble() == 2:
                                amide = True
                    elif pb_sym == 'O':
                        no2 += 1
                    if pb_hyb == 'SP2' or pb_hyb == 'SP':
                        sp2 += 1
                if no2 >= 2:
                    self.set_ptype(p, 'no')
                elif amide:
                    numHs = p.GetTotalNumHs(includeNeighbors=True)
                    if numHs == 1:
                        self.set_ptype(p, 'ns')
                    elif numHs == 2:
                        self.set_ptype(p, 'nt')
                    else:
                        self.set_ptype(p, 'n')
                    
                elif p.GetIsAromatic():
                    self.set_ptype(p, 'na')
                elif sp2 >= 2:
                    self.set_ptype(p, 'na')
                elif aromatic_ring:
                    numHs = p.GetTotalNumHs(includeNeighbors=True)
                    if numHs == 1:
                        self.set_ptype(p, 'nu')
                    elif numHs == 2:
                        self.set_ptype(p, 'nv')
                    else:
                        self.set_ptype(p, 'nh')
                    
                else:
                    numHs = p.GetTotalNumHs(includeNeighbors=True)
                    if numHs == 1:
                        self.set_ptype(p, 'n7')
                    elif numHs == 2:
                        self.set_ptype(p, 'n8')
                    elif numHs == 3:
                        self.set_ptype(p, 'n9') # NH3
                    else:
                        self.set_ptype(p, 'n3')
                    
                    
            elif degree == 4:
                numHs = p.GetTotalNumHs(includeNeighbors=True)
                if numHs == 1:
                    self.set_ptype(p, 'nx')
                elif numHs == 2:
                    self.set_ptype(p, 'ny')
                elif numHs == 3:
                    self.set_ptype(p, 'nz')
                elif numHs == 4:
                    self.set_ptype(p, 'n+') # NH4+
                else:
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
            p_idx = p.GetIdx()
            degree = p.GetTotalDegree()

            if p.GetIsAromatic():
                self.set_ptype(p, 'pb')
                
            elif degree == 2:
                conj = 0
                for b in p.GetBonds():
                    if b.GetIsConjugated():
                        conj += 1

                if conj >= 2:
                    if utils.is_in_ring(p, max_size=self.max_ring_size):
                        self.set_ptype(p, 'pc')
                    else:
                        self.set_ptype(p, 'pe')
                else:
                    self.set_ptype(p, 'p2')
                
            elif degree == 3:
                bond_orders = [x.GetBondTypeAsDouble() for x in p.GetBonds()]
                if 2 in bond_orders:
                    conj = False
                    for pb in p.GetNeighbors():
                        for b in pb.GetBonds():
                            if b.GetBeginAtom().GetIdx() != p_idx and b.GetEndAtom().GetIdx() != p_idx:
                                if b.GetBondTypeAsDouble() >= 1.5:
                                    conj = True
                    if conj:
                        self.set_ptype(p, 'px')
                    else:
                        self.set_ptype(p, 'p4')
                else:
                    self.set_ptype(p, 'p3')
                    
            elif degree == 4:
                conj = False
                for pb in p.GetNeighbors():
                    for b in pb.GetBonds():
                        if b.GetBeginAtom().GetIdx() != p_idx and b.GetEndAtom().GetIdx() != p_idx:
                            if b.GetBondTypeAsDouble() >= 1.5:
                                conj = True
                if conj:
                    self.set_ptype(p, 'py')
                else:
                    self.set_ptype(p, 'p5')

            elif degree == 5 or degree == 6:
                self.set_ptype(p, 'p5')
                    
            else:
                utils.radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
                            % (p.GetIdx(), p.GetSymbol(), p.GetTotalDegree(), str(p.GetHybridization())), level=2 )
                result_flag = False



        ######################################
        # Assignment routine of S
        ######################################
        elif p.GetSymbol() == 'S':
            p_idx = p.GetIdx()
            degree = p.GetTotalDegree()

            if degree == 1:
                self.set_ptype(p, 's')
                
            elif degree == 2:
                bond_orders = [x.GetBondTypeAsDouble() for x in p.GetBonds()]
                if p.GetIsAromatic():
                    self.set_ptype(p, 'ss')
                elif p.GetTotalNumHs(includeNeighbors=True) == 1:
                    self.set_ptype(p, 'sh')
                elif 2 in bond_orders:
                    self.set_ptype(p, 's2')
                else:
                    self.set_ptype(p, 'ss')
                
            elif degree == 3:
                conj = False
                for pb in p.GetNeighbors():
                    for b in pb.GetBonds():
                        if b.GetBeginAtom().GetIdx() != p_idx and b.GetEndAtom().GetIdx() != p_idx:
                            if b.GetBondTypeAsDouble() >= 1.5:
                                conj = True
                if conj:
                    self.set_ptype(p, 'sx')
                else:
                    self.set_ptype(p, 's4')
                    
            elif degree == 4:
                conj = False
                for pb in p.GetNeighbors():
                    for b in pb.GetBonds():
                        if b.GetBeginAtom().GetIdx() != p_idx and b.GetEndAtom().GetIdx() != p_idx:
                            if b.GetBondTypeAsDouble() >= 1.5:
                                conj = True
                if conj:
                    self.set_ptype(p, 'sy')
                else:
                    self.set_ptype(p, 's6')

            elif degree == 5 or degree == 6:
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
        
        return result_flag

