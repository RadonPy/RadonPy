#  Copyright (c) 2022. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# ******************************************************************************
# preset.__init__ module
# ******************************************************************************

import os
import numpy as np
from ...core import utils

class Preset():
    def __init__(self, mol, prefix='', work_dir=None, save_dir=None, solver_path=None, **kwargs):
        self.mol = utils.deepcopy_mol(mol)
        self.prefix = prefix if prefix == '' else prefix+'_'
        self.work_dir = work_dir if work_dir is not None else './'
        self.save_dir = save_dir if save_dir is not None else os.path.join(self.work_dir, 'analyze')
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        self.solver_path = solver_path
        self.in_file = kwargs.get('in_file', '%seq1.in' % prefix)
        self.top_file = kwargs.get('top_file', '%seq1.data' % prefix)
        self.pdb_file = kwargs.get('pdb_file', '%stopology.pdb' % prefix)
        self.log_file = kwargs.get('log_file', '%seq1.log' % prefix)
        self.dump_file = kwargs.get('dump_file', '%seq1.dump' % prefix)
        self.xtc_file = kwargs.get('xtc_file', '%seq1.xtc' % prefix)
        self.last_str = kwargs.get('last_str', '%seq2_last.dump' % prefix)
        self.last_data = kwargs.get('last_data', '%seq2_last.data' % prefix)
        self.pair_style = kwargs.get('pair_style', 'lj/charmm/coul/long')
        self.cutoff_in = kwargs.get('cutoff_in', 8.0)
        self.cutoff_out = kwargs.get('cutoff_out', 12.0)
        self.neighbor_dis = kwargs.get('neighbor_dis', 2.0)
        self.kspace_style = kwargs.get('kspace_style', 'pppm')
        self.kspace_style_accuracy = kwargs.get('kspace_style_accuracy', '1e-6')

        self.uwstr = np.array([])
        self.wstr = np.array([])
        self.cell = np.array([])
        self.vel = np.array([])
        self.force = np.array([])

