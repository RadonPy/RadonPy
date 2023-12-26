#  Copyright (c) 2023. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# ******************************************************************************
# sim.qm_wrapper module
# ******************************************************************************

from ..core import utils
from .psi4_wrapper import Psi4w

# Try to load unpublished modules
try:
    from ..dev.sim.ntchem_wrapper import NTChemw
    ntchem_avail = True
except ImportError:
    ntchem_avail = False

try:
    from ..dev.sim.nwchem_wrapper import NWChemw
    nwchem_avail = True
except ImportError:
    nwchem_avail = False

__version__ = '0.2.9'



def QMw(mol, confId=0, work_dir=None, tmp_dir=None, name=None, qm_solver='psi4', **kwargs):

    qm_solver = qm_solver.lower()

    if qm_solver == 'ntchem':
        if ntchem_avail:
            return NTChemw(mol, confId=confId, work_dir=work_dir, tmp_dir=tmp_dir,
                           name=name, **kwargs)
        else:
            utils.radon_print('NTChem is not available.', level=3)

    elif qm_solver == 'nwchem':
        if nwchem_avail:
            return NWChemw(mol, confId=confId, work_dir=work_dir, tmp_dir=tmp_dir,
                           name=name, **kwargs)
        else:
            utils.radon_print('NWChem is not available.', level=3)

    else:
        return Psi4w(mol, confId=confId, work_dir=work_dir, tmp_dir=tmp_dir,
                     name=name, **kwargs)
