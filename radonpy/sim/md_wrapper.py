#  Copyright (c) 2023. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# ******************************************************************************
# sim.md_wrapper module
# ******************************************************************************

from ..core import utils
from . import lammps

# Try to load unpublished modules
try:
    from ..dev.sim import gromacs
    gromacs_avail = True
except ImportError:
    gromacs_avail = False


__version__ = '0.2.9'


def MD_solver(md_solver='lammps', work_dir=None, solver_path=None, **kwargs):
    md_solver = md_solver.lower()

    if md_solver == 'lammps':
        return lammps.LAMMPS(work_dir=work_dir, solver_path=solver_path, **kwargs)
            
    elif md_solver == 'gromacs':
        if gromacs_avail:
            return gromacs.Gromacs(work_dir=work_dir, solver_path=solver_path, **kwargs)
        else:
            utils.radon_print('Gromacs is not available.', level=3)



def MD_analyzer(md_analyzer='lammps', **kwargs):
    md_analyzer = md_analyzer.lower()

    if md_analyzer == 'lammps':
        return lammps.Analyze(**kwargs)
            
    elif md_analyzer == 'gromacs':
        if gromacs_avail:
            return gromacs.Analyze(**kwargs)
        else:
            utils.radon_print('Gromacs is not available.', level=3)


