#!/usr/bin/env python3

#  Copyright (c) 2026. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

__version__ = '1.0b2'

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import os
import sys

# Preset module for solubility parameter calculation require to install "TALLY" package in LAMMPS
# For user installed LAMMPS
# from radonpy.core import const
# const.lammps_exec = '/path/to/lammps/binary'

# For Fugaku
# from radonpy.core import const
# const.mpi_cmd = 'mpiexec -n %i'
# const.check_package_disable = True
# const.lammps_exec = '/vol0003/hp210264/data/radonpy/lammps/lmp_tuned'
# os.environ['RadonPy_LAMMPS_INTEL'] = 'off'
# os.environ['RadonPy_LAMMPS_OPT'] = 'off'

from radonpy.core import utils
from radonpy.sim import helper
from radonpy.sim.preset import sp


if __name__ == '__main__':
    data = {
        'DBID': os.environ.get('RadonPy_DBID'),
        **helper.get_version(),
        'preset_sp_ver': sp.__version__,
    }
    
    omp = int(os.environ.get('RadonPy_OMP', 0))
    mpi = 16 # mpi should always be 16, 64 or 128 for rerun.
    #mpi = int(os.environ.get('RadonPy_MPI', utils.cpu_count()))
    gpu = int(os.environ.get('RadonPy_GPU', 0))
    rst_json_file = os.environ.get('RadonPy_JSON_File', None) 
    rst_pickle_file = os.environ.get('RadonPy_Pickle_File', None)
    sp_force = bool(os.environ.get('RadonPy_SP_Force', False) == 'True')
    # Same engine selection as 1_eq.py.
    sp_options = {
        'solver': os.environ.get('RadonPy_MD_Solver', 'lammps'),
        'interaction_profile': os.environ.get('RadonPy_Interaction_Profile') or None,
    }

    work_dir = './%s' % data['DBID']
    save_dir = os.path.join(work_dir, 'analyze')
    io = helper.IO_Helper(work_dir, save_dir)
    
    # Load results.csv or input_data.csv file
    data = io.load_md_csv(data)
    if not data['check_eq'] and not sp_force:
        print('check_eq: FALSE')
        sys.exit(0)

    # Load JSON file, pickle file, or LAMMPS data file
    mol = io.load_md_obj(rst_json_file=rst_json_file, rst_pickle_file=rst_pickle_file)

    # SP calculation (using rerun)
    spmd = sp.Rerun(mol, work_dir=work_dir, **sp_options)
    mol, sp_results = spmd.exec(temp=data['temp'], press=data['press'], mpi=mpi, omp=omp, gpu=gpu)    

    # Reload MD csv data
    data = io.load_md_csv(data)

    #tg_results is the dictionary including results of SPMD
    data.update(sp_results)

    # Data output after SPMD
    io.output_md_data(data)

