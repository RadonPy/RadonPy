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

# For Fugaku
# from radonpy.core import const
# const.mpi_cmd = 'mpiexec -n %i'
# const.check_package_disable = True
# const.lammps_exec = '/vol0003/hp210264/data/radonpy/lammps/lmp_tuned'
# os.environ['RadonPy_LAMMPS_INTEL'] = 'off'
# os.environ['RadonPy_LAMMPS_OPT'] = 'off'
# os.environ['RadonPy_No_Traj'] = str(os.environ.get('RadonPy_No_Traj', 'True'))

from radonpy.core import utils
from radonpy.sim import helper
from radonpy.sim.preset import tg


if __name__ == '__main__':
    data = {
        'DBID': os.environ.get('RadonPy_DBID'),
        **helper.get_version(),
        'preset_tg_ver': tg.__version__,
    }
    
    omp = int(os.environ.get('RadonPy_OMP', 0))
    mpi = int(os.environ.get('RadonPy_MPI', utils.cpu_count()))
    gpu = int(os.environ.get('RadonPy_GPU', 0))
    intel = os.environ.get('RadonPy_LAMMPS_INTEL', 'off')
    opt = os.environ.get('RadonPy_LAMMPS_OPT', 'off')
    rst_json_file = os.environ.get('RadonPy_JSON_File', None) 
    rst_pickle_file = os.environ.get('RadonPy_Pickle_File', None)
    tg_force = bool(os.environ.get('RadonPy_Tg_Force', False) == 'True')
    no_traj = bool(os.environ.get('RadonPy_No_Traj', False) == 'True')
    # Same engine selection as 1_eq.py; GROMACS needs RadonPy_Barostat=C-rescale (or Parrinello-Rahman).
    tg_options = {
        'solver': os.environ.get('RadonPy_MD_Solver', 'lammps'),
        'interaction_profile': os.environ.get('RadonPy_Interaction_Profile') or None,
    }
    tg_kwargs = {'barostat': os.environ['RadonPy_Barostat']} if os.environ.get('RadonPy_Barostat') else {}

    work_dir = './%s' % data['DBID']
    save_dir = os.path.join(work_dir, 'analyze')
    io = helper.IO_Helper(work_dir, save_dir)
    
    # Load results.csv or input_data.csv file
    data = io.load_md_csv(data)
    if not data['check_eq'] and not tg_force:
        print('check_eq: FALSE')
        sys.exit(0)

    # Load JSON file, pickle file, or LAMMPS data file
    mol = io.load_md_obj(rst_json_file=rst_json_file, rst_pickle_file=rst_pickle_file)

    # Tg calculation
    tgmd = tg.TGMD(mol, work_dir=work_dir, no_traj=no_traj, **tg_options)
    mol, tg_results = tgmd.exec(temp=data['temp'], mpi=mpi, omp=omp, gpu=gpu, cooling_rate=8e3, intel=intel, opt=opt,
                                **tg_kwargs)

    # Reload MD csv data
    data = io.load_md_csv(data)

    #tg_results is the dictionary including results of TGMD
    data.update(tg_results)

    # Data output after TGMD
    io.output_md_data(data)

