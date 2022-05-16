#!/usr/bin/env python3

#  Copyright (c) 2022. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

__version__ = '0.2.1'

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import os
import sys
import datetime
import shutil
import platform
import radonpy

# For Fugaku
#from radonpy.core import const
#const.mpi_cmd = 'mpiexec -stdout ./%%n.%%j.out -stderr ./%%n.%%j.err -n %i'

from radonpy.core import utils, calc
from radonpy.sim.preset import eq, tc


if __name__ == '__main__':
    data = {
        'DBID': os.environ.get('RadonPy_DBID'),
        'temp': float(os.environ.get('RadonPy_Temp', 300.0)),
        'Python_ver': platform.python_version(),
        'RadonPy_ver': radonpy.__version__,
        'preset_tc_ver': tc.__version__,
        'check_tc': False
    }

    omp = int(os.environ.get('RadonPy_OMP', 0))
    mpi = int(os.environ.get('RadonPy_MPI', utils.cpu_count()))
    gpu = int(os.environ.get('RadonPy_GPU', 0))
    rst_pickle_file = os.environ.get('RadonPy_Pickle_File', None)
    rst_data_file = os.environ.get('RadonPy_LAMMPS_Data_File', None)
    tc_force = os.environ.get('RadonPy_TC_Force', False)


    work_dir = './%s' % data['DBID']
    save_dir = os.path.join(work_dir, 'analyze')

    # Load results.csv file
    input_df = pd.read_csv(os.path.join(save_dir, 'results.csv'), index_col=0)
    input_data = input_df.iloc[0].to_dict()
    data = {**input_data, **data}
    if not data['do_TC'] and not tc_force:
        sys.exit(0)

    # Load pickle file or LAMMPS data file
    mol = None
    if not rst_pickle_file:
        rst_pickle_file = eq.get_final_pickle([save_dir, work_dir])
        if rst_pickle_file:
            mol = utils.pickle_load(rst_pickle_file)
    else:
        mol = utils.pickle_load(os.path.join(work_dir, rst_pickle_file))

    if mol is None:
        rst_data_file = eq.get_final_data(work_dir)
        if rst_data_file:
            mol = lammps.MolFromLAMMPSdata(rst_data_file)

    # Non-equilibrium MD for thermal condultivity
    nemd = tc.NEMD_MP(mol, work_dir=work_dir)
    mol = nemd.exec(decomp=True, temp=data['temp'], mpi=mpi, omp=omp, gpu=gpu)
    nemd_analy = nemd.analyze()
    TC = nemd_analy.calc_tc(decomp=True, save=True)

    if not nemd_analy.Tgrad_data['Tgrad_check']:
        data['remarks'] = str(data['remarks']) + '[ERROR: Low linearity of temperature gradient.]'
    else:
        data['check_tc'] = True

    prop_data = {
        'thermal_conductivity': TC,
        'thermal_diffusivity': calc.thermal_diffusivity(TC, data['density'], data['Cp']),
        **nemd_analy.TCdecomp_data
    }
    data.update(prop_data)

    # Backup results.csv file
    now = datetime.datetime.now()
    shutil.copyfile(os.path.join(save_dir, 'results.csv'),
        os.path.join(save_dir, 'results_%i-%i-%i-%i-%i-%i.csv' % (now.year, now.month, now.day, now.hour, now.minute, now.second)))

    # Data output after NEMD
    data_df = pd.DataFrame(data, index=[0]).set_index('DBID')
    data_df.to_csv(os.path.join(save_dir, 'results.csv'))

