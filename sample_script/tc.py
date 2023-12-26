#!/usr/bin/env python3

#  Copyright (c) 2023. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

__version__ = '0.2.8'

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import os

from radonpy.core import utils, calc
from radonpy.sim import helper
from radonpy.sim.preset import tc


if __name__ == '__main__':
    data = {
        'DBID': os.environ.get('RadonPy_DBID'),
        'temp': float(os.environ.get('RadonPy_Temp', 300.0)),
        **helper.get_version(),
        'preset_tc_ver': tc.__version__,
        'check_tc': False
    }

    omp = int(os.environ.get('RadonPy_OMP', 1))
    mpi = int(os.environ.get('RadonPy_MPI', utils.cpu_count()))
    gpu = int(os.environ.get('RadonPy_GPU', 0))
    rst_pickle_file = os.environ.get('RadonPy_Pickle_File', None)
    rst_data_file = os.environ.get('RadonPy_LAMMPS_Data_File', None)
    tc_force = os.environ.get('RadonPy_TC_Force', False)


    work_dir = './%s' % data['DBID']
    save_dir = os.path.join(work_dir, 'analyze')
    io = helper.IO_Helper(work_dir, save_dir)

    # Load results.csv file
    data = io.load_md_csv(data)
    if not data['do_TC'] and not tc_force:
        sys.exit(0)

    # Load pickle file or LAMMPS data file
    mol = io.load_md_obj(rst_pickle_file=rst_pickle_file)

    # Non-equilibrium MD for thermal condultivity
    nemd = tc.NEMD_MP(mol, work_dir=work_dir)
    mol = nemd.exec(decomp=True, temp=data['temp'], mpi=mpi, omp=omp, gpu=gpu)
    nemd_analy = nemd.analyze()
    TC = nemd_analy.calc_tc(decomp=True, save=True)

    # Reload MD csv data
    data = io.load_md_csv(data)

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

    # Data output after NEMD
    io.output_md_data(data)

