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
from radonpy.sim.preset import eq


if __name__ == '__main__':
    data = {
        'DBID': os.environ.get('RadonPy_DBID'),
        'temp': float(os.environ.get('RadonPy_Temp', 300.0)),
        'press': float(os.environ.get('RadonPy_Press', 1.0)),
        'remarks': os.environ.get('RadonPy_Remarks', ''),
        **helper.get_version(),
        'preset_eq_ver': eq.__version__,
        'check_tc': False
    }

    omp = int(os.environ.get('RadonPy_OMP', 1))
    mpi = int(os.environ.get('RadonPy_MPI', utils.cpu_count()))
    gpu = int(os.environ.get('RadonPy_GPU', 0))
    retry_eq = int(os.environ.get('RadonPy_RetryEQ', 2))
    retry_eq = 2 if retry_eq == 0 else retry_eq
    rst_pickle_file = os.environ.get('RadonPy_Pickle_File', None)
    rst_data_file = os.environ.get('RadonPy_LAMMPS_Data_File', None)
    skip_init_analy = bool(int(os.environ.get('RadonPy_Skip_Init_Analy', 0)))


    work_dir = './%s' % data['DBID']
    save_dir = os.path.join(work_dir, 'analyze')
    io = helper.IO_Helper(work_dir, save_dir)

    # Load results.csv or input_data.csv file
    data = io.load_md_csv(data)

    # Load pickle file or LAMMPS data file
    mol = io.load_md_obj(rst_pickle_file=rst_pickle_file)

    # Analyze the results of equilibrium MD
    if skip_init_analy:
        result = False
    else:
        last_idx = eq.get_final_idx(work_dir)
        eqmd = eq.Additional(mol, work_dir=work_dir, idx=last_idx)
        analy = eqmd.analyze()
        analy.pdb_file = os.path.join(work_dir, 'eq1.pdb')
        prop_data = analy.get_all_prop(temp=data['temp'], press=data['press'], save=True)
        result = analy.check_eq()

    # Additional equilibration MD
    for i in range(retry_eq):
        if result: break
        eqmd = eq.Additional(mol, work_dir=work_dir)
        mol = eqmd.exec(temp=data['temp'], press=data['press'], mpi=mpi, omp=omp, gpu=gpu)
        analy = eqmd.analyze()
        analy.pdb_file = os.path.join(work_dir, 'eq1.pdb')
        prop_data = analy.get_all_prop(temp=data['temp'], press=data['press'], save=True)
        result = analy.check_eq()

    # Reload MD csv data
    data = io.load_md_csv(data)

    # Calculate refractive index
    polarizability = [data[x] for x in data.keys() if 'qm_polarizability_monomer' in str(x)]
    mol_weight = [data[x] for x in data.keys() if 'mol_weight_monomer' in str(x)]
    ratio = [float(x) for x in str(data['copoly_ratio_list']).split(',')] if 'copoly_ratio_list' in data.keys() else None
    if len(polarizability) > 0 and len(mol_weight) > 0:
        prop_data['refractive_index'] = calc.refractive_index(polarizability, prop_data['density'], mol_weight, ratio=ratio)

    data.update(prop_data)
    data['check_eq'] = result
    data['do_TC'] = result
    if not result:
        data['remarks'] += '[ERROR: Did not reach an equilibrium state.]'

    if prop_data['nematic_order_parameter'] >= 0.1:
        data['remarks'] += '[ERROR: The system is partially oriented.]'
        data['do_TC'] = False

    io.output_md_data(data)

