#!/usr/bin/env python3

__version__ = '0.2.0b3'

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import os
import datetime
import shutil
import platform
import radonpy

# For Fugaku
#from radonpy.core import const
#const.mpi_cmd = 'mpiexec -stdout ./%%n.%%j.out -stderr ./%%n.%%j.err -n %i'

from radonpy.core import utils, calc
from radonpy.sim import lammps
from radonpy.sim.preset import eq


if __name__ == '__main__':
    data = {
        'DBID': os.environ.get('RadonPy_DBID'),
        'temp': float(os.environ.get('RadonPy_Temp', 300.0)),
        'press': float(os.environ.get('RadonPy_Press', 1.0)),
        'remarks': os.environ.get('RadonPy_Remarks', ''),
        'Python_ver': platform.python_version(),
        'RadonPy_ver': radonpy.__version__,
        'preset_eq_ver': eq.__version__,
        'check_tc': False
    }

    omp = int(os.environ.get('RadonPy_OMP', 1))
    mpi = int(os.environ.get('RadonPy_MPI', utils.cpu_count()))
    gpu = int(os.environ.get('RadonPy_GPU', 0))
    retry_eq = int(os.environ.get('RadonPy_RetryEQ', 3))
    rst_pickle_file = os.environ.get('RadonPy_Pickle_File', None)
    rst_data_file = os.environ.get('RadonPy_LAMMPS_Data_File', None)


    work_dir = './%s' % data['DBID']
    save_dir = os.path.join(work_dir, 'analyze')

    # Load input_data.csv file
    if os.path.isfile(os.path.join(save_dir, 'results.csv')):
        input_df = pd.read_csv(os.path.join(save_dir, 'results.csv'), index_col=0)
    elif os.path.isfile(os.path.join(save_dir, 'input_data.csv')):
        input_df = pd.read_csv(os.path.join(save_dir, 'input_data.csv'), index_col=0)
    else:
        input_df = pd.read_csv(os.path.join(work_dir, 'input_data.csv'), index_col=0)
    input_data = input_df.iloc[0].to_dict()
    data = {**input_data, **data}

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

    # Analyze the results of equilibrium MD
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

    # Calculate refractive index
    polarizability = [data[x] for x in data.keys() if 'qm_polarizability_monomer' in str(x)]
    mol_weight = [data[x] for x in data.keys() if 'mol_weight_monomer' in str(x)]
    ratio = [float(x) for x in str(data['copoly_ratio_list']).split(',')] if 'copoly_ratio_list' in data.keys() else None
    prop_data['refractive_index'] = calc.refractive_index(polarizability, prop_data['density'], mol_weight, ratio=ratio)

    data.update(prop_data)
    data['check_eq'] = result
    data['do_TC'] = result
    if not result:
        data['remarks'] += '[ERROR: Did not reach an equilibrium state.]'

    if prop_data['nematic_order_parameter'] >= 0.1:
        data['remarks'] += '[ERROR: The system is partially oriented.]'
        data['do_TC'] = False

    if os.path.isfile(os.path.join(save_dir, 'results.csv')):
        now = datetime.datetime.now()
        shutil.copyfile(os.path.join(save_dir, 'results.csv'),
            os.path.join(save_dir, 'results_%i-%i-%i-%i-%i-%i.csv' % (now.year, now.month, now.day, now.hour, now.minute, now.second)))

    # Data output after equilibration MD
    eq_data_df = pd.DataFrame(data, index=[0]).set_index('DBID')
    eq_data_df.to_csv(os.path.join(save_dir, 'results.csv'))

