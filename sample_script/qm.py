#!/usr/bin/env python3

#  Copyright (c) 2022. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

__version__ = '0.2.1'

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import os
import platform
import radonpy

from radonpy.core import const
const.mpi4py_avail = os.environ.get('RadonPy_mpi4py', False) == 'True'
#const.mpi_cmd = 'mpiexec -stdout ./%%n.%%j.out -stderr ./%%n.%%j.err -n %i'
#const.check_package_disable = True

from radonpy.core import utils, calc
from radonpy.ff.gaff2_mod import GAFF2_mod
from radonpy.sim import qm


if __name__ == '__main__':
    data = {
        'DBID': os.environ.get('RadonPy_DBID'),
        'monomer_ID': os.environ.get('RadonPy_Monomer_ID', None),
        'smiles_list': os.environ.get('RadonPy_SMILES'),
        'smiles_ter_1': os.environ.get('RadonPy_SMILES_TER', '*C'),
        'ter_ID_1': os.environ.get('RadonPy_TER_ID', 'CH3'),
        'qm_method': os.environ.get('RadonPy_QM_Method', 'wb97m-d3bj'),
        'charge': os.environ.get('RadonPy_Charge', 'RESP'),
        'remarks': os.environ.get('RadonPy_Remarks', ''),
        'Python_ver': platform.python_version(),
        'RadonPy_ver': radonpy.__version__,
    }

    omp_psi4 = int(os.environ.get('RadonPy_OMP_Psi4', 4))
    mem_psi4 = int(os.environ.get('RadonPy_MEM_Psi4', 1000))

    conf_mm_omp = int(os.environ.get('RadonPy_Conf_MM_OMP', 0))
    conf_mm_mpi = int(os.environ.get('RadonPy_Conf_MM_MPI', utils.cpu_count()))
    conf_mm_gpu = int(os.environ.get('RadonPy_Conf_MM_GPU', 0))
    conf_mm_mp = int(os.environ.get('RadonPy_Conf_MM_MP', 0))
    conf_psi4_omp = int(os.environ.get('RadonPy_Conf_Psi4_OMP', omp_psi4))
    conf_psi4_mp = int(os.environ.get('RadonPy_Conf_Psi4_MP', 0))


    work_dir = './%s' % data['DBID']
    if not os.path.isdir(work_dir):
        os.makedirs(work_dir)
    save_dir = os.path.join(work_dir, 'analyze')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    tmp_dir = os.environ.get('RadonPy_TMP_Dir', work_dir)
    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir)

    smi_list = data['smiles_list'].split(',')
    if data['monomer_ID']: monomer_id = data['monomer_ID'].split(',')

    ff = GAFF2_mod()
    mols = []

    for i, smi in enumerate(smi_list):
        monomer_data = {
            'smiles': smi,
            'qm_method': data['qm_method'],
            'charge': data['charge'],
            'remarks': data['remarks'],
            'Python_ver': data['Python_ver'],
            'RadonPy_ver': data['RadonPy_ver'],
        }
        data['smiles_%i' % (i+1)] = smi

        # Conformation search and RESP charge calculation of a repeating unit
        mol = utils.mol_from_smiles(smi)
        mol, energy = qm.conformation_search(mol, ff=ff, work_dir=work_dir, tmp_dir=tmp_dir, opt_method=data['qm_method'],
            psi4_omp=conf_psi4_omp, psi4_mp=conf_psi4_mp, mpi=conf_mm_mpi, omp=conf_mm_omp, gpu=conf_mm_gpu,
            mm_mp=conf_mm_mp, log_name='monomer%i' % (i+1), memory=mem_psi4)
        qm.assign_charges(mol, charge=data['charge'], work_dir=work_dir, tmp_dir=tmp_dir, omp=omp_psi4, opt=False, log_name='monomer%i' % (i+1), memory=mem_psi4)
        mols.append(mol)
        if data['monomer_ID']:
            data['monomer_ID_%i' % (i+1)] = monomer_data['monomer_ID'] = monomer_id[i]
            utils.pickle_dump(mol, os.path.join(save_dir, 'monomer_%s.pickle' % monomer_id[i]))
        else:
            utils.pickle_dump(mol, os.path.join(save_dir, 'monomer%i.pickle' % (i+1)))

        # Get monomer properties
        data['mol_weight_monomer%i' % (i+1)] = monomer_data['mol_weight'] = calc.molecular_weight(mol)
        data['vdw_volume_monomer%i' % (i+1)] = monomer_data['vdw_volume'] = calc.vdw_volume(mol)
        qm_data = qm.sp_prop(mol, opt=False, work_dir=work_dir, tmp_dir=tmp_dir, sp_method=data['qm_method'],
                            omp=omp_psi4, log_name='monomer%i' % (i+1), memory=mem_psi4)
        polar_data = qm.polarizability(mol, opt=False, work_dir=work_dir, tmp_dir=tmp_dir, polar_method=data['qm_method'], omp=conf_psi4_omp, mp=conf_psi4_mp,
                                        log_name='monomer%i' % (i+1), memory=mem_psi4)
        qm_data.update(polar_data)
        for k in qm_data.keys(): data['%s_monomer%i' % (k, i+1)] = qm_data[k]

        monomer_data.update(qm_data)
        monomer_df = pd.DataFrame(monomer_data, index=[0])
        if data['monomer_ID']:
            monomer_df = monomer_df.set_index('monomer_ID')
            monomer_df.to_csv(os.path.join(save_dir, 'monomer_%s_data.csv' % monomer_id[i]))
        else:
            monomer_df.to_csv(os.path.join(save_dir, 'monomer%i_data.csv' % (i+1)))


    # RESP charge calculation of a termination unit
    ter = utils.mol_from_smiles(data['smiles_ter_1'])
    qm.assign_charges(ter, charge=data['charge'], work_dir=work_dir, tmp_dir=tmp_dir, opt_method=data['qm_method'], omp=omp_psi4, log_name='ter1', memory=mem_psi4)
    if data['ter_ID_1']:
        utils.pickle_dump(ter, os.path.join(save_dir, 'ter_%s.pickle' % data['ter_ID_1']))
    else:
        utils.pickle_dump(ter, os.path.join(save_dir, 'ter1.pickle'))

    # Input data and monomer properties are outputted
    data_df = pd.DataFrame(data, index=[0]).set_index('DBID')
    data_df.to_csv(os.path.join(save_dir, 'qm_data.csv'))

