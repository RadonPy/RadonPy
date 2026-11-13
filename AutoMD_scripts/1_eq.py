#!/usr/bin/env python3

#  Copyright (c) 2026. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

__version__ = '1.0b2'

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import os
import uuid

# For Fugaku
# from radonpy.core import const
# const.mpi_cmd = 'mpiexec -n %i'
# const.check_package_disable = True
# const.lammps_exec = '/vol0003/hp210264/data/radonpy/lammps/lmp_tuned'
# os.environ['RadonPy_LAMMPS_INTEL'] = 'off'
# os.environ['RadonPy_LAMMPS_OPT'] = 'off'
# os.environ['RadonPy_No_Traj'] = str(os.environ.get('RadonPy_No_Traj', 'True'))
# os.environ['RadonPy_Del_Traj'] = str(os.environ.get('RadonPy_Del_Traj', 'True'))

from radonpy.core import utils, poly
from radonpy.sim import helper
from radonpy.sim.preset import eq
from radonpy.ff.gaff import GAFF
from radonpy.ff.gaff2 import GAFF2
from radonpy.ff.gaff2_mod import GAFF2_mod
from radonpy.ff.dreiding import Dreiding, Dreiding_UT


if __name__ == '__main__':
    data = {
        'UUID': str(uuid.uuid4()),
        'DBID': os.environ.get('RadonPy_DBID'),
        'monomer_ID': os.environ.get('RadonPy_Monomer_ID', None),
        'ter_ID_1': os.environ.get('RadonPy_TER_ID', 'CH3'),
        'ter_ID_2': os.environ.get('RadonPy_TER_ID2', None),
        'smiles_list': os.environ.get('RadonPy_SMILES'),
        'monomer_dir': os.environ.get('RadonPy_Monomer_Dir', None),
        'ter_dir': os.environ.get('RadonPy_Ter_Dir', None),
        'copoly_ratio_list': os.environ.get('RadonPy_Copoly_Ratio', '1'),
        'copoly_type': os.environ.get('RadonPy_Copoly_Type', 'random'),
        'input_natom': int(os.environ.get('RadonPy_NAtom', 1000)),
        'input_nchain': int(os.environ.get('RadonPy_NChain', 10)),
        'ini_density': float(os.environ.get('RadonPy_Ini_Density', 0.05)),
        'temp': float(os.environ.get('RadonPy_Temp', 300.0)),
        'press': float(os.environ.get('RadonPy_Press', 1.0)),
        'input_tacticity': os.environ.get('RadonPy_Tacticity', 'atactic'),
        'tacticity': '',
        'forcefield': str(os.environ.get('RadonPy_FF', 'GAFF2_mod')),
        'remarks': os.environ.get('RadonPy_Remarks', ''),
        **helper.get_version(),
        'preset_eq_ver': eq.__version__,
        'check_eq': False,
        'check_tc': False
    }

    omp = int(os.environ.get('RadonPy_OMP', 1))
    mpi = int(os.environ.get('RadonPy_MPI', utils.cpu_count()))
    gpu = int(os.environ.get('RadonPy_GPU', 0))
    intel = os.environ.get('RadonPy_LAMMPS_INTEL', 'auto')
    opt = os.environ.get('RadonPy_LAMMPS_OPT', 'auto')
    retry_eq = int(os.environ.get('RadonPy_RetryEQ', 0))
    no_traj = bool(os.environ.get('RadonPy_No_Traj', False) == 'True')
    del_traj = bool(os.environ.get('RadonPy_Del_Traj', False) == 'True')
    # MD engine of eq presets; the default keeps the LAMMPS settings.
    # GROMACS needs RadonPy_Barostat=C-rescale (or Parrinello-Rahman); RadonPy_Interaction_Profile=portable
    # selects the validation model.
    eq_options = {
        'solver': os.environ.get('RadonPy_MD_Solver', 'lammps'),
        'interaction_profile': os.environ.get('RadonPy_Interaction_Profile') or None,
    }
    eq_kwargs = {'barostat': os.environ['RadonPy_Barostat']} if os.environ.get('RadonPy_Barostat') else {}


    work_dir = './%s' % data['DBID']
    if not os.path.isdir(work_dir):
        os.makedirs(work_dir)
    save_dir = os.path.join(work_dir, 'analyze')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    tmp_dir = os.environ.get('RadonPy_TMP_Dir', work_dir)
    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir)

    # Setting for random walk of polymer builder
    rw_setting = {
        'retry': 100,
        'rollback': 5,
        'rollback_shaking': True,
        'retry_step': 200,
        'retry_opt_step': 5,
        'mp': 4,
        'omp': 0,
        'mpi': 12,
        'gpu': 0,
        'work_dir': tmp_dir,
    }

    io = helper.IO_Helper(work_dir, save_dir)
    monomer_dir = data['monomer_dir'].split(',') if data['monomer_dir'] else []
    ter_dir = data['ter_dir'] if data['ter_dir'] else save_dir
    ratio = [float(x) for x in str(data['copoly_ratio_list']).split(',')]
    rw_setting['work_dir'] = work_dir


    # Load monomer_data.csv file
    data = io.load_monomer_csv(share_dir=monomer_dir, data_dict=data)

    # Load monomer object file
    mols = io.load_monomer_obj(share_dir=monomer_dir, data_dict=data)
    ter, ter2 = io.load_terminal_obj(share_dir=ter_dir, data_dict=data)

    if data['forcefield'] == 'GAFF':
        ff = GAFF()
    elif data['forcefield'] == 'GAFF2':
        ff = GAFF2()
    elif data['forcefield'] == 'GAFF2_mod':
        ff = GAFF2_mod()
    elif data['forcefield'] == 'Dreiding':
        ff = Dreiding()
    elif data['forcefield'] == 'Dreiding_UT':
        ff = Dreiding_UT()
    else:
        raise ValueError("Force field %s is not available." % data['forcefield'])

    n = poly.calc_n_from_num_atoms(mols, data['input_natom'], ratio=ratio, terminal1=ter, terminal2=ter2)
    data['DP'] = n

    # Generate homopolymer chain
    if len(mols) == 1:
        data['copoly_ratio_list'] = '1'
        ratio = [1]
        data['copoly_type'] = ''
        
        homopoly = poly.polymerize_rw_mp(mols[0], n, tacticity=data['input_tacticity'], ter1=ter, ter2=ter2, nchain=1, **rw_setting)[0]
        data['tacticity'] = poly.get_tacticity(homopoly)

        # Force field assignment
        result = ff.ff_assign(homopoly)
        if not result:
            data['remarks'] += '[ERROR: Can not assign force field parameters.]'
        utils.MolToJSON(homopoly, os.path.join(save_dir, 'polymer.json'))
        utils.pickle_dump(homopoly, os.path.join(save_dir, 'polymer.pickle'))

        # Generate amorphous cell
        ac = poly.amorphous_cell(homopoly, data['input_nchain'], density=data['ini_density'])

    # Generate random copolymer chain
    elif len(mols) > 1 and data['copoly_type'] == 'random':
        rw_setting['mp'] = min([max([1,rw_setting['omp']*rw_setting['mpi'],rw_setting['omp'],rw_setting['mpi']]), data['input_nchain'], 60])
        copoly_list = poly.random_copolymerize_rw_mp(mols, n, ratio=ratio, tacticity=data['input_tacticity'], ter1=ter, ter2=ter2,
                                                     nchain=data['input_nchain'], **rw_setting)
        for i in range(data['input_nchain']):
            # Force field assignment
            result = ff.ff_assign(copoly_list[i])
            if not result:
                data['remarks'] += '[ERROR: Can not assign force field parameters.]'
            utils.MolToJSON(copoly_list[i], os.path.join(save_dir, 'polymer%i.json' % i))
            utils.pickle_dump(copoly_list[i], os.path.join(save_dir, 'polymer%i.pickle' % i))

        data['tacticity'] = poly.get_tacticity(copoly_list[0])

        # Generate amorphous cell
        ac = poly.amorphous_mixture_cell(copoly_list, [1]*data['input_nchain'], density=data['ini_density'])

    # Generate alternating copolymer chain
    elif len(mols) > 1 and data['copoly_type'] == 'alternating':
        ratio = [1/len(mols)]*len(mols)
        data['copoly_ratio_list'] = ','.join([str(x) for x in ratio])
        n = poly.calc_n_from_num_atoms(mols, data['input_natom'], ratio=ratio, terminal1=ter, terminal2=ter2)
        n = round(n/len(mols))
        data['DP'] = n
        
        copoly = poly.copolymerize_rw_mp(mols, n, tacticity=data['input_tacticity'], ter1=ter, ter2=ter2, nchain=1, **rw_setting)[0]
        data['tacticity'] = poly.get_tacticity(copoly)

        # Force field assignment
        result = ff.ff_assign(copoly)
        if not result:
            data['remarks'] += '[ERROR: Can not assign force field parameters.]'
        utils.MolToJSON(copoly, os.path.join(save_dir, 'polymer.json'))
        utils.pickle_dump(copoly, os.path.join(save_dir, 'polymer.pickle'))
        
        # Generate amorphous cell
        ac = poly.amorphous_cell(copoly, data['input_nchain'], density=data['ini_density'])

    # Generate block copolymer chain
    elif len(mols) > 1 and data['copoly_type'] == 'block':
        n_list = [round(n*(x/sum(ratio))) for x in ratio]
        
        copoly = poly.block_copolymerize_rw_mp(mols, n_list, tacticity=data['input_tacticity'], ter1=ter, ter2=ter2, nchain=1, **rw_setting)[0]
        data['tacticity'] = poly.get_tacticity(copoly)

        # Force field assignment
        result = ff.ff_assign(copoly)
        if not result:
            data['remarks'] += '[ERROR: Can not assign force field parameters.]'
        utils.MolToJSON(copoly, os.path.join(save_dir, 'polymer.json'))
        utils.pickle_dump(copoly, os.path.join(save_dir, 'polymer.pickle'))
        
        # Generate amorphous cell
        ac = poly.amorphous_cell(copoly, data['input_nchain'], density=data['ini_density'])
        
    utils.MolToJSON(ac, os.path.join(save_dir, 'amorphous.json'))
    utils.pickle_dump(ac, os.path.join(save_dir, 'amorphous.pickle'))

    # Input data and monomer properties are outputted
    poly_stats = poly.polymer_stats(ac, df=False, join=True)
    data.update(poly_stats)
    data_df = pd.DataFrame(data, index=[0]).set_index('DBID')
    data_df.to_csv(os.path.join(save_dir, 'input_data.csv'))

    # Equilibration MD
    eqmd = eq.EQ21step(ac, work_dir=work_dir, no_traj_ann=no_traj, **eq_options)
    ac = eqmd.exec(temp=data['temp'], press=data['press'], mpi=mpi, omp=omp, gpu=gpu, intel=intel, opt=opt, **eq_kwargs)
    analy = eqmd.analyze()
    prop_data = analy.get_all_prop(temp=data['temp'], press=data['press'], save=True)
    result = analy.check_eq()

    # Additional equilibration MD
    for i in range(retry_eq):
        if result: break
        eqmd = eq.Additional(ac, work_dir=work_dir, **eq_options)
        ac = eqmd.exec(temp=data['temp'], press=data['press'], mpi=mpi, omp=omp, gpu=gpu, intel=intel, opt=opt, **eq_kwargs)
        analy = eqmd.analyze()
        prop_data = analy.get_all_prop(temp=data['temp'], press=data['press'], save=True)
        result = analy.check_eq()
        if del_traj:
            eq.del_dump(work_dir)

    # Reload monomer csv data
    data = io.load_monomer_csv(share_dir=monomer_dir, data_dict=data)

    # # Calculate refractive index
    # polarizability = [data[x] for x in data.keys() if 'qm_polarizability_monomer' in str(x)]
    # mol_weight = [data[x] for x in data.keys() if 'mol_weight_monomer' in str(x)]
    # if len(polarizability) > 0 and len(mol_weight) > 0:
    #     prop_data['refractive_index'] = calc.refractive_index(polarizability, prop_data['density'], mol_weight, ratio=ratio)

    data['check_eq'] = result
    data['do_TC'] = result
    if not result:
        data['remarks'] += '[ERROR: Did not reach an equilibrium state.]'

    if prop_data['nematic_order_parameter'] >= 0.1:
        data['remarks'] += '[ERROR: The system is partially oriented.]'
        data['do_TC'] = False

    # Data output after equilibration MD
    io.output_md_data({**data, **prop_data})

