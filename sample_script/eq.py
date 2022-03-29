#!/usr/bin/env python3

__version__ = '0.2.0b3'

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import os
import platform
import radonpy

# For Fugaku
#from radonpy.core import const
#const.mpi_cmd = 'mpiexec -stdout ./%%n.%%j.out -stderr ./%%n.%%j.err -n %i'

from radonpy.core import utils, calc, poly
from radonpy.ff.gaff2_mod import GAFF2_mod
from radonpy.sim.preset import eq


if __name__ == '__main__':
    data = {
        'DBID': os.environ.get('RadonPy_DBID'),
        'monomer_ID': os.environ.get('RadonPy_Monomer_ID', None),
        'ter_ID_1': os.environ.get('RadonPy_TER_ID', 'CH3'),
        'smiles_list': os.environ.get('RadonPy_SMILES'),
        'monomer_dir': os.environ.get('RadonPy_Monomer_Dir', None),
        'copoly_ratio_list': os.environ.get('RadonPy_Copoly_Ratio', '1'),
        'copoly_type': os.environ.get('RadonPy_Copoly_Type', 'random'),
        'input_natom': int(os.environ.get('RadonPy_NAtom', 1000)),
        'input_nchain': int(os.environ.get('RadonPy_NChain', 10)),
        'ini_density': float(os.environ.get('RadonPy_Ini_Density', 0.05)),
        'temp': float(os.environ.get('RadonPy_Temp', 300.0)),
        'press': float(os.environ.get('RadonPy_Press', 1.0)),
        'input_tacticity': os.environ.get('RadonPy_Tacticity', 'atactic'),
        'tacticity': '',
        'remarks': os.environ.get('RadonPy_Remarks', ''),
        'Python_ver': platform.python_version(),
        'RadonPy_ver': radonpy.__version__,
        'preset_eq_ver': eq.__version__,
        'check_eq': False,
        'check_tc': False
    }

    omp = int(os.environ.get('RadonPy_OMP', 1))
    mpi = int(os.environ.get('RadonPy_MPI', utils.cpu_count()))
    gpu = int(os.environ.get('RadonPy_GPU', 0))
    retry_eq = int(os.environ.get('RadonPy_RetryEQ', 3))


    work_dir = './%s' % data['DBID']
    if not os.path.isdir(work_dir):
        os.makedirs(work_dir)
    save_dir = os.path.join(work_dir, 'analyze')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    monomer_dir = data['monomer_dir'] if data['monomer_dir'] else None
    smi_list = data['smiles_list'].split(',')
    ratio = [float(x) for x in str(data['copoly_ratio_list']).split(',')]
    if data['monomer_ID']: monomer_id = data['monomer_ID'].split(',')

    # Load shared monomer_data.csv file
    if data['monomer_dir'] and data['monomer_ID']:
        for i, mid in enumerate(monomer_id):
            monomer_df = pd.read_csv(os.path.join(monomer_dir, 'monomer_%s_data.csv' % mid), index_col=0)
            monomer_data = monomer_df.iloc[0].to_dict()
            data['smiles_%i' % (i+1)] = monomer_data.pop('smiles')
            data['monomer_ID_%i' % (i+1)] = mid
            data['copoly_ratio_%i' % (i+1)] = ratio[i]
            for k in monomer_data.keys(): data['%s_monomer%i' % (k, i+1)] = monomer_data[k]

    # Load qm_data.csv file
    elif os.path.isfile(os.path.join(save_dir, 'qm_data.csv')):
        qm_df = pd.read_csv(os.path.join(save_dir, 'qm_data.csv'), index_col=0)
        qm_data = qm_df.iloc[0].to_dict()
        data = {**qm_data, **data}
        if monomer_dir is None: monomer_dir = save_dir
    elif os.path.isfile(os.path.join(work_dir, 'qm_data.csv')):
        qm_df = pd.read_csv(os.path.join(work_dir, 'qm_data.csv'), index_col=0)
        qm_data = qm_df.iloc[0].to_dict()
        data = {**qm_data, **data}
        if monomer_dir is None: monomer_dir = work_dir
    else:
        print('ERROR: Cannot find monomer data.')

    # Load monomer pickle file
    mols = []
    for i in range(len(smi_list)):
        if data['monomer_ID']:
            mol = utils.pickle_load(os.path.join(monomer_dir, 'monomer_%s.pickle' % (monomer_id[i])))
        else:
            mol = utils.pickle_load(os.path.join(monomer_dir, 'monomer%i.pickle' % (i+1)))
        mols.append(mol)

    if data['ter_ID_1']:
        ter = utils.pickle_load(os.path.join(monomer_dir, 'ter_%s.pickle' % data['ter_ID_1']))
    else:
        ter = utils.pickle_load(os.path.join(monomer_dir, 'ter1.pickle'))


    ff = GAFF2_mod()
    n = poly.calc_n_from_num_atoms(mols, data['input_natom'], ratio=ratio, terminal1=ter)
    data['DP'] = n

    # Generate homopolymer chain
    if len(mols) == 1:
        data['copoly_ratio_list'] = '1'
        ratio = [1]
        data['copoly_type'] = ''
        
        homopoly = poly.polymerize_rw(mols[0], n, tacticity=data['input_tacticity'])
        homopoly = poly.terminate_rw(homopoly, ter)
        data['tacticity'] = poly.get_tacticity(homopoly)

        # Force field assignment
        result = ff.ff_assign(homopoly)
        if not result:
            data['remarks'] += '[ERROR: Can not assign force field parameters.]'
        utils.pickle_dump(homopoly, os.path.join(save_dir, 'polymer.pickle'))

        # Generate amorphous cell
        ac = poly.amorphous_cell(homopoly, data['input_nchain'], density=data['ini_density'])

    # Generate random copolymer chain
    elif len(mols) > 1 and data['copoly_type'] == 'random':
        copoly_list = poly.random_copolymerize_rw_mp(mols, n, ratio=ratio, tacticity=data['input_tacticity'],
                                                     nchain=data['input_nchain'], mp=min([omp*mpi, data['input_nchain'], 60]))
        for i in range(data['input_nchain']):
            copoly_list[i] = poly.terminate_rw(copoly_list[i], ter)

            # Force field assignment
            result = ff.ff_assign(copoly_list[i])
            if not result:
                data['remarks'] += '[ERROR: Can not assign force field parameters.]'
            utils.pickle_dump(copoly_list[i], os.path.join(save_dir, 'polymer%i.pickle' % i))

        data['tacticity'] = poly.get_tacticity(copoly_list[0])

        # Generate amorphous cell
        ac = poly.amorphous_mixture_cell(copoly_list, [1]*data['input_nchain'], density=data['ini_density'])

    # Generate alternating copolymer chain
    elif len(mols) > 1 and data['copoly_type'] == 'alternating':
        ratio = [1/len(mols)]*len(mols)
        data['copoly_ratio_list'] = ','.join([str(x) for x in ratio])
        n = poly.calc_n_from_num_atoms(mols, data['input_natom'], ratio=ratio, terminal1=ter)
        n = round(n/len(mols))
        data['DP'] = n
        
        copoly = poly.copolymerize_rw(mols, n, tacticity=data['input_tacticity'])
        copoly = poly.terminate_rw(copoly, ter)
        data['tacticity'] = poly.get_tacticity(copoly)

        # Force field assignment
        result = ff.ff_assign(copoly)
        if not result:
            data['remarks'] += '[ERROR: Can not assign force field parameters.]'
        utils.pickle_dump(copoly, os.path.join(save_dir, 'polymer.pickle'))
        
        # Generate amorphous cell
        ac = poly.amorphous_cell(copoly, data['input_nchain'], density=data['ini_density'])

    # Generate block copolymer chain
    elif len(mols) > 1 and data['copoly_type'] == 'block':
        n_list = [round(n*(x/sum(ratio))) for x in ratio]
        
        copoly = poly.block_copolymerize_rw(mols, n_list, tacticity=data['input_tacticity'])
        copoly = poly.terminate_rw(copoly, ter)
        data['tacticity'] = poly.get_tacticity(copoly)

        # Force field assignment
        result = ff.ff_assign(copoly)
        if not result:
            data['remarks'] += '[ERROR: Can not assign force field parameters.]'
        utils.pickle_dump(copoly, os.path.join(save_dir, 'polymer.pickle'))
        
        # Generate amorphous cell
        ac = poly.amorphous_cell(copoly, data['input_nchain'], density=data['ini_density'])
    
    utils.pickle_dump(ac, os.path.join(save_dir, 'amorphous.pickle'))

    # Input data and monomer properties are outputted
    poly_stats_df = poly.polymer_stats(ac, df=True)
    data_df = pd.concat([pd.DataFrame(data, index=[0]), poly_stats_df], axis=1).set_index('DBID')
    data_df.to_csv(os.path.join(save_dir, 'input_data.csv'))

    # Equilibration MD
    eqmd = eq.EQ21step(ac, work_dir=work_dir)
    ac = eqmd.exec(temp=data['temp'], press=data['press'], mpi=mpi, omp=omp, gpu=gpu)
    analy = eqmd.analyze()
    prop_data = analy.get_all_prop(temp=data['temp'], press=data['press'], save=True)
    result = analy.check_eq()

    # Additional equilibration MD
    for i in range(retry_eq):
        if result: break
        eqmd = eq.Additional(ac, work_dir=work_dir)
        ac = eqmd.exec(temp=data['temp'], press=data['press'], mpi=mpi, omp=omp, gpu=gpu)
        analy = eqmd.analyze()
        prop_data = analy.get_all_prop(temp=data['temp'], press=data['press'], save=True)
        result = analy.check_eq()

    # Calculate refractive index
    polarizability = [data[x] for x in data.keys() if 'qm_polarizability_monomer' in str(x)]
    mol_weight = [data[x] for x in data.keys() if 'mol_weight_monomer' in str(x)]
    prop_data['refractive_index'] = calc.refractive_index(polarizability, prop_data['density'], mol_weight, ratio=ratio)

    data_df.loc[data['DBID'], 'check_eq'] = result
    data_df.loc[data['DBID'], 'do_TC'] = result
    if not result:
        data_df.loc[data['DBID'], 'remarks'] += '[ERROR: Did not reach an equilibrium state.]'

    if prop_data['nematic_order_parameter'] >= 0.1:
        data_df.loc[data['DBID'], 'remarks'] += '[ERROR: The system is partially oriented.]'
        data_df.loc[data['DBID'], 'do_TC'] = False

    # Data output after equilibration MD
    eq_data_df = pd.concat([data_df, pd.DataFrame(prop_data, index=[data['DBID']])], axis=1)
    eq_data_df.index.name = 'DBID'
    eq_data_df.to_csv(os.path.join(save_dir, 'results.csv'))

