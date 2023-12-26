#!/usr/bin/env python3

#  Copyright (c) 2023. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

__version__ = '0.2.9'

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import os
import uuid

from radonpy.core import utils, calc, poly
from radonpy.sim import helper
from radonpy.sim.preset import eq
from radonpy.ff.gaff import GAFF
from radonpy.ff.gaff2 import GAFF2
from radonpy.ff.gaff2_mod import GAFF2_mod
try:
    from radonpy.dev.ff.dreiding import Dreiding, Dreiding_UT
    dreiding_avail = True
except ImportError:
    dreiding_avail = False


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
    retry_eq = int(os.environ.get('RadonPy_RetryEQ', 0))


    work_dir = './%s' % data['DBID']
    if not os.path.isdir(work_dir):
        os.makedirs(work_dir)
    save_dir = os.path.join(work_dir, 'analyze')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    io = helper.IO_Helper(work_dir, save_dir)
    monomer_dir = data['monomer_dir'].split(',') if data['monomer_dir'] else []
    ter_dir = data['ter_dir'] if data['ter_dir'] else save_dir
    ratio = [float(x) for x in str(data['copoly_ratio_list']).split(',')]

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
    elif data['forcefield'] == 'Dreiding' and dreiding_avail:
        ff = Dreiding()
    elif data['forcefield'] == 'Dreiding_UT' and dreiding_avail:
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
        
        homopoly = poly.polymerize_rw(mols[0], n, tacticity=data['input_tacticity'])
        homopoly = poly.terminate_rw(homopoly, ter, ter2)
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
        mp = min([max([1,omp*mpi,omp,mpi]), data['input_nchain'], 60])
        copoly_list = poly.random_copolymerize_rw_mp(mols, n, ratio=ratio, tacticity=data['input_tacticity'],
                                                     nchain=data['input_nchain'], mp=mp)
        for i in range(data['input_nchain']):
            copoly_list[i] = poly.terminate_rw(copoly_list[i], ter, ter2)

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
        
        copoly = poly.copolymerize_rw(mols, n, tacticity=data['input_tacticity'])
        copoly = poly.terminate_rw(copoly, ter, ter2)
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
        
        copoly = poly.block_copolymerize_rw(mols, n_list, tacticity=data['input_tacticity'])
        copoly = poly.terminate_rw(copoly, ter, ter2)
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

    # Reload monomer csv data
    data = io.load_monomer_csv(share_dir=monomer_dir, data_dict=data)

    # Calculate refractive index
    polarizability = [data[x] for x in data.keys() if 'qm_polarizability_monomer' in str(x)]
    mol_weight = [data[x] for x in data.keys() if 'mol_weight_monomer' in str(x)]
    if len(polarizability) > 0 and len(mol_weight) > 0:
        prop_data['refractive_index'] = calc.refractive_index(polarizability, prop_data['density'], mol_weight, ratio=ratio)

    data['check_eq'] = result
    data['do_TC'] = result
    if not result:
        data['remarks'] += '[ERROR: Did not reach an equilibrium state.]'

    if prop_data['nematic_order_parameter'] >= 0.1:
        data['remarks'] += '[ERROR: The system is partially oriented.]'
        data['do_TC'] = False

    # Data output after equilibration MD
    io.output_md_data({**data, **prop_data})

