#  Copyright (c) 2022. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# ******************************************************************************
# sim.helper module
# ******************************************************************************

import os
import re
import datetime
import shutil
import platform
import importlib
import pandas as pd
import rdkit

from ..core import utils
from ..__init__ import __version__ as radonpy_ver
from .lammps import LAMMPS

psi4_ver = None
try:
    from psi4 import __version__ as psi4_ver
except ImportError:
    psi4_ver = None

__version__ = '0.2.0'


class Pipeline_Helper():
    def __init__(self, read_csv=None, load_monomer=False, **kwargs):

        # Set default values
        self.data = {
            'DBID': os.environ.get('RadonPy_DBID'),
            'monomer_ID': None,
            'smiles_list': None,
            'smiles_ter_1': '*C',
            'smiles_ter_2': None,
            'ter_ID_1': 'CH3',
            'ter_ID_2': None,
            'qm_method': 'wb97m-d3bj',
            'charge': 'RESP',

            'monomer_dir': None,
            'copoly_ratio_list': '1',
            'copoly_type': 'random',
            'input_natom': 1000,
            'input_nchain': 10,
            'ini_density': 0.05,
            'temp': 300.0,
            'press': 1.0,
            'input_tacticity': 'atactic',
            'tacticity': None,

            'remarks': '',
            'date': None,
            'Python_ver': None,
            'RadonPy_ver': None,
            'RDKit_ver': None,
            'Psi4_ver': None,
            'LAMMPS_ver': None,
        }

        # Set number of parallel
        self.omp = int(os.environ.get('RadonPy_OMP', 1))
        self.mpi = int(os.environ.get('RadonPy_MPI', utils.cpu_count()))
        self.gpu = int(os.environ.get('RadonPy_GPU', 0))
        self.retry_eq = int(os.environ.get('RadonPy_RetryEQ', 3))

        self.psi4_omp = int(os.environ.get('RadonPy_OMP_Psi4', 4))
        self.psi4_mem = int(os.environ.get('RadonPy_MEM_Psi4', 1000))

        self.conf_mm_omp = int(os.environ.get('RadonPy_Conf_MM_OMP', 1))
        self.conf_mm_mpi = int(os.environ.get('RadonPy_Conf_MM_MPI', utils.cpu_count()))
        self.conf_mm_gpu = int(os.environ.get('RadonPy_Conf_MM_GPU', 0))
        self.conf_mm_mp = int(os.environ.get('RadonPy_Conf_MM_MP', 0))
        self.conf_psi4_omp = int(os.environ.get('RadonPy_Conf_Psi4_OMP', self.psi4_omp))
        self.conf_psi4_mp = int(os.environ.get('RadonPy_Conf_Psi4_MP', 0))

        # Set work, save, temp directories
        self.work_dir = './%s' % self.data['DBID']
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

        self.save_dir = os.path.join(self.work_dir, os.environ.get('RadonPy_Save_Dir', 'analyze'))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.tmp_dir = os.environ.get('RadonPy_TMP_Dir', None)
        if self.tmp_dir is not None and not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        self.mols = []
        self.monomer_data = []

        # Read csv data
        if read_csv is not None:
            self.read_csv(read_csv)

        # Set input data
        now = datetime.datetime.now()
        self.indata = {
            'DBID': os.environ.get('RadonPy_DBID'),
            # 'Monomer_ID': os.environ.get('RadonPy_Monomer_ID', self.data['monomer_ID']),
            # 'smiles_list': os.environ.get('RadonPy_SMILES', self.data['smiles_list']),
            # 'smiles_ter_1': os.environ.get('RadonPy_SMILES_TER', self.data['smiles_ter_1']),
            # 'smiles_ter_2': os.environ.get('RadonPy_SMILES_TER2', self.data['smiles_ter_2']),
            # 'ter_ID_1': os.environ.get('RadonPy_TER_ID', self.data['ter_ID_1']),
            # 'ter_ID_2': os.environ.get('RadonPy_TER_ID2', self.data['ter_ID_2']),
            # 'qm_method': os.environ.get('RadonPy_QM_Method', self.data['qm_method']),
            # 'charge': os.environ.get('RadonPy_Charge', self.data['charge']),

            # 'monomer_dir': os.environ.get('RadonPy_Monomer_Dir', self.data['monomer_dir']),
            # 'copoly_ratio_list': os.environ.get('RadonPy_Copoly_Ratio', self.data['copoly_ratio_list']),
            # 'copoly_type': os.environ.get('RadonPy_Copoly_Type', self.data['copoly_type']),
            # 'input_natom': int(os.environ.get('RadonPy_NAtom', self.data['input_natom'])),
            # 'input_nchain': int(os.environ.get('RadonPy_NChain', self.data['input_nchain'])),
            # 'ini_density': float(os.environ.get('RadonPy_Ini_Density', self.data['ini_density'])),
            # 'temp': float(os.environ.get('RadonPy_Temp', self.data['temp'])),
            # 'press': float(os.environ.get('RadonPy_Press', self.data['press'])),
            # 'input_tacticity': os.environ.get('RadonPy_Tacticity', self.data['input_tacticity']),
            # 'tacticity': str(''),

            # Meta data are allways overwritten by new informations
            'remarks': os.environ.get('RadonPy_Remarks', str('')),
            'date': '%04i-%02i-%02i-%02i-%02i-%02i' % (now.year, now.month, now.day, now.hour, now.minute, now.second),
            'Python_ver': platform.python_version(),
            'RadonPy_ver': radonpy_ver,
            'RDKit_ver': rdkit.__version__,
            'Psi4_ver': psi4_ver,
            'LAMMPS_ver': LAMMPS().get_version(),
        }

        envkeys = {
            'RadonPy_Monomer_ID':   'monomer_ID',
            'RadonPy_SMILES':       'smiles_list',
            'RadonPy_SMILES_TER':   'smiles_ter_1',
            'RadonPy_SMILES_TER2':  'smiles_ter_2',
            'RadonPy_TER_ID':       'ter_ID_1',
            'RadonPy_TER_ID2':      'ter_ID_1',
            'RadonPy_QM_Method':    'qm_method',
            'RadonPy_Charge':       'charge',
            'RadonPy_Monomer_Dir':  'monomer_dir',
            'RadonPy_TER_Dir':      'ter_dir',
            'RadonPy_Copoly_Ratio': 'copoly_ratio_list',
            'RadonPy_Copoly_Type':  'copoly_type',
            'RadonPy_NAtom':        'input_natom',
            'RadonPy_NChain':       'input_nchain',
            'RadonPy_Ini_Density':  'ini_density',
            'RadonPy_Temp':         'temp',
            'RadonPy_Press':        'press',
            'RadonPy_Tacticity':    'input_tacticity',
        }

        for k, v in envkeys.items():
            if os.environ.get(k):
                self.indata[v] = os.environ.get(k)

        # Import preset modules
        self.preset = type('', (), {})()
        preset_dir = 'preset'
        preset_files = os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), preset_dir))
        for pf in preset_files:
            if pf.endswith('py'):
                path = os.path.join(preset_dir, pf)
                mname = os.path.splitext(os.path.basename(pf))[0]
                if mname == '__init__':
                    continue
                mpath = '..' + os.path.splitext(path)[0].replace(os.path.sep, '.')
                try:
                    m = importlib.import_module(mpath, package=__name__)
                except ImportError:
                    utils.radon_print('Cannot import %s' % mpath)
                    continue

                setattr(self.preset, mname, m)

                # Initialize preset options
                if hasattr(m, '__version__'):
                    self.indata['preset_%s_ver' % mname] = m.__version__
                if hasattr(m, 'helper_options'):
                    mop = m.helper_options()
                    for k, v in mop.items():
                        if k not in self.data.keys():
                            self.data[k] = v
                        if os.environ.get('RadonPy_%s' % k):
                            self.indata[k] = v

        # Overwritten by input data
        self.data.update(self.indata)

        # Set monomer dir
        self.monomer_dir = self.data['monomer_dir'] if self.data['monomer_dir'] else self.save_dir

        # Parse smiles list
        self.smi_list = self.data['smiles_list'].split(',') if self.data['smiles_list'] else []

        # Set input of copolymer
        if len(self.smi_list) == 1:
            self.data['copoly_ratio_list'] = '1'
            self.copoly_ratio = [1]
            self.data['copoly_type'] = ''
        else:
            self.copoly_ratio = [float(x) for x in str(self.data['copoly_ratio_list']).split(',')]

        # Set monomer ID
        if self.data['monomer_ID']:
            self.monomer_id = self.data['monomer_ID'].split(',')
        else:
            self.monomer_id = [None]*len(self.smi_list)

        # Initialize monomer data
        if load_monomer:
            self.load_monomer_data()

        elif len(self.monomer_data) == 0:
            for i, smi in enumerate(self.smi_list):
                mon = {
                    'monomer_ID':   None,
                    'smiles':       smi,
                    'qm_method':    self.data['qm_method'],
                    'charge':       self.data['charge'],
                    'copoly_ratio': self.copoly_ratio[i],
                    'remarks':      self.data['remarks'],
                    'date':         self.data['date'],
                    'Python_ver':   self.data['Python_ver'],
                    'RadonPy_ver':  self.data['RadonPy_ver'],
                    'RDKit_ver':    self.data['RDKit_ver'],
                    'Psi4_ver':     self.data['Psi4_ver'],
                }
                if self.data['monomer_ID']:
                    mon['monomer_ID'] = self.monomer_id[i]
                    self.data['monomer_ID_%i' % (i+1)] = self.monomer_id[i]

                self.data['smiles_%i' % (i+1)] = smi

                self.monomer_data.append(mon)


    def read_csv(self, file='results.csv', overwrite=True):
        if not os.path.isfile(os.path.join(self.save_dir, file)):
            utils.radon_print('Cannot find monomer data.', level=3)

        df = pd.read_csv(os.path.join(self.save_dir, file), index_col=0)
        data = df.iloc[0].to_dict()
        data['DBID'] = df.index.tolist()[0]
        if 'remarks' in data.keys() and data['remarks'] is None:
            data['remarks'] = str('')

        if data['DBID'] != self.data['DBID']:
            utils.radon_print('DBID in %s (%s) does not match the input DBID (%s).'
                % (file, data['DBID'], self.data['DBID']), level=3)

        if overwrite:
            self.data = {**self.data, **data}
        else:
            self.data = {**data, **self.data}

        if len(self.monomer_data) == 0:
            now = datetime.datetime.now()
            mon = {
                'monomer_ID': None,
                'smiles': None,
                'qm_method': 'wb97m-d3bj',
                'charge': 'RESP',
                'copoly_ratio': None,
                'remarks': '',
                'date': '%i-%i-%i-%i-%i-%i' % (now.year, now.month, now.day, now.hour, now.minute, now.second),
                'Python_ver': None,
                'RadonPy_ver': None,
                'RDKit_ver': None,
                'Psi4_ver': None,
            }
            smi_list = self.data['smiles_list'].split(',')
            self.monomer_data = [mon for x in range(len(smi_list))]

        for k, v in self.data.items():
            if re.search('_monomer\d+', k):
                m = re.search('(.+)_monomer(\d+)', k)
                key = str(m.group(1))
                idx = int(m.group(2))-1
                self.monomer_data[idx][key] = v
            elif re.search('smiles_\d+', k):
                m = re.search('smiles_(\d+)', k)
                idx = int(m.group(1))-1
                self.monomer_data[idx]['smiles'] = v
            elif re.search('monomer_ID_\d+', k):
                m = re.search('monomer_ID_(\d+)', k)
                idx = int(m.group(1))-1
                self.monomer_data[idx]['monomer_ID'] = v


    def to_csv(self, file='results.csv'):
        now = datetime.datetime.now()
        self.data['date'] = '%04i-%02i-%02i-%02i-%02i-%02i' % (now.year, now.month, now.day, now.hour, now.minute, now.second)

        if os.path.isfile(os.path.join(self.save_dir, file)):
            shutil.copyfile(os.path.join(self.save_dir, file),
                os.path.join(self.save_dir, '%s_%04i-%02i-%02i-%02i-%02i-%02i.csv' % (file, now.year, now.month, now.day, now.hour, now.minute, now.second)))

        df = pd.DataFrame(self.data, index=[0]).set_index('DBID')
        df.to_csv(os.path.join(self.save_dir, file))


    def read_monomer_csv(self, idx, file, overwrite=True):
        monomer_df = pd.read_csv(os.path.join(self.monomer_dir, file), index_col=0)
        mon_data = {'monomer_ID': monomer_df.index.tolist()[0], **monomer_df.iloc[0].to_dict()}

        if overwrite:
            self.monomer_data[idx] = {**self.monomer_data[idx], **mon_data}
        else:
            self.monomer_data[idx] = {**mon_data, **self.monomer_data[idx]}

        for k in self.monomer_data[idx].keys():
            if k == 'monomer_ID':
                self.data['monomer_ID_%i' % (idx+1)] = self.monomer_data[idx]['monomer_ID']
            elif k == 'smiles':
                self.data['smiles_%i' % (idx+1)] = self.monomer_data[idx]['smiles']
            else:
                self.data['%s_monomer%i' % (k, idx+1)] = self.monomer_data[idx][k]


    def to_monomer_csv(self, idx, file=None):
        now = datetime.datetime.now()
        self.monomer_data[idx]['date'] = '%04i-%02i-%02i-%02i-%02i-%02i' % (now.year, now.month, now.day, now.hour, now.minute, now.second)

        if self.data['monomer_ID']:
            data_df = pd.DataFrame(self.monomer_data[idx], index=[0]).set_index('monomer_ID')
        else:
            data_df = pd.DataFrame(self.monomer_data[idx], index=[0])

        if file is None:
            if self.data['monomer_ID']:
                file = 'monomer_%s.csv' % self.monomer_data[idx]['monomer_ID']
            else:
                file = 'monomer%i.csv' % (idx+1)

        if os.path.isfile(os.path.join(self.save_dir, file)):
            shutil.copyfile(os.path.join(self.save_dir, file),
                os.path.join(self.save_dir, '%s_%04i-%02i-%02i-%02i-%02i-%02i.csv' % (file, now.year, now.month, now.day, now.hour, now.minute, now.second)))

        data_df.to_csv(os.path.join(self.save_dir, file))


    def update(self, **kwargs):
        self.data.update(kwargs)


    def update_monomer(self, idx, **kwargs):
        self.monomer_data[idx].update(kwargs)
        for k in kwargs.keys():
            if k == 'monomer_ID':
                self.data['monomer_ID_%i' % (idx+1)] = self.monomer_data[idx]['monomer_ID']
            elif k == 'smiles':
                self.data['smiles_%i' % (idx+1)] = self.monomer_data[idx]['smiles']
            else:
                self.data['%s_monomer%i' % (k, idx+1)] = self.monomer_data[idx][k]


    def monomer_pickle_dump(self, mol, idx, file=None):
        if file is None:
            if self.data['monomer_ID']:
                file = 'monomer_%s.pickle' % self.monomer_data[idx]['monomer_ID']
            else:
                file = 'monomer%i.pickle' % (idx+1)

        now = datetime.datetime.now()
        if os.path.isfile(os.path.join(self.save_dir, file)):
            shutil.copyfile(os.path.join(self.save_dir, file),
                os.path.join(self.save_dir, '%s_%04i-%02i-%02i-%02i-%02i-%02i.pickle' % (file, now.year, now.month, now.day, now.hour, now.minute, now.second)))

        utils.pickle_dump(mol, os.path.join(self.save_dir, file))


    def load_monomer_data(self):

        # Load monomer_data.csv file
        if self.data['monomer_dir'] and self.data['monomer_ID']:
            for i, mid in enumerate(self.monomer_id):
                self.read_monomer_csv(i, 'monomer_%s_data.csv' % mid)

                mol = utils.pickle_load(os.path.join(self.monomer_dir, 'monomer_%s.pickle' % mid))
                self.mols.append(mol)

        # Load qm_data.csv file
        elif os.path.isfile(os.path.join(self.save_dir, 'qm_data.csv')):
            self.read_csv('qm_data.csv')

            for idx in range(len(self.monomer_data)):
                if self.data['monomer_ID']:
                    mol = utils.pickle_load(os.path.join(self.save_dir, 'monomer_%s.pickle' % self.monomer_id[idx]))
                else:
                    mol = utils.pickle_load(os.path.join(self.save_dir, 'monomer%i.pickle' % (idx+1)))
                self.mols.append(mol)

        else:
            utils.radon_print('Cannot find monomer data.', level=3)

        if self.data['ter_ID_1'] and os.path.isfile(os.path.join(self.monomer_dir, 'ter_%s.pickle' % self.data['ter_ID_1'])):
            self.ter1 = utils.pickle_load(os.path.join(self.monomer_dir, 'ter_%s.pickle' % self.data['ter_ID_1']))
        elif os.path.isfile(os.path.join(self.save_dir, 'ter1.pickle')):
            self.ter1 = utils.pickle_load(os.path.join(self.save_dir, 'ter1.pickle'))

        if self.data['ter_ID_2'] and os.path.isfile(os.path.join(self.monomer_dir, 'ter_%s.pickle' % self.data['ter_ID_2'])):
            self.ter2 = utils.pickle_load(os.path.join(self.monomer_dir, 'ter_%s.pickle' % self.data['ter_ID_2']))
        elif os.path.isfile(os.path.join(self.save_dir, 'ter2.pickle')):
            self.ter2 = utils.pickle_load(os.path.join(self.save_dir, 'ter2.pickle'))

        self.data.update(self.indata)


    def restore(self, mod='eq', **kwargs):
        if hasattr(self.preset, mod):
            m = getattr(self.preset, mod)
            if hasattr(m, 'restore'):
                mol = m.restore(self.save_dir, **kwargs)
                return mol

        utils.radon_print('Cannot restore mod=%s.' % mod, level=3)

