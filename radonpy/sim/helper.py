#  Copyright (c) 2023. RadonPy developers. All rights reserved.
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
from . import lammps
from .preset import eq

__version__ = '0.2.9'


def get_version():
    from ..__init__ import __version__ as radonpy_ver

    try:
        from psi4 import __version__ as psi4_ver
    except ImportError:
        psi4_ver = None

    try:
        lammps_ver = lammps.LAMMPS().get_version()
    except:
        lammps_ver = None

    ver = {
        'Python_ver': platform.python_version(),
        'RadonPy_ver': radonpy_ver,
        'RDKit_ver': rdkit.__version__,
        'Psi4_ver': psi4_ver,
        'LAMMPS_ver': lammps_ver,
    }

    return ver 


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

        ver_info = get_version()

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
            'Python_ver': ver_info['Python_ver'],
            'RadonPy_ver': ver_info['RadonPy_ver'],
            'RDKit_ver': ver_info['RDKit_ver'],
            'Psi4_ver': ver_info['Psi4_ver'],
            'LAMMPS_ver': ver_info['LAMMPS_ver'],
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


class IO_Helper():
    def __init__(self, work_dir, save_dir, share_dir=[]):
        self.work_dir = work_dir
        self.save_dir = save_dir
        if type(share_dir) is not list:
            share_dir = [share_dir]
        self.share_dir = share_dir

    def update_monomer_data(self, update_dict, data_dict, monomer_dict, monomer_idx=0):
        monomer_dict.update(update_dict)
        monomer_df = pd.DataFrame(monomer_dict, index=[0])
        if data_dict.get('monomer_ID'):
            monomer_id = data_dict['monomer_ID'].split(',')
            monomer_df = monomer_df.set_index('monomer_ID')
            monomer_df.to_csv(os.path.join(self.save_dir, 'monomer_%s_data.csv' % monomer_id[monomer_idx]))
        else:
            monomer_df.to_csv(os.path.join(self.save_dir, 'monomer%i_data.csv' % (monomer_idx+1)))

        for k in update_dict.keys():
            data_dict['%s_monomer%i' % (k, monomer_idx+1)] = update_dict[k]
        data_df = pd.DataFrame(data_dict, index=[0]).set_index('DBID')
        data_df.to_csv(os.path.join(self.save_dir, 'qm_data.csv'))

        return data_dict, monomer_dict


    def load_monomer_csv(self, share_dir=[], data_dict={}):
        if type(share_dir) is not list:
            share_dir = [share_dir]
        share_dir = [self.save_dir, self.work_dir, *share_dir, *self.share_dir]

        if len(share_dir) > 0 and data_dict.get('monomer_ID'):
            monomer_id = data_dict['monomer_ID'].split(',')
            monomer_dir = []
            if data_dict.get('copoly_ratio_list'):
                if len(str(data_dict['copoly_ratio_list']).split(',')) == len(monomer_id):
                    ratio = [float(x) for x in str(data_dict['copoly_ratio_list']).split(',')]
                else:
                    ratio = [1.0 for x in range(len(monomer_id))]
            else:
                ratio = [1.0 for x in range(len(monomer_id))]

            for i, mid in enumerate(monomer_id):
                monomer_path = None
                for d in share_dir:
                    monomer_path = os.path.join(d, 'monomer_%s_data.csv' % mid)
                    if os.path.isfile(monomer_path):
                        break
                if monomer_path is None:
                    raise Exception('ERROR: Cannot find monomer data of %s.' % mid)

                monomer_dir.append(d)
                monomer_df = pd.read_csv(monomer_path, index_col=0)
                monomer_data = monomer_df.iloc[0].to_dict()
                data_dict['smiles_%i' % (i+1)] = monomer_data.pop('smiles')
                data_dict['monomer_ID_%i' % (i+1)] = mid
                data_dict['copoly_ratio_%i' % (i+1)] = ratio[i]
                for k in monomer_data.keys():
                    data_dict['%s_monomer%i' % (k, i+1)] = monomer_data[k]

        elif os.path.isfile(os.path.join(self.save_dir, 'qm_data.csv')):
            qm_df = pd.read_csv(os.path.join(self.save_dir, 'qm_data.csv'), index_col=0)
            qm_data = qm_df.iloc[0].to_dict()
            data_dict = {**qm_data, **data_dict}
            data_dict['monomer_dir'] = self.save_dir

        elif os.path.isfile(os.path.join(self.work_dir, 'qm_data.csv')):
            qm_df = pd.read_csv(os.path.join(self.work_dir, 'qm_data.csv'), index_col=0)
            qm_data = qm_df.iloc[0].to_dict()
            data_dict = {**qm_data, **data_dict}
            data_dict['monomer_dir'] = self.work_dir

        else:
            raise Exception('ERROR: Cannot find monomer data.')

        return data_dict


    def load_monomer_obj(self, share_dir=[], data_dict={}):
        if type(share_dir) is not list:
            share_dir = [share_dir]
        share_dir = [self.save_dir, self.work_dir, *share_dir, *self.share_dir]

        mols = []
        if data_dict.get('monomer_ID'):
            monomer_id = data_dict['monomer_ID'].split(',')
            for i in range(len(monomer_id)):
                mol = None
                for d in share_dir:
                    json_path = os.path.join(d, 'monomer_%s.json' % (monomer_id[i]))
                    if os.path.isfile(json_path):
                        mol = utils.JSONToMol(json_path)
                        break

                    pickle_path = os.path.join(d, 'monomer_%s.pickle' % (monomer_id[i]))
                    if os.path.isfile(pickle_path):
                        mol = utils.pickle_load(pickle_path)
                        break

                if mol is None:
                    raise Exception('ERROR: Cannot find monomer object file of %s.' % monomer_id[i])

                mols.append(mol)

        else:
            smi_list = data_dict['smiles_list'].split(',')
            for i in range(len(smi_list)):
                mol = None
                for d in share_dir:
                    json_path = os.path.join(d, 'monomer%i.json' % (i+1))
                    if os.path.isfile(json_path):
                        mol = utils.JSONToMol(json_path)
                        break

                    pickle_path = os.path.join(d, 'monomer%i.pickle' % (i+1))
                    if os.path.isfile(pickle_path):
                        mol = utils.pickle_load(pickle_path)
                        break

                if mol is None:
                    raise Exception('ERROR: Cannot find monomer object file of monomer%i.' % (i+1))

                mols.append(mol)

        return mols


    def load_terminal_obj(self, share_dir=[], data_dict={}):
        if type(share_dir) is not list:
            share_dir = [share_dir]

        share_dir = [self.save_dir, self.work_dir, *share_dir, *self.share_dir]
        ter1, ter2 = None, None

        if data_dict.get('ter_ID_1'):
            for d in share_dir:
                json_path = os.path.join(d, 'ter_%s.json' % data_dict['ter_ID_1'])
                if os.path.isfile(json_path):
                    ter1 = utils.JSONToMol(json_path)
                    break

                pickle_path = os.path.join(d, 'ter_%s.pickle' % data_dict['ter_ID_1'])
                if os.path.isfile(pickle_path):
                    ter1 = utils.pickle_load(pickle_path)
                    break

            if ter1 is None:
                raise Exception('ERROR: Cannot find monomer object file of ter_%s.' % data_dict['ter_ID_1'])

        else:
            for d in share_dir:
                json_path = os.path.join(d, 'ter1.json')
                if os.path.isfile(json_path):
                    ter1 = utils.JSONToMol(json_path)
                    break

                pickle_path = os.path.join(d, 'ter1.pickle')
                if os.path.isfile(pickle_path):
                    ter1 = utils.pickle_load(pickle_path)
                    break

            if ter1 is None:
                raise Exception('ERROR: Cannot find monomer object file of ter1.')

        if data_dict.get('ter_ID_2'):
            for d in share_dir:
                json_path = os.path.join(d, 'ter_%s.json' % data_dict['ter_ID_2'])
                if os.path.isfile(json_path):
                    ter2 = utils.JSONToMol(json_path)
                    break

                pickle_path = os.path.join(d, 'ter_%s.pickle' % data_dict['ter_ID_2'])
                if os.path.isfile(pickle_path):
                    ter2 = utils.pickle_load(pickle_path)
                    break

            if ter2 is None:
                raise Exception('ERROR: Cannot find monomer object file of ter_%s.' % data_dict['ter_ID_2'])

        else:
            for d in share_dir:
                json_path = os.path.join(d, 'ter2.json')
                if os.path.isfile(json_path):
                    ter2 = utils.JSONToMol(json_path)
                    break

                pickle_path = os.path.join(d, 'ter2.pickle')
                if os.path.isfile(pickle_path):
                    ter2 = utils.pickle_load(pickle_path)
                    break

        return ter1, ter2


    def exist_monomer_csv(self, monomer_id, share_dir=[]):
        exi = False

        if type(share_dir) is not list:
            share_dir = [share_dir]
        share_dir = [self.save_dir, self.work_dir, *share_dir, *self.share_dir]

        for d in share_dir:
            monomer_path = os.path.join(d, 'monomer_%s_data.csv' % monomer_id)
            if os.path.isfile(monomer_path):
                exi = True
                break
                
        return exi


    def exist_monomer_obj(self, monomer_id, share_dir=[]):
        exi = False

        if type(share_dir) is not list:
            share_dir = [share_dir]
        share_dir = [self.save_dir, self.work_dir, *share_dir, *self.share_dir]

        for d in share_dir:
            json_path = os.path.join(d, 'monomer_%s.json' % (monomer_id))
            if os.path.isfile(json_path):
                exi = True
                break

            pickle_path = os.path.join(d, 'monomer_%s.pickle' % (monomer_id))
            if os.path.isfile(pickle_path):
                exi = True
                break

        return exi


    def load_md_csv(self, data_dict):
        # Load input_data.csv file
        if os.path.isfile(os.path.join(self.save_dir, 'results.csv')):
            input_df = pd.read_csv(os.path.join(self.save_dir, 'results.csv'), index_col=0)
        elif os.path.isfile(os.path.join(self.save_dir, 'input_data.csv')):
            input_df = pd.read_csv(os.path.join(self.save_dir, 'input_data.csv'), index_col=0)
        else:
            input_df = pd.read_csv(os.path.join(self.work_dir, 'input_data.csv'), index_col=0)
        input_data = input_df.iloc[0].to_dict()
        data_dict = {**input_data, **data_dict}

        return data_dict


    def load_md_obj(self, rst_json_file=None, rst_pickle_file=None):
        # Load JSON file or pickle file or LAMMPS data file
        mol = None
        if not rst_json_file:
            rst_json_file = eq.get_final_json([self.save_dir, self.work_dir])
            if rst_json_file:
                mol = utils.JSONToMol(rst_json_file)
        else:
            mol = utils.JSONToMol(os.path.join(self.work_dir, rst_json_file))

        if mol is None:
            if not rst_pickle_file:
                rst_pickle_file = eq.get_final_pickle([self.save_dir, self.work_dir])
                if rst_pickle_file:
                    mol = utils.pickle_load(rst_pickle_file)
            else:
                mol = utils.pickle_load(os.path.join(self.work_dir, rst_pickle_file))

        if mol is None:
            rst_data_file = eq.get_final_data(self.work_dir)
            if rst_data_file:
                mol = lammps.MolFromLAMMPSdata(rst_data_file)

        return mol


    def output_md_data(self, data_dict):
        if os.path.isfile(os.path.join(self.save_dir, 'results.csv')):
            now = datetime.datetime.now()
            shutil.copyfile(os.path.join(self.save_dir, 'results.csv'),
                os.path.join(self.save_dir, 'results_%i-%i-%i-%i-%i-%i.csv' % (now.year, now.month, now.day, now.hour, now.minute, now.second)))

        # Data output after equilibration MD
        eq_data_df = pd.DataFrame(data_dict, index=[0]).set_index('DBID')
        eq_data_df.to_csv(os.path.join(self.save_dir, 'results.csv'))

