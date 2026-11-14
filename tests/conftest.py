import os
from pathlib import Path
import shutil

import pytest
from radonpy.core import utils


def pytest_addoption(parser):
    parser.addoption('--require-engines', action='store_true',
                     help='Fail instead of skip when either MD engine is absent')


@pytest.fixture
def molecule():
    return utils.JSONToMol(str(Path(__file__).parent/'fixtures/butane_gaff2.json'))


@pytest.fixture
def lammps_exec(request):
    return _executable(request, 'LAMMPS_EXEC', 'lmp')


@pytest.fixture
def gromacs_exec(request):
    return _executable(request, 'GROMACS_EXEC', 'gmx')


def _executable(request, variable, default):
    executable = os.environ.get(variable, default)
    if shutil.which(executable) is None:
        message = 'Missing MD executable: %s=%s' % (variable, executable)
        if request.config.getoption('--require-engines'):
            pytest.fail(message)
        pytest.skip(message)
    return executable


@pytest.fixture
def engines(lammps_exec, gromacs_exec):
    return {'lammps': lammps_exec, 'gromacs': gromacs_exec}
