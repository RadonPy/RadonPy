import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from radonpy.core import utils
from radonpy.sim import md, lammps

FIXTURES = Path(__file__).parent/'fixtures'


def test_legacy_input_settings_are_unchanged(molecule, tmp_path):
    solver = lammps.LAMMPS(work_dir=str(tmp_path), check_lammps_package=False)
    options = md.MD(mol=molecule)
    solver.make_input(options)
    text = (tmp_path/options.input_file).read_text()
    assert 'pair_style lj/charmm/coul/long 8.0 12.0' in text
    assert 'special_bonds amber' in text
    assert 'pair_modify mix arithmetic\n' in text


@pytest.mark.engines
@pytest.mark.parametrize('fixture', ['butane_gaff2'] + [p+'_'+f for p in ('pe', 'ps', 'pmma')
                                                      for f in ('gaff', 'gaff2', 'gaff2_mod')])
def test_legacy_against_original_checkout(fixture, lammps_exec, tmp_path):
    reference = json.loads((FIXTURES/'lammps_reference.json').read_text())['fixtures'][fixture+'.json']
    source = FIXTURES/(fixture+'.json')
    assert hashlib.sha256(source.read_bytes()).hexdigest() == reference['sha256']
    energy, force = md.quick_energy(utils.JSONToMol(str(source)), work_dir=str(tmp_path),
                                   solver_path=lammps_exec, omp=0, mpi=0)
    np.testing.assert_allclose(energy, reference['energy_kcal_mol'], atol=1e-5, rtol=1e-6)
    np.testing.assert_allclose(force, reference['force_kcal_mol_angstrom'], atol=1e-4, rtol=1e-5)
