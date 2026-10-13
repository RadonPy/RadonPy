import numpy as np
import pytest
from radonpy.core import calc
from tests.helpers import gromacs, lammps


@pytest.mark.engines
@pytest.mark.parametrize('engine', ['lammps', 'gromacs'])
def test_nve_drift_decreases_with_timestep(molecule, request, tmp_path, engine):
    executable = request.getfixturevalue(engine+'_exec')
    np.random.seed(130)
    calc.set_velocity(molecule, 300)
    drifts = []
    for timestep in (.1, .05):
        work = tmp_path/str(timestep)
        runner = lammps if engine == 'lammps' else gromacs
        frame, _ = runner.run(molecule, work, executable,
                              steps=int(20/timestep), time_step=timestep, stride=10)
        energy = (frame.TotEng.to_numpy()*4.184 if engine == 'lammps'
                  else frame['Total Energy'])/molecule.GetNumAtoms()
        drifts.append(np.max(np.abs(energy-energy[0])))
    assert drifts[0] < .01
    assert drifts[1] < .6*drifts[0]+1e-5
