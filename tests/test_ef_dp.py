from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from radonpy.core import poly, utils
from radonpy.sim import gromacs as backend
from radonpy.sim.preset import ef_dp

FIXTURES = Path(__file__).parent/'fixtures'


@pytest.fixture
def liquid():
    np.random.seed(5)
    octane = utils.JSONToMol(str(FIXTURES/'pe_gaff2.json'))
    return poly.amorphous_cell(octane, 30, density=0.25)


def preset(mol, work, **kwargs):
    work.mkdir(exist_ok=True)
    return ef_dp.further_Additional(mol, work_dir=str(work), solver='gromacs', interaction_profile='portable',
                                    **kwargs)


def test_ac_field_and_dipole_are_accepted_by_gromacs(liquid, tmp_path):
    p = preset(liquid, tmp_path, axis='z', evalue=0.05, freq=1e11, process='dp')
    solver = backend.Gromacs(tmp_path)
    solver.make_dat(liquid)
    # AutoMD passes float output frequencies; LAMMPS writes them as integers.
    md = p.sampling_dp(step=1000, thermo_freq=100.0, dump_freq=100.0, barostat='C-rescale')
    solver.make_input(md, file_name='check.json')
    options = dict(line.split(' = ') for line in backend.Gromacs._mdp(md, md.wf[0], liquid, 0.0).splitlines())
    # LAMMPS: 0.05 V/angstrom * cos(2 pi 1e11 Hz * time); GROMACS: V/nm and rad/ps.
    assert options['electric-field-z'] == '0.5 %.12g 0 0' % (2*np.pi*0.1)
    assert 'electric-field-x' not in options and options['nstenergy'] == '100'
    tuning = preset(liquid, tmp_path/'tuning', axis='z', evalue=0.5, process='tuning')
    with pytest.raises(NotImplementedError, match='linear ramps'):
        solver.make_input(tuning.sampling_dp(step=1000, barostat='C-rescale'), file_name='check.json')


def test_dipole_observable_is_the_charge_weighted_sum(liquid):
    rng = np.random.default_rng(3)
    charges = np.array([a.GetDoubleProp('AtomicCharge') for a in liquid.GetAtoms()])
    initial = liquid.GetConformer().GetPositions()
    wf = SimpleNamespace(rg=False, msd=False, dipole=True, time_step=1.0)
    observer = backend.Observables(liquid, initial, wf, SimpleNamespace(thermo_freq=10), elapsed=0.0)
    frames = [initial+rng.normal(scale=.1, size=initial.shape) for _ in range(3)]
    for step, xyz in zip((0, 10, 20), frames):
        observer(step/1000, xyz)
    # Every frame, the stage start included, as LAMMPS prints the dipole at step 0.
    assert [s for s, _ in observer.dipole] == [0, 10, 20]
    np.testing.assert_allclose([m for _, m in observer.dipole], [charges@xyz for xyz in frames])


@pytest.mark.engines
def test_ef_dp_runs_with_gromacs(liquid, gromacs_exec, tmp_path):
    np.random.seed(13)
    p = preset(liquid, tmp_path/'ef', axis='z', evalue=0.05, freq=5e12, process='dp', solver_path=gromacs_exec)
    # Five periods of a 5 THz field.
    p.exec_dp(eq_step=1e-3, thermo_freq=10, dump_freq=10, barostat='C-rescale', omp=2, mpi=0)
    log = p.log_reader()(log_file=str(tmp_path/'ef'/p.log_file)).dfs[-1]
    time = log['Time'].to_numpy()
    np.testing.assert_allclose(log['v_efac'], 0.05*np.cos(2*np.pi*5e12*time*1e-15))
    # The dipole at the start is that of the initial structure.
    charges = np.array([a.GetDoubleProp('AtomicCharge') for a in liquid.GetAtoms()])
    np.testing.assert_allclose(log[['v_mux', 'v_muy', 'v_muz']].iloc[0], charges@liquid.GetConformer().GetPositions(),
                               atol=1e-3)
    assert np.isfinite(log['v_mu']).all()
    result = p.dipole_anal(printout=False, save=False)
    assert np.isfinite(result['real part of permittivity'].iat[0])
