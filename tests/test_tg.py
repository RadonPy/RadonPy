import json
from pathlib import Path

import numpy as np
import pytest
from radonpy.core import poly, utils
from radonpy.sim import gromacs as backend, lammps
from radonpy.sim.preset import tg
from tests.test_equilibration import _shorten

FIXTURES = Path(__file__).parent/'fixtures'


@pytest.fixture
def liquid():
    np.random.seed(5)
    octane = utils.JSONToMol(str(FIXTURES/'pe_gaff2.json'))
    return poly.amorphous_cell(octane, 30, density=0.25)


def gromacs_tgmd(mol, work, **kwargs):
    return tg.TGMD(mol, work_dir=str(work), solver='gromacs', interaction_profile='portable', **kwargs)


def mdp(md, wf, mol):
    return dict(line.split(' = ') for line in backend.Gromacs._mdp(md, wf, mol, 0.0).splitlines())


def test_tg_workflows_are_accepted_by_gromacs(liquid, tmp_path):
    p = gromacs_tgmd(liquid, tmp_path)
    assert p.analyzer() is tg.TGMD_analyze_gromacs
    solver = backend.Gromacs(tmp_path)
    solver.make_dat(liquid)
    kwargs = {'barostat': 'C-rescale'}
    search = p.density_checker(t_start=300, t_stop=800, step=(800-300)*2e2, **kwargs)
    equilibration = p.equilibration(temp=800, step=5e5, **kwargs)
    for md in (search, equilibration, p.step_wise(800, 700, cooling_rate=8e3, **kwargs),
               p.quench(800, 700, cooling_rate=8e3, **kwargs), p.continuous(800, 700, cooling_rate=8e3, **kwargs)):
        solver.make_input(md, file_name='check.json')
    # Step counts given as floats (5e5) are integers, as in LAMMPS' run %i.
    assert mdp(equilibration, equilibration.wf[0], liquid)['nsteps'] == '500000'
    assert mdp(search, search.wf[0], liquid)['annealing-temp'] == '300 800'


def test_legacy_lammps_tg_is_unchanged(liquid, tmp_path):
    p = tg.TGMD(liquid, work_dir=str(tmp_path))
    assert p.solver == 'lammps' and p.interaction_profile is None and p.analyzer() is tg.TGMD_analyze
    solver = lammps.LAMMPS(work_dir=str(tmp_path), check_lammps_package=False)
    solver.make_input(p.step_wise(800, 700, cooling_rate=8e3), file_name='tg.in')
    text = (tmp_path/'tg.in').read_text()
    assert 'pair_style lj/charmm/coul/long 8.0 12.0' in text
    # Ten temperatures, each held and then ramped, and a final hold.
    assert text.count('\nrun 4000\n') == 21


@pytest.mark.parametrize('shift', [0.0, 0.02])
def test_step_wise_analysis_of_gromacs_stages(tmp_path, shift):
    # Density falls linearly with two slopes that meet at 400 K; the melt at
    # the highest temperature is below the 0.45 g/cm3 the analysis starts from,
    # or, shifted, stays above it.
    def density(t):
        return 1.04+shift-(2e-4 if t < 400 else 1.5e-3)*(t-400)
    frames, step = [], 0
    for t in np.arange(800, 50, -10):
        for temps in ([t]*5, np.linspace(t, t-10, 5)):
            frames.append({'Step': list(range(step, step+500, 100)), 'Temp': list(map(float, temps)),
                           'Density': [density(x) for x in temps]})
            step += 500
    frames.append({'Step': list(range(step, step+500, 100)), 'Temp': [50.0]*5, 'Density': [density(50)]*5})
    log = tmp_path/'tg.log'
    Path(str(log)+'.gmx.json').write_text(json.dumps({'frames': frames}))
    result = tg.TGMD_analyze_gromacs(log_file=str(log), save_dir=str(tmp_path)).step_wise()
    assert result['tg'] == pytest.approx(400, abs=1)
    assert result['tg_thermal_expansion_coef(upper_tg)'] == pytest.approx(-1.5e-3, rel=1e-6)
    assert result['tg_thermal_expansion_coef(below_tg)'] == pytest.approx(-2e-4, rel=1e-6)


@pytest.mark.engines
def test_tgmd_runs_with_gromacs(liquid, gromacs_exec, tmp_path, monkeypatch):
    np.random.seed(11)
    p = gromacs_tgmd(liquid, tmp_path, solver_path=gromacs_exec)
    for name in ('density_checker', 'equilibration', 'step_wise'):
        setattr(p, name, _shorten(getattr(p, name), 200))
    # Repeat the density search once: each GROMACS run keeps its own files.
    # The runs are too short for the density fit, so only their logs are read.
    searches = []

    def search(self, index):
        frame = self.read_log(self.log_file)[-1]
        searches.append(self.log_file)
        return {'tg_init_density': frame['Density'].iloc[-1], 'tg_init_density_check': index > 0, 'tg_next_temp': 850}
    stages = []

    def step_wise(self):
        stages.extend(self.read_log(self.log_file))
        return {'tg': 0.0}
    monkeypatch.setattr(tg.TGMD_analysis, 'density_checker', search)
    monkeypatch.setattr(tg.TGMD_analysis, 'step_wise', step_wise)
    mol, data = p.exec(temp=300, min_temp=810, interval_temp=10, cooling_rate=400, barostat='C-rescale',
                       omp=2, mpi=0)
    assert [Path(s).name for s in searches] == ['tg_pre_01.log', 'tg_pre_02.log']
    assert data['tg_max_temp'] == 850
    # 850, 840, 830 and 820 K held and ramped, then held at 810 K.
    assert len(stages) == 9
    held = [frame['Temp'].mean() for frame in stages[0::2]]
    np.testing.assert_allclose(held, [850, 840, 830, 820, 810], atol=60)
    assert mol.GetNumAtoms() == liquid.GetNumAtoms()
