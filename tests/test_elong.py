from pathlib import Path

import numpy as np
import pytest
from radonpy.core import calc, poly, utils
from radonpy.sim import gromacs as backend, lammps, md
from radonpy.sim.preset import elong

FIXTURES = Path(__file__).parent/'fixtures'
LATERAL = {'py_start': 1.0, 'py_stop': 1.0, 'py_dump': 1000.0, 'pz_start': 1.0, 'pz_stop': 1.0, 'pz_dump': 1000.0}


@pytest.fixture
def liquid():
    np.random.seed(5)
    octane = utils.JSONToMol(str(FIXTURES/'pe_gaff2.json'))
    return poly.amorphous_cell(octane, 30, density=0.25)


def deformation(mol, work, **kwargs):
    work.mkdir(exist_ok=True)
    return elong.Deformation(mol, work_dir=str(work), axis='x', deform_strain=0.1, **kwargs)


def test_deformation_stages_for_gromacs(liquid, tmp_path):
    p = deformation(liquid, tmp_path, solver='gromacs', interaction_profile='portable')
    md = p.sampling_deform(tstead=300, tmax=320, heating_step=1000, relax_step=1000, deform_rate=1e-5,
                           deform_step=10000, cooling_step=100, barostat='Parrinello-Rahman', **LATERAL)
    solver = backend.Gromacs(tmp_path)
    solver.make_dat(liquid)
    solver.make_input(md, file_name='check.json')
    # Heating and relaxation, the deformation, 320 and 310 K held and ramped, and the final relaxation.
    assert [wf.deform for wf in md.wf] == [None, None, 'erate', None, None, None, None, None]
    deform = md.wf[2]
    assert deform.p_aniso and deform.py_start == 1.0 and deform.px_start is None
    assert not md.wf[0].p_aniso


def lammps_deformation_input(mol, work, **kwargs):
    p = deformation(mol, work, **kwargs)
    md = p.sampling_deform(tstead=300, tmax=320, deform_rate=1e-5, deform_step=10000, **LATERAL)
    solver = lammps.LAMMPS(work_dir=str(work), check_lammps_package=False)
    solver.make_input(md, file_name='deform.in')
    return (work/'deform.in').read_text()


def test_legacy_lammps_deformation_is_unchanged(liquid, tmp_path):
    text = lammps_deformation_input(liquid, tmp_path)
    assert 'fix DEF1 all deform 1 x erate 1.0e-05 units box remap x' in text
    assert 'variable EngStrain equal' in text and 'fix momentum1 all momentum 50 linear 1 1 1' in text
    assert 'pair_style lj/charmm/coul/long 8.0 12.0' in text
    # The thermo_style in effect for the deformation run reports the strain and the surface tension.
    deform = text[:text.index('\nrun 10000\n')]
    assert deform[deform.rindex('thermo_style'):].split('\n')[0].endswith(' v_GamNsurf v_EngStrain')


def test_lammps_deformation_uses_the_interaction_profile(liquid, tmp_path):
    text = lammps_deformation_input(liquid, tmp_path, interaction_profile='portable')
    assert 'pair_style lj/cut/coul/long 12.0 12.0' in text and 'pair_modify mix arithmetic shift no tail no' in text


@pytest.mark.engines
@pytest.mark.parametrize('engine', ['lammps', 'gromacs'])
def test_deformation_runs(engine, engines, tmp_path):
    np.random.seed(17)
    # Dense and large enough that the laterally contracting box stays above twice the cutoff.
    octane = utils.JSONToMol(str(FIXTURES/'pe_gaff2.json'))
    liquid = poly.amorphous_cell(octane, 120, density=0.5)
    # elong starts from an equilibrated structure; a minimized one is enough here.
    (tmp_path/'min').mkdir()
    liquid, _, _ = md.quick_min(liquid, solver=engine, solver_path=engines[engine], work_dir=str(tmp_path/'min'),
                                interaction_profile='portable', ftol=1.0, maxiter=500)
    calc.set_velocity(liquid, 300)
    work = tmp_path/'deform'
    p = deformation(liquid, work, solver=engine, interaction_profile='portable', solver_path=engines[engine])
    # GROMACS C-rescale has no anisotropic coupling.
    barostat = {'barostat': 'Parrinello-Rahman'} if engine == 'gromacs' else {}
    # 10% strain at 1e-5/fs over 10,000 steps, without heating or cooling.
    p.exec_deform(init_temp=300, max_temp=300, relax_step=500, deform_rate=1e-5, deform_step=10000,
                  omp=2, mpi=0, **barostat, **LATERAL)
    analysis = elong.Deformation_Analysis(work_dir=str(work), axis='x', deform_strain=0.1, solver=engine)
    whole, deform = analysis.log_extraction()
    assert deform['v_EngStrain'].iat[-1] == pytest.approx(0.1, rel=1e-3)
    # The lateral axes contract under 1 atm while x is stretched.
    assert deform['Ly'].iat[-1] < deform['Ly'].iat[0] and np.isfinite(deform['v_GamNsurf']).all()
    assert len(whole) > len(deform)
    analysis.density_plot()
    analysis.nemorder_plot()
    assert (work/'analyze'/'density_plot.png').exists() and (work/'analyze'/'nematic_order.csv').exists()
