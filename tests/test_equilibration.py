from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from radonpy.core import const, poly, utils
from radonpy.sim import gromacs as backend, lammps
from radonpy.sim.preset import eq

FIXTURES = Path(__file__).parent/'fixtures'
PROPERTIES = ['density', 'Rg', 'self-diffusion', 'Cp', 'Cv', 'compressibility',
              'isentropic_compressibility', 'bulk_modulus', 'isentropic_bulk_modulus',
              'volume_expansion', 'linear_expansion', 'r2', 'static_dielectric_const',
              'nematic_order_parameter']


@pytest.fixture
def liquid():
    np.random.seed(5)
    octane = utils.JSONToMol(str(FIXTURES/'pe_gaff2.json'))
    return poly.amorphous_cell(octane, 30, density=0.25)


def preset(cls, mol, work, **kwargs):
    work.mkdir(exist_ok=True)
    return cls(mol, work_dir=str(work), solver='gromacs', interaction_profile='portable', **kwargs)


def test_preset_workflows_are_accepted_by_gromacs(liquid, tmp_path):
    p = preset(eq.EQ21step, liquid, tmp_path)
    solver = backend.Gromacs(tmp_path)
    solver.make_dat(liquid)
    kwargs = {'barostat': 'C-rescale'}
    for md in (p.packing(f_density=0.8, **kwargs), p.annealing(**kwargs),
               p.eq21step(set_init_velocity=True, **kwargs), p.sampling(**kwargs)):
        solver.make_input(md, file_name='check.json')
    with pytest.raises(ValueError, match='MTK'):
        solver.make_input(p.sampling())
    with pytest.raises(NotImplementedError, match='LAMMPS'):
        p.make_lammps_input()


def test_packing_stage_is_charge_free_and_deforms(liquid, tmp_path):
    p = preset(eq.EQ21step, liquid, tmp_path)
    md = p.packing(f_density=0.8)
    assert md.pair_style == 'lj/cut' and md.cutoff_in == 3.0
    text = backend.Gromacs._mdp(md, md.wf[-1], liquid, 0.0)
    options = dict(line.split(' = ') for line in text.splitlines())
    assert options['coulombtype'] == 'Cut-off' and options['rvdw'] == '0.3'
    mass = sum(a.GetMass() for a in liquid.GetAtoms())
    # packing() deforms to [-f_length, f_length]; f_length is half the edge.
    target = np.cbrt((mass/const.NA)/(0.8/const.cm2ang**3))/10
    duration = md.wf[-1].step*md.wf[-1].time_step/1000
    rates = [float(x) for x in options['deform'].split()]
    np.testing.assert_allclose(rates[:3], (target-liquid.cell.dx/10)/duration, rtol=1e-9)
    assert rates[3:] == [0, 0, 0]
    charges = [line.split()[6] for line in backend.topology(liquid, electrostatics=False).splitlines()
               if line.split()[3:5] == ['MOL', 'C']]
    assert set(charges) == {'0'}


def test_legacy_lammps_preset_is_unchanged(liquid, tmp_path):
    p = eq.EQ21step(liquid, work_dir=str(tmp_path))
    assert p.solver == 'lammps' and p.interaction_profile is None
    solver = lammps.LAMMPS(work_dir=str(tmp_path), check_lammps_package=False)
    solver.make_input(p.sampling(), file_name='eq3.in')
    text = (tmp_path/'eq3.in').read_text()
    assert 'pair_style lj/charmm/coul/long 8.0 12.0' in text
    assert 'compute msd' in text and 'gyration/chunk' in text


def test_observables_match_definitions_and_lammps_rg_format(tmp_path):
    from rdkit import Chem
    mol = Chem.AddHs(Chem.MolFromSmiles('C.C'))
    masses = np.array([a.GetMass() for a in mol.GetAtoms()])
    rng = np.random.default_rng(1)
    initial = rng.normal(size=(mol.GetNumAtoms(), 3))
    wf = SimpleNamespace(rg=True, rg_ave_length=20, msd=True, time_step=2.0)
    md = SimpleNamespace(thermo_freq=10)
    observer = backend.Observables(mol, initial, wf, md, elapsed=100.0)
    frames = {}
    for step in range(0, 41, 5):
        xyz = initial+rng.normal(scale=.1, size=initial.shape)+[step*.1, 0, 0]
        frames[step] = xyz
        observer((100.0+step*2.0)/1000, xyz)

    def msd(xyz):
        delta = xyz-initial
        delta -= np.average(delta, axis=0, weights=masses)
        return np.mean(np.sum(delta**2, axis=1))

    def rg(xyz, f):
        f = list(f)
        center = np.average(xyz[f], axis=0, weights=masses[f])
        return np.sqrt(np.average(np.sum((xyz[f]-center)**2, axis=1), weights=masses[f]))

    # Windows exclude step 0 and end at multiples of the window length.
    assert [s for s, _ in observer.msd] == [10, 20, 30, 40]
    assert observer.msd[0][1] == pytest.approx(np.mean([msd(frames[5]), msd(frames[10])]))
    fragments = Chem.GetMolFrags(mol)
    assert [s for s, _ in observer.rg] == [20, 40]
    expected = [np.mean([rg(frames[s], f) for s in (25, 30, 35, 40)]) for f in fragments]
    np.testing.assert_allclose(observer.rg[1][1], expected)
    observer.write_rg(tmp_path/'rg.profile', step_offset=1000)
    analysis = lammps.Analyze.__new__(lammps.Analyze)
    result = analysis.calc_rg(rg_file=str(tmp_path/'rg.profile'), init=0)
    np.testing.assert_allclose(result['mean'], np.mean([observer.rg[0][1], observer.rg[1][1]], axis=0))


def _shorten(builder, cap):
    def wrapped(*args, **kwargs):
        md = builder(*args, **kwargs)
        md.thermo_freq = md.dump_freq = 100
        for wf in md.wf:
            if wf.type == 'md':
                wf.step = min(wf.step, cap)
                if getattr(wf, 'rg', False):
                    wf.rg_ave_length = 500
            else:
                wf.maxiter = 200
        return md
    return wrapped


@pytest.mark.engines
def test_eq21step_additional_and_restore(liquid, gromacs_exec, tmp_path, monkeypatch):
    monkeypatch.setenv('GROMACS_EXEC', gromacs_exec)
    np.random.seed(9)
    work = tmp_path/'eq'
    p = preset(eq.EQ21step, liquid, work, solver_path=gromacs_exec)
    p.packing = _shorten(p.packing, 300)
    p.eq21step = _shorten(p.eq21step, 50)
    p.sampling = _shorten(p.sampling, 2000)
    mol = p.exec(f_density=0.3, max_press=1000, barostat='C-rescale', omp=2, mpi=0)
    packed = utils.JSONToMol(str(work/'analyze/eq1_last.json'))
    mass = sum(a.GetMass() for a in liquid.GetAtoms())
    assert packed.cell.dx == pytest.approx(np.cbrt((mass/const.NA)/(0.3/const.cm2ang**3)), rel=1e-5)
    analysis = p.analyze()
    assert 'v_msd' in analysis.dfs[-1] and analysis.dfs[-1]['v_msd'].iat[-1] > 0
    prop = analysis.get_all_prop(temp=300.0, press=1.0, width=5, init=5, f_width=5)
    assert all(np.isfinite(prop[name]) for name in PROPERTIES), prop
    assert prop['density'] == pytest.approx(analysis.dfs[-1].Density.iloc[-5:].mean(), rel=.05)

    additional = preset(eq.Additional, mol, work, solver_path=gromacs_exec)
    assert additional.idx == 4
    additional.sampling = _shorten(additional.sampling, 500)
    additional.exec(barostat='C-rescale', omp=2, mpi=0)
    assert eq.get_final_idx(str(work/'analyze')) == 4
    restored = eq.restore(str(work/'analyze'))
    assert restored.GetNumAtoms() == liquid.GetNumAtoms()
