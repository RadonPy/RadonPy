import json
from pathlib import Path

import numpy as np
import pytest
from radonpy.core import calc, utils
from radonpy.sim import gromacs as backend, md
from radonpy.sim.md_wrapper import MD_analyzer
from tests.helpers import gromacs, lammps

FIXTURES = Path(__file__).parent/'fixtures'


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
        frame, _, _ = runner.run(molecule, work, executable,
                                 steps=int(20/timestep), time_step=timestep, stride=10)
        energy = frame.TotEng.to_numpy()*4.184/molecule.GetNumAtoms()
        drifts.append(np.max(np.abs(energy-energy[0])))
    assert drifts[0] < .01
    assert drifts[1] < .6*drifts[0]+1e-5


@pytest.mark.engines
def test_nve_trajectory_matches_lammps(engines, tmp_path):
    # Both engines integrate with velocity Verlet; with the same Hamiltonian
    # and initial velocities, short trajectories must coincide.
    molecule = utils.JSONToMol(str(FIXTURES/'pmma_gaff2.json'))
    np.random.seed(7)
    calc.set_velocity(molecule, 300)
    # GROMACS removes center-of-mass motion; start without it, as LAMMPS
    # "velocity ... mom yes" does.
    masses = np.array([a.GetMass() for a in molecule.GetAtoms()])
    velocity = np.array([[a.GetDoubleProp(k) for k in ('vx', 'vy', 'vz')] for a in molecule.GetAtoms()])
    velocity -= np.average(velocity, axis=0, weights=masses)
    for atom, v in zip(molecule.GetAtoms(), velocity):
        for k, value in zip(('vx', 'vy', 'vz'), v):
            atom.SetDoubleProp(k, float(value))
    results = {name: runner.run(molecule, tmp_path/name, engines[name], steps=200, time_step=.5, stride=20)
               for name, runner in (('lammps', lammps), ('gromacs', gromacs))}
    (lf, _, (lx, lv)), (gf, _, (gx, gv)) = results['lammps'], results['gromacs']
    assert list(lf.index) == list(gf.index)
    atoms = molecule.GetNumAtoms()
    np.testing.assert_allclose(gf.PotEng*4.184/atoms, lf.PotEng*4.184/atoms, atol=2e-4)
    np.testing.assert_allclose(gf.KinEng*4.184/atoms, lf.KinEng*4.184/atoms, atol=2e-4)
    np.testing.assert_allclose(gx, lx, atol=1e-4, rtol=0)
    np.testing.assert_allclose(gv, lv, atol=1e-6, rtol=0)


def _short_md(molecule, gromacs_exec, work, ensemble, **kwargs):
    kwargs.setdefault('step', 200)
    kwargs.setdefault('time_step', 1.0)
    return getattr(md, 'quick_'+ensemble)(
        molecule, solver='gromacs', solver_path=gromacs_exec, work_dir=str(work),
        interaction_profile='portable', **kwargs)


@pytest.mark.engines
@pytest.mark.parametrize('ensemble, options', [
    ('nvt', {}), ('nvt', {'thermostat': 'csvr'}), ('nvt', {'thermostat': 'Langevin'}),
    ('nvt', {'f_temp': 500}), ('npt', {'barostat': 'Parrinello-Rahman'}),
    ('npt', {'barostat': 'C-rescale', 'press': 1000.0}),
])
def test_thermostats_and_barostats_run(molecule, gromacs_exec, tmp_path, ensemble, options):
    np.random.seed(23)
    calc.set_velocity(molecule, 300)
    result = _short_md(molecule, gromacs_exec, tmp_path, ensemble, **options)
    frame = MD_analyzer('gromacs', log_file=str(tmp_path/'radon_md.log')).dfs[-1]
    assert list(frame.index) == [0, 200]
    assert frame.Time.iat[-1] == pytest.approx(200.0)
    assert np.isfinite(frame.to_numpy()).all()
    assert not np.array_equal(result[1], molecule.GetConformer().GetPositions())
    assert all(result[0].GetAtomWithIdx(0).HasProp(k) for k in ('vx', 'vy', 'vz'))
    if ensemble == 'npt':
        cell = result[2]
        assert cell.shape == (3, 2)
        assert result[0].cell.dx == pytest.approx(cell[0, 1]-cell[0, 0])
        assert frame.Volume.iat[-1] == pytest.approx(result[0].cell.volume, rel=1e-6)


@pytest.mark.engines
def test_shake_keeps_hydrogen_bond_lengths(molecule, gromacs_exec, tmp_path):
    np.random.seed(5)
    calc.set_velocity(molecule, 300)
    _, xyz = _short_md(molecule, gromacs_exec, tmp_path, 'nvt', shake=True, time_step=2.0)
    for bond in molecule.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        length = np.linalg.norm(xyz[i]-xyz[j])
        if backend.is_constrained(molecule, i, j):
            assert length == pytest.approx(bond.GetDoubleProp('ff_r0'), rel=1e-4)
        else:
            assert abs(length-bond.GetDoubleProp('ff_r0')) < .3


@pytest.mark.engines
def test_unwrapped_coordinates_follow_molecules_across_boundaries(engines, tmp_path):
    # PMMA chains straddle the cell boundary; a hot run moves them further.
    molecule = utils.JSONToMol(str(FIXTURES/'pmma_gaff2.json'))
    np.random.seed(11)
    calc.set_velocity(molecule, 600)
    options = md.MD(mol=molecule, interaction_profile='portable', thermo_freq=50, dump_freq=50,
                    xtc_file='radon_md.xtc')
    options.add_md('nvt', 2000, time_step=1.0, t_start=600, t_stop=600)
    solver = backend.Gromacs(work_dir=str(tmp_path), solver_path=engines['gromacs'])
    solver.make_dat(molecule, file_name=options.dat_file)
    solver.make_input(options)
    solver.exec()
    uwstr, wstr, cell, _, _ = solver.read_traj_simple(tmp_path/options.outstr)
    for bond in molecule.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        assert np.linalg.norm(uwstr[i]-uwstr[j]) < 2.0
    assert ((wstr >= cell[:, 0]) & (wstr < cell[:, 1])).all()
    from mdtraj.formats import XTCTrajectoryFile
    with XTCTrajectoryFile(str(tmp_path/'radon_md.xtc')) as trajectory:
        xyz, time, _, _ = trajectory.read()
    assert len(time) == 41 and np.all(np.diff(time) > 0)
    np.testing.assert_allclose(xyz[-1]*10, uwstr, atol=2e-2)
    steps = np.linalg.norm(np.diff(xyz*10, axis=0), axis=2)
    assert steps.max() < 5.0


def test_unwrap_trajectory_follows_images(tmp_path):
    from mdtraj.formats import XTCTrajectoryFile
    # One atom crossing +x twice and -y once in a 2 nm box (nm).
    path = np.array([[1.8, .1, 1.], [1.95, .05, 1.], [2.1, -.05, 1.], [2.3, -.1, 1.],
                     [2.9, -.15, 1.], [3.6, -.15, 1.], [4.1, -.15, 1.]])
    box = np.tile(np.eye(3, dtype=np.float32)*2.0, (len(path), 1, 1))
    with XTCTrajectoryFile(str(tmp_path/'in.xtc'), 'w') as stream:
        stream.write((path % 2.0)[:, None, :].astype(np.float32), time=np.arange(len(path), dtype=np.float32),
                     step=np.arange(len(path)), box=box)
    last = backend.unwrap_trajectory(tmp_path/'in.xtc', path[:1]*10, tmp_path/'out.xtc')
    np.testing.assert_allclose(last, path[-1:]*10, atol=1e-3)
    with XTCTrajectoryFile(str(tmp_path/'out.xtc')) as stream:
        xyz = stream.read()[0]
    np.testing.assert_allclose(xyz[:, 0], path, atol=1e-4)


@pytest.mark.engines
def test_multistage_workflow_is_continuous(molecule, gromacs_exec, tmp_path):
    np.random.seed(3)
    options = md.MD(mol=molecule, interaction_profile='portable', thermo_freq=50, dump_freq=50,
                    xtc_file='radon_md.xtc', set_init_velocity=300.0)
    options.add_min(min_style='cg', ftol=1e-2, maxiter=200)
    options.add_md('nvt', 200, time_step=1.0)
    options.add_md('npt', 100, time_step=2.0, shake=True, barostat='C-rescale')
    solver = backend.Gromacs(work_dir=str(tmp_path), solver_path=gromacs_exec)
    solver.make_dat(molecule, file_name=options.dat_file, velocity=False)
    solver.make_input(options)
    solver.exec()
    frames = MD_analyzer('gromacs', log_file=str(tmp_path/options.log_file)).dfs
    assert len(frames) == 3
    nvt, npt = frames[1], frames[2]
    assert nvt.index[0] == frames[0].index[-1] and nvt.index[-1] == nvt.index[0]+200
    assert npt.index[0] == nvt.index[-1] and npt.index[-1] == npt.index[0]+100
    assert nvt.Time.iat[0] == 0 and nvt.Time.iat[-1] == pytest.approx(200)
    assert npt.Time.iat[0] == pytest.approx(200) and npt.Time.iat[-1] == pytest.approx(400)
    from mdtraj.formats import XTCTrajectoryFile
    with XTCTrajectoryFile(str(tmp_path/'radon_md.xtc')) as trajectory:
        time = trajectory.read()[1]
    # dump_freq counts steps: every 50 fs at 1 fs, then every 100 fs at 2 fs.
    np.testing.assert_allclose(time, [0, .05, .1, .15, .2, .3, .4], atol=1e-6)


@pytest.mark.engines
def test_interrupted_run_resumes_from_checkpoint(molecule, gromacs_exec, tmp_path, monkeypatch):
    np.random.seed(17)
    calc.set_velocity(molecule, 300)

    def options():
        result = md.MD(mol=molecule, interaction_profile='portable', thermo_freq=100, dump_freq=100)
        result.add_md('nvt', 300, time_step=1.0)
        result.add_md('nvt', 3000, time_step=1.0)
        return result

    solver = backend.Gromacs(work_dir=str(tmp_path), solver_path=gromacs_exec)
    solver.make_dat(molecule, file_name='snapshot')
    solver.make_input(options())
    command = backend.Gromacs._command
    calls = []

    def stop_second_stage_early(self, args, cwd, **kwargs):
        if args[0] == 'mdrun' and Path(cwd).name == 'stage_001' and not calls:
            calls.append(cwd)
            # A tiny wall-time budget makes mdrun stop at a checkpoint.
            args = list(args)+['-maxh', '1e-7']
        return command(self, args, cwd, **kwargs)

    monkeypatch.setattr(backend.Gromacs, '_command', stop_second_stage_early)
    with pytest.raises(RuntimeError, match='resume=True'):
        solver.exec()
    assert (tmp_path/'radon_gmx.json.gmx/stage_000/completed.json').exists()
    assert (tmp_path/'radon_gmx.json.gmx/stage_001/run.cpt').exists()
    assert not (tmp_path/'radon_md.log.gmx.json').exists()
    with pytest.raises(FileExistsError):
        solver.exec()
    solver.exec(resume=True)
    log = (tmp_path/'radon_gmx.json.gmx/stage_001/commands.log').read_text()
    assert '"-cpi"' in log
    frames = MD_analyzer('gromacs', log_file=str(tmp_path/'radon_md.log')).dfs
    assert frames[1].index[0] == 300 and frames[1].index[-1] == 3300
    assert frames[1].Time.iat[-1] == pytest.approx(3300)
    assert not frames[1].index.duplicated().any()
    # Resuming refuses a manifest that no longer matches the run.
    manifest = json.loads((tmp_path/'radon_gmx.json').read_text())
    manifest['workflow'][1]['step'] = 10
    (tmp_path/'radon_gmx.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='changed'):
        solver.exec(resume=True)


def test_unsupported_dynamics_are_rejected(molecule, tmp_path):
    solver = backend.Gromacs(tmp_path)
    solver.make_dat(molecule)
    cases = [
        ('npt', {}, ValueError, 'MTK'),
        ('nph', {}, NotImplementedError, 'nve, nvt and npt'),
        ('nvt', {'thermostat': 'Berendsen'}, NotImplementedError, 'thermostats'),
        ('npt', {'barostat': 'C-rescale', 'p_stop': 2.0}, NotImplementedError, 'ramps'),
        ('npt', {'barostat': 'C-rescale', 'p_aniso': True}, NotImplementedError, 'Parrinello-Rahman'),
        ('nve', {'nve_limit': .1}, NotImplementedError, 'nve/limit'),
        ('nvt', {'timeave': True}, NotImplementedError, 'timeave'),
    ]
    for ensemble, kwargs, error, message in cases:
        options = md.MD(mol=molecule, interaction_profile='portable')
        options.add_md(ensemble, 10, **kwargs)
        with pytest.raises(error, match=message):
            solver.make_input(options)
    # Constant and cosine fields are supported; the ef_dp tuning ramp is not.
    options = md.MD(mol=molecule, interaction_profile='portable')
    options.add_md('nvt', 10)
    options.wf[0].add_efield(evalue=0.1, axis='x', rate=1e-6)
    with pytest.raises(NotImplementedError, match='linear ramps'):
        solver.make_input(options)
    options = md.MD(mol=molecule, interaction_profile='portable')
    options.add_md('nvt', 10)
    options.wf[0].add_deform(dftype='scale', axis='x')
    with pytest.raises(NotImplementedError, match='isotropic NVT deformation'):
        solver.make_input(options)


def test_gpu_numerics_change_only_scheduling_and_pme_order(tmp_path):
    from radonpy.core import poly
    np.random.seed(5)
    octane = utils.JSONToMol(str(FIXTURES/'pe_gaff2.json'))
    cell = poly.amorphous_cell(octane, 30, density=0.25)
    options = md.MD(mol=cell, interaction_profile='portable', thermo_freq=1000)
    options.add_md('npt', 10, barostat='C-rescale', t_dump=100.0, p_dump=1000.0, shake=True)

    def mdp(gpu):
        np.random.seed(0)
        return dict(line.split(' = ') for line in
                    backend.Gromacs._mdp(options, options.wf[0], cell, 0.0, gpu=gpu).splitlines())
    cpu, gpu = mdp(False), mdp(True)
    changed = {k: (cpu[k], gpu[k]) for k in cpu if cpu[k] != gpu[k]}
    assert changed == {'pme-order': ('6', '4'), 'verlet-buffer-tolerance': ('-1', '0.005'),
                       'nstlist': ('1', '10'), 'nstcalcenergy': ('1', '100'),
                       'nstpcouple': ('1', '10')}
    assert cpu.keys() == gpu.keys()
    # Nose-Hoover stays coupled every step; v-rescale within 20 steps per tau-t.
    options.wf[0].thermostat = 'csvr'
    assert mdp(True)['nsttcouple'] == '5'


def test_lammps_gpu_uses_kokkos_without_gpu_package(tmp_path):
    from radonpy.sim import lammps as lmp
    solver = lmp.LAMMPS(work_dir=str(tmp_path), solver_path='lmp', check_lammps_package=False)
    solver.package.update(omp=False, intel=False, opt=False, gpu=False, kokkos_gpu=False)
    assert '-sf' not in solver.exec(gpu=1, mpi=0, return_cmd=True)
    solver.package['kokkos_gpu'] = True
    assert '-k on g 1 -sf kk' in solver.exec(gpu=1, mpi=0, return_cmd=True)
    assert '-k on' not in solver.exec(gpu=0, mpi=0, return_cmd=True)


def test_strain_rate_deformation_matches_lammps(molecule, tmp_path):
    from radonpy.sim import lammps
    def stage(**kwargs):
        options = md.MD(mol=molecule, interaction_profile='portable')
        options.add_md('npt', 1000, shake=True, **kwargs)
        options.wf[0].add_deform(dftype='erate', axis='x', deform_rate=5e-6, remap='x')
        return options
    # LAMMPS: L(t) = L0 (1 + 5e-6/fs t) along x, with y and z barostatted (as the elong preset writes it).
    solver = lammps.LAMMPS(work_dir=str(tmp_path), check_lammps_package=False)
    solver.make_input(stage(), file_name='deform.in')
    text = (tmp_path/'deform.in').read_text()
    assert 'fix DEF1 all deform 1 x erate 5.000000e-06 units box remap x' in text
    assert ' y 1.000000 1.000000 1000.000000 z 1.000000 1.000000 1000.000000 ' in text
    # GROMACS: a constant box velocity L0*rate and zero compressibility along x.
    options = stage(barostat='Parrinello-Rahman')
    gromacs = backend.Gromacs(tmp_path)
    gromacs.make_dat(molecule)
    gromacs.make_input(options, file_name='check.json')
    mdp = dict(line.split(' = ') for line in backend.Gromacs._mdp(options, options.wf[0], molecule, 0.0).splitlines())
    velocity = [float(x) for x in mdp['deform'].split()]
    assert velocity[0] == pytest.approx(molecule.cell.dx/10*5e-6*1000) and velocity[1:] == [0]*5
    assert mdp['pcoupltype'] == 'anisotropic'
    assert mdp['compressibility'].split() == ['0', '4.5e-05', '4.5e-05', '0', '0', '0']
    with pytest.raises(NotImplementedError, match='Parrinello-Rahman'):
        crescale = md.MD(mol=molecule, interaction_profile='portable')
        crescale.add_md('npt', 1000, barostat='C-rescale', shake=True)
        crescale.wf[0].add_deform(dftype='erate', axis='x', deform_rate=5e-6, remap='x')
        gromacs.make_input(crescale, file_name='check.json')
