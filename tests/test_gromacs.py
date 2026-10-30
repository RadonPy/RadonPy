from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from radonpy.sim import gromacs as backend, md
from radonpy.sim.md_wrapper import MD_solver, MD_analyzer
from tests.helpers import gromacs, lammps
from tests.helpers.validation import compare_static
from radonpy.core import utils

FIXTURES = ['butane_gaff2'] + [p+'_'+f for p in ('pe', 'ps', 'pmma')
                               for f in ('gaff', 'gaff2', 'gaff2_mod')]


def test_conversion(molecule):
    text = backend.topology(molecule)
    bond = molecule.GetBondWithIdx(0)
    assert '%.16g' % (bond.GetDoubleProp('ff_k')*836.8) in text
    assert '1 2 yes 0.5 0.8333333333333333' in text
    pairs = set(backend._pairs14(molecule))
    assert (0, 3) in pairs and (0, 2) not in pairs
    assert len(pairs) == sum(utils.Chem.GetDistanceMatrix(molecule)[i, j] == 3
                             for i in range(molecule.GetNumAtoms())
                             for j in range(i+1, molecule.GetNumAtoms()))


def test_comparison_settings_are_local(molecule, tmp_path):
    options = lammps.comparison_options(molecule)
    solver = lammps.lammps.LAMMPS(work_dir=str(tmp_path), check_lammps_package=False)
    solver.make_input(options)
    generated = (tmp_path/options.input_file).read_text()
    assert 'pair_style lj/cut/coul/long 12.0 12.0' in generated
    assert 'pair_modify mix arithmetic shift no tail no' in generated
    assert 'kspace_style pppm 1e-8' in generated
    assert md.MD(mol=molecule).pair_style == 'lj/charmm/coul/long'


def test_portable_profile_matches_comparison_settings(molecule, tmp_path):
    texts = []
    for name, options in [('helper', lammps.comparison_options(molecule)),
                          ('profile', md.MD(mol=molecule, interaction_profile='portable'))]:
        work = tmp_path/name
        work.mkdir()
        lammps.lammps.LAMMPS(work_dir=str(work), check_lammps_package=False).make_input(options)
        texts.append((work/options.input_file).read_text())
    assert texts[0] == texts[1]
    with pytest.raises(ValueError, match='interaction_profile'):
        lammps.lammps.LAMMPS(work_dir=str(tmp_path), check_lammps_package=False).make_input(
            md.MD(mol=molecule, interaction_profile='typo'))


def test_default_profile_follows_lammps_defaults(molecule, tmp_path):
    solver = backend.Gromacs(tmp_path)
    solver.make_dat(molecule)
    options = md.MD(mol=molecule)
    options.add_min(min_style='cg')
    options.add_md('nvt', 100, time_step=0.1)
    options.add_md('npt', 100, time_step=1.0, barostat='C-rescale', shake=True)
    solver.make_input(options)

    def mdp(wf, options=options):
        return dict(line.split(' = ') for line in backend.Gromacs._mdp(options, wf, molecule, 0.0).splitlines())
    minimize, packing, sampling = (mdp(wf) for wf in options.wf)
    # lj/charmm/coul/long 8-12 angstrom and PPPM 1e-6 as a potential switch and PME of that tolerance.
    for stage in (minimize, packing, sampling):
        assert (stage['vdw-modifier'], stage['rvdw-switch'], stage['rvdw'], stage['rcoulomb']) == \
            ('Potential-switch', '0.8', '1.2', '1.2')
        assert (stage['coulombtype'], stage['ewald-rtol'], stage['pme-order']) == ('PME', '1e-06', '4')
    # A buffered pair list only once the system is past minimization and sub-fs packing.
    assert (sampling['verlet-buffer-tolerance'], sampling['nstlist']) == ('0.005', '10')
    for stage in (minimize, packing):
        assert (stage['verlet-buffer-tolerance'], stage['nstlist']) == ('-1', '1')
    portable = md.MD(mol=molecule, interaction_profile='portable')
    portable.add_md('npt', 100, time_step=1.0, barostat='C-rescale', shake=True)
    exact = mdp(portable.wf[0], portable)
    assert (exact['vdw-modifier'], exact['ewald-rtol'], exact['pme-order'], exact['nstlist']) == \
        ('None', '1e-08', '6', '1')
    options = md.MD(mol=molecule, interaction_profile='portable')
    options.add_min(min_style='fire')
    with pytest.raises(NotImplementedError, match='cg and sd'):
        solver.make_input(options)


def test_unsupported_and_unknown(molecule):
    for factory in (MD_solver, MD_analyzer):
        with pytest.raises(ValueError, match='Unknown'):
            factory('typo')
    nonperiodic = utils.deepcopy_mol(molecule)
    del nonperiodic.cell
    with pytest.raises(NotImplementedError, match='periodic'):
        backend.topology(nonperiodic)
    molecule.GetAtomWithIdx(0).SetDoubleProp('AtomicCharge', 1)
    with pytest.raises(NotImplementedError, match='charge-neutral'):
        backend.topology(molecule)


def test_extended_precision_roundtrip(molecule, tmp_path):
    molecule.GetConformer().SetAtomPosition(0, [1.12345678912, 2.33333333333, -4.23456789123])
    path = backend.Gromacs(tmp_path).make_dat(molecule)
    np.testing.assert_allclose(backend._read_molecule(path).GetConformer().GetPositions(),
                               molecule.GetConformer().GetPositions(), atol=1e-10, rtol=0)


def test_minimized_box_allows_single_precision(molecule, tmp_path):
    edge = 95.62750141196
    molecule.cell = utils.Cell(edge/2, -edge/2, edge/2, -edge/2, edge/2, -edge/2)
    xyz = (molecule.GetConformer().GetPositions()+edge/2)/10

    def final_state(box):
        lines = ['POSITION'] + ['%5d MOL   C  %5d%15.9f%15.9f%15.9f' % (i+1, i+1, *p) for i, p in enumerate(xyz)]
        lines += ['END', 'BOX', '%15.9f%15.9f%15.9f' % (box, box, box), 'END']
        (tmp_path/'final.g96').write_text('\n'.join(lines)+'\n')
        wf = SimpleNamespace(type='minimize')
        return backend.Gromacs(tmp_path)._final_state(tmp_path, utils.deepcopy_mol(molecule), 0, wf, None, 0.0, 0)

    # Mixed-precision mdrun writes the box rounded to single precision.
    result = final_state(np.float32(edge/10))
    assert result.cell.dx == pytest.approx(edge, rel=1e-6)
    with pytest.raises(RuntimeError, match='changed the box'):
        final_state(edge/10*1.0001)


def test_ring_pairs_are_not_torsion_counted():
    from rdkit import Chem
    ring = Chem.MolFromSmiles('C1CCCCC1')
    assert set(backend._pairs14(ring)) == {(0, 3), (1, 4), (2, 5)}


@pytest.mark.engines
@pytest.mark.parametrize('fixture', FIXTURES)
def test_single_point_parity(fixture, engines, tmp_path):
    molecule = utils.JSONToMol(str(Path(__file__).parent/'fixtures'/(fixture+'.json')))
    reference = lammps.single_point(molecule, tmp_path/'lammps', engines['lammps'])
    candidate = gromacs.single_point(molecule, tmp_path/'gromacs', engines['gromacs'])
    comparison = compare_static(reference, candidate, molecule.GetNumAtoms())
    assert comparison['status'] == 'pass', comparison


@pytest.mark.engines
def test_public_profile_matches_lammps_comparison_run(molecule, engines, tmp_path):
    work = tmp_path/'profile'
    work.mkdir()
    profile = md.quick_energy(utils.deepcopy_mol(molecule), solver_path=engines['lammps'],
                              work_dir=str(work), omp=1, mpi=0, interaction_profile='portable')
    reference = lammps.single_point(molecule, tmp_path/'helper', engines['lammps'])
    assert compare_static(reference, profile, molecule.GetNumAtoms())['status'] == 'pass'


@pytest.mark.engines
def test_inter_molecular_and_boundary_parity(molecule, engines, tmp_path):
    from radonpy.core import poly
    atoms = molecule.GetNumAtoms()
    cell = poly.super_cell(molecule, x=2, y=1, z=1)
    xyz = cell.GetConformer().GetPositions()
    xyz[atoms:] -= [molecule.cell.dx-6, 0, 0]
    # Wrap individual atoms: bonded interactions and exclusions must remain
    # valid even when a molecule straddles the periodic boundary.
    origin = np.array([cell.cell.xlo, cell.cell.ylo, cell.cell.zlo])
    lengths = np.array([cell.cell.dx, cell.cell.dy, cell.cell.dz])
    xyz += origin-xyz[0]
    xyz = (xyz-origin) % lengths+origin
    for i, position in enumerate(xyz):
        cell.GetConformer().SetAtomPosition(i, position)
    reference = lammps.single_point(cell, tmp_path/'lammps', engines['lammps'])
    candidate = gromacs.single_point(cell, tmp_path/'gromacs', engines['gromacs'])
    comparison = compare_static(reference, candidate, cell.GetNumAtoms())
    assert comparison['status'] == 'pass', comparison


@pytest.mark.engines
def test_tmp_clear_removes_only_run_artifacts(molecule, gromacs_exec, tmp_path):
    (tmp_path/'keep.txt').write_text('user file')
    md.quick_energy(molecule, solver='gromacs', solver_path=gromacs_exec, work_dir=str(tmp_path),
                    interaction_profile='portable', tmp_clear=True)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['keep.txt']


@pytest.mark.engines
def test_failure_cannot_reuse_previous_result(molecule, gromacs_exec, tmp_path):
    solver = backend.Gromacs(tmp_path, solver_path=gromacs_exec)
    solver.make_dat(molecule)
    solver.make_input(md.MD(mol=molecule, interaction_profile='portable'))
    solver.exec()
    with pytest.raises(FileExistsError, match='already exists'):
        solver.exec()


@pytest.mark.engines
@pytest.mark.parametrize('min_style', ['cg', 'sd'])
def test_minimization_lowers_energy_consistently(min_style, engines, tmp_path):
    # PMMA has impropers and a cell where chains cross the periodic boundary.
    molecule = utils.JSONToMol(str(Path(__file__).parent/'fixtures/pmma_gaff2.json'))
    initial = gromacs.single_point(molecule, tmp_path/'initial', engines['gromacs'])
    work = tmp_path/'min'
    work.mkdir()
    result, energy, uwstr = md.quick_min(
        molecule, min_style=min_style, solver='gromacs', solver_path=engines['gromacs'],
        work_dir=str(work), interaction_profile='portable', ftol=1e-2, maxiter=2000)
    assert energy < initial[0]-1
    # Unwrapped output preserves bonded geometry without image jumps.
    for bond in molecule.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        assert np.linalg.norm(uwstr[i]-uwstr[j]) < 2.0
    _, _, _, _, force = backend.Gromacs(work).read_traj_simple(work/'radon_md_last.dump')
    # Steepest descent does not reach ftol within maxiter, and how far it gets
    # depends on the CPU (max force 0.75-1.2 from 20), so check its progress.
    limit = 1.0 if min_style == 'cg' else 0.1*np.max(np.abs(initial[1]))
    assert np.max(np.abs(force)) < limit
    # The minimized structure is evaluated identically by both engines.
    minimized = utils.deepcopy_mol(molecule)
    for i, position in enumerate(uwstr):
        minimized.GetConformer().SetAtomPosition(i, position)
    reference = lammps.single_point(minimized, tmp_path/'lammps', engines['lammps'])
    assert compare_static(reference, (energy, force), molecule.GetNumAtoms())['status'] == 'pass'
    assert result.GetNumConformers() == molecule.GetNumConformers()
    # Minimization neither generates nor changes velocities.
    for k in ('vx', 'vy', 'vz'):
        assert [a.GetDoubleProp(k) for a in result.GetAtoms()] == [a.GetDoubleProp(k) for a in molecule.GetAtoms()]


@pytest.mark.engines
def test_minimize_all_conformers_keeps_ids(molecule, gromacs_exec, tmp_path):
    from rdkit import Chem
    molecule.AddConformer(Chem.Conformer(molecule.GetConformer()), assignId=True)
    for conformer, cid in zip(molecule.GetConformers(), (7, 13)):
        conformer.SetId(cid)
    result, energies, xyz = md.quick_min_all(
        molecule, solver='gromacs', solver_path=gromacs_exec, work_dir=str(tmp_path), mpi=0,
        interaction_profile='portable', ftol=.01, maxiter=500)
    assert [c.GetId() for c in result.GetConformers()] == [7, 13]
    assert len(energies) == len(xyz) == 2
    np.testing.assert_allclose(energies[0], energies[1], atol=1e-8)
    np.testing.assert_allclose(xyz[0], xyz[1], atol=1e-8)
