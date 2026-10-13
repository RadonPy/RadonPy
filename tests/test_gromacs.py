from pathlib import Path

import numpy as np
import pytest

from radonpy.sim import md
from tests.helpers import gromacs, lammps
from tests.helpers.validation import compare_static
from radonpy.core import utils


def test_conversion(molecule):
    text = gromacs.topology(molecule)
    bond = molecule.GetBondWithIdx(0)
    assert '%.16g' % (bond.GetDoubleProp('ff_k')*836.8) in text
    assert '1 2 yes 0.5 0.8333333333333333' in text
    pairs = set(gromacs._pairs14(molecule))
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


def test_unsupported_charge(molecule):
    molecule.GetAtomWithIdx(0).SetDoubleProp('AtomicCharge', 1)
    with pytest.raises(NotImplementedError, match='charge-neutral'):
        gromacs.topology(molecule)


def test_extended_precision_roundtrip(molecule, tmp_path):
    molecule.GetConformer().SetAtomPosition(0, [1.12345678912, 2.33333333333, -4.23456789123])
    path = tmp_path/'molecule.json'
    gromacs._write_molecule(molecule, path)
    np.testing.assert_allclose(gromacs._read_molecule(path).GetConformer().GetPositions(),
                               molecule.GetConformer().GetPositions(), atol=1e-10, rtol=0)


@pytest.mark.engines
@pytest.mark.parametrize('fixture', ['butane_gaff2'] + [p+'_'+f for p in ('pe', 'ps', 'pmma')
                                                       for f in ('gaff', 'gaff2', 'gaff2_mod')])
def test_single_point_parity(fixture, engines, tmp_path):
    molecule = utils.JSONToMol(str(Path(__file__).parent/'fixtures'/(fixture+'.json')))
    reference = lammps.single_point(molecule, tmp_path/'lammps', engines['lammps'])
    candidate = gromacs.single_point(molecule, tmp_path/'gromacs', engines['gromacs'])
    comparison = compare_static(reference, candidate, molecule.GetNumAtoms())
    assert comparison['status'] == 'pass', comparison


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
def test_failure_cannot_reuse_previous_result(molecule, gromacs_exec, tmp_path):
    work = tmp_path/'gromacs'
    gromacs.single_point(molecule, work, gromacs_exec)
    with pytest.raises(FileExistsError):
        gromacs.single_point(molecule, work, gromacs_exec)


def test_ring_pairs_are_not_torsion_counted():
    from rdkit import Chem
    ring = Chem.MolFromSmiles('C1CCCCC1')
    assert set(gromacs._pairs14(ring)) == {(0, 3), (1, 4), (2, 5)}
