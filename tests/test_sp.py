from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from radonpy.core import poly, utils
from radonpy.sim.preset import sp

FIXTURES = Path(__file__).parent/'fixtures'


@pytest.fixture
def cell():
    np.random.seed(7)
    ester = utils.JSONToMol(str(FIXTURES/'pmma_gaff2.json'))
    return poly.amorphous_cell(ester, 6, density=0.2)


def frames(cell, n=3):
    """Perturbed structures in slightly different boxes, on the XTC grid (0.01 angstrom) so both engines read them exactly."""
    rng = np.random.default_rng(1)
    xyz = cell.GetConformer().GetPositions()
    lengths = np.array([cell.cell.dx, cell.cell.dy, cell.cell.dz])
    return (np.round([xyz+rng.normal(scale=0.05, size=xyz.shape) for _ in range(n)], 2),
            np.round([lengths*(1+0.01*k) for k in range(n)], 2))


def write_eq_trajectory(work, xyz, lengths):
    """The eq3 trajectory as LAMMPS (dump, wrapped with image flags) and GROMACS (xtc) write it."""
    from mdtraj.formats import XTCTrajectoryFile
    work.mkdir()
    with open(work/'eq3.dump', 'w') as dump:
        for k, (x, box) in enumerate(zip(xyz, lengths)):
            image = np.floor(x/box).astype(int)
            dump.write('ITEM: TIMESTEP\n%d\nITEM: NUMBER OF ATOMS\n%d\nITEM: BOX BOUNDS pp pp pp\n' % (k*1000, len(x)))
            dump.write(''.join('0 %.10f\n' % b for b in box))
            dump.write('ITEM: ATOMS id x y z ix iy iz vx vy vz\n')
            dump.write(''.join('%d %.10f %.10f %.10f %d %d %d 0 0 0\n' % (i+1, *p, *m)
                               for i, (p, m) in enumerate(zip(x-image*box, image))))
    boxes = np.zeros((len(xyz), 3, 3))
    boxes[:, [0, 1, 2], [0, 1, 2]] = lengths/10
    with XTCTrajectoryFile(str(work/'eq3.xtc'), 'w') as xtc:
        xtc.write(xyz/10, time=np.arange(len(xyz), dtype=float), step=np.arange(len(xyz))*1000, box=boxes)


def rerun(cell, engine, executable, work, xyz, lengths):
    write_eq_trajectory(work, xyz, lengths)
    p = sp.Rerun(cell, work_dir=str(work), solver_path=executable, idx=3, solver=engine, interaction_profile='portable')
    p.exec(omp=1, mpi=0, gpu=0)
    return p.analyze().dfs[-1].reset_index(drop=True)


@pytest.mark.engines
def test_energy_decomposition_matches_lammps(cell, engines, tmp_path):
    xyz, lengths = frames(cell)
    lmp, gmx = (rerun(cell, engine, engines[engine], tmp_path/engine, xyz, lengths) for engine in ('lammps', 'gromacs'))
    assert len(lmp) == len(gmx) == len(xyz)
    np.testing.assert_allclose(gmx['Volume'], np.prod(lengths, axis=1), rtol=1e-6)
    # Real and reciprocal space are split differently by PPPM and PME, so electrostatics are compared as sums.
    np.testing.assert_allclose(gmx['E_vdwl'], lmp['E_vdwl'], rtol=1e-5)
    np.testing.assert_allclose(gmx['E_coul']+gmx['E_long'], lmp['E_coul']+lmp['E_long'], rtol=1e-5, atol=1e-3)
    sums = {}
    for i in range(6):
        vdw, coul, long = ('v_mol%i_%s_intra' % (i, k) for k in ('evdw', 'ecoul', 'elong'))
        np.testing.assert_allclose(gmx[vdw], lmp[vdw], rtol=1e-5, atol=5e-4)
        sums[i] = gmx[coul]+gmx[long]
        # LAMMPS group/group kspace is accurate to about 1e-4 of its +-60 kcal/mol parts.
        np.testing.assert_allclose(sums[i], lmp[coul]+lmp[long], atol=0.05)
    # The GROMACS intramolecular electrostatics are those of the molecule alone, which plain LAMMPS PPPM gives exactly.
    alone = utils.deepcopy_mol(cell)
    atoms = Chem.GetMolFrags(alone)[0]
    for atom in alone.GetAtoms():
        if atom.GetIdx() not in atoms:
            atom.SetDoubleProp('AtomicCharge', 0.0)
            atom.SetDoubleProp('ff_epsilon', 0.0)
    exact = rerun(alone, 'lammps', engines['lammps'], tmp_path/'alone', xyz, lengths)
    np.testing.assert_allclose(-sums[0], exact['E_coul']+exact['E_long'], rtol=1e-5, atol=1e-3)
