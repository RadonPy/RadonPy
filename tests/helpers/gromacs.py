"""GROMACS runners for validation tests, using the public RadonPy API.

Native files stay in the test's temporary directory.
"""
from pathlib import Path

from radonpy.core import utils
from radonpy.sim import gromacs, md
from radonpy.sim.md_wrapper import MD_analyzer


def run(mol, work, executable, *, steps=0, time_step=.1, stride=1):
    """Run a single point or NVE (velocity Verlet) in a fresh directory.

    Return thermodynamic frames (RadonPy units), forces in kcal/(mol angstrom)
    for a single point, and the final unwrapped coordinates and velocities.
    This runner uses one CPU thread.
    """
    work = Path(work)
    work.mkdir(parents=True, exist_ok=False)
    state = utils.deepcopy_mol(mol)
    options = md.MD(mol=state, interaction_profile='portable', thermo_freq=stride, dump_freq=stride)
    if steps:
        options.add_md('nve', steps, time_step=time_step)
    solver = gromacs.Gromacs(work_dir=str(work), solver_path=executable)
    solver.make_dat(state, file_name=options.dat_file, velocity=False)
    solver.make_input(options)
    solver.exec(omp=1, mpi=0, gpu=0)
    frame = MD_analyzer('gromacs', log_file=str(work/options.log_file)).dfs[-1]
    uwstr, _, _, velocity, force = solver.read_traj_simple(str(work/options.outstr))
    return frame, (force if steps == 0 else None), (uwstr, velocity)


def single_point(mol, work, executable):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=False)
    return md.quick_energy(utils.deepcopy_mol(mol), solver='gromacs', solver_path=executable,
                           work_dir=str(work), omp=1, mpi=0, interaction_profile='portable')
