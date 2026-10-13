"""Use the existing LAMMPS API with explicit settings for comparison tests."""
from pathlib import Path

from radonpy.core import utils
from radonpy.sim import lammps, md


def comparison_options(mol, **kwargs):
    # MD(mol=...) reads the molecule's styles, so set the comparison settings
    # afterward. Nothing is added to the production API or its defaults.
    options = md.MD(mol=mol, **kwargs)
    options.pair_style = 'lj/cut/coul/long'
    options.cutoff_in = options.cutoff_out = 12.0
    options.kspace_style = 'pppm'
    options.kspace_style_accuracy = '1e-8'
    options.pair_modify = 'mix arithmetic shift no tail no'
    options.special_bonds = 'amber'
    return options


def run(mol, work, executable, *, steps=0, time_step=.1, stride=1):
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=False)
    state = utils.deepcopy_mol(mol)
    options = comparison_options(state, thermo_freq=stride, dump_freq=stride)
    if steps:
        options.add_md('nve', steps, time_step=time_step)
    solver = lammps.LAMMPS(work_dir=str(work), solver_path=executable)
    solver.make_dat(state, file_name=options.dat_file, velocity=bool(steps))
    solver.make_input(options)
    result = solver.exec(omp=1, mpi=0, gpu=0)
    if result.returncode:
        raise RuntimeError('LAMMPS failed; see %s' % work)
    frame = lammps.Analyze(log_file=str(work/options.log_file)).dfs[-1]
    force = None
    if steps == 0:
        force = solver.read_traj_simple(str(work/options.outstr))[-1]
    return frame, force


def single_point(mol, work, executable):
    frame, force = run(mol, work, executable)
    return frame['PotEng'].iat[-1], force
