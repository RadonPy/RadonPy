"""Opt-in ensemble comparison of LAMMPS and GROMACS dynamics (portable profile).

Not collected by pytest. Each command writes report.json with block-averaged
estimates; "consistent" means the engines differ by less than three combined
standard errors; quantities with fewer than 20 blocks of five autocorrelation
times are reported as inconclusive. Use as evidence for review, not as a
release gate. Existing engine outputs in DIR are reused, so a report can be
regenerated without rerunning the simulations.

  python -m tests.ensemble_validation nvt --output DIR [--steps 500000]
  python -m tests.ensemble_validation npt --output DIR [--steps 400000]

Options: --threads N (CPU threads or MPI ranks per arm), --gpu (one GPU per
arm: GROMACS GPU numerics with a mixed-precision build, LAMMPS KOKKOS; select
the device with CUDA_VISIBLE_DEVICES), --shake (constrain bonds to hydrogen),
--sequential (run the arms one after another).

NVT: one PMMA oligomer with the CSVR (Bussi) thermostat in both engines;
mean potential energy, and kinetic energy mean/variance against the canonical
values for 3N-3 degrees of freedom.
NPT: liquid octane (150 molecules) at 300 K and 1 atm from one shared
equilibrated state; mean density with LAMMPS Nose-Hoover (MTK) and GROMACS
C-rescale and Parrinello-Rahman barostats.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import numpy as np

from radonpy.core import calc, poly, utils
from radonpy.sim import gromacs, lammps, md
from radonpy.sim.md_wrapper import MD_analyzer
from tests.helpers.validation import autocorrelation_time

FIXTURES = Path(__file__).parent/'fixtures'
KB = 0.0019872043  # kcal/(mol K)


def estimate(values):
    """Mean and standard error from blocks of five autocorrelation times."""
    values = np.asarray(values, dtype=float)
    tau = autocorrelation_time(values)
    size = max(1, int(np.ceil(5*tau)))
    blocks = len(values)//size
    result = {'mean': float(values.mean()), 'tau_samples': float(tau), 'blocks': blocks}
    if blocks < 20:
        return {**result, 'se': None, 'status': 'insufficient sampling'}
    means = values[:blocks*size].reshape(blocks, size).mean(axis=1)
    return {**result, 'se': float(means.std(ddof=1)/np.sqrt(blocks)), 'status': 'ok'}


def compare(a, b):
    diff = b['mean']-a['mean']
    if a['se'] is None or b['se'] is None:
        return {'difference': diff, 'se': None, 'z': None, 'consistent': 'inconclusive'}
    se = float(np.hypot(a['se'], b['se']))
    return {'difference': diff, 'se': se, 'z': diff/se if se else float('inf'),
            'consistent': bool(abs(diff) < 3*se)}


def zero_momentum(mol):
    masses = np.array([a.GetMass() for a in mol.GetAtoms()])
    velocity = np.array([[a.GetDoubleProp(k) for k in ('vx', 'vy', 'vz')] for a in mol.GetAtoms()])
    velocity -= np.average(velocity, axis=0, weights=masses)
    for atom, v in zip(mol.GetAtoms(), velocity):
        for k, value in zip(('vx', 'vy', 'vz'), v):
            atom.SetDoubleProp(k, float(value))


def simulate(engine, mol, work, steps, ensemble, threads, stride=10, gpu=False, **kwargs):
    work = Path(work)
    log = work/'radon_md.log'
    if (work/'radon_md.log.gmx.json').exists() or (engine == 'lammps' and (work/'radon_md_last.dump').exists()):
        return None, MD_analyzer(engine, log_file=str(log)).dfs[-1]
    work.mkdir(parents=True, exist_ok=False)
    mol = utils.deepcopy_mol(mol)
    options = md.MD(mol=mol, interaction_profile='portable', thermo_freq=stride,
                    dump_freq=max(stride, 1000), rst=False)
    options.add_md(ensemble, steps, time_step=1.0, **kwargs)
    if engine == 'lammps':
        solver = lammps.LAMMPS(work_dir=str(work), solver_path=os.environ.get('LAMMPS_EXEC', 'lmp'))
        solver.make_dat(mol, file_name=options.dat_file)
        # KOKKOS runs one rank per GPU.
        mol = solver.run(options, mol=mol, omp=0, mpi=1 if gpu else threads, gpu=int(gpu))
    else:
        solver = gromacs.Gromacs(work_dir=str(work), solver_path=os.environ.get('GROMACS_EXEC', 'gmx'))
        mol = solver.run(options, mol=mol, omp=threads, gpu=int(gpu))
    frame = MD_analyzer(engine, log_file=str(work/options.log_file)).dfs[-1]
    return mol, frame


def run_arms(arms, workers):
    """Run {name: (callable, args, kwargs)} concurrently or one at a time."""
    with ThreadPoolExecutor(len(arms) if workers is None else workers) as pool:
        jobs = {name: pool.submit(f, *a, **kw) for name, (f, a, kw) in arms.items()}
        return {name: job.result()[1] for name, job in jobs.items()}


def nvt(output, steps, threads, workers=None, **options):
    mol = utils.JSONToMol(str(FIXTURES/'pmma_gaff2.json'))
    np.random.seed(2024)
    calc.set_velocity(mol, 300)
    zero_momentum(mol)
    dof = 3*mol.GetNumAtoms()-3
    arms = {'lammps': {'thermostat': 'csvr'}, 'gromacs': {'thermostat': 'csvr'}}
    frames = run_arms({name: (simulate, (name, mol, output/name, steps, 'nvt', threads),
                              dict(t_start=300.0, t_stop=300.0, t_dump=100.0, **options, **kw))
                       for name, kw in arms.items()}, workers)
    report = {'system': 'pmma_gaff2 single chain', 'atoms': mol.GetNumAtoms(), 'steps': steps,
              'options': options,
              'time_step_fs': 1.0, 'expected': {'KinEng': dof*KB*300/2, 'KinEng_var': dof*(KB*300)**2/2},
              'engines': {}}
    for name, frame in frames.items():
        frame = frame.iloc[len(frame)//5:]
        ke = frame.KinEng.to_numpy()
        report['engines'][name] = {'PotEng': estimate(frame.PotEng), 'KinEng': estimate(ke),
                                   'KinEng_var': float(ke.var(ddof=1)), 'Temp': estimate(frame.Temp)}
    a, b = report['engines']['lammps'], report['engines']['gromacs']
    report['gromacs_minus_lammps'] = {key: compare(a[key], b[key]) for key in ('PotEng', 'KinEng', 'Temp')}
    return report


def prepare_liquid(output, threads, gpu=False):
    """Pack and compress liquid octane once; both engines start from this state."""
    path = output/'liquid.json'
    if path.exists():
        return gromacs._read_molecule(path)
    octane = utils.JSONToMol(str(FIXTURES/'pe_gaff2.json'))
    np.random.seed(7)
    cell = poly.amorphous_cell(octane, 150, density=0.25)
    work = output/'prepare'
    common = dict(solver='gromacs', interaction_profile='portable', omp=threads, gpu=int(gpu))
    cell, _, _ = md.quick_min(cell, work_dir=str(work/'min'), ftol=1.0, maxiter=5000, **common)
    cell, _ = md.quick_nvt(cell, step=20000, temp=300.0, work_dir=str(work/'nvt'), **common)
    cell, _, _ = md.quick_npt(cell, step=50000, temp=300.0, press=1000.0, barostat='C-rescale',
                              work_dir=str(work/'compress'), **common)
    cell, _, _ = md.quick_npt(cell, step=100000, temp=300.0, press=1.0, barostat='C-rescale',
                              work_dir=str(work/'relax'), **common)
    zero_momentum(cell)
    gromacs._write_molecule(cell, path)
    return cell


def npt(output, steps, threads, workers=None, **options):
    cell = prepare_liquid(output, threads, options.get('gpu', False))
    arms = {'lammps': ('lammps', {'barostat': 'Nose-Hoover'}),
            'gromacs_c_rescale': ('gromacs', {'barostat': 'C-rescale'}),
            'gromacs_parrinello_rahman': ('gromacs', {'barostat': 'Parrinello-Rahman'})}
    frames = run_arms({name: (simulate, (engine, cell, output/name, steps, 'npt', threads),
                              dict(t_start=300.0, t_stop=300.0, t_dump=100.0, p_start=1.0,
                                   p_stop=1.0, p_dump=1000.0, **options, **kw))
                       for name, (engine, kw) in arms.items()}, workers)
    report = {'system': '150 octane (pe_gaff2), 300 K, 1 atm', 'atoms': cell.GetNumAtoms(),
              'options': options,
              'steps': steps, 'time_step_fs': 1.0, 'engines': {}}
    for name, frame in frames.items():
        frame = frame.iloc[len(frame)//5:]
        report['engines'][name] = {key: estimate(frame[key]) for key in ('Density', 'PotEng', 'Temp')}
    reference = report['engines']['lammps']
    report['minus_lammps'] = {name: {key: compare(reference[key], values[key])
                                     for key in ('Density', 'PotEng')}
                              for name, values in report['engines'].items() if name != 'lammps'}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('ensemble', choices=['nvt', 'npt'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--shake', action='store_true')
    parser.add_argument('--sequential', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    command = nvt if args.ensemble == 'nvt' else npt
    steps = args.steps or (500000 if args.ensemble == 'nvt' else 400000)
    report = command(args.output, steps, args.threads, workers=1 if args.sequential else None,
                     gpu=args.gpu, shake=args.shake)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
