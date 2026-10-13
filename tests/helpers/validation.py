"""Test-only numerical and statistical equivalence checks for MD engines.

Independent cells, not correlated trajectory frames, are the replicates in
``compare_properties``. A confidence interval overlapping zero is NOT evidence
of equivalence: the whole interval must lie within the prespecified margin.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy import stats

PROPERTY_MARGINS = {
    'density': .01, 'Rg': .05, 'r2': .05, 'Cp': .10, 'Cv': .10,
    'compressibility': .10, 'isentropic_compressibility': .10,
    'bulk_modulus': .10, 'isentropic_bulk_modulus': .10,
    'volume_expansion': .10, 'linear_expansion': .10,
    'static_dielectric_const': .10, 'self-diffusion': .20,
    'nematic_order_parameter': None,
}


def compare_static(reference, candidate, atoms):
    """Inputs are (kcal/mol energy, kcal/(mol angstrom) force array)."""
    if atoms <= 0:
        raise ValueError('atoms must be positive')
    re, rf = reference; ce, cf = candidate
    rf, cf = np.asarray(rf)*41.84, np.asarray(cf)*41.84
    if rf.shape != (atoms, 3) or cf.shape != rf.shape:
        raise ValueError('Force arrays must have shape (atoms, 3) in the same atom order')
    energy_error = abs(ce-re)*4.184/atoms
    energy_limit = 1e-4+1e-5*abs(re)*4.184/atoms
    force_error = np.abs(cf-rf)
    force_limit = 1e-2+1e-4*np.abs(rf)
    finite = np.isfinite([re, ce]).all() and np.isfinite(rf).all() and np.isfinite(cf).all()
    return {'status': 'pass' if finite and energy_error <= energy_limit and
            np.all(force_error <= force_limit) else 'fail',
            'energy_error_kj_mol_atom': float(energy_error) if finite else None,
            'energy_limit_kj_mol_atom': float(energy_limit) if finite else None,
            'force_max_error_kj_mol_nm': float(force_error.max()) if finite else None,
            'force_rms_error_kj_mol_nm': float(np.sqrt(np.mean(force_error**2))) if finite else None,
            'force_components_outside_tolerance': int(np.sum(force_error > force_limit)) if finite else None}


def block_means(values, block_size):
    """Discard an incomplete tail; require at least 20 blocks for diagnostics."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or block_size < 1 or not np.isfinite(values).all():
        raise ValueError('Expected a finite time series and positive block size')
    count = len(values)//block_size
    if count < 20:
        raise ValueError('Insufficient sampling: fewer than 20 blocks')
    return values[:count*block_size].reshape(count, block_size).mean(axis=1)


def autocorrelation_time(values):
    """Integrated autocorrelation time using the initial positive sequence."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 4 or not np.isfinite(x).all():
        raise ValueError('Expected at least four finite samples')
    x = x-x.mean()
    variance = np.dot(x, x)
    if variance == 0:
        return float('inf')  # A frozen simulation is not evidence of convergence.
    size = 1 << (2*len(x)-1).bit_length()
    fft = np.fft.rfft(x, n=size)
    ac = np.fft.irfft(fft*fft.conj(), n=size)[:len(x)]/variance
    total = 0.0
    for i in range(1, len(ac)-1, 2):
        pair = ac[i]+ac[i+1]
        if pair <= 0:
            break
        total += pair
    return max(1., 1+2*total)


def compare_properties(reference, candidate, absolute_margins=None, reference_scales=None):
    """Paired 95% Student intervals across >=5 independently prepared cells.

Each record contains replica_id, converged, and properties. Missing records,
non-finite values and unresolved near-zero reference scales are inconclusive.
Absolute margins and pilot reference scales must be fixed before production.
"""
    absolute_margins = {'nematic_order_parameter': .02, **(absolute_margins or {})}
    reference_scales = reference_scales or {}
    left = {r['replica_id']: r for r in reference}
    right = {r['replica_id']: r for r in candidate}
    if len(left) != len(reference) or len(right) != len(candidate):
        raise ValueError('Duplicate replica_id')
    if set(left) != set(right) or len(left) < 5:
        return {'status': 'inconclusive', 'reason': 'Need >=5 matching independent replicas', 'properties': {}}
    if not all(r.get('converged') is True for r in reference+candidate):
        return {'status': 'inconclusive', 'reason': 'At least one replica is unconverged', 'properties': {}}
    results = {}
    for name, relative in PROPERTY_MARGINS.items():
        a = np.array([left[k]['properties'].get(name, np.nan) for k in sorted(left)], dtype=float)
        b = np.array([right[k]['properties'].get(name, np.nan) for k in sorted(left)], dtype=float)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            results[name] = {'status': 'inconclusive', 'reason': 'Missing or non-finite property'}
            continue
        critical = stats.t.ppf(.975, len(a)-1)
        ref_half = critical*stats.sem(a)
        if name in absolute_margins:
            margin = float(absolute_margins[name])
        elif name in reference_scales:
            margin = relative*abs(float(reference_scales[name]))
        elif abs(a.mean()) <= ref_half or abs(a.mean()) < np.finfo(float).eps:
            results[name] = {'status': 'inconclusive', 'reason': 'Near-zero reference needs a prespecified absolute margin'}
            continue
        else:
            margin = relative*abs(a.mean())
        if not np.isfinite(margin) or margin <= 0:
            raise ValueError('Equivalence margin must be finite and positive')
        difference = b-a
        half = critical*stats.sem(difference)
        low, high = difference.mean()-half, difference.mean()+half
        status = 'pass' if low >= -margin and high <= margin else (
            'fail' if low > margin or high < -margin else 'inconclusive')
        results[name] = {'status': status, 'reference_mean': float(a.mean()),
                         'candidate_mean': float(b.mean()), 'difference_ci95': [float(low), float(high)],
                         'margin': margin, 'replicates': len(a)}
    states = {r['status'] for r in results.values()}
    return {'status': 'fail' if 'fail' in states else ('pass' if states == {'pass'} else 'inconclusive'),
            'properties': results}


def static_report(fixture, output, lammps_exec=None, gromacs_exec=None):
    from radonpy.core import utils
    from radonpy.sim import md
    from . import gromacs, lammps
    lammps_exec = lammps_exec or os.environ.get('LAMMPS_EXEC', 'lmp')
    gromacs_exec = gromacs_exec or os.environ.get('GROMACS_EXEC', 'gmx')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    molecule = utils.JSONToMol(str(fixture))
    results, details = {}, {}
    for label, engine, profile, executable in [('A', 'lammps', None, lammps_exec),
            ('B', 'lammps', 'portable', lammps_exec), ('C', 'gromacs', 'portable', gromacs_exec)]:
        work = output/label
        start = time.perf_counter()
        if label == 'A':
            work.mkdir(exist_ok=False)
            result = md.quick_energy(utils.deepcopy_mol(molecule), solver_path=executable,
                                     work_dir=str(work), omp=1, mpi=0)
        else:
            runner = lammps if engine == 'lammps' else gromacs
            result = runner.single_point(molecule, work, executable)
        elapsed = time.perf_counter()-start
        version = ((work/'version.txt').read_text() if engine == 'gromacs' else
                   lammps.lammps.LAMMPS(work_dir=str(work), solver_path=executable).get_version())
        results[label] = result
        np.savez(work/'result.npz', energy=result[0], force=result[1])
        details[label] = {'engine': engine, 'profile': profile, 'version': str(version),
                          'seconds_including_io': elapsed, 'energy_kj_mol': float(result[0])*4.184}
    report = {'schema': 1, 'fixture': str(Path(fixture).resolve()),
              'fixture_sha256': hashlib.sha256(Path(fixture).read_bytes()).hexdigest(),
              'python': sys.version, 'platform': platform.platform(), 'runs': details,
              'B_vs_C': compare_static(results['B'], results['C'], molecule.GetNumAtoms()),
              'A_vs_B': compare_static(results['A'], results['B'], molecule.GetNumAtoms()),
              'legacy_property_equivalence': 'not tested by single-point evaluation'}
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    static = commands.add_parser('static')
    static.add_argument('fixture', type=Path)
    static.add_argument('--output', type=Path, required=True)
    static.add_argument('--lammps-exec')
    static.add_argument('--gromacs-exec')
    props = commands.add_parser('properties')
    props.add_argument('reference', type=Path)
    props.add_argument('candidate', type=Path)
    props.add_argument('--margins', type=Path, help='Prespecified absolute_margins and reference_scales JSON')
    props.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'static':
        report = static_report(args.fixture, args.output, args.lammps_exec, args.gromacs_exec)
        status = report['B_vs_C']['status']
    else:
        margins = json.loads(args.margins.read_text()) if args.margins else {}
        report = compare_properties(json.loads(args.reference.read_text()),
                                    json.loads(args.candidate.read_text()), **margins)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        status = report['status']
    print(json.dumps(report, indent=2))
    return 0 if status == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
