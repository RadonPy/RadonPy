"""Opt-in LAMMPS/GROMACS campaign of the EQ21step preset (stage 3 gate).

Not collected by pytest. Checks on replicas (amorphous cells built from their
own random seeds):

1. Properties (sample): every engine runs the sampling stage of EQ21step
   (NPT, 1 atm, 5 ns) from the same equilibrated structure with the same
   initial velocities, and the 14 properties are computed from each engine's
   own output. This checks that each engine's dynamics and output conversion
   give the same properties. The seed-to-seed spread of one engine (a second
   velocity seed) is the noise floor: an engine difference is consistent when
   its replica mean is within three standard errors implied by that spread.
   The first SKIP_ROWS thermo rows are dropped for every engine: LAMMPS reports
   the temperature before SHAKE removes the bond components of the seeded
   velocities at step 0. A higher --temp (a melt) checks diffusion.
2. Equilibration (run): every arm runs the full protocol (packing, 21-step
   compression/decompression, sampling) from the same initial cell. This checks
   that independently equilibrated states give equivalent properties; replica
   means are compared with their combined standard errors.
3. Dynamics (dynamics): NVE energy conservation and NVT potential energies from
   the same start; --integrator md-vv runs GROMACS with velocity Verlet, the
   LAMMPS integrator, instead of leapfrog.
4. Additional (additional): the AutoMD retry path, Additional equilibration of
   a copy of a finished run, then analysis, check_eq and del_dump.
5. Tg (tg): TGMD with the AutoMD settings (cooling_rate 8e3) from the source
   arm's final equilibrated structure, in every engine.
6. Dielectric response (efdp): ef_dp with a given AC field (10 periods, as
   AutoMD) from the final equilibrated structure, in every engine.
7. Deformation (elong): uniaxial strain-rate deformation at 1 atm laterally from
   the final equilibrated structure, in every engine.
8. Solubility parameter (sp): the sp energy decomposition of one sampling run's
   frames (5 ns, 5001 frames) by every engine, on identical coordinates.

Run each simulation in its own process (LAMMPS changes the working directory):

  python -m tests.eq_campaign build --output DIR [--polymer pmma] [--replicas 3] [--charges resp]
  python -m tests.eq_campaign sample --output DIR --replica 0 --arm B --seed 1 [--source C] [--temp 300] [--gpu]
  python -m tests.eq_campaign run --output DIR --replica 0 --arm C [--gpu] [--threads 2]
  python -m tests.eq_campaign dynamics --output DIR --replica 0 --arm C --ensemble nve [--integrator md-vv]
  python -m tests.eq_campaign additional --output DIR --replica 0 --arm C
  python -m tests.eq_campaign tg --output DIR --replica 0 --arm B [--tg-max-temp 1100]
  python -m tests.eq_campaign efdp --output DIR --replica 0 --arm B [--freq 1e10]
  python -m tests.eq_campaign elong --output DIR --replica 0 --arm B [--strain 0.5]
  python -m tests.eq_campaign sp --output DIR --replica 0 --arm B [--frames sample_C_s1]
  python -m tests.eq_campaign report --output DIR
  python -m tests.eq_campaign plot --output DIR

--cap STEPS shortens every MD stage (and the analysis windows) to check that a
protocol runs before its long version is launched; use a separate --output.

Arms: A = LAMMPS defaults, B = LAMMPS portable, C = GROMACS portable, D = GROMACS
defaults (C-rescale barostat for C and D). B and C have the same interactions and
are compared; D approximates A (a potential switch for the CHARMM switch) and is
compared with it; A differs from B in its interactions and is reported for
information.
`sample` and `dynamics` start from the eq2 (21-step) structure of the --source arm.
--source-output DIR takes that structure from another campaign directory.

--accuracy loose keeps the portable interactions with production accuracy, to
check that the engines also agree at a speed usable in practice: PPPM 1e-6 in
LAMMPS; PME ewald-rtol 1e-6, order 4 and 0.10 nm spacing in GROMACS with a
buffered pair list (nstlist 10, tolerance 0.005 kJ/(mol ps); minimization and
stages under 1 fs keep the exact pair list), energies every
gcd(nstenergy, 100) steps and barostat coupling every 10 steps. The thermostat
stays coupled every step. Use a separate --output.

The cells use Gasteiger charges and shorter chains than AutoMD (RESP charges,
10 chains of 1000 atoms) to keep the campaign affordable; the comparison is
between engines for the same model, not against experiment. --charges resp
uses RESP charges (HF/6-31G(d) on a force-field geometry) as in AutoMD.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from radonpy.core import calc, poly, utils
from radonpy.ff.gaff2_mod import GAFF2_mod
from radonpy.sim import gromacs
from radonpy.sim.preset import ef_dp, elong, eq, sp, tg
from tests.ensemble_validation import zero_momentum

POLYMERS = {'pe': '*CC*', 'ps': '*CC(*)c1ccccc1', 'pmma': '*CC(*)(C)C(=O)OC'}
ARMS = {'A': ('lammps', None), 'B': ('lammps', 'portable'), 'C': ('gromacs', 'portable'), 'D': ('gromacs', None)}
TEMP, PRESS = 300.0, 1.0
SKIP_ROWS = 10
PROPERTIES = ['density', 'Rg', 'self-diffusion', 'Cp', 'Cv', 'compressibility',
              'isentropic_compressibility', 'bulk_modulus', 'isentropic_bulk_modulus',
              'volume_expansion', 'linear_expansion', 'r2', 'static_dielectric_const',
              'nematic_order_parameter']
ENERGIES = ['Temp', 'TotEng', 'KinEng', 'E_bond', 'E_angle', 'E_dihed', 'E_impro', 'E_vdwl', 'E_coul', 'E_long']
ACCURACY = 'portable'
SOURCE_OUTPUT = None


def _loosen():
    """Production accuracy for both engines (--accuracy loose); interactions are unchanged."""
    import math
    import re
    from radonpy.sim import lammps, profiles

    def apply_profile(md):
        md = profiles.apply_profile(md)
        if md.kspace_style == 'pppm':
            md.kspace_style_accuracy = '1e-6'
        return md
    lammps.apply_profile = gromacs.apply_profile = apply_profile

    mdp = gromacs.Gromacs._mdp

    def loose_mdp(*args, **kwargs):
        text = mdp(*args, **kwargs)
        nstenergy = re.search(r'^nstenergy = (\d+)$', text, flags=re.M)
        settings = {'ewald-rtol': '1e-06', 'fourierspacing': '0.1', 'pme-order': '4', 'nstlist': '10',
                    'verlet-buffer-tolerance': '0.005', 'nstpcouple': '10',
                    'nstcalcenergy': str(math.gcd(int(nstenergy.group(1)), 100) if nstenergy else 1)}
        dt = re.search(r'^dt = (\S+)$', text, flags=re.M)
        if not dt or float(dt.group(1)) < 0.001:
            # Minimization and sub-fs packing stages start far from equilibrium, where
            # the buffer estimate does not hold (a PMMA cell minimized to 2e9 kJ/(mol nm)
            # forces and exploded): keep the exact pair list.
            del settings['nstlist'], settings['verlet-buffer-tolerance']
        for key, value in settings.items():
            text = re.sub(r'^%s = .*$' % re.escape(key), '%s = %s' % (key, value), text, flags=re.M)
        return text
    gromacs.Gromacs._mdp = staticmethod(loose_mdp)


def _resp(mol, work, threads):
    from radonpy.sim import qm
    assert qm.assign_charges(mol, charge='RESP', opt=False, work_dir=str(work), tmp_dir=str(work),
                             omp=threads, memory=4000, log_name='resp')
    return mol


def build(output, polymer, replicas, natom, nchain, seed=4100, charges='gasteiger', threads=1):
    """One amorphous cell per replica at the AutoMD initial density."""
    meta = {'polymer': polymer, 'smiles': POLYMERS[polymer], 'force_field': 'gaff2_mod',
            'charges': charges, 'chain_atoms': natom, 'chains': nchain, 'seed': seed, 'replicas': {}}
    for replica in range(replicas):
        path = output/f'r{replica}'/'initial.json'
        if not path.exists():
            np.random.seed(seed+replica)
            monomer = utils.mol_from_smiles(POLYMERS[polymer])
            terminal = utils.mol_from_smiles('*C')
            if charges == 'resp':
                for name, mol in (('monomer', monomer), ('terminal', terminal)):
                    (output/'resp'/name).mkdir(parents=True, exist_ok=True)
                    _resp(mol, output/'resp'/name, threads)
            n = poly.calc_n_from_num_atoms(monomer, natom, terminal1=terminal)
            # AutoMD random-walk settings; geometry shaking uses one LAMMPS rank.
            work = output/f'r{replica}'/'build'
            work.mkdir(parents=True, exist_ok=True)
            chain = poly.polymerize_rw(monomer, n, tacticity='atactic', ter1=terminal, retry=100, rollback=5,
                                       rollback_shaking=True, retry_step=200, retry_opt_step=5,
                                       work_dir=str(work), omp=0, mpi=1, gpu=0)
            # Without a charge model ff_assign keeps the RESP charges, as in AutoMD.
            assert GAFF2_mod().ff_assign(chain, charge='gasteiger' if charges == 'gasteiger' else None)
            cell = poly.amorphous_cell(chain, nchain, density=0.05)
            path.parent.mkdir(parents=True, exist_ok=True)
            gromacs._write_molecule(cell, path)
        cell = gromacs._read_molecule(path)
        meta['replicas'][replica] = {'atoms': cell.GetNumAtoms(),
                                     'net_charge': sum(a.GetDoubleProp('AtomicCharge') for a in cell.GetAtoms())}
    (output/'campaign.json').write_text(json.dumps(meta, indent=2)+'\n')
    return meta


def _preset(mol, work, arm, cls=eq.EQ21step, **kwargs):
    engine, profile = ARMS[arm]
    path = os.environ.get('LAMMPS_EXEC' if engine == 'lammps' else 'GROMACS_EXEC')
    return cls(mol, work_dir=str(work), solver_path=path, solver=engine, interaction_profile=profile, **kwargs)


def _resources(arm, threads, gpu):
    if ARMS[arm][0] == 'lammps':
        # KOKKOS runs one rank per GPU.
        return {'omp': 0, 'mpi': 1 if gpu else threads, 'gpu': int(gpu)}
    return {'omp': threads, 'mpi': 0, 'gpu': int(gpu), 'barostat': 'C-rescale'}


def _cap(preset, steps, stages=('packing', 'eq21step', 'sampling')):
    """Shorten every MD stage of the preset's protocol to at most steps."""
    for name in stages:
        def capped(*args, _builder=getattr(preset, name), **kwargs):
            md = _builder(*args, **kwargs)
            for wf in md.wf:
                if wf.type == 'md':
                    wf.step = min(wf.step, steps)
                else:
                    wf.maxiter = min(wf.maxiter, 1000)
            return md
        setattr(preset, name, capped)


def _window(steps, skip=0):
    """Analysis window of a capped run: 40% of its thermo rows (2000 of 5000 in the full run)."""
    if not steps:
        return {}
    width = max(5, int(0.4*(steps//1000+1-skip)))
    return {'width': width, 'init': width, 'f_width': width}


def _started(mol, seed, temp):
    np.random.seed(seed)
    calc.set_velocity(mol, temp)
    zero_momentum(mol)
    return mol


def _means(df):
    """Mean thermodynamic quantities; Coulomb as one sum since PPPM and PME split it differently."""
    means = {k: float(df[k].mean()) for k in ENERGIES+['Volume', 'Density'] if k in df}
    means['PotEng'] = means['TotEng']-means['KinEng']
    if 'E_coul' in means and 'E_long' in means:
        means['E_coul+E_long'] = means['E_coul']+means['E_long']
    return means


def _analyze(preset, work, done, record, temp=TEMP, skip=0, **options):
    analysis = preset.analyze()
    if skip:
        analysis.dfs[-1] = analysis.dfs[-1].iloc[skip:].reset_index(drop=True)
    prop = analysis.get_all_prop(temp=temp, press=PRESS, save=True, **options)
    result = {**record, 'accuracy': ACCURACY, **json.loads(done.read_text()), 'check_eq': bool(analysis.check_eq()),
              'properties': {k: float(v) for k, v in prop.items() if np.isscalar(v) and np.isreal(v)},
              # Over the rows the properties average (the last width rows).
              'thermo': _means(analysis.dfs[-1].iloc[-options.get('width', 2000):])}
    (work/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def _workdir(work):
    """Create work for a new run; a directory left by an unfinished run is not reused."""
    done = work/'simulated.json'
    if not done.exists():
        work.mkdir(exist_ok=False)
    return done


def _simulate(done, solver_run):
    if not done.exists():
        start = time.time()
        solver_run()
        done.write_text(json.dumps({'elapsed_s': time.time()-start})+'\n')


def run(output, replica, arm, threads, gpu, cap=None):
    """Full EQ21step of one arm (unless already done), then analysis."""
    cell = gromacs._read_molecule(output/f'r{replica}'/'initial.json')
    work = output/f'r{replica}'/arm
    done = work/'simulated.json'
    if not done.exists():
        work.mkdir(exist_ok=False)
    preset = _preset(cell, work, arm)
    if cap:
        _cap(preset, cap)
    if not done.exists():
        start = time.time()
        preset.exec(temp=TEMP, press=PRESS, **_resources(arm, threads, gpu))
        done.write_text(json.dumps({'elapsed_s': time.time()-start, 'gpu': bool(gpu), 'threads': threads})+'\n')
    engine, profile = ARMS[arm]
    return _analyze(preset, work, done, {'arm': arm, 'engine': engine, 'interaction_profile': profile, 'cap': cap},
                    **_window(cap))


def _source(output, replica, source):
    return utils.pickle_load(str((SOURCE_OUTPUT or output)/f'r{replica}'/source/'analyze'/'eq2_last.pickle'))


def sample(output, replica, arm, seed, threads, gpu, source='C', temp=TEMP, cap=None):
    """Sampling stage only, from the source arm's eq2 structure with seeded velocities."""
    suffix = '' if temp == TEMP else '_T%g' % temp
    work = output/f'r{replica}'/f'sample_{arm}_s{seed}{suffix}'
    done = _workdir(work)
    preset = _preset(_started(_source(output, replica, source), seed, temp), work, arm)
    resources = _resources(arm, threads, gpu)
    barostat = {'barostat': resources.pop('barostat')} if 'barostat' in resources else {}

    def simulate():
        utils.MolToPDBFile(preset.mol, str(work/preset.pdb_file))
        solver = preset.get_solver()
        # In the full protocol the preceding stage writes this data file.
        solver.make_dat(preset.mol, file_name=preset.dat_file)
        solver.run(preset.sampling(temp=temp, press=PRESS, step=cap or 5000000, **barostat), mol=preset.mol,
                   input_file=preset.in_file, last_str=preset.last_str, last_data=preset.last_data, **resources)
    _simulate(done, simulate)
    engine, profile = ARMS[arm]
    record = {'arm': arm, 'engine': engine, 'interaction_profile': profile, 'seed': seed, 'source': source,
              'temp': temp, 'cap': cap, 'gpu': bool(gpu), 'threads': threads}
    return _analyze(preset, work, done, record, temp=temp, skip=SKIP_ROWS, **_window(cap, SKIP_ROWS))


def dynamics(output, replica, arm, seed, threads, gpu, ensemble='nve', integrator=None, source='C', steps=100000):
    """NVE or NVT at TEMP from the source eq2 structure; energy drift and mean energies."""
    suffix = '_vv' if integrator == 'md-vv' else ''
    work = output/f'r{replica}'/f'{ensemble}_{arm}_s{seed}{suffix}'
    done = _workdir(work)
    preset = _preset(_started(_source(output, replica, source), seed, TEMP), work, arm)
    resources = _resources(arm, threads, gpu)
    resources.pop('barostat', None)
    # The sampling stage's settings and outputs, with its NPT step replaced.
    md = preset.sampling(temp=TEMP, press=PRESS, step=steps)
    md.wf = []
    md.add_md(ensemble, steps, time_step=1.0, shake=True, t_start=TEMP, t_stop=TEMP)

    def simulate():
        solver = preset.get_solver()
        solver.make_dat(preset.mol, file_name=preset.dat_file)
        mdp = gromacs.Gromacs._mdp
        if integrator == 'md-vv':
            assert ARMS[arm][0] == 'gromacs'
            gromacs.Gromacs._mdp = staticmethod(lambda *a, **k: mdp(*a, **k).replace('integrator = md\n', 'integrator = md-vv\n'))
        try:
            solver.run(md, mol=preset.mol, input_file=preset.in_file, last_str=preset.last_str,
                       last_data=preset.last_data, **resources)
        finally:
            gromacs.Gromacs._mdp = staticmethod(mdp)
    _simulate(done, simulate)
    df = preset.analyze().dfs[-1].iloc[SKIP_ROWS:]
    t, energy = df['Time'].to_numpy()/1000, df['TotEng'].to_numpy()  # ns, kcal/mol
    atoms = preset.mol.GetNumAtoms()
    drift = float(np.polyfit(t, energy, 1)[0])
    means = _means(df)
    result = {'arm': arm, 'engine': ARMS[arm][0], 'ensemble': ensemble, 'integrator': integrator or 'default',
              'seed': seed, 'steps': steps, **json.loads(done.read_text()), 'atoms': atoms,
              'drift_kcal_mol_ns': drift, 'drift_kT_per_atom_ns': drift/(atoms*TEMP*1.987204e-3),
              'totene_sd': float(energy.std()), 'means': means}
    (work/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def additional(output, replica, arm, threads, gpu, steps=5000000, cap=None):
    """AutoMD retry path on a copy of a finished run: Additional, analysis, check_eq, del_dump."""
    source = output/f'r{replica}'/arm
    work = output/f'r{replica}'/f'{arm}_additional'
    if not work.exists():
        shutil.copytree(source, work)
    mol = utils.pickle_load(eq.get_final_pickle(str(work/'analyze')))
    preset = _preset(mol, work, arm, cls=eq.Additional)
    resources = _resources(arm, threads, gpu)
    start = time.time()
    preset.exec(temp=TEMP, press=PRESS, eq_step=(cap or steps)/1e6, **resources)
    elapsed = time.time()-start
    analysis = preset.analyze()
    options = _window(cap)
    prop = analysis.get_all_prop(temp=TEMP, press=PRESS, save=True, **options)
    before = sorted(p.name for p in work.iterdir())
    eq.del_dump(str(work))
    result = {'arm': arm, 'idx': preset.idx, 'final_idx': eq.get_final_idx(str(work/'analyze')),
              'elapsed_s': elapsed, 'check_eq': bool(analysis.check_eq()),
              'deleted': sorted(set(before)-set(p.name for p in work.iterdir())),
              'restored_atoms': eq.restore(str(work/'analyze')).GetNumAtoms(),
              'properties': {k: float(v) for k, v in prop.items() if np.isscalar(v) and np.isreal(v)}}
    (work/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def _fixed_search(preset, max_temp):
    """Replace the density search's stopping rule: heat 300 -> 800 (-> max_temp) K in every engine.

    The search otherwise stops where its fitted density falls below 0.5, which the engines reach
    at different temperatures (1100-1150 K in LAMMPS, 1300-1450 K in GROMACS, where the cell
    cavitated and did not recompress on cooling).
    """
    analyzer = preset.analyzer()

    class Fixed(analyzer):
        def density_checker(self, index):
            try:
                data = super().density_checker(index)
            except (TypeError, ValueError, IndexError):
                data = {'tg_init_density': float('nan')}
            # The first run heats to 800 K; a higher max_temp takes a second one.
            data['tg_init_density_check'] = index >= 1 or max_temp <= 800
            data['tg_next_temp'] = max_temp
            return data

        def step_wise(self):
            # The fit selects holds below 0.45 g/cm3, which a cell heated to max_temp need
            # not reach; the curve is compared directly, so a failed fit only means no Tg.
            try:
                return super().step_wise()
            except (TypeError, ValueError, IndexError):
                return None
    preset.analyzer = lambda: Fixed


def tg_run(output, replica, arm, threads, gpu, source='C', cap=None, cooling_rate=8e3, max_temp=None, barostat=None):
    """TGMD from the source arm's final equilibrated structure (the same structure and velocities for every arm).

    barostat replaces the GROMACS barostat (C-rescale cavitated PMMA at 1100 K, where LAMMPS MTK kept 0.5 g/cm3).
    """
    name = f'tg_{arm}' if max_temp is None else f'tg_{arm}_T{max_temp:g}'
    if barostat is not None:
        assert ARMS[arm][0] == 'gromacs', 'only the GROMACS barostat is selectable'
        name += '_'+{'Parrinello-Rahman': 'PR', 'C-rescale': 'CR'}[barostat]
    work = output/f'r{replica}'/name
    done = _workdir(work)
    mol = utils.pickle_load(str(output/f'r{replica}'/source/'analyze'/'eq3_last.pickle'))
    preset = _preset(mol, work, arm, cls=tg.TGMD)
    if max_temp is not None:
        _fixed_search(preset, max_temp)
    if cap:
        # The density search must heat until the density falls, so it keeps its length.
        _cap(preset, cap, stages=('equilibration', 'step_wise'))

    def simulate():
        try:
            resources = _resources(arm, threads, gpu)
            if barostat is not None:
                resources['barostat'] = barostat
            preset.exec(temp=TEMP, cooling_rate=cooling_rate, **resources)
        except utils.RadonPyError as error:
            # The runs finished but the density curve gave no Tg.
            if 'Tg could not be calculated' not in str(error):
                raise
        (work/'tg_data.json').write_text(json.dumps(preset.data, indent=2, default=float)+'\n')
    _simulate(done, simulate)
    data = json.loads((work/'tg_data.json').read_text())
    analysis = preset.analyzer()(log_file=os.path.join(preset.work_dir_tg, preset.log_file), save_dir=preset.save_dir)
    # Mean temperature and density of every held temperature, as the Tg fit uses them.
    curve = [[float(f['Temp'].mean()), float(f['Density'].mean())] for f in analysis.read_log(analysis.log_file)[0::2]]
    engine, profile = ARMS[arm]
    result = {'arm': arm, 'engine': engine, 'interaction_profile': profile, 'source': source, 'cap': cap,
              'cooling_rate': cooling_rate, **json.loads(done.read_text()), 'tg': data.get('tg'), 'data': data,
              'curve': curve}
    (work/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def _final(output, replica, source):
    return utils.pickle_load(str(output/f'r{replica}'/source/'analyze'/'eq3_last.pickle'))


def efdp_run(output, replica, arm, threads, gpu, source='C', freq=1e10, evalue=0.05, axis='z', waves=10):
    """ef_dp 'dp' at a given field from the source arm's final equilibrated structure (identical for every arm)."""
    work = output/f'r{replica}'/f'efdp_{arm}_{freq/1e9:g}GHz'
    done = _workdir(work)
    preset = _preset(_final(output, replica, source), work, arm, cls=ef_dp.further_Additional,
                     axis=axis, evalue=evalue, freq=freq, process='dp')
    steps = int(round(waves/freq*1e15))  # 1 fs steps
    # AutoMD: 1000 frames for >= 10 GHz, else 5000 per run of 10 periods.
    every = max(1, steps//(1000 if freq >= 1e10 else 5000))
    _simulate(done, lambda: preset.exec_dp(temp=TEMP, press=PRESS, eq_step=steps/1e6, thermo_freq=every,
                                           dump_freq=every, ensemble_s='npt', **_resources(arm, threads, gpu)))
    fit = preset.dipole_anal(printout=True, save=True).iloc[0].to_dict()
    engine, profile = ARMS[arm]
    result = {'arm': arm, 'engine': engine, 'interaction_profile': profile, 'source': source, 'freq': freq,
              'evalue': evalue, 'axis': axis, 'steps': steps, **json.loads(done.read_text()),
              'fit': {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in fit.items()}}
    (work/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def elong_run(output, replica, arm, threads, gpu, source='C', strain=0.5, rate=5e-6, axis='x', relax_step=500000):
    """Uniaxial deformation at TEMP with 1 atm laterally, from the source arm's final equilibrated structure."""
    work = output/f'r{replica}'/f'elong_{arm}_{int(strain*100)}percent'
    done = _workdir(work)
    preset = _preset(_final(output, replica, source), work, arm, cls=elong.Deformation, axis=axis, deform_strain=strain)
    lateral = {'p%s_%s' % (a, k): v for a in 'xyz' if a != axis for k, v in (('start', PRESS), ('stop', PRESS), ('dump', 1000.0))}
    resources = _resources(arm, threads, gpu)
    if 'barostat' in resources:
        # GROMACS C-rescale has no anisotropic coupling.
        resources['barostat'] = 'Parrinello-Rahman'
    _simulate(done, lambda: preset.exec_deform(init_temp=TEMP, max_temp=TEMP, relax_step=relax_step, deform_rate=rate,
                                               deform_step=int(round(strain/rate)), **lateral, **resources))
    engine, profile = ARMS[arm]
    analysis = elong.Deformation_Analysis(work_dir=str(work), axis=axis, deform_strain=strain, solver=engine)
    _, deform = analysis.log_extraction()
    analysis.density_plot()
    analysis.nemorder_plot()
    curve = deform[['v_EngStrain', 'Density', 'v_GamNsurf']].astype(float).to_numpy().tolist()
    result = {'arm': arm, 'engine': engine, 'interaction_profile': profile, 'source': source, 'strain': strain,
              'rate': rate, 'axis': axis, **json.loads(done.read_text()), 'curve': curve,
              'nematic_order': pd.read_csv(work/'analyze'/'nematic_order.csv')['Nematic order'].tolist()}
    (work/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def sp_run(output, replica, arm, threads, frames='sample_C_s1'):
    """The sp decomposition by one engine of the frames of a sampling run (the same coordinates for every engine)."""
    from mdtraj.formats import XTCTrajectoryFile
    source = output/f'r{replica}'/frames
    work = output/f'r{replica}'/f'sp_{arm}_{frames}'
    done = _workdir(work)
    engine, profile = ARMS[arm]
    preset = _preset(_source(output, replica, 'C'), work, arm, cls=sp.Rerun, idx=3)

    def simulate():
        if engine == 'gromacs':
            (work/'eq3.xtc').symlink_to(source/'eq3.xtc')
        else:
            # The XTC coordinates as a LAMMPS dump (wrapped, with image flags).
            with XTCTrajectoryFile(str(source/'eq3.xtc')) as xtc:
                xyz, _, _, box = xtc.read()
            xyz, lengths = xyz.astype(float)*10, box[:, [0, 1, 2], [0, 1, 2]].astype(float)*10
            with open(work/'eq3.dump', 'w') as dump:
                for k, (x, b) in enumerate(zip(xyz, lengths)):
                    image = np.floor(x/b).astype(int)
                    dump.write('ITEM: TIMESTEP\n%d\nITEM: NUMBER OF ATOMS\n%d\nITEM: BOX BOUNDS pp pp pp\n' % (k*1000, len(x)))
                    dump.write(''.join('0 %.6f\n' % v for v in b))
                    dump.write('ITEM: ATOMS id x y z ix iy iz vx vy vz\n')
                    dump.write(''.join('%d %.6f %.6f %.6f %d %d %d 0 0 0\n' % (i+1, *q, *m)
                                       for i, (q, m) in enumerate(zip(x-image*b, image))))
        # Energies only: CPU numerics for LAMMPS (pe/mol/tally) and GROMACS.
        preset.exec(omp=threads, mpi=0, gpu=0)
    _simulate(done, simulate)
    data = preset.analyze(save_dir=preset.save_dir).Solubility_Parameter()
    result = {'arm': arm, 'engine': engine, 'interaction_profile': profile, 'frames': frames,
              **json.loads(done.read_text()), 'sp': {k: float(v) for k, v in data.items()}}
    (work/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def summarize(values):
    values = np.asarray(values, dtype=float)
    se = float(values.std(ddof=1)/np.sqrt(len(values))) if len(values) > 1 else None
    return {'mean': float(values.mean()), 'se': se, 'n': len(values), 'values': values.tolist()}


def sampling_report(output, reference='B', candidate='C', seeds=(1, 2), temp=TEMP):
    """Engine differences from identical starts against the candidate's seed-to-seed spread."""
    runs = {}
    for path in output.glob('r*/sample_*/result.json'):
        result = json.loads(path.read_text())
        if result.get('temp', TEMP) == temp:
            runs[path.parent.parent.name, result['arm'], result['seed']] = result['properties']
    replicas = sorted(r for r, arm, seed in runs if arm == reference and seed == seeds[0]
                      and (r, candidate, seeds[0]) in runs)
    noise = [r for r in replicas if (r, candidate, seeds[1]) in runs]
    table = {}
    for name in PROPERTIES:
        ref = [runs[r, reference, seeds[0]][name] for r in replicas]
        diff = [runs[r, candidate, seeds[0]][name]-runs[r, reference, seeds[0]][name] for r in replicas]
        spread = [runs[r, candidate, seeds[1]][name]-runs[r, candidate, seeds[0]][name] for r in noise]
        entry = {'reference': ref, 'differences': diff, 'seed_differences': spread}
        if diff:
            mean = float(np.mean(diff))
            entry.update(mean_difference=mean, relative=mean/np.mean(ref) if np.mean(ref) else None)
        if diff and spread:
            # A seed difference is the difference of two runs: its variance is twice a run's.
            sd = float(np.sqrt(np.mean(np.square(spread))/2))
            se = sd*np.sqrt(2/len(diff))
            entry.update(seed_sd=sd, se=se, z=mean/se if se else None, consistent=bool(abs(mean) <= 3*se))
        else:
            entry['consistent'] = 'inconclusive'
        table[name] = entry
    return {'reference': reference, 'candidate': candidate, 'seeds': list(seeds), 'temp': temp,
            'replicas': replicas, 'noise_replicas': noise, 'properties': table}


def report(output):
    results = {}
    for path in sorted(output.glob('r*/[A-D]/result.json')):
        result = json.loads(path.read_text())
        results.setdefault(result['arm'], {})[path.parent.parent.name] = result
    arms = {arm: {} for arm in results}
    for arm, replicas in results.items():
        names = set.intersection(*(set(r['properties']) for r in replicas.values()))
        arms[arm] = {name: summarize([r['properties'][name] for r in replicas.values()]) for name in sorted(names)}
    comparisons = {}
    for pair in (('B', 'C'), ('B', 'A'), ('A', 'D')):
        if not all(arm in arms for arm in pair):
            continue
        a, b = (arms[arm] for arm in pair)
        table = {}
        for name in sorted(set(a) & set(b)):
            diff = b[name]['mean']-a[name]['mean']
            if a[name]['se'] is None or b[name]['se'] is None:
                table[name] = {'difference': diff, 'se': None, 'z': None, 'consistent': 'inconclusive'}
                continue
            se = float(np.hypot(a[name]['se'], b[name]['se']))
            table[name] = {'difference': diff, 'relative': diff/a[name]['mean'] if a[name]['mean'] else None,
                           'se': se, 'z': diff/se if se else None, 'consistent': bool(abs(diff) <= 3*se)}
        comparisons['%s_minus_%s' % (pair[1], pair[0])] = table
    meta = json.loads((output/'campaign.json').read_text())
    temps = sorted({json.loads(p.read_text()).get('temp', TEMP) for p in output.glob('r*/sample_*/result.json')})
    dynamics_runs = {f'{p.parent.parent.name}/{p.parent.name}': json.loads(p.read_text())
                     for p in sorted(output.glob('r*/n[vV][eEtT]_*/result.json'))}
    return {'campaign': meta, 'sampling': {str(t): sampling_report(output, temp=t) for t in temps},
            'arms': arms, 'comparisons': comparisons, 'dynamics': dynamics_runs,
            'runs': {arm: {k: {'elapsed_s': r['elapsed_s'], 'check_eq': r['check_eq'], 'gpu': r['gpu']}
                           for k, r in replicas.items()} for arm, replicas in results.items()}}


PANELS = [  # (title, result section, [(key, label)], axis limit in %)
    ('Structure', 'properties', [('density', 'Density'), ('Rg', 'Rg'), ('r2', 'r2 (end-to-end)')], 2),
    ('Fluctuation / transport / order', 'properties', [
        ('Cp', 'Cp'), ('Cv', 'Cv'), ('compressibility', 'Isothermal compressibility'),
        ('isentropic_compressibility', 'Isentropic compressibility'), ('volume_expansion', 'Volume expansion'),
        ('static_dielectric_const', 'Dielectric (eps-1)'), ('self-diffusion', 'Self-diffusion'),
        ('nematic_order_parameter', 'Nematic order')], 100),
    ('Thermodynamic averages', 'thermo', [
        ('Temp', 'Temperature'), ('Density', 'Density'), ('PotEng', 'Potential energy'), ('E_bond', 'Bond'),
        ('E_angle', 'Angle'), ('E_dihed', 'Dihedral'), ('E_vdwl', 'vdW'), ('E_coul+E_long', 'Coulomb (real+kspace)')], 4),
]
COLORS = ['#2364aa', '#c4552a', '#3a8a4f', '#7b4ea3']


def _relative(value, reference, key):
    base = reference-1 if key == 'static_dielectric_const' else reference
    return (value-reference)/abs(base)*100 if base else np.nan


def _bars(path, title, xlabel, series):
    """series: [(label, color, filled, {key: relative %})]; one bar per series and quantity."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9, 12), gridspec_kw={'height_ratios': [3, 8, 8]})
    for ax, (name, section, keys, limit) in zip(axes, PANELS):
        y = np.arange(len(keys))[::-1]
        height = 0.8/len(series)
        for i, (label, color, filled, values) in enumerate(series):
            v = np.array([values[section].get(k, np.nan) for k, _ in keys])
            shown = np.clip(v, -limit, limit)
            offset = ((len(series)-1)/2-i)*height
            ax.barh(y+offset, shown, height*0.9, color=color if filled else 'white', edgecolor=color, lw=1.2,
                    label=label)
            for yy, value, bar in zip(y+offset, v, shown):
                if abs(value) > limit:
                    ax.text(bar*0.98, yy, '%+.0f%%' % value, va='center', ha='right' if value > 0 else 'left',
                            fontsize=7, bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.5})
        ax.axvline(0, color='black', lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([label for _, label in keys])
        ax.set_xlim(-limit, limit)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(name)
        ax.grid(axis='x', alpha=0.3)
    axes[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.22), fontsize=9, ncol=min(len(series), 3),
                   frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def _pairs(runs, candidate, reference):
    """Per replica: candidate relative to reference, both from the identical start."""
    series = []
    for i, replica in enumerate(sorted(r for r, arm in runs if arm == reference and (r, candidate) in runs)):
        a, b = runs[replica, candidate], runs[replica, reference]
        values = {section: {k: _relative(a[section][k], b[section][k], k) for k in b.get(section, {})
                            if k in a.get(section, {})} for section in ('properties', 'thermo')}
        series.append((replica.replace('r', 'replica '), COLORS[i % len(COLORS)], True, values))
    return series


def plot(output):
    """Bar charts of the engine differences; returns the PNG paths."""
    polymer = json.loads((output/'campaign.json').read_text())['polymer'].upper()
    paths = []
    samples = {}
    for path in output.glob('r*/sample_*_s1*/result.json'):
        result = json.loads(path.read_text())
        samples.setdefault(result.get('temp', TEMP), {})[path.parent.parent.name, result['arm']] = result
    for temp, runs in sorted(samples.items()):
        window = '%g-%g ps' % (SKIP_ROWS, (runs[next(iter(runs))].get('cap') or 5000000)/1000)
        for candidate, reference, name, label in (('B', 'C', 'sampling', 'LAMMPS vs GROMACS'),
                                                  ('A', 'B', 'legacy', 'LAMMPS legacy vs portable interactions')):
            series = _pairs(runs, candidate, reference)
            if series:
                paths.append(_bars(output/f'{name}_T{temp:g}.png',
                                   f'{polymer} sampling at {temp:g} K: {label}\n'
                                   f'identical structure and velocities, same window ({window})',
                                   f'({ARMS[candidate][0].upper()} {candidate} - {ARMS[reference][0].upper()} {reference})'
                                   f' / {reference} [%]', series))
    full = {}
    for path in output.glob('r*/[BC]/result.json'):
        result = json.loads(path.read_text())
        full.setdefault(result['arm'], {})[path.parent.parent.name] = result
    if 'B' in full and 'C' in full:
        mean = {section: {k: np.mean([r[section][k] for r in full['C'].values()]) for k in
                          next(iter(full['C'].values())).get(section, {})} for section in ('properties', 'thermo')}
        series = []
        for arm, filled in (('C', False), ('B', True)):
            for i, (replica, result) in enumerate(sorted(full[arm].items())):
                values = {section: {k: _relative(result[section][k], mean[section][k], k) for k in mean[section]
                                    if k in result.get(section, {})} for section in ('properties', 'thermo')}
                cap = f" (cap {result['cap']})" if result.get('cap') else ''
                series.append((f'{ARMS[arm][0].upper()} {replica}{cap}', COLORS[i % len(COLORS)], filled, values))
        paths.append(_bars(output/'equilibration.png',
                           f'{polymer} full EQ21step from the same initial cells\n'
                           'each run relative to the GROMACS replica mean (outlined: GROMACS, filled: LAMMPS)',
                           'Relative to the GROMACS mean [%]', series))
    tgs = {}
    for path in output.glob('r*/tg_[A-D]/result.json'):
        result = json.loads(path.read_text())
        tgs.setdefault(path.parent.parent.name, {})[result['arm']] = result
    if tgs:
        paths.append(_tg_figure(output/'tg.png', polymer, tgs))
    efdp = {}
    for path in output.glob('r*/efdp_[A-D]_*GHz/result.json'):
        result = json.loads(path.read_text())
        efdp.setdefault(result['freq'], {}).setdefault(path.parent.parent.name, {})[result['arm']] = result
    for freq, runs in sorted(efdp.items()):
        paths.append(_efdp_figure(output/f'efdp_{freq/1e9:g}GHz.png', polymer, freq, runs))
    deformations = {}
    for path in output.glob('r*/elong_[A-D]_*percent/result.json'):
        result = json.loads(path.read_text())
        deformations.setdefault(path.parent.parent.name, {})[result['arm']] = result
    if deformations:
        paths.append(_elong_figure(output/'elong.png', polymer, deformations))
    solubility = {}
    for path in output.glob('r*/sp_[A-D]_*/result.json'):
        result = json.loads(path.read_text())
        solubility.setdefault(path.parent.parent.name, {})[result['arm']] = result
    if solubility:
        paths.append(_sp_figure(output/'sp.png', polymer, solubility))
    return paths


def _engine_bars(ax, replicas, values, fmt='%.4g'):
    """Paired bars per replica (LAMMPS filled, GROMACS outlined) labelled with their values; the tick labels carry
    LAMMPS relative to GROMACS, which the bar heights cannot show when the engines agree closely."""
    width = 0.38
    for j, (arm, filled) in enumerate((('B', True), ('C', False))):
        color = '#2364aa' if arm == 'B' else '#c4552a'
        x = np.arange(len(replicas))+(j-0.5)*width
        ax.bar(x, values[arm], width*0.9, color=color if filled else 'white', edgecolor=color, lw=1.2,
               label=ARMS[arm][0].upper())
        for xx, v in zip(x, values[arm]):
            if np.isfinite(v):
                # Vertical, so that the labels of neighbouring bars do not overlap.
                ax.annotate(fmt % v, (xx, v), xytext=(0, 2 if v >= 0 else -2), textcoords='offset points',
                            ha='center', va='bottom' if v >= 0 else 'top', fontsize=7, rotation=90)
    ticks = []
    for replica, b, c in zip(replicas, values['B'], values['C']):
        ticks.append(replica if not (np.isfinite(b) and np.isfinite(c) and c) else
                     '%s\n%+.2g%%' % (replica, (b-c)/abs(c)*100))
    ax.set_xticks(np.arange(len(replicas)))
    ax.set_xticklabels(ticks, fontsize=8)
    ax.set_xlabel('LAMMPS relative to GROMACS below each replica', fontsize=8)
    ax.set_xlim(-0.6, len(replicas)-0.4)
    ax.margins(y=0.25)
    ax.grid(axis='y', alpha=0.3)


def _engine_legend(fig, ax):
    """The engine legend of _engine_bars in the top left corner of the figure, clear of the bars."""
    fig.legend(*ax.get_legend_handles_labels(), loc='upper left', ncol=2, fontsize=9, frameon=False)


def _sp_figure(path, polymer, runs):
    """Solubility parameters and cohesive energy density of every engine, per replica."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys = [('sp_total', 'SP total'), ('sp_vdw', 'SP vdW'), ('sp_ele', 'SP electrostatic'), ('sp_ced', 'CED')]
    replicas = sorted(runs)
    fig, axes = plt.subplots(1, len(keys), figsize=(14, 4))
    for ax, (key, label) in zip(axes, keys):
        _engine_bars(ax, replicas, {arm: [runs[r].get(arm, {}).get('sp', {}).get(key, np.nan) for r in replicas]
                                    for arm in 'BC'}, fmt='%.5g')
        ax.set_title(label + (' [J/cm$^3$]' if key == 'sp_ced' else ' [MPa$^{1/2}$]'), fontsize=10)
    _engine_legend(fig, axes[0])
    fig.suptitle(f'{polymer} solubility parameter from identical frames', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def _efdp_figure(path, polymer, freq, runs):
    """Real and imaginary permittivity of every engine, per replica."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    replicas = sorted(runs)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, label in zip(axes, ('real part of permittivity', 'imaginary part of permittivity'), ("eps'", "eps''")):
        _engine_bars(ax, replicas, {arm: [runs[r].get(arm, {}).get('fit', {}).get(key, np.nan) for r in replicas]
                                    for arm in 'BC'})
        ax.set_ylabel(label)
    _engine_legend(fig, axes[0])
    fig.suptitle(f'{polymer} dielectric response at {freq/1e9:g} GHz from the same equilibrated structure', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def _elong_figure(path, polymer, runs):
    """Density and normal stress difference against strain, every engine and replica."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, replica in enumerate(sorted(runs)):
        for arm, style in (('B', '-'), ('C', '--')):
            if arm in runs[replica]:
                strain, density, gamma = np.array(runs[replica][arm]['curve']).T
                label = f'{replica} {ARMS[arm][0].upper()}'
                axes[0].plot(strain, density, style, color=COLORS[i % len(COLORS)], lw=1.2, label=label)
                axes[1].plot(strain, gamma, style, color=COLORS[i % len(COLORS)], lw=1.0, label=label)
    axes[0].set_ylabel('Density [g/cm$^3$]')
    axes[1].set_ylabel('GamNsurf (LAMMPS elong variable)')
    for ax in axes:
        ax.set_xlabel('Engineering strain')
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle(f'{polymer} uniaxial deformation from the same equilibrated structure', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def _tg_figure(path, polymer, tgs):
    """Density-temperature curves and Tg of every engine, per replica."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (curves, bars) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [3, 2]})
    replicas = sorted(tgs)
    for i, replica in enumerate(replicas):
        for arm, style in (('B', '-'), ('C', '--')):
            if arm in tgs[replica]:
                t, d = np.array(tgs[replica][arm]['curve']).T
                curves.plot(t, d, style, color=COLORS[i % len(COLORS)], lw=1.2,
                            label=f'{replica} {ARMS[arm][0].upper()}')
    curves.set_xlabel('Temperature [K]')
    curves.set_ylabel('Density [g/cm$^3$]')
    curves.legend(fontsize=8, ncol=2)
    curves.grid(alpha=0.3)
    _engine_bars(bars, replicas, {arm: [tgs[r].get(arm, {}).get('tg') or np.nan for r in replicas] for arm in 'BC'},
                 fmt='%.0f')
    bars.set_ylabel('Tg [K]')
    _engine_legend(fig, bars)
    fig.suptitle(f'{polymer} Tg (step-wise cooling) from the same equilibrated structure', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['build', 'sample', 'run', 'dynamics', 'additional', 'tg', 'efdp', 'elong', 'sp', 'report', 'plot'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--polymer', choices=sorted(POLYMERS), default='pmma')
    parser.add_argument('--replicas', type=int, default=3)
    parser.add_argument('--chain-atoms', type=int, default=400)
    parser.add_argument('--chains', type=int, default=10)
    parser.add_argument('--charges', choices=['gasteiger', 'resp'], default='gasteiger')
    parser.add_argument('--replica', type=int)
    parser.add_argument('--arm', choices=sorted(ARMS))
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--source', choices=sorted(ARMS), default='C')
    parser.add_argument('--temp', type=float, default=TEMP)
    parser.add_argument('--ensemble', choices=['nve', 'nvt'], default='nve')
    parser.add_argument('--integrator', choices=['md-vv'])
    parser.add_argument('--steps', type=int, default=100000, help='dynamics length')
    parser.add_argument('--cap', type=int, help='shorten every MD stage to this many steps')
    parser.add_argument('--freq', type=float, default=1e10, help='efdp field frequency (Hz)')
    parser.add_argument('--strain', type=float, default=0.5, help='elong final engineering strain')
    parser.add_argument('--barostat', choices=['C-rescale', 'Parrinello-Rahman'], help='tg: GROMACS barostat')
    parser.add_argument('--tg-max-temp', type=float, help='tg: heat 300 -> 800 -> this temperature instead of the density search')
    parser.add_argument('--frames', default='sample_C_s1', help='sp: the sampling run whose frames are decomposed')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--accuracy', choices=['portable', 'loose'], default='portable')
    parser.add_argument('--source-output', type=Path, help='campaign directory of the --source structure')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    global ACCURACY, SOURCE_OUTPUT
    ACCURACY, SOURCE_OUTPUT = args.accuracy, args.source_output
    # One accuracy per campaign directory, so reports never mix them.
    marker = args.output/'accuracy.txt'
    if args.command not in ('report', 'plot'):
        if not marker.exists() and ACCURACY != 'portable':
            marker.write_text(ACCURACY+'\n')
        if (marker.read_text().strip() if marker.exists() else 'portable') != ACCURACY:
            parser.error('%s was run with a different --accuracy' % args.output)
    if ACCURACY == 'loose':
        _loosen()
    if args.command == 'build':
        result = build(args.output, args.polymer, args.replicas, args.chain_atoms, args.chains,
                       charges=args.charges, threads=args.threads)
    elif args.command == 'sample':
        result = sample(args.output, args.replica, args.arm, args.seed, args.threads, args.gpu, args.source,
                        args.temp, args.cap)
    elif args.command == 'run':
        result = run(args.output, args.replica, args.arm, args.threads, args.gpu, args.cap)
    elif args.command == 'dynamics':
        result = dynamics(args.output, args.replica, args.arm, args.seed, args.threads, args.gpu,
                          args.ensemble, args.integrator, args.source, args.steps)
    elif args.command == 'additional':
        result = additional(args.output, args.replica, args.arm, args.threads, args.gpu, cap=args.cap)
    elif args.command == 'tg':
        result = tg_run(args.output, args.replica, args.arm, args.threads, args.gpu, args.source, args.cap,
                        max_temp=args.tg_max_temp, barostat=args.barostat)
    elif args.command == 'efdp':
        result = efdp_run(args.output, args.replica, args.arm, args.threads, args.gpu, args.source, args.freq)
    elif args.command == 'elong':
        result = elong_run(args.output, args.replica, args.arm, args.threads, args.gpu, args.source, args.strain,
                           relax_step=args.cap or 500000)
    elif args.command == 'sp':
        result = sp_run(args.output, args.replica, args.arm, args.threads, args.frames)
    elif args.command == 'plot':
        result = plot(args.output)
    else:
        result = report(args.output)
        (args.output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
