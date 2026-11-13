#  Copyright (c) 2025. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# ******************************************************************************
# sim.profiles module
# ******************************************************************************
"""Interaction profiles shared by MD backends.

None (the default) is RadonPy's model in every engine: LAMMPS lj/charmm/coul/long
(LJ energy switch between cutoff_in and cutoff_out) with PPPM at
kspace_style_accuracy; GROMACS approximates it with a potential switch over the
same range (a different switching polynomial) and PME at the same tolerance.

'portable' is for development and validation: a model both engines evaluate
identically, with tight PPPM/PME accuracy, used by the parity tests and the
engine comparisons.
"""

PROFILES = (None, 'portable')


def apply_profile(md):
    """Overwrite the nonbonded settings of an MD object with its declared profile.

    portable: LJ cut off at 12 angstrom without shift, switch or tail
    correction, arithmetic mixing, AMBER 1-4 scaling and long-range Coulomb
    with a tight PPPM/PME accuracy. It can be evaluated identically by LAMMPS
    and GROMACS.
    """
    profile = getattr(md, 'interaction_profile', None)
    if profile not in PROFILES:
        raise ValueError('Unknown interaction_profile: %s' % profile)
    if profile is None:
        return md
    if not md.pbc:
        raise NotImplementedError('The portable interaction profile requires a periodic cell')
    # Packing deliberately uses short-range LJ without electrostatics.
    if md.pair_style != 'lj/cut':
        md.pair_style = 'lj/cut/coul/long'
        md.cutoff_in = 12.0
        md.cutoff_out = 12.0
        md.kspace_style = 'pppm'
        md.kspace_style_accuracy = '1e-8'
    md.pair_modify = 'mix arithmetic shift no tail no'
    md.special_bonds = 'amber'
    return md
