#  Copyright (c) 2022. RadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# ******************************************************************************
# core.const module
# ******************************************************************************
import os

__version__ = '0.2.3'

print_level = 1
tqdm_disable = True
debug = False

# Do not check installing package in LAMMPS
check_package_disable = False

# Use mpi4py
mpi4py_avail = False

# Path to LAMMPS binary
lammps_exec = os.getenv('LAMMPS_EXEC', 'lmp_mpi')

# %i: number of process
mpi_cmd = 'mpirun -n %i'

# 1st %s: Path of the LAMMPS binary
# 2nd %s: Accelerate options (GPU, OpenMP, Intel)
# 3rd %s: Path of the input file
# 4th %s: Path of the output file
lmp_cmd = '%s %s -in %s -sc none -log %s'

# Conversion factors
atm2pa = 101325
bohr2ang = 0.52917720859
cal2j = 4.184
au2kcal = 627.5095
au2ev = 27.2114
au2kj = au2kcal * cal2j
au2debye = 2.541765
ang2m = 1e-10
m2ang = 1e+10
ang2cm = 1e-8
cm2ang = 1e+8

# Physical constants
kB = 1.3806504e-23 # J/K
NA = 6.02214076e+23 # mol^-1
R = kB * NA # J/(K mol)
h = 6.62607015e-34 # J s
c = 2.99792458e+8 # m/s
e = 1.602e-19 # C
eps0 = 8.8541878128e-12 # F/m

# Constants
pdb_id = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

