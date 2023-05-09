![logo](https://user-images.githubusercontent.com/83273612/160471242-40d7d7f1-d2cd-4658-b4e1-75f5e608665d.png)

## Overview
RadonPy is the first open-source Python library for fully automated calculation for a comprehensive set of polymer properties, using all-atom classical MD simulations. For a given polymer repeating unit with its chemical structure, the entire process of the MD simulation can be carried out fully automatically, including molecular modelling, equilibrium and non-equilibrium MD simulations, automatic determination of the completion of equilibration, scheduling of restarts in case of failure to converge, and property calculations in the post-process step. In this release, the library comprises the calculation of 15 properties at the amorphous state.

## Requirement
- Python 3.7, 3.8, 3.9, 3.10
- LAMMPS >= 3Mar20
- rdkit >= 2020.03
- psi4 >= 1.5
- resp
- dftd3
- mdtraj >= 1.9
- scipy
- matplotlib

## Installation and usage
User manual and PyPI, conda packages are currently in preparation.

[PDF file](https://github.com/RadonPy/RadonPy/blob/develop/docs/RadonPy_tutorial_20220331.pdf) of RadonPy tutorial is available.

### Instllation for conda:
1. Create conda environment
```
conda create -n radonpy python=3.9
conda activate radonpy
```

2. Installation of requirement packages by conda
```
conda install -c psi4 -c conda-forge rdkit psi4 resp mdtraj matplotlib
```

3. Installation of LAMMPS by conda
```
conda install -c conda-forge lammps
```

or manually build from source of [LAMMPS official site](https://www.lammps.org/).
In this case, the environment variable must be set:
```
export LAMMPS_EXEC=<Path-to-LAMMPS-binary>
```

4. Installation of RadonPy
```
git clone -b main https://github.com/RadonPy/RadonPy.git
export PYTHONPATH=<Path-to-RadonPy>:$PYTHONPATH
```


## Features
- Fully automated all-atom classical MD calculation for polymeric materials
	- Conformation search
	- Cherge calculation (RESP, ESP, Mulliken, Lowdin, Gasteiger)
	- Electronic property calculation (HOMO, LUMO, dipole moment, polarizability)
	- Generation of a polymer chain
		- Homopolymer
		- Alternating copolymer
		- Random copolymer
		- Block copolymer
	- Generation of a simulation cell
		- Amorphous
		- Polymer mixture
		- Polymer solution
		- Crystalline polymer
		- Oriented structure
	- Run for equilibration MD
	- Checking archivement of equilibrium
	- Run for non-equilibrium MD (NEMD)
	- Calculation of physical properties from the MD calculation results
		- Thermal conductivity
		- Thermal diffusivity
		- Density
		- Cp
		- Cv
		- Linear expansion coefficient
		- Volumetric expansion coefficient
		- Compressibility
		- Bulk modulus
		- Isentropic compressibility
		- Isentropic bulk modulus
		- Static dielectric constant
		- Refractive index
		- Radius of gyration
		- End-to-end distance
		- Nematic order parameter
	- Using LAMMPS and Psi4 as calculation engines of MD and DFT calculations
- Implementation of add-on like presets to allow for proper and easy execution of polymer MD calculations
	- Equilibration MD
	- Calculation of thermal conductivity with NEMD
- Easy installation
    - Only using open-source software
- Tools for polymer informatics
	- Force field descriptor
	- Generator of macrocyclic oligomer for descriptor construction of polymers
	- Full and substruct match function for polymer SMILES
	- Extractor of mainchain in a polymer backbone
	- Monomerization of oligomer SMILES
	- Emulator of polymer classification in PoLyInfo

## MD calculated data
- [1070 amorphous polymers](https://github.com/RadonPy/RadonPy/blob/develop/data/PI1070.csv)

## Publications
1. [Y. Hayashi, J. Shiomi, J. Morikawa, R. Yoshida, "RadonPy: Automated Physical Property Calculation using All-atom Classical Molecular Dynamics Simulations for Polymer Informatics," npj Comput. Mater. 8:222 (2022)](https://www.nature.com/articles/s41524-022-00906-4)

## Contributors
- Yoshihiro Hayashi (The Institute of Statistical Mathematics)

## Related projects
- [XenonPy (Machine learning tools for materials informatics)](https://github.com/yoshida-lab/XenonPy)
- [SMiPoly (Polymerization rule-based virtual polymer generator)](https://github.com/PEJpOhno/SMiPoly)

## Copyright and licence
©Copyright 2023 The RadonPy developers, all rights reserved.
Released under the `BSD-3 license`.


![Radon_ikaho](https://user-images.githubusercontent.com/83273612/158885745-224f6e7a-4b1d-46f4-b5c6-80455827c904.png)

