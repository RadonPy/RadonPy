"""Explicit fixture regeneration; never run automatically by tests.

Run from the repository root: python -m tests.generate_fixtures
These small oligomer cells test translation and execution, not bulk properties.
"""
from pathlib import Path
import json
import hashlib

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from radonpy.core import utils, poly, calc
from radonpy.ff.gaff import GAFF
from radonpy.ff.gaff2 import GAFF2
from radonpy.ff.gaff2_mod import GAFF2_mod
from tests.helpers.gromacs import _write_molecule


def main():
    target = Path(__file__).parent/'fixtures'
    target.mkdir(exist_ok=True)
    sources = {'pe': '*CC*', 'ps': '*CC(*)c1ccccc1', 'pmma': '*CC(*)(C)C(=O)OC'}
    manifest = {'seed': 1701, 'rdkit': rdBase.rdkitVersion,
                'purpose': 'small fixed oligomers; NOT equilibrated bulk reference data', 'fixtures': {}}
    for name, smiles in sources.items():
        np.random.seed(1701)
        unit = utils.mol_from_smiles(smiles, coord=False)
        assert AllChem.EmbedMolecule(unit, randomSeed=1701) == 0
        terminal = utils.mol_from_smiles('*C', coord=False)
        assert AllChem.EmbedMolecule(terminal, randomSeed=1702) == 0
        chain = poly.terminate_mols(poly.polymerize_mols(unit, 3), terminal)
        AllChem.MMFFOptimizeMolecule(chain)
        chain.cell = utils.Cell(20, -20, 20, -20, 20, -20)
        utils.set_mol_id(chain)
        for ff in (GAFF(), GAFF2(), GAFF2_mod()):
            molecule = utils.deepcopy_mol(chain)
            assert ff.ff_assign(molecule, charge='gasteiger')
            np.random.seed(1703)
            calc.set_velocity(molecule, 300)
            path = target/('%s_%s.json' % (name, ff.name))
            _write_molecule(molecule, path)
            manifest['fixtures'][path.name] = {'smiles': Chem.MolToSmiles(molecule),
                'repeat_unit': smiles, 'degree': 3, 'atoms': molecule.GetNumAtoms(),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (target/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
