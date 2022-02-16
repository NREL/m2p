import pytest
from rdkit import Chem

from m2p import Monomer


def test_acid_ol_monomer():
    acid_ol_smiles = "O=C(O)CC(C)CCO"
    m = Monomer(acid_ol_smiles)
    assert m.molecular_weight == 132.078644244
    assert m.canonical_smiles == Chem.CanonSmiles(acid_ol_smiles)
    assert m.esterification_enantiomers == [
        ("C[C@@H](CCO)CC(=O)O", "C[C@H](CCO)CC(=O)O")
    ]


def test_acid_ol_multiple_stereo():
    acid_ol_smiles = "O=C(O)CC(C)C(C)C(C)CO"
    m = Monomer(acid_ol_smiles)

    assert m.esterification_enantiomers == [
        ("CC(CC(=O)O)C(C)[C@@H](C)CO", "CC(CC(=O)O)C(C)[C@H](C)CO"),
        ("CC(CO)C(C)[C@H](C)CC(=O)O", "CC(CO)C(C)[C@@H](C)CC(=O)O"),
        ("CC(CO)[C@H](C)C(C)CC(=O)O", "CC(CO)[C@@H](C)C(C)CC(=O)O"),
    ]
