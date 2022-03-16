import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from m2p import PolyMaker
from m2p.stereo_reactions import (
    get_vinyl_init_dict,
    get_vinyl_prop_dict,
    vinyl_prop_stereo,
    vinyl_terminate_stereo,
)

pm = PolyMaker()


@pytest.fixture
def vinyl():
    return pm.thermoplastic("C=C", DP=10, mechanism="vinyl").smiles_polymer[0]


@pytest.fixture
def RS_ester_reactants():
    poly_name = ["poly(RS-3-hydroxybutyrate)"]
    smiles = ["C[C@@H](O)CC(=O)O.C[C@H](O)CC(=O)O"]

    reactants = pd.DataFrame(
        {"smiles_monomer": smiles, "replicate_structures": [1]},
        columns=["smiles_monomer", "replicate_structures"],
        index=poly_name,
    )
    reactants["monomers"] = reactants.smiles_monomer.apply(
        lambda s: pm.get_monomers(s, stereochemistry=True)
    )
    reactants["mechanism"] = "ester_stereo"

    return reactants


@pytest.fixture
def RS_vinyl_reactants():
    poly_name = ["Polypropylene"]
    smiles = ["CC=C"]

    reactants = pd.DataFrame(
        {"smiles_monomer": smiles, "replicate_structures": [1]}, index=poly_name
    )
    reactants["monomers"] = reactants.smiles_monomer.apply(
        lambda s: pm.get_monomers(s, stereochemistry=True)
    )
    reactants["mechanism"] = "vinyl_stereo"

    return reactants


def calc_pm(smi):
    # Dict to count
    RS_list = []
    mol = Chem.MolFromSmiles(smi)

    for atom in mol.GetAtoms():
        try:
            chiral_center = atom.GetProp("_CIPCode")
            RS_list.append(chiral_center)
        except:
            pass

    # Zip together list to get neighbors and determine R vs M addition
    is_M_add = list(map(lambda t: t[0] == t[1], zip(*(RS_list, RS_list[1:]))))
    pm = sum(is_M_add) / len(is_M_add)

    return pm


def test_vinyl(vinyl):
    assert vinyl == "CCCCCCCCCCCCCCCCCCCC"


def test_ester_stereo_iso(RS_ester_reactants):
    poly_df = pm.thermoplastic_stereo(RS_ester_reactants, DP=10, pm=1, verbose=False)
    pmeso = calc_pm(poly_df["smiles_polymer"][0])

    assert len(poly_df) == 1
    assert pmeso == 1


def test_ester_stereo_syn(RS_ester_reactants):
    poly_df = pm.thermoplastic_stereo(RS_ester_reactants, DP=10, pm=0, verbose=False)
    pmeso = calc_pm(poly_df["smiles_polymer"][0])

    assert len(poly_df) == 1
    assert pmeso == 0


def test_ester_stereo_a(RS_ester_reactants):
    poly_df = pm.thermoplastic_stereo(RS_ester_reactants, DP=10, pm=0.5, verbose=False)
    pmeso = calc_pm(poly_df["smiles_polymer"][0])

    assert len(poly_df) == 1
    assert 0 < pmeso and pmeso < 1


def test_df_pm_ester_stereo(RS_ester_reactants):
    RS_ester_reactants["pm"] = 1

    poly_df = pm.thermoplastic_stereo(RS_ester_reactants, DP=10, verbose=False)
    pmeso = calc_pm(poly_df["smiles_polymer"][0])

    assert len(poly_df) == 1
    assert pmeso == 1


def test_vinyl_enantiomer_dicts():
    def m2s(mol):
        return AllChem.CanonSmiles(AllChem.MolToSmiles(mol))

    smis = ["C=Cc1ccccc1", "C=C([Cl])C"]

    smis_prop_dict = get_vinyl_prop_dict(smis)
    smis_init_dict = get_vinyl_init_dict(smis)

    assert m2s(smis_prop_dict[0]["R"]) == "[Kr]C[C@H]([XeH])c1ccccc1"
    assert m2s(smis_prop_dict[0]["S"]) == "[Kr]C[C@@H]([XeH])c1ccccc1"

    assert m2s(smis_prop_dict[1]["R"]) == "C[C@](Cl)([XeH])C[Kr]"
    assert m2s(smis_prop_dict[1]["S"]) == "C[C@@](Cl)([XeH])C[Kr]"

    assert m2s(smis_init_dict[0]["R"]) == "[XeH][C@@H](C[Hg])c1ccccc1"
    assert m2s(smis_init_dict[0]["S"]) == "[XeH][C@H](C[Hg])c1ccccc1"

    assert m2s(smis_init_dict[1]["R"]) == "C[C@](Cl)([XeH])C[Hg]"
    assert m2s(smis_init_dict[1]["S"]) == "C[C@@](Cl)([XeH])C[Hg]"


def test_vinyl_prop():
    smi = ["C=CC"]
    prop_dict = get_vinyl_prop_dict(smi)
    init_dict = get_vinyl_init_dict(smi)

    CIP_assignments = ["R", "S", "R", "S"]

    chain = init_dict[0][CIP_assignments[0]]
    for CIP in CIP_assignments[1:]:
        chain = vinyl_prop_stereo(chain, prop_dict[0][CIP])
    chain = vinyl_terminate_stereo(chain)

    assert AllChem.MolToSmiles(chain) == "CCC[C@H](C)C[C@@H](C)CC(C)C"


def test_vinyl(RS_vinyl_reactants):
    poly_df = pm.thermoplastic_stereo(RS_vinyl_reactants, DP=10, pm=0, verbose=False)
    assert len(poly_df) == 1


def test_vinyl_and_ester(RS_ester_reactants, RS_vinyl_reactants):
    df = pd.concat([RS_vinyl_reactants, RS_ester_reactants]).reset_index(drop=True)
    poly_df = pm.thermoplastic_stereo(df, DP=10, pm=0, verbose=False)
    assert len(poly_df) == 2
