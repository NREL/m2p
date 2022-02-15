"""Monomer class with parameters and methods."""

from typing import Tuple, Union

import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdchem import Mol

functionality_smarts = {
    "ols": "[C,c;!$(C=O)][OH]",
    "aliphatic_ols": "[C;!$(C=O);!$([a])][OH]",
    "acids": "[#6][#6](=[#8:4])([F,Cl,Br,I,#8H,O-])",
    "prime_amines": "[#6;!$(C=O)][NH2;!$([NH2+])]",
    "carbonates": "[O]=[C]([F,Cl,Br,I,O])([F,Cl,Br,I,O])",
    "acidanhydrides": "[#8]([#6](=[#8]))([#6](=[#8]))",
    "prime_thiols": "[#6;!$(C=O)][SH]",
}


def molecule_from_smiles(smiles: str) -> Union[Mol, None]:
    """Generate rdkit mol from smiles

    Parameters
    ----------
    smiles : str
        SMILES string

    Returns
    -------
    Union[Mol, None]
        RDKit Molecule, or None if can't generate
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        mol = None

    return mol


def get_functionality(reactants, distribution=[]):
    """gets the functional groups from a list of reactants

    inputs: list of smiles
    output: dataframe with count of functional groups
    """

    def id_functionality(r):
        mol = Chem.MolFromSmiles(r.name)
        r.ols = len(
            mol.GetSubstructMatches(Chem.MolFromSmarts(functionality_smarts["ols"]))
        )
        r.aliphatic_ols = len(
            mol.GetSubstructMatches(
                Chem.MolFromSmarts(functionality_smarts["aliphatic_ols"])
            )
        )
        r.acids = len(
            mol.GetSubstructMatches(Chem.MolFromSmarts(functionality_smarts["acids"]))
        )
        r.prime_amines = len(
            mol.GetSubstructMatches(
                Chem.MolFromSmarts(functionality_smarts["prime_amines"])
            )
        )
        r.carbonates = len(
            mol.GetSubstructMatches(
                Chem.MolFromSmarts(functionality_smarts["carbonates"])
            )
        )
        r.acidanhydrides = len(
            mol.GetSubstructMatches(
                Chem.MolFromSmarts(functionality_smarts["acidanhydrides"])
            )
        )
        return r

    df_func = pd.DataFrame(
        data=0,
        index=reactants,
        columns=[
            "ols",
            "acids",
            "prime_amines",
            "carbonates",
            "aliphatic_ols",
            "acidanhydrides",
        ],
    )
    df_func = df_func.apply(lambda r: id_functionality(r), axis=1)

    # appends distribution to dataframe

    if len(distribution) == 0:
        df_func["distribution"] = [1] * df_func.shape[0]
    else:
        df_func["distribution"] = list(distribution)
    return df_func


def enumerate_ester_enantiomers(smiles: str) -> Tuple[Tuple[str]]:
    """Generate enantiomer pairs for a monomer that would participate in an esterification reaction

    Parameters
    ----------
    smiles : str
        SMILES string of the monomer

    Returns
    -------
    Tuple[Tuple[str]]
        The enantiomer pairs, or a single string if no enantiomers are created
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        # Get the atom ids for the acid and ol functionalities
        acid = Chem.MolFromSmarts("[O]-[C]=[O]")
        ol = Chem.MolFromSmarts("[C;X4]-[O]")
        acid_atoms = mol.GetSubstructMatch(acid)
        ol_atoms = mol.GetSubstructMatch(ol)

        # Get the carbon atoms for both, doesn't matter for acid, but does for ol
        atoms = mol.GetAtoms()

        for atom_i in acid_atoms:
            if atoms[atom_i].GetAtomicNum() == 8:
                acid_O = atom_i

        for atom_i in ol_atoms:
            if atoms[atom_i].GetAtomicNum() == 8:
                ol_O = atom_i

        # Get shortest path (backbone) and make stereo based on that
        # Does shortest path make sense?
        atom_path = Chem.GetShortestPath(mol, acid_O, ol_O)

        # Get sites that can be stereo
        stereo_sites = [
            sinfo.centeredOn for sinfo in Chem.rdmolops.FindPotentialStereo(mol)
        ]

        # Find the enantiomers based on stereo sites within the shortest path
        enantiomers = []

        for atom_i in set(atom_path).intersection(stereo_sites):
            atoms[atom_i].SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW)
            smilesR = Chem.MolToSmiles(mol)
            atoms[atom_i].SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)
            smilesS = Chem.MolToSmiles(mol)

            enantiomers.append((smilesR, smilesS))
            atoms[atom_i].SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)

    except BaseException as e:
        enantiomers = [[smiles]]

    if not enantiomers:
        enantiomers = [[smiles]]

    return enantiomers


class Monomer:
    def __init__(self, smiles: str):
        self.smiles = smiles
        self.canonical_smiles = Chem.CanonSmiles(smiles)
        self.molecule = molecule_from_smiles(self.smiles)
        self.molecular_weight = ExactMolWt(self.molecule)

        df = pd.DataFrame([self.smiles], columns=["smiles"])
        self.functionality = get_functionality(df.smiles)

    def __repr__(self) -> str:
        if self.is_valid:
            return f"Valid Monomer with smiles {self.smiles}"
        else:
            return f"Invalid Monomer with smiles {self.smiles}"

    @property
    def esterification_enantiomers(self) -> Tuple[Tuple[str]]:
        """Get possible enantiomers that would participate in an esterification reaction

        Returns
        -------
        Tuple[Tuple[str, str]]
            Tuple containing diads of enantiomers
        """
        return enumerate_ester_enantiomers(self.smiles)
