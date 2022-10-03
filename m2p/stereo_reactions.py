"""Functions to generate """
import ast
from random import choices
from typing import List, Union

import pandas as pd
from rdkit.Chem import AllChem

from .utils import get_CIP_assignments, get_monomer_sequences


def get_valid_mols(smi_list):
    mol_list = []
    for smi in smi_list:
        try:
            AllChem.SanitizeMol(smi[0])
            mol_list.append(smi[0])
        except Exception as e:
            print(e)
            pass

    return mol_list


def polyvinyl_stereo(
    reactants: Union[List, str],
    DP: int,
    replicate_structures: int = 1,
    distribution: Union[List[float], List] = [],
    pm: float = 1,
) -> pd.DataFrame:
    """Create polymers ustilizing radical polymerization.

    Parameters
    ----------
    reactants : Union[List, str]
        The smiles to react.
    DP : int
        The degree of polymerization, measured by number of monomers incorporated.
    replicate_structures : int, optional
        How many replicate structures to generate, by default 1
    distribution : Union[List[float], List], optional
        The distribution of monomers determined by their relative abundance. I.e. [1, 2] or [0.33, 0.66] would generate a polymer with two times monomer two than one. Distribution values can be given as any set of numbers, by default []
    pm : float, optional
        The pm value for the polymer, by default 1

    Returns
    -------
    pd.DataFrame
        A dataframe containing the polymerization information of the supplied monomers and conditions.
    """
    try:
        if isinstance(reactants, str):
            reactants = ast.literal_eval(reactants)

        if isinstance(distribution, str):
            distribution = ast.literal_eval(distribution)

        if len(distribution) != len(reactants):
            distribution = [1 for i in reactants]

        if not distribution:
            distribution = [1]

        # first get the list of CIP assignments and monomer sequences to produce
        poly_list = []
        monomer_sequences = get_monomer_sequences(
            replicate_structures, distribution, DP
        )
        CIP_assignments = get_CIP_assignments(replicate_structures, pm, DP)

        prop_dict = get_vinyl_prop_dict(reactants)
        init_dict = get_vinyl_init_dict(reactants)

        for i, CIPs in enumerate(CIP_assignments):
            poly_id = monomer_sequences[i][0]
            poly = init_dict[poly_id][CIPs[0]]
            for j, CIP in enumerate(CIPs[1:]):
                poly_id = monomer_sequences[i][j + 1]
                poly = vinyl_prop_stereo(poly, prop_dict[poly_id][CIP])

            poly = vinyl_terminate_stereo(poly)

            poly_list.append(AllChem.MolToSmiles(poly))

        return_poly = pd.DataFrame(data=poly_list, columns=["smiles_polymer"])

    except BaseException as e:
        return_poly = "ERROR:Vinyl_Stereo_ReactionFailed"

    return return_poly


def get_vinyl_prop_dict(smiles_list: List[str]):
    """Generate the propogation dictionary for radical polymerization."""

    def stm(smi):
        return AllChem.MolFromSmiles(smi)

    enantiomer_dict = dict()
    for i, smi in enumerate(smiles_list):
        # Generate dict entries one by one
        mol = stm(smi)
        rxn_smarts = AllChem.ReactionFromSmarts(
            "[C:1]=[C:2][!#1:3]>>[Kr][C:1][C@@:2]([!#1:3])[Xe]"
        )

        products = [
            AllChem.MolFromSmiles(AllChem.MolToSmiles(prod[0]))
            for prod in rxn_smarts.RunReactants((mol,))
        ]

        if len(products) == 1:
            rxn_smarts2 = AllChem.ReactionFromSmarts(
                "[C:1]=[C:2][!#1:3]>>[Kr][C:1][C@:2]([!#1:3])[Xe]"
            )

            products.extend(
                [
                    AllChem.MolFromSmiles(AllChem.MolToSmiles(prod[0]))
                    for prod in rxn_smarts2.RunReactants((mol,))
                ]
            )

        enantiomer_dict[i] = {"R": products[0], "S": products[1]}

    return enantiomer_dict


def get_vinyl_init_dict(smi: str):
    """Generate the initiator dictionary for vinyl polymerization."""
    enantiomer_dict = get_vinyl_prop_dict(smi)
    make_terminal = AllChem.ReactionFromSmarts("[Kr][C:1]>>[Hg][C:1]")

    for key in enantiomer_dict:
        enantiomer_dict[key]["R"] = make_terminal.RunReactants(
            (enantiomer_dict[key]["R"],)
        )[0][0]
        enantiomer_dict[key]["S"] = make_terminal.RunReactants(
            (enantiomer_dict[key]["S"],)
        )[0][0]

    return enantiomer_dict


def vinyl_prop_stereo(poly: AllChem.rdchem.Mol, monomer: AllChem.rdchem.Mol):
    """Carry out a single propogation step."""
    prop_rxn = AllChem.ReactionFromSmarts("[C:1][Xe].[Kr][C:2]>>[C:1][C:2]")

    products = prop_rxn.RunReactants(
        (
            poly,
            monomer,
        )
    )

    return products[0][0]


def vinyl_terminate_stereo(poly: AllChem.rdchem.Mol):
    """Carry out a single termination step."""
    term_rxn = AllChem.ReactionFromSmarts("([Xe][C:1].[Hg][C:2])>>([C:1].[C:2])")

    products = term_rxn.RunReactants((poly,))

    return products[0][0]


# #############
# Polyesters
def polyester_stereo(
    reactants: Union[List, str],
    DP: int,
    replicate_structures: int = 1,
    distribution: Union[List[float], List] = [1],
    pm: float = 1,
):
    """Create polyesters with stereochemical information.

    Parameters
    ----------
    reactants : Union[List, str]
        The smiles to react.
    DP : int
        The degree of polymerization, measured by number of monomers incorporated.
    replicate_structures : int, optional
        How many replicate structures to generate, by default 1
    distribution : Union[List[float], List], optional
        The distribution of monomers determined by their relative abundance. I.e. [1, 2] or [0.33, 0.66] would generate a polymer with two times monomer two than one. Distribution values can be given as any set of numbers, by default []
    pm : float, optional
        The pm value for the polymer, by default 1

    Returns
    -------
    pd.DataFrame
        A dataframe containing the polymerization information of the supplied monomers and conditions.
    """
    try:
        if isinstance(reactants, str):
            reactants = ast.literal_eval(reactants)

        if isinstance(distribution, str):
            distribution = ast.literal_eval(distribution)

        if not distribution:
            distribution = []

        # Get the smiles into correct form
        reactants = [[x, y] for x, y in zip(reactants[::2], reactants[1::2])]

        # first get the list of CIP assignments and monomer sequences to produce
        poly_list = []
        monomer_sequences = get_monomer_sequences(
            replicate_structures, distribution, DP
        )
        CIP_assignments = get_CIP_assignments(replicate_structures, pm, DP)

        # Next generate prop and init dicts
        prop_dict = get_ester_prop_dict(reactants)
        init_dict = get_ester_init_dict(reactants)

        for i, CIPs in enumerate(CIP_assignments):
            # Init
            poly_id = monomer_sequences[i][0]
            poly = init_dict[poly_id][CIPs[0]]

            for j, CIP in enumerate(CIPs[1:]):
                poly_id = monomer_sequences[i][j + 1]
                poly = ester_prop_stereo(poly, prop_dict[poly_id][CIP])

            poly = ester_terminate_stereo(poly)

            poly_list.append(AllChem.MolToSmiles(poly))

        return_poly = pd.DataFrame(data=poly_list, columns=["smiles_polymer"])

    except BaseException as e:
        return_poly = "ERROR:Ester_Stereo_ReactionFailed"

    return return_poly


def get_ester_prop_dict(smiles_list: str):
    """Generate the propogation dictionary. Contains monomers + enantiomers."""
    # Xe reacts with Argon
    def stm(smi):
        return AllChem.MolFromSmiles(smi)

    rxn_smarts = AllChem.ReactionFromSmarts(
        "([O;D1:1][C:2]=[O:3].[C:4][O;D1:5])>>([Ar:1][C:2]=[O:3].[C:4][Xe:3])"
    )

    enantiomer_dict = dict()
    for i, monomers in enumerate(smiles_list):
        products = [rxn_smarts.RunReactants((stm(smi),))[0][0] for smi in monomers]

        enantiomer_dict[i] = {"R": products[0], "S": products[1]}

    return enantiomer_dict


def get_ester_init_dict(smiles_list: str):
    """Generate the initiation dictionary. Contains monomers + enantiomers."""
    # Pb is terminal on the carboxylic acid, doesn't react
    # Xe reacts with Argon
    def stm(smi):
        return AllChem.MolFromSmiles(smi)

    rxn_smarts = AllChem.ReactionFromSmarts(
        "([O;D1:1][C:2]=[O:3].[C:4][O;D1:5])>>([Pb:1][C:2]=[O:3].[C:4][Xe:3])"
    )

    enantiomer_dict = dict()
    for i, monomers in enumerate(smiles_list):
        products = [rxn_smarts.RunReactants((stm(smi),))[0][0] for smi in monomers]

        enantiomer_dict[i] = {"R": products[0], "S": products[1]}

    return enantiomer_dict


def ester_prop_stereo(poly: AllChem.rdchem.Mol, monomer: AllChem.rdchem.Mol):
    prop_rxn = AllChem.ReactionFromSmarts("[C:1][Xe].[Ar][C:2]>>[C:1][O][C:2]")

    products = prop_rxn.RunReactants(
        (
            poly,
            monomer,
        )
    )

    return products[0][0]


def ester_terminate_stereo(poly: AllChem.rdchem.Mol):
    prop_rxn = AllChem.ReactionFromSmarts("([Pb][C:1].[Xe][C:2])>>([O][C:1].[O][C:2])")

    products = prop_rxn.RunReactants((poly,))

    return products[0][0]
