"""Functions to generate """
import ast
from random import choices
from typing import List, Union

import pandas as pd
from rdkit.Chem import AllChem

from .utils import get_CIP_assignments


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
):
    try:
        if isinstance(reactants, str):
            reactants = ast.literal_eval(reactants)

        if isinstance(distribution, str):
            distribution = ast.literal_eval(distribution)

        if len(distribution) != len(reactants):
            distribution = [1 for i in reactants]

        if not distribution:
            distribution = [1]

        # first get the list of CIP assignments to produce
        poly_list = []
        CIP_assignments = get_CIP_assignments(replicate_structures, pm, DP)
        poly_ids = [i for i, _ in enumerate(reactants)]
        prop_dict = get_vinyl_prop_dict(reactants)
        init_dict = get_vinyl_init_dict(reactants)

        for CIPs in CIP_assignments:
            poly_id = choices(poly_ids, distribution)[0]
            poly = init_dict[poly_id][CIPs[0]]
            for CIP in CIPs[1:]:
                poly_id = choices(poly_ids, distribution)[0]
                poly = vinyl_prop_stereo(poly, prop_dict[poly_id][CIP])

            poly = vinyl_terminate_stereo(poly)

            poly_list.append(AllChem.MolToSmiles(poly))

        return_poly = pd.DataFrame(data=poly_list, columns=["smiles_polymer"])

    except BaseException as e:
        return_poly = "ERROR:Vinyl_Stereo_ReactionFailed"

    return return_poly


def get_vinyl_prop_dict(smiles_list):
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


def get_vinyl_init_dict(smi):
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


def vinyl_prop_stereo(poly, monomer):
    prop_rxn = AllChem.ReactionFromSmarts("[C:1][Xe].[Kr][C:2]>>[C:1][C:2]")

    products = prop_rxn.RunReactants((poly, monomer,))

    return products[0][0]


def vinyl_terminate_stereo(poly):
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
    try:
        if isinstance(reactants, str):
            reactants = ast.literal_eval(reactants)

        if not (isinstance(reactants[0], list) or isinstance(reactants[0], tuple)):
            reactants = [reactants]

        if isinstance(distribution, str):
            distribution = ast.literal_eval(distribution)

        if len(distribution) != len(reactants):
            distribution = [1 for i in reactants]

        if not distribution:
            distribution = [1]

        # first get the list of CIP assignments to produce
        poly_list = []
        poly_ids = [i for i, _ in enumerate(reactants)]
        CIP_assignments = get_CIP_assignments(replicate_structures, pm, DP)
        prop_dict = get_ester_prop_dict(reactants)
        init_dict = get_ester_init_dict(reactants)

        for CIPs in CIP_assignments:
            # Init
            poly_id = choices(poly_ids, distribution)[0]
            poly = init_dict[poly_id][CIPs[0]]

            for CIP in CIPs[1:]:
                last_poly = poly
                poly_id = choices(poly_ids, distribution)[0]
                poly = ester_prop_stereo(poly, prop_dict[poly_id][CIP])
                foo = 1

            poly = ester_terminate_stereo(poly)

            poly_list.append(AllChem.MolToSmiles(poly))

        return_poly = pd.DataFrame(data=poly_list, columns=["smiles_polymer"])

    except BaseException as e:
        return_poly = "ERROR:Ester_Stereo_ReactionFailed"

    return return_poly


def get_ester_prop_dict(smiles_list):
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


def get_ester_init_dict(smiles_list):
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


def ester_prop_stereo(poly, monomer):
    prop_rxn = AllChem.ReactionFromSmarts("[C:1][Xe].[Ar][C:2]>>[C:1][O][C:2]")

    products = prop_rxn.RunReactants((poly, monomer,))

    return products[0][0]


def ester_terminate_stereo(poly):
    prop_rxn = AllChem.ReactionFromSmarts("([Pb][C:1].[Xe][C:2])>>([O][C:1].[O][C:2])")

    products = prop_rxn.RunReactants((poly,))

    return products[0][0]
