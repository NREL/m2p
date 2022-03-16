"""Useful functions."""
import random
import ast

import numpy as np
from math import ceil, floor
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import ChiralType


def get_monomer_sequences(n_structures, distribution, DP):
    """Only supports homopolymer/copolymer of 2 monomers. Anything larger will be given
    a valid sequence that is near the actual distribution, but no consideration for rounding
    will be taken."""

    # homopolymer
    if not distribution or len(distribution) == 1:
        return [[0] * DP] * n_structures

    if len(distribution) == 2:
        # First normalize the distribution and multiply by DP to sum to DP
        distribution = [
            DP * monomer_d / sum(distribution) for monomer_d in distribution
        ]

        bias_first = [ceil(distribution[0]), floor(distribution[1])]
        bias_second = [floor(distribution[0]), ceil(distribution[1])]
        monomer_sequences = [
            [0] * bias_first[0] + [1] * bias_first[1],
            [0] * bias_second[0] + [1] * bias_second[1],
        ]

        # Calculate weights of first/second bias
        weight_first = 1 - (bias_first[0] - distribution[0])
        weight_second = 1 - weight_first
        weights = [weight_first, weight_second]

        sequences = []
        i = 0
        i_max = 100
        while len(sequences) < n_structures:
            sequence_list = random.choices(monomer_sequences, weights)[0]

            perm = tuple(np.random.permutation(sequence_list))
            if (perm not in sequences) or (i >= i_max):
                sequences.append(perm)
                i = 0
            else:
                i += 1

    else:
        distribution = [
            DP * monomer_d / sum(distribution) for monomer_d in distribution
        ]
        ceils = [ceil(i) for i in distribution]
        sequence_list = [[i] * n for i, n in enumerate(ceils)]
        sequence_list = [item for sublist in sequence_list for item in sublist]

        sequences = []
        i = 0
        i_max = 100
        while len(sequences) < n_structures:
            perm = tuple(np.random.permutation(sequence_list))
            perm = perm[:DP]
            if (perm not in sequences) or (i >= i_max):
                sequences.append(perm)
                i = 0
            else:
                i += 1

    return sequences


def get_CIP_assignments(n_structures, pm, DP):
    """Generate a a list containing the R/S assignemnts for compounds. First calculates diads then converts to R/S."""
    # Initiate info to generate a diad list
    n_diads = DP - 1
    lower, upper = [int(floor(n_diads * pm)), int(ceil(n_diads * pm))]
    diad_lists = [
        np.array([1] * lower + [0] * (n_diads - lower)),
        np.array([1] * upper + [0] * (n_diads - upper)),
    ]

    # Get weights of sampling so average pm (with infinite replicates) is equal to specified pm
    if lower != upper:
        r_weight = (n_diads * pm - lower / n_diads) / (upper - lower)
        l_weight = 1 - r_weight
        weights = [l_weight, r_weight]
    else:
        weights = [1, 1]

    # Generate n_structures amount of diad lists
    # Ideally unique, but only try a max iteration of 10^3
    i = 0
    replicates = []
    max_iter = 10 ** 3
    while len(replicates) < n_structures:
        # Generate a random permutation and convert it into a string
        diad_list = random.choices(diad_lists, weights)[0]
        perm = tuple(np.random.permutation(diad_list))
        perm_string = "".join([f"{num}" for num in perm])

        # Add permutation to replicate list and reset iteration count
        if (perm_string not in replicates) or (i >= max_iter):
            replicates.append(perm_string)
            i = 0

        i += 1

    # Convert to R/S with arbitrary starting stereo for first atom
    RS = ["R", "S"]
    RS_replicates = []
    for diads in replicates:
        current_RS = np.random.randint(0, 2)
        RS_list = [RS[current_RS % 2]]

        for diad in diads:
            if diad == "0":
                current_RS += 1
            RS_list.append(RS[current_RS % 2])
        RS_replicates.append(RS_list)

    return RS_replicates


def assign_CIP_to_poly(poly, CIP_list, smarts1, smarts2):
    chiral_dict = {
        "R": ChiralType.CHI_TETRAHEDRAL_CCW,
        "S": ChiralType.CHI_TETRAHEDRAL_CW,
    }

    # Get the chiral centers on backbone
    atom1 = poly.GetSubstructMatch(AllChem.MolFromSmarts(smarts1))[0]
    atom2 = poly.GetSubstructMatch(AllChem.MolFromSmarts(smarts2))[0]
    chiral_centers = AllChem.FindMolChiralCenters(poly, includeUnassigned=True)
    backbone = AllChem.rdmolops.GetShortestPath(poly, atom1, atom2)
    chiral_backbone_centers = [cc[0] for cc in chiral_centers if cc[0] in backbone]

    specified_chirality = {
        atom_i: CIP_id for atom_i, CIP_id in zip(chiral_backbone_centers, CIP_list)
    }

    # First pass of assigning CIP_id
    atoms = poly.GetAtoms()
    for atom_i, CIP_id in zip(chiral_backbone_centers, CIP_list):
        atoms[atom_i].SetChiralTag(chiral_dict[CIP_id])
    AllChem.AssignCIPLabels(poly)

    # Identify incorrect assignments
    current_chirality = AllChem.FindMolChiralCenters(poly, includeCIP=True)
    incorrect_set = set(specified_chirality.items()).symmetric_difference(
        current_chirality
    )
    incorrect_atom_set = set()
    for mol_chirality in incorrect_set:
        incorrect_atom_set.add(mol_chirality[0])

    for atom_i in incorrect_atom_set:
        current_chirality = atoms[atom_i].GetChiralTag()
        if current_chirality == chiral_dict["R"]:
            atoms[atom_i].SetChiralTag(chiral_dict["S"])
        if current_chirality == chiral_dict["S"]:
            atoms[atom_i].SetChiralTag(chiral_dict["R"])

    AllChem.AssignCIPLabels(poly)

    return AllChem.MolToSmiles(poly)


def assign_CIP_to_polyvinyl(poly, CIP_list):
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

    def vinyl_term(mol):
        """Terminate vinyl reaction"""
        rxn = AllChem.ReactionFromSmarts("([Pb][C:1].[C:2][Xe])>>([C:1].[C:2])")
        products = rxn.RunReactants((mol,))
        product_mols = get_valid_mols(products)[0]

        return product_mols

    chiral_dict = {
        "R": ChiralType.CHI_TETRAHEDRAL_CCW,
        "S": ChiralType.CHI_TETRAHEDRAL_CW,
    }

    smarts1 = "[Xe]"
    smarts2 = "[Pb]"

    # Get the chiral centers on backbone
    atom1 = poly.GetSubstructMatch(AllChem.MolFromSmarts(smarts1))[0]
    atom2 = poly.GetSubstructMatch(AllChem.MolFromSmarts(smarts2))[0]
    chiral_centers = AllChem.FindMolChiralCenters(poly, includeUnassigned=True)
    backbone = AllChem.rdmolops.GetShortestPath(poly, atom1, atom2)
    chiral_backbone_centers = [cc[0] for cc in chiral_centers if cc[0] in backbone]

    specified_chirality = {
        atom_i: CIP_id for atom_i, CIP_id in zip(chiral_backbone_centers, CIP_list)
    }

    # First pass of assigning CIP_id
    poly = vinyl_term(poly)
    atoms = poly.GetAtoms()
    for atom_i, CIP_id in zip(chiral_backbone_centers, CIP_list):
        atoms[atom_i].SetChiralTag(chiral_dict[CIP_id])

    AllChem.AssignCIPLabels(poly)

    # Identify incorrect assignments
    current_chirality = AllChem.FindMolChiralCenters(poly, includeCIP=True)
    incorrect_set = set(specified_chirality.items()).symmetric_difference(
        current_chirality
    )
    incorrect_atom_set = set()
    for mol_chirality in incorrect_set:
        incorrect_atom_set.add(mol_chirality[0])

    for atom_i in incorrect_atom_set:
        current_chirality = atoms[atom_i].GetChiralTag()
        if current_chirality == chiral_dict["R"]:
            atoms[atom_i].SetChiralTag(chiral_dict["S"])
        if current_chirality == chiral_dict["S"]:
            atoms[atom_i].SetChiralTag(chiral_dict["R"])

    AllChem.AssignCIPLabels(poly)

    return poly
