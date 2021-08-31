import pandas as pd
from m2p import PolyMaker
from rdkit import Chem

pm = PolyMaker()

polymer_name = ['poly(RS-3-hydroxybutyrate)', 'poly(RS-3-hydroxybutyrate)', 'poly(RS-3-hydroxybutyrate)']
smiles = ['C[C@@H](O)CC(=O)O.C[C@H](O)CC(=O)O', 'C[C@@H](O)CC(=O)O.C[C@H](O)CC(=O)O', 'C[C@@H](O)CC(=O)O.C[C@H](O)CC(=O)O']
pmeso = [6/7, 4/7.5, 1/7]

# define input dataframe
poly_df = pd.DataFrame({'smiles': smiles, 'pm': pmeso}, index=polymer_name)
poly_df['monomers'] = poly_df.smiles.apply(
    lambda s: pm.get_monomers(s, stereochemistry=True)
    )

# Distribution 
# Length of monomers (or single value), doesn't have to add up to any specific value
distribution = [1]

replicate_structures = 5

# Syndiotactic
pmeso = .5

DP = 22

poly_df = pm.thermoplastic_tactic(poly_df, DP, 'ester', replicate_structures, distribution, pmeso, verbose=True)

def get_RS(smi):
    mol = Chem.MolFromSmiles(smi)
    chiral_centers = {}
    for i, atom in enumerate(mol.GetAtoms()):
        try:
            chiral_centers[i] = atom.GetProp('_CIPCode')
        except:
            pass
    
    return chiral_centers

def get_pm(smi):
    rs = list(get_RS(smi).values())
    diads = list(zip(*(list(rs), list(rs[1:]))))

    s = 0
    for diad in diads:
        if diad[0] == diad[1]:
            s += 1

    pm = s / len(diads)
    return pm

get_pm(poly_df.polymer[0])

print('ok')