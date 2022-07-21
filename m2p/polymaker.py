import ast
import itertools
import random
from ast import literal_eval as leval
from copy import deepcopy
from typing import List, Union

import numpy as np
import pandas as pd
from casadi.casadi import ceil, floor
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem, Descriptors
from tqdm import tqdm

from .stereo_reactions import polyester_stereo, polyvinyl_stereo

tqdm.pandas()
lg = RDLogger.logger()
# lg.setLevel(RDLogger.ERROR)
lg.setLevel(4)


class PolyMaker:
    def __init__(self):
        self.smiles_templates = {
            "ols": "[C,c;!$(C=O)][OH]",
            "aliphatic_ols": "[C;!$(C=O);!$([a])][OH]",
            "acids": "[#6][#6](=[#8:4])([F,Cl,Br,I,#8H,O-])",
            "prime_amines": "[#6;!$(C=O)][NH2;!$([NH2+])]",
            "carbonates": "[O]=[C]([F,Cl,Br,I,O])([F,Cl,Br,I,O])",
            "cyclic_carbonates": "[O]=[C]1[O][C][C][O]1",
            "acidanhydrides": "[#8]([#6](=[#8]))([#6](=[#8]))",
            "prime_thiols": "[#6;!$(C=O)][SH]",
            "isocyanates": "[#6][#7;!$([#7+])]=[#6]=[#8]",
            "acrylates": "[CH2]=[C][C](=[O])",
            "double_bond": "C=C",
            "vinyls": "[CH2]=[CH!$(CC=O)]",
            "double_bond_secondary": "[CH]=[CH]",
            "double_bond_tertiary": "C[C]=[C!$([CH])]C",
            "double_bond_quaternary": "C[C!$([CH])]=[C!$([CH])]C",
        }
        self.reactions = {
            "ester": {
                "diols_acids": "[C;!$(C=O);!$([a]):6][OH:1].[#6:2][#6:3](=[O:4])([F,Cl,Br,I,#8H,O-:5])>>"
                "[C:6][O:1][#6:3](=[O:4])([#6:2])",
                "diacids_ols": "[#6:2][#6:3](=[O:4])([F,Cl,Br,I,#8H,O-:5]).[C;!$(C=O);!$([a]):6][OH:1]>>"
                "[C:6][O:1][#6:3](=[O:4])([#6:2])",
                "infinite_chain": "([C;!$(C=O);!$([a]):1][OH:2].[#6:3][#6:4](=[O:5])([F,Cl,Br,I,OH,O-:6]))>>"
                "[*:1][*:2][*:4](=[*:5])[*:3]",
            },
            "amide": {
                "diamines_acids": "[#6;!$(C=O):0][NH2;!$([NH2+]):1].[#6:2][#6:3](=[O:4])([#8H,O-:5])>>"
                "[#6:0][NH:1][#6:3](=[O:4])([#6:2])",
                "diacids_amines": "[#6:2][#6:3](=[O:4])([#8H,O-:5]).[#6;!$(C=O):0][NH2;!$([NH2+]):1]>>"
                "[#6:0][NH:1][#6:3](=[O:4])([#6:2])",
                "infinite_chain": "([#6;!$(C=O):1][NH2;!$([NH2+]):2].[#6:3][#6:4](=[O:5])([#8H,O-:6]))>>"
                "[*:1][*:2][*:4](=[*:5])[*:3]",
            },
            "carbonate": {
                "phosgene": {
                    "diols_carbonates": "[C,c;!$(C=O):0][OH:1].[O:2]=[C:3]([F,Cl,Br,I,O:4])([F,Cl,Br,I:5])>>"
                    "[O:2]=[C:3]([O:1][C,c:0])[X:4]",
                    "carbonates_diols": "[O:2]=[C:3]([F,Cl,Br,I,O:4])([F,Cl,Br,I:5]).[C,c;!$(C=O):0][OH:1]>>"
                    "[O:2]=[C:3]([O:1][C,c:0])[X:4]",
                    "infinite_chain": "([C,c;!$(C=O):0][OH:1].[O:2]=[C:3]([F,Cl,Br,I,O:4])([F,Cl,Br,I:5]))>>"
                    "[O:2]=[C:3]([O:4])([O:1][C,c:0])",
                },
                "nonphosgene": {
                    "diols_carbonates": "[C,c;!$(C=O):0][OH:1].[O:2]=[C:3]([O:4][C,c:6])([O:5][C,c])>>"
                    "[O:2]=[C:3]([O:1][C,c:0])[O:4][C,c:6]",
                    "carbonates_diols": "[O:2]=[C:3]([O:4][C,c:6])([O:5][C,c]).[C,c;!$(C=O):0][OH:1]>>"
                    "[O:2]=[C:3]([O:1][C,c:0])[O:4][C,c:6]",
                    "infinite_chain": "([C,c;!$(C=O):0][OH:1].[O:2]=[C:3]([O:4][C,c:6])([O:5][C,c]))>>"
                    "[O:2]=[C:3]([O:1][C,c:0])[O:4][C,c:6]",
                },
            },
            "imide": {
                "diacidanhydrides_amines": "[#8:3]([#6:4](=[#8:5]))([#6:6](=[#8:7])).[#6;!$(C=O):0][NH2:1]>>"
                "[#6:0][N:1]([#6:4](=[#8:5]))([#6:6](=[#8:7]))",
                "diamines_acidanhydrides": "[#6;!$(C=O):0][NH2:1].[#8:3]([#6:4](=[#8:5]))([#6:6](=[#8:7]))>>"
                "[#6:0][N:1]([#6:4](=[#8:5]))([#6:6](=[#8:7]))",
                "infinite_chain": "([#8:3]([#6:4](=[#8:5]))([#6:6](=[#8:7])).[#6;!$(C=O):0][NH2:1])>>"
                "[#6:0][N:1]([#6:4](=[#8:5]))([#6:6](=[#8:7]))",
            },
            "open_acidanhydrides": {
                "add_OH": "[#8:3]([#6:4](=[#8:5]))([#6:6](=[#8:7]))>>"
                "[#8:3]([#6:4](=[#8:5])(O))([#6:6](=[#8:7]))"
            },
            "NIPU": {
                "dicycliccarbonates_amine": "[O:0]=[C:1]1[O:2][C:3][C:4][O:5]1.[NH2;!$([NH2+]):6][#6;!$(C=O):7]>>"
                "[C:7][NH:6][C:1](=[O:0])[O:2][C:3][C:4][O:5]",
                "diamine_cycliccarbonate": "[NH2;!$([NH2+]):6][#6;!$(C=O):7].[O:0]=[C:1]1[O:2][C:3][C:4][O:5]1>>"
                "[C:7][NH:6][C:1](=[O:0])[O:2][C:3][C:4][O:5]",
                "infinite_chain": "to complete",
            },
            "urethane": {
                "diisocyanates_ol": "[#6:0][#7;!$([#7+]):1]=[#6:2]=[#8:3].[C,c;!$(C=O):4][OH:5]>>"
                "[#6:0][#7;!$([#7+]):1]-[#6:2](=[#8:3])[O:5][C;!$(C=O):4]",
                "diols_isocyanate": "[C,c;!$(C=O):4][OH:5].[#6:0][#7;!$([#7+]):1]=[#6:2]=[#8:3]>>"
                "[#6:0][#7;!$([#7+]):1]-[#6:2](=[#8:3])[O:5][C;!$(C=O):4]",
                "infinite_chain": "to complete",
            },
        }
        self.__verison__ = "0.1.7.1"

    @staticmethod
    def checksmile(smi: str) -> Union[str, None]:
        """Checks if a smiles string creates a valid RDKit object and returns the canonical smiles if valid.

        Parameters
        ----------
        smi : str
            Compound SMILES string.

        Returns
        -------
        Union[str, None]
            Canonical SMILES string if valid or None
        """
        rdBase.DisableLog("rdApp.error")
        try:
            smi = Chem.CanonSmiles(smi)
        except:
            smi = ""
        rdBase.EnableLog("rdApp.error")

        return smi

    @staticmethod
    def get_monomers(smi: str, stereochemistry: bool = True) -> List[str]:
        """Convert a string of monomers into a list of monomers.

        The input string must contain monomer SMILES seperated by ".". Each monomer is checked for validity.

        Parameters
        ----------
        smi : str
            SMILES strings seperated by "." if multiple SMILES.
        stereochemistry : bool, optional
            Whether or not to retain stereochemistry in SMILES strings, by default False

        Returns
        -------
        List[str]
            A list of monomer strings.
        """
        try:
            smi = ast.literal_eval(smi)
        except:
            pass

        # Determine whether or no enantiomers are present
        if ".." not in smi:
            if type(smi) == str:
                smi = smi.split(".")
                if not stereochemistry:
                    smi = [smi_i.replace("/", "").replace("@", "") for smi_i in smi]
                monomers = tuple([PolyMaker.checksmile(smi_i) for smi_i in smi])
            elif type(smi) == tuple:
                monomers = smi
            else:
                monomers = None
        else:
            # assume for now there are monomers with R and S and specified like
            # M1R.M1S..M2R.M2S
            monomers = [
                [PolyMaker.checksmile(enant) for enant in enants.split(".")]
                for enants in smi.split("..")
                if enants
            ]

        # Check to see if there are any bad smiles in flattened list
        if any([item == "" for sublist in monomers for item in sublist]):
            monomers == None

        return monomers

    def thermoset(
        self,
        reactants,
        mechanism,
        crosslinker=[],
        distribution=[],
        DP=10,
        replicate_structures=1,
        verbose=True,
    ):
        """Inputs:
            reactants: contains smiles strings for reactants used in the polymer for both backbone and crosslinks
                a tuple
                or a strings of monomers
                or a pandas dataframe containing a list of monomers as strings with column title 'monomers'

            crosslinker: a list of 0's and 1's
                each value will correspond to the mononmers in reactants
                0's will indicate the corresponding monomer is part of the backbone
                1's will indicate the corresponding monomer is part of the crosslink

                a list of integers
                or a column in dataframe that is named 'crosslinker'

                example: [0,0,0,1]

            distribution: number of mols for each monomer in the reaction. values should be in samer order as reactancts
                list of floats
                or column in dataframe that is named 'mols'

                example: [10,10,3,1]

            DP:  degree of polymerization which is the number of monomer units in the polymer
                an integer, if an integer the same DP will be used for the backbone and the crosslinks
                a tuple, will contain only 2 values, the first value will be for the backbone and the second
                    for the crosslinks

            mechanism: one of the following strings,
                upe: unsaturated polyester, backbone will be a polyester with unsaturated bonds, crosslinks will be vinyls, olefins, acrylates

            replicate_structures: integer, number of replicate structures which will be generated


        Returns:
            polymer: string

        #"""

        returnpoly = pd.DataFrame()

        # converts monomers to tuple if reactants is dataframee
        if type(reactants) == pd.DataFrame:
            try:
                reactants.loc[:, "monomers"] = reactants.apply(
                    lambda row: self.get_monomers(row.monomers), axis=1
                )
            except:
                pass

        for rep in range(0, replicate_structures):
            returnpoly_i = pd.DataFrame()

            # reactants,crosslinks,etc should be a tuple but as a string going into polymerization methods
            # this puts everthing into dataframe before generating structures

            # fixing reactants and build dataframe
            if type(reactants) == pd.DataFrame:
                returnpoly_i = reactants
                if "mechanism" not in reactants.columns:
                    returnpoly_i.loc[:, "mechanism"] = mechanism
                returnpoly_i.loc[:, "replicate_structure"] = rep
                returnpoly_i.loc[:, "monomers"] = returnpoly_i.monomers.astype(str)
                returnpoly_i.loc[:, "mechanism"] = mechanism
            elif type(reactants) == str:
                try:
                    reactants_i = ast.literal_eval(reactants)
                except:
                    reactants_i = self.get_monomers(reactants)
                returnpoly_i.loc[:, "monomers"] = pd.Series(str(reactants_i))
                returnpoly_i.loc[:, "distribution"] = pd.Series(str(distribution))
                returnpoly_i.loc[:, "crosslinker"] = pd.Series(str(crosslinker))

                returnpoly_i.loc[:, "replicate_structure"] = rep
                returnpoly_i.loc[:, "monomers"] = returnpoly_i.monomers.astype(str)
                returnpoly_i.loc[:, "mechanism"] = mechanism

            elif type(reactants) == tuple:
                returnpoly_i.loc[:, "monomers"] = pd.Series(str(reactants))
                returnpoly_i.loc[:, "distribution"] = pd.Series(str(distribution))
                returnpoly_i.loc[:, "crosslinker"] = pd.Series(str(crosslinker))

                returnpoly_i.loc[:, "replicate_structure"] = rep
                returnpoly_i.loc[:, "monomers"] = returnpoly_i.monomers.astype(str)
                returnpoly_i.loc[:, "mechanism"] = mechanism

            else:
                raise ValueError("Data type not recognized")

            # building dataframe
            returnpoly = pd.concat([returnpoly, returnpoly_i])
        # build polymers
        if verbose:
            returnpoly[["smiles_polymer", "mechanism"]] = returnpoly.progress_apply(
                lambda row: self.__polymerizemechanism_thermoset(
                    leval(row.monomers),
                    row.mechanism,
                    leval(row.crosslinker),
                    leval(row.distribution),
                    DP,
                ),
                axis=1,
            )
        else:
            returnpoly[["smiles_polymer", "mechanism"]] = returnpoly.apply(
                lambda row: self.__polymerizemechanism_thermoset(
                    leval(row.monomers),
                    row.mechanism,
                    leval(row.crosslinker),
                    leval(row.distribution),
                    DP,
                ),
                axis=1,
            )
        returnpoly = returnpoly.sort_index().sort_values("replicate_structure")

        # BUILD STRUCTURE

        return returnpoly

    def thermoplastic_stereo(
        self,
        reactants: pd.DataFrame,
        DP: int = 2,
        replicate_structures: int = 1,
        distribution: Union[List[float], List] = [],
        pm: float = 1,
        verbose: bool = True,
    ):
        """thermoplastic_stereo generates stereochemical structures using the non-stereo thermoplastic function

        Parameters
        ----------
        reactants : pd.DataFrame
            A pandas dataframe containing a smiles column, a monomers column (generated by get_monomers),
            a mechanism column, an optional pm column, and an optinional distribution column
        DP : int, optional
            Degree of polymerization, by default 2
            by default ''
        replicate_structures : int, optional
            Number of replicate structures per monomer entry, by default 1
        distribution : list, optional
            Distribution of copolymers, by default []
        pm : int, optional
            Default pm value to use if none is provided in reactants dataframe, by default 1
        verbose : bool, optional
            Whether or not to print polymerization progress, by default True

        Returns
        -------
        pd.DataFrame
            A dataframe containing the polymerized inputs
        """
        # Ensure there is a Pm value, default is 1
        if "pm" not in reactants:
            reactants["pm"] = pm
        reactants[["pm"]] = reactants[["pm"]].fillna(value=pm)

        if "distribution" in reactants:
            reactants["distribution"] = reactants["distribution"].fillna(
                str(distribution)
            )
            reactants["distribution"] = reactants["distribution"].astype(str)
        else:
            reactants.loc[:, "distribution"] = str(distribution)

        if "replicate_structures" in reactants:
            reactants["replicate_structures"] = reactants[
                "replicate_structures"
            ].fillna(int(replicate_structures))
        else:
            reactants["replicate_structures"] = int(replicate_structures)

        reactants["DP"] = DP

        df_return_mapped = reactants

        if verbose:
            df_return_mapped = df_return_mapped.progress_apply(
                self._polymerizemechanism_thermoplastic_stereo,
                axis=1,
            )
        else:
            df_return_mapped = df_return_mapped.apply(
                self._polymerizemechanism_thermoplastic_stereo,
                axis=1,
            )

        if not df_return_mapped.empty:
            df_return_mapped = pd.concat([res for res in df_return_mapped]).reset_index(
                drop=True
            )

            # if not df_return_mapped.empty:
            #     df_return_mapped.sort_index().sort_values("replicate_structure")

        if "index" in df_return_mapped.columns:
            df_return_mapped = df_return_mapped.drop(columns=["index"])
        if "replicate_structures" in df_return_mapped.columns:
            df_return_mapped = df_return_mapped.drop(columns=["replicate_structures"])
        df_return_mapped = df_return_mapped.reset_index(drop=True)

        return df_return_mapped

    def _polymerizemechanism_thermoplastic_stereo(self, row):
        df_poly = pd.DataFrame()
        if (row.mechanism == "vinyl_stereo") | (row.mechanism == "acrylate_stereo"):
            df_poly = polyvinyl_stereo(
                row.monomers,
                row.DP,
                row.replicate_structures,
                row.distribution,
                row.pm,
            )
        elif row.mechanism == "ester_stereo":
            df_poly = polyester_stereo(
                row.monomers,
                row.DP,
                row.replicate_structures,
                row.distribution,
                row.pm,
            )

        if isinstance(df_poly, str) or df_poly.empty:
            # TODO: how to integrate
            return pd.DataFrame()
        else:
            df_poly = pd.merge(df_poly, row.to_frame().T, how="cross")
            df_poly["replicate_structure"] = [
                i for i in range(row.replicate_structures)
            ]
            return df_poly

    def thermoplastic(
        self,
        reactants,
        DP=2,
        mechanism="",
        replicate_structures=1,
        distribution=[],
        infinite_chain=False,
        verbose=True,
    ):

        """Polymerization method for building thermoplastics structures

        Parameters
        ----------
        reactants: a tuple
                   or a strings of monomers
                   or a pandas dataframe containing a column titled "monomers" (generated by get_monomers)
                   optional column "distribution" containing list of molar ratio that maps to monomers in "monomers" column
                   optional column mechanism containing desired mechanism to use for the row

        DP: int, optional,
            Degree of polymerization, by default 2

        mechanism: string,
            vinyl: performs addition reaction vinyl or acrylate groups
            ester: performs condensation reaction on diacid + diol or  hydroxyacid
            carbonate: performs condensation reaction on carbonate + diol
            amide: performs condensation reaction on diacid + diamine or aminoacid
            urethane: performs condensation reaction on diisocyanate + diamine
            NIPU: performs condensation reaction multi-functional cyclic carbonate + diamine
            imide: performs condensation reaction on diacidanhydride and diamine

            all: will perform all mechanims, may cause multiple reaction mechanisms if not specified

        replicate_structures: integer, optional
            number of replicate structures which will be generated

        Returns
        -------
            A dataframe containing the polymerized inputs in column titled "smiles_polymer"

        """
        returnpoly = pd.DataFrame()

        for rep in range(0, replicate_structures):
            returnpoly_i = pd.DataFrame()

            # reactants should be a tuple but as a string going into polymerization methods
            # this puts everthing into dataframe before generating structures
            if type(reactants) == str:
                try:
                    reactants_i = ast.literal_eval(reactants)
                except:
                    reactants_i = self.get_monomers(reactants)
                returnpoly_i.loc[:, "monomers"] = pd.Series(str(reactants_i))
                returnpoly_i.loc[:, "distribution"] = str(distribution)
            elif type(reactants) == tuple:
                returnpoly_i.loc[:, "monomers"] = pd.Series(str(reactants))
                returnpoly_i.loc[:, "distribution"] = str(distribution)
            elif type(reactants) == pd.DataFrame:
                returnpoly_i = reactants

                if "distribution" in returnpoly_i:
                    returnpoly_i["distribution"] = returnpoly_i["distribution"].fillna(
                        str(distribution)
                    )
                    returnpoly_i["distribution"] = returnpoly_i["distribution"].astype(
                        str
                    )
                else:
                    returnpoly_i.loc[:, "distribution"] = str(distribution)

            else:
                raise ValueError("Data type not recognized")
            returnpoly_i.loc[:, "replicate_structure"] = rep
            returnpoly_i.loc[:, "monomers"] = returnpoly_i.monomers.astype(str)
            returnpoly = pd.concat([returnpoly, returnpoly_i])

        if verbose:
            apply_method = returnpoly.progress_apply
        else:
            apply_method = returnpoly.apply

        if (returnpoly.keys() == "mechanism").any() & ((mechanism == "")):
            returnpoly[["smiles_polymer", "mechanism"]] = apply_method(
                lambda row: self.__polymerizemechanism_thermoplastic(
                    ast.literal_eval(row.monomers),
                    DP,
                    row.mechanism,
                    ast.literal_eval(row.distribution),
                    infinite_chain,
                ),
                axis=1,
            )
        else:
            returnpoly[["smiles_polymer", "mechanism"]] = apply_method(
                lambda row: self.__polymerizemechanism_thermoplastic(
                    ast.literal_eval(row.monomers),
                    DP,
                    mechanism,
                    ast.literal_eval(row.distribution),
                    infinite_chain,
                ),
                axis=1,
            )

        returnpoly = returnpoly.sort_index().sort_values("replicate_structure")
        return returnpoly

    def get_functionality(self, reactants, distribution=[], verbose=0):
        """gets the functional groups from a list of reactants

        inputs: list of smiles
        output: dataframe with count of functional groups
        """

        def id_functionality(r):
            mol = Chem.MolFromSmiles(r.name)
            for k, v in self.smiles_templates.items():
                r[k] = len(mol.GetSubstructMatches(Chem.MolFromSmarts(v)))
            return r

        df_func = pd.DataFrame(
            data=0,
            index=reactants,
            columns=self.smiles_templates.keys(),
        )
        if verbose == 0:
            df_func = df_func.apply(lambda r: id_functionality(r), axis=1)
        if verbose >= 1:
            df_func = df_func.progress_apply(lambda r: id_functionality(r), axis=1)

        # appends distribution to dataframe

        if len(distribution) == 0:
            df_func["distribution"] = [1] * df_func.shape[0]
        else:
            df_func["distribution"] = list(distribution)
        return df_func

    def _returnvalid(self, prodlist):
        """verifies list of molecule smiles is valid

        Input: list of strings

        Return: list of strings
        """
        returnlist = []
        rdBase.DisableLog("rdApp.error")
        for x in prodlist:
            try:
                Chem.SanitizeMol(Chem.MolFromSmiles(x))
                returnlist.append(x)
            except:
                pass
        rdBase.EnableLog("rdApp.error")
        return returnlist

    def __get_distributed_reactants(self, reactants, distribution=[]):

        if len(distribution) != 0:
            distribution = self.__integerize_distribution(distribution)
            smiles_list = []
            for reactant, mol in zip(reactants, distribution):
                smiles_list = smiles_list + [reactant] * mol
            return_reactants = self.get_monomers(".".join(smiles_list))
        else:
            return_reactants = reactants
        return return_reactants

    def __integerize_distribution(self, distribution):
        numdecimals = max([str(d)[::-1].find(".") for d in distribution])
        if numdecimals == -1:
            numdecimals = 0

        distribution = [int(d * 10**numdecimals) for d in distribution]

        try:
            distribution = distribution / np.gcd.reduce(distribution)
        except:
            pass

        return [int(d) for d in distribution]

    def __polymerizemechanism_thermoplastic(
        self, reactants, DP, mechanism, distribution=[], infinite_chain=False, rep=None
    ):
        """directs polymerization to correct method for mechanism"""

        returnpoly = ""
        # reactants = self.__get_distributed_reactants(reactants,distribution=distribution)

        if (mechanism == "vinyl") | (mechanism == "acrylate"):
            polydata = self.__poly_vinyl(reactants, DP, distribution, infinite_chain)
            returnpoly = polydata[0]
            mechanism = polydata[1]

        elif mechanism == "ester":
            polydata = self.__poly_ester(reactants, DP, distribution, infinite_chain)
            returnpoly = polydata[0]
            mechanism = polydata[1]

        elif mechanism == "amide":
            polydata = self.__poly_amide(reactants, DP, distribution, infinite_chain)
            returnpoly = polydata[0]
            mechanism = polydata[1]

        elif mechanism == "carbonate":
            polydata = self.__poly_carbonate(
                reactants, DP, distribution, infinite_chain
            )
            returnpoly = polydata[0]
            mechanism = polydata[1]

        elif mechanism == "imide":
            polydata = self.__poly_imide(reactants, DP, distribution, infinite_chain)
            returnpoly = polydata[0]
            mechanism = polydata[1]

        elif mechanism == "NIPU":
            polydata = self.__poly_NIPU(reactants, DP, distribution, infinite_chain)
            returnpoly = polydata[0]
            mechanism = polydata[1]

        elif mechanism == "urethane":
            polydata = self.__poly_urethane(reactants, DP, distribution, infinite_chain)
            returnpoly = polydata[0]
            mechanism = polydata[1]

        elif mechanism == "all":
            polylist = [
                self.__poly_vinyl(reactants, DP, distribution, infinite_chain),
                self.__poly_ester(reactants, DP, distribution, infinite_chain),
                self.__poly_amide(reactants, DP, distribution, infinite_chain),
                self.__poly_carbonate(reactants, DP, distribution, infinite_chain),
                self.__poly_imide(reactants, DP, distribution, infinite_chain),
                self.__poly_NIPU(reactants, DP, distribution, infinite_chain),
                self.__poly_urethane(reactants, DP, distribution, infinite_chain),
            ]

            polylist = [
                p
                for p in polylist
                if p[0]
                not in [
                    "ERROR:Vinyl_ReactionFailed",
                    "ERROR:Ester_ReactionFailed",
                    "ERROR:Amide_ReactionFailed",
                    "ERROR:Carbonate_ReactionFailed",
                    "ERROR:Imide_ReactionFailed",
                    "ERROR:NIPU_ReactionFailed",
                    "ERROR:Urethane_ReactionFailed",
                    "",
                ]
            ]
            if len(polylist) == 1:
                returnpoly = polylist[0][0]
                mechanism = polylist[0][1]
            elif len(polylist) > 1:
                returnpoly = "ERROR_02:MultiplePolymerizations"
            else:
                returnpoly = "ERROR_01:NoReaction"
        else:
            returnpoly = "ERROR_03:MechanismNotRecognized"

        return pd.Series([returnpoly, mechanism])

    def __polymerizemechanism_thermoset(
        self, reactants, mechanism, crosslinker, distribution, DP
    ):
        """directs polymerization to correct method for mechanism"""
        returnpoly = ""

        if mechanism == "UPE":
            polydata = self.__poly_upe(reactants, crosslinker, distribution, DP)
            returnpoly = polydata[0]
            mechanism = polydata[1]

        else:
            returnpoly = "ERROR_03:MechanismNotRecognized"

        return pd.Series([returnpoly, mechanism])

    def __poly_vinyl_init(self, mola, molb):
        """performs propagation rxn of vinyl polymer"""

        # rxn definition
        rxn = AllChem.ReactionFromSmarts(
            "[C:1]=[C:2].[C:3]=[C:4]>>[Kr][C:1][C:2][C:3][C:4][Xe]"
        )

        # product creation and validation
        prod = rxn.RunReactants((mola, molb))
        prodlist = [Chem.MolToSmiles(x[0]) for x in prod]
        molprodlist = [Chem.MolFromSmiles(p) for p in self._returnvalid(prodlist)]
        return molprodlist

    def __poly_vinyl_prop(self, mola, molb):
        """performs propagation rxn of vinyl polymer"""

        # rxn definition
        rxn = AllChem.ReactionFromSmarts(
            "[C:0][C:1][C:2][C:3][Xe].[C:4]=[C:5]>>[C:0][C:1][C:2][C:3][C:4][C:5][Xe]"
        )

        # product creation and validation
        prod = rxn.RunReactants((mola, molb))
        prodlist = [Chem.MolToSmiles(x[0]) for x in prod]
        molprodlist = [Chem.MolFromSmiles(p) for p in self._returnvalid(prodlist)]
        return molprodlist

    def __poly_vinyl_term(self, mola, molb, infinite_chain=False, single_rxn=False):
        """performs termination rxn of vinyl polymer"""

        # rxn definition
        if single_rxn:
            rxn1 = AllChem.ReactionFromSmarts(
                "[C:0]=[C:1].[C:2]=[C:3]>>[C:0][C:1][C:2][C:3]"
            )
            prod = rxn1.RunReactants((mola, molb))

        elif infinite_chain:
            # terminates and removes Xe
            rxn1 = AllChem.ReactionFromSmarts(
                "[C:0][C:1][C:2][C:3][Xe].[C:4]=[C:5]>>[C:0][C:1][C:2][C:3][C:4][C:5][Xe]"
            )
            prod = rxn1.RunReactants((mola, molb))
            # ring closes
            rxn2 = AllChem.ReactionFromSmarts(
                "([Kr][C:0][C:1].[C:2][C:3][Xe])>>[C:1][C:0][C:3][C:2]"
            )
            prod = [rxn2.RunReactants((r,)) for r in list(itertools.chain(*prod))]
            prod = list(itertools.chain(*prod))
        else:
            # terminates and removes Xe
            rxn1 = AllChem.ReactionFromSmarts(
                "[C:0][C:1][C:2][C:3][Xe].[C:4]=[C:5]>>[C:0][C:1][C:2][C:3][C:4][C:5]"
            )
            prod = rxn1.RunReactants((mola, molb))
            # removes Kr
            rxn2 = AllChem.ReactionFromSmarts(
                "[C:0][C:1][C:2][C:3][Kr]>>[C:0][C:1][C:2][C:3]"
            )
            prod = [rxn2.RunReactants((r,)) for r in list(itertools.chain(*prod))]
            prod = list(itertools.chain(*prod))

        # preps for return
        prod = list(itertools.chain(*prod))
        prodlist = [Chem.MolToSmiles(p) for p in prod]
        molprodlist = [Chem.MolFromSmiles(p) for p in self._returnvalid(prodlist)]
        return molprodlist

    def __poly_vinyl(
        self, reactants, DP=3, distribution=[], infinite_chain=False, crosslink=False
    ):
        """performs vinyl polymerization"""
        try:

            if len(distribution) == 0:
                distribution = [1] * len(reactants)
            # mol conversion and parsing
            mols = [Chem.MolFromSmiles(r) for r in reactants]

            if crosslink:
                distribution = [1, 1] + list(distribution)
                dfreactants = pd.DataFrame(
                    data=[reactants, mols, distribution],
                    index=["reactants", "mols", "distribution"],
                ).T

                dfmolA = pd.DataFrame(dfreactants.iloc[0]).T
                dfmolB = pd.DataFrame(dfreactants.iloc[1]).T
                dfreactants = dfreactants.iloc[2:]
            else:
                dfreactants = pd.DataFrame(
                    data=[reactants, mols, distribution],
                    index=["reactants", "mols", "distribution"],
                ).T
                dfmolA = dfreactants
                dfmolB = dfreactants
                dfreactants = dfreactants

            # polymerization
            assert DP > 1

            if DP > 2:

                # initiate
                molA = (
                    dfmolA.sample(1, weights=dfmolA.distribution, replace=True)
                    .iloc[0]
                    .loc["mols"]
                )
                mol = (
                    dfreactants.sample(
                        1, weights=dfreactants.distribution, replace=True
                    )
                    .iloc[0]
                    .loc["mols"]
                )
                polymer = self.__poly_vinyl_init(molA, mol)

                # propagate
                for r in range(0, DP - 3):
                    assert len(polymer) >= 1
                    polymer = random.choice(polymer)
                    mol = (
                        dfreactants.sample(
                            1, weights=dfreactants.distribution, replace=True
                        )
                        .iloc[0]
                        .loc["mols"]
                    )
                    polymer = self.__poly_vinyl_prop(polymer, mol)

                # terminate

                polymer = random.choice(polymer)
                molB = (
                    dfmolB.sample(1, weights=dfmolB.distribution, replace=True)
                    .iloc[0]
                    .loc["mols"]
                )
                polymer = self.__poly_vinyl_term(polymer, molB, infinite_chain)

            if DP == 2:
                molA = (
                    dfmolA.sample(1, weights=dfmolA.distribution, replace=True)
                    .iloc[0]
                    .loc["mols"]
                )
                molB = (
                    dfmolB.sample(1, weights=dfmolB.distribution, replace=True)
                    .iloc[0]
                    .loc["mols"]
                )
                polymer = self.__poly_vinyl_term(molA, molB, single_rxn=True)

            polymer = Chem.MolToSmiles(random.choice(polymer))
        except:
            polymer = "ERROR:Vinyl_ReactionFailed"
        return polymer, "vinyl"

    def __protect_substructure(self, mol, substructure, n_unprotected=0):
        """protects atoms in the group identified

        mol: rdkit mol object
        substructure: SMARTS string to match to
        n_uprotected: number of substructures that will not be protected"""
        if type(mol) == str:
            mol = Chem.MolFromSmiles(mol)

        mol = deepcopy(mol)
        protect = list(mol.GetSubstructMatches(Chem.MolFromSmarts(substructure)))
        random.shuffle(protect)

        protect = protect[n_unprotected:]
        protect = list(itertools.chain(*protect))

        for atom in mol.GetAtoms():
            if atom.GetIdx() in protect:
                atom.SetProp("_protected", "1")
            else:
                pass
        return [mol, len(protect)]

    def __unprotect_atoms(self, mol):
        """unprotects all atoms in molecule"""
        mol = deepcopy(mol)
        for atom in mol.GetAtoms():
            try:
                atom.ClearProp("_protected")
            except:
                pass
        return mol

    def __poly_ester(self, reactants, DP=2, distribution=[], infinite_chain=False):
        """performs condenstation reaction on dicarboxyl and  diols"""

        try:
            # open acid anhydrides
            def replace_acidanhydrides(reactant):
                mol = Chem.MolFromSmiles(reactant)
                if (
                    len(
                        mol.GetSubstructMatches(
                            Chem.MolFromSmarts(self.smiles_templates["acidanhydrides"])
                        )
                    )
                    > 0
                ):
                    reactant = self.__openacidanyhydride(reactant)
                else:
                    pass
                return reactant

            reactants = pd.Series(reactants).apply(replace_acidanhydrides).tolist()

            rxn_dic = self.reactions["ester"]
            df_func = self.get_functionality(reactants, distribution=distribution)

            # select initial monomer as polymer chain
            df_poly = df_func.sample(1)
            df_func.loc["smiles_polymer"] = df_poly.sample(1).values[0]
            poly = df_poly.index[0]
            molpoly = Chem.MolFromSmiles(poly)

            DP_count = 1
            DP_actual = 1

            while DP_count < DP:

                # select rxn rule and reactant
                if (df_func.loc["smiles_polymer", "aliphatic_ols"] >= 1) & (
                    df_func.loc["smiles_polymer", "acids"] >= 1
                ):
                    msk = ((df_func.acids >= 1) | (df_func.aliphatic_ols >= 1)) & (
                        df_func.index != "smiles_polymer"
                    )
                    df_func_select = df_func.loc[msk]
                    a = df_func_select.sample(
                        1, weights=df_func.distribution, replace=True
                    ).index.values[0]
                    if df_func.loc[a].aliphatic_ols >= 1:
                        rxn_selector = "diacids_ols"
                    if df_func.loc[a].acids >= 1:
                        rxn_selector = "diols_acids"
                elif df_func.loc["smiles_polymer", "aliphatic_ols"] >= 2:
                    msk = (df_func.acids >= 1) & (df_func.index != "smiles_polymer")
                    df_func_select = df_func.loc[msk]
                    a = df_func_select.sample(
                        1, weights=df_func.distribution, replace=True
                    ).index.values[0]
                    rxn_selector = "diols_acids"
                elif df_func.loc["smiles_polymer", "acids"] >= 2:
                    msk = (df_func.aliphatic_ols >= 1) & (
                        df_func.index != "smiles_polymer"
                    )
                    df_func_select = df_func.loc[msk]
                    a = df_func_select.sample(
                        1, weights=df_func.distribution, replace=True
                    ).index.values[0]
                    rxn_selector = "diacids_ols"
                else:
                    assert False
                rxn = Chem.AllChem.ReactionFromSmarts(rxn_dic[rxn_selector])

                # update df_func table
                df_func.loc["smiles_polymer"] = (
                    df_func.loc["smiles_polymer"] + df_func.loc[a]
                )  # adding polymer and a
                for column_name in ["aliphatic_ols", "ols", "acids"]:
                    df_func.loc[
                        "smiles_polymer", column_name
                    ] += -1  # substracting off functionality
                assert (
                    df_func.loc["smiles_polymer"][
                        df_func.loc["smiles_polymer"] > -1
                    ].shape
                    == df_func.loc["smiles_polymer"].shape
                )

                # React and select product
                mola = Chem.MolFromSmiles(a)
                prod = rxn.RunReactants((molpoly, mola))
                prodlist = [Chem.MolToSmiles(x[0]) for x in prod]
                prodlist = self._returnvalid(prodlist)
                poly = random.choice(prodlist)
                molpoly = Chem.MolFromSmiles(poly)

                # manage loop and ring close
                if (infinite_chain) & (DP_count == DP - 1):
                    # logic for closing ring
                    if (df_func.loc["smiles_polymer", "aliphatic_ols"] > 0) & (
                        df_func.loc["smiles_polymer", "acids"]
                    ) > 0:
                        # case for when has can ring close
                        DP_count += 1
                        DP_actual += 1
                    else:
                        # case for when has same terminal ends so can't ring close
                        DP_count = DP_count
                        DP_actual += 1
                else:
                    DP_count += 1
                    DP_actual += 1

            if infinite_chain:  # closes ring
                rxn = Chem.AllChem.ReactionFromSmarts(rxn_dic["infinite_chain"])
                prod = rxn.RunReactants((molpoly,))
                prodlist = [Chem.MolToSmiles(x[0]) for x in prod]
                prodlist = self._returnvalid(prodlist)
                poly = random.choice(prodlist)
                molpoly = Chem.MolFromSmiles(poly)

        except:
            poly = "ERROR:Ester_ReactionFailed"

        return poly, "ester"

    def __poly_amide(self, reactants, DP=2, distribution=[], infinite_chain=False):
        """performs condenstation reaction on dicarboxyl and  diols"""
        # function

        try:
            # 	initial
            rxn_dic = self.reactions["amide"]
            df_func = self.get_functionality(reactants, distribution=distribution)

            # select initial monomer as polymer chain
            df_poly = df_func.sample(1)
            df_func.loc["smiles_polymer"] = df_poly.sample(1).values[0]
            poly = df_poly.index[0]
            molpoly = Chem.MolFromSmiles(poly)

            DP_count = 1
            DP_actual = 1

            while DP_count < DP:

                # select rxn rule and reactant
                if (df_func.loc["smiles_polymer", "prime_amines"] >= 1) & (
                    df_func.loc["smiles_polymer", "acids"] >= 1
                ):
                    msk = ((df_func.acids >= 1) | (df_func.prime_amines >= 1)) & (
                        df_func.index != "smiles_polymer"
                    )
                    df_func_select = df_func.loc[msk]
                    a = df_func_select.sample(
                        1, weights=df_func.distribution, replace=True
                    ).index.values[0]
                    if df_func.loc[a].prime_amines >= 1:
                        rxn_selector = "diacids_amines"
                    if df_func.loc[a].acids >= 1:
                        rxn_selector = "diamines_acids"
                elif df_func.loc["smiles_polymer", "prime_amines"] >= 2:
                    msk = (df_func.acids >= 1) & (df_func.index != "smiles_polymer")
                    df_func_select = df_func.loc[msk]
                    a = df_func_select.sample(
                        1, weights=df_func.distribution, replace=True
                    ).index.values[0]
                    rxn_selector = "diamines_acids"
                elif df_func.loc["smiles_polymer", "acids"] >= 2:
                    msk = (df_func.prime_amines >= 1) & (
                        df_func.index != "smiles_polymer"
                    )
                    df_func_select = df_func.loc[msk]
                    a = df_func_select.sample(
                        1, weights=df_func.distribution, replace=True
                    ).index.values[0]
                    rxn_selector = "diacids_amines"
                else:
                    assert False

                rxn = Chem.AllChem.ReactionFromSmarts(rxn_dic[rxn_selector])

                # update df_func table
                df_func.loc["smiles_polymer"] = (
                    df_func.loc["smiles_polymer"] + df_func.loc[a]
                )  # adding polymer and a
                for column_name in ["prime_amines", "acids"]:
                    df_func.loc["smiles_polymer", column_name] += -1
                assert (
                    df_func.loc["smiles_polymer"][
                        df_func.loc["smiles_polymer"] > -1
                    ].shape
                    == df_func.loc["smiles_polymer"].shape
                )

                # React and select product
                mola = Chem.MolFromSmiles(a)
                prod = rxn.RunReactants((molpoly, mola))
                prodlist = [Chem.MolToSmiles(x[0]) for x in prod]
                prodlist = self._returnvalid(prodlist)
                poly = random.choice(prodlist)
                molpoly = Chem.MolFromSmiles(poly)

                # manage loop and ring close
                if (infinite_chain) & (DP_count == DP - 1):
                    # logic for closing ring
                    if (df_func.loc["smiles_polymer", "prime_amines"] > 0) & (
                        df_func.loc["smiles_polymer", "acids"]
                    ) > 0:
                        # case for when can ring close
                        DP_count += 1
                        DP_actual += 1
                    else:
                        # case for when has same terminal ends so can't ring close
                        DP_count = DP_count
                        DP_actual += 1
                else:
                    DP_count += 1
                    DP_actual += 1

            if infinite_chain:  # closes ring

                rxn = Chem.AllChem.ReactionFromSmarts(rxn_dic["infinite_chain"])
                prod = rxn.RunReactants((molpoly,))
                prodlist = [Chem.MolToSmiles(x[0]) for x in prod]
                prodlist = self._returnvalid(prodlist)
                poly = random.choice(prodlist)
                molpoly = Chem.MolFromSmiles(poly)

        except:
            poly = "ERROR:Amide_ReactionFailed"
        return poly, "amide"

    def __poly_carbonate(self, reactants, DP=2, distribution=[], infinite_chain=False):
        def choose_carbonyltype(reactants):
            # this chooses the right rxn scheme depeneding on the carbonate monomer
            template_phosgene = "[O:2]=[C:3]([F,Cl,Br,I,O:4])([F,Cl,Br,I:5])"
            template_nonphosgene = "[O:2]=[C:3]([O:4][C,c:6])([O:5][C,c])"
            if np.any(
                [
                    len(
                        Chem.MolFromSmiles(r).GetSubstructMatch(
                            Chem.MolFromSmarts(template_phosgene)
                        )
                    )
                    for r in reactants
                ]
            ):
                carbonyltype = "phosgene"
            if np.any(
                [
                    len(
                        Chem.MolFromSmiles(r).GetSubstructMatch(
                            Chem.MolFromSmarts(template_nonphosgene)
                        )
                    )
                    for r in reactants
                ]
            ):
                carbonyltype = "nonphosgene"

            return carbonyltype

        def get_prods_matching_mw(
            molpoly, mola, prodlist, leavegroup_MW, infinite_chain=False
        ):
            returnlist = []

            if not infinite_chain:
                mwexpected = np.round(
                    Descriptors.MolWt(molpoly)
                    + Descriptors.MolWt(mola)
                    - leavegroup_MW,
                    2,
                )
            else:
                mwexpected = np.round(Descriptors.MolWt(molpoly) - leavegroup_MW, 2)

            for prod in prodlist:
                mprod = Chem.MolFromSmiles(prod)
                mwprod = np.round(Descriptors.MolWt(mprod), 2)
                if mwexpected - 0.1 < mwprod < mwexpected + 0.1:
                    returnlist.append(prod)
            return returnlist

        try:

            # initial
            carbonyltype = choose_carbonyltype(reactants)
            rxn_dic = self.reactions["carbonate"][carbonyltype]
            df_func = self.get_functionality(reactants, distribution=distribution)

            # select initial monomer as polymer chain
            df_poly = df_func.sample(1)
            df_func.loc["smiles_polymer"] = df_poly.sample(1).values[0]
            poly = df_poly.index[0]
            molpoly = Chem.MolFromSmiles(poly)

            DP_count = 1
            DP_actual = 1

            while DP_count < DP:
                # select rxn rule and reactant
                if (df_func.loc["smiles_polymer", "ols"] >= 1) & (
                    df_func.loc["smiles_polymer", "carbonates"] >= 0.5
                ):
                    msk = ((df_func.ols >= 1) | (df_func.carbonates >= 0.5)) & (
                        df_func.index != "smiles_polymer"
                    )
                    df_func_select = df_func.loc[msk]
                    a = df_func_select.sample(
                        1, weights=df_func.distribution, replace=True
                    ).index.values[0]
                    if np.all(df_func.loc[a].ols >= 1):
                        rxn_selector = "carbonates_diols"
                    if np.all(df_func.loc[a].carbonates >= 0.5):
                        rxn_selector = "diols_carbonates"
                elif df_func.loc["smiles_polymer", "ols"] >= 2:
                    msk = (df_func.carbonates >= 0.5) & (
                        df_func.index != "smiles_polymer"
                    )
                    df_func_select = df_func.loc[msk]
                    a = df_func_select.sample(
                        1, weights=df_func.distribution, replace=True
                    ).index.values[0]
                    rxn_selector = "diols_carbonates"
                elif df_func.loc["smiles_polymer", "carbonates"] >= 1:
                    msk = (df_func.ols >= 1) & (df_func.index != "smiles_polymer")
                    df_func_select = df_func.loc[msk]
                    a = df_func_select.sample(
                        1, weights=df_func.distribution, replace=True
                    ).index.values[0]
                    rxn_selector = "carbonates_diols"
                else:
                    assert False
                rxn = Chem.AllChem.ReactionFromSmarts(rxn_dic[rxn_selector])

                # update df_func table
                df_func.loc["smiles_polymer"] = (
                    df_func.loc["smiles_polymer"] + df_func.loc[a]
                )  # adding polymer and a
                for column_name, adder in zip(["ols", "carbonates"], [-1, -0.5]):
                    df_func.loc["smiles_polymer", column_name] += adder
                assert (
                    df_func.loc["smiles_polymer"][
                        df_func.loc["smiles_polymer"] > -1
                    ].shape
                    == df_func.loc["smiles_polymer"].shape
                )

                mola = Chem.MolFromSmiles(a)

                if (DP_count - 1 == 0) & (rxn_selector == "diols_carbonates"):
                    leavegroup_MW = (
                        Descriptors.MolWt(mola)
                        - Descriptors.MolWt(Chem.MolFromSmiles("C=O"))
                        + 4
                    ) / 2
                if (DP_count - 1 == 0) & (rxn_selector == "carbonates_diols"):
                    leavegroup_MW = (
                        Descriptors.MolWt(molpoly)
                        - Descriptors.MolWt(Chem.MolFromSmiles("C=O"))
                        + 4
                    ) / 2
                prods = rxn.RunReactants((molpoly, mola))
                allprodlist = [Chem.MolToSmiles(x[0]) for x in prods]
                prodlist = pd.Series(self._returnvalid(allprodlist)).unique().tolist()
                prodlist = get_prods_matching_mw(molpoly, mola, prodlist, leavegroup_MW)
                poly = random.choice(prodlist)
                molpoly = Chem.MolFromSmiles(poly)

                # manage loop and ring close
                if (infinite_chain) & (DP_count == DP - 1):
                    # logic for closing ring
                    if (df_func.loc["smiles_polymer", "ols"] > 0) & (
                        df_func.loc["smiles_polymer", "carbonates"] > 0
                    ):
                        # case for when has can ring close
                        DP_count += 1
                        DP_actual += 1
                    else:
                        # case for when has same terminal ends so can't ring close
                        DP_count = DP_count
                        DP_actual += 1
                else:
                    DP_count += 1
                    DP_actual += 1

            if infinite_chain:  # closes ring

                rxn = Chem.AllChem.ReactionFromSmarts(rxn_dic["infinite_chain"])
                prod = rxn.RunReactants((molpoly,))
                prodlist = [Chem.MolToSmiles(x[0]) for x in prod]
                prodlist = self._returnvalid(prodlist)
                prodlist = get_prods_matching_mw(
                    molpoly, mola, prodlist, leavegroup_MW, infinite_chain=True
                )
                poly = random.choice(prodlist)
                molpoly = Chem.MolFromSmiles(poly)

        except:
            poly = "ERROR:Carbonate_ReactionFailed"
        return poly, "carbonate"

    def __poly_imide(self, reactants, DP=2, distribution=[], infinite_chain=False):
        """performs condenstation reaction on dianhydride and  diamine"""
        # function

        try:

            # initial
            rxn_dic = self.reactions["imide"]
            df_func = self.get_functionality(reactants, distribution=distribution)

            # select initial monomer as polymer chain
            df_poly = df_func.sample(1)
            df_func.loc["smiles_polymer"] = df_poly.sample(1).values[0]
            poly = df_poly.index[0]
            molpoly = Chem.MolFromSmiles(poly)

            DP_count = 1
            DP_actual = 1
            while DP_count < DP:

                # select rxn rule and reactant
                if (df_func.loc["smiles_polymer", "prime_amines"] >= 1) & (
                    df_func.loc["smiles_polymer", "acidanhydrides"] >= 1
                ):
                    msk = ((df_func.acids >= 1) | (df_func.prime_amines >= 1)) & (
                        df_func.index != "smiles_polymer"
                    )
                    df_func_select = df_func.loc[msk]
                    a = df_func_select.sample(
                        1, weights=df_func.distribution, replace=True
                    ).index.values[0]
                    if np.all(df_func.loc[a].prime_amines >= 1):
                        rxn_selector = "diacidanhydrides_amines"
                    if np.all(df_func.loc[a].acidanhydrides >= 1):
                        rxn_selector = "diamines_acidanhydrides"
                elif df_func.loc["smiles_polymer", "prime_amines"] >= 2:
                    msk = (df_func.acidanhydrides >= 1) & (
                        df_func.index != "smiles_polymer"
                    )
                    df_func_select = df_func.loc[msk]
                    a = df_func_select.sample(
                        1, weights=df_func.distribution, replace=True
                    ).index.values[0]
                    rxn_selector = "diamines_acidanhydrides"
                elif df_func.loc["smiles_polymer", "acidanhydrides"] >= 2:
                    msk = (df_func.prime_amines >= 1) & (
                        df_func.index != "smiles_polymer"
                    )
                    df_func_select = df_func.loc[msk]
                    a = df_func_select.sample(
                        1, weights=df_func.distribution, replace=True
                    ).index.values[0]
                    rxn_selector = "diacidanhydrides_amines"
                else:
                    assert False
                rxn = Chem.AllChem.ReactionFromSmarts(rxn_dic[rxn_selector])

                # update df_func table
                df_func.loc["smiles_polymer"] = (
                    df_func.loc["smiles_polymer"] + df_func.loc[a]
                )  # adding polymer and a
                for column_name, adder in zip(
                    ["prime_amines", "acidanhydrides"], [-1, -1]
                ):
                    df_func.loc["smiles_polymer", column_name] += adder
                assert (
                    df_func.loc["smiles_polymer"][
                        df_func.loc["smiles_polymer"] > -1
                    ].shape
                    == df_func.loc["smiles_polymer"].shape
                )

                # React and select product
                mola = Chem.MolFromSmiles(a)
                prod = rxn.RunReactants((molpoly, mola))
                prodlist = [Chem.MolToSmiles(x[0]) for x in prod]
                prodlist = self._returnvalid(prodlist)
                poly = random.choice(prodlist)
                molpoly = Chem.MolFromSmiles(poly)

                # manage loop and ring close
                if (infinite_chain) & (DP_count == DP - 1):
                    # logic for closing ring
                    if (df_func.loc["smiles_polymer", "prime_amines"] > 0) & (
                        df_func.loc["smiles_polymer", "acidanhydrides"]
                    ) > 0:
                        # case for when has can ring close
                        DP_count += 1
                        DP_actual += 1
                    else:
                        # case for when has same terminal ends so can't ring close
                        DP_count = DP_count
                        DP_actual += 1
                else:
                    DP_count += 1
                    DP_actual += 1

            if infinite_chain:  # closes ring
                rxn = Chem.AllChem.ReactionFromSmarts(rxn_dic["infinite_chain"])
                prod = rxn.RunReactants((molpoly,))
                prodlist = [Chem.MolToSmiles(x[0]) for x in prod]
                prodlist = self._returnvalid(prodlist)
                poly = random.choice(prodlist)
                molpoly = Chem.MolFromSmiles(poly)

        except:
            poly = "ERROR:Imide_ReactionFailed"
        return poly, "imide"

    def __poly_NIPU(self, reactants, DP=2, distribution=[], infinite_chain=False):
        """performs condenstation reaction on dicarboxyl and  diols"""
        # function

        try:
            # initial
            rxn_dic = self.reactions["NIPU"]
            df_func = self.get_functionality(reactants, distribution=distribution)

            # select initial monomer as polymer chain
            df_poly = df_func.sample(1)
            df_func.loc["smiles_polymer"] = df_poly.sample(1).values[0]
            poly = df_poly.index[0]
            molpoly = Chem.MolFromSmiles(poly)

            DP_count = 1
            DP_actual = 1

            while DP_count < DP:

                # select rxn rule and reactant
                if (df_func.loc["smiles_polymer", "prime_amines"] >= 1) & (
                    df_func.loc["smiles_polymer", "cyclic_carbonates"] >= 1
                ):
                    msk = (
                        (df_func.cyclic_carbonates >= 1) | (df_func.prime_amines >= 1)
                    ) & (df_func.index != "smiles_polymer")
                    df_func_select = df_func.loc[msk]
                    a = df_func_select.sample(
                        1, weights=df_func.distribution, replace=True
                    ).index.values[0]
                    if df_func.loc[a].prime_amines >= 1:
                        rxn_selector = "dicycliccarbonates_amine"
                    if df_func.loc[a].cyclic_carbonates >= 1:
                        rxn_selector = "diamine_cycliccarbonate"
                elif df_func.loc["smiles_polymer", "prime_amines"] >= 2:
                    msk = (df_func.cyclic_carbonates >= 1) & (
                        df_func.index != "smiles_polymer"
                    )
                    df_func_select = df_func.loc[msk]
                    a = df_func_select.sample(
                        1, weights=df_func.distribution, replace=True
                    ).index.values[0]
                    rxn_selector = "diamine_cycliccarbonate"
                elif df_func.loc["smiles_polymer", "cyclic_carbonates"] >= 2:
                    msk = (df_func.prime_amines >= 1) & (
                        df_func.index != "smiles_polymer"
                    )
                    df_func_select = df_func.loc[msk]
                    a = df_func_select.sample(
                        1, weights=df_func.distribution, replace=True
                    ).index.values[0]
                    rxn_selector = "dicycliccarbonates_amine"
                else:
                    assert False

                rxn = Chem.AllChem.ReactionFromSmarts(rxn_dic[rxn_selector])

                # update df_func table
                df_func.loc["smiles_polymer"] = (
                    df_func.loc["smiles_polymer"] + df_func.loc[a]
                )  # adding polymer and a
                for column_name in ["prime_amines", "cyclic_carbonates"]:
                    df_func.loc["smiles_polymer", column_name] += -1
                assert (
                    df_func.loc["smiles_polymer"][
                        df_func.loc["smiles_polymer"] > -1
                    ].shape
                    == df_func.loc["smiles_polymer"].shape
                )

                # React and select product
                mola = Chem.MolFromSmiles(a)
                prod = rxn.RunReactants((molpoly, mola))
                prodlist = [Chem.MolToSmiles(x[0]) for x in prod]
                prodlist = self._returnvalid(prodlist)
                poly = random.choice(prodlist)
                molpoly = Chem.MolFromSmiles(poly)

                # manage loop and ring close
                if (infinite_chain) & (DP_count == DP - 1):
                    # logic for closing ring
                    if (df_func.loc["smiles_polymer", "prime_amines"] > 0) & (
                        df_func.loc["smiles_polymer", "cyclic_carbonates"]
                    ) > 0:
                        # case for when can ring close
                        DP_count += 1
                        DP_actual += 1
                    else:
                        # case for when has same terminal ends so can't ring close
                        DP_count = DP_count
                        DP_actual += 1
                else:
                    DP_count += 1
                    DP_actual += 1

            if infinite_chain:  # closes ring

                rxn = Chem.AllChem.ReactionFromSmarts(rxn_dic["infinite_chain"])
                prod = rxn.RunReactants((molpoly,))
                prodlist = [Chem.MolToSmiles(x[0]) for x in prod]
                prodlist = self._returnvalid(prodlist)
                poly = random.choice(prodlist)
                molpoly = Chem.MolFromSmiles(poly)
        except:
            poly = "ERROR:NIPU_ReactionFailed"

        return poly, "NIPU"

    def __poly_urethane(self, reactants, DP=2, distribution=[], infinite_chain=False):
        """performs condenstation reaction on diisocyanates and  diols"""
        # function

        try:
            # initial
            rxn_dic = self.reactions["urethane"]
            df_func = self.get_functionality(reactants, distribution=distribution)

            # select initial monomer as polymer chain
            df_poly = df_func.sample(1)
            df_func.loc["smiles_polymer"] = df_poly.sample(1).values[0]
            poly = df_poly.index[0]
            molpoly = Chem.MolFromSmiles(poly)

            DP_count = 1
            DP_actual = 1

            while DP_count < DP:

                # select rxn rule and reactant
                if (df_func.loc["smiles_polymer", "isocyanates"] >= 1) & (
                    df_func.loc["smiles_polymer", "ols"] >= 1
                ):
                    msk = ((df_func.ols >= 1) | (df_func.isocyanates >= 1)) & (
                        df_func.index != "smiles_polymer"
                    )
                    df_func_select = df_func.loc[msk]
                    a = df_func_select.sample(
                        1, weights=df_func.distribution, replace=True
                    ).index.values[0]
                    if df_func.loc[a].isocyanates >= 1:
                        rxn_selector = "diols_isocyanate"
                    if df_func.loc[a].ols >= 1:
                        rxn_selector = "diisocyanates_ol"
                elif df_func.loc["smiles_polymer", "isocyanates"] >= 2:
                    msk = (df_func.ols >= 1) & (df_func.index != "smiles_polymer")
                    df_func_select = df_func.loc[msk]
                    a = df_func_select.sample(
                        1, weights=df_func.distribution, replace=True
                    ).index.values[0]
                    rxn_selector = "diisocyanates_ol"
                elif df_func.loc["smiles_polymer", "ols"] >= 2:
                    msk = (df_func.isocyanates >= 1) & (
                        df_func.index != "smiles_polymer"
                    )
                    df_func_select = df_func.loc[msk]
                    a = df_func_select.sample(
                        1, weights=df_func.distribution, replace=True
                    ).index.values[0]
                    rxn_selector = "diols_isocyanate"
                else:
                    assert False

                rxn = Chem.AllChem.ReactionFromSmarts(rxn_dic[rxn_selector])

                # update df_func table
                df_func.loc["smiles_polymer"] = (
                    df_func.loc["smiles_polymer"] + df_func.loc[a]
                )  # adding polymer and a
                for column_name in ["isocyanates", "ols"]:
                    df_func.loc["smiles_polymer", column_name] += -1
                assert (
                    df_func.loc["smiles_polymer"][
                        df_func.loc["smiles_polymer"] > -1
                    ].shape
                    == df_func.loc["smiles_polymer"].shape
                )

                # React and select product
                mola = Chem.MolFromSmiles(a)
                prod = rxn.RunReactants((molpoly, mola))
                prodlist = [Chem.MolToSmiles(x[0]) for x in prod]
                prodlist = self._returnvalid(prodlist)
                poly = random.choice(prodlist)
                molpoly = Chem.MolFromSmiles(poly)

                # manage loop and ring close
                if (infinite_chain) & (DP_count == DP - 1):
                    # logic for closing ring
                    if (df_func.loc["smiles_polymer", "isocyanates"] > 0) & (
                        df_func.loc["smiles_polymer", "ols"]
                    ) > 0:
                        # case for when can ring close
                        DP_count += 1
                        DP_actual += 1
                    else:
                        # case for when has same terminal ends so can't ring close
                        DP_count = DP_count
                        DP_actual += 1
                else:
                    DP_count += 1
                    DP_actual += 1

            if infinite_chain:  # closes ring

                rxn = Chem.AllChem.ReactionFromSmarts(rxn_dic["infinite_chain"])
                prod = rxn.RunReactants((molpoly,))
                prodlist = [Chem.MolToSmiles(x[0]) for x in prod]
                prodlist = self._returnvalid(prodlist)
                poly = random.choice(prodlist)
                molpoly = Chem.MolFromSmiles(poly)
        except:
            poly = "ERROR:Urethane_ReactionFailed"

        return poly, "urethane"

    def __poly_upe(self, reactants, crosslinker, distribution, DP):
        """generates 2 ringed thermoset
        reactants: list of smiles
        crosslinker: boolean list indicating which reactants are in the ring structure and which are in the crosslink
        mols: number of mols in reaction, this is not just the molar ratio and should be actual mols
        DP: integer, degree of polymerization
        """
        # getting distributed reactants and parsing monomers
        reactants = np.array(reactants)
        crosslinker = np.array(crosslinker, dtype=bool)
        distribution = np.array(distribution)

        reactants_backbone = reactants[~crosslinker]
        reactants_backbone = tuple(reactants[np.isin(reactants, reactants_backbone)])
        distribution_backbone = tuple(
            distribution[np.isin(reactants, reactants_backbone)]
        )

        reactants_crosslinker = reactants[crosslinker]
        reactants_crosslinker = tuple(
            reactants[np.isin(reactants, reactants_crosslinker)]
        )
        distribution_crosslinker = tuple(
            distribution[np.isin(reactants, reactants_crosslinker)]
        )

        # parse DP
        # to be inserted

        # make rings by generating ring structures, makes 20 attempts to have ring with a reaction cite and protects any other reactions cites
        ring1 = ring2 = "ERROR"
        i = 0
        while ring1 == "ERROR" or ring2 == "ERROR":
            dfrings = self.thermoplastic(
                reactants_backbone,
                mechanism="ester",
                DP=DP,
                replicate_structures=1,
                infinite_chain=True,
                verbose=False,
            )

            if dfrings.smiles_polymer.str.contains(
                "ERROR"
            ).any():  # makes sure the ester reaction worked before trying to protect
                pass
            else:
                mol, p = dfrings.apply(
                    lambda row: self.__protect_substructure(
                        row.smiles_polymer, "C=C", n_unprotected=1
                    ),
                    axis=1,
                )[0]
                if p > 0:
                    if ring1 == "ERROR":
                        ring1 = mol
                    else:
                        ring2 = mol
            i += 1
            if i > 20:
                break

        if (
            type(ring1) == str or type(ring2) == str
        ):  # makes sure rings have been assigned, if error could be because ring didnt have rxn site or bc ester rxn failed
            poly = "ERROR:Ester_ReactionFailed"
        else:
            rings = [Chem.MolToSmiles(s) for s in [ring1, ring2]]

            ## connect rings
            reactant_ringclose = rings + list(reactants_crosslinker)
            poly = self.__poly_vinyl(
                reactant_ringclose,
                DP=DP,
                distribution=distribution_crosslinker,
                crosslink=True,
            )[0]

            if "ERROR" in poly:
                poly = "ERROR:Vinyl_ReactionFailed"

        return poly, "UPE"

    def _openacidanyhydride(self, reactant):

        rxn = Chem.AllChem.ReactionFromSmarts(
            self.reactions["open_acidanhydrides"]["add_OH"]
        )
        mol = Chem.MolFromSmiles(reactant)
        prod = rxn.RunReactants((mol,))
        prod = random.choice(prod)[0]

        mol = Chem.RWMol(prod)
        mol.RemoveBond(0, 1)
        return Chem.MolToSmiles(mol)
