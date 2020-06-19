import pandas as pd 
import numpy as np
import re
import random
import ast
import warnings

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from rdkit import RDLogger

from copy import deepcopy
from tqdm import tqdm, tqdm_pandas, tqdm_notebook

tqdm.pandas(tqdm_notebook)
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)


class PolyMaker():
	def __init__ (self):
		self.smiles_req = {'ols':'[C,c;!$(C=O)][OH]',
							'aliphatic_ols':'[C;!$(C=O);!$([a])][OH]',
							'acids':'[#6][#6](=[#8:4])([F,Cl,Br,I,#8H,O-])',
							'prime_amines':'[#6;!$(C=O)][NH2;!$([NH2+])]',
							'carbonates':'[O]=[C]([F,Cl,Br,I,O])([F,Cl,Br,I,O])',
							'acidanhydrides':'[#8]([#6](=[#8]))([#6](=[#8]))',
							'prime_thiols':'[#6;!$(C=O)][SH]'}
		self.__verison__ = 'm2p version: 2019.07.01'

	def checksmile(self,s):
		'''checks to make sure monomer is readable by rdkit and 
		returns canonical smile

		Input: string
		
		Returns: string
		'''
		rdBase.DisableLog('rdApp.error')
		try:
			mol = Chem.MolFromSmiles(s)
			mol = Chem.MolToSmiles(mol)
		except:
			mol = ''
		rdBase.EnableLog('rdApp.error')
		return mol

	def get_monomers(self,s,stereochemistry=False):
		'''parses a string into a list of monomers
		the string is separated by '.' and each monomer is checked
		for validity

		Input: string

		Returns: list of strings
		'''
		s = s.split('.')
		if not stereochemistry:s = [s_i.replace('/','').replace('@','') for s_i in s]
		monomers = tuple([self.checksmile(s_i) for s_i in s])
		if np.any(np.array(monomers)=='',):monomers==None
		return monomers

	def __returnvalid(self,prodlist):
		'''verifies molecule is valid
		
		Input: list of strings
		
		Return: list of strings
		'''
		returnlist = []
		rdBase.DisableLog('rdApp.error')
		for x in prodlist:
			try:
				Chem.SanitizeMol(Chem.MolFromSmiles(x))
				returnlist.append(x)
			except:
				 pass
		rdBase.EnableLog('rdApp.error')
		return returnlist

	def polymerize(self,reactants,DP=2,mechanism='',replicate_structures=1,distribution=[]):
		'''Main polymerization method

		Inputs:
			reactants: tuple of monomers as strings or 
						a pandas dataframe containing a list of monomers as strings with column title monomers
			DP: integer, degree of polymerization which is the number of monomer units in the polymer
			mechanism: string, 
				
				vinyl: performs polymerization along vinyl groups
				ester: performs condensation reaction on dicarboxylic acid + diol
				amide: performs condensation reaction on dicarboxylic acid + diamine
				carbonate: performs condensation reaction on carbonate + diol
		
		Returns:
			polymer: string
			
		''' 

		returnpoly = pd.DataFrame()
		for rep in range(0,replicate_structures):
			returnpoly_i = pd.DataFrame()

			# reactants should be a tuple but as a string going into polymerization methods 
			# this puts everthing into dataframe before generating structures
			if type(reactants)==str:
				try:
					reactants_i = ast.literal_eval(reactants)
				except:
					reactants_i = self.get_monomers(reactants)
				returnpoly_i['monomers']=pd.Series(str(reactants_i))
			elif type(reactants)==tuple:
				returnpoly_i['monomers']=pd.Series(str(reactants))
			elif type(reactants)==pd.DataFrame:
				returnpoly_i = reactants
			else:
				raise ValueError('Data type not recognized')
			returnpoly_i['replicate_structure']=rep
			returnpoly_i['monomers'] = returnpoly_i.monomers.astype(str)
			returnpoly_i[['polymer','mechanism']] = returnpoly_i.progress_apply(
																			lambda row: 
																				self.__polymerizemechanism(
																					ast.literal_eval(row.monomers),
																					DP,
																					mechanism),
																			axis=1)
			
			returnpoly = pd.concat([returnpoly,returnpoly_i])
		return returnpoly

	def __polymerizemechanism(self,reactants,DP,mechanism,distribution=[]):
		'''directs polymerization to correct method for mechanism'''
		returnpoly = ''

		if (mechanism=='vinyl')|(mechanism=='acrylate'):
			polydata = self.__poly_vinyl(reactants,DP)
			returnpoly = polydata[0]
			mechanism = polydata[1]

		elif mechanism=='ester':
			polydata = self.__poly_ester(reactants,DP,distribution)
			returnpoly = polydata[0]
			mechanism = polydata[1]
		
		elif mechanism=='amide':
			polydata = self.__poly_amide(reactants,DP,distribution)
			returnpoly = polydata[0]
			mechanism = polydata[1]
		
		elif mechanism=='carbonate':
			polydata = self.__poly_carbonate(reactants,DP,distribution)
			returnpoly = polydata[0]
			mechanism = polydata[1]

		elif mechanism=='imide':
			polydata = self.__poly_imide(reactants,DP,distribution)
			returnpoly = polydata[0]
			mechanism = polydata[1]

		elif mechanism=='all':
				polylist = [self.__poly_vinyl(reactants,DP),
							self.__poly_ester(reactants,DP,distribution),
							self.__poly_amide(reactants,DP,distribution),
							self.__poly_carbonate(reactants,DP,distribution),
							self.__poly_imide(reactants,DP,distribution)]

				polylist = [p for p in polylist if p[0] not in ['ERROR:Vinyl_ReactionFailed',
															'ERROR:Ester_ReactionFailed',
															'ERROR:Amide_ReactionFailed',
															'ERROR:Carbonate_ReactionFailed',
															'ERROR:Imide_ReactionFailed',
															'']]
				if len(polylist)==1:
					returnpoly = polylist[0][0]
					mechanism = polylist[0][1]
				elif len(polylist) > 1:
					returnpoly = 'ERROR_02:MultiplePolymerizations'
				else:
					returnpoly = 'ERROR_01:NoReaction'
		else:
			returnpoly='ERROR_03:MechanismNotRecognized'

		return pd.Series([returnpoly,mechanism])

	def __poly_vinyl_init(self,a,b):
		'''performs propagation rxn of vinyl polymer'''

		#mol conversion
		mola = Chem.MolFromSmiles(a)
		molb = Chem.MolFromSmiles(b)

		#rxn definition
		rxn = AllChem.ReactionFromSmarts('[C:1]=[C:2].[C:3]=[C:4]>>[C:1][C:2][C:3]=[C:4]')

		#product creation and validation
		prod = rxn.RunReactants((mola,molb))
		prodlist = [Chem.MolToSmiles(x[0]) for x in prod]

		return self.__returnvalid(prodlist)

	def __poly_vinyl_prop(self,a,b):
		'''performs propagation rxn of vinyl polymer'''

		#mol conversion
		mola = Chem.MolFromSmiles(a)
		molb = Chem.MolFromSmiles(b)

		#rxn definition
		rxn = AllChem.ReactionFromSmarts('[C:0][C:1][C:2]=[C:3].[C:4]=[C:5]>>[C:0][C:1][C:2][C:3][C:4]=[C:5]')

		#product creation and validation
		prod = rxn.RunReactants((mola,molb))
		prodlist = [Chem.MolToSmiles(x[0]) for x in prod]

		return self.__returnvalid(prodlist)

	def __poly_vinyl_term(self,a):
		'''performs termination rxn of vinyl polymer'''
		#mol conversion

		mola = Chem.MolFromSmiles(a)

		#rxn definition
		rxn = AllChem.ReactionFromSmarts('[C:0][C:1][C:2]=[C:3]>>[C:0][C:1][C:2][C:3]')

		#product creation and validation
		prod = rxn.RunReactants((mola,))
		prodlist = [Chem.MolToSmiles(x[0]) for x in prod]

		return self.__returnvalid(prodlist)

	def __poly_vinyl(self,reactants,DP=2):
		''' performs vinyl polymerization'''
		try:
			assert DP>=2

			# initiate 
			polymer = self.__poly_vinyl_init(random.choice(reactants),
											random.choice(reactants))
		
			# propagate
			for r in range(0,DP-2):
				assert len(polymer)>=1
				polymer = self.__poly_vinyl_prop(random.choice(polymer),
												random.choice(reactants)) #propagate along polymer

			#terminate 
			polymer = self.__poly_vinyl_term(random.choice(polymer))
			polymer = random.choice(polymer)

		except:
			polymer = 'ERROR:Vinyl_ReactionFailed'
		return polymer, 'vinyl'

	def __protect_carboxyols(self,mol,randomselect=False):
		'''protects all but the first hydroxyl and carboxyl in a molecule'''

		mol = deepcopy(mol)   

		#randomly select which functional group should be protected
		if randomselect:
			rc = random.choice(['first','last'])
		else:
			rc = 'first'
		
		if rc=='first':
			#protect all but one ol
			for match in mol.GetSubstructMatches(Chem.MolFromSmarts('[C][OH]'))[1:]:
				mol.GetAtomWithIdx(match[0]).SetProp('_protected','1')
			#protect all but one carboxyl
			for match in mol.GetSubstructMatches(Chem.MolFromSmarts('[O][C]=[O]'))[1:]:
				mol.GetAtomWithIdx(match[1]).SetProp('_protected','1')
		else:
			#protect all but one ol
			for match in mol.GetSubstructMatches(Chem.MolFromSmarts('[C][OH]'))[:1]:
				mol.GetAtomWithIdx(match[0]).SetProp('_protected','1')
			#protect all but one carboxyl
			for match in mol.GetSubstructMatches(Chem.MolFromSmarts('[O][C]=[O]'))[:1]:
				mol.GetAtomWithIdx(match[1]).SetProp('_protected','1')    
		return mol

	def __protect_carboxyamines(self,mol,randomselect=False):
		'''protects all but the first hydroxyl and carboxyl in a molecule'''

		mol = deepcopy(mol)   

		if randomselect:
			rc = random.choice(['first','last'])
		else:
			rc = 'first'
		
		if rc=='first':
			#protect all but one ol
			for match in mol.GetSubstructMatches(Chem.MolFromSmarts('[#6][NH2]'))[1:]:
				mol.GetAtomWithIdx(match[0]).SetProp('_protected','1')
			#protect all but one carboxyl
			for match in mol.GetSubstructMatches(Chem.MolFromSmarts('[O][C]=[O]'))[1:]:
				mol.GetAtomWithIdx(match[1]).SetProp('_protected','1')
		else:
			#protect all but one ol
			for match in mol.GetSubstructMatches(Chem.MolFromSmarts('[#6][NH2]'))[:1]:
				mol.GetAtomWithIdx(match[0]).SetProp('_protected','1')
			#protect all but one carboxyl
			for match in mol.GetSubstructMatches(Chem.MolFromSmarts('[O][C]=[O]'))[:1]:
				mol.GetAtomWithIdx(match[1]).SetProp('_protected','1')    
		return mol

	def get_functionality(self,reactants,distribution=[]):

		def id_functionality(r):
			mol = Chem.MolFromSmiles(r.name)
			r.ols = 			len(mol.GetSubstructMatches(Chem.MolFromSmarts(self.smiles_req['ols'])))
			r.aliphatic_ols = 	len(mol.GetSubstructMatches(Chem.MolFromSmarts(self.smiles_req['aliphatic_ols'])))
			r.acids = 			len(mol.GetSubstructMatches(Chem.MolFromSmarts(self.smiles_req['acids'])))
			r.prime_amines = 	len(mol.GetSubstructMatches(Chem.MolFromSmarts(self.smiles_req['prime_amines'])))
			r.carbonates = 		len(mol.GetSubstructMatches(Chem.MolFromSmarts(self.smiles_req['carbonates'])))
			r.acidanhydrides = 	len(mol.GetSubstructMatches(Chem.MolFromSmarts(self.smiles_req['acidanhydrides'])))
			return r    

		df_func = pd.DataFrame(data = 0,index=reactants,columns=['ols','acids','prime_amines','carbonates','aliphatic_ols','acidanhydrides'])
		return df_func.apply(lambda r: id_functionality(r),axis=1)

	def __poly_ester(self,reactants,DP=2,distribution=[]):
		'''performs condenstation reaction on dicarboxyl and  diols'''
		# function

		try:
			# initial
			if len(distribution)==0:distribution = [1/len(reactants)]*len(reactants)
			rxn_dic = {'diols_acids':'[C;!$(C=O);!$([a]):6][OH:1].[#6:2][#6:3](=[O:4])([F,Cl,Br,I,#8H,O-:5])>>'
			           '[C:6][O:1][#6:3](=[O:4])([#6:2])',
			           'diacids_ols':'[#6:2][#6:3](=[O:4])([F,Cl,Br,I,#8H,O-:5]).[C;!$(C=O);!$([a]):6][OH:1]>>'
			           '[C:6][O:1][#6:3](=[O:4])([#6:2])'}
			df_func = self.get_functionality(reactants,distribution)

			#select initial monomer as polymer chain
			df_poly = df_func.sample(1)
			df_func.loc['polymer'] = df_poly.sample(1).values[0]
			poly = df_poly.index[0]
			molpoly = Chem.MolFromSmiles(poly)

			for i in range(DP-1):
				#select rxn rule and reactant
				if (df_func.loc['polymer','aliphatic_ols']>=1)&(df_func.loc['polymer','acids']>=1):
				    a = random.choice(df_func.loc[((df_func.acids>=1)|(df_func.aliphatic_ols>=1))&(df_func.index!='polymer')].index.tolist())
				    if df_func.loc[a].aliphatic_ols>=1: rxn_selector ='diacids_ols'
				    if df_func.loc[a].acids >=1: rxn_selector = 'diols_acids'
				    rxn = Chem.AllChem.ReactionFromSmarts(rxn_dic[rxn_selector])
				elif df_func.loc['polymer','aliphatic_ols'] >=2:
				    a = random.choice(df_func.loc[(df_func.acids>=1)&(df_func.index!='polymer')].index.tolist())
				    rxn = Chem.AllChem.ReactionFromSmarts(rxn_dic['diols_acids'])
				elif df_func.loc['polymer','acids']>=2:
				    a = random.choice(df_func.loc[(df_func.aliphatic_ols>=1)&(df_func.index!='polymer')].index.tolist())
				    rxn = Chem.AllChem.ReactionFromSmarts(rxn_dic['diacids_ols'])

				#update df_func table
				df_func.loc['polymer']+=df_func.loc[a]
				df_func.loc['polymer','aliphatic_ols'] += -1
				df_func.loc['polymer','ols'] += -1
				df_func.loc['polymer','acids'] += -1
				assert df_func.loc['polymer'][df_func.loc['polymer']>-1].shape==df_func.loc['polymer'].shape
				#React and select product
				mola = Chem.MolFromSmiles(a)
				prod = rxn.RunReactants((molpoly,mola))
				prodlist = [Chem.MolToSmiles(x[0]) for x in prod]
				prodlist = self.__returnvalid(prodlist)
				poly = random.choice(prodlist)
				molpoly = Chem.MolFromSmiles(poly)

		except:
			poly='ERROR:Ester_ReactionFailed'
		return poly, 'ester'

	def __poly_amide(self,reactants,DP=2,distribution=[]):
		'''performs condenstation reaction on dicarboxyl and  diols'''
		# function

		try:
			# initial
			if len(distribution)==0:distribution = [1/len(reactants)]*len(reactants)
			rxn_dic = {'diamines_acids':'[#6;!$(C=O):0][NH2;!$([NH2+]):1].[#6:2][#6:3](=[O:4])([#8H,O-:5])>>[#6:0][NH:1][#6:3](=[O:4])([#6:2])',
					   'diacids_amines':'[#6:2][#6:3](=[O:4])([#8H,O-:5]).[#6;!$(C=O):0][NH2;!$([NH2+]):1]>>[#6:0][NH:1][#6:3](=[O:4])([#6:2])'}

			df_func = self.get_functionality(reactants,distribution)

			#select initial monomer as polymer chain
			df_poly = df_func.sample(1)
			df_func.loc['polymer'] = df_poly.sample(1).values[0]
			poly = df_poly.index[0]
			molpoly = Chem.MolFromSmiles(poly)

			for i in range(DP-1):
				#select rxn rule and reactant
				if (df_func.loc['polymer','prime_amines']>=1)&(df_func.loc['polymer','acids']>=1):
				    a = random.choice(df_func.loc[((df_func.acids>=1)|(df_func.prime_amines>=1))&(df_func.index!='polymer')].index.tolist())
				    if df_func.loc[a].prime_amines>=1: rxn_selector ='diacids_amines'
				    if df_func.loc[a].acids >=1: rxn_selector = 'diamines_acids'
				    rxn = Chem.AllChem.ReactionFromSmarts(rxn_dic[rxn_selector])
				elif df_func.loc['polymer','prime_amines'] >=2:
				    a = random.choice(df_func.loc[(df_func.acids>=1)&(df_func.index!='polymer')].index.tolist())
				    rxn = Chem.AllChem.ReactionFromSmarts(rxn_dic['diamines_acids'])
				elif df_func.loc['polymer','acids']>=2:
				    a = random.choice(df_func.loc[(df_func.prime_amines>=1)&(df_func.index!='polymer')].index.tolist())
				    rxn = Chem.AllChem.ReactionFromSmarts(rxn_dic['diacids_amines'])

				#update df_func table
				df_func.loc['polymer']+=df_func.loc[a]
				df_func.loc['polymer','prime_amines'] += -1
				df_func.loc['polymer','acids'] += -1

				#React and select product
				mola = Chem.MolFromSmiles(a)
				prod = rxn.RunReactants((molpoly,mola))
				prodlist = [Chem.MolToSmiles(x[0]) for x in prod]
				prodlist = self.__returnvalid(prodlist)
				poly = random.choice(prodlist)
				molpoly = Chem.MolFromSmiles(poly)

		except:
			poly='ERROR:Amide_ReactionFailed'
		return poly, 'amide'

	def __poly_carbonate(self,reactants,DP=2,distribution=[]):
		try:
			# initial
			if len(distribution)==0:distribution = [1/len(reactants)]*len(reactants)
			rxn_dic = {'diols_carbonates':'[C,c;!$(C=O):0][OH:1].[O:2]=[C:3]([F,Cl,Br,I,O:4])([F,Cl,Br,I:5])>>'
			                                              '[O:2]=[C:3]([O:1][C,c:0])[X:4]',
			           'carbonates_diols':'[O:2]=[C:3]([F,Cl,Br,I,O:4])([F,Cl,Br,I:5]).[C,c;!$(C=O):0][OH:1]>>'
			                                              '[O:2]=[C:3]([O:1][C,c:0])[X:4]'}
			df_func = self.get_functionality(reactants,distribution)

			#select initial monomer as polymer chain
			df_poly = df_func.sample(1)
			df_func.loc['polymer'] = df_poly.sample(1).values[0]
			poly = df_poly.index[0]
			molpoly = Chem.MolFromSmiles(poly)
			for i in range(DP-1):
				#select rxn rule and reactant
				if (df_func.loc['polymer','ols']>=1)&(df_func.loc['polymer','carbonates']>=0.5):
				    a = random.choice(df_func.loc[((df_func.ols>=1)|(df_func.carbonates>=0.5))&(df_func.index!='polymer')].index.tolist())
				    if (df_func.loc[a].ols>=1): 
				    	rxn_selector ='carbonates_diols'
				    if (df_func.loc[a].carbonates >=0.5): 
				    	rxn_selector = 'diols_carbonates'
				    rxn = Chem.AllChem.ReactionFromSmarts(rxn_dic[rxn_selector])

				elif df_func.loc['polymer','ols'] >=2:
				    a = random.choice(df_func.loc[(df_func.carbonates>=0.5)&(df_func.index!='polymer')].index.tolist())
				    rxn = Chem.AllChem.ReactionFromSmarts(rxn_dic['diols_carbonates'])
				elif df_func.loc['polymer','carbonates']>=1:
				    a = random.choice(df_func.loc[(df_func.ols>=1)&(df_func.index!='polymer')].index.tolist())
				    rxn = Chem.AllChem.ReactionFromSmarts(rxn_dic['carbonates_diols'])
				#update df_func table
				df_func.loc['polymer']+=df_func.loc[a]
				df_func.loc['polymer','ols'] += -1
				df_func.loc['polymer','carbonates'] += -0.5
				assert df_func.loc['polymer'][df_func.loc['polymer']>-1].shape==df_func.loc['polymer'].shape
				# React and select product
				mola = Chem.MolFromSmiles(a)
				prod = rxn.RunReactants((molpoly,mola))
				prodlist = [Chem.MolToSmiles(x[0]) for x in prod]
				prodlist = self.__returnvalid(prodlist)
				poly = random.choice(prodlist)
				molpoly = Chem.MolFromSmiles(poly)
		except:
			poly='ERROR:Carbonate_ReactionFailed'
		return poly, 'carbonate'

	def __poly_imide(self,reactants,DP=2,distribution=[]):
		'''performs condenstation reaction on dianhydride and  diamine'''
		# function

		try:

			# initial
			if len(distribution)==0:distribution = [1/len(reactants)]*len(reactants)
			rxn_dic = {'diacidanhydrides_amines':'[#8:3]([#6:4](=[#8:5]))([#6:6](=[#8:7])).[#6;!$(C=O):0][NH2:1]>>[#6:0][N:1]([#6:4](=[#8:5]))([#6:6](=[#8:7]))',
					   'diamines_acidanhydrides':'[#6;!$(C=O):0][NH2:1].[#8:3]([#6:4](=[#8:5]))([#6:6](=[#8:7]))>>[#6:0][N:1]([#6:4](=[#8:5]))([#6:6](=[#8:7]))'}
			
			df_func = self.get_functionality(reactants,distribution)

			#select initial monomer as polymer chain
			df_poly = df_func.sample(1)
			df_func.loc['polymer'] = df_poly.sample(1).values[0]
			poly = df_poly.index[0]
			molpoly = Chem.MolFromSmiles(poly)

			for i in range(DP-1):
				#select rxn rule and reactant
				if (df_func.loc['polymer','prime_amines']>=1)&(df_func.loc['polymer','acidanhydrides']>=1):
				    a = random.choice(df_func.loc[((df_func.acids>=1)|(df_func.prime_amines>=1))&(df_func.index!='polymer')].index.tolist())
				    if df_func.loc[a].prime_amines>=1: rxn_selector ='diacidanhydrides_amines'
				    if df_func.loc[a].acidanhydrides >=1: rxn_selector = 'diamines_acidanhydrides'
				    rxn = Chem.AllChem.ReactionFromSmarts(rxn_dic[rxn_selector])
				elif df_func.loc['polymer','prime_amines'] >=2:
				    a = random.choice(df_func.loc[(df_func.acidanhydrides>=1)&(df_func.index!='polymer')].index.tolist())
				    rxn = Chem.AllChem.ReactionFromSmarts(rxn_dic['diamines_acidanhydrides'])
				elif df_func.loc['polymer','acidanhydrides']>=2:
				    a = random.choice(df_func.loc[(df_func.prime_amines>=1)&(df_func.index!='polymer')].index.tolist())
				    rxn = Chem.AllChem.ReactionFromSmarts(rxn_dic['diacidanhydrides_amines'])

				#update df_func table
				df_func.loc['polymer']+=df_func.loc[a]
				df_func.loc['polymer','prime_amines'] += -1
				df_func.loc['polymer','acidanhydrides'] += -1

				#React and select product
				mola = Chem.MolFromSmiles(a)
				prod = rxn.RunReactants((molpoly,mola))
				prodlist = [Chem.MolToSmiles(x[0]) for x in prod]
				prodlist = self.__returnvalid(prodlist)
				poly = random.choice(prodlist)
				molpoly = Chem.MolFromSmiles(poly)

		except:
			poly='ERROR:Amide_ReactionFailed'
		return poly, 'imide'


