import pandas as pd 
import numpy as np
import re
import random
import ast
import warnings
import itertools
import time

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from rdkit import RDLogger
from rdkit.Chem import Descriptors

from copy import deepcopy
from tqdm import tqdm

import casadi as cas
from casadi import SX,integrator,vertcat

from .polymaker import PolyMaker

tqdm.pandas()
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)


class PolyEstimator():
	def __init__ (self):
		self.pm = PolyMaker()
		self.dfsmiles_mapper = pd.read_csv('./m2p/201117_smiles_map_rdkit2thermo.csv',index_col=0)


	def get_monomer_density(self,smiles):
		try:
			thermo_smiles = self.dfsmiles_mapper[self.dfsmiles_mapper.rdkit_smiles==smiles].thermo_smiles.values[0]
			chemical = thermo.Chemical('SMILES={}'.format(thermo_smiles))
			rho = chemical.rho/1000
		except:
			rho = 1
		return rho

	def get_initial_system(self,chain,crosslinkers,mols_chain,mols_crosslinkers,mechanism):
		'''
		Gets the initial parameters of the system to feed into the crosslink density estimator
		
		inputs:
			chain:
				description: A smiles string of all monomers used as the backbone of the thermoset.
				type: string
				example: 'OC(=O)C=CC(=O)O.OC(=O)CCC(O)=O.OCCCO'
			crosslinker:
				description: A smiles string of all monomers used in the crosslinks of the thermoset.
				type: string
				example: 'C=Cc1ccccc1'
			mols_chain:
				description: A list of floats containing the number of mols used in the polymerization of the backbone of the thermoset
				type: list
				example: [10, 40.5, 50.5]
			mols_crosslinkers:
				description: A list of floats containing the number of mols used in the polymerization of the crosslinks of the thermoset
				type: list
				example: [10]
			mechansim:            
				description: a description from list of options below to identify the mechanisms involved with making the thermoset
				type: string
				example: 'UPE'
				options:
					UPE: unsaturate polyesters
		
		returns: dictionary
		
				
		'''
		#get smiles
		smiles_chain = self.pm.get_monomers(chain)
		smiles_crosslinkers = self.pm.get_monomers(crosslinkers)

		# get mols
		molobject_chain = [Chem.MolFromSmiles(s) for s in smiles_chain]
		molobject_crosslinkers = [Chem.MolFromSmiles(s) for s in smiles_crosslinkers]

		# get MW
		MW_chain = [Chem.Descriptors.MolWt(m) for m in molobject_chain]
		MW_leavinggroup = Chem.Descriptors.MolWt(Chem.MolFromSmiles('O'))
		MW_crosslinkers = [Chem.Descriptors.MolWt(m) for m in molobject_crosslinkers]

		# get density
		density_crosslinkers = [self.get_monomer_density(s) for s in smiles_crosslinkers]
		density_polymer = 1

		# get mass
		mass_polymer = np.sum([mols_i*MW_i for mols_i,MW_i in zip(mols_chain,MW_chain)]) - MW_leavinggroup*np.sum(mols_chain)/2

		# get volumes
		V_crosslinkers = np.sum([mols_i*MW_i/rho_i for mols_i,MW_i,rho_i in zip(mols_crosslinkers,MW_crosslinkers,density_crosslinkers)])
		V_polymer = mass_polymer/density_polymer
		V_sys = V_polymer+V_crosslinkers

		# get initiation point concentration
		initiationpointcount_chain = [len(m.GetSubstructMatches(Chem.MolFromSmarts('C=C'))) for m in molobject_chain]
		mols_initiationpoints = np.sum([n*i for n,i in zip(mols_chain,initiationpointcount_chain)])
		C_initiationpoints = mols_initiationpoints/V_sys

		# get crosslinker monomer concentration
		C_crosslinkers = np.sum(mols_crosslinkers)/V_sys

		return {'C_crosslinkers':C_crosslinkers,
				'C_initiationpoints':C_initiationpoints,
				'V_sys':V_sys,
				'mass_polymer':mass_polymer}
	
	def get_species(self,initialsystem):
		M  = SX.sym('monomer')    # concentration of monomer
		Mr = SX.sym('monomer*')   # concentration of monomer with radical
		P  = SX.sym('polymer')    # concentration of polymer
		Pr = SX.sym('polymer*')   # concentration of polymer with radical
		I  = SX.sym('initationpoint')    # concentration of initiation point on chain backbone
		A  = SX.sym('arm')    # concentration of arms from chain backbone
		Ar = SX.sym('arm*')   # concentration of arms from chain backbone with radical
		C  = SX.sym('crosslink')    # concentration of crosslinks

		
		Ci = [initialsystem['C_crosslinkers'],0 ,0,0 ,initialsystem['C_initiationpoints'],0,0 ,0]
		species_dict = {s.name():{'sx':s,'Ci':i} for s,i in zip([M,Mr,P,Pr,I,A,Ar,C],Ci)}
		# species_dict['vertcat'] = vertcat(M ,Mr,P,Pr,I,A,Ar,C)

		return species_dict
	
	def get_rate_consts(self,**kwargs):
		rate_consts_base = {'ki':10*60,             # L/mol/min
							'kp' : 2.090000e+02*60, # L/mol/min
							'kt' : 1.400000e+07*60  # L/mol/min
						   }
		
		# update base constants with kwards
		for k,v in kwargs.items():
			if k in rate_consts_base.keys(): rate_consts_base[k] = v
		rate_consts_all = { 'k1' : rate_consts_base['ki'],
							'k2' : rate_consts_base['ki'],
							'k3' : rate_consts_base['ki'],
							'k4' : rate_consts_base['kp'],
							'k5' : rate_consts_base['kp'],
							'k6' : rate_consts_base['kt'],
							'k7' : rate_consts_base['kt'],
							'k8' : rate_consts_base['kt'],
							'k9' : rate_consts_base['kt']/10,
							'k10': rate_consts_base['kt'],
							'k11': rate_consts_base['kt'],
							'k12': rate_consts_base['kt'],
							'k13': rate_consts_base['kt']        
							}
		
		# update all constants with kwargs and add ones not in dic unless its a base const
		for k,v in kwargs.items():
			if k in rate_consts_all.keys(): rate_consts_all[k] = v
			elif k not in rate_consts_base.keys(): rate_consts_all[k] = v
		return rate_consts_all

	def get_reactions(self,species_dict,rate_consts_dict):

		# Reactions

		# initiation
		r1  = rate_consts_dict['k1']*species_dict['monomer']['sx']
		r2  = rate_consts_dict['k2']*species_dict['monomer*']['sx']*species_dict['initationpoint']['sx']
		r3  = rate_consts_dict['k3']*species_dict['monomer*']['sx']*species_dict['monomer']['sx']

		# propagation
		r4  = rate_consts_dict['k4']*species_dict['polymer*']['sx']*species_dict['monomer']['sx']
		r5  = rate_consts_dict['k5']*species_dict['arm*']['sx']*species_dict['monomer']['sx']

		# termination
		r6  = rate_consts_dict['k6']*species_dict['polymer*']['sx']*species_dict['monomer*']['sx']
		r7  = rate_consts_dict['k7']*species_dict['polymer*']['sx']**2
		r8  = rate_consts_dict['k8']*species_dict['polymer*']['sx']*species_dict['initationpoint']['sx']
		r9  = rate_consts_dict['k9']*species_dict['arm*']['sx']*species_dict['monomer*']['sx']
		r10 = rate_consts_dict['k10']*species_dict['arm*']['sx']*species_dict['polymer*']['sx']
		r11 = rate_consts_dict['k11']*species_dict['arm*']['sx']*species_dict['initationpoint']['sx']
		r12 = rate_consts_dict['k12']*species_dict['arm*']['sx']**2
		
		#Right hand side of ODEs
		rhs_M  = -r1-r3-r4-r5
		rhs_Mr = r1-r2-r3-r6-r9
		rhs_P  = r6+r7
		rhs_Pr = r3+r4-r4-r6-r7-r8
		rhs_I  = -r2-r8-r11
		rhs_A  = r8+r9+r10
		rhs_Ar = r2+r5-r5-r9-r10-r11-r12
		rhs_C  = r11+r12

		reactions = vertcat(rhs_M,rhs_Mr,rhs_P,rhs_Pr,rhs_I,rhs_A,rhs_Ar,rhs_C)
		
		return reactions

	def run_reaction(self,species_dict,reactions,t_i=0,t_f=20,t_step=0.1):

		# ODE and Bounds
		t= np.arange(t_i,t_f,t_step)
		ode = {'x':vertcat(*[v['sx'] for v in species_dict.values()]),
			   'ode':reactions}

		# Solving
		opts = {'grid':t, 'output_t0':True}
		F = cas.integrator('F','cvodes',ode,opts)
		results = F(x0=[v['Ci'] for v in species_dict.values()])
		
		# Clean results
		dfresults = pd.DataFrame(np.array(results['xf']),index=species_dict.keys()).T.set_index(t)

		return dfresults, results

	def calc_Mc(self,dfresults,initialsystem):
		# Calc molecule weight between crosslinks
		C_crosslinks = dfresults.tail(1).crosslink.values[0]
		N_crosslinks = C_crosslinks*initialsystem['V_sys']
		Mc = initialsystem['mass_polymer'] / N_crosslinks
		return Mc