from .utils import *
from .constants import METALS, HETEROATOMS, COORDATOMS, LIGANDS, FUNCTIONALGROUPS

import fairchem.data.oc
from fairchem.core import FAIRChemCalculator, pretrained_mlip
from fairchem.data.oc.core import Adsorbate, AdsorbateSlabConfig, Bulk, Slab
from fairchem.data.oc.utils import DetectTrajAnomaly
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import VoronoiNN

from ase import Atoms, Atom
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from ase.optimize import BFGS
from ase.visualize.plot import plot_atoms

from ast import literal_eval
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path

import numpy as np
import random
import secrets
import textwrap
import copy

class ModificationTools:
    def __init__(self):
        self.strategies = {
            'substitute_metal': self._metal_subs,
            'substitute_2nd_shell': self._2nd_shell_subs,
            'substitute_coordination': self._coord_subs,
            'defect_coordination': self._coord_defect,
            'recover_coordination': self._coord_recov,
            'modify_ligand': self._modify_ligand,
            'add_functional_group': self._add_func,
            'remove_functional_group' : self._remove_func,
        } 
        self.stacked_strategies = []
        self.status = {
            'metal': "None",
            'coordination': ["N","N","N","N"],
            'defect_count' : 0,
            'ligand' : "None",
            'functional_group' : ["None", "None"],
            '2nd_shell' : ["C","C","C","C","C","C","C","C"],
            'complexity' : 0,
        }
        self.initial_positions = None

    def set_positions(self, atoms):
        self.initial_positions = get_vnn_positions(atoms)

    def set_status(self, status):
        self.status = copy.deepcopy(status)
    
    def initialize_status(self):
        self.status = {
            'metal': "None",
            'coordination': ["N","N","N","N"],
            'defect_count' : 0,
            'ligand' : "None",
            'functional_group' : ["None", "None"],
            '2nd_shell' : ["C","C","C","C","C","C","C","C"],
            'complexity' : 0,
        }
    
    def apply(self, modification_type:str, parameters:list, current_atoms):
        ### Save modification history:
        self.stacked_strategies.append([modification_type, parameters])
            
        strategy_func = self.strategies.get(modification_type)
        
        if not strategy_func:
            raise ValueError(f"Unknown modification subtype: {modification_type}")
        return strategy_func(parameters, current_atoms)

    def _metal_subs(self, parameters, current_atoms):
        current_metal, new_metal = parameters[0], parameters[1]
        modified_atoms = current_atoms.copy()
        self.status['metal'] = new_metal
        
        for atom in modified_atoms:
            if atom.symbol in METALS:
                atom.symbol = new_metal
        return modified_atoms

    def _2nd_shell_subs(self, parameters, current_atoms):
        current_element, new_element = parameters[0], parameters[1]
        modified_atoms = current_atoms.copy()
        tmp_catalyst =  current_atoms.copy()
        
        _, _, snn_idx, _ = get_vnn_idx(self.coordination_recovery(tmp_catalyst))
        doped_idx = [snn_idx[i] for i in [0,4,2,6,7,3,1,5]]
        doped_elements = [modified_atoms[idx].symbol for idx in doped_idx]

        # If suggested element not in atoms, choose first element
        if current_element not in doped_elements:
            current_element = doped_elements[0]
        
        for i,idx in enumerate(doped_idx):
            if modified_atoms[idx].symbol == current_element:
                modified_atoms[idx].symbol = new_element
                break

        self.status['2nd_shell'][i] = new_element

        if new_element == 'C':
            complexity = -1
        elif current_element == 'C':
            complexity = 1
        else:
            complexity = 0
            
        self.status['complexity'] += complexity
        return modified_atoms

    def _coord_subs(self, parameters, current_atoms):
        current_element, new_element = parameters[0], parameters[1]
        modified_atoms = current_atoms.copy()

        _, coord_idx, _, _ = get_vnn_idx(modified_atoms)
        coord_elements = [modified_atoms[idx].symbol for idx in coord_idx]

        if current_element not in coord_elements:
            current_element = coord_elements[0]

        for i, idx in enumerate(coord_idx):
            if modified_atoms[idx].symbol == current_element:
                modified_atoms[idx].symbol = new_element
                break
        
        self.status['coordination']= [modified_atoms[i].symbol for i in coord_idx]
                
        if new_element == 'N':
            complexity = -1
        elif current_element == 'N':
            complexity = 1
        else:
            complexity = 0
        self.status['complexity'] += complexity
        return modified_atoms

    def _coord_defect(self, parameters, current_atoms):
        current_element = parameters[0]
        modified_atoms = current_atoms.copy()
        self.status['defect_count'] += 1
        
        _, coord_idx, _, _ = get_vnn_idx(modified_atoms)
        coord_elements = [modified_atoms[idx].symbol for idx in coord_idx]

        if current_element not in coord_elements:
            current_element = coord_elements[0]

        for idx in coord_idx:
            if modified_atoms[idx].symbol == current_element:
                del modified_atoms[idx]
                break

        self.status['complexity'] += 1
        return modified_atoms

    def _coord_recov(self, parameters, current_atoms):
        new_element = parameters[0]
        modified_atoms = current_atoms.copy()        
        fnn_pos = self.initial_positions[1]
        self.status['defect_count'] -= 1
        
        for pos in fnn_pos:
            if is_empty(modified_atoms, pos):
                modified_atoms += Atoms(symbols=new_element, positions=[pos], tags=[1])
                break

        self.status['complexity'] -= 1
        return modified_atoms

    def _modify_ligand(self, parameters, current_atoms):
        new_ligand = parameters[1]
        
        modified_atoms = current_atoms.copy()
        current_ligand = self.status['ligand']
        self.status['ligand'] = new_ligand
        center_pos, _, _, _ = get_vnn_positions(modified_atoms) # get position of support metal
        
        # Remove existing ligand first
        if current_ligand != 'None':
            ads_idx = [idx for idx, atom in enumerate(modified_atoms) if atom.tag == 2]
            ligand_idx = []
            
            for idx in ads_idx:
                if modified_atoms[idx].position[2] < center_pos[2]:
                    ligand_idx.append(idx)
            del modified_atoms[ligand_idx]
            
        # Case of addition
        if new_ligand in ['O','OH']:
            modified_atoms += Atoms('O', [center_pos + np.array([0.0,0.0,-1.8])], tags=[2])
            if new_ligand == 'O':
                pass
            elif new_ligand == 'OH':
                modified_atoms += Atoms('H', [center_pos + np.array([0,0.8,-2.3])], tags=[2])
    
        # Case of removal
        elif new_ligand == 'None':
            pass

        if (current_ligand in ['O','OH']) and (new_ligand == 'None'):
            complexity = -1
        elif (current_ligand == 'None') and (new_ligand in ['O','OH']):
            complexity = 1
        else:
            complexity = 0
        
        self.status['complexity'] += complexity
        return modified_atoms           
        
    def _add_func(self, parameters, current_atoms):
        new_func = parameters[0]
        modified_atoms = current_atoms.copy()
        snn_pos = self.initial_positions[2]
        
        func_status = self.status['functional_group']
        num_blank = func_status.count("None")
        if num_blank == 2:
            pos = snn_pos[0]
            func_idx = 0
        elif num_blank == 1:
            pos = snn_pos[5]
            func_idx = 1
        else:
            return modified_atoms
            
        if new_func == 'COC':
            modified_atoms += Atoms(symbols='O',positions=[pos + np.array([0.6,-0.4,1.4])],tags=[1])
        elif new_func == 'CO':
            modified_atoms += Atoms(symbols='O',positions=[pos + np.array([0.0,0.0,1.8])],tags=[1])
        elif new_func == 'COH':
            modified_atoms += Atoms(symbols='O',positions=[pos + np.array([0.0,0.0,1.8])],tags=[1])
            modified_atoms += Atoms(symbols='H',positions=[pos + np.array([-0.4,0.4,2.3])],tags=[1])
        elif new_func == 'CCOOH':
            modified_atoms += Atoms(symbols='C',positions=[pos + np.array([0.0,0.0,1.6])],tags=[1])
            modified_atoms += Atoms(symbols='O',positions=[pos + np.array([-1.0,0.7,2.2])],tags=[1])
            modified_atoms += Atoms(symbols='O',positions=[pos + np.array([1.0,-0.7,2.2])],tags=[1])
            modified_atoms += Atoms(symbols='H',positions=[pos + np.array([1.0,-0.7,3.2])],tags=[1])
        self.status['functional_group'][func_idx] = new_func
        self.status['complexity'] += 1
        return modified_atoms

    def _remove_func(self, parameters, current_atoms):
        current_func = parameters[0]
        modified_atoms = current_atoms.copy()       
        snn_pos = self.initial_positions[2]
        
        func_status = self.status['functional_group']
        func_idx = func_status.index(current_func)
        if func_idx == 0:
            pos = snn_pos[0]
        elif func_idx == 1:
            pos = snn_pos[5]
        else:
            return modified_atoms 

        func_atoms_idx = []
        for idx, atom in enumerate(modified_atoms):
            if atom.tag == 1:
                position = atom.position
                xy_distance = np.linalg.norm(position[:2] - pos[:2])
                if (xy_distance < 3) and (position[2] > pos[2]):
                    func_atoms_idx.append(idx)
                    
        del modified_atoms[func_atoms_idx]
        self.status['functional_group'][func_idx] = "None"
        self.status['complexity'] -= 1
        return modified_atoms

    def coordination_recovery(self, atoms):
        atoms = atoms.copy()
        if self.status['defect_count'] != 0:
            save_defect_count = self.status['defect_count']
            save_complexity = self.status['complexity']
            for i in range(self.status['defect_count']):
                atoms = self._coord_recov(["N","None"], atoms)
            self.status['defect_count'] = save_defect_count
            self.status['complexity'] = save_complexity
            return atoms
        else:
            return atoms
    
    def get_random_modification(self, current_atoms):
        rng = random.Random(secrets.randbits(128))
        def _choice(items, exclude=None):
            if exclude is None:
                return rng.choice(items)
            candidates = [x for x in items if x != exclude]
            return rng.choice(candidates) if candidates else random.choice(items)
      
    
        modification_list = [
            'substitute_metal','substitute_2nd_shell','substitute_coordination','defect_coordination','recover_coordination',
            'modify_ligand','add_functional_group','remove_functional_group'
        ]
        mod_type = rng.choice(modification_list)
        current_metal = self.status['metal']
        current_coord = self.status['coordination']
        current_hetero = self.status['2nd_shell']
        current_ligand = self.status['ligand']
        current_func = self.status['functional_group']
        current_defect = self.status['defect_count']
        
        if mod_type == 'substitute_metal':
            param_1 = current_metal
            param_2 = _choice(METALS, exclude=param_1)
            params = [param_1, param_2]
                
        elif mod_type == 'substitute_2nd_shell':
            param_1 = rng.choice(current_hetero)
            param_2 = _choice(HETEROATOMS, exclude=param_1)
            params = [param_1, param_2]
            
        elif mod_type == 'substitute_coordination':
            param_1 = rng.choice(current_coord)
            param_2 = _choice(COORDATOMS, exclude=param_1)
            params = [param_1, param_2]
            
        elif mod_type == 'defect_coordination':
            params = [rng.choice(current_coord), "None"]
            
        elif mod_type == 'recover_coordination':
            params = [rng.choice(COORDATOMS), "None"]
            
        elif mod_type == 'modify_ligand':
            param_1 = current_ligand
            param_2 = _choice(LIGANDS, exclude=param_1)
            params = [param_2, "None"]
            
        elif mod_type == 'add_functional_group':
            params = [rng.choice(FUNCTIONALGROUPS),"None"]
            
        elif mod_type == 'remove_functional_group':
            params = [rng.choice(current_func),"None"]
            
        return mod_type, params
            
    def is_valid_modification(self, modification_type:str, parameters:list, current_atoms):
        try:
            if modification_type not in self.strategies:
                return False, "Suggested modification is not in list", 0
            if not isinstance(parameters, (list, tuple)):
                return False, "Format of parameters is wrong", 0
            if len(parameters) != 2:
                return False,  "Parameters need 2 elements", 0
            
            if modification_type == 'substitute_metal':
                current_metal, new_metal = parameters
                if new_metal not in METALS:
                    return False, "Suggested metal element not in list", 0
                if not any(atom.symbol in METALS for atom in current_atoms):
                    return False, "Suggested metal element not in catalyst", 0
                return True, '_', 0

            if modification_type == 'substitute_2nd_shell':
                current_element, new_element = parameters
                if current_element not in HETEROATOMS or new_element not in HETEROATOMS:
                    return False, "Suggested element not in list or catalyst", 0

                if new_element == 'C':
                    complexity = -1
                else:
                    complexity = 1
                return True, '_', complexity

            if modification_type == 'substitute_coordination':
                current_element, new_element = parameters
                
                _, coord_idx, _, _ = get_vnn_idx(current_atoms)
                coord_elements = [current_atoms[idx].symbol for idx in coord_idx]
                
                if current_element not in COORDATOMS or new_element not in COORDATOMS:
                    return False, "Suggested element not in list", 0
                elif current_element not in coord_elements:
                    return False, "Suggested element not in catalyst", 0
                    
                if new_element == 'N':
                    complexity = -1
                else:
                    complexity = 1
                return True, '_', complexity

            if modification_type == 'defect_coordination':
                current_element = parameters[0]
                if current_element not in COORDATOMS:
                    return False, "Suggested element not in catalyst", 0
                return True, '_', 1

            if modification_type == 'recover_coordination':
                new_element = parameters[0]
                if new_element not in COORDATOMS:
                    return False, "Suggested element not in list", 0
                if self.status['defect_count'] <= 0:
                    return False, "There is no defect", 0
                return True, '_', -1

            if modification_type == 'modify_ligand':
                current_ligand, new_ligand = parameters
                if current_ligand != self.status['ligand']:
                    return False, "Current ligand is different", 0
                
                if new_ligand not in LIGANDS:
                    return False, "Suggested ligand not in list", 0

                if new_ligand == 'None':
                    complexity = -1
                else:
                    complexity = 1
                return True, '_', complexity

            if modification_type == 'add_functional_group':
                new_func = parameters[0]
                if new_func not in FUNCTIONALGROUPS:
                    return False, "Suggested functional group not in list", 0
                if self.status['functional_group'].count("None") == 0:
                    return False,  "There is no place for additional functional group", 0
                return True, '_', 1

            if modification_type == 'remove_functional_group':
                current_func = parameters[0]
                if current_func not in FUNCTIONALGROUPS:
                    return False, "Suggested functional group not in list", 0
                if current_func not in self.status['functional_group']:
                    return False, "Suggested functional group not in catalyst", 0
                return True, '_', 1
            return False, "Unknown reason", 0

        except Exception:
            return False, "Unknown reason", 0
            
            
class CalculationTools:
    def __init__(self, calculator_name:str = 'UMA'):
        self.calculator_name = calculator_name

        if calculator_name == 'UMA':
            self.db_path = Path(fairchem.data.oc.__file__).parent / Path("databases/pkls/adsorbates.pkl")
            self.atomic_reference_energies = {
                "H": -3.477, "N": -8.083, "O": -7.204, "C": -7.282,
            }
            self.metal_reference_energies= {'pt': -4.594, 'pd': -4.607, 'ir': -8.133, 'ru': -8.602, 'fe': -7.197, 'co': -6.271, 'mn': -8.433, 
                                            'cu': -3.199, 'ni': -4.874, 'cr': -8.911, 'v': -8.76, 'ti': -7.377, 'mo': -9.638, 'w': -11.785, 
                                            'zr': -8.118, 'hf': -9.553, 'na': -1.24, 'ta': -11.369, 'ag': -2.443, 'au': -2.59, 'zn': -0.66, 
                                            'sn': -3.27, 'sb': -3.851, 'bi': -3.587}

            self.metal_cohesive_energies = {'pt': -4.395, 'pd': -3.103, 'ir': -7.967, 'ru': -8.056, 'fe': -6.572, 'co': -6.025, 'mn': -7.463, 
                                            'cu': -3.163, 'ni': -4.773, 'cr': -8.301, 'v': -8.032, 'ti': -6.277, 'mo': -9.429, 'w': -10.153, 
                                            'zr': -6.744, 'hf': -6.836, 'na': -1.212, 'ta': -9.138, 'ag': -2.408, 'au': -2.557, 'zn': -0.649, 
                                            'sn': -3.194, 'sb': -3.758, 'bi': -3.505}

            self.metal_reduction_potential= {'pt':(1.188,2), 'pd':(0.951,2), 'ir':(1.156,3), 'ru':(0.455,2), 'fe':(-0.447,2), 'co':(-0.28,2),
                                             'mn':(-1.185,2), 'cu':(0.337,2), 'ni':(-0.275,2), 'cr':(-0.74,3), 'v':(-1.13,2), 'ti':(-1.37,3), 
                                             'mo':(-0.20,3), 'w': (-0.119,4), 'na': (-2.71,1), 'ta': (-0.6,3), 'ag':(0.7996,1),'au': (1.52,3),
                                             'zn': (-0.7618,2), 'sn': (-0.13,2), 'bi': (0.308,3)}
            
            self.free_energy_correciton = {
                #      ZPE  + Cp   - TS
                "*O":  0.07 + 0.03 - 0.06,
                "*OH": 0.36 + 0.03 - 0.04,
                "*OOH":0.44 + 0.05 - 0.09,
            }
            self.predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
        else:
            pass

    def _optimize_atoms(self, atoms, max_steps=200):
        atoms = atoms.copy()
        atoms.set_calculator(FAIRChemCalculator(self.predictor, task_name="oc20"))
        opt = BFGS(atoms,logfile=None)
    
        opt_check = opt.run(0.05, max_steps)
        if opt_check:
            energy = atoms.get_potential_energy()
        else:
            energy = None
        return atoms, energy
        
    def _add_adsorbate(self, atoms, adsorbate):
        atoms = atoms.copy()
        center_pos, _, _, _ = get_vnn_positions(atoms)
        
        if adsorbate == '*O':
            atoms += Atoms('O', [center_pos + np.array([0.0,0.0,1.8])], tags=[2])
        elif adsorbate == '*OH':
            atoms += Atoms('O', [center_pos + np.array([0.0,0.0,1.8])], tags=[2])
            atoms += Atoms('H', [center_pos + np.array([0.0,0.8,2.3])], tags=[2])
        elif adsorbate == '*OOH':
            atoms += Atoms('O', [center_pos + np.array([ 0.0,0.0,1.8])], tags=[2])
            atoms += Atoms('O', [center_pos + np.array([-1.2,0.0,2.5])], tags=[2])
            atoms += Atoms('H', [center_pos + np.array([-1.2,1.0,2.5])], tags=[2])

        return atoms

    def calculate_metal_binidng_energy(self, atoms):
        slabs = atoms.copy()
        cavity = atoms.copy()
        
        for atom in slabs:
            atom.tag = 1
            
        center_idx, _, _, _ = get_vnn_idx(slabs)
        metal_element = slabs[center_idx].symbol
        mu = self.metal_reference_energies[metal_element.lower()]
        del cavity[center_idx]

        optimized_slabs, slabs_e = self._optimize_atoms(slabs)
        optimized_cavity, cavity_e = self._optimize_atoms(cavity)

        if None in (slabs_e, cavity_e):
            return None
        return np.round(slabs_e - cavity_e - mu,3)

    def calculate_dissolution_potential(self, atoms):
        slabs = atoms.copy()

        for atom in slabs:
            atom.tag = 1
            
        center_idx, _, _, _ = get_vnn_idx(slabs)
        metal_element = slabs[center_idx].symbol
        
        srp, electron = self.metal_reduction_potential[metal_element.lower()]
        metal_bind_e = self.calculate_metal_binidng_energy(atoms)

        if metal_bind_e == None:
            return None
        return np.round(srp - (metal_bind_e/electron),3)
        
    def calculate_binding_energy(self, atoms, adsorbates):
        results = {'optimized_adslabs':[], 'binding_sites':[]}
        slabs = atoms.copy()
        
        # Reset tag for binding energy calculation
        for atom in slabs:
            atom.tag = 1
        
        optimized_slabs, slabs_e = self._optimize_atoms(slabs)
        if slabs_e == None:
            return None, None
        for adsorbate in adsorbates:
            adslabs = self._add_adsorbate(optimized_slabs, adsorbate)
            optimized_adslabs, adslabs_e = self._optimize_atoms(adslabs)
            if adslabs_e == None:
                return None, None
            
            results['optimized_adslabs'].append(optimized_adslabs)
            results['binding_sites'].append(get_active_string(optimized_adslabs))
            gas_e = sum([self.atomic_reference_energies[atom.symbol] for atom in adslabs if atom.tag == 2])
            free_e = self.free_energy_correciton[adsorbate]
            bind_e = adslabs_e + free_e - slabs_e - gas_e
            
            results[f'{adsorbate}_gibbs_free_bind_e'] = bind_e

        return optimized_slabs, results
        
    def calculate_overpotential(self, results, reaction):
        if reaction == 'ORR':
            dG_OOH = results['*OOH_gibbs_free_bind_e']
            dG_OH = results['*OH_gibbs_free_bind_e']
            dG_O = results['*O_gibbs_free_bind_e']
            
            dG1 = dG_OOH - 4.92 # * + O2 + H+ + e- ¡æ *OOH
            dG2 = dG_O - dG_OOH # *OOH + H+ + e- ¡æ *O + H2O
            dG3 = dG_OH - dG_O  # *O + H+ + e- ¡æ *OH
            dG4 = 0.00 - dG_OH  # *OH + H+ + e- ¡æ * + H2O

            dGs = np.array([dG1, dG2, dG3, dG4])
            RDS = ['deltaG_OOH - 4.92', 'deltaG_O - deltaG_OOH', 'deltaG_OH - deltaG_O', '0.00 - deltaG_OH'][np.argmax(-dGs)]
            UL = -np.max(dGs)
            OP = 1.23 - UL

        return OP, RDS


class PromptingTools:
    def __init__(self):
        pass
    def format_history_prompts(self, history_list:list, start, end):
        history_prompts = ""
        for idx, history in enumerate(history_list[start:end]):
            if history["iteration"] == 0:
                pass
            else:
                proposal = self.format_proposal(history['proposal']['modifications'])
                history_prompts += textwrap.dedent(f"""
                At iteration {history['iteration']}, Design agent suggested {proposal},
                As a result, previous catalys {history['prev_catalyst']} modified in to \
                {history['current_catalyst']}.
                Then, reflection agent demonstrated following self-reflection: {history['reflection']['reflection']}\
                and suggested following catalyst as a starting point of next modification: {history['suggested_catalyst']}
                \n
                """).strip()
        return textwrap.dedent(history_prompts).strip()

    def format_simplified_history(self, history:dict):
        catalyst = history['prev_catalyst'].split(', Gibbs')[0]+')'
        modification = self.format_modification(history['modification'])
        prev_energy = np.array([float(history['prev_catalyst'].split(f'of {ads}: ')[1].split(',')[0]) for ads in ['*OOH','*O','*OH']])
        curr_energy = np.array([float(history['current_catalyst'].split(f'of {ads}: ')[1].split(',')[0]) for ads in ['*OOH','*O','*OH']])
        diff = [f"+{np.round(i,3)} eV" if i > 0 else f"{np.round(i,3)} eV" for i in (curr_energy - prev_energy)]
        
        prompt = textwrap.dedent(
            f"""
            Iteration {history['iteration']}: After applying the modification ({modification}) to the catalyst ({catalyst}), Gibbs free energy changed as follows: 
            *OOH energy: From {prev_energy[0]} to {curr_energy[0]} eV (Change: {diff[0]})
            *O energy: From {prev_energy[1]} to {curr_energy[1]} eV (Change: {diff[1]})
            *OH energy: From {prev_energy[2]} to {curr_energy[2]} eV (Change: {diff[2]}).
            """)
        return prompt
        
    def format_catalyst_string(self, target_type, **kwargs):
        if "Gibbs free energy" in target_type:
            return self._binding_energy_format(**kwargs)
        elif "overpotential" in target_type:
            return self._overpotential_format(**kwargs)

    def format_proposal(self, modifications):
        num_mod = len(modifications)
        if type(modifications[0]) == dict:
            joined_modifications = " and ".join([f"{modifications[k]['modification_type']}-{modifications[k]['parameters']}" for k in range(num_mod)])    
        else:
            joined_modifications = " and ".join([f"{modifications[k].modification_type}-{modifications[k].parameters}" for k in range(num_mod)])
        return joined_modifications

    def format_modification(self, modification):
        if type(modification) != dict:
            modification = modification.model_dump()

        modification_type = modification['modification_type']
        param_1, param_2 = modification['parameters']
        if modification_type == 'substitute_metal':
            prompt = f"Substitute {param_1} metal atom to {param_2}"
        elif modification_type == 'substitute_2nd_shell':
            prompt = f"Substitute one 2nd shell {param_1} atom to {param_2}"
        elif modification_type == 'substitute_coordination':
            prompt = f"Substitute one 1st shell {param_1} atom to {param_2}"
        elif modification_type == 'defect_coordination':
            prompt = f"Remove one 1st shell {param_1} atom to make defect"
        elif modification_type == 'recover_coordination':
            prompt = f"Add one 1st shell {param_1} atom to recover defect"
        elif modification_type == 'modify_ligand':
            prompt = f"Change ligand from {param_1} to {param_2}"
        elif modification_type == 'add_functional_group':
            prompt = f"Add {param_1} functional group to 2nd shell"
        elif modification_type == 'remove_functional_group':
            prompt = f"Remove {param_1} functional group from 2nd shell"
        
        return prompt
    
    def _overpotential_format(self, metal, sites, coordination, heteroatoms, ligand, func_group, defect, energies, rds, target, stability, complexity):
        energies = np.round(energies,2)
        return (
            f'(Metal: {metal}, Binding Site: [*OOH: {sites[0]}, *O: {sites[1]}, *OH: {sites[2]}], Coordination: {coordination}, 2nd Shell: {heteroatoms}, '
            f'Ligand: {ligand}, Functional Group: {func_group}, Number of Defect: {defect}, '
            f'Gibbs Free Energy of *OOH: {energies[2]}, Gibbs Free Energy of *O: {energies[0]}, Gibbs Free Energy of *OH: {energies[1]}, '
            f'Rate-determinig Step: {rds}, Overpotential: {np.round(target,2)} V,  Dissolution Potential: {np.round(stability, 3)} V, Complexity: {complexity})'
        )
    def _binding_energy_format(self, metal, coordination, heteroatoms, ligand, func_group, defect, adsorbate, target):
        return (
            f'(Metal: {metal}, Binding Site: {site}, Coordination: {coordination}, 2nd Shell: {heteroatoms}, '
            f'Ligand: {ligand}, Functional Group: {func_group}, Number of Defect: {defect}, '
            f'Adsorbate: {adsorbate}, Gibbs Free Energy: {np.round(target,2)} eV,  Dissolution Potential: {np.round(stability, 3)} V, Complexity: {complexity})'
                )