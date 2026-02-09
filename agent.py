from catagent.project.agents import get_llm_model, get_design_agent, get_review_agent, get_summarize_agent, get_reflect_agent, get_report_agent
from catagent.project.tools import modification_report_base
from catagent.tools import CalculationTools, ModificationTools, PromptingTools
from catagent.graphs import DesignNode, ReviewNode, CalculationNode, ReflectionNode, SummaryNode, GraphState
from catagent.utils import get_vnn_idx, save_pickle, get_attempt_indices

from pydantic_graph import Graph

from omegaconf import OmegaConf
from ase.io import read
from dataclasses import fields, is_dataclass
from pathlib import Path

import os
import asyncio
import time

BASE_DIR = Path(__file__).resolve().parent
os.environ["GEMINI_API_KEY"] = 'YOUR_API_KEY_HERE'
os.environ["GEMINI_API_KEY"] = 'YOUR_API_KEY_HERE'

def is_primitive(val):
    return isinstance(val, (str, int, float, bool)) or val is None

def is_primitive_list(val):
    if isinstance(val, list):
        return all(is_primitive(v) or isinstance(v, dict) for v in val)
    return False

def is_primitive_dict(val):
    if isinstance(val, dict):
        return all(is_primitive(v) or is_primitive_list(v) for v in val.values())
    return False

def serialize_state(state):
    result = {}
    if not is_dataclass(state):
        return result
    for f in fields(state):
        key = f.name
        val = getattr(state, key)

        if is_primitive(val):
            result[key] = val

        elif is_primitive_list(val):
            result[key] = val

        elif is_primitive_dict(val):
            result[key] = val

        else:
            pass

    return result


def run_catalyst_discovery_agent(
    initial_atoms, 
    target_value,
    run_name,
    strategy="exploration",
    model_name="gpt-4.1-mini",
    save_path="./outputs/",
    knowledge_path="./data/modification_knowledge.txt",
    max_iter=10,
    threshold=0.1,
    num_history=5,
    adsorbates=["*O","*OH","*OOH"],
    reaction="ORR",
    use_history_summary=True,
    calculator_name="UMA",
    temperature=1.0,
):  

    model = get_llm_model(model_name)
    
    design_agent = get_design_agent(model, prompt_type=strategy)
    review_agent = get_review_agent(model, prompt_type=strategy)
    reflect_agent = get_reflect_agent(model, prompt_type=strategy)
    summarize_agent = get_summarize_agent(model, prompt_type=strategy)
    report_agent = get_report_agent(model, prompt_type=strategy) 
    
    calculator = CalculationTools(calculator_name)
    modifier = ModificationTools()
    formatter = PromptingTools()
    
    center_idx, fnn_idx, _, _ = get_vnn_idx(initial_atoms)
    modifier.set_status({
            'metal': initial_atoms[center_idx].symbol,
            'coordination': ["N","N","N","N"],
            'defect_count' : 0,
            'ligand' : "None",
            'functional_group' : ["None", "None"],
            '2nd_shell' : ["C","C","C","C","C","C","C","C"],
            'complexity' : 0,
        })
    
    modifier.set_positions(initial_atoms)
    dst = os.path.join(save_path, run_name)
    os.makedirs(dst,exist_ok=True)
    
    print("Initializing Agent System...")
    try:
        print("Start initial calculation...")
        initial_slabs, initial_result = calculator.calculate_binding_energy(initial_atoms, adsorbates)
        initial_stability = calculator.calculate_dissolution_potential(initial_atoms)
        modification_status = modifier.status
        
        metal = modification_status['metal']
        sites = initial_result['binding_sites']
        coordination = f"{modification_status['coordination']}"
        ligand = modification_status['ligand']
        defect = f"{modification_status['defect_count']}"
        heteroatoms = f"{modification_status['2nd_shell']}"
        functional_group = f"{modification_status['functional_group']}"
        complexity = modification_status['complexity']

        if len(adsorbates) == 1:
            adsorbate = adsorbates[0]
            initial_value = initial_result[f"{adsorbate}_gibbs_free_bind_e"]
            target_type = "Gibbs free energy (ΔG)"
            catalyst_dict ={
                'metal': metal, 'sites': sites[0], 'coordination': coordination, 'heteroatoms': heteroatoms, 'ligand': ligand, 'func_group': functional_group,
                'defect': defect, 'adsorbate': adsorbate, 'target': initial_value, 'stability': initial_stability, 'complexity': complexity
            }
        else:
            initial_value, initial_rds = calculator.calculate_overpotential(initial_result, reaction)
            target_type = "overpotential"
            catalyst_dict ={
                'metal': metal, 'sites': sites, 'coordination': coordination, 'heteroatoms': heteroatoms, 'ligand': ligand, 'func_group': functional_group,
                'defect': defect, 'energies': [initial_result['*O_gibbs_free_bind_e'],initial_result['*OH_gibbs_free_bind_e'],initial_result['*OOH_gibbs_free_bind_e']], 
                'rds': initial_rds, 'target': initial_value, 'stability': initial_stability, 'complexity': complexity
            }

        initial_atoms_desc= formatter.format_catalyst_string(
                    target_type=target_type, **catalyst_dict
                )
        
        print(f"Initial catalyst description : {initial_atoms_desc}")
    except Exception as e:
        print(f"Initialization Failed: {e}")
        return [], [], []

    initial_state = GraphState(
        current_atoms=initial_atoms,
        prev_atoms=initial_atoms,
        calculator=calculator,
        modifier=modifier,
        formatter=formatter,
        design_agent=design_agent,
        review_agent=review_agent,
        reflect_agent=reflect_agent,
        summarize_agent=summarize_agent,
        report_agent=report_agent,
        strategy=strategy,
        target=target_value,
        max_iteration=max_iter,
        threshold=threshold,
        num_history=num_history,
        name=run_name,
        save_path=dst,
        adsorbates=adsorbates,
        reaction=reaction,
        #use_history_summary=use_history_summary,
        current_atoms_desc=initial_atoms_desc,
        prev_atoms_desc=initial_atoms_desc, 
        current_value=initial_value,
        prev_value=initial_value,
        current_time=time.time()
    )
    initial_state.slabs_list.append(initial_atoms)
    initial_state.prev_slabs_list.append(initial_atoms)
    initial_state.history_list.append({
        "iteration": 0,
        "current_catalyst": initial_atoms_desc
    })
    initial_state.status_list.append(modifier.status)
    initial_state.best_state['best_value'] = 100.
    initial_state.best_state['best_atoms'] = initial_atoms
    initial_state.best_state['best_atoms_desc'] = initial_atoms_desc
    initial_state.best_state['best_modifier_status'] = initial_state.modifier.status 
    
    initial_state.temperature['design'] = temperature
    initial_state.temperature['review'] = temperature
    initial_state.temperature['report'] = temperature
    initial_state.temperature['reflection'] = temperature
    initial_state.temperature['summary'] = temperature
    
    if strategy == "knowledge_base":
        with open(knowledge_path, 'r', encoding='utf-8') as f:
            knowledge = f.read()
        initial_state.last_report = knowledge
    elif strategy == "rag_base":
        initial_state.rag_store = modification_report_base(knowledge_path)
    
    initial_state_dict = serialize_state(initial_state)
    save_pickle(dst+'/state.pkl',initial_state_dict)
    
    catalyst_graph = Graph(nodes=[DesignNode, ReviewNode, CalculationNode, ReflectionNode, SummaryNode])
    
    print(f"Start Graph Run. Target: {target_value}")
    
    try:
        final_state = catalyst_graph.run_sync(DesignNode(), state=initial_state)
        print("Graph Finished Successfully.")
        return final_state.output
        
    except Exception as e:
        print(f"\nCRITICAL ERROR during Graph Execution: {e}")
        import traceback
        traceback.print_exc()

        return initial_state
        
        
if __name__ == '__main__':
    params = OmegaConf.load('./config/config.yml')
    
    BASE_DIR = Path(__file__).resolve().parent
    save_root = (BASE_DIR / params.save_path).resolve()
    atoms = read((BASE_DIR / params.atoms_path).resolve())
  
    attempts = get_attempt_indices(
      base_dir=save_root,
      name=params.name,
      n_attempts=params.attempts,
    )
    
    for attempt in attempts:
        run_name = f"{params.name}/{attempt}/"
        run_catalyst_discovery_agent(
            initial_atoms=atoms,
            target_value=params.target,
            run_name=run_name,
            model_name=params.model_name,
            strategy=params.strategy,
            save_path=str(save_root),
            max_iter=params.max_iteration,
            threshold=params.threshold,
            temperature=params.temperature,
            num_history=params.num_history,
            adsorbates=params.adsorbates,
            reaction=params.reaction,
            use_history_summary=params.use_history_summary,
            calculator_name=params.calculator_name
        )
