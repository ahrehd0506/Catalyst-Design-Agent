from typing import List, Optional, Literal, Any, Dict
from dataclasses import dataclass, field

from ase.io import write

from pydantic_ai import ImageUrl
from pydantic_ai.models import ModelSettings
from pydantic_graph import BaseNode, End, Graph, GraphRunContext
from .project.schemas import *
from .tools import *
from .utils import save_pickle, make_figure

from datetime import datetime
import os
import textwrap
import base64
import time

@dataclass
class GraphState:
    calculator: Any
    modifier: Any
    formatter: Any
    
    design_agent: Any
    review_agent: Any
    reflect_agent: Any
    summarize_agent: Any
    report_agent: Any
    
    target:float
    max_iteration:int = 50
    threshold:float = 0.1
    current_iteration:int = 0
    num_history:int = 3
    num_recent_catalysts:int = 5
    name:str = None
    save_path:str = None
    adsorbates: List[str] = field(default_factory=lambda: ["*O"])
    reaction:str = "ORR"
    use_history_summary:bool = True
    use_image:bool = True
    use_review:bool = False
    
    strategy:str = "exploration"
    current_strategy:str = "exploration"
    strategy_change:float = 0.5
    rag_store:Any = None

    history_list: List[dict] = field(default_factory=list)
    prev_slabs_list: List[Any] = field(default_factory=list)
    slabs_list: List[Any] = field(default_factory=list)
    adslabs_list: List[Any] = field(default_factory=list)
    rejected_list: List[Any] = field(default_factory=list)
    status_list: List[Any] = field(default_factory=list)
    
    current_atoms: Any = None
    prev_atoms: Any = None
    current_atoms_desc:str = ""
    prev_atoms_desc:str = ""
    current_value:float = 0.0 
    prev_value:float = 0.0
    
    current_rejection_count:int = 0
    complexity_change:List[int] = field(default_factory=list)
    max_complexity:int = 5

    retry_flag:bool = False
    failed_modification_list: List[Any] = field(default_factory=list)
    retry_count:int = 0
    retry_reason:str = ""
    max_retry_count:int = 5

    total_input_tokens: List[int] = field(default_factory=list)
    total_output_tokens: List[int] = field(default_factory=list)
    total_requests: List[int] = field(default_factory=list)
    iter_input_tokens: int = 0
    iter_output_tokens: int = 0
    iter_requests: int = 0
    current_time:float = 0.0
    iter_time:float = 0.0
    iter_time_list:List[float] = field(default_factory=list)
    
    last_proposal: Optional[DesignOutputs] = None
    last_modification: Optional[ModificationProposal] = None
    last_feedback: Optional[ReviewOutputs] = None
    last_reflection:str = ""
    last_report:str = ""
    recent_catalysts: List[Any] = field(default_factory=list)
    best_state: Dict[str, Any] = field(
        default_factory=lambda: {
            "best_atoms": None,
            "best_atoms_desc": "",
            "best_value": float("inf"),
            "best_modifier_status": {},
        }
    )
    history_summary:str = "Initial State."
    simplified_history:str = "Initial State."
    input_prompt_history: Dict[str, List[Any]] = field(
        default_factory=lambda: {
            "design_prompts": [],
            "review_prompts": [],
            "reflection_prompts": [],
            "summary_prompts": [],
        }
    )

    temperature: Dict[str, float] = field(
        default_factory=lambda: {
            "design": 1,
            "review": 1,
            "report": 1,
            "reflection": 1,
            "summary": 1,
        }
    )
    
    @property
    def target_type(self) -> str:
        if len(self.adsorbates) > 1 and self.reaction:
            return f"{self.reaction} overpotential"        
        else:
            return f"{self.adsorbates[0]} Gibbs free energy (ΔG)"

@dataclass
class DesignNode(BaseNode[GraphState]): 
    async def run(self, ctx: GraphRunContext[GraphState]) -> 'ReviewNode | End':
        state = ctx.state
        print(f"\n Iteration {state.current_iteration + 1} ")
        if state.current_iteration >= state.max_iteration:
            print("\n Max Iterations Reached.")
            return End(state)
        
        if state.strategy == "exploration":
            if state.strategy_change*state.max_iteration < state.current_iteration:
                state.current_strategy = "exploitation"
            else:
                state.current_strategy = "exploration"
                
        elif state.strategy == "random":
            print(f"[Designer] Suggest Random Modification...")
            valid = False
            loop_stack = 0
            
            while not valid:
                loop_stack += 1
                modification_type, params = state.modifier.get_random_modification(state.current_atoms)
                print(f"[Designer] Random Proposed: {modification_type}, {params}")  
                if state.modifier.is_valid_modification(modification_type, params, state.current_atoms)[0]:
                    valid = True
                    
                if loop_stack >= 10:
                    print("[Designer] Too many invalid random modifications, stopping.")
                    return End(state)
                
            modifications = ModificationProposal(
                modification_type = modification_type,
                parameters=params,
                reasoning="None",
            )
            state.last_proposal = DesignOutputs(modifications=[modifications])
            state.last_modification = modifications

            modifications_string = state.formatter.format_proposal([modifications])
            print(f"[Designer] Proposed: {modifications_string}")
            return CalculationNode()
        
        else:
            state.current_strategy = None
        
        
        print(f"[Designer] Thinking...")
        
        if state.strategy == "rag_base":
            inputs = DesignInputs(
                current_atoms_desc=state.current_atoms_desc,
                target_value=str(state.target),
                target_type=state.target_type,
                review_feedback=state.last_feedback.feedback if state.last_feedback else None,
                history_summary=state.history_summary,
                vector_store=state.rag_store
            )
          
        else:
            inputs = DesignInputs(
                current_atoms_desc=state.current_atoms_desc,
                target_value=str(state.target),
                target_type=state.target_type,
                review_feedback=state.last_feedback.feedback if state.last_feedback else None,
                history_summary=state.history_summary,
            )
          
            
        if state.current_strategy == "exploration":
            if state.current_iteration > 0:
                modification_list = [state.formatter.format_modification(history['modification']) for history in state.history_list[1:]]
            else:
                modification_list = "This is first iteration"
            
            strat_prompt = f"""
            ### STRATEGY: EXPLORATION
            Your primary objective is to explore uncertain but plausible regions of the catalyst space
            Rule: 
            - You MUST not suggest the modifications alreday used in history
            - Just make sure the complexity doesn't exceed the maximum, do not care about it below that.

            Below is the list of modifications that alreday used:
            {modification_list}
            
            You must still aim toward the target, but information gain has priority over immediate best performance.
            """

        elif state.current_strategy == "exploitation":
            strat_prompt =  f"""
            ### STRATEGY: EXPLOITATION
            Your primary objective is to explore uncertain but plausible regions of the catalyst space
            Rule: 
            - Choose proper the modifications to reach the target by referring to the Gibbs free energy change in the previous history and report.
            - Try to maintain complexity of catalyst moderate
            - Never suggest the failed combination of the modification + current catalysts already used in history.

            Below is the report that summarized the exploration steps:
            {state.last_report}
            Performance toward the target has priority over novelty.
            """
        
        else:
            strat_prompt = ""
            
            
        if "Gibbs" in state.target_type:
            target_prompt = textwrap.dedent(f"""
            Propose a modification to reach target {state.target} based on given information.
            
            The type of target value is {state.target_type}\
            """).strip()
        elif "overpotential" in state.target_type:
            target_prompt = textwrap.dedent(f"""
            Propose modifications to tune its Gibbs free energy (ΔG) of *O, *OH, *OOH adsorbates to a target value to reduce ORR overpotential based on given information.
            Target Gibbs free energy of *O: {2.46 - state.threshold} ~ {2.46 + state.threshold} eV
            Target Gibbs free energy of *OH: {1.23 - state.threshold} ~ {1.23 + state.threshold} eV
            Target Gibbs free energy of *OOH: {3.69 - state.threshold} ~ {3.69 + state.threshold} eV\
            """).strip()
            
        if state.last_feedback and "reject" in state.last_feedback.decision:
            print(f"[Designer] Thinking again based on feedback...")
            state.last_feedback.decision = "choose_one"
            prev_modifications = state.formatter.format_proposal(state.last_proposal.modifications)
            feedback_prompt = textwrap.dedent(f"""
            Your recent modifications {prev_modifications} are rejected by reviewer.
            The feedback of reviewer is {state.last_feedback.feedback}.
            Please re-propose the modification based on the feedback and given information.
            NEVER suggest same modification with previous modification
            """).strip()
        elif state.retry_flag:
            print(f"[Designer] Suggesting correct modification...")
            
            state.retry_flag = False
            prev_modifications = state.formatter.format_proposal(state.failed_modification_list)
            feedback_prompt = textwrap.dedent(f"""
            Your recent modifications {prev_modifications} are failed.
            The reason of the most recent failure is {state.retry_reason}.
            Please re-propose the modification based on the given format and information.
            NEVER suggest same modification with previous failed modification
            """).strip()

        else:
            feedback_prompt = ""

        if state.use_image:
            image_prompt = "Provided image is side and top view of current catalyst. Please consider geometrical effect too."
        elif state.use_image:
            image_prompt = ""
        
        
        if state.strategy == "no_icl":
            design_prompt = textwrap.dedent(f"""
            {feedback_prompt}\
            {target_prompt}
        
            Current state of catalyst is {state.current_atoms_desc}
            {image_prompt}
            """).strip()
        elif state.strategy == "knowledge_base":
            history_prompt = state.formatter.format_history_prompts(state.history_list, -state.num_history, None)
        
            design_prompt = textwrap.dedent(f"""
            {feedback_prompt}\
            {target_prompt}
        
            Current state of catalyst is {state.current_atoms_desc}
            {image_prompt}
            The {state.num_history} recent modificaitons, reasonings, feedbacks and self-reflections are following:
            {history_prompt}.
            The summary of modification history is {state.history_summary}.
            The simplified history of previous modifications is {state.simplified_history} except recent {state.num_history}.
            
            Please refer knowledge about modification as follow:
            {state.last_report} 
            """).strip()    
        
        else:
            history_prompt = state.formatter.format_history_prompts(state.history_list, -state.num_history, None)
        
            design_prompt = textwrap.dedent(f"""
            {strat_prompt}\
            {feedback_prompt}\
            {target_prompt}
        
            Current state of catalyst is {state.current_atoms_desc}
            {image_prompt}
            The {state.num_history} recent modificaitons, reasonings, feedbacks and self-reflections are following:
            {history_prompt}.
            The summary of modification history is {state.history_summary}.
            The simplified history of previous modifications is {state.simplified_history} except recent {state.num_history}.
            """).strip()
        state.input_prompt_history['design_prompts'].append(design_prompt)
          
        if state.use_image:
            image = make_figure(state.current_atoms)
            encoded_image = base64.b64encode(image).decode("utf-8")
            design_prompt = [
                design_prompt,
                ImageUrl(url=f"data:image/png;base64,{encoded_image}"),
            ]
            
        result = await state.design_agent.run(
            user_prompt=design_prompt,
            deps=inputs,
            model_settings=ModelSettings(temperature=state.temperature['design'])
        )
        if result.usage() is not None:
            u = result.usage()
            state.iter_input_tokens += u.input_tokens
            state.iter_output_tokens += u.output_tokens
            state.iter_requests += u.requests
        
        state.last_proposal = result.output
        for modification in result.output.modifications:
            validity, reason, complexity_change = state.modifier.is_valid_modification(modification.modification_type, modification.parameters, state.current_atoms)
            if not validity:
                print(f"[Designer] Wrong Modification: {modification}")
                state.retry_flag = True
                state.retry_count += 1
                state.retry_reason = reason
                state.failed_modification_list.append(modification)
                state.complexity_change = []
                if state.retry_count > state.max_retry_count:
                    print("[Designer] Too many invalid modifications, stopping.")
                    return End(state)
                return DesignNode()
            else:
                state.retry_flag = False
                state.complexity_change.append(complexity_change)
                pass

        modifications = state.formatter.format_proposal(result.output.modifications)
        print(f"[Designer] Proposed: {modifications}")
        
        return ReviewNode()

@dataclass
class ReviewNode(BaseNode[GraphState]):
    async def run(self, ctx: GraphRunContext[GraphState]) -> 'CalculationNode | DesignNode':
        state = ctx.state
        print(f"[Review] Reviewing proposal...")

        # history check
        recent_history = state.history_list[-state.num_history:]
        
        if not state.use_review:
            state.last_feedback = ReviewOutputs(
                decision='choose_one', selected_index=0, feedback="Skip review at this time" 
            )
            state.last_modification = state.last_proposal.modifications[0]
            state.current_rejection_count = 0
            state.complexity_change = []
            return CalculationNode()
            
        inputs = ReviewInputs(
            proposal=state.last_proposal,
            current_atoms_desc=state.current_atoms_desc,
            target_value=str(state.target),
            history_summary=state.history_summary,
            recent_history=recent_history
        )
        
        if "Gibbs" in state.target_type:
            target_prompt = textwrap.dedent(f"""
            The target Gibbs free energy (ΔG) value is {state.target}\
            """).strip()
        elif "overpotential" in state.target_type:
            target_prompt = textwrap.dedent(f"""
            The target Gibbs free energy (ΔG) to reduce overpotential is below:.
            Target Gibbs free energy of *O: {2.46 - state.threshold} ~ {2.46 + state.threshold} eV
            Target Gibbs free energy of *OH: {1.23 - state.threshold} ~ {1.23 + state.threshold} eV
            Target Gibbs free energy of *OOH: {3.69 - state.threshold} ~ {3.69 + state.threshold} eV\
            """).strip()

        history_prompt = state.formatter.format_history_prompts(state.history_list, -state.num_history, None)
        review_prompt = textwrap.dedent(f"""
        Review given proposal of design agent based on scientific principles and history.
        {target_prompt}

        Current state of catalyst is {state.current_atoms_desc}.
        The proposal of design agent is {state.last_proposal}. Expected complexity change is {state.complexity_change}. \
        Recommended maximum value of complexity is {state.max_complexity}. Please choose one or reject all.
        The current recjection count is {state.current_rejection_count}. If rejection count is over 3, you MUST approved suggestion whatever it is.

        The {state.num_history} recent modificaitons, reasonings and feedbacks is following:
        {history_prompt}.
        The summary of history is {state.history_summary}.   
        The simplified history of previous modifications is {state.simplified_history} except recent {state.num_history}.
        """).strip()
        state.input_prompt_history['review_prompts'].append(review_prompt)
        
        result = await state.review_agent.run(
            user_prompt=review_prompt,
            deps=inputs,
            model_settings=ModelSettings(temperature=state.temperature['review'])
        )
        
        if result.usage() is not None:
            u = result.usage()
            state.iter_input_tokens += u.input_tokens
            state.iter_output_tokens += u.output_tokens
            state.iter_requests += u.requests
            
        state.last_feedback = result.output
        
        if result.output.decision == 'choose_one':
            selected_idx = result.output.selected_index
            print(f"[Reviewer] Index {selected_idx} Approved.")
            state.last_modification = state.last_proposal.modifications[selected_idx]
            state.current_rejection_count = 0
            state.complexity_change = []
            return CalculationNode()
        else:
            print(f"[Reviewer] All Rejected. Feedback: {result.output.feedback}")
            state.rejected_list.append({
                'proposal':state.last_proposal,
                'feedback':state.last_feedback
            })
            state.current_rejection_count += 1
            state.complexity_change = []
            if state.current_rejection_count > 5:
                return End(state)
            return DesignNode()

@dataclass
class CalculationNode(BaseNode[GraphState]):
    async def run(self, ctx: GraphRunContext[GraphState]) -> 'ReflectionNode | End':
        state = ctx.state
        
        try:
            modification_type = state.last_modification.modification_type
            parameters = state.last_modification.parameters

            print(f"[Calculator] Running simulation...")
            print(f"[Calculator] Applying suggested modification to {state.current_atoms_desc}")
            modified_atoms = state.modifier.apply(modification_type, parameters, state.current_atoms)
            
            stability = state.calculator.calculate_dissolution_potential(modified_atoms)
            optimized_slabs, calc_results = state.calculator.calculate_binding_energy(modified_atoms, state.adsorbates)

            if (optimized_slabs == None) or (stability == None) :
                print(f"[Calculator] Optimization failed... Return to Design Node")
                state.retry_flag = True
                state.retry_count += 1
                state.failed_modification_list.append(state.last_modification)
                state.retry_reason = "Optimization Failed"
                state.complexity_change = []
                if state.retry_count > state.max_retry_count:
                    print("[Calculator] Too many failed optimization, stopping.")
                    return End(state)
                return DesignNode()
            else:
                state.retry_count = 0
                state.failed_modification_list = []
            
                
            state.prev_slabs_list.append(state.current_atoms)
            state.prev_value = state.current_value
            state.prev_atoms = state.current_atoms
            state.prev_atoms_desc = state.current_atoms_desc
            
            state.current_atoms = modified_atoms
            state.slabs_list.append(modified_atoms)
            state.adslabs_list.extend(calc_results['optimized_adslabs'])

            # Get parameters
            center_idx, fnn_idx, _, _ = get_vnn_idx(modified_atoms)
            modification_status = state.modifier.status

            metal = modified_atoms[center_idx].symbol
            sites = calc_results['binding_sites']
            coordination = [modified_atoms[i].symbol for i in fnn_idx]
            complexity = modification_status['complexity']
            ligand = modification_status['ligand']
            defect = f"{modification_status['defect_count']}"
            heteroatoms = f"{modification_status['2nd_shell']}"
            functional_group = f"{modification_status['functional_group']}"
            
            # formatting
            if "overpotential" in state.target_type:
                value, rds = state.calculator.calculate_overpotential(calc_results, state.reaction)
                catalyst_dict = {
                        'metal': metal, 'sites': sites, 'coordination': coordination, 'heteroatoms': heteroatoms, 'ligand': ligand, 'func_group': functional_group,
                        'defect': defect, 'energies': [calc_results['*O_gibbs_free_bind_e'],calc_results['*OH_gibbs_free_bind_e'],calc_results['*OOH_gibbs_free_bind_e']], 
                        'rds': rds, 'target': value, 'stability': stability, 'complexity': complexity
                    }
                
            elif "Gibbs" in state.target_type:
                value = calc_results[f"{state.adsorbates[0]}_gibbs_free_bind_e"]
                catalyst_dict ={
                        'metal': metal, 'sites': sites[0], 'coordination': coordination, 'heteroatoms': heteroatoms, 'ligand': ligand, 'func_group': functional_group,
                        'defect': defect, 'adsorbate': state.adsorbates[0], 'target': value, 'stability': stability, 'complexity': complexity
                    }

            state.current_value = value
            current_atoms_desc = state.formatter.format_catalyst_string(target_type=state.target_type, **catalyst_dict)
            state.current_atoms_desc = current_atoms_desc

            if state.best_state['best_value'] > np.abs(state.target - value):
                state.best_state['best_atoms'] = state.current_atoms
                state.best_state['best_atoms_desc'] = state.current_atoms_desc
                state.best_state['best_value'] = np.abs(state.target - value)
                state.best_state['best_modifier_status'] = state.modifier.status
            
            if len(state.recent_catalysts) == state.num_recent_catalysts:
                state.recent_catalysts.pop(0)
            state.recent_catalysts.append([state.current_atoms, state.current_atoms_desc, state.modifier.status])
            
            print(f"[Calculator] Result: {current_atoms_desc} (Target: {state.target})")
            
            '''
            if abs(value - state.target) <= state.threshold:
                print("\n>>> TARGET REACHED! Stopping.")
                final_entry = {
                    "iteration": state.current_iteration+1,
                    "prev_catalyst": state.prev_atoms_desc,
                    "current_catalyst": state.current_atoms_desc,
                    "proposal": state.last_proposal.model_dump(),
                    "modification": state.last_modification.model_dump(),
                    "feedback": state.last_feedback.model_dump(),
                }
                state.history_list.append(final_entry)
                return End(state)
            '''
            
            return ReflectionNode()

        except Exception as e:
            print(f"\n[Error] Calculation Failed: {e}")
            import traceback
            traceback.print_exc()
            return End(state)

@dataclass
class ReflectionNode(BaseNode[GraphState]):
    async def run(self, ctx: GraphRunContext[GraphState]) -> 'SummaryNode':
        state = ctx.state
        
        if state.strategy in ["no_icl","random"]:
            print("[Reflection] Skipping self-reflection...")
            history = {
                "iteration": state.current_iteration+1,
                "prev_catalyst": state.prev_atoms_desc,
                "current_catalyst": state.current_atoms_desc,
                "proposal": state.last_proposal.model_dump(),
                "modification": state.last_modification.model_dump(),
            }
        
            state.history_list.append(history)
            return SummaryNode()
        
        print(f"[Reflection] Analyzing result...")
        recent_history = state.history_list[-state.num_history:]
            
        if state.strategy == "exploration":
            if state.current_strategy == "exploration":
                strat_prompt = """
                ### STRATEGY: EXPLORATION
                Your priority is information gain and coverage of the catalyst space.
                    - Prefer continuing from RECENT catalyst that has potential to expand catalyst space and avoid choose catalysts already used. 
                    - MUST explore as many catalyst + modification possibilities as possible.
                """
            elif state.current_strategy == 'exploitation':
                strat_prompt = """
                ### STRATEGY: EXPLOITATION
                Your priority is fast convergence toward the target with reliable, low-risk steps.
                    - Recommends you judge to be the most suitable catalyst for the designer to achieve the target..
                    - Balance the use of RECENT and BEST. DO NOT stick to the BEST and select catalysts with the potential to achieve new BEST. 
                """
        else:
            strat_prompt = ""
            
        if "Gibbs" in state.target_type:
            target_prompt = textwrap.dedent(f"""
            Please write a brief post-action reflection on the modification in less than five sentences, \
            explaining how successful it was in achieving {state.target} eV and the reasons for its success or failure.\
            """).strip()
        elif "overpotential" in state.target_type:
            target_prompt = textwrap.dedent(f"""
            Please write a brief post-action reflection on the modification in less than five sentences, \
            explaining how successful it was in achieving {2.46 - state.threshold} ~ {2.46 + state.threshold} eV for Gibbs free energy of *O\
            {1.23 - state.threshold} ~ {1.23 + state.threshold} for Gibbs free energy of *OH and \
            {3.69 - state.threshold} ~ {3.69 + state.threshold} for Gibbs free energy of *OOH, and \
            the reasons for its success or failure.\
            """).strip()

        history_prompt = state.formatter.format_history_prompts(state.history_list, -state.num_history, None)
        recent_catalysts_prompt = "\n".join([f"[{idx}] :: {catal[1]}" for idx, catal in enumerate(state.recent_catalysts)]) + " (recent)"

        if state.use_image:
            image_prompt = textwrap.dedent(f"""
            Provided images are top and side view of optimized slabs and catalysts with adsorbates.\
            If there are anomalies like atomic distortion within an optimized slab, please reflect on that point as well.
            """).strip()
        else:
            image_prompt = ""
            
        reflect_prompt = textwrap.dedent(f"""
        {strat_prompt}\
        
        Design agent suggested following hypothesis and modification.
        Hypothesis: {state.last_modification.reasoning}
        Modification: {state.formatter.format_modification(state.last_modification)}

        After completing the modification, we obtained the following catalyst
        Before modification: {state.prev_atoms_desc}
        After modification: {state.current_atoms_desc}
        {target_prompt}
        {image_prompt}
        
        The {state.num_history} recent modificaitons, reasonings and feedbacks is following:
        {history_prompt}.
        The summary of history is {state.history_summary}
        The simplified history of previous modifications is {state.simplified_history} except recent {state.num_history}.
        Recommended maximum value of complexity is {state.max_complexity}.

        Global best catalyst found so far is {state.best_state['best_atoms_desc']}.
        Recent catalyst list is following (oldest to newest):
        {recent_catalysts_prompt}
        If all target value of recent catalysts is fall apart from the best, choose 'best'
        """).strip()
        state.input_prompt_history['reflection_prompts'].append(reflect_prompt)

        if state.use_image:
            reflect_prompt = [reflect_prompt] 
            slab_image = make_figure(state.slabs_list[-1])
            adslab_images = [make_figure(adslab) for adslab in state.adslabs_list[-3:]]
            
            encoded_image = base64.b64encode(slab_image).decode("utf-8")
            reflect_prompt.append(ImageUrl(url=f"data:image/png;base64,{encoded_image}"))

            for adslab_image in adslab_images:
                encoded_image = base64.b64encode(adslab_image).decode("utf-8")
                reflect_prompt.append(ImageUrl(url=f"data:image/png;base64,{encoded_image}"))
        
        result = await state.reflect_agent.run(
            user_prompt=reflect_prompt,
            model_settings=ModelSettings(temperature=state.temperature['reflection'])
            )
        if result.usage() is not None:
            u = result.usage()
            state.iter_input_tokens += u.input_tokens
            state.iter_output_tokens += u.output_tokens
            state.iter_requests += u.requests
            
        state.last_reflection = result.output
        next_atoms_type = result.output.next_catalyst_type
        next_atoms_index = result.output.next_catalyst_index

        if next_atoms_type == 'recent':
            print(f"[Reflection] Index {next_atoms_index+1}/{len(state.recent_catalysts)} Selected")
            if next_atoms_index >= state.num_recent_catalysts:
                print("[Reflection] next_catalyst_index is out of range, keep current catalyst.")
                next_atoms = state.current_atoms
                next_atoms_desc = state.current_atoms_desc 
                state.modifier.set_status(state.recent_catalysts[-1][2])
                state.status_list.append(state.recent_catalysts[-1][2])
            else:
                next_atoms = state.recent_catalysts[next_atoms_index][0]
                next_atoms_desc = state.recent_catalysts[next_atoms_index][1]
                state.modifier.set_status(state.recent_catalysts[next_atoms_index][2])
                state.status_list.append(state.recent_catalysts[next_atoms_index][2])
        elif next_atoms_type == 'best':
            print(f"[Reflection] Best Catalyst Selected")
            next_atoms = state.best_state['best_atoms']
            next_atoms_desc = state.best_state['best_atoms_desc']
            state.modifier.set_status(state.best_state['best_modifier_status'])
            state.status_list.append(state.best_state['best_modifier_status'])
        else:
            print("[Reflection] Invalid next_catalyst_type, keep current catalyst.")
            next_atoms = state.current_atoms
            next_atoms_desc = state.current_atoms_desc
            
        history = {
            "iteration": state.current_iteration+1,
            "prev_catalyst": state.prev_atoms_desc,
            "current_catalyst": state.current_atoms_desc,
            "proposal": state.last_proposal.model_dump(),
            "modification": state.last_modification.model_dump(),
            "feedback": state.last_feedback.model_dump(),
            "reflection": state.last_reflection.model_dump(),
            "suggested_catalyst": next_atoms_desc,
            "history": state.history_summary,
        }
        
        state.history_list.append(history)
        state.current_atoms = next_atoms
        state.current_atoms_desc = next_atoms_desc

        return SummaryNode()

@dataclass
class SummaryNode(BaseNode[GraphState]):
    async def run(self, ctx: GraphRunContext[GraphState]) -> 'DesignNode':
        state = ctx.state
        
        if state.strategy in ["no_icl","random"]:
            state.current_iteration += 1
            state.total_input_tokens.append(state.iter_input_tokens)
            state.total_output_tokens.append(state.iter_output_tokens)
            state.total_requests.append(state.iter_requests)
    
            state.iter_input_tokens = 0
            state.iter_output_tokens = 0
            state.iter_requests = 0
            state.retry_flag = False
            state.retry_count = 0
            state.complexity_change = []
    
            state.iter_time = np.round(time.time() - state.current_time,4)
            state.iter_time_list.append(state.iter_time)
            state.current_time = time.time()
            
            if state.save_path is not None:
                write(f'{state.save_path}/slabs_list.traj',state.slabs_list)
                write(f'{state.save_path}/prev_slabs_list.traj',state.prev_slabs_list)
                write(f'{state.save_path}/adslabs_list.traj',state.adslabs_list)
                save_pickle(f'{state.save_path}/history.pkl',state.history_list)
                save_pickle(
                    f'{state.save_path}/cost.pkl',
                    {
                        "input_tokens": state.total_input_tokens,
                        "output_tokens": state.total_output_tokens,
                        "requests": state.total_requests,
                        "times": state.iter_time_list
                    }
                )
                save_pickle(
                    f'{state.save_path}/etc.pkl', 
                    {
                        "exploration_report": state.last_report,
                        "input_prompts": state.input_prompt_history
                    }
                )
                
            return DesignNode()
            
        elif state.strategy == "exploration":
            if state.strategy_change*state.max_iteration == state.current_iteration:
                print(f"[Report] Writing report...")
                report_prompt = textwrap.dedent(f"""
                Following is detailed history:
                {state.formatter.format_history_prompts(state.history_list, None, None)}
    
                Please write the report based on history
                """).strip()
    
                result = await state.report_agent.run(
                    user_prompt=report_prompt,
                    model_settings=ModelSettings(temperature=state.temperature['summary'])
                )
                state.last_report = result.output
                    
        else:
            pass
          
        if state.num_history >= state.current_iteration:
            pass
        else:
            history_text = ""
            for idx, h in enumerate(state.history_list[1:-state.num_history]):
                if 'prev_catalyst' not in h:
                    continue
                history_text += state.formatter.format_simplified_history(h)
            history_text = str(history_text)
            
            print(f"[Summary] Updating history summary...")
            if state.use_history_summary:
                summary_prompt = textwrap.dedent(f"""
                Following is recent summarized history by yourself:
                {state.history_summary}
                And following is recent detailed history:
                {state.formatter.format_history_prompts(state.history_list, -state.num_history - 3, -state.num_history)}
                
                Please summarize history based on given summarized and detailed history\
                """).strip()
                state.input_prompt_history['summary_prompts'].append(summary_prompt)
                
                result = await state.summarize_agent.run(
                    user_prompt=summary_prompt,
                    model_settings=ModelSettings(temperature=state.temperature['summary'])
                )
                if result.usage() is not None:
                    u = result.usage()
                    state.iter_input_tokens += u.input_tokens
                    state.iter_output_tokens += u.output_tokens
                    state.iter_requests += u.requests
                    
                state.history_summary = result.output
                state.simplified_history = history_text
            else:
                state.simplified_history = history_text

        if state.name is None:
            state.name = datetime.now().strftime("%Y-%m-%d")
            
            
        state.current_iteration += 1
        state.total_input_tokens.append(state.iter_input_tokens)
        state.total_output_tokens.append(state.iter_output_tokens)
        state.total_requests.append(state.iter_requests)

        state.iter_input_tokens = 0
        state.iter_output_tokens = 0
        state.iter_requests = 0
        state.retry_flag = False
        state.retry_count = 0
        state.complexity_change = []
    
        state.iter_time = np.round(time.time() - state.current_time,4)
        state.iter_time_list.append(state.iter_time)
        state.current_time = time.time()
        
        
        if state.save_path is not None:
            write(f'{state.save_path}/slabs_list.traj',state.slabs_list)
            write(f'{state.save_path}/prev_slabs_list.traj',state.prev_slabs_list)
            write(f'{state.save_path}/adslabs_list.traj',state.adslabs_list)
            save_pickle(f'{state.save_path}/history.pkl',state.history_list)
            save_pickle(
                f'{state.save_path}/cost.pkl',
                {
                    "input_tokens": state.total_input_tokens,
                    "output_tokens": state.total_output_tokens,
                    "requests": state.total_requests,
                    "times": state.iter_time_list
                }
            )
            save_pickle(
                f'{state.save_path}/etc.pkl', 
                {
                    "exploration_report": state.last_report,
                    "input_prompts": state.input_prompt_history
                }
            )
            
        return DesignNode()
