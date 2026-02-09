from pydantic import BaseModel, Field, conint
from dataclasses import dataclass
from typing import List, Optional, Literal, Any, Dict
from langchain_community.vectorstores import FAISS

@dataclass
class DesignInputs:
    current_atoms_desc: str
    target_value: str
    target_type: str
    review_feedback: Optional[str] = None
    history_summary: Optional[str] = None
    vector_store: FAISS = None


class ModificationProposal(BaseModel):
    modification_type: str = Field(
        description="The type of modification applied"
    )
    parameters: List[str] = Field(
        description="The parameters used for this modification"
    )
    reasoning: str = Field(
        description="Why this modification was chosen and how it is expected to affect the target property"
    )

class DesignOutputs(BaseModel):
    modifications: List[ModificationProposal] = Field(
        description="A list of one candidate modification for the current SAC",
        min_length=1,
        max_length=1,
    )

@dataclass
class ReviewInputs:
    proposal: DesignOutputs
    current_atoms_desc: str
    target_value: str
    history_summary: str
    recent_history: List[Dict]

class ReviewOutputs(BaseModel):
    decision: Literal['choose_one', 'reject_all'] = Field(
        description=(
            "'choose_one': Select exactly one modification from the Designer's proposals. "
            "'reject_all': Reject all proposed modifications."
        )
    )
    selected_index: Optional[int] = Field(
        default=None,
        description=(
            "If decision is 'choose_one', this is the 0-based index of the selected "
            "modification in proposal.modifications. Must be None when decision is 'reject_all'."
        )
    )
    feedback: str = Field(
        description=(
            "A global summary of the evaluation. If one modification is chosen, briefly explain "
            "why it is preferred and why the others are less suitable. If all are rejected, "
            "identify the common issues and give guidance for improvement in the next iteration."
        )
    )

class ReflectionOutputs(BaseModel):
    reflection: str = Field(
        description="Concise reflection (≤5 sentences) on whether the modification succeeded or failed and why."
    )
    next_catalyst_type: Literal['recent', 'best'] = Field(
        description=(
            "Choose the source of the next starting catalyst. "
            "'recent' means pick one candidate from the N most recent catalysts (RECENT list). "
            "'best' means revert to the global best catalyst found so far (BEST)."
        )
    )
    next_catalyst_index: conint(ge=0) = Field(
        description=(
            "If next_catalyst_type is 'recent', choose the 0-based index into RECENT. (0=oldest, N-1=most recent)"
            "If next_catalyst_type is 'best', set this to 0 (dummy value, ignored)."
        )
    )
        
    next_catalyst_reason: str = Field(
        description=(
            "Short justification (4~5 sentences) for why this catalyst is the best starting point for the next iteration and why didn't choose other options."
            "If you choose 'recent', explain why didn't choose 'best'."
        )
    )
