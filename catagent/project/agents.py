from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel

from .tools import knowledge_for_review_agent, knowledge_for_design_agent, knowledge_for_reflect_agent, modification_report_for_design_agent
from .schemas import DesignInputs, DesignOutputs, ReflectionOutputs, ReviewInputs, ReviewOutputs
from .prompts import get_design_prompt, get_reflect_prompt, get_summarize_prompt, get_report_prompt, get_review_prompt

import os

def get_llm_model(model_name):
    if 'gpt' in model_name:
        model = OpenAIChatModel(model_name)
    elif 'gemini' in model_name:
        model = GoogleModel(model_name)
    return model 

def get_design_agent(model, prompt_type="exploration"):
    system_prompt = get_design_prompt(prompt_type)
    
    if prompt_type=="rag_base":
        agent = Agent(
            model,
            name="DesignAgent",
            deps_type=DesignInputs,       
            output_type=DesignOutputs,
            tools=[modification_report_for_design_agent],
            system_prompt=system_prompt,   
        )
               
    else:
        agent = Agent(
            model,
            name="DesignAgent",
            deps_type=DesignInputs,       
            output_type=DesignOutputs,
            tools=[knowledge_for_design_agent],
            system_prompt=system_prompt,   
        )
    return agent
    
def get_reflect_agent(model, prompt_type="exploration"):
    system_prompt = get_reflect_prompt(prompt_type)
    
    agent = Agent(
        model,
        name="ReflectAgent",
        output_type=ReflectionOutputs,
        tools=[knowledge_for_reflect_agent],
        system_prompt=system_prompt,   
    )
    return agent

def get_summarize_agent(model, prompt_type="exploration"):
    system_prompt = get_summarize_prompt(prompt_type)
    
    agent = Agent(
        model,
        name="SummarizeAgent",
        system_prompt=system_prompt
    )
    return agent

def get_report_agent(model, prompt_type="exploration"):
    system_prompt = get_report_prompt(prompt_type)
    
    agent = Agent(
        model,
        name="ReportAgent",
        system_prompt=system_prompt,
    )
    return agent
    
def get_review_agent(model, prompt_type="exploration"):
    system_prompt = get_review_prompt(prompt_type)
    
    agent = Agent(
    model,
    name="ReviewAgent",
    deps_type=ReviewInputs,
    output_type=ReviewOutputs,
    tools=[knowledge_for_review_agent],
    system_prompt=system_prompt    
    )
    return agent
    