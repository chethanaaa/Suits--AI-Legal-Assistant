# Imports
import json
from typing import Literal
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain.prompts import PromptTemplate
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

class RuleSelection(BaseModel):
    selected_rule: Literal[
        "habitability_violation",
        "security_deposit_issue",
        "failure_to_provide_essential_services",
        "retaliatory_eviction",
        "illegal_lockout",
        "failure_to_address_health_hazard",
        "unauthorized_entry_by_landlord",
        "notice_of_foreclosure"
    ]

class DecisionReadiness(BaseModel):
    decision_ready: Literal["yes", "no"]


def create_rules_chain(openai_key): 
    # Prompt template for LLM
    prompt_template = """
    You are an expert in tenant-landlord laws. Based on the details of a case, select the most applicable rule from the following knowledge base:

    1. habitability_violation:
    - Conditions: Unresolved maintenance issue, health hazard, tenant notified landlord, no repair in reasonable time.
    - Action: Tenant entitled to rent reduction or abatement.

    2. security_deposit_issue:
    - Conditions: Security deposit not returned, tenant vacated unit, no damages or unpaid rent.
    - Action: Tenant entitled to double the security deposit.

    3. failure_to_provide_essential_services:
    - Conditions: No heat, water, or electricity; tenant provided written notice; landlord noncompliant for 24 hours.
    - Action: Tenant entitled to substitute housing or rent deduction.

    4. retaliatory_eviction:
    - Conditions: Tenant reported violation, landlord issued eviction notice, within six months of the report.
    - Action: Tenant entitled to remain and landlord fined.

    5. illegal_lockout:
    - Conditions: Tenant locked out without court order, lock changed or entry blocked.
    - Action: Tenant entitled to reentry and damages.

    6. failure_to_address_health_hazard:
    - Conditions: Mold or pest issue, tenant has medical report, inspection confirms issue.
    - Action: Tenant entitled to injunctive relief and damages.

    7. unauthorized_entry_by_landlord:
    - Conditions: Landlord entered without notice, no emergency or consent.
    - Action: Tenant entitled to damages for privacy violation.

    8. notice_of_foreclosure:
    - Conditions: Landlord facing foreclosure, tenant not informed in writing.
    - Action: Tenant entitled to terminate the lease.

    Given the following case details:
    {case_details}

    Respond with only the rule name.
    """

    # llm = OllamaFunctions(model="llama3-groq-tool-use", temperature=0)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_key)
    structured_llm = llm.with_structured_output(RuleSelection)

    rules_prompt = PromptTemplate(input_variables=['case_details'], template=prompt_template)

    rules_chain = rules_prompt | structured_llm

    return rules_chain

def create_decision_chain(openai_key):
    prompt_template = """
    You are an expert in tenant-landlord laws. Your task is to determine if there is enough information in the given conversation between the user and the LLM to reasonably assess which rule applies. The rules are based on the following conditions:

    1. Habitability Violation:
    - Conditions: Unresolved maintenance issue, health hazard, tenant notified landlord, no repair in reasonable time.

    2. Security Deposit Issue:
    - Conditions: Security deposit not returned, tenant vacated unit, no damages or unpaid rent.

    3. Failure to Provide Essential Services:
    - Conditions: No heat, water, or electricity; tenant provided written notice; landlord noncompliant for 24 hours.

    4. Retaliatory Eviction:
    - Conditions: Tenant reported violation, landlord issued eviction notice, within six months of the report.

    5. Illegal Lockout:
    - Conditions: Tenant locked out without court order, lock changed or entry blocked.

    6. Failure to Address Health Hazard:
    - Conditions: Mold or pest issue, tenant has medical report, inspection confirms issue.

    7. Unauthorized Entry by Landlord:
    - Conditions: Landlord entered without notice, no emergency or consent.

    8. Notice of Foreclosure:
    - Conditions: Landlord facing foreclosure, tenant not informed in writing.

    To decide if there is enough information:
    - All conditions for at least one rule should be met or clearly implied.
    - If most required conditions for one rule are met or strongly implied, respond with "yes."
    - If critical conditions are missing or highly ambiguous, respond with "no."

    Given the following conversation:
    {conversation}

    Respond with only "yes" or "no".
    """

    # llm = OllamaFunctions(model="llama3-groq-tool-use", temperature=0)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_key)
    structured_llm = llm.with_structured_output(DecisionReadiness)

    decision_prompt = PromptTemplate(input_variables=['conversation'], template=prompt_template)

    decision_chain = decision_prompt | structured_llm

    return decision_chain