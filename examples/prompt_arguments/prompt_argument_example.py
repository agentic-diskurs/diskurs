import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from jinja2 import Template

from diskurs import (
    PromptArgument,
    prompt_field,
    LongtermMemory,
    ConductorAgent,
    Conversation,
    AsynchronousConversationDispatcher,
    Forum,
    tool,
    ImmutableConversation,
    MultiStepAgent,
    PromptValidationError,
)
from diskurs.azure_llm_client import AzureOpenAIClient
from diskurs.entities import RoutingRule, DiskursInput, AgentDescription
from diskurs.filesystem_conversation_store import AsyncFilesystemConversationStore
from diskurs.prompt import ConductorPrompt, MultistepPrompt
from diskurs.tools import ToolExecutor

load_dotenv()

##########################################
#         Setting up the system          #
##########################################


@tool
def retrieve_customer_data(customer_id: str) -> dict:
    """
    A tool that retrieves customer data from a database.
    :param customer_id: The ID of the customer to retrieve data for
    :return: A text description of the customer
    """
    # Simulate a database call
    return {
        "customer_id": customer_id,
        "name": "John Doe Inc",
        "country": "USA",
        "industry": "Software",
        "size": 500,
    }


@tool
def predict_future_growth(country: str, industry: str, size: int) -> float:
    """
    A tool that predicts future growth for a company based on its data.
    Returns a growth index between 0-100.

    :param country: The country where the company is located
    :param industry: The industry of the company
    :param size: The number of employees in the company
    :return: Growth index from 0-100
    """
    # Base growth score
    growth_score = 50.0

    # Country factors
    country_factors = {
        "USA": 15.0,
        "China": 12.0,
        "India": 10.0,
        "Germany": 8.0,
        "UK": 7.0,
        "Japan": 6.0,
        "Canada": 7.0,
        "Australia": 6.0,
        # Default for other countries
    }
    growth_score += country_factors.get(country, 5.0)

    # Industry factors
    industry_factors = {
        "Software": 18.0,
        "Technology": 15.0,
        "Healthcare": 12.0,
        "Renewable Energy": 14.0,
        "Finance": 10.0,
        "Manufacturing": 5.0,
        "Retail": 4.0,
        "Hospitality": 3.0,
        # Default for other industries
    }
    growth_score += industry_factors.get(industry, 5.0)

    # Size factors
    if size < 50:
        growth_score += 8.0  # Small companies can grow faster
    elif size < 200:
        growth_score += 12.0  # Medium companies with good growth potential
    elif size < 500:
        growth_score += 10.0  # Larger medium companies
    else:
        growth_score += 7.0  # Large companies grow slower

    # Ensure the score is between 0-100
    growth_score = max(0.0, min(100.0, growth_score))

    return growth_score


@tool
def calculate_customer_score(size: int, industry: str) -> int:
    """
    A tool that calculates a customer score based on their data.
    :param size: The number of employees in the company
    :param industry: The industry of the company
    :return: Customer score from 0-100
    """
    # Simulate a scoring algorithm
    score = 0
    if size > 100:
        score += 10
    if industry == "Software":
        score += 5
    return score


my_tool_executor = ToolExecutor()
my_tool_executor.register_tools([retrieve_customer_data, calculate_customer_score])

my_agent_dispatcher = AsynchronousConversationDispatcher()

llm_client = AzureOpenAIClient.create(
    type="azure",
    api_version="2024-08-01-preview",
    model_name="gpt-4-0613",
    endpoint=os.getenv("AZURE_AI_ENDPOINT"),
    use_entra_id=True,
    modelMaxTokens=8192,
)

#############################################
# Example of a sales domain expert agent    #
#############################################

sales_agent_description = (
    "The Sales Agent uses the information about a company to decide if it would fit in the ideal customer profile. "
    "For the sales agent to make a reasonable decision, it needs to know details about the company"
)

sales_agent_system_template = Template(
    """You are a sales agent. Your task is to analyze the information about a company and decide if it fits 
    in the ideal customer profile.
    - You carefully analyze the information about the company
    - You will decide if the company fits in the ideal customer profile
    - If the company fits in the ideal customer profile, you will return a verdict of *trustworthy* otherwise *not*
    """
)
sales_agent_user_template = Template(
    """**Company Information**
    - Country: {{ country }}
    - Industry: {{ company_industry }}
    - Size: {{ company_size }}
    - Customer Score: {{ customer_score }}
    - Growth Index: {{ growth_index }}
    -----
    I have to analyze the information about the company, who's details are provided above, and decide if it fits in the ideal customer profile.
    - I will carefully analyze the information about the company
    - I will decide if the company fits in the ideal customer profile
    - If the company fits in the ideal customer profile, I will return a verdict of *trustworthy* otherwise *not*
    - Only if it fits in the ideal customer profile, I will return a management summary about the company and the reasons why it fits
    """
)


@dataclass
class SalesAgentPromptArgument(PromptArgument):
    company_country: Annotated[str, prompt_field(include=False)] = ""
    company_industry: Annotated[str, prompt_field(include=False)] = ""
    company_size: Annotated[int, prompt_field(include=False)] = ""
    verdict: Annotated[str, prompt_field(include=True)] = ""
    management_summary: Annotated[str, prompt_field(include=True)] = ""


sales_agent_prompt = MultistepPrompt(
    agent_description=sales_agent_description,
    system_template=sales_agent_system_template,
    user_template=sales_agent_user_template,
    prompt_argument=SalesAgentPromptArgument,
)


sales_agent = MultiStepAgent(
    name="Sales_Agent",
    prompt=sales_agent_prompt,
    llm_client=llm_client,
    dispatcher=my_agent_dispatcher,
    tool_executor=my_tool_executor,
)

#####################################################
# Example of a customer success domain expert agent #
#####################################################

customer_success_agent_description = (
    "The customer success agent knows how to get detailed information about a company. "
    "For the customer success agent to find information about a company, it needs to know the name of the company"
)

customer_success_system_template = Template(
    """You are a customer success agent. Your task is to obtain detailed information about a company
    - You use the company name to obtain detailed information about a company
    """
)

customer_success_user_template = Template(
    """Company Name: {{ company_name }}

    I have to use the company name to obtain detailed information about the company.
    """
)


@dataclass
class CustomerSuccessAgentPromptArgument(PromptArgument):
    company_name: Annotated[str, prompt_field(include=False)] = ""
    detailed_company_information: Annotated[str, prompt_field(include=True)] = ""


customer_success_agent_prompt = MultistepPrompt(
    agent_description=customer_success_agent_description,
    system_template=customer_success_system_template,
    user_template=customer_success_user_template,
    prompt_argument=CustomerSuccessAgentPromptArgument,
)


customer_success_agent = MultiStepAgent(
    name="Customer_Success_Agent",
    prompt=customer_success_agent_prompt,
    llm_client=llm_client,
    dispatcher=my_agent_dispatcher,
    tool_executor=my_tool_executor,
)

#####################################################
# Example of a customer analyst domain expert agent #
#####################################################

customer_analyst_agent_description = (
    "The customer analyst agent knows how to assess a company to obtain the most important metrics. "
    "For the analyst agent to compute the most important metrics of a company, it uses dedicated tools."
    "For the analyst agent to be able to do its job, it needs the required input data."
)

customer_analyst_system_template = Template(
    """You are a customer analyst agent. Your task is to assess a company to obtain the most important metrics for the company.
    - You use the companies country, industry, and size to obtain metrics about the company
    """
)

customer_analyst_user_template = Template(
    """**Company: {{ company_name }}**
    - Company country: {{ company_country }}
    - Company industry: {{ company_industry }}
    - Company size: {{ company_size }}

    I have to use the the company country, industry, and size to obtain the customer score and future growth index for the company.
    """
)


@dataclass
class CustomerAnalystAgentPromptArgument(PromptArgument):
    company_name: Annotated[str, prompt_field(include=False)] = ""
    detailed_company_information: Annotated[str, prompt_field(include=True)] = ""
    company_score: Annotated[int, prompt_field(include=False)] = ""
    growth_index: Annotated[float, prompt_field(include=False)] = ""


customer_analyst_agent_prompt = MultistepPrompt(
    agent_description=customer_analyst_agent_description,
    system_template=customer_analyst_system_template,
    user_template=customer_analyst_user_template,
    prompt_argument=CustomerSuccessAgentPromptArgument,
)


customer_analyst_agent = MultiStepAgent(
    name="Analyst_Agent",
    prompt=customer_analyst_agent_prompt,
    llm_client=llm_client,
    dispatcher=my_agent_dispatcher,
    tool_executor=my_tool_executor,
)

###########################################
# Example of a customer termination agent #
###########################################

customer_termination_agent = None


##########################################
# Example of a custom conductor agent    #
##########################################

my_conductor_agent_description = ""

my_conductor_system_template = Template(
    """You are a conductor agent. You are adept in evluating the information provided to you, to find the best agent to continue.
    You will receive a user query, a set of agent descriptions and the chat history.
    Your job is to determine which agent should be the next agent based on the provided information.
    {% if example_trajectories %}You will also receive a set of example trajectories to help you understand better how to route the user query.{% endif %}
    
    **User Query:**
    {{ user_query }}
    
    **Agent Descriptions:**
    {% for agent in agent_descriptions %}
    - Name: {{ agent.name }}
    - Description: {{ agent.description }}
    {% if agent.inputs %}- Inputs: {{ agent.inputs }}{% endif %}
    {% if agent.outputs %}- Outputs: {{ agent.outputs }}{% endif %}
    {% endfor %}
    {% if example_trajectories %}
    **Example Trajectories:**
    {% for example in example_trajectories %}
    - Example: {{ example.example }}
    {% endfor %}
    {% endif %}
    """
)

my_conductor_user_template = Template(
    "I have to analyze the information available to me and decide which agent should be the next agent to handle the user query."
    "I have to specify the next agent name which has to be one of the names of the agents provided in the agent descriptions."
)


@dataclass
class MyConductorLongtermMemory(LongtermMemory):
    user_query: str = ""
    company_name: str = ""
    company_country: str = ""
    company_industry: str = ""
    company_size: int = ""
    verdict: str = ""


@dataclass
class MyConductorPromptArgument(PromptArgument):
    agent_descriptions: Annotated[list[dict[str, str]], prompt_field(include=False)] = field(default_factory=list)
    example_trajectories: Annotated[list[dict[str, str]], prompt_field(include=False)] = field(default_factory=list)
    user_query: Annotated[str, prompt_field(include=False)] = ""
    next_agent: Annotated[str, prompt_field(include=True)] = ""


with open(
    Path(__file__).parent.parent.parent / "diskurs" / "assets" / "json_formatting.jinja2", encoding="utf-8"
) as f:
    json_formatting_template = f.read()
    json_formatting_template = Template(json_formatting_template)


def get_in_out(prompt_argument: PromptArgument) -> dict[str, dict[str, str]]:
    """
    Gets information about the inputs and outputs of the agent.
    It uses the prompt arguments fields "prompt_field" to get the names of the inputs and outputs.
    If the field is marked as "include=True", it is considered an output, otherwise an input.
    """
    inputs = {}
    outputs = {}
    for field_name, field_value in prompt_argument.__annotations__.items():
        if hasattr(field_value, "__metadata__") and field_value.__metadata__[0].include:
            outputs[field_name] = getattr(prompt_argument, field_name)
        else:
            inputs[field_name] = getattr(prompt_argument, field_name)
    return {"inputs": inputs, "outputs": outputs}


agent_descriptions: list[AgentDescription] = [
    AgentDescription(
        name=agent.name, description=agent.prompt.agent_description, **get_in_out(agent.prompt.prompt_argument)
    )
    for agent in [customer_success_agent, customer_analyst_agent]
]


def is_valid(prompt_argument: MyConductorPromptArgument) -> bool:
    """
    A rule that checks if the conversation is valid.
    This is a placeholder for the actual logic.
    """
    valid_next_agents = [agent.name for agent in agent_descriptions]
    if prompt_argument in valid_next_agents:
        return True
    else:
        raise PromptValidationError(f"next_agent must be one of {valid_next_agents}")


my_conductor_prompt = ConductorPrompt(
    agent_description=my_conductor_agent_description,
    system_template=my_conductor_system_template,
    user_template=my_conductor_user_template,
    prompt_argument_class=MyConductorPromptArgument,
    longterm_memory=MyConductorLongtermMemory,
    json_formatting_template=json_formatting_template,
    is_valid=is_valid,
)


def can_conclude(conversation: Conversation) -> bool:
    """
    A rule that checks if the conversation can be concluded.
    This is a placeholder for the actual logic.
    """
    longterm_memory: MyConductorLongtermMemory = conversation.active_longterm_memory
    return longterm_memory.verdict in ["trustworthy", "aaa"] and longterm_memory.company_size > 200


my_conductor_finish_rule = [
    RoutingRule(
        name="test_rule_1",
        description="Rule that always returns true",
        condition=can_conclude,
        target_agent=sales_agent.name,
    ),
]


my_conductor_agent = ConductorAgent(
    name="My_Conductor_Agent",
    prompt=my_conductor_prompt,
    llm_client=llm_client,
    handoff=[agent.name for agent in agent_descriptions],
    agent_descriptions=agent_descriptions,
    dispatcher=my_agent_dispatcher,
    rules=my_conductor_finish_rule,
)

agents = [
    sales_agent,
    customer_success_agent,
    customer_analyst_agent,
    my_conductor_agent,
]

my_conversation_store = AsyncFilesystemConversationStore(
    agents=agents,
    conversation_class=ImmutableConversation,
    is_persistent=True,
    storage_path=Path(__file__).parent / "conversations",
)

for agent in agents:
    my_agent_dispatcher.subscribe(topic=agent.name, subscriber=agent)

forum = Forum(
    agents=agents,
    dispatcher=my_agent_dispatcher,
    tool_executor=my_tool_executor,
    first_contact=my_conductor_agent,
    conversation_store=my_conversation_store,
    conversation_class=ImmutableConversation,
)

if __name__ == "__main__":
    asyncio.run(
        forum.ama(
            DiskursInput(
                user_query="What is the customer score of customer 12345?",
                metadata={"customer_id": "12345"},
                conversation_id="12345",
            )
        )
    )
