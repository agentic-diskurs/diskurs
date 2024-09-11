import os
from dataclasses import dataclass

from dotenv import load_dotenv
from jinja2 import Template

from agent import Agent
from entities import (
    PromptArgument,
)
from llm_client import OpenAILLMClient
from prompt import Prompt
from tools import tool, ToolExecutor

load_dotenv()


@tool
def get_current_temperature(location: str, unit: str) -> str:
    """
    Get the current temperature for a specific location.

    :param location: The city and state, e.g., San Francisco, CA.
    :param unit: The temperature unit to use, either 'Celsius' or 'Fahrenheit'.

    :return: A string with the current temperature and unit for the specified location.
    """

    temperature_data = {
        "San Francisco, CA": {"Celsius": 18, "Fahrenheit": 64},
        "New York, NY": {"Celsius": 25, "Fahrenheit": 77},
    }

    temp_info = temperature_data.get(location, {"Celsius": 20, "Fahrenheit": 68})

    temperature = temp_info[unit]
    return f"The current temperature in {location} is {temperature}° {unit}."


@tool
def get_rain_probability(location: str) -> str:
    """
    Get the probability of rain for a specific location.

    :param location: The city and state, e.g., San Francisco, CA.

    :return: A string with the rain probability for the specified location.
    """

    rain_probability_data = {
        "San Francisco, CA": 20,  # 20% chance of rain
        "New York, NY": 50,  # 50% chance of rain
    }

    probability = rain_probability_data.get(location, 30)  # Default to 30%

    return f"The probability of rain in {location} is {probability}%."


# TODO: make it, such that if the user does not provide PromptArgument or validate/is_final functions, the system will provide default ones
system_template = Template(
    """You are a weather bot. Use the provided functions to answer questions."""
)


@dataclass
class SystemPromptArgument(PromptArgument):
    pass


user_template = Template("""{{ content }}""")


@dataclass
class UserPromptArgument(PromptArgument):
    content: str = ""


def is_valid(arg: UserPromptArgument) -> bool:
    return True


def is_final(arg: UserPromptArgument) -> bool:
    return True


prompt = Prompt(
    system_template=system_template,
    user_template=user_template,
    system_prompt_argument=SystemPromptArgument,
    user_prompt_argument=UserPromptArgument,
    is_valid=is_valid,
    is_final=is_final,
)

tool_executor = ToolExecutor()
tool_executor.register_tools([get_current_temperature, get_rain_probability])

llm_client = OpenAILLMClient.create(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

agent = Agent(
    name="weather_bot",
    prompt=prompt,
    llm_client=llm_client,
    tool_executor=tool_executor,
    tools=[],
)

agent.register_tools([get_current_temperature, get_rain_probability])

if __name__ == "__main__":
    res = agent.invoke("What's the weather like in Boston today?")
    print(res)
