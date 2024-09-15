import os

from dotenv import load_dotenv

from diskurs.multistep_agent import MultiStepAgent
from diskurs.entities import Conversation

load_dotenv()


def test_create():
    agent = MultiStepAgent.create(llm_api_key=os.getenv("OPENAI_API_KEY"), llm_model="gpt-4o-mini")
    assert isinstance(agent, MultiStepAgent)


def test_invoke():
    agent = MultiStepAgent.create(os.getenv("OPENAI_API_KEY"), llm_model="gpt-4o-mini")
    agent_input = "Hello World"

    res = agent.invoke(Conversation(agent_input))
    assert isinstance(res, Conversation)
    print(res)
