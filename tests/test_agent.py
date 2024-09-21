import os

from dotenv import load_dotenv


from diskurs.entities import Conversation

load_dotenv()


def test_create():
    agent = MultiStepAgent.create(,
    assert isinstance(agent, MultiStepAgent)


def test_invoke():
    agent = MultiStepAgent.create(os.getenv("OPENAI_API_KEY"),,
    agent_input = "Hello World"

    res = agent.invoke(Conversation(agent_input))
    assert isinstance(res, Conversation)
    print(res)
