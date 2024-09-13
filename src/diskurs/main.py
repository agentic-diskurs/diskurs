from dataclasses import asdict
from pathlib import Path
from typing import Callable

from agent import Agent
from config import load_config_from_yaml, ToolConfig
from llm_client import OpenAILLMClient, AzureOpenAIClient
from messaging import SynchronousConversationDispatcher
from prompt import Prompt
from tools import ToolExecutor, load_tools

llm_map = {"openai": OpenAILLMClient.create, "azure": AzureOpenAIClient.create}
# TODO: better naming for agent types, and consider enum or something
agent_map = {
    "multistep": Agent,
}

# TODO: Consider making methods non-static, to be able to instantiate the class, and keep the agent_map and
#  llm_map as class attributes. Doing so would allow the user to easily extend the class and add new agent types and
#  LLM clients, by e.g. calling `Forum.register_agent_type("new_agent_type", NewAgentType)`
#  and Forum.register_llm_client("new_llm", NewLLMClient)`.


class Forum:
    def __init__(
        self,
        agents: list[Agent],
        dispatcher: SynchronousConversationDispatcher,
        tool_executor: ToolExecutor,
        first_contact: Agent,
    ):
        self.agents = agents
        self.dispatcher = dispatcher
        self.tool_executor = tool_executor
        self.conductor = first_contact

    @classmethod
    def create_forum(cls, config_path: Path):
        config = load_config_from_yaml(config_path)

        tool_executor = cls.create_executor()
        llm_clients = cls.create_llm_clients(config.llms)

        tools = cls.load_tools(config.tools)

        dispatcher = cls.create_dispatcher()

        tool_executor.register_tools(tools)

        agents = cls.create_agents(config.agents, llm_clients, dispatcher, tool_executor, tools)

        first_contact = next((agent for agent in agents if agent.name == config.first_contact), None)

        return cls(agents=agents, dispatcher=dispatcher, tool_executor=tool_executor, first_contact=first_contact)

    @classmethod
    def create_executor(cls):
        return ToolExecutor()

    @classmethod
    def create_llm_clients(cls, llm_config):
        return {llm_conf.name: llm_map[llm_conf.type](**asdict(llm_conf)) for llm_conf in llm_config}

    @classmethod
    def load_tools(cls, tools: list[ToolConfig]) -> list[Callable]:
        return load_tools(tools)

    @classmethod
    def create_dispatcher(cls):
        return SynchronousConversationDispatcher()

    @classmethod
    def create_agents(cls, agent_configs, llm_clients, dispatcher, tool_executor, tools) -> list[Agent]:
        agents = []
        for agent_config in agent_configs:
            prompt = Prompt.create(
                location=agent_config.prompt.prompt_assets,
                system_prompt_argument_class=agent_config.prompt.system_prompt_argument_class,
                user_prompt_argument_class=agent_config.prompt.user_prompt_argument_class,
            )
            agent_tools = [tool for tool in tools if tool.name in agent_config.tools]

            agent = agent_map[agent_config.type].create(
                name=agent_config.name,
                prompt=prompt,
                llm_client=llm_clients[agent_config.llm],
                dispatcher=dispatcher,
                tool_executor=tool_executor,
                tools=agent_tools,
            )

            for topic in agent_config.topics:
                dispatcher.subscribe(topic=topic, subscriber=agent)

            agents.append(agent)

        return agents

    def ama(self, question: str):
        answer = self.conductor.process_conversation(question)
        return answer
