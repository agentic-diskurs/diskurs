import importlib
from dataclasses import asdict
from pathlib import Path
from typing import List

from config import load_config_from_yaml
from protocols import Agent, ConversationParticipant
from registry import AGENT_REGISTRY, LLM_REGISTRY, TOOL_EXECUTOR_REGISTRY, DISPATCHER_REGISTRY, PROMPT_REGISTRY
from tools import load_tools
from utils import load_module_from_path


class Forum:
    def __init__(
        self,
        agents: List[Agent],
        dispatcher,
        tool_executor,
        first_contact: ConversationParticipant,
    ):
        self.agents = agents
        self.dispatcher = dispatcher
        self.tool_executor = tool_executor
        self.conductor = first_contact

    def ama(self, question: dict):
        answer = self.dispatcher.run(self.conductor, question)
        return answer


class ForumFactory:
    def __init__(self, config_path: Path, base_path: Path):
        self.config = load_config_from_yaml(config=config_path, base_path=base_path)
        self.llm_clients = {}
        self.agents = []
        self.tools = []
        self.dispatcher = None
        self.tool_executor = None
        self.first_contact = None
        self.modules_to_import = ["llm_client", "dispatcher", "agent", "conductor_agent", "prompt"]

    # TODO: find a cleaner solution for modules_to_import

    def create_forum(self) -> Forum:
        self.import_modules()
        self.load_custom_modules()
        self.create_tool_executor()
        self.load_and_register_tools()
        self.create_dispatcher()
        self.create_llm_clients()
        self.create_agents()
        self.prepare_conductors()
        self.identify_first_contact_agent()

        return Forum(
            agents=self.agents,
            dispatcher=self.dispatcher,
            tool_executor=self.tool_executor,
            first_contact=self.first_contact,
        )

    def import_modules(self):
        """Dynamically import modules required for registration."""
        for module_name in self.modules_to_import:
            importlib.import_module(name=module_name, package="diskurs")

    def load_custom_modules(self):
        """Load custom modules specified in the configuration."""
        for module_path_str in self.config.custom_modules:
            module_path = Path(module_path_str)
            load_module_from_path(module_path.stem, module_path)

    def create_tool_executor(self):
        """Create a tool executor instance based on the configuration."""
        tool_executor_cls = TOOL_EXECUTOR_REGISTRY.get(self.config.tool_executor_type)
        if tool_executor_cls is None:
            raise ValueError(f"ToolExecutor type '{self.config.tool_executor_type}' is not registered.")
        self.tool_executor = tool_executor_cls()

    def load_and_register_tools(self):
        """Load and register tools with the tool executor."""
        self.tools = load_tools(self.config.tools)
        self.tool_executor.register_tools(self.tools)

    def create_dispatcher(self):
        """Create a dispatcher instance based on the configuration."""
        dispatcher_cls = DISPATCHER_REGISTRY.get(self.config.dispatcher_type)
        if dispatcher_cls is None:
            raise ValueError(f"Dispatcher type '{self.config.dispatcher_type}' is not registered.")
        self.dispatcher = dispatcher_cls()

    def create_llm_clients(self):
        """Create LLM clients based on the configuration."""
        for llm_conf in self.config.llms:
            llm_type = llm_conf.type
            llm_cls = LLM_REGISTRY.get(llm_type)
            if llm_cls is None:
                raise ValueError(f"LLM type '{llm_type}' is not registered.")
            self.llm_clients[llm_conf.name] = llm_cls.create(**asdict(llm_conf))

    def create_agents(self):
        """Create agent instances based on the configuration."""
        for agent_conf in self.config.agents:
            additional_args = {}

            if hasattr(agent_conf, "prompt"):
                prompt_cls = PROMPT_REGISTRY.get(agent_conf.prompt.type)
                prompt = prompt_cls.create(**asdict(agent_conf.prompt))
                additional_args["prompt"] = prompt
            if hasattr(agent_conf, "llm"):
                additional_args["llm_client"] = self.llm_clients[agent_conf.llm]
            if hasattr(agent_conf, "tools"):
                agent_conf.tools = [tool for tool in self.tools if tool.name in agent_conf.tools]
                additional_args["tool_executor"] = self.tool_executor
            if hasattr(agent_conf, "topics") and self.dispatcher:
                additional_args["topics"] = agent_conf.topics
            if hasattr(agent_conf, "finalizer_name"):
                additional_args["finalizer_name"] = agent_conf.finalizer_name

            additional_args["dispatcher"] = self.dispatcher

            agent_type = agent_conf.type
            agent_cls = AGENT_REGISTRY.get(agent_type)
            if agent_cls is None:
                raise ValueError(f"Agent type '{agent_type}' is not registered.")

            agent = agent_cls.create(name=agent_conf.name, **additional_args)

            self.dispatcher.subscribe(topic=agent.name, subscriber=agent)

            self.agents.append(agent)

    def prepare_conductors(self):
        # get the configs from each conductor agent
        conductor_configs = [agent for agent in self.config.agents if agent.type == "conductor"]

        # for each conductor agent, get the agent descriptions for the agents that are in the topics
        for conf in conductor_configs:
            conductor = next((agent for agent in self.agents if agent.name == conf.name))
            conductor.agent_descriptions = {
                agent.name: agent.prompt.agent_description for agent in self.agents if agent.name in conf.topics
            }

    def identify_first_contact_agent(self):
        """Identify the first contact agent from the list of agents."""
        first_contact_name = self.config.first_contact
        self.first_contact = next((agent for agent in self.agents if agent.name == first_contact_name), None)
        if self.first_contact is None:
            raise ValueError(f"First contact agent '{first_contact_name}' not found among agents.")


def create_forum_from_config(config_path: Path, base_path: Path) -> Forum:
    factory = ForumFactory(config_path=config_path, base_path=base_path)
    return factory.create_forum()
