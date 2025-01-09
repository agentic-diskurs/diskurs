import logging
from dataclasses import asdict
from pathlib import Path
from typing import List, Callable, Type, Any

from asyncer import asyncify

from diskurs.config import load_config_from_yaml
from diskurs.entities import (
    ToolDescription,
    DiskursInput,
    ChatMessage,
    Role,
    MessageType,
)
from diskurs.logger_setup import get_logger
from diskurs.protocols import (
    Agent,
    ConversationParticipant,
    ConversationStore,
    Conversation,
    ConductorAgent,
)
from diskurs.registry import (
    AGENT_REGISTRY,
    LLM_REGISTRY,
    TOOL_EXECUTOR_REGISTRY,
    DISPATCHER_REGISTRY,
    PROMPT_REGISTRY,
    CONVERSATION_REGISTRY,
    CONVERSATION_STORE_REGISTRY,
)
from diskurs.tools import load_tools, load_dependencies
from diskurs.utils import load_module_from_path

logging.basicConfig(level=logging.WARNING)


def filter_conductor_agents(agents: list[Agent]) -> list[ConductorAgent]:
    return [agent for agent in agents if isinstance(agent, ConductorAgent)]


def init_longterm_memories(agents: list[Agent]) -> dict[str, Any]:
    longterm_memory = {}

    for conductor in filter_conductor_agents(agents):
        longterm_memory[conductor.name] = conductor.prompt.init_longterm_memory()

    return longterm_memory


class Forum:
    def __init__(
        self,
        agents: List[Agent],
        dispatcher,
        tool_executor,
        first_contact: ConversationParticipant,
        conversation_store: ConversationStore,
        conversation_class: Type[Conversation],
    ):
        self.agents = agents
        self.dispatcher = dispatcher
        self.tool_executor = tool_executor
        self.conversation_store = conversation_store
        self.first_contact = first_contact
        self.conversation_class = conversation_class
        self.logger = get_logger(f"diskurs.{__name__}.forum")

        self.logger.info("Initializing forum")

    async def fetch_or_create_conversation(self, diskurs_input: DiskursInput) -> Conversation:
        """
        If a conversation store is present, we try to getch an existing conversation from the conversation store
        or creates a new one if it doesn't exist.
        """
        if self.conversation_store and await self.conversation_store.exists(diskurs_input.conversation_id):
            return await self.conversation_store.fetch(diskurs_input.conversation_id)
        else:
            longterm_memory = init_longterm_memories(self.agents)
            conversation = self.conversation_class(
                metadata=diskurs_input.metadata,
                conversation_id=diskurs_input.conversation_id,
                longterm_memory=longterm_memory,
            )
        if diskurs_input.user_query:
            conversation = conversation.append(
                ChatMessage(
                    Role.USER,
                    content=diskurs_input.user_query,
                    name="forum",
                    type=MessageType.CONVERSATION,
                )
            )
        return conversation

    async def ama(self, diskurs_input: DiskursInput):
        if not diskurs_input.conversation_id:
            self.logger.warning("Conversation ID not provided. Using default value 'default'.")
            diskurs_input.conversation_id = "default"

        conversation = await self.fetch_or_create_conversation(diskurs_input)

        answer = await self.dispatcher.run(participant=self.first_contact, conversation=conversation)
        return answer


class ForumFactory:
    def __init__(
        self,
        config_path: Path,
        base_path: Path,
        conversation_store: ConversationStore = None,
    ):
        self.tool_dependencies = None
        self.base_path = base_path
        self.config = load_config_from_yaml(config=config_path, base_path=base_path)
        self.llm_clients = {}
        self.agents = []
        self.tools: list[Callable] = []
        self.dispatcher = None
        self.tool_executor = None
        self.first_contact = None
        self.modules_to_import = [
            Path(__file__).parent / mdls
            for mdls in [
                "llm_client.py",
                "azure_llm_client.py",
                "dispatcher.py",
                "agent.py",
                "conductor_agent.py",
                "prompt.py",
                "immutable_conversation.py",
                "filesystem_conversation_store.py",
                "heuristic_agent.py",
            ]
        ]
        self.conversation_store = conversation_store
        self.conversation_cls = None
        self.logger = get_logger(f"diskurs.{__name__}.forum_factory")

        self.logger.info("Initializing forum factory")

    def create_forum(self) -> Forum:
        self.import_modules()
        self.load_custom_modules()
        self.create_tool_executor()
        self.load_and_register_tool_dependencies()
        self.load_and_register_tools()
        self.create_dispatcher()
        self.create_llm_clients()
        self.create_agents()
        self.prepare_conductors()
        self.identify_first_contact_agent()
        self.load_conversation()
        self.load_conversation_store()

        self.logger.info("*** Forum created successfully ***")

        return Forum(
            agents=self.agents,
            dispatcher=self.dispatcher,
            tool_executor=self.tool_executor,
            first_contact=self.first_contact,
            conversation_store=self.conversation_store,
            conversation_class=self.conversation_cls,
        )

    def import_modules(self):
        """Dynamically import modules required for registration."""
        for module_path in self.modules_to_import:
            load_module_from_path(module_path)

    def load_custom_modules(self):
        """Load custom modules specified in the configuration."""
        for custom_module in self.config.custom_modules:
            module_path = (self.base_path / custom_module["location"]).resolve()
            load_module_from_path(module_path)

    def create_tool_executor(self):
        """Create a tool executor instance based on the configuration."""
        tool_executor_cls = TOOL_EXECUTOR_REGISTRY.get(self.config.tool_executor_type)
        if tool_executor_cls is None:
            raise ValueError(f"ToolExecutor type '{self.config.tool_executor_type}' is not registered.")
        self.tool_executor = tool_executor_cls()

    def load_conversation(self):
        """Load conversation class from the configuration."""
        self.conversation_cls = CONVERSATION_REGISTRY.get(self.config.conversation_type)
        if self.conversation_cls is None:
            raise ValueError(f"Conversation class '{self.config.conversation_type}' is not registered.")

    def load_and_register_tool_dependencies(self):
        """Load and register tools with the tool executor."""
        if self.config.tool_dependencies:
            self.tool_dependencies = load_dependencies(
                self.config.tool_dependencies, self.config.custom_modules, self.base_path
            )
            self.tool_executor.register_dependencies(self.tool_dependencies)

    def load_and_register_tools(self):
        """Load and register tools with the tool executor."""
        if self.config.tools:
            self.tools = load_tools(
                self.tool_dependencies, self.config.tools, self.config.custom_modules, self.base_path
            )
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

            if hasattr(agent_conf, "max_trials"):
                additional_args["max_trials"] = agent_conf.max_trials
            if hasattr(agent_conf, "max_reasoning_steps"):
                additional_args["max_reasoning_steps"] = agent_conf.max_reasoning_steps
            if hasattr(agent_conf, "final_properties"):
                additional_args["final_properties"] = agent_conf.final_properties
            if hasattr(agent_conf, "prompt"):
                agent_conf.prompt.location = agent_conf.prompt.location.resolve()
                prompt_creation_arguments = asdict(agent_conf.prompt)

                if agent_conf.prompt.type == "conductor_prompt":
                    prompt_creation_arguments["topics"] = agent_conf.topics

                prompt_cls = PROMPT_REGISTRY.get(agent_conf.prompt.type)
                prompt = prompt_cls.create(**prompt_creation_arguments)
                additional_args["prompt"] = prompt
            if hasattr(agent_conf, "llm"):
                additional_args["llm_client"] = self.llm_clients[agent_conf.llm]
            if hasattr(agent_conf, "tools") and agent_conf.tools:
                additional_args["tools"] = [
                    ToolDescription.from_function(tool) for tool in self.tools if tool.__name__ in agent_conf.tools
                ]
                additional_args["tool_executor"] = self.tool_executor
            if hasattr(agent_conf, "topics") and self.dispatcher:
                additional_args["topics"] = agent_conf.topics
            if hasattr(agent_conf, "finalizer_name"):
                additional_args["finalizer_name"] = agent_conf.finalizer_name
            if hasattr(agent_conf, "can_finalize_name"):
                additional_args["can_finalize_name"] = agent_conf.can_finalize_name

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

        self.logger.info(f"Identifying first contact agent {first_contact_name}")

        self.first_contact = next((agent for agent in self.agents if agent.name == first_contact_name), None)
        if self.first_contact is None:
            raise ValueError(f"First contact agent '{first_contact_name}' not found among agents.")

    def load_conversation_store(self):
        """Load conversation store class from the configuration."""
        conversation_store_cls = CONVERSATION_STORE_REGISTRY.get(self.config.conversation_store.type)
        conversation_store_config = asdict(self.config.conversation_store)
        conversation_store_config.pop("type")

        self.conversation_store = conversation_store_cls.create(
            **{"agents": self.agents, "conversation_class": self.conversation_cls},
            **conversation_store_config,
        )
        if self.conversation_store is None:
            raise ValueError(f"Conversation store '{self.config.conversation_store}' is not registered.")


def create_forum_from_config(config_path: Path, base_path: Path) -> Forum:
    factory = ForumFactory(config_path=config_path, base_path=base_path)
    return factory.create_forum()
