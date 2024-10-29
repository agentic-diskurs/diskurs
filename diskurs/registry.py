from typing import Type

AGENT_REGISTRY: dict[str, Type] = {}
LLM_REGISTRY: dict[str, Type] = {}
TOOL_EXECUTOR_REGISTRY: dict[str, Type] = {}
DISPATCHER_REGISTRY: dict[str, Type] = {}
PROMPT_REGISTRY: dict[str, Type] = {}
CONVERSATION_REGISTRY: dict[str, Type] = {}
CONVERSATION_STORE_REGISTRY: dict[str, Type] = {}


def register_agent(name: str):
    """Decorator to register an Agent class."""

    def decorator(cls):
        AGENT_REGISTRY[name] = cls
        return cls

    return decorator


def register_llm(name: str):
    """Decorator to register an LLM class."""

    def decorator(cls):
        LLM_REGISTRY[name] = cls
        return cls

    return decorator


def register_tool_executor(name: str):
    """Decorator to register a ToolExecutor class."""

    def decorator(cls):
        TOOL_EXECUTOR_REGISTRY[name] = cls
        return cls

    return decorator


def register_dispatcher(name: str):
    """Decorator to register a Dispatcher class."""

    def decorator(cls):
        DISPATCHER_REGISTRY[name] = cls
        return cls

    return decorator


def register_prompt(name: str):
    """Decorator to register a Prompt class."""

    def decorator(cls):
        PROMPT_REGISTRY[name] = cls
        return cls

    return decorator


def register_conversation(name: str):
    """Decorator to register a Conversation class."""

    def decorator(cls):
        CONVERSATION_REGISTRY[name] = cls
        return cls

    return decorator


def register_conversation_store(name: str):
    """Decorator to register a ConversationStore class."""

    def decorator(cls):
        CONVERSATION_STORE_REGISTRY[name] = cls
        return cls

    return decorator
