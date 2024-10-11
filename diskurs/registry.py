from typing import Type, Dict

AGENT_REGISTRY: Dict[str, Type] = {}
LLM_REGISTRY: Dict[str, Type] = {}
TOOL_EXECUTOR_REGISTRY: Dict[str, Type] = {}
DISPATCHER_REGISTRY: Dict[str, Type] = {}
PROMPT_REGISTRY: Dict[str, Type] = {}


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
