from dataclasses import dataclass

from diskurs import ToolDependencyConfig


@dataclass(kw_only=True)
class SomeExternalDependencyConfig(ToolDependencyConfig):
    user_name: str
    street: str
