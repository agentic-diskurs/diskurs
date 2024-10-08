from dataclasses import dataclass

from diskurs import ToolDependency


@dataclass(kw_only=True)
class SomeExternalDependency(ToolDependency):
    user_name: str
    street: str