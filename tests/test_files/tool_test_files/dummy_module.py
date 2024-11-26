# dummy_module.py
from typing import Self

from diskurs.protocols import ToolDependency


def simple_function(configs, dep1=None, dep2=None):
    return f"Configs: {configs}, Dependencies: {dep1}, {dep2}"


def create_sample_function(configs, dummy_dependency_name=None, dep2=None):
    def inner_function():
        return f"Inner Configs: {configs}, Inner Dependencies: {dummy_dependency_name}, {dep2}"

    return inner_function


class ExampleDependency(ToolDependency):
    def __init__(self, name: str, foo: str, bar: str):
        self.name = name
        self.foo = foo
        self.bar = bar

    @classmethod
    def create(cls, name: str, foo: str, bar: str) -> Self:
        return cls(name, foo, bar)

    def close(self) -> None:
        pass
