import logging
from pathlib import Path
from pprint import pprint

import pytest

from diskurs.config import ToolConfig, ToolDependencyConfig
from diskurs.entities import ToolDescription
from diskurs.tools import tool, ToolExecutor, create_func_with_closure, load_tools, load_dependencies


@pytest.fixture()
def tool_configs():
    return [
        ToolConfig(
            name="sample_function",
            module_path=Path("test_files") / "tool_test_files" / "dummy_module.py",
            function_name="sample_function",
            dependencies=["dummy_dependency_name"],
            configs={"param": "value"},
        ),
        ToolConfig(
            name="simple_function",
            module_path=Path("test_files") / "tool_test_files" / "dummy_module.py",
            function_name="simple_function",
            dependencies=None,
            configs=None,
        ),
    ]


@pytest.fixture()
def dependency_config():
    my_dep_conf = ToolDependencyConfig(
        name="dummy_dependency_name",
        module_path=Path(__file__).parent / "test_files" / "tool_test_files" / "dummy_module.py",
        class_name="ExampleDependency",
        parameters={
            "foo": "example_value1",
            "bar": "example_value2",
        },
    )
    return my_dep_conf


def test_load_dependencies(dependency_config):
    dependencies = load_dependencies([dependency_config])
    assert len(dependencies) == 1
    assert any(dep.name == "dummy_dependency_name" for dep in dependencies)
    assert isinstance(dependencies[0], object)


def test_create_func_with_closure(dependency_config):

    def create_dummy_func(configs, dummy_dependency_name=None):

        def dummy_func():
            return {
                "config_param": configs["param"],
                "dep_foo": dummy_dependency_name.foo,
                "dep_bar": dummy_dependency_name.bar,
            }

        return dummy_func

    config = ToolConfig(
        name="dummy_func",
        module_path=Path("test_files") / "tool_test_files" / "dummy_module.py",
        function_name="dummy_func",
        dependencies=["dummy_dependency_name"],
        configs={"param": "value"},
    )

    dependencies = load_dependencies([dependency_config])

    result_func = create_func_with_closure(create_dummy_func, config, dependencies)
    result = result_func()

    assert result["config_param"] == "value"
    assert result["dep_foo"] == dependency_config.parameters["foo"]
    assert result["dep_bar"] == dependency_config.parameters["bar"]


def test_load_tools(tool_configs, dependency_config):
    dependencies = load_dependencies([dependency_config])

    loaded_functions = load_tools(tool_configs, dependencies)

    assert len(loaded_functions) == len(tool_configs)
    for func in loaded_functions:
        assert callable(func)


def test_annotation():
    @tool
    def get_delivery_date(order_id: str) -> str:
        """
        Get the delivery date for a customer's order.

        :param order_id: The customer's order ID.
        :return: The delivery date as a string.
        """
        return "2024-09-03"

    @tool
    def get_holiday_date(season: str, job: str) -> str:
        """
        Get the delivery date for a customer's order.

        :param job: the job you're working in
        :param season: season of the year
        :return: Your well deserved holiday
        """
        return "2024-09-03"

    @tool
    def is_holiday_deserved(name: str) -> bool:
        """
        Whether you deserve your holiday or not.
        :param name: your name
        :return: Your well deserved holiday
        """
        return True

    # Access the tool descriptor
    tool_description = [
        ToolDescription.from_function(fun) for fun in [get_delivery_date, is_holiday_deserved, get_holiday_date]
    ]
    pprint(tool_description)


# Sample tools for testing
def tool_a():
    pass


def tool_b():
    pass


def tool_c():
    pass


@pytest.fixture
def tool_executor():
    return ToolExecutor()


def test_register_single_tool(tool_executor):
    # Register a single tool
    tool_executor.register_tools(tool_a)

    # Check that the tool is registered correctly
    assert "tool_a" in tool_executor.tools
    assert tool_executor.tools["tool_a"] == tool_a


def test_register_multiple_tools(tool_executor):
    # Register multiple tools as a dictionary
    tool_list = [tool_b, tool_c]
    tool_executor.register_tools(tool_list)

    # Check that both tools are registered correctly
    assert "tool_b" in tool_executor.tools
    assert "tool_c" in tool_executor.tools
    assert tool_executor.tools["tool_b"] == tool_b
    assert tool_executor.tools["tool_c"] == tool_c


def test_register_tool_overwrite_warning(tool_executor, caplog):
    # Register a tool
    tool_executor.register_tools(tool_a)

    # Re-register the same tool to trigger overwrite warning
    with caplog.at_level(logging.WARNING):
        tool_executor.register_tools(tool_a)

    # Check that the warning is logged
    assert "Tool 'tool_a' already exists and will be overwritten." in caplog.text


def test_register_multiple_tools_with_overwrite(tool_executor, caplog):
    # Register multiple tools
    tool_list = [tool_b, tool_c]
    tool_executor.register_tools(tool_list)

    # Re-register a tool to trigger the overwrite warning
    with caplog.at_level(logging.WARNING):
        tool_executor.register_tools(tool_c)

    # Check that the warning is logged
    assert "Tool 'tool_c' already exists and will be overwritten." in caplog.text


def test_invisible_params():
    @tool
    def get_delivery_date(order_id: str, company_id: str, season: str) -> str:
        """
        Get the delivery date for a customer's order.

        :param order_id: The customer's order ID.
        :param company_id: [invisible] the job you're working in
        :param season: season of the year
        :return: The delivery date as a string.
        """
        return "2024-09-03"

    # Access the tool descriptor
    tool_description = ToolDescription.from_function(get_delivery_date)
    assert get_delivery_date.invisible_args == {
        "company_id": {"title": "company_id", "type": "str", "description": "[invisible] the job you're working in"}
    }
    assert "season" in tool_description.arguments.keys()
    assert "company_id" not in tool_description.arguments.keys()


def test_create_func_with_closure_missing_dependency():
    def dummy_func(configs, dep1=None):
        def inner_func():
            return {"config_param": configs["param"], "dep_foo": dep1.foo, "dep_bar": dep1.bar}

        return inner_func

    config = ToolConfig(
        name="dummy_func",
        module_path=Path("test_files") / "tool_test_files" / "dummy_module.py",
        function_name="dummy_func",
        dependencies=["dep1"],
        configs={"param": "value"},
    )

    try:
        create_func_with_closure(dummy_func, config, [])
    except ValueError as e:
        assert str(e) == "Missing configurations for dependencies: dep1"


def test_create_func_with_closure_no_dependencies():
    def dummy_func(configs):
        def inner_func():
            return {"config_param": configs["param"]}

        return inner_func

    config = ToolConfig(
        name="dummy_func",
        module_path=Path("test_files") / "tool_test_files" / "dummy_module.py",
        function_name="dummy_func",
        dependencies=None,
        configs={"param": "value"},
    )

    result_func = create_func_with_closure(dummy_func, config, [])
    result = result_func()

    assert result["config_param"] == "value"
