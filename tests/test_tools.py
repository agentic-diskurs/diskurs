from pprint import pprint

from diskurs.tools import tool, ToolExecutor
from entities import ToolDescription


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


import pytest
import logging
from unittest.mock import MagicMock


# Assuming the ToolExecutor class and logger have been imported
# from your_module import ToolExecutor, logger


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


import pytest


# Sample tools for testing
def tool_with_args(arg1, arg2):
    return arg1 + arg2


def tool_without_args():
    return "success"


@pytest.fixture
def tool_executor_with_tools():
    # Initialize ToolExecutor with some tools
    executor = ToolExecutor()
    executor.register_tools(tool_with_args)
    executor.register_tools(tool_without_args)
    return executor


def test_execute_existing_tool_without_args(tool_executor_with_tools):
    # Execute a tool without arguments
    result = tool_executor_with_tools.execute_tool(None)

    # Assert that the tool executed correctly
    assert result == "success"


def test_execute_existing_tool_with_args(tool_executor_with_tools):
    # Execute a tool with arguments
    result = tool_executor_with_tools.execute_tool(None)

    # Assert that the tool executed correctly with the provided arguments
    assert result == 15


def test_execute_non_existent_tool(tool_executor_with_tools):
    # Try to execute a tool that doesn't exist
    with pytest.raises(ValueError, match="Tool 'non_existent_tool' not found."):
        tool_executor_with_tools.execute_tool(None)


def test_execute_tool_missing_arguments(tool_executor_with_tools):
    # Try to execute a tool with missing arguments
    with pytest.raises(TypeError):
        tool_executor_with_tools.execute_tool(None)
