import pytest
from dataclasses import asdict

from diskurs.llm_compiler.prompts import PlanningSystemPromptArgument
from diskurs.llm_compiler.entities import PlanStep
from diskurs.utils import load_template_from_package


class TestLLMCompilerTemplates:
    @pytest.fixture
    def planning_system_template(self):
        """Load the planning system template from the package."""
        return load_template_from_package("diskurs.assets", "llm_compiler_planning_system_template.jinja2")

    @pytest.fixture
    def planning_tools(self):
        """Sample tools for testing."""
        return [
            {"name": "search_web", "description": "Search the web for information", "arguments": {"query": "str"}},
            {"name": "calculate", "description": "Perform a calculation", "arguments": {"expression": "str"}},
        ]

    @pytest.fixture
    def execution_plan(self):
        return [
            PlanStep(
                step_id="1",
                description="Search for information about AI",
                function="search_web",
                parameters={"query": "artificial intelligence"},
                result="Information about AI...",
                status="completed",
            ),
            PlanStep(
                step_id="2",
                description="Calculate statistics",
                function="calculate",
                parameters={"expression": "2+2"},
                result="4",
                status="completed",
            ),
        ]

    def test_planning_mode_with_empty_execution_plan(self, planning_system_template, planning_tools):
        """Test rendering in planning mode with empty execution_plan."""
        # Create prompt args for planning mode
        prompt_args = PlanningSystemPromptArgument(
            tools=planning_tools, execution_plan=[], user_query="Whats the meaning of life?"
        )

        rendered = planning_system_template.render(**asdict(prompt_args))

        # Check for planning content
        assert "User query: Whats the meaning of life?" in rendered
        assert "Each plan should comprise an action from the following 2 types" in rendered

        # Check that all tools are included - match the actual format in the template
        assert "- Name: search_web" in rendered
        assert "Description: Search the web for information" in rendered

    def test_planning_mode_with_undefined_execution_plan(self, planning_system_template, planning_tools):
        """Test rendering in planning mode when execution_plan is None."""
        prompt_args = PlanningSystemPromptArgument(tools=planning_tools, execution_plan=None)

        rendered = planning_system_template.render(**asdict(prompt_args))

        # Check for planning content - match what's actually in the template
        assert "Given the above user query, create a plan to solve it" in rendered
        assert "Each plan should comprise an action from the following" in rendered

        # Check that all tools are included
        assert "- Name: search_web" in rendered
        assert "Description: Search the web for information" in rendered

    def test_summary_mode_with_executed_steps(self, planning_system_template, execution_plan):
        """Test rendering in summary mode with executed_steps parameter."""
        # Include user_query in the prompt_args to avoid duplicate parameter
        prompt_args = PlanningSystemPromptArgument(
            tools=[], execution_plan=execution_plan, user_query="Tell me about AI and calculate 2+2"
        )

        # Render the template without additional user_query parameter
        rendered = planning_system_template.render(**asdict(prompt_args))

        # Should contain summary content
        assert "You are an AI assistant tasked with synthesizing" in rendered
        assert "Original user query: Tell me about AI and calculate 2+2" in rendered

        # Should contain all execution steps
        assert "Description: Search for information about AI" in rendered
        assert "Description: Calculate statistics" in rendered

    def test_template_handles_edge_cases(self, planning_system_template):
        """Test that the template handles edge cases like missing variables gracefully."""
        # Test rendering with minimal required arguments
        rendered = planning_system_template.render(tools=[], execution_plan=[])

        # Should render without errors and include planning content - match what's in the template
        assert "Given the above user query, create a plan to solve it" in rendered

        # Test with empty dictionary
        rendered = planning_system_template.render({})

        # Template should handle missing variables without raising exceptions
        assert len(rendered) > 0  # At least renders something
