from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, Mock

import pytest
from jinja2 import Template

from tests.conftest import are_classes_structurally_similar
from diskurs import ToolExecutor, PromptValidationError
from diskurs.immutable_conversation import ImmutableConversation
from diskurs.entities import (
    ChatMessage,
    Role,
    PromptArgument,
    OutputField,
    InputField,
    LockedField,
)
from diskurs.prompt import (
    MultistepPrompt,
    validate_dataclass,
    ConductorPrompt,
    HeuristicPrompt,
    validate_json,
    BasePrompt,
)
from diskurs.utils import load_template_from_package
from tests.test_files.heuristic_agent_test_files.prompt import MyHeuristicPromptArgument
from tests.test_files.prompt_test_files.prompt import MyPromptArgument, MyUserPromptWithArrayArgument, Step


@pytest.fixture
def prompt_instance():
    return MultistepPrompt.create(
        location=Path(__file__).parent / "test_files" / "prompt_test_files",
        prompt_argument_class="MyPromptArgument",
    )


@pytest.fixture
def prompt_with_array_instance() -> MultistepPrompt:
    return MultistepPrompt.create(
        location=Path(__file__).parent / "test_files" / "prompt_test_files",
        prompt_argument_class="MyUserPromptWithArrayArgument",
    )


@pytest.fixture
def prompt_testing_conversation(longterm_memories):
    ltm1, ltm2 = longterm_memories
    conversation = ImmutableConversation(
        prompt_argument=MyPromptArgument(
            name="",
            topic="",
            user_question="",
            answer="",
        ),
        chat=[ChatMessage(role=Role.USER, content="Hello, world!", name="Alice")],
        longterm_memory={
            "my_conductor": ltm1(
                field1="longterm_val1",
                field2="longterm_val2",
                field3="longterm_val3",
                user_query="How's the weather?",
            ),
            "my_conductor_2": ltm2(
                user_query="How's the aquarium?",
            ),
        },
        active_agent="my_conductor",
        conversation_id="my_conversation_id",
    )
    return conversation


mock_llm_response = """{
  "name": "John Doe",
  "topic": "Python Programming",
  "user_question": "What is a decorator in Python?",
  "answer": "A decorator in Python is a function that modifies the behavior of another function."
}"""

mock_illegal_llm_response = """{
  "name": "John Doe"
  "topic": "Python Programming",
  "user_question": "What is a decorator in Python?",
  "answer": "A decorator in Python is a function that modifies the behavior of another function."
}"""


def test_parse_prompt(prompt_instance, prompt_testing_conversation):
    res = prompt_instance.parse_user_prompt(
        name="test_agent",
        llm_response=mock_llm_response,
        old_prompt_argument=prompt_testing_conversation.prompt_argument,
    )

    assert res.name == "John Doe"
    assert res.topic == "Python Programming"


def test_fail_parse_prompt(prompt_instance, prompt_testing_conversation):
    res = prompt_instance.parse_user_prompt(
        name="test_agent",
        llm_response=mock_illegal_llm_response,
        old_prompt_argument=prompt_testing_conversation.prompt_argument,
    )

    assert (
        res.content
        == "Invalid JSON: Expecting ',' delimiter at line 3, column 3. Please ensure the response is valid JSON."
    )


@dataclass
class ExamplePromptArg(PromptArgument):
    url: str = ""
    comment: str = ""
    username: str = ""


@dataclass
class ExampleTypedPromptArg(PromptArgument):
    url: Optional[str] = ""
    is_valid: Optional[bool] = None
    comments: Optional[list[str]] = None


def test_validate_dataclass():
    response = {"url": "https://diskurs.dev", "comment": "Do what thou wilt", "username": "Jane"}
    res_prompt_arg = validate_dataclass(parsed_response=response, prompt_argument=ExamplePromptArg)

    assert (
        res_prompt_arg.url == response["url"]
        and res_prompt_arg.comment == response["comment"]
        and res_prompt_arg.username == response["username"]
    )


def test_validate_dataclass_typed():
    response = {
        "url": "https://diskurs.dev",
        "is_valid": "true",
        "comments": ["Do what thou wilt", "Do what thou wilt"],
    }
    res_prompt_arg = validate_dataclass(parsed_response=response, prompt_argument=ExampleTypedPromptArg)

    assert (
        res_prompt_arg.url == response["url"]
        and res_prompt_arg.is_valid == True
        and type(res_prompt_arg.comments) == list
        and type(res_prompt_arg.comments[0]) == str
    )


def test_validate_dataclass_typed_empty():
    response = {
        "url": "https://diskurs.dev",
        "comments": ["Do what thou wilt", "Do what thou wilt"],
    }
    res_prompt_arg = validate_dataclass(parsed_response=response, prompt_argument=ExampleTypedPromptArg)

    assert (
        res_prompt_arg.url == response["url"]
        and type(res_prompt_arg.comments) == list
        and res_prompt_arg.is_valid is None
        and type(res_prompt_arg.comments[0]) == str
    )


def test_validate_dataclass_additional_fields():
    response = {"url": "https://www.diskurs.dev", "foo": "just foo"}

    with pytest.raises(PromptValidationError) as exc_info:
        res_prompt_arg = validate_dataclass(parsed_response=response, prompt_argument=ExamplePromptArg)
    assert (
        str(exc_info.value)
        == "Extra fields provided: foo. Please remove them. Valid fields are: url, comment, username."
    )


@dataclass
class BooleanTestPromptArg(PromptArgument):
    is_enabled: bool = False
    is_valid: bool = True
    is_active: Optional[bool] = None


def test_validate_dataclass_boolean_string_values():
    """Test that string representations of booleans are correctly converted to boolean values."""
    # Test case for string "true"/"false" values (lowercase)
    response = {"is_enabled": "true", "is_valid": "false"}
    result = validate_dataclass(parsed_response=response, prompt_argument=BooleanTestPromptArg)
    assert result.is_enabled is True
    assert result.is_valid is False

    # Test case for string "True"/"False" values (capitalized)
    response = {"is_enabled": "True", "is_valid": "False"}
    result = validate_dataclass(parsed_response=response, prompt_argument=BooleanTestPromptArg)
    assert result.is_enabled is True
    assert result.is_valid is False

    # Test case for string "TRUE"/"FALSE" values (uppercase)
    response = {"is_enabled": "TRUE", "is_valid": "FALSE"}
    result = validate_dataclass(parsed_response=response, prompt_argument=BooleanTestPromptArg)
    assert result.is_enabled is True
    assert result.is_valid is False


def test_validate_dataclass_mixed_boolean_values():
    """Test handling of mixed boolean value types."""
    response = {
        "is_enabled": True,  # Python boolean
        "is_valid": "false",  # String representation
        "is_active": False,  # Python boolean for optional field
    }
    result = validate_dataclass(parsed_response=response, prompt_argument=BooleanTestPromptArg)
    assert result.is_enabled is True
    assert result.is_valid is False
    assert result.is_active is False


def test_exclude_input_fields_in_json_schema():
    """Test that fields with ACCESS_MODE.INPUT are excluded from JSON schema."""

    @dataclass
    class TestPromptArgWithAccessMode(PromptArgument):
        output_field: OutputField[str] = "output"
        input_field: InputField[str] = "input"
        locked_field: LockedField[str] = "locked"

    prompt = BasePrompt(
        agent_description="Test Agent",
        system_template=Template("system"),
        user_template=Template("user"),
        prompt_argument_class=TestPromptArgWithAccessMode,
        json_formatting_template=Template("{{ schema | tojson }}"),
    )

    # Get rendered system template
    system_prompt = prompt.render_system_template(name="test", prompt_argument=TestPromptArgWithAccessMode())

    # Check that only OUTPUT fields are in the JSON schema
    assert "output_field" in system_prompt.content
    assert "input_field" not in system_prompt.content
    assert "locked_field" not in system_prompt.content


def test_conductor_system_template_excludes_hidden_fields():
    """Test that ConductorPrompt's system template correctly excludes INPUT fields."""

    @dataclass
    class TestConductorPromptArg(PromptArgument):
        agent_descriptions: InputField[dict[str, str]] = None
        next_agent: OutputField[str] = None

    prompt = ConductorPrompt(
        agent_description="Test Conductor",
        system_template=Template("Agent descriptions should not be in JSON schema"),
        user_template=Template("user template"),
        prompt_argument_class=TestConductorPromptArg,
        json_formatting_template=Template("Fields: {{ schema | tojson }}"),
        can_finalize=lambda x: True,
        finalize=lambda x: x,
        fail=lambda x: {"error": "fail"},
        longterm_memory=None,
    )

    prompt_arg = TestConductorPromptArg(
        agent_descriptions={"agent1": "Description 1", "agent2": "Description 2"}, next_agent="agent1"
    )

    system_prompt = prompt.render_system_template(name="test_conductor", prompt_argument=prompt_arg)

    assert "next_agent" in system_prompt.content
    assert "agent_descriptions" not in system_prompt.content


prompt_config = {
    "location": Path(__file__).parent / "test_files" / "conductor_test_files",
    "prompt_argument_class": "ConductorPromptArgument",
    "longterm_memory_class": "MyConductorLongtermMemory",
    "can_finalize_name": "can_finalize",
    "fail_name": "fail",
}


def test_conductor_custom_system_prompt():
    prompt = ConductorPrompt.create(**prompt_config)
    prompt_arg = prompt.prompt_argument(
        agent_descriptions={"first_agent": "I am the first agent", "second_agent": "I am the second agent"},
        next_agent="first_agent",
    )

    rendered_system_prompt = prompt.render_system_template(name="test_conductor", prompt_argument=prompt_arg)

    assert rendered_system_prompt.content.startswith("Custom system template")

    json_schema_part = rendered_system_prompt.content.split("```json")[1]

    assert "agent_descriptions" not in json_schema_part
    assert "next_agent" in json_schema_part


prompt_config_no_finalize = {
    "location": Path(__file__).parent / "test_files" / "conductor_no_finalize_test_files",
    "prompt_argument_class": "ConductorPromptArgument",
    "longterm_memory_class": "MyConductorLongtermMemory",
    "can_finalize_name": "can_finalize",
    "fail_name": "fail",
}


def test_conductor_no_finalize_function():
    prompt = ConductorPrompt.create(**prompt_config_no_finalize)

    assert prompt._can_finalize.__name__ == "can_finalize"


def test_parse_user_prompt(prompt_instance, prompt_testing_conversation):
    res = prompt_instance.parse_user_prompt(
        name="test_agent",
        llm_response='"{\\"topic\\": \\"Secure Web Gateway\\", \\"name\\": \\"Hans Ruedi\\", \\"user_question\\": \\"Where is my sandwich?\\"}"',
        old_prompt_argument=prompt_testing_conversation.prompt_argument,
    )

    assert isinstance(res, PromptArgument)
    assert res.topic == "Secure Web Gateway"


def test_parse_user_prompt_json_array(prompt_with_array_instance, prompt_testing_conversation):
    prompt_with_array_instance.is_valid = lambda _: True

    res = prompt_with_array_instance.parse_user_prompt(
        name="test_agent",
        llm_response='{"steps": [{"topic": "Secure Web Gateway"}, {"topic": "Secure Web Gateway 2"}]}',
        old_prompt_argument=MyUserPromptWithArrayArgument(),
    )

    assert are_classes_structurally_similar(type(res), MyUserPromptWithArrayArgument)
    assert isinstance(res.steps, list)
    assert are_classes_structurally_similar(type(res.steps[0]), Step)
    assert res.steps[0].topic == "Secure Web Gateway"
    assert res.steps[1].topic == "Secure Web Gateway 2"


def test_fail():
    prompt = ConductorPrompt.create(**prompt_config)
    msg = prompt.fail(prompt.longterm_memory())
    assert msg["error"] == "Failed to finalize"


heuristic_prompt_config = {
    "location": Path(__file__).parent / "test_files" / "heuristic_agent_test_files",
    "prompt_argument_class": "MyHeuristicPromptArgument",
    "heuristic_sequence_name": "heuristic_sequence",
}


def test_heuristic_prompt_create():
    prompt = HeuristicPrompt.create(**heuristic_prompt_config)

    assert callable(prompt.heuristic_sequence)
    assert are_classes_structurally_similar(prompt.prompt_argument, MyHeuristicPromptArgument)


@pytest.fixture
def tool_executor():
    executor = Mock(spec=ToolExecutor)
    return executor


@pytest.fixture
def heuristic_prompt(conversation):
    prompt = AsyncMock(spec=HeuristicPrompt)  # Change to AsyncMock

    prompt_arg_instance = MyHeuristicPromptArgument()

    prompt.create_prompt_argument.return_value = prompt_arg_instance

    async def heuristic_sequence_side_effect(prompt_argument, metadata, call_tool):
        assert prompt_argument == conversation.prompt_argument, "Expected correct prompt_argument"
        assert metadata == conversation.metadata
        return prompt_arg_instance

    prompt.heuristic_sequence.side_effect = heuristic_sequence_side_effect

    return prompt


@pytest.mark.asyncio
async def test_heuristic_prompt(heuristic_prompt, conversation):
    result = heuristic_prompt.create_prompt_argument()
    assert isinstance(result, MyHeuristicPromptArgument)

    result = await heuristic_prompt.heuristic_sequence(
        prompt_argument=conversation.prompt_argument,
        metadata=conversation.metadata,
        call_tool=lambda x: x,
    )
    assert isinstance(result, MyHeuristicPromptArgument)


def test_create_loads_agent_description():
    location = Path(__file__).parent / "test_files" / "heuristic_agent_test_files"

    with open(location / "agent_description.txt") as f:
        agent_description = f.read()

    prompt = HeuristicPrompt.create(
        location=location,
        prompt_argument_class="MyHeuristicPromptArgument",
    )

    assert prompt.agent_description == agent_description


json_string = """```json\n{\n    "answer": "Die Übersetzung wie folgt: \'Wartungsfenster: Internetleitungsbetriebssupport. Mit freundlichen Grüßen.\' "\n}\n```"""


def test_validate_json():
    res = validate_json(json_string)
    assert (
        res["answer"]
        == """Die Übersetzung wie folgt: 'Wartungsfenster: Internetleitungsbetriebssupport. Mit freundlichen Grüßen.' """
    )


json_string_2 = """{"comment": "To add the domain \\".abc.com\\" to the config file, you will first need to edit the .pac file. Here\\"s a step by step guide:\n\n1. Use the command \\"vault edit company $COMPANY squid/pacs/proxy.pac.auto\\" to edit an automatically generated proxy.pac.auto file. Alternatively, if it\\"s a manually written file, use \\"vault edit company $COMPANY squid/pacs/proxy.pac\\".\n\n2. Inside the file, add the following line of code:\n   \\"dnsDomainIs(host, \\".abc.com\\") || host == \\"abc.com\\"\\"\n\n3. Save the changes and close the file.\n\n4. To implement your changes, you\\"ll need to run buildall. This can be done using the command \\"scheduler rollall active and os linux and company $COMPANY and dbprop \\"squid=1\\"\\".\n\nPlease note: If it\\"s a newly set up PAC file, use this command to begin: \\"[VAULT] company/open $ cp /vault/current/sample/squid/pacs/proxy.pac.auto squid/pacs/\\". Remember to replace \\"$COMPANY\\" with your own company details where needed.\n\nIf you face any issues during these steps, do not hesitate to contact us for further assistance. "}"""


def test_validate_json_2():
    res = validate_json(json_string_2)
    assert isinstance(res, dict)
    assert "comment" in res
    assert res["comment"].startswith("To add the domain")
    assert '".abc.com"' in res["comment"]
    assert "scheduler rollall" in res["comment"]
    assert "buildall" in res["comment"]


def test_render_json_formatting_prompt(prompt_instance, prompt_testing_conversation):
    prompt_args = prompt_instance.create_prompt_argument(name="test", topic="example")
    result = prompt_instance.render_json_formatting_prompt(prompt_args)
    assert "name" in result
    assert "topic" in result


def test_render_json_formatting_prompt_with_access_modes():
    @dataclass
    class TestPromptArg(PromptArgument):
        output_field: OutputField[str] = "output"
        input_field: InputField[str] = "input"
        locked_field: LockedField[str] = "locked"

    # Create a minimal template for testing
    template = Template(
        "Fields to include: {% for key in schema.keys() %}{{ key }}{% if not loop.last %}, {% endif %}{% endfor %}"
    )

    prompt = BasePrompt(
        agent_description="Test Agent",
        system_template=Template("system"),
        user_template=Template("user"),
        prompt_argument_class=TestPromptArg,
        json_formatting_template=template,
    )

    # Create a proper instance of TestPromptArg instead of using a dictionary
    prompt_args = TestPromptArg(
        output_field="should appear", input_field="should not appear", locked_field="should not appear"
    )

    result = prompt.render_json_formatting_prompt(prompt_args)

    assert "output_field" in result
    assert "input_field" not in result
    assert "locked_field" not in result


def test_render_json_formatting_prompt_empty_args():
    @dataclass
    class EmptyPromptArg(PromptArgument):
        pass

    template = Template(
        "Fields: {% for key in schema.keys() %}{{ key }}{% if not loop.last %}, {% endif %}{% endfor %}"
    )

    prompt = BasePrompt(
        agent_description="Test Agent",
        system_template=Template("system"),
        user_template=Template("user"),
        prompt_argument_class=EmptyPromptArg,
        json_formatting_template=template,
    )

    # Create a proper instance of EmptyPromptArg instead of using an empty dictionary
    prompt_args = EmptyPromptArg()
    result = prompt.render_json_formatting_prompt(prompt_args)
    assert result == "Fields: "


def test_render_json_formatting_prompt_missing_template():
    prompt = BasePrompt(
        agent_description="Test Agent",
        system_template=Template("system"),
        user_template=Template("user"),
        prompt_argument_class=PromptArgument,
        json_formatting_template=None,
    )

    with pytest.raises(ValueError) as exc_info:
        prompt.render_json_formatting_prompt({"test": "value"})
    assert str(exc_info.value) == "json_formatting_template is not set."


def test_render_json_formatting_prompt_inheritance():
    @dataclass
    class BasePromptArg(PromptArgument):
        base_output: OutputField[str] = "base"
        base_input: InputField[str] = "input"

    @dataclass
    class ChildPromptArg(BasePromptArg):
        child_output: OutputField[str] = "child"
        child_locked: LockedField[str] = "locked"

    template = Template(
        "Fields: {% for key in schema.keys() %}{{ key }}{% if not loop.last %}, {% endif %}{% endfor %}"
    )

    prompt = BasePrompt(
        agent_description="Test Agent",
        system_template=Template("system"),
        user_template=Template("user"),
        prompt_argument_class=ChildPromptArg,
        json_formatting_template=template,
    )

    # Create a proper instance of ChildPromptArg instead of using a dictionary
    prompt_args = ChildPromptArg(base_output="base", base_input="input", child_output="child", child_locked="locked")

    result = prompt.render_json_formatting_prompt(prompt_args)

    # Verify output fields from both parent and child classes are included
    assert "base_output" in result
    assert "child_output" in result
    # Verify input and locked fields are excluded
    assert "base_input" not in result
    assert "child_locked" not in result


def test_generate_json_schema_with_nested_dataclass():
    @dataclass
    class NestedClass:
        field1: str = ""
        field2: int = 0

    @dataclass
    class TestPromptArg(PromptArgument):
        name: str = ""
        nested: NestedClass = field(default_factory=NestedClass)

    prompt = BasePrompt(
        agent_description="Test Agent",
        system_template=Template("system"),
        user_template=Template("user"),
        prompt_argument_class=TestPromptArg,
        json_formatting_template=Template("{{ schema | tojson(indent=2) }}"),
    )

    schema = prompt._generate_json_schema({"name": str, "nested": NestedClass})

    assert "name" in schema
    assert "nested" in schema
    assert isinstance(schema["nested"], dict)
    assert "field1" in schema["nested"]
    assert "field2" in schema["nested"]
    assert schema["nested"]["field1"] == ""
    assert schema["nested"]["field2"] == 0


def test_generate_json_schema_with_list_of_dataclasses():
    @dataclass
    class ListItem:
        topic: str = ""
        priority: int = 0

    @dataclass
    class TestPromptArg(PromptArgument):
        name: str = ""
        items: list[ListItem] = field(default_factory=list)

    prompt = BasePrompt(
        agent_description="Test Agent",
        system_template=Template("system"),
        user_template=Template("user"),
        prompt_argument_class=TestPromptArg,
        json_formatting_template=Template("{{ schema | tojson(indent=2) }}"),
    )

    schema = prompt._generate_json_schema({"name": str, "items": list[ListItem]})

    assert "name" in schema
    assert "items" in schema
    assert isinstance(schema["items"], list)
    assert len(schema["items"]) > 0
    assert "topic" in schema["items"][0]
    assert "priority" in schema["items"][0]
    assert schema["items"][0]["topic"] == ""
    assert schema["items"][0]["priority"] == 0


def test_render_json_formatting_prompt_with_array_type():
    @dataclass
    class PlanStep:
        step_id: str = ""
        description: str = ""
        function: str = ""
        parameters: dict = field(default_factory=dict)
        depends_on: list[str] = field(default_factory=list)

    @dataclass
    class TestArrayArgument(PromptArgument):
        user_query: str = ""
        execution_plan: list[PlanStep] = field(default_factory=list)
        summary: str = ""

    template = load_template_from_package("diskurs.assets", "json_formatting.jinja2")

    prompt = BasePrompt(
        agent_description="Test Agent",
        system_template=Template("system"),
        user_template=Template("user"),
        prompt_argument_class=TestArrayArgument,
        json_formatting_template=template,
    )

    # Create a proper instance of TestArrayArgument instead of using a dictionary
    prompt_args = TestArrayArgument(user_query="", execution_plan=[], summary="")

    result = prompt.render_json_formatting_prompt(prompt_args)

    assert '"user_query"' in result
    assert '"execution_plan"' in result
    assert '"step_id"' in result
    assert '"description"' in result
    assert '"function"' in result
    assert '"parameters"' in result
    assert '"depends_on"' in result
    assert '"summary"' in result


def test_json_formatting_with_real_llm_compiler_prompt(prompt_with_array_instance):
    from diskurs.llm_compiler.prompts import PlanningPromptArgument

    template = load_template_from_package("diskurs.assets", "json_formatting.jinja2")

    prompt = BasePrompt(
        agent_description="LLM Compiler",
        system_template=Template("system"),
        user_template=Template("user"),
        prompt_argument_class=PlanningPromptArgument,
        json_formatting_template=template,
    )

    # Create a proper instance of PlanningPromptArgument with the correct fields
    prompt_args = PlanningPromptArgument(
        user_query="", execution_plan=[], answer="", tools=[], replan=False, replan_explanation=""
    )

    result = prompt.render_json_formatting_prompt(prompt_args)

    assert "JSON" in result
    assert '"user_query"' in result
    assert '"execution_plan"' in result
    assert '"answer"' in result
    assert '"tools"' in result
    # Check specific fields in the execution plan structure
    assert '"step_id"' in result
    assert '"description"' in result
    assert '"function"' in result
    assert '"parameters"' in result
    assert '"depends_on"' in result

    assert "[" in result and "]" in result


def test_access_mode_annotations():
    """
    Test that all AccessMode types are properly respected in the JSON schema generation.
    """

    @dataclass
    class TestPromptArgWithAccessModes(PromptArgument):
        # Default field (should be included by default)
        default_field: str = "default"

        # Field with OUTPUT mode
        output_field: OutputField[str] = "output"

        # Field with INPUT mode
        input_field: InputField[str] = "input"

        # Field with LOCKED mode
        locked_field: LockedField[str] = "locked"

        # Field with multiple annotations, including mode=INPUT (should be excluded)
        multi_annotated_input: InputField[str] = "multi-input"

        # Field with multiple annotations, including mode=OUTPUT (should be included)
        multi_annotated_output: OutputField[str] = "multi-output"

    prompt = BasePrompt(
        agent_description="Test Agent",
        system_template=Template("Testing access mode annotations"),
        user_template=Template("user template"),
        prompt_argument_class=TestPromptArgWithAccessModes,
        json_formatting_template=Template("{{ schema | tojson }}"),
    )

    system_prompt = prompt.render_system_template(name="test", prompt_argument=TestPromptArgWithAccessModes())
    content = system_prompt.content

    assert '"default_field"' in content
    assert '"output_field"' in content
    assert '"multi_annotated_output"' in content

    assert '"input_field"' not in content
    assert '"locked_field"' not in content
    assert '"multi_annotated_input"' not in content

    instance = TestPromptArgWithAccessModes(
        default_field="custom default",
        output_field="custom output",
        input_field="custom input",
        locked_field="custom locked",
        multi_annotated_input="custom multi-input",
        multi_annotated_output="custom multi-output",
    )

    system_prompt = prompt.render_system_template(name="test", prompt_argument=instance)
    content = system_prompt.content

    assert '"default_field"' in content
    assert '"output_field"' in content
    assert '"multi_annotated_output"' in content
    assert '"input_field"' not in content
    assert '"locked_field"' not in content
    assert '"multi_annotated_input"' not in content


def test_validate_dataclass_output_field_next_agent():
    """
    Test the specific case for the next_agent field with OutputField[str] annotation
    that was causing type conversion issues.
    """
    from diskurs.prompt import DefaultConductorPromptArgument, validate_dataclass

    # JSON response that would come from LLM
    response = {"next_agent": "Maintenance_Conductor_Agent"}  # String value as returned by LLM

    # Validate the dataclass with the string value for next_agent
    result = validate_dataclass(parsed_response=response, prompt_argument=DefaultConductorPromptArgument)

    # Assert the result has the correct next_agent value
    assert result.next_agent == "Maintenance_Conductor_Agent"

    # Also test a more complex case with agent_descriptions and next_agent
    complex_response = {
        "next_agent": "Maintenance_Conductor_Agent",
        # agent_descriptions is a LockedField, so it shouldn't be validated
        # but we include it here to test that it doesn't interfere
        "agent_descriptions": {"agent1": "Description 1"},
    }

    complex_result = validate_dataclass(
        parsed_response=complex_response, prompt_argument=DefaultConductorPromptArgument
    )
    assert complex_result.next_agent == "Maintenance_Conductor_Agent"
    assert complex_result.agent_descriptions is None  # Default value since it's a LockedField
