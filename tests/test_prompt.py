from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Annotated
from unittest.mock import AsyncMock, Mock

import pytest
from jinja2 import Template

from conftest import are_classes_structurally_similar
from diskurs import ImmutableConversation, ToolExecutor, PromptValidationError
from diskurs.entities import ChatMessage, Role, PromptArgument, prompt_field
from diskurs.prompt import (
    MultistepPrompt,
    validate_dataclass,
    ConductorPrompt,
    HeuristicPrompt,
    validate_json,
    BasePrompt,
)
from diskurs.utils import load_template_from_package
from test_files.heuristic_agent_test_files.prompt import MyHeuristicPromptArgument
from test_files.prompt_test_files.prompt import MyUserPromptArgument, MyUserPromptWithArrayArgument, Step


@pytest.fixture
def prompt_instance():
    return MultistepPrompt.create(
        location=Path(__file__).parent / "test_files" / "prompt_test_files",
        system_prompt_argument_class="MySystemPromptArgument",
        prompt_argument_class="MyUserPromptArgument",
    )


@pytest.fixture
def prompt_with_array_instance() -> MultistepPrompt:
    return MultistepPrompt.create(
        location=Path(__file__).parent / "test_files" / "prompt_test_files",
        system_prompt_argument_class="MySystemPromptArgument",
        prompt_argument_class="MyUserPromptWithArrayArgument",
    )


@pytest.fixture
def prompt_testing_conversation(longterm_memories):
    ltm1, ltm2 = longterm_memories
    conversation = ImmutableConversation(
        prompt_argument=MyUserPromptArgument(
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


prompt_config = {
    "location": Path(__file__).parent / "test_files" / "conductor_test_files",
    "prompt_argument_class": "ConductorUserPromptArgument",
    "system_prompt_argument_class": "ConductorSystemPromptArgument",
    "longterm_memory_class": "MyConductorLongtermMemory",
    "can_finalize_name": "can_finalize",
    "fail_name": "fail",
}


def test_conductor_custom_system_prompt():
    prompt = ConductorPrompt.create(**prompt_config)
    rendered_system_prompt = prompt.render_system_template(
        name="test_conductor",
        prompt_args=prompt.system_prompt_argument(
            agent_descriptions={"first_agent": "I am the first agent", "second_agen": "I am the second agent"}
        ),
    )
    print(rendered_system_prompt)
    assert rendered_system_prompt.content.startswith("Custom system template")


prompt_config_no_finalize = {
    "location": Path(__file__).parent / "test_files" / "conductor_no_finalize_test_files",
    "prompt_argument_class": "ConductorUserPromptArgument",
    "system_prompt_argument_class": "ConductorSystemPromptArgument",
    "longterm_memory_class": "MyConductorLongtermMemory",
    "can_finalize_name": "can_finalize",
    "fail_name": "fail",
}


def test_conductor_no_finalize_function():
    prompt = ConductorPrompt.create(**prompt_config_no_finalize)

    assert prompt._can_finalize.__name__ == "can_finalize"


def test_parse_user_prompt_partial_update(prompt_instance, prompt_testing_conversation):
    old_prompt_argument = MyUserPromptArgument(name="Alice", topic="Wonderland")
    returned_property = """{
        "user_question": "Am I updated correctly?"
        }"""
    conversation_with_prompt_args = prompt_testing_conversation.update(prompt_argument=old_prompt_argument)
    print(conversation_with_prompt_args.prompt_argument)

    res = prompt_instance.parse_user_prompt(
        name="test_agent",
        llm_response=returned_property,
        old_prompt_argument=conversation_with_prompt_args.prompt_argument,
    )

    assert res.name == "Alice"
    assert res.topic == "Wonderland"
    assert res.user_question == "Am I updated correctly?"


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

    # Use are_classes_structurally_similar instead of direct class comparison
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

    # Create an instance of MyHeuristicPromptArgument to be returned
    prompt_arg_instance = MyHeuristicPromptArgument()

    # Configure create_prompt_argument to return the specific instance
    prompt.create_prompt_argument.return_value = prompt_arg_instance

    # Side effect function for heuristic_sequence
    async def heuristic_sequence_side_effect(prompt_argument, metadata, call_tool):
        # Assert that heuristic_sequence is called with the correct Conversation instance
        assert prompt_argument == conversation.prompt_argument, "Expected correct prompt_argument"
        assert metadata == conversation.metadata, "Expected correct metadata"
        # Return the specific instance
        return prompt_arg_instance

    # Set the heuristic_sequence to the side effect function
    prompt.heuristic_sequence.side_effect = heuristic_sequence_side_effect

    return prompt


@pytest.mark.asyncio
async def test_heuristic_prompt(heuristic_prompt, conversation):
    result = heuristic_prompt.create_prompt_argument()
    assert isinstance(result, MyHeuristicPromptArgument)

    result = await heuristic_prompt.heuristic_sequence(
        prompt_argument=conversation.prompt_argument,
        metadata=conversation.metadata,
        call_tool=lambda x: x,  # Mock or real function as needed
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


def test_render_json_formatting_prompt(prompt_instance):
    # Test with a simple prompt argument
    prompt_args = {"name": "test", "topic": "example"}
    result = prompt_instance.render_json_formatting_prompt(prompt_args)
    assert "name" in result
    assert "topic" in result


def test_render_json_formatting_prompt_with_prompt_fields():
    @dataclass
    class TestPromptArg(PromptArgument):
        visible_field: str = "visible"
        hidden_field: Annotated[str, prompt_field(include=False)] = "hidden"
        another_visible: Annotated[str, prompt_field(include=True)] = "visible2"

    # Create a minimal template for testing
    template = Template(
        "Fields to include: {% for key in schema.keys() %}{{ key }}{% if not loop.last %}, {% endif %}{% endfor %}"
    )

    prompt = BasePrompt(
        agent_description="Test Agent",
        system_template=Template("system"),
        user_template=Template("user"),
        system_prompt_argument_class=TestPromptArg,
        prompt_argument_class=TestPromptArg,
        json_formatting_template=template,
    )

    result = prompt.render_json_formatting_prompt(
        {"visible_field": "test", "hidden_field": "should not appear", "another_visible": "should appear"}
    )

    # Check that visible fields are included
    assert "visible_field" in result
    assert "another_visible" in result
    # Check that hidden field is excluded
    assert "hidden_field" not in result


def test_render_json_formatting_prompt_empty_args():
    @dataclass
    class EmptyPromptArg(PromptArgument):
        pass

    template = Template("Fields: {% for key in keys %}{{ key }}{% if not loop.last %}, {% endif %}{% endfor %}")

    prompt = BasePrompt(
        agent_description="Test Agent",
        system_template=Template("system"),
        user_template=Template("user"),
        system_prompt_argument_class=EmptyPromptArg,
        prompt_argument_class=EmptyPromptArg,
        json_formatting_template=template,
    )

    result = prompt.render_json_formatting_prompt({})
    assert result == "Fields: "


def test_render_json_formatting_prompt_missing_template():
    prompt = BasePrompt(
        agent_description="Test Agent",
        system_template=Template("system"),
        user_template=Template("user"),
        system_prompt_argument_class=PromptArgument,
        prompt_argument_class=PromptArgument,
        json_formatting_template=None,
    )

    with pytest.raises(ValueError) as exc_info:
        prompt.render_json_formatting_prompt({"test": "value"})
    assert str(exc_info.value) == "json_formatting_template is not set."


def test_render_json_formatting_prompt_inheritance():
    @dataclass
    class BasePromptArg(PromptArgument):
        base_visible: str = "base"
        base_hidden: Annotated[str, prompt_field(include=False)] = "hidden"

    @dataclass
    class ChildPromptArg(BasePromptArg):
        child_visible: str = "child"
        child_hidden: Annotated[str, prompt_field(include=False)] = "hidden"

    template = Template("Fields: {% for key in keys %}{{ key }}{% if not loop.last %}, {% endif %}{% endfor %}")

    prompt = BasePrompt(
        agent_description="Test Agent",
        system_template=Template("system"),
        user_template=Template("user"),
        system_prompt_argument_class=ChildPromptArg,
        prompt_argument_class=ChildPromptArg,
        json_formatting_template=template,
    )


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
        system_prompt_argument_class=TestPromptArg,
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
        system_prompt_argument_class=TestPromptArg,
        prompt_argument_class=TestPromptArg,
        json_formatting_template=Template("{{ schema | tojson(indent=2) }}"),
    )

    schema = prompt._generate_json_schema({"name": str, "items": list[ListItem]})

    assert "name" in schema
    assert "items" in schema
    assert isinstance(schema["items"], list)
    assert len(schema["items"]) > 0  # Should have at least one example
    assert "topic" in schema["items"][0]
    assert "priority" in schema["items"][0]
    assert schema["items"][0]["topic"] == ""
    assert schema["items"][0]["priority"] == 0


def test_render_json_formatting_prompt_with_array_type():
    # Test with PlanningUserPromptArgument-like class
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
        system_prompt_argument_class=TestArrayArgument,
        prompt_argument_class=TestArrayArgument,
        json_formatting_template=template,
    )

    result = prompt.render_json_formatting_prompt({"user_query": "", "execution_plan": [], "summary": ""})

    # Check for the structure in the formatted output
    assert '"user_query"' in result
    assert '"execution_plan"' in result
    assert '"step_id"' in result  # From the example step
    assert '"description"' in result
    assert '"function"' in result
    assert '"parameters"' in result
    assert '"depends_on"' in result
    assert '"summary"' in result


def test_json_formatting_with_real_llm_compiler_prompt(prompt_with_array_instance):
    from diskurs.llm_compiler.prompts import PlanningUserPromptArgument

    # Create an instance with proper template
    template = load_template_from_package("diskurs.assets", "json_formatting.jinja2")

    prompt = BasePrompt(
        agent_description="LLM Compiler",
        system_template=Template("system"),
        user_template=Template("user"),
        system_prompt_argument_class=PlanningUserPromptArgument,
        prompt_argument_class=PlanningUserPromptArgument,
        json_formatting_template=template,
    )

    # Generate the JSON formatting instructions
    result = prompt.render_json_formatting_prompt({"user_query": "", "execution_plan": [], "summary": ""})

    # Validate the generated format
    assert "JSON" in result
    assert '"user_query"' in result
    assert '"execution_plan"' in result
    assert '"step_id"' in result
    assert '"description"' in result
    assert '"function"' in result
    assert '"parameters"' in result
    assert '"depends_on"' in result
    assert '"summary"' in result

    # Make sure the result shows an array structure for execution_plan
    assert "[" in result and "]" in result
