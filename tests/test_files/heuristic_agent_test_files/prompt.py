from dataclasses import dataclass

from diskurs import CallTool, PromptArgument, PromptValidationError


@dataclass
class MyHeuristicPromptArgument(PromptArgument):
    name: str = ""
    topic: str = ""
    user_question: str = ""
    answer: str = ""


def heuristic_sequence(
    prompt_argument: MyHeuristicPromptArgument, metadata: dict, call_tool: CallTool
) -> MyHeuristicPromptArgument:

    ticket_context = call_tool(
        function_name="extract_ticket_context",
        arguments={"ticket_id": metadata["ticket_id"], "host_id": prompt_argument.topic},
    )

    # ... a lot of other stuff

    return MyHeuristicPromptArgument(
        name=ticket_context["name"],
        topic=ticket_context["topic"],
        user_question=ticket_context["user_question"],
        answer=ticket_context["answer"],
    )
