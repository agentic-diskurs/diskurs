from diskurs import Conversation
from prompt import URLInstruction, ParallelMultiStepUserPrompt


def branch_conversation(conversation: Conversation) -> list[Conversation]:
    conversations: list[Conversation] = []
    for url in conversation.user_prompt_argument.urls:
        prompt_arguments = conversation.user_prompt_argument
        prompt_arguments.urls = url
        prompt_arguments.branching = False
        prompt_arguments.joining = True

        conversations.append(conversation.update(user_prompt_argument=prompt_arguments))

    return conversations


def join_conversations(conversations: list[Conversation]) -> Conversation:
    final_instructions = []
    for conv in conversations:
        url = (
            conv.user_prompt_argument.urls[0]
            if isinstance(conv.user_prompt_argument.urls, list)
            else conv.user_prompt_argument.urls
        )
        list_name = conv.user_prompt_argument.list_name
        action = conv.user_prompt_argument.action

        final_instructions.append(URLInstruction(url=url, list_name=list_name, action=action))
    return conversations[0].update(user_prompt_argument=ParallelMultiStepUserPrompt(instructions=final_instructions))
