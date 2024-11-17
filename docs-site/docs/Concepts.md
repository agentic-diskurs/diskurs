# Concepts

## Agents

As a multi-agent system, Diskurs uses the concept of agents to group semantically similar parts of a workflow. An agent can be thought of as a persona capable of handling a specific domain or sub-domain of a broader space.

There are two types of agents:

1. **Conductor agents:** These agents coordinate the overall execution of a workflow by dispatching tasks to different agents.
2. **Multi-step agents:** These agents carry out specific tasks, such as extracting information from a user's query, answering questions, or acquiring and enriching information using tools.

### Multi-Step Agents

Agents are designed using a combination of Python code and prompt templates in Jinja2 format. You can pass all the respective assets directly into an agent or use specific files to organize an agent's assets in a more structured way.

Below is an example of how you would typically organize an agent's files. The `system_prompt.jinja2` and `user_prompt.jinja2` files contain the prompts to instruct the agent, while the `prompt.py` file contains the code linked to those templates.

```bash
agent_assets
├── agent_description.txt
├── prompt.py
├── system_prompt.jinja2
└── user_prompt.jinja2
```

### Agent Description

This text file contains a concise description of the agent. It should mention the agent's capabilities and under what circumstances it should be addressed. This description is used by the conductor agent to correctly coordinate a workflow by dispatching a query to the most appropriate agent for solving the current task.

### System Prompt

A system prompt provides an overall description of the persona a given agent should impersonate. It should describe precisely what an agent's responsibilities and capabilities are.

Within the system prompt template, you can use all of Jinja2's functionality, including arbitrary placeholders such as `topic` and `mode`.

```jinja2
- You are a helpful assistant, a seasoned expert in the area of {{ topic }}.
- Your audience consists of {{ mode }}.
- You can answer a user's queries and extract entities from the provided text.
```

In the `prompt.py` file, you can specify the prompt arguments for the system prompt. The system prompt argument class is specific to each agent, containing the placeholders for a specific prompt as its properties. Each concrete prompt argument should extend the `PromptArgument` class. There are two ways to specify the concrete values of the system prompt:

1. Specify them directly as default arguments of the system prompt argument class.
2. Specify them in the `config.yml`.

The second approach is recommended as it allows for better separation of concerns and follows the pattern of managing all configuration from a central location.

```python
from dataclasses import dataclass
from diskurs.entities import PromptArgument

@dataclass
class MySystemPromptArgument(PromptArgument):
    topic: str = "Cybersecurity"
    mode: str = "professional network engineers"
```

### User Prompt

The user prompt is an integral part of Diskurs. It is similar to the system prompt in that it also consists of a Jinja2 template and a corresponding prompt argument class, but it extends its functionality. The user prompt argument has four purposes:

1. **Replacement of placeholders:** Similar to the system prompt arguments, you can use the properties of the user prompt argument to replace placeholders in the user prompt.
2. **Conditional rendering:** You can use properties of the user prompt argument to conditionally display or hide certain parts of the user prompt.
3. **Validation of the LLM's response:** Diskurs can automatically try to parse an LLM's response into the user prompt arguments of a given agent. If it fails, it will automatically send a request to the LLM containing instructions on how to correct its output so that Diskurs can parse it into a valid prompt argument. You can extend this mechanism by implementing a validation function defining specific constraints on the allowed values or combinations of values contained in the prompt arguments.
4. **Finishing an agent's turn:** The values of the user prompt arguments are used to decide whether an agent has reached a state where it can hand off to the conductor agent again. This occurs when the currently active agent has finished its task and wants to return the information it acquired.

Below is an example of a user template:

```jinja2
{% if not name or not topic or not user_question %}
Extract all the following entities from the user's query:
- name: the user's name
- topic: the topic of the user's query
- user_question: a concise description of the user's question
{% else %}
- The user's name is {{ name }}.
- They are asking questions about {{ topic }}.
- The user's query is: {{ user_question }}.
Answer the user's question and output the answer in the answer field.
{% endif %}
```

The first line shows an example of *conditional rendering*. Here, we show the text contained within the *if-block* whenever one of `name`, `topic`, or `user_question` is empty; otherwise, we show the *else* block. Lines 7-9 use the same approach for replacing prompt arguments as already shown for the system prompt.

Below are the user prompt arguments:

```python
@dataclass
class FirstUserPromptArgument(PromptArgument):
    name: str = ""
    topic: str = ""
    user_question: str = ""
    answer: str = ""
```

The user prompt argument class contains all the variables used for conditional rendering and/or replacement contained in the Jinja2 template. These same arguments are also used to decide whether the answer returned by the LLM is valid. Syntactic validation of the returned answer is automatically carried out by Diskurs. Think of the validation function as a means to perform *semantic validation*:

```python
valid_topics = ["firewall", "proxy", "connectivity"]

def is_valid(arg: FirstUserPromptArgument) -> bool:
    if topic := arg.topic:
        if topic not in valid_topics:
            raise PromptValidationError(f"Please assign a valid topic. Valid topics are: {valid_topics}")
    if arg.topic and not arg.user_question:
        raise PromptValidationError("Please extract a concise description of the user's question")
    return True
```

In the above example, we first ensure that if the LLM extracted a topic, it belongs to one of the valid topics. If it does not belong to a valid topic, we instruct the LLM to do so by raising a `PromptValidationError`. This error will prompt Diskurs to relay the message specified within the error directly to the LLM. Therefore, it is important to specify a descriptive error message to help the LLM correct its own output.

In the second case, we ensure that the user's question has been extracted, as we know that we were able to already extract the respective topic.

Lastly, we have to implement an `is_final` function, which determines when we consider a given agent's task as finished:

```python
def is_final(arg: FirstUserPromptArgument) -> bool:
    return len(arg.answer) > 10 and arg.topic
```

Here, we require the user prompt argument to contain a topic and an answer longer than 10 characters. If both conditions are fulfilled, the agent will stop and return its result to the conductor.

## Conductor Agents

The conductor agent builds on the same concepts as already introduced for the multi-step agent. As the task of the conductor agent is often identical across different workflows, you don't have to specify the system and user prompts for a conductor agent. If no prompts are provided, we use the default conductor prompt. The system prompt does the following:

- It renders a list of all agents' names and their descriptions.
- It adds this list to the system prompt.
- It instructs the conductor agent to pick the agent most suitable to solve the problem at hand.

The conductor's user prompt instructs the agent to analyze the conversation so far and, based on it, suggest the next agent that best contributes to the conversation's context and progression.

### Long-term Memory

In addition to the prompt arguments, each conductor agent contains its own long-term memory. We use the long-term memory to retain information acquired across multiple agents' turns. Whenever an agent hands off to the conductor agent, the conductor agent updates its long-term memory by checking the properties of the last agent's prompt arguments, extracting the values where the property names match those of its long-term memory, and updating the values of its long-term memory.

### Finalize

This section should provide a summary or conclusion of the documentation, but it appears to be missing. Please provide the necessary content to complete this section.