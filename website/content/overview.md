---
title: Overview
sidebar: false  
sidebarlogo: fresh-white-alt # From (static/images/logo/)
include_footer: false # or false to display the footer
---

# Declarative Configuration
Diskurs simplifies the setup and management of multi-agent systems with a declarative approach to configuration.
Instead of hardcoding system details, everything is centrally defined in a YAML configuration file, offering unparalleled flexibility and ease of use.

## What You Can Configure
In the YAML configuration file, you can define:

LLM settings: Specify the language model to use, including its parameters.
Agent definitions: Identify agent types and the locations of their associated assets.
Communication patterns: Map which agents interact with each other.
Tooling: Outline the tools available, their code locations, and the agents that rely on them.
External dependencies: Configure connections to databases, object storage, or other external services.
## Why Declarative Configuration Matters
Our approach empowers you to adapt your system with ease. Whether it’s adjusting communication patterns, assigning different LLMs to agents, or reconfiguring external service integrations, everything can be modified by simply updating the YAML file.
This eliminates the need to alter code, streamlining the development process and reducing the risk of introducing errors.

By providing this central configuration file, Diskurs automatically sets up a fully operational system tailored to your specifications, allowing you to focus on building functionality rather than managing infrastructure.

# Jinja2 Templating
Diskurs adheres to the principle of separation of concerns by using Jinja2 templates for all prompts.
This includes prompts for agents as well as those used internally by the framework.

By leveraging Jinja2's powerful features, you can create highly sophisticated prompts. Conditional rendering, loops, and other advanced templating capabilities allow you to dynamically adapt prompts to various contexts. For instance, you can tailor responses based on specific agent states, include or exclude information conditionally, or iterate over dynamic lists of input data—all without modifying the underlying code.

Moreover, you can easily customize prompts to meet your specific needs, such as translating them into different languages or aligning them with unique project requirements. This flexibility ensures that Diskurs remains adaptable and user-friendly.

# Extensions
Diskurs is built on a modular architecture with entities like LLM clients, agents, a message bus, and more.
These entities are registered through decorators, enabling Diskurs to automatically instantiate them based on the YAML configuration file.

We’ve designed Diskurs to offer maximum flexibility for customization. If you need to implement deep customizations, simply adhere to the exposed interface of the respective entity. For minor adjustments, you can subclass the entities provided by Diskurs, allowing you to tweak specific details without starting from scratch.

This modularity and extensibility ensure that Diskurs adapts seamlessly to your unique requirements. To add new entities:

Apply the appropriate decorators to your custom entity.
List the module in the configuration file.
Diskurs will detect and integrate the new entities automatically, enabling your system to evolve effortlessly as your needs grow.