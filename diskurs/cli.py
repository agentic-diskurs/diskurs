from pathlib import Path
import click


TEMPLATES_DIR = Path(__file__).parent / "agent_templates"


@click.group()
def cli():
    """Command-line interface for the diskurs package."""
    pass


@cli.command()
@click.option("--name", prompt="Agent name", help="Name of the new agent")
def create_agent(name):
    """Create a new agent with the specified name."""
    agent_dir = Path.cwd() / name

    if not agent_dir.exists():
        agent_dir.mkdir(parents=True)

        # Read and write each template file to the appropriate location
        for template_name in [
            "prompt.py",
            "user_template.jinja2",
            "system_template.jinja2",
        ]:
            template_path = TEMPLATES_DIR / template_name
            output_file = agent_dir / template_name.replace(".jinja2", "")

            with open(template_path, "r") as template_file:
                content = template_file.read()

            with open(output_file, "w") as output:
                output.write(content)

        click.echo(f"Agent {name} created successfully at {agent_dir}")
    else:
        click.echo(f"Agent {name} already exists.")


if __name__ == "__main__":
    cli()
