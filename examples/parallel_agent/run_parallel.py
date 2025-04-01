import asyncio
from pathlib import Path
from dotenv import load_dotenv

from diskurs import create_forum_from_config, DiskursInput

load_dotenv()


async def main() -> None:
    """
    This example demonstrates how to run a parallel agent, which branches and joins
    a conversation.
    """
    forum = create_forum_from_config(config_path=Path("parallel_config.yaml"), base_path=Path(__file__).parent)

    input_data = DiskursInput(
        user_query="Please configure the following URLs: "
        "www.acme.com: add to whitelist"
        "www.example.com: add to whitelist"
        "add the following to our blacklist:"
        "www.google.com, 192.168.1.10"
        "remove the following from our blacklist:"
        "innocence.org, helpfule.com"
    )

    response = await forum.ama(input_data)
    for instruction in response["instructions"]:
        print(f"{'='*20}\nAnswer:\n{instruction}\n")
    print(f"{'='*20}")


if __name__ == "__main__":
    asyncio.run(main())
