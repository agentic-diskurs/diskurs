from pathlib import Path
from dotenv import load_dotenv

from diskurs import create_forum_from_config, DiskursInput

load_dotenv()


def main(config: Path):
    forum = create_forum_from_config(config_path=config, base_path=Path(__file__).parent)

    diskurs_input = DiskursInput(
        user_query="What is the meaning of life?",
        metadata={"user_id": "1234"},
        conversation_id="1234",
    )

    res = forum.ama(diskurs_input)
    print(res)


if __name__ == "__main__":
    main(Path(__file__).parent / "config.yml")
