import asyncio
from pathlib import Path
import logging

from dotenv import load_dotenv
from diskurs import create_forum_from_config, DiskursInput

logging.basicConfig(level=logging.DEBUG)

load_dotenv()


async def main() -> None:
    forum = create_forum_from_config(config_path=Path("business_report_config.yaml"), base_path=Path(__file__).parent)

    input_data = DiskursInput(
        user_query="Generate a quarterly business report with the following information: "
        "1. Calculate revenue growth from our sales data "
        "2. Analyze employee performance metrics "
        "3. Estimate budget projections for next quarter "
        "4. Provide 3 strategic recommendations based on this analysis"
    )

    response = await forum.ama(input_data)
    print(f"{'='*20}\nAnswer:\n{response}\n{'='*20}")


if __name__ == "__main__":
    asyncio.run(main())
