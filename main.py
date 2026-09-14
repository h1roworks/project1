"""Knowledge Hub v1 entry point."""

from pathlib import Path

from core.settings import Settings, load_settings
from observability.logger import get_logger

logger = get_logger()

CONFIG_PATH = Path(__file__).parent / "config" / "settings.yaml"


def main() -> None:
    settings: Settings = load_settings(CONFIG_PATH)
    logger.info(
        "Knowledge Hub v1 starting | llm=%s/%s | embedding=%s/%s",
        settings.llm.provider,
        settings.llm.model,
        settings.embedding.provider,
        settings.embedding.model,
    )
    print("Knowledge Hub v1 started.")


if __name__ == "__main__":
    main()
