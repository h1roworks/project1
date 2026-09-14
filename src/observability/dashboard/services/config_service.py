"""Read and format application settings for the Dashboard UI.

Keeping this small adapter outside the Streamlit page means the page only
renders data; configuration loading and display labels remain easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.settings import Settings, load_settings


@dataclass(frozen=True)
class ComponentConfig:
    """A concise, display-ready description of one pluggable component."""

    name: str
    provider: str
    model: str
    details: str


class ConfigService:
    """Provide Dashboard-friendly views of :class:`core.settings.Settings`.

    ``settings`` can be supplied by tests or another caller.  In normal use,
    the service loads ``config/settings.yaml`` from the project root.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        self._settings = settings
        self._config_path = Path(config_path) if config_path else self.default_config_path()

    @staticmethod
    def default_config_path() -> Path:
        """Return the repository's default configuration path."""
        return Path(__file__).resolve().parents[4] / "config" / "settings.yaml"

    def get_settings(self) -> Settings:
        """Return injected settings or load them lazily from disk."""
        if self._settings is None:
            self._settings = load_settings(self._config_path)
        return self._settings

    def get_component_configs(self) -> list[ComponentConfig]:
        """Format the configurable RAG components shown on the overview page."""
        settings = self.get_settings()
        reranker = settings.rerank
        reranker_provider = reranker.provider if reranker.enabled else "disabled"
        reranker_model = reranker.model if reranker.enabled else "-"
        reranker_details = (
            f"Top {reranker.top_m}" if reranker.enabled else "Reranking is not enabled"
        )

        metrics = ", ".join(settings.evaluation.metrics) or "No metrics configured"
        return [
            ComponentConfig(
                name="LLM",
                provider=settings.llm.provider,
                model=settings.llm.model,
                details=f"temperature={settings.llm.temperature}, max_tokens={settings.llm.max_tokens}",
            ),
            ComponentConfig(
                name="Embedding",
                provider=settings.embedding.provider,
                model=settings.embedding.model,
                details=(
                    f"dimensions={settings.embedding.dimensions}, "
                    f"batch_size={settings.embedding.batch_size}"
                ),
            ),
            ComponentConfig(
                name="Splitter",
                provider=settings.splitter.strategy,
                model="-",
                details=(
                    f"chunk_size={settings.splitter.chunk_size}, "
                    f"overlap={settings.splitter.chunk_overlap}"
                ),
            ),
            ComponentConfig(
                name="Reranker",
                provider=reranker_provider,
                model=reranker_model,
                details=reranker_details,
            ),
            ComponentConfig(
                name="Evaluator",
                provider=settings.evaluation.provider,
                model="-",
                details=metrics,
            ),
        ]
