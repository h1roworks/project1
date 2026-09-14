"""Phase A3: settings loading and validation tests."""

from pathlib import Path

import pytest

from core.settings import (
    EmbeddingSettings,
    LLMSettings,
    Settings,
    SettingsError,
    VectorStoreSettings,
    load_settings,
    validate_settings,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_CONFIG = REPO_ROOT / "config" / "settings.yaml"

MINIMAL_VALID_YAML = """
llm:
  provider: openai
  model: gpt-4o
embedding:
  provider: openai
  model: text-embedding-3-small
vector_store:
  provider: chroma
  collection: test
retrieval:
  top_k: 5
observability:
  log_dir: logs
  trace_file: logs/traces.jsonl
"""


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "settings.yaml"
    p.write_text(content, encoding="utf-8")
    return p


def test_load_real_config() -> None:
    """The real config/settings.yaml must load and validate."""
    settings = load_settings(REAL_CONFIG)
    assert isinstance(settings, Settings)
    assert settings.llm.provider
    assert settings.llm.model
    assert settings.embedding.provider
    assert settings.embedding.model
    assert settings.vector_store.provider == "chroma"


def test_load_valid_minimal(tmp_path: Path) -> None:
    settings = load_settings(_write(tmp_path, MINIMAL_VALID_YAML))
    assert settings.retrieval.top_k == 5
    assert settings.rerank.provider == "none"  # default applied


def test_missing_required_field_reports_path(tmp_path: Path) -> None:
    """Missing embedding.provider must raise with the exact field path."""
    cfg = MINIMAL_VALID_YAML.replace(
        "embedding:\n  provider: openai\n  model: text-embedding-3-small\n", ""
    )
    with pytest.raises(SettingsError, match=r"embedding\.provider"):
        load_settings(_write(tmp_path, cfg))


def test_missing_model_field_reports_path(tmp_path: Path) -> None:
    cfg = MINIMAL_VALID_YAML.replace("  model: gpt-4o", "")
    with pytest.raises(SettingsError, match=r"llm\.model"):
        load_settings(_write(tmp_path, cfg))


def test_nonexistent_file(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="不存在"):
        load_settings(tmp_path / "nope.yaml")


def test_invalid_yaml(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="解析失败"):
        load_settings(_write(tmp_path, "llm: [unclosed"))


def test_env_placeholder_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "secret-123")
    cfg = MINIMAL_VALID_YAML + "rerank:\n  provider: llm\n  model: m\n"
    settings = load_settings(_write(tmp_path, cfg))
    assert isinstance(settings.llm, LLMSettings)
    assert isinstance(settings.embedding, EmbeddingSettings)


def test_validate_settings_ok() -> None:
    s = Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o"),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreSettings(provider="chroma", collection="test"),
    )
    validate_settings(s)


def test_validate_settings_missing_field() -> None:
    s = Settings(
        llm=LLMSettings(provider="", model="gpt-4o"),
        embedding=EmbeddingSettings(provider="openai", model="m"),
    )
    with pytest.raises(SettingsError, match=r"llm\.provider"):
        validate_settings(s)
