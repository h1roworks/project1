"""Configuration loading and validation.

Loads config/settings.yaml into strongly-typed Settings dataclasses.
Performs structural (required-field) validation only -- no network or IO
initialization happens here.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{(\w+)\}")


class SettingsError(ValueError):
    """Raised when settings are missing or malformed."""


@dataclass
class LLMSettings:
    provider: str = ""
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.1
    max_tokens: int = 2048


@dataclass
class EmbeddingSettings:
    provider: str = ""
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    dimensions: int = 1024
    batch_size: int = 32


@dataclass
class VisionLLMSettings:
    provider: str = ""
    model: str = ""
    api_key: str = ""
    base_url: str = ""


@dataclass
class VectorStoreSettings:
    provider: str = ""
    collection: str = ""
    persist_dir: str = ""


@dataclass
class RetrievalSettings:
    top_k: int = 10
    dense_top_k: int = 20
    sparse_top_k: int = 20
    fusion_k: int = 60
    hard_filters: list[str] = field(default_factory=list)


@dataclass
class SplitterSettings:
    strategy: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list[str] = field(default_factory=list)


@dataclass
class RerankSettings:
    provider: str = "none"
    enabled: bool = False
    top_m: int = 10
    model: str = ""
    timeout_seconds: int = 10


@dataclass
class EvaluationSettings:
    provider: str = "composite"
    metrics: list[str] = field(default_factory=list)
    golden_test_set_path: str = ""


@dataclass
class ObservabilitySettings:
    log_dir: str = "logs"
    trace_file: str = "logs/traces.jsonl"
    app_log: str = "logs/app.log"
    dashboard: dict[str, Any] = field(default_factory=dict)


@dataclass
class Settings:
    llm: LLMSettings
    embedding: EmbeddingSettings
    vision_llm: VisionLLMSettings = field(default_factory=VisionLLMSettings)
    vector_store: VectorStoreSettings = field(default_factory=VectorStoreSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    splitter: SplitterSettings = field(default_factory=SplitterSettings)
    rerank: RerankSettings = field(default_factory=RerankSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    observability: ObservabilitySettings = field(default_factory=ObservabilitySettings)


# (section, field) pairs that must be present and non-empty after load.
_REQUIRED_FIELDS: list[tuple[str, str]] = [
    ("llm", "provider"),
    ("llm", "model"),
    ("embedding", "provider"),
    ("embedding", "model"),
    ("vector_store", "provider"),
    ("vector_store", "collection"),
    ("retrieval", "top_k"),
    ("observability", "log_dir"),
    ("observability", "trace_file"),
]

_SECTION_TYPES = {
    "llm": LLMSettings,
    "embedding": EmbeddingSettings,
    "vision_llm": VisionLLMSettings,
    "vector_store": VectorStoreSettings,
    "retrieval": RetrievalSettings,
    "splitter": SplitterSettings,
    "rerank": RerankSettings,
    "evaluation": EvaluationSettings,
    "observability": ObservabilitySettings,
}


def _resolve_env(value: Any) -> Any:
    """Replace ${ENV_VAR} placeholders with their environment values.

    Unresolvable placeholders become empty strings (the caller decides whether
    the field is required). Non-string values pass through unchanged.
    """
    if not isinstance(value, str):
        return value

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        return os.environ.get(name, "")

    return _ENV_PATTERN.sub(_replace, value)


def _resolve_env_recursive(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _resolve_env_recursive(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_recursive(v) for v in value]
    return _resolve_env(value)


def _build_section(cls: type[Any], section: str, data: Any) -> Any:
    if not isinstance(data, dict):
        raise SettingsError(f"配置段 '{section}' 必须是一个映射（dict）")
    return cls(**_resolve_env_recursive(data))


def validate_settings(settings: Settings) -> None:
    """Check that all required fields are present and non-empty.

    Error messages include the full field path, e.g. 'embedding.provider'.
    """
    for section, field_name in _REQUIRED_FIELDS:
        value = getattr(getattr(settings, section), field_name, None)
        if value is None or value == "" or value == []:
            path = f"{section}.{field_name}"
            raise SettingsError(f"缺少必填配置字段: '{path}'")


def load_settings(path: str | Path) -> Settings:
    """Read a YAML config file into a validated Settings object."""
    path = Path(path)
    if not path.exists():
        raise SettingsError(f"配置文件不存在: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SettingsError(f"配置文件 YAML 解析失败: {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise SettingsError(f"配置文件顶层必须是映射（dict）: {path}")

    sections: dict[str, Any] = {}
    for name, cls in _SECTION_TYPES.items():
        data = raw.get(name, {})
        sections[name] = _build_section(cls, name, data)

    settings = Settings(**sections)
    validate_settings(settings)
    return settings
