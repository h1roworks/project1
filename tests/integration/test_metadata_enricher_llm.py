"""C6 集成测试：MetadataEnricher 真实 LLM 调用（连通性与降级）。

对齐 DEV_SPEC C6 验收标准"开启 LLM 的集成测试用例"：
- 配置就绪（config/settings.yaml 中 dashscope provider），开启 enrich_with_llm 时
  必须真实调用 LLM，并产出语义丰富的 title/summary/tags；
- 无效模型名 / 无 API Key 时优雅降级到规则结果，不抛致命异常。

⚠️ 会产生真实 API 调用与费用；未配置 DASHSCOPE_API_KEY 时自动跳过。
"""

from pathlib import Path

import pytest

from core.settings import load_settings
from core.types import Chunk
from ingestion.transform.metadata_enricher import MetadataEnricher

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


def _make_chunk(text: str) -> Chunk:
    return Chunk(
        id="doc_001_0000_abc12345",
        text=text,
        metadata={"source_path": "/data/sample.pdf", "chunk_index": 0},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc_001",
    )


def _settings(enrich_with_llm: bool = True) -> object:
    """加载真实配置，但把 transform 开关指向 C6 的 enrich_with_llm。"""
    base = load_settings(CONFIG_PATH)
    return type(
        "FullSettings",
        (),
        {
            "transform": type(
                "T",
                (),
                {
                    "refine": True,
                    "refine_with_llm": False,
                    "enrich_with_llm": enrich_with_llm,
                    "prompt_path": "",
                },
            )(),
            "llm": base.llm,
        },
    )()


@pytest.mark.integration
def test_real_llm_enriches_metadata_with_key() -> None:
    """真实 LLM 连通性：开启开关后产出语义丰富的 title/summary/tags。"""
    settings = _settings()
    if not settings.llm.api_key:
        pytest.skip("DASHSCOPE_API_KEY 未配置，跳过真实 LLM 集成测试")

    enricher = MetadataEnricher(settings=settings)
    out = enricher.transform(
        _make_chunk(
            "RAG（检索增强生成）通过混合检索结合稠密向量与稀疏关键词召回，"
            "在生成前把相关上下文注入提示词，从而减少幻觉并提升回答质量。"
        )
    )
    assert out.metadata["enriched"] is True
    assert out.metadata["enriched_by_llm"] is True
    assert out.metadata["llm_error"] not in out.metadata
    assert out.metadata["title"]
    assert out.metadata["summary"]
    assert out.metadata["tags"]
    assert len(out.metadata["tags"]) >= 1
    print("\n[LLM title]", out.metadata["title"])
    print("[LLM summary]", out.metadata["summary"])
    print("[LLM tags]", out.metadata["tags"])


@pytest.mark.integration
def test_invalid_model_degrades_gracefully() -> None:
    """降级机制：无效模型名（或无 key）时回退到规则结果，不崩溃。"""
    settings = _settings()
    if not settings.llm.api_key:
        pytest.skip("DASHSCOPE_API_KEY 未配置，跳过真实 LLM 集成测试")

    import dataclasses

    bad = dataclasses.replace(settings.llm, model="this-model-does-not-exist")
    settings.llm = bad
    enricher = MetadataEnricher(settings=settings)
    out = enricher.transform(_make_chunk("RAG 混合检索。\n正文内容。"))
    assert out.metadata["enriched_by_llm"] is False
    assert "llm_error" in out.metadata  # 标记降级原因，但不抛异常
    assert out.metadata["title"]
    assert out.metadata["summary"]
    assert out.metadata["tags"]
