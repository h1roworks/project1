"""C6: MetadataEnricher 单元测试。

验证：
- 规则增强：title 取标题行/首行，summary 取首句，tags 抽取中英文关键词；
  空内容兜底非空；幂等
- metadata 契约：title/summary/tags 恒非空；enriched / enriched_by_llm 标志；
  入参不被修改；id / source_ref / offset / text 作为稳定锚点保持不变
- 可选 LLM 增强：注入 Fake LLM 验证 JSON 改写（含代码围栏/纯文本容错）、
  失败/空输出/缺字段静默回退、未启用时不调用、prompt 模板注入、
  经 settings.llm 工厂创建
- 与 Chunker 的链路（C4 → C6）
"""

import hashlib
import json
from types import SimpleNamespace

import pytest

from core.settings import LLMSettings, TransformSettings
from core.types import Chunk
from ingestion.chunking.document_chunker import DocumentChunker
from ingestion.transform.base_transform import BaseTransform
from ingestion.transform.metadata_enricher import MetadataEnricher
from libs.llm.base_llm import BaseLLM, ChatResponse
from libs.llm.llm_factory import LLMFactory

# ---------- fixtures ----------


def make_chunk(text: str, doc_id: str = "doc_001", index: int = 0, **metadata) -> Chunk:
    meta = {"source_path": "/data/sample.pdf", "chunk_index": index, "image_refs": []}
    meta.update(metadata)
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return Chunk(
        id=f"{doc_id}_{index:04d}_{content_hash}",
        text=text,
        metadata=meta,
        start_offset=0,
        end_offset=len(text),
        source_ref=doc_id,
    )


def make_enriched(extra: dict[str, object]) -> str:
    """构造合法 JSON 响应（可选叠加 title/summary/tags 覆盖）。"""
    payload = {
        "title": "LLM 生成的标题",
        "summary": "LLM 生成的摘要。",
        "tags": ["tag_a", "tag_b", "tag_c"],
    }
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


class MockLLM:
    """注入到 MetadataEnricher 的假 LLM，绕开真实调用只测编排逻辑。"""

    def __init__(self, content: str = "", exc: Exception | None = None) -> None:
        self.content = content
        self.exc = exc
        self.calls: list[list[dict]] = []

    def chat(self, messages, trace=None, **kwargs) -> ChatResponse:
        self.calls.append(messages)
        if self.exc:
            raise self.exc
        return ChatResponse(content=self.content, model="mock")


# ---------- 规则增强：title ----------


def test_rule_title_from_heading() -> None:
    out = MetadataEnricher().transform(make_chunk("# RAG Overview\n正文内容。"))
    assert out.metadata["title"] == "RAG Overview"


def test_rule_title_from_first_line() -> None:
    out = MetadataEnricher().transform(make_chunk("混合检索架构\n正文内容。"))
    assert out.metadata["title"] == "混合检索架构"


def test_rule_title_truncated() -> None:
    long = "标题" + "很长很长的文字" * 30
    out = MetadataEnricher().transform(make_chunk(f"{long}\n正文内容。"))
    assert out.metadata["title"].endswith("…")
    assert len(out.metadata["title"]) <= 80


# ---------- 规则增强：summary ----------


def test_rule_summary_from_first_sentence() -> None:
    out = MetadataEnricher().transform(make_chunk("这是第一句。这是第二句。"))
    assert out.metadata["summary"] == "这是第一句。"


def test_rule_summary_without_punctuation_takes_all() -> None:
    out = MetadataEnricher().transform(make_chunk("没有句号的一段内容"))
    assert out.metadata["summary"] == "没有句号的一段内容"


# ---------- 规则增强：tags ----------


def test_rule_tags_extract_chinese_and_english() -> None:
    text = "RAG 混合检索 dense retrieval 与 sparse retrieval"
    out = MetadataEnricher().transform(make_chunk(text))
    tags = out.metadata["tags"]
    assert "RAG" in tags
    assert "混合检索" in tags
    assert "dense" in tags
    assert "retrieval" in tags
    assert "sparse" in tags


def test_rule_tags_deduplicated_and_limited() -> None:
    text = "apple apple banana banana cherry cherry"
    tags = MetadataEnricher().transform(make_chunk(text)).metadata["tags"]
    assert len(tags) == 3  # 去重
    assert tags.count("apple") == 1


def test_rule_tags_fallback_to_general() -> None:
    text = "the and for this that"  # 全是停用词
    out = MetadataEnricher().transform(make_chunk(text))
    assert out.metadata["tags"] == ["general"]


def test_rule_tags_skip_stopwords() -> None:
    text = "the RAG and 检索 for 增强"
    tags = MetadataEnricher().transform(make_chunk(text)).metadata["tags"]
    assert "the" not in tags
    assert "RAG" in tags
    assert "检索" in tags
    assert "增强" in tags


# ---------- 规则增强：兜底非空 ----------


def test_empty_chunk_still_non_empty_metadata() -> None:
    out = MetadataEnricher().transform(make_chunk("   \n  "))
    assert out.metadata["title"]
    assert out.metadata["summary"]
    assert out.metadata["tags"]


def test_metadata_always_has_three_keys() -> None:
    for text in ("# 只有标题", "只有一句话。", "没有标点符号的内容", "   "):
        meta = MetadataEnricher().transform(make_chunk(text)).metadata
        assert "title" in meta and meta["title"]
        assert "summary" in meta and meta["summary"]
        assert "tags" in meta and meta["tags"]


# ---------- metadata 契约 / 锚点 ----------


def test_metadata_flags_and_input_not_mutated() -> None:
    chunk = make_chunk("RAG 混合检索。\n正文内容。")
    before = dict(chunk.metadata)
    out = MetadataEnricher().transform(chunk)
    assert out.metadata["enriched"] is True
    assert out.metadata["enriched_by_llm"] is False
    assert chunk.metadata == before  # 入参未被修改
    assert "enriched" not in chunk.metadata


def test_anchors_preserved() -> None:
    chunk = make_chunk("正文内容。\n\n- 3 -\n\n结尾内容。")
    out = MetadataEnricher().transform(chunk)
    assert out.id == chunk.id
    assert out.text == chunk.text
    assert out.source_ref == chunk.source_ref
    assert out.start_offset == chunk.start_offset
    assert out.end_offset == chunk.end_offset


def test_rule_enrich_is_idempotent() -> None:
    text = "RAG 混合检索结合了 dense 与 sparse 两种召回方式。"
    enricher = MetadataEnricher()
    out1 = enricher.transform(make_chunk(text))
    out2 = enricher.transform(make_chunk(text))
    assert out1 == out2


def test_transform_many_degrades_per_chunk() -> None:
    """BaseTransform 原子化契约：单 chunk 失败不拖垮整批（此处让 Enricher 抛异常）。"""

    class Boom(MetadataEnricher):
        def transform(self, chunk, trace=None) -> Chunk:
            if "bad" in chunk.text:
                raise ValueError("boom")
            return super().transform(chunk, trace)

    chunks = [make_chunk("好内容。"), make_chunk("bad content"), make_chunk("好内容二。")]
    out = Boom().transform_many(chunks)
    assert out[1].metadata["transform_errors"] == ["metadata_enricher: boom"]
    assert "enriched" not in out[1].metadata  # 失败的 chunk 原样返回，未写入增强标志
    assert out[0].metadata["title"]  # 其余 chunk 正常增强
    assert [c.id for c in out] == [c.id for c in chunks]


# ---------- 可选 LLM 增强 ----------


def test_llm_enrich_rewrites_when_enabled() -> None:
    settings = TransformSettings(enrich_with_llm=True)
    llm = MockLLM(content=make_enriched({}))
    enricher = MetadataEnricher(settings=settings, llm=llm)
    out = enricher.transform(make_chunk("原始内容。"))
    assert out.metadata["title"] == "LLM 生成的标题"
    assert out.metadata["summary"] == "LLM 生成的摘要。"
    assert out.metadata["tags"] == ["tag_a", "tag_b", "tag_c"]
    assert out.metadata["enriched_by_llm"] is True
    assert len(llm.calls) == 1


def test_llm_enrich_parses_code_fenced_json() -> None:
    settings = TransformSettings(enrich_with_llm=True)
    fenced = f"```json\n{make_enriched({})}\n```"
    enricher = MetadataEnricher(settings=settings, llm=MockLLM(content=fenced))
    out = enricher.transform(make_chunk("原始内容。"))
    assert out.metadata["title"] == "LLM 生成的标题"
    assert out.metadata["enriched_by_llm"] is True


def test_llm_failure_degrades_to_rule_result() -> None:
    settings = TransformSettings(enrich_with_llm=True)
    llm = MockLLM(exc=ConnectionError("llm down"))
    enricher = MetadataEnricher(settings=settings, llm=llm)
    out = enricher.transform(make_chunk("RAG 混合检索。\n正文内容。"))
    assert out.metadata["enriched_by_llm"] is False
    assert "llm down" in out.metadata["llm_error"]
    assert out.metadata["title"]  # 规则结果保留
    assert out.metadata["summary"]
    assert out.metadata["tags"]


def test_llm_empty_output_degrades() -> None:
    settings = TransformSettings(enrich_with_llm=True)
    enricher = MetadataEnricher(settings=settings, llm=MockLLM(content="   \n  "))
    out = enricher.transform(make_chunk("RAG 混合检索。"))
    assert out.metadata["enriched_by_llm"] is False
    assert "空文本" in out.metadata["llm_error"]
    assert out.metadata["title"]


def test_llm_invalid_json_degrades() -> None:
    settings = TransformSettings(enrich_with_llm=True)
    enricher = MetadataEnricher(settings=settings, llm=MockLLM(content="抱歉，我无法理解"))
    out = enricher.transform(make_chunk("RAG 混合检索。"))
    assert out.metadata["enriched_by_llm"] is False
    assert "llm_error" in out.metadata
    assert out.metadata["title"]  # 规则结果保留


def test_llm_missing_fields_degrades() -> None:
    settings = TransformSettings(enrich_with_llm=True)
    bad = make_enriched({"tags": []})
    enricher = MetadataEnricher(settings=settings, llm=MockLLM(content=bad))
    out = enricher.transform(make_chunk("RAG 混合检索。"))
    assert out.metadata["enriched_by_llm"] is False
    assert "缺少必要字段" in out.metadata["llm_error"]


def test_llm_not_called_when_disabled() -> None:
    llm = MockLLM(content=make_enriched({}))
    enricher = MetadataEnricher(settings=TransformSettings(enrich_with_llm=False), llm=llm)
    out = enricher.transform(make_chunk("正文内容。"))
    assert llm.calls == []
    assert out.metadata["enriched_by_llm"] is False


def test_missing_llm_degrades_silently() -> None:
    enricher = MetadataEnricher(settings=TransformSettings(enrich_with_llm=True))
    out = enricher.transform(make_chunk("正文内容。"))
    # settings 是纯 TransformSettings（无 .llm），无法创建 LLM → 静默降级不抛异常
    assert out.metadata["enriched_by_llm"] is False
    assert "llm_error" not in out.metadata
    assert out.metadata["title"]


def test_custom_prompt_used(tmp_path) -> None:
    prompt = tmp_path / "custom_enrich.txt"
    prompt.write_text("CUSTOM_ENRICH_PROMPT", encoding="utf-8")
    settings = TransformSettings(enrich_with_llm=True)
    llm = MockLLM(content=make_enriched({}))
    enricher = MetadataEnricher(settings=settings, llm=llm, prompt_path=prompt)
    enricher.transform(make_chunk("原文内容。"))
    content = llm.calls[0][0]["content"]
    assert "CUSTOM_ENRICH_PROMPT" in content
    assert "原文内容。" in content


def test_llm_created_from_settings() -> None:
    """未注入 llm 时，从 settings.llm 经 LLMFactory 创建（可插拔）。"""

    class FakeLLM(BaseLLM):
        provider = "c6fake"

        def __init__(self, settings=None) -> None:
            pass

        def chat(self, messages, trace=None, **kwargs) -> ChatResponse:
            return ChatResponse(content=make_enriched({}), model="c6fake")

    LLMFactory.register("c6fake", FakeLLM)
    full = SimpleNamespace(
        refine=True,
        refine_with_llm=False,
        enrich_with_llm=True,
        prompt_path="",
        llm=LLMSettings(provider="c6fake"),
    )
    enricher = MetadataEnricher(settings=full)
    out = enricher.transform(make_chunk("原文内容。"))
    assert out.metadata["enriched_by_llm"] is True
    assert out.metadata["title"] == "LLM 生成的标题"


# ---------- 与 Chunker 的链路（C4 → C6） ----------


def test_enricher_chains_with_document_chunker(tmp_path) -> None:
    """真实 PDF 经 PDFLoader → DocumentChunker → MetadataEnricher 产出增强 Chunk。"""
    from reportlab.pdfgen import canvas

    from core.settings import SplitterSettings
    from libs.loader.pdf_loader import PDFLoader

    pdf = tmp_path / "enrich_chain.pdf"
    c = canvas.Canvas(str(pdf))
    c.setFont("Helvetica", 14)
    c.drawString(72, 800, "# RAG Overview")
    c.drawString(72, 770, "RAG combines retrieval with generation.")
    c.save()

    doc = PDFLoader().load(str(pdf))
    chunker = DocumentChunker(settings=SplitterSettings(chunk_size=200, chunk_overlap=0))
    chunks = chunker.chunk(doc)
    assert chunks

    enricher = MetadataEnricher()
    enriched = enricher.transform_many(chunks)
    assert len(enriched) == len(chunks)
    for r, c in zip(enriched, chunks):
        assert r.metadata["enriched"] is True
        assert r.metadata["title"]
        assert r.metadata["summary"]
        assert r.metadata["tags"]
        assert r.metadata["chunk_index"] == c.metadata["chunk_index"]
        assert r.source_ref == doc.id
        assert 0 <= r.start_offset <= r.end_offset <= len(doc.text)
