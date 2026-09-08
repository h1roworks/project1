"""C5: Transform 基类 + ChunkRefiner 单元测试。

验证：
- BaseTransform 抽象契约（抽象方法 / transform_many 原子化 + 失败降级不阻塞）
- ChunkRefiner 规则去噪：换页符归一化、页码噪声剔除、空白规整、
  段内续行合并（遇标题/列表/表格/代码围栏止步）、幂等
- metadata 标志：refined / denoised / refined_by_llm；入参不被修改；
  id / source_ref / offset 作为稳定锚点保持不变
- 可选 LLM 二次加工：注入 Fake LLM 验证改写、失败/空输出静默回退、
  未启用时不调用、prompt 模板注入、经 settings.llm 工厂创建
- 边界：整块噪声保留原文；与 DocumentChunker 的真实链路（C4 → C5）
"""

import hashlib
from types import SimpleNamespace

import pytest

from core.settings import LLMSettings, TransformSettings
from core.types import Chunk
from ingestion.chunking.document_chunker import DocumentChunker
from ingestion.transform.base_transform import BaseTransform
from ingestion.transform.chunk_refiner import ChunkRefiner
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


class MockLLM:
    """注入到 ChunkRefiner 的假 LLM，绕开真实调用只测编排逻辑。"""

    def __init__(self, content: str = "", exc: Exception | None = None) -> None:
        self.content = content
        self.exc = exc
        self.calls: list[list[dict]] = []

    def chat(self, messages, trace=None, **kwargs) -> ChatResponse:
        self.calls.append(messages)
        if self.exc:
            raise self.exc
        return ChatResponse(content=self.content, model="mock")


class UpperTransform(BaseTransform):
    """测试用 Transform：文本转大写；含 "bad" 的 chunk 抛异常。"""

    name = "upper"

    def transform(self, chunk: Chunk, trace=None) -> Chunk:
        if "bad" in chunk.text:
            raise ValueError("boom")
        metadata = dict(chunk.metadata)
        metadata["upper"] = True
        return Chunk(
            id=chunk.id,
            text=chunk.text.upper(),
            metadata=metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref,
        )


# ---------- BaseTransform 抽象契约 ----------


def test_base_transform_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseTransform()  # type: ignore[abstract]


def test_transform_many_degrades_per_chunk() -> None:
    """单个 chunk 失败 → 原样返回 + 记录错误，不拖垮整批。"""
    chunks = [make_chunk("good one"), make_chunk("bad one"), make_chunk("good two")]
    out = UpperTransform().transform_many(chunks)
    assert out[0].text == "GOOD ONE"
    assert out[0].metadata["upper"] is True
    assert out[1].text == "bad one"  # 原样保留
    assert out[1].metadata["transform_errors"] == ["upper: boom"]
    assert "upper" not in out[1].metadata
    assert out[2].text == "GOOD TWO"
    # 保序，且入参未被修改
    assert [c.id for c in out] == [c.id for c in chunks]
    assert chunks[1].text == "bad one"
    assert "transform_errors" not in chunks[1].metadata


def test_transform_many_accepts_empty() -> None:
    assert UpperTransform().transform_many([]) == []


# ---------- ChunkRefiner：规则去噪 ----------


def test_form_feed_normalized() -> None:
    out = ChunkRefiner().transform(make_chunk("page one\x0cpage two"))
    assert "\x0c" not in out.text
    assert "page one" in out.text and "page two" in out.text


def test_page_number_noise_removed() -> None:
    text = (
        "正文内容第一段。\n\n"
        "- 3 -\n\n第 4 页\n\nPage 2 of 10\n\n3 / 10\n\n正文内容第二段。"
    )
    out = ChunkRefiner().transform(make_chunk(text))
    for noise in ("- 3 -", "第 4 页", "Page 2 of 10", "3 / 10"):
        assert noise not in out.text
    assert "正文内容第一段。" in out.text
    assert "正文内容第二段。" in out.text


def test_whitespace_normalized() -> None:
    out = ChunkRefiner().transform(make_chunk("  第一行  \n\n\n\n  第二行  "))
    assert not out.text.startswith(" ")
    assert not out.text.endswith(" ")
    assert "\n\n\n" not in out.text
    assert "第一行" in out.text and "第二行" in out.text


def test_broken_lines_merged_into_paragraph() -> None:
    text = "RAG combines retrieval with\ngeneration.\n\n第二段开始"
    out = ChunkRefiner().transform(make_chunk(text))
    assert "RAG combines retrieval with generation." in out.text


def test_no_merge_across_structural_lines() -> None:
    text = (
        "无标点行\n# 新标题\n\n无标点行二\n- 列表项\n\n"
        "无标点行三\n| 表格行 |\n\n`code` 行"
    )
    out = ChunkRefiner().transform(make_chunk(text))
    assert "# 新标题" in out.text
    assert "- 列表项" in out.text
    assert "| 表格行 |" in out.text
    assert "无标点行三| 表格行 |" not in out.text  # 结构行止步，未错误合并


def test_code_fence_lines_untouched() -> None:
    text = "前面说明\n```python\nprint('a')\nprint('b')\n```\n后面说明。"
    out = ChunkRefiner().transform(make_chunk(text))
    assert "print('a')\nprint('b')" in out.text  # 代码行保持原样


def test_image_placeholder_survives_refinement() -> None:
    text = "[IMAGE: img_a_0] 柱状图显示\n销售趋势。\n\n后续段落。"
    out = ChunkRefiner().transform(make_chunk(text, image_refs=["img_a_0"]))
    assert "[IMAGE: img_a_0]" in out.text
    assert out.metadata["image_refs"] == ["img_a_0"]


def test_denoised_flag() -> None:
    dirty = make_chunk("正文。\n\n- 3 -\n\n结尾。")
    out = ChunkRefiner().transform(dirty)
    assert out.metadata["refined"] is True
    assert out.metadata["denoised"] is True
    assert out.metadata["refined_by_llm"] is False

    clean = make_chunk("正文内容完整。")
    out2 = ChunkRefiner().transform(clean)
    assert out2.metadata["denoised"] is False
    assert out2.text == "正文内容完整。"


def test_all_noise_chunk_keeps_original() -> None:
    chunk = make_chunk("- 3 -\n\n第 4 页")
    out = ChunkRefiner().transform(chunk)
    assert out.text == chunk.text  # 整块噪声不清空，保留原文
    assert out.metadata["denoised"] is False


def test_transform_idempotent_on_same_input() -> None:
    text = "第一段无标点续行\n第二段结尾。\n\n- 3 -\n\n第三段内容。"
    refiner = ChunkRefiner()
    out1 = refiner.transform(make_chunk(text))
    out2 = refiner.transform(make_chunk(text))
    assert out1 == out2  # 同一输入两次 → 完全一致（含 metadata）


def test_transform_text_is_fixed_point() -> None:
    text = "第一段无标点续行\n第二段结尾。\n\n- 3 -\n\n第三段内容。"
    refiner = ChunkRefiner()
    out1 = refiner.transform(make_chunk(text))
    out2 = refiner.transform(out1)
    assert out2.text == out1.text  # 二次加工文本不再变化


def test_anchors_preserved_and_input_not_mutated() -> None:
    chunk = make_chunk("正文内容。\n\n- 3 -\n\n结尾内容。")
    before = dict(chunk.metadata)
    out = ChunkRefiner().transform(chunk)
    assert out.id == chunk.id
    assert out.source_ref == chunk.source_ref
    assert out.start_offset == chunk.start_offset
    assert out.end_offset == chunk.end_offset
    assert chunk.metadata == before  # 入参未被修改
    assert "refined" not in chunk.metadata


# ---------- ChunkRefiner：可选 LLM 二次加工 ----------


def test_llm_refine_rewrites_text_when_enabled() -> None:
    settings = TransformSettings(refine_with_llm=True)
    llm = MockLLM(content="精炼后的干净文本。")
    refiner = ChunkRefiner(settings=settings, llm=llm)
    out = refiner.transform(make_chunk("原始噪声文本。"))
    assert out.text == "精炼后的干净文本。"
    assert out.metadata["refined_by_llm"] is True
    assert out.metadata["refined"] is True
    assert len(llm.calls) == 1


def test_llm_failure_degrades_to_rule_result() -> None:
    settings = TransformSettings(refine_with_llm=True)
    llm = MockLLM(exc=ConnectionError("llm down"))
    refiner = ChunkRefiner(settings=settings, llm=llm)
    out = refiner.transform(make_chunk("噪声开头。\n\n- 3 -\n\n正文内容。"))
    assert "正文内容。" in out.text  # 规则结果保留
    assert "- 3 -" not in out.text
    assert out.metadata["refined_by_llm"] is False
    assert "llm down" in out.metadata["llm_error"]


def test_llm_empty_output_degrades() -> None:
    settings = TransformSettings(refine_with_llm=True)
    refiner = ChunkRefiner(settings=settings, llm=MockLLM(content="   \n  "))
    out = refiner.transform(make_chunk("正文内容。"))
    assert out.text == "正文内容。"  # 空输出 → 保留规则结果
    assert out.metadata["refined_by_llm"] is False
    assert "空文本" in out.metadata["llm_error"]


def test_llm_not_called_when_disabled() -> None:
    llm = MockLLM(content="不应被使用")
    refiner = ChunkRefiner(settings=TransformSettings(refine_with_llm=False), llm=llm)
    out = refiner.transform(make_chunk("正文内容。"))
    assert llm.calls == []
    assert out.metadata["refined_by_llm"] is False


def test_missing_llm_degrades_silently() -> None:
    refiner = ChunkRefiner(settings=TransformSettings(refine_with_llm=True))
    out = refiner.transform(make_chunk("正文内容。"))
    # settings 是纯 TransformSettings（无 .llm），无法创建 LLM → 静默降级不抛异常
    assert out.text == "正文内容。"
    assert out.metadata["refined_by_llm"] is False
    assert "llm_error" not in out.metadata


def test_custom_prompt_used(tmp_path) -> None:
    prompt = tmp_path / "custom_refine.txt"
    prompt.write_text("CUSTOM_REFINE_PROMPT", encoding="utf-8")
    settings = TransformSettings(refine_with_llm=True)
    llm = MockLLM(content="结果。")
    refiner = ChunkRefiner(settings=settings, llm=llm, prompt_path=prompt)
    out = refiner.transform(make_chunk("原文内容。"))
    assert out.text == "结果。"
    content = llm.calls[0][0]["content"]
    assert "CUSTOM_REFINE_PROMPT" in content
    assert "原文内容。" in content


def test_llm_created_from_settings() -> None:
    """未注入 llm 时，从 settings.llm 经 LLMFactory 创建（可插拔）。"""

    class FakeLLM(BaseLLM):
        provider = "c5fake"

        def __init__(self, settings=None) -> None:
            pass

        def chat(self, messages, trace=None, **kwargs) -> ChatResponse:
            return ChatResponse(content="工厂创建的结果。", model="c5fake")

    LLMFactory.register("c5fake", FakeLLM)
    full = SimpleNamespace(
        refine=True,
        refine_with_llm=True,
        prompt_path="",
        llm=LLMSettings(provider="c5fake"),
    )
    refiner = ChunkRefiner(settings=full)
    out = refiner.transform(make_chunk("原文内容。"))
    assert out.text == "工厂创建的结果。"
    assert out.metadata["refined_by_llm"] is True


# ---------- 与 Chunker 的链路（C4 → C5） ----------


def test_refiner_chains_with_document_chunker(tmp_path) -> None:
    """真实 PDF 经 PDFLoader → DocumentChunker → ChunkRefiner 产出增强 Chunk。"""
    from reportlab.pdfgen import canvas

    from core.settings import SplitterSettings
    from libs.loader.pdf_loader import PDFLoader

    pdf = tmp_path / "refine_chain.pdf"
    c = canvas.Canvas(str(pdf))
    c.setFont("Helvetica", 14)
    c.drawString(72, 800, "# RAG Overview")
    c.drawString(72, 770, "RAG combines retrieval with")
    c.drawString(72, 740, "generation.")
    c.save()

    doc = PDFLoader().load(str(pdf))
    chunker = DocumentChunker(settings=SplitterSettings(chunk_size=200, chunk_overlap=0))
    chunks = chunker.chunk(doc)
    assert chunks

    refiner = ChunkRefiner()
    refined = refiner.transform_many(chunks)
    assert len(refined) == len(chunks)
    for r, c in zip(refined, chunks):
        assert r.metadata["refined"] is True
        assert r.metadata["chunk_index"] == c.metadata["chunk_index"]
        assert r.source_ref == doc.id
        assert 0 <= r.start_offset <= r.end_offset <= len(doc.text)
