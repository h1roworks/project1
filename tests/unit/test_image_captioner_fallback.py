"""C7: ImageCaptioner 单元测试。

验证：
- 启用模式（caption_with_vision=True + Vision LLM 可用）：
  存在 image_refs 时逐张调用 Vision LLM 生成 caption，写回 metadata.image_captions；
  自定义 prompt 注入；经 settings.vision_llm 工厂创建；入参不被修改；
  id / source_ref / offset 作为稳定锚点保持不变；幂等
- 降级模式（不阻塞）：
  禁用 / 未注入且无配置 / 工厂创建失败 / 调用异常 / 空输出 / 部分失败 /
  image_refs 缺对应 ImageRef 时，chunk 保留 image_refs、不生成 caption 且标记
  has_unprocessed_images
- 无图片引用的 chunk 原样返回；BaseTransform 原子化契约
- 与 DocumentChunker 的链路（C4 → C7）
"""

import hashlib
from types import SimpleNamespace

from core.settings import TransformSettings, VisionLLMSettings
from core.types import Chunk, ImageRef
from ingestion.chunking.document_chunker import DocumentChunker
from ingestion.transform.image_captioner import ImageCaptioner
from libs.llm.base_llm import ChatResponse
from libs.llm.base_vision_llm import BaseVisionLLM
from libs.llm.llm_factory import LLMFactory

# ---------- fixtures ----------


def make_image_dict(image_id: str, path: str) -> dict:
    return {"id": image_id, "path": path, "page": 1, "text_offset": 0, "text_length": 0, "position": {}}


def make_chunk(text: str = "正文内容。", doc_id: str = "doc_001", index: int = 0, **metadata) -> Chunk:
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


class MockVisionLLM:
    """注入到 ImageCaptioner 的假 Vision LLM，绕开真实调用只测编排逻辑。"""

    def __init__(self, content: str = "", exc: Exception | None = None) -> None:
        self.content = content
        self.exc = exc
        self.calls: list[tuple[str, str]] = []  # (prompt, image_path)

    def chat_with_image(self, text, image_path, trace=None, **kwargs) -> ChatResponse:
        self.calls.append((text, image_path))
        if self.exc:
            raise self.exc
        return ChatResponse(content=self.content, model="mock-vision")


def enabled_captioner(vision, **settings_kwargs) -> ImageCaptioner:
    settings = TransformSettings(caption_with_vision=True, **settings_kwargs)
    return ImageCaptioner(settings=settings, vision_llm=vision)


def image_chunk() -> Chunk:
    return make_chunk(
        image_refs=["img_1"],
        images=[make_image_dict("img_1", "/data/images/img_1.png")],
    )


# ---------- 启用模式：生成 caption ----------


def test_caption_written_when_enabled() -> None:
    vision = MockVisionLLM(content="一张 RAG 架构图。")
    out = enabled_captioner(vision).transform(image_chunk())
    assert out.metadata["image_captions"] == {"img_1": "一张 RAG 架构图。"}
    assert out.metadata["captioned"] is True
    assert out.metadata["has_unprocessed_images"] is False
    assert len(vision.calls) == 1
    prompt, image_path = vision.calls[0]
    assert image_path == "/data/images/img_1.png"
    assert "image" in prompt.lower()  # 默认 prompt 来自 config/prompts/image_captioning.txt


def test_multiple_images_all_captioned() -> None:
    chunk = make_chunk(
        image_refs=["img_1", "img_2"],
        images=[make_image_dict("img_1", "/p/1.png"), make_image_dict("img_2", "/p/2.png")],
    )
    vision = MockVisionLLM(content="图表。")
    out = enabled_captioner(vision).transform(chunk)
    assert out.metadata["image_captions"] == {"img_1": "图表。", "img_2": "图表。"}
    assert out.metadata["captioned"] is True
    assert out.metadata["has_unprocessed_images"] is False
    assert [p for _, p in vision.calls] == ["/p/1.png", "/p/2.png"]


def test_image_ref_object_supported() -> None:
    """metadata.images 既可能是 dict（Loader 产出）也可能是 ImageRef 对象，都要支持。"""
    chunk = make_chunk(image_refs=["img_1"], images=[ImageRef(id="img_1", path="/p/1.png")])
    vision = MockVisionLLM(content="描述。")
    out = enabled_captioner(vision).transform(chunk)
    assert out.metadata["image_captions"] == {"img_1": "描述。"}


def test_custom_prompt_used(tmp_path) -> None:
    prompt = tmp_path / "custom_caption.txt"
    prompt.write_text("CUSTOM_CAPTION_PROMPT", encoding="utf-8")
    vision = MockVisionLLM(content="描述。")
    cap = ImageCaptioner(
        settings=TransformSettings(caption_with_vision=True),
        vision_llm=vision,
        prompt_path=prompt,
    )
    cap.transform(image_chunk())
    assert vision.calls[0][0] == "CUSTOM_CAPTION_PROMPT"


def test_vision_created_from_settings() -> None:
    """未注入 vision_llm 时，从 settings.vision_llm 经 LLMFactory 创建（可插拔）。"""

    class FakeVision(BaseVisionLLM):
        provider = "c7fake"

        def __init__(self, settings: VisionLLMSettings | None = None) -> None:
            self.settings = settings

        def chat_with_image(self, text, image_path, trace=None, **kwargs) -> ChatResponse:
            return ChatResponse(content="fake caption", model="c7fake")

    LLMFactory.register_vision("c7fake", FakeVision)
    try:
        full = SimpleNamespace(
            transform=TransformSettings(caption_with_vision=True),
            vision_llm=VisionLLMSettings(provider="c7fake", model="fake-vl"),
        )
        out = ImageCaptioner(settings=full).transform(image_chunk())
        assert out.metadata["image_captions"] == {"img_1": "fake caption"}
        assert out.metadata["captioned"] is True
    finally:
        LLMFactory._vision_registry.pop("c7fake", None)


# ---------- metadata 契约 / 锚点 ----------


def test_metadata_flags_and_input_not_mutated() -> None:
    chunk = image_chunk()
    before = dict(chunk.metadata)
    out = enabled_captioner(MockVisionLLM(content="描述。")).transform(chunk)
    assert chunk.metadata == before  # 入参未被修改
    assert "image_captions" not in chunk.metadata
    assert "has_unprocessed_images" not in chunk.metadata


def test_anchors_preserved() -> None:
    chunk = image_chunk()
    out = enabled_captioner(MockVisionLLM(content="描述。")).transform(chunk)
    assert out.id == chunk.id
    assert out.text == chunk.text
    assert out.source_ref == chunk.source_ref
    assert out.start_offset == chunk.start_offset
    assert out.end_offset == chunk.end_offset


def test_captioning_is_idempotent() -> None:
    cap = ImageCaptioner(
        settings=TransformSettings(caption_with_vision=True),
        vision_llm=MockVisionLLM(content="描述。"),
    )
    out1 = cap.transform(image_chunk())
    out2 = cap.transform(image_chunk())
    assert out1 == out2


# ---------- 降级模式：禁用 / 不可用 / 异常不阻塞 ----------


def test_disabled_marks_unprocessed_and_keeps_image_refs() -> None:
    chunk = image_chunk()
    vision = MockVisionLLM(content="不应被调用")
    out = ImageCaptioner(
        settings=TransformSettings(caption_with_vision=False), vision_llm=vision
    ).transform(chunk)
    assert out.metadata["image_refs"] == ["img_1"]  # 引用始终保留
    assert out.metadata["has_unprocessed_images"] is True
    assert out.metadata["captioned"] is False
    assert "image_captions" not in out.metadata
    assert vision.calls == []


def test_missing_vision_degrades_silently() -> None:
    out = ImageCaptioner(settings=TransformSettings(caption_with_vision=True)).transform(image_chunk())
    assert out.metadata["has_unprocessed_images"] is True
    assert "caption_error" not in out.metadata  # 静默降级，不记错误
    assert "image_captions" not in out.metadata


def test_vision_exception_degrades() -> None:
    vision = MockVisionLLM(exc=ConnectionError("vision down"))
    out = enabled_captioner(vision).transform(image_chunk())
    assert out.metadata["has_unprocessed_images"] is True
    assert "image_captions" not in out.metadata
    assert any("vision down" in e for e in out.metadata["caption_errors"])
    assert out.metadata["image_refs"] == ["img_1"]


def test_empty_caption_degrades() -> None:
    vision = MockVisionLLM(content="   \n  ")
    out = enabled_captioner(vision).transform(image_chunk())
    assert out.metadata["has_unprocessed_images"] is True
    assert "image_captions" not in out.metadata
    assert any("空 caption" in e for e in out.metadata["caption_errors"])


def test_partial_failure_marks_unprocessed() -> None:
    chunk = make_chunk(
        image_refs=["good", "bad"],
        images=[make_image_dict("good", "/p/good.png"), make_image_dict("bad", "/p/bad.png")],
    )

    class PickyVision:
        def chat_with_image(self, text, image_path, trace=None, **kwargs) -> ChatResponse:
            if image_path.endswith("bad.png"):
                raise RuntimeError("bad image")
            return ChatResponse(content="ok caption", model="mock")

    out = enabled_captioner(PickyVision()).transform(chunk)
    assert out.metadata["image_captions"] == {"good": "ok caption"}
    assert out.metadata["captioned"] is True  # 部分成功也算标注过
    assert out.metadata["has_unprocessed_images"] is True
    assert any("bad" in e for e in out.metadata["caption_errors"])


def test_missing_imageref_degrades() -> None:
    """image_refs 里的 id 在 images 中找不到对应 ImageRef → 无法定位路径，标记未处理。"""
    chunk = make_chunk(image_refs=["img_1"], images=[])
    vision = MockVisionLLM(content="描述。")
    out = enabled_captioner(vision).transform(chunk)
    assert out.metadata["has_unprocessed_images"] is True
    assert "image_captions" not in out.metadata
    assert vision.calls == []
    assert any("img_1" in e for e in out.metadata["caption_errors"])


def test_no_image_refs_unchanged() -> None:
    chunk = make_chunk(text="没有图片的普通文本。")
    out = enabled_captioner(MockVisionLLM(content="x")).transform(chunk)
    assert out == chunk  # 无图片引用：原样返回，不加任何标志


def test_transform_many_degrades_per_chunk() -> None:
    """BaseTransform 原子化契约：单 chunk 失败不拖垮整批（此处让 Captioner 抛异常）。"""

    class Boom(ImageCaptioner):
        def transform(self, chunk, trace=None) -> Chunk:
            if "bad" in chunk.text:
                raise ValueError("boom")
            return super().transform(chunk, trace)

    chunks = [
        make_chunk(text="A", image_refs=["img_1"], images=[make_image_dict("img_1", "/p/1.png")]),
        make_chunk(text="bad content"),
        make_chunk(text="B", image_refs=["img_1"], images=[make_image_dict("img_1", "/p/1.png")]),
    ]
    out = Boom(
        settings=TransformSettings(caption_with_vision=True),
        vision_llm=MockVisionLLM(content="描述。"),
    ).transform_many(chunks)
    assert out[0].metadata["image_captions"] == {"img_1": "描述。"}
    assert out[1].metadata["transform_errors"] == ["image_captioner: boom"]
    assert out[2].metadata["image_captions"] == {"img_1": "描述。"}
    assert [c.id for c in out] == [c.id for c in chunks]


# ---------- 与 Chunker 的链路（C4 → C7） ----------


def test_captioner_chains_with_document_chunker() -> None:
    """Document 经 DocumentChunker 切分后，含图片占位符的 chunk 可被 ImageCaptioner 标注。"""
    from core.settings import SplitterSettings
    from core.types import Document, make_image_placeholder

    doc = Document(
        id="chain_doc",
        text=f"这是一段文字，包含一张图片 {make_image_placeholder('img_1')}。",
        metadata={
            "source_path": "/data/x.pdf",
            "images": [make_image_dict("img_1", "/data/images/img_1.png")],
        },
    )
    chunker = DocumentChunker(settings=SplitterSettings(chunk_size=500, chunk_overlap=0))
    chunks = chunker.chunk(doc)
    assert chunks and chunks[0].metadata["image_refs"] == ["img_1"]

    out = enabled_captioner(MockVisionLLM(content="一张图片的说明。")).transform_many(chunks)
    captioned = [c for c in out if c.metadata["image_refs"]]
    assert captioned
    assert captioned[0].metadata["image_captions"] == {"img_1": "一张图片的说明。"}
    assert captioned[0].metadata["has_unprocessed_images"] is False
    assert captioned[0].source_ref == "chain_doc"
