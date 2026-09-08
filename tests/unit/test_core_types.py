"""C1: 核心数据类型/契约测试。

验证 Document/Chunk/ChunkRecord/ImageRef 的字段稳定、dict/json 可序列化，
以及 metadata.images 与 [IMAGE: {id}] 占位符契约。
"""

import json

import pytest

from core.types import (
    Chunk,
    ChunkRecord,
    Document,
    ImageRef,
    IMAGE_PLACEHOLDER_PATTERN,
    extract_image_ids,
    make_image_placeholder,
)

# ---------- 基础 fixture ----------


def make_document() -> Document:
    return Document(
        id="doc-001",
        text="# 标题\n正文内容\n[IMAGE: abc_0_0]\n更多正文",
        metadata={
            "source_path": "/data/documents/sample.pdf",
            "doc_type": "pdf",
            "title": "示例文档",
            "page": 1,
        },
    )


def make_chunk() -> Chunk:
    return Chunk(
        id="doc-001_0000_1a2b3c4d",
        text="正文内容",
        metadata={
            "source_path": "/data/documents/sample.pdf",
            "doc_type": "pdf",
            "chunk_index": 0,
            "image_refs": ["abc_0_0"],
        },
        start_offset=10,
        end_offset=22,
        source_ref="doc-001",
    )


def make_image_ref() -> ImageRef:
    return ImageRef(
        id="abc_0_0",
        path="data/images/default/abc_0_0.png",
        page=1,
        text_offset=14,
        text_length=len(make_image_placeholder("abc_0_0")),
        position={"x": 10.0, "y": 20.0, "width": 300, "height": 200},
    )


# ---------- 字段稳定性 ----------

def test_document_field_stability() -> None:
    assert set(make_document().to_dict()) == {"id", "text", "metadata"}


def test_chunk_field_stability() -> None:
    assert set(make_chunk().to_dict()) == {
        "id", "text", "metadata", "start_offset", "end_offset", "source_ref",
    }


def test_chunk_record_field_stability() -> None:
    rec = ChunkRecord(id="c1", text="t", metadata={})
    assert set(rec.to_dict()) == {
        "id", "text", "metadata", "dense_vector", "sparse_vector",
    }


def test_image_ref_field_stability() -> None:
    assert set(make_image_ref().to_dict()) == {
        "id", "path", "page", "text_offset", "text_length", "position",
    }


# ---------- Document ----------

def test_document_roundtrip() -> None:
    doc = make_document()
    restored = Document.from_dict(doc.to_dict())
    assert restored == doc
    assert isinstance(restored, Document)


def test_document_metadata_preserved() -> None:
    doc = make_document()
    assert doc.to_dict()["metadata"]["source_path"] == "/data/documents/sample.pdf"
    assert "doc_type" in doc.metadata


def test_document_requires_source_path_convention() -> None:
    """metadata 约定最少包含 source_path，缺省时字段仍在契约中（docstring 约定）。"""
    doc = Document(id="d", text="t", metadata={"source_path": "a.pdf"})
    assert doc.metadata["source_path"] == "a.pdf"


def test_document_json_serializable() -> None:
    payload = json.dumps(make_document().to_dict(), ensure_ascii=False)
    assert isinstance(payload, str)
    assert json.loads(payload)["id"] == "doc-001"


# ---------- Chunk ----------

def test_chunk_roundtrip() -> None:
    chunk = make_chunk()
    restored = Chunk.from_dict(chunk.to_dict())
    assert restored == chunk
    assert isinstance(restored, Chunk)


def test_chunk_defaults() -> None:
    chunk = Chunk(id="c", text="t", metadata={"source_path": "a.pdf"})
    assert chunk.start_offset == 0
    assert chunk.end_offset == 0
    assert chunk.source_ref == ""
    restored = Chunk.from_dict(chunk.to_dict())
    assert restored == chunk


def test_chunk_json_serializable() -> None:
    assert json.loads(json.dumps(make_chunk().to_dict()))["id"].startswith("doc-001")


# ---------- ChunkRecord ----------

def test_chunk_record_roundtrip_with_vectors() -> None:
    rec = ChunkRecord(
        id="doc-001_0000_1a2b3c4d",
        text="正文内容",
        metadata={"source_path": "a.pdf", "image_refs": []},
        dense_vector=[0.1, 0.2, 0.3],
        sparse_vector={"正文": 1.5, "内容": 0.8},
    )
    restored = ChunkRecord.from_dict(rec.to_dict())
    assert restored == rec
    assert restored.dense_vector == [0.1, 0.2, 0.3]
    assert restored.sparse_vector == {"正文": 1.5, "内容": 0.8}


def test_chunk_record_vectors_optional() -> None:
    rec = ChunkRecord(id="c", text="t", metadata={})
    assert rec.dense_vector is None
    assert rec.sparse_vector is None
    restored = ChunkRecord.from_dict(rec.to_dict())
    assert restored.dense_vector is None
    assert restored.sparse_vector is None


def test_chunk_record_json_serializable() -> None:
    rec = ChunkRecord(
        id="c", text="t", metadata={},
        dense_vector=[0.5], sparse_vector={"t": 1.0},
    )
    assert json.loads(json.dumps(rec.to_dict()))["dense_vector"] == [0.5]


# ---------- metadata.images 契约 ----------

def test_image_ref_roundtrip() -> None:
    ref = make_image_ref()
    restored = ImageRef.from_dict(ref.to_dict())
    assert restored == ref


def test_metadata_images_structure_matches_spec() -> None:
    """metadata.images 条目结构需符合 C1 规范字段。"""
    d = make_image_ref().to_dict()
    assert set(d) == {"id", "path", "page", "text_offset", "text_length", "position"}
    assert d["text_length"] == len("[IMAGE: abc_0_0]")


def test_metadata_images_embedded_in_chunk() -> None:
    chunk = Chunk(
        id="c",
        text="正文 [IMAGE: abc_0_0] 结束",
        metadata={
            "source_path": "a.pdf",
            "images": [make_image_ref().to_dict()],
        },
    )
    restored = Chunk.from_dict(chunk.to_dict())
    assert restored == chunk
    assert restored.metadata["images"][0]["id"] == "abc_0_0"


def test_metadata_images_empty_ok() -> None:
    chunk = Chunk(id="c", text="t", metadata={"source_path": "a.pdf", "images": []})
    restored = Chunk.from_dict(chunk.to_dict())
    assert restored == chunk
    assert restored.metadata["images"] == []


def test_image_ref_json_serializable() -> None:
    assert json.loads(json.dumps(make_image_ref().to_dict()))["id"] == "abc_0_0"


# ---------- 图片占位符契约 ----------

def test_make_image_placeholder_format() -> None:
    assert make_image_placeholder("abc_0_0") == "[IMAGE: abc_0_0]"


def test_placeholder_pattern_matches() -> None:
    text = "前文 [IMAGE: a_1_0] 中段 [IMAGE: a_1_2] 后文"
    ids = IMAGE_PLACEHOLDER_PATTERN.findall(text)
    assert ids == ["a_1_0", "a_1_2"]


def test_extract_image_ids_preserves_order() -> None:
    text = "前 [IMAGE: b_0_1] 中 [IMAGE: b_0_0] 后"
    assert extract_image_ids(text) == ["b_0_1", "b_0_0"]


def test_extract_image_ids_empty() -> None:
    assert extract_image_ids("没有任何占位符") == []


@pytest.mark.parametrize("image_id", ["abc_1_0", "doc-hash_2_3", "img42"])
def test_placeholder_roundtrip(image_id: str) -> None:
    placeholder = make_image_placeholder(image_id)
    assert extract_image_ids(placeholder) == [image_id]
