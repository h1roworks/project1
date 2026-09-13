"""核心数据类型/契约（全链路复用）。

本模块是 ingestion → retrieval → mcp tools 共用的数据契约中心：
- ``Document``：Loader 产出的统一文档对象（text + metadata）
- ``Chunk``：切分后的语义单元，携带稳定的定位与溯源信息
- ``ChunkRecord``：存储/检索载体（Chunk + 双路向量），字段按 C8~C12 演进
- ``ProcessedQuery``：查询预处理产物（关键词 + 稀疏词项 + filters），D1 引入
- ``ImageRef``：``metadata.images`` 条目的结构化描述，支持多模态索引

**元数据约定**：
- ``metadata`` 最少包含 ``source_path``，其余字段允许增量扩展但不得破坏兼容。
- ``metadata.images`` 为图片引用列表，每项符合 ``ImageRef`` 结构：
  ``{"id", "path", "page", "text_offset", "text_length", "position"}``。

**图片占位符约定**：在 ``Document.text`` / ``Chunk.text`` 中，图片位置使用
``[IMAGE: {image_id}]`` 标记（见 ``make_image_placeholder``），通过
``text_offset``/``text_length`` 可精确定位其在原文中的位置。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\[IMAGE: ([^\]]+)\]")


def make_image_placeholder(image_id: str) -> str:
    """生成文本中的图片占位符：``[IMAGE: {image_id}]``。"""
    return f"[IMAGE: {image_id}]"


def extract_image_ids(text: str) -> list[str]:
    """按出现顺序提取文本中所有 ``[IMAGE: {id}]`` 占位符的 image_id。"""
    return IMAGE_PLACEHOLDER_PATTERN.findall(text)


@dataclass
class ImageRef:
    """``metadata.images`` 的一个条目：图片引用信息。

    Args:
        id: 全局唯一图片标识（建议格式 ``{doc_hash}_{page}_{seq}``）。
        path: 图片文件存储路径（约定 ``data/images/{collection}/{image_id}.png``）。
        page: 图片在原文档中的页码（可选，适用于 PDF 等分页文档）。
        text_offset: 占位符在文本中的起始字符位置（从 0 开始计数）。
        text_length: 占位符字符长度（通常为 ``len("[IMAGE: {image_id}]")``）。
        position: 图片在原文档中的物理位置信息（可选，如 PDF 坐标/尺寸）。
    """

    id: str
    path: str
    page: int = 0
    text_offset: int = 0
    text_length: int = 0
    position: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "page": self.page,
            "text_offset": self.text_offset,
            "text_length": self.text_length,
            "position": self.position,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageRef":
        return cls(
            id=data["id"],
            path=data["path"],
            page=data.get("page", 0),
            text_offset=data.get("text_offset", 0),
            text_length=data.get("text_length", 0),
            position=data.get("position", {}),
        )


@dataclass
class Document:
    """Loader 产出的统一文档对象。

    ``text`` 为规范化 Markdown 文本（内含 ``[IMAGE: {id}]`` 占位符），
    ``metadata`` 至少包含 ``source_path``，并可携带 ``doc_type``、
    ``title/heading_outline``、``page``、``images`` 等定位与引用信息。
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        return cls(
            id=data["id"],
            text=data["text"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class Chunk:
    """切分后的语义单元。

    ``id`` 需在整个文档中唯一且确定（如 ``{doc_id}_{index:04d}_{hash}``），
    ``start_offset``/``end_offset`` 记录其在 ``Document.text`` 中的字符区间，
    ``source_ref`` 指向父 ``Document.id`` 支持溯源。
    ``metadata`` 继承自 Document 并追加 ``chunk_index``、``image_refs`` 等字段。
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    start_offset: int = 0
    end_offset: int = 0
    source_ref: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "source_ref": self.source_ref,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(
            id=data["id"],
            text=data["text"],
            metadata=data.get("metadata", {}),
            start_offset=data.get("start_offset", 0),
            end_offset=data.get("end_offset", 0),
            source_ref=data.get("source_ref", ""),
        )


@dataclass
class ChunkRecord:
    """存储/检索载体：Chunk 原文 + metadata + 双路向量。

    ``dense_vector`` 为语义向量（list[float]），``sparse_vector`` 为
    关键词权重映射（``{term: weight}``，由 SparseEncoder/BM25 产生）。
    两者均为可选，字段按 C8~C12 逐步填实，保持向后兼容。
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    dense_vector: list[float] | None = None
    sparse_vector: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "dense_vector": self.dense_vector,
            "sparse_vector": self.sparse_vector,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkRecord":
        return cls(
            id=data["id"],
            text=data["text"],
            metadata=data.get("metadata", {}),
            dense_vector=data.get("dense_vector"),
            sparse_vector=data.get("sparse_vector"),
        )


@dataclass
class ProcessedQuery:
    """查询预处理产物（D1：QueryProcessor 输出）。

    ``keywords`` 为原始关键词（去停用词、去重、保序）；``sparse_terms`` 为
    稀疏检索词项及其权重（原始关键词 1.0，扩展同义词 0.8）；``dense_query``
    为稠密检索输入文本（默认取剥离过滤约束后的查询）；``filters`` 为解析出的
    结构化元数据过滤条件；``method`` 记录处理方式（当前 ``"rule"``，预留
    LLM 增强）。
    """

    original_query: str
    keywords: list[str] = field(default_factory=list)
    sparse_terms: dict[str, float] = field(default_factory=dict)
    dense_query: str = ""
    filters: dict[str, Any] = field(default_factory=dict)
    method: str = "rule"

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "keywords": self.keywords,
            "sparse_terms": self.sparse_terms,
            "dense_query": self.dense_query,
            "filters": self.filters,
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessedQuery":
        return cls(
            original_query=data["original_query"],
            keywords=data.get("keywords", []),
            sparse_terms=data.get("sparse_terms", {}),
            dense_query=data.get("dense_query", ""),
            filters=data.get("filters", {}),
            method=data.get("method", "rule"),
        )


@dataclass
class RetrievalResult:
    """统一的检索结果，供 Dense/Sparse/Hybrid 检索层共同使用。

    ``chunk_id`` 是命中文本块的稳定标识；``score`` 是当前检索器给出的
    相关性分数（分数的具体计算方式由检索器决定）；``text`` 和 ``metadata``
    让调用方无需再访问向量库就能展示结果和来源。
    """

    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的普通字典。"""
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "text": self.text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievalResult":
        """从 :meth:`to_dict` 产生的字典还原对象。"""
        return cls(
            chunk_id=data["chunk_id"],
            score=float(data["score"]),
            text=data["text"],
            metadata=data.get("metadata", {}),
        )
