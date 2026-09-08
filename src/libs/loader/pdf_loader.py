"""PDF Loader：基于 MarkItDown 将 PDF 解析为规范化 Markdown 的 ``Document``。

MarkItDown 是默认 PDF 解析引擎（DEV_SPEC 3.1.1 首选方案）：
- ``pdfminer`` 负责文本提取，MarkItDown 规整为 Markdown 文本；
- ``result.text_content`` 即规范化 Markdown，pdfminer 会在跨页处插入换页符
  ``\\x0c``，这里将其规整为换行，保证切分时语义边界干净；
- MarkItDown 不产出 ``#`` 标题层级（那是结构化解析的职责），因此
  ``heading_outline`` 仅从 Markdown 文本中按 ``#`` 语法提取，多数 PDF 为空属正常。

职责边界：
- 只做"格式统一 + 结构抽取"，不切分、不写存储。
- ``metadata.images`` 在本 MVP 阶段为空列表；PDF 内嵌图片的提取与
  ``[IMAGE: {id}]`` 占位符注入将在 C13（ImageStorage）阶段落地，
  占位符契约已由 ``core.types`` 预先定义。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from markitdown import MarkItDown

from core.types import Document
from libs.loader.base_loader import BaseLoader


class PDFLoader(BaseLoader):
    loader_name = "markitdown"
    supported_extensions = (".pdf",)

    def __init__(self, doc_id: str | None = None, markitdown: Any | None = None) -> None:
        """初始化。

        Args:
            doc_id: 可选的文档 ID；缺省时按文件内容哈希生成稳定 ID。
            markitdown: 可注入的 MarkItDown 实例（测试可传 Fake，缺省自动创建）。
        """
        self._doc_id = doc_id
        self._markitdown = markitdown

    def load(self, path: str) -> Document:
        source_path = str(path)
        if not Path(source_path).exists():
            raise FileNotFoundError(f"PDF 文件不存在: {source_path}")

        converter = self._markitdown or MarkItDown()
        result = converter.convert(source_path)
        text = _clean_markdown(result.text_content or "")
        if not text:
            raise ValueError(f"无法从 PDF 提取任何文本: {source_path}")

        doc_id = self._doc_id or _content_based_doc_id(source_path)
        title = getattr(result, "title", None) or _first_heading(text) or Path(source_path).stem

        metadata = {
            "source_path": source_path,
            "doc_type": "pdf",
            "loader": self.loader_name,
            "title": title,
            "heading_outline": _extract_heading_outline(text),
            "images": [],
        }
        return Document(id=doc_id, text=text, metadata=metadata)


# ---------- 私有工具函数 ----------


def _clean_markdown(text: str) -> str:
    """规整 pdfminer 产出的文本：换页符 → 空行，并清除收尾空白。"""
    return re.sub(r"[ \t]*\x0c[ \t]*", "\n\n", text).strip()


def _content_based_doc_id(path: str) -> str:
    """按文件内容生成稳定文档 ID：``{stem}_{sha256 前 8 位}``。内容不变则 ID 不变。"""
    return f"{Path(path).stem}_{_sha256_hex(path)[:8]}"


def _sha256_hex(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):  # 分块读取，避免大文件占满内存
            digest.update(block)
    return digest.hexdigest()


def _first_heading(text: str) -> str | None:
    """取 Markdown 中第一个 ``#`` 标题文本（无则返回 None）。"""
    m = re.search(r"^#{1,6}\s+(.+)$", text, flags=re.MULTILINE)
    return m.group(1).strip() if m else None


def _extract_heading_outline(text: str) -> list[dict[str, Any]]:
    """从 Markdown 文本中提取标题大纲：``[{"level": 1, "text": "..."}]``。"""
    outline: list[dict[str, Any]] = []
    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if m:
            outline.append({"level": len(m.group(1)), "text": m.group(2).strip()})
    return outline
