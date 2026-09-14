"""Markdown Loader for local text fixtures and lightweight documentation."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from core.types import Document
from libs.loader.base_loader import BaseLoader


class MarkdownLoader(BaseLoader):
    """Load a UTF-8 Markdown file into the common ``Document`` contract."""

    loader_name = "markdown"
    supported_extensions = (".md", ".markdown")

    def __init__(self, doc_id: str | None = None) -> None:
        self._doc_id = doc_id

    def load(self, path: str) -> Document:
        source_path = str(path)
        file_path = Path(source_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Markdown 文件不存在: {source_path}")

        text = file_path.read_text(encoding="utf-8-sig").strip()
        if not text:
            raise ValueError(f"Markdown 文件没有可提取的文本: {source_path}")

        return Document(
            id=self._doc_id or _content_based_doc_id(file_path),
            text=text,
            metadata={
                "source_path": source_path,
                "doc_type": "markdown",
                "loader": self.loader_name,
                "title": _first_heading(text) or file_path.stem,
                "heading_outline": _extract_heading_outline(text),
                "images": [],
            },
        )


def _content_based_doc_id(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:8]
    return f"{path.stem}_{digest}"


def _first_heading(text: str) -> str | None:
    match = re.search(r"^#{1,6}\s+(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def _extract_heading_outline(text: str) -> list[dict[str, str | int]]:
    outline: list[dict[str, str | int]] = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if match:
            outline.append({"level": len(match.group(1)), "text": match.group(2).strip()})
    return outline
