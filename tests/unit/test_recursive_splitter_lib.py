"""B7.5: Recursive Splitter（封装 LangChain）测试。

覆盖工厂创建、chunk 尺寸约束、内容保留（whitespace 归位不影响正文）、空文本，
以及 Markdown 结构（标题 / 代码块）不被切断。
"""

import re

import pytest

from core.settings import SplitterSettings
from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.recursive_splitter import RecursiveSplitter
from libs.splitter.splitter_factory import SplitterFactory


def make_settings(**overrides) -> SplitterSettings:
    base = {"strategy": "recursive", "chunk_size": 50, "chunk_overlap": 0, "separators": []}
    base.update(overrides)
    return SplitterSettings(**base)


def test_factory_creates_recursive() -> None:
    s = SplitterFactory.create(make_settings())
    assert isinstance(s, RecursiveSplitter)
    assert isinstance(s, BaseSplitter)


def test_chunks_respect_chunk_size() -> None:
    s = SplitterFactory.create(make_settings(chunk_size=20, chunk_overlap=0))
    chunks = s.split_text("a" * 100)
    assert chunks
    assert all(len(c) <= 20 for c in chunks)


def test_split_preserves_content_without_overlap() -> None:
    text = "第一段内容。\n\n第二段内容。\n\n第三段内容。\n\n" * 10
    s = SplitterFactory.create(make_settings(chunk_size=40, chunk_overlap=0))
    chunks = s.split_text(text)
    # LangChain 切分会把空白字符重新归位，但正文内容一个字符都不能丢
    def strip_ws(t: str) -> str:
        return re.sub(r"\s", "", t)

    assert strip_ws("".join(chunks)) == strip_ws(text)


def test_empty_text_returns_empty() -> None:
    s = SplitterFactory.create(make_settings())
    assert s.split_text("") == []


def test_markdown_heading_not_split() -> None:
    heading = "# 项目介绍"
    body = "这是正文内容。" * 30
    s = SplitterFactory.create(make_settings(chunk_size=50, chunk_overlap=0))
    chunks = s.split_text(heading + "\n\n" + body)
    # 标题保持在某个 chunk 的开头，没有被拦腰截断
    assert any(c.startswith("# 项目介绍") for c in chunks)


def test_code_block_kept_intact() -> None:
    code = '```python\nprint("hello world")\nprint("second line")\n```'
    intro = "前面的介绍文字。" * 20  # 足够长，保证文本确实被切分
    s = SplitterFactory.create(make_settings(chunk_size=60, chunk_overlap=0))
    chunks = s.split_text(intro + "\n\n" + code)
    assert len(chunks) > 1
    # 完整代码块落在同一个 chunk 内，未被中间切断
    assert any(code in c for c in chunks)


def test_custom_separators_override() -> None:
    text = "x" * 20 + "\n" + "y" * 20 + "\n" + "z" * 20
    s = SplitterFactory.create(
        make_settings(chunk_size=25, chunk_overlap=0, separators=["\n"])
    )
    chunks = s.split_text(text)
    # 自定义分隔符生效：结果按行切分，且无正文丢失
    assert "".join(chunks).replace("\n", "") == text.replace("\n", "")
    assert len(chunks) >= 3  # 25 字符装不下 41 字符的整行，必然切分成多块
