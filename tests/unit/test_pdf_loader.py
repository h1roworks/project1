"""C3: Loader 抽象基类与 PDF Loader 单元测试。

验证：
- BaseLoader 抽象基类契约（抽象方法 / can_handle 扩展名判定）
- PDFLoader 用 MarkItDown 将 PDF 转为规范化 Markdown Document（真实转换）
- metadata 契约（source_path / doc_type / title / heading_outline / images）
- Document.id 稳定性（同内容同 id、内容变更 id 变更、可显式指定）
- 边界处理（文件不存在 / 空文本 / 注入 Fake converter 纯逻辑测试）
"""

import shutil
from types import SimpleNamespace

import pytest
from reportlab.pdfgen import canvas

from core.types import Document
from libs.loader.base_loader import BaseLoader
from libs.loader.pdf_loader import PDFLoader

# ---------- fixture ----------


def make_pdf(path, pages):
    """用 reportlab 生成多页 PDF。``pages`` 为每页文本行列表（list[list[str]]）。"""
    c = canvas.Canvas(str(path))
    for lines in pages:
        c.setFont("Helvetica", 14)
        y = 800
        for line in lines:
            c.drawString(72, y, line)
            y -= 30
        c.showPage()
    c.save()


class FakeConverter:
    """注入到 PDFLoader 的假 MarkItDown，绕开真实解析、只测纯逻辑。"""

    def __init__(self, text_content="", title=None):
        self._text = text_content
        self._title = title

    def convert(self, path):
        return SimpleNamespace(text_content=self._text, title=self._title)


# ---------- BaseLoader 抽象契约 ----------

def test_base_loader_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseLoader()  # type: ignore[abstract]


def test_pdf_loader_is_a_base_loader() -> None:
    assert isinstance(PDFLoader(), BaseLoader)


def test_supported_extensions_declared() -> None:
    assert PDFLoader.supported_extensions == (".pdf",)


def test_can_handle_by_extension() -> None:
    assert PDFLoader.can_handle("doc.pdf") is True
    assert PDFLoader.can_handle("DOC.PDF") is True  # 大小写不敏感
    assert PDFLoader.can_handle("doc.md") is False
    assert PDFLoader.can_handle("doc.pdf.txt") is False  # 按真实后缀判定


def test_can_handle_is_inherited() -> None:
    assert BaseLoader.can_handle("any.pdf") is False  # 基类默认不支持任何扩展名


# ---------- 真实 PDF 转换（MarkItDown） ----------

def test_load_returns_document(tmp_path) -> None:
    pdf = tmp_path / "intro.pdf"
    make_pdf(pdf, [["# Introduction to RAG", "RAG is a framework."], ["# Hybrid Search", "BM25 matches keywords."]])
    doc = PDFLoader().load(str(pdf))
    assert isinstance(doc, Document)
    assert "RAG is a framework." in doc.text
    assert "BM25 matches keywords." in doc.text


def test_load_metadata_contract(tmp_path) -> None:
    pdf = tmp_path / "intro.pdf"
    make_pdf(pdf, [["# Introduction to RAG", "RAG is a framework."]])
    doc = PDFLoader().load(str(pdf))
    md = doc.metadata
    assert md["source_path"] == str(pdf)
    assert md["doc_type"] == "pdf"
    assert md["loader"] == "markitdown"
    assert md["title"] == "Introduction to RAG"
    assert isinstance(md["heading_outline"], list)
    assert md["images"] == []


def test_load_doc_id_stable_for_same_content(tmp_path) -> None:
    """同内容（字节一致）→ 同 id，即使路径不同。"""
    src = tmp_path / "same.pdf"
    make_pdf(src, [["Stable content"]])
    copy_dir = tmp_path / "copy"
    copy_dir.mkdir()
    dst = copy_dir / "same.pdf"
    shutil.copyfile(src, dst)
    assert PDFLoader().load(str(src)).id == PDFLoader().load(str(dst)).id


def test_load_doc_id_changes_when_content_changes(tmp_path) -> None:
    pdf = tmp_path / "doc.pdf"
    make_pdf(pdf, [["version one"]])
    id1 = PDFLoader().load(str(pdf)).id
    make_pdf(pdf, [["version two"]])  # 覆盖为不同内容
    id2 = PDFLoader().load(str(pdf)).id
    assert id1 != id2


def test_load_doc_id_respects_explicit_override(tmp_path) -> None:
    pdf = tmp_path / "doc.pdf"
    make_pdf(pdf, [["anything"]])
    assert PDFLoader(doc_id="my-doc").load(str(pdf)).id == "my-doc"


def test_load_title_falls_back_to_stem(tmp_path) -> None:
    """PDF 内没有 # 标题时，title 降级为文件名（去扩展名）。"""
    pdf = tmp_path / "no_heading.pdf"
    make_pdf(pdf, [["plain text without heading"]])
    doc = PDFLoader().load(str(pdf))
    assert doc.metadata["title"] == "no_heading"


# ---------- 边界处理 ----------

def test_load_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        PDFLoader().load(str(tmp_path / "not_exist.pdf"))


def test_load_empty_text_raises(tmp_path) -> None:
    pdf = tmp_path / "empty.pdf"
    pdf.write_bytes(b"%PDF-1.4 placeholder")  # 存在但无有效文本
    loader = PDFLoader(markitdown=FakeConverter(text_content=""))
    with pytest.raises(ValueError):
        loader.load(str(pdf))


# ---------- 注入 Fake converter：纯逻辑 ----------

def test_injected_converter_skips_real_parsing(tmp_path) -> None:
    """注入 Fake 后不再触碰真实 PDF，绕开对 pdfminer 的依赖。"""
    pdf = tmp_path / "fake.pdf"
    pdf.write_text("not a real pdf", encoding="utf-8")
    fake = FakeConverter(text_content="hello", title="hello")
    doc = PDFLoader(markitdown=fake).load(str(pdf))
    assert doc.text == "hello"
    assert doc.metadata["title"] == "hello"


def test_title_heading_outline_from_markdown(tmp_path) -> None:
    pdf = tmp_path / "markdown.pdf"
    pdf.write_text("x", encoding="utf-8")
    fake = FakeConverter(text_content="# Title\n\n## Sub\nbody text")
    doc = PDFLoader(markitdown=fake).load(str(pdf))
    assert doc.metadata["title"] == "Title"
    assert doc.metadata["heading_outline"] == [
        {"level": 1, "text": "Title"},
        {"level": 2, "text": "Sub"},
    ]


def test_form_feed_normalized_to_blank_line(tmp_path) -> None:
    """pdfminer 的换页符 \\x0c 应规整为空行，避免污染切分。"""
    pdf = tmp_path / "pages.pdf"
    pdf.write_text("x", encoding="utf-8")
    fake = FakeConverter(text_content="page one\x0cpage two")
    doc = PDFLoader(markitdown=fake).load(str(pdf))
    assert doc.text == "page one\n\npage two"


def test_leading_trailing_whitespace_stripped(tmp_path) -> None:
    pdf = tmp_path / "ws.pdf"
    pdf.write_text("x", encoding="utf-8")
    fake = FakeConverter(text_content="  \n\n  content  \n\n  ")
    doc = PDFLoader(markitdown=fake).load(str(pdf))
    assert doc.text == "content"
