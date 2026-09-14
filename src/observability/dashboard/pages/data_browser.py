"""Streamlit page for browsing ingested documents and their chunks."""

from __future__ import annotations

from typing import Any

import streamlit as st

from ingestion.document_manager import DocumentDetail
from observability.dashboard.services.data_service import DataService


def _document_label(document: Any) -> str:
    return f"{document.title} — {document.source_path}"


def _render_document_table(documents: list[Any]) -> None:
    st.dataframe(
        [
            {
                "文档": document.title,
                "来源路径": document.source_path,
                "Collection": document.collection,
                "Chunks": document.chunk_count,
                "图片": document.image_count,
                "摄取时间": document.processed_at or "-",
            }
            for document in documents
        ],
        hide_index=True,
        width="stretch",
    )


def _render_images(service: DataService, images: list[dict[str, Any]]) -> None:
    if not images:
        st.caption("此文档没有关联图片。")
        return
    columns = st.columns(min(3, len(images)))
    for index, image in enumerate(images):
        image_id = image["image_id"]
        with columns[index % len(columns)]:
            data = service.read_image(image_id)
            if data is None:
                st.warning(f"图片文件不可用：{image_id}")
            else:
                st.image(data, caption=f"{image_id}（第 {image.get('page_num') or '-'} 页）")


def _render_detail(service: DataService, detail: DocumentDetail) -> None:
    st.divider()
    st.subheader(f"文档详情：{detail.title}")
    st.caption(detail.source_path)
    st.markdown("##### 关联图片")
    _render_images(service, detail.images)

    st.markdown("##### Chunks")
    for index, chunk in enumerate(detail.chunks, start=1):
        with st.expander(f"Chunk {index} · {chunk.chunk_id}"):
            st.text(chunk.text)
            st.caption("Metadata")
            st.json(chunk.metadata)


def render() -> None:
    """Render the data-browser route registered in ``dashboard.app``."""
    st.title("数据浏览")
    st.caption("浏览已摄取的文档、切分后的 Chunk，以及关联图片。")

    try:
        service = DataService()
        collections = service.list_collections()
        selected_collection = st.selectbox("选择 Collection", collections)
        documents = service.list_documents(selected_collection)
    except Exception as exc:  # The page stays usable even when local storage is absent.
        st.error(f"无法读取知识库数据：{exc}")
        return

    if not documents:
        st.info("这个 Collection 中还没有已摄取的文档。请先使用 ingest.py 摄取文件。")
        return

    _render_document_table(documents)
    selected_id = st.selectbox(
        "选择一个文档以查看详情",
        [document.doc_id for document in documents],
        format_func=lambda doc_id: _document_label(
            next(document for document in documents if document.doc_id == doc_id)
        ),
    )
    detail = service.get_document_detail(selected_id)
    if detail is None:
        st.warning("该文档详情已不可用；请刷新页面后重试。")
        return
    _render_detail(service, detail)


if __name__ == "__main__":
    render()
