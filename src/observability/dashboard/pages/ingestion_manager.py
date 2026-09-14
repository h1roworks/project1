"""Streamlit page for uploading, ingesting, and deleting documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from observability.dashboard.services.ingestion_service import IngestionService

_STAGES = ("load", "split", "transform", "embed", "upsert")


def _progress_percent(stage: str, current: int, total: int) -> int:
    """Map Pipeline's stage callback into one 0-100 Streamlit progress value."""
    stage_index = _STAGES.index(stage) if stage in _STAGES else 0
    fraction = current / total if total > 0 else 0.0
    return min(100, max(0, int((stage_index + fraction) / len(_STAGES) * 100)))


def _run_ingestion(service: IngestionService, path: Path, collection: str) -> None:
    progress_bar = st.progress(0, text="准备开始摄取…")

    def on_progress(stage: str, current: int, total: int) -> None:
        progress_bar.progress(
            _progress_percent(stage, current, total),
            text=f"正在执行 {stage}：{current}/{total}",
        )

    with st.spinner("正在处理文档，请稍候…"):
        result = service.ingest_path(path, collection, on_progress=on_progress)
    progress_bar.progress(100, text="摄取流程结束")

    if result.error:
        st.error(f"摄取失败：{result.error}")
    elif result.skipped:
        st.warning("该文件内容已摄取过，因此被增量机制跳过。")
    else:
        st.success(f"摄取成功：{result.total_chunks} 个 Chunk、{result.total_images} 张图片")
        st.json(result.to_dict())


def _render_ingestion_form(service: IngestionService) -> None:
    st.subheader("摄取新文档")
    collection = st.text_input(
        "目标 Collection",
        value=service.default_collection(),
        help="新 collection 可直接输入名称；当前 MVP 的 Chroma 仍使用配置中的物理集合。",
    ).strip()
    if not collection:
        st.warning("请输入目标 Collection 名称。")
        return

    uploaded = st.file_uploader("上传 PDF 文件", type=["pdf"])
    local_path = st.text_input("或填写本机 PDF 路径")
    if st.button("开始摄取", type="primary"):
        if uploaded is not None:
            path = service.save_upload(uploaded.name, uploaded.getvalue())
        elif local_path.strip():
            path = Path(local_path.strip())
            if not path.is_file():
                st.error("本机路径不存在，或不是文件。")
                return
        else:
            st.error("请上传一个 PDF，或填写一个本机 PDF 路径。")
            return
        _run_ingestion(service, path, collection)


def _render_delete_section(service: IngestionService) -> None:
    st.divider()
    st.subheader("删除已摄取文档")
    collections = service.list_collections()
    collection = st.selectbox("选择要管理的 Collection", collections, key="delete_collection")
    documents = service.list_documents(collection)
    if not documents:
        st.info("这个 Collection 目前没有可删除的文档。")
        return

    document = st.selectbox(
        "选择要删除的文档",
        documents,
        format_func=lambda item: f"{item.title} — {item.source_path}",
    )
    confirmed = st.checkbox("我确认要从本地知识库中删除该文档及其索引和图片。")
    if st.button("删除文档", type="secondary", disabled=not confirmed):
        try:
            result = service.delete_document(document.source_path, collection)
        except Exception as exc:  # Display incomplete-delete failures honestly to the operator.
            st.error(f"删除失败：{exc}")
            return
        st.success(
            "已删除："
            f"{result.chunks_deleted} 个 Chunk、{result.bm25_entries_deleted} 条 BM25 条目、"
            f"{result.images_deleted} 张图片。"
        )
        st.rerun()


def render() -> None:
    """Render the ingestion-management route registered in ``dashboard.app``."""
    st.title("Ingestion 管理")
    st.caption("上传或选择本机 PDF，观察摄取进度，并管理已摄取文档。")
    try:
        service = IngestionService()
        _render_ingestion_form(service)
        _render_delete_section(service)
    except Exception as exc:
        st.error(f"无法初始化摄取管理：{exc}")


if __name__ == "__main__":
    render()
