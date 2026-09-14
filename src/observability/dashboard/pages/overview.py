"""System overview page for the Streamlit Dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from libs.vector_store.chroma_store import ChromaStore
from observability.dashboard.services.config_service import ConfigService


def _render_component_cards(service: ConfigService) -> None:
    st.subheader("当前组件配置")
    cards = service.get_component_configs()
    for start in range(0, len(cards), 2):
        columns = st.columns(2)
        for column, component in zip(columns, cards[start : start + 2]):
            with column:
                st.markdown(f"#### {component.name}")
                st.write(f"**Provider:** {component.provider}")
                st.write(f"**Model:** {component.model}")
                st.caption(component.details)


def _render_data_stats(settings: Any) -> None:
    st.subheader("数据资产统计")
    try:
        stats = ChromaStore(settings.vector_store).get_collection_stats()
    except Exception as exc:  # A missing/unavailable local database should not block the UI.
        st.warning(f"暂时无法读取向量库统计：{exc}")
        return

    documents, chunks, images, size = st.columns(4)
    documents.metric("文档数", stats["documents"])
    chunks.metric("Chunk 数", stats["chunks"])
    images.metric("图片数", stats["images"])
    size.metric("本地库大小", _format_bytes(stats["database_size_bytes"]))
    st.caption(f"当前集合：{stats['collection']}")


def _format_bytes(size: int) -> str:
    """Display a byte value in a compact human-readable form."""
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024
    return f"{size} B"


def render() -> None:
    """Render the first Dashboard page."""
    st.title("系统总览")
    st.caption("查看当前 RAG 组件配置，以及本地知识库中已保存的数据量。")

    service = ConfigService()
    try:
        settings = service.get_settings()
        _render_component_cards(service)
        st.divider()
        _render_data_stats(settings)
    except Exception as exc:
        st.error(f"Dashboard 初始化失败：{exc}")
        st.info("请检查 config/settings.yaml 是否存在且配置格式正确。")
        return

    with st.expander("查看当前数据目录"):
        persist_dir = Path(settings.vector_store.persist_dir).resolve()
        st.code(str(persist_dir), language=None)
