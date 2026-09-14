"""Streamlit entry point and navigation for the management Dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

from observability.dashboard.pages import (
    data_browser,
    evaluation_panel,
    ingestion_manager,
    ingestion_traces,
    overview,
    query_traces,
)


def main() -> None:
    """Configure the app, register all six routes, and run the selected page."""
    st.set_page_config(page_title="Knowledge Hub v1", page_icon="📚", layout="wide")

    navigation = st.navigation(
        {
            "概览": [
                st.Page(
                    overview.render,
                    title="系统总览",
                    icon="🏠",
                    url_path="overview",
                    default=True,
                )
            ],
            "知识库管理": [
                st.Page(data_browser.render, title="数据浏览", icon="🗂️", url_path="data-browser"),
                st.Page(
                    ingestion_manager.render,
                    title="Ingestion 管理",
                    icon="📥",
                    url_path="ingestion-manager",
                ),
            ],
            "可观测性": [
                st.Page(
                    ingestion_traces.render,
                    title="Ingestion 追踪",
                    icon="📈",
                    url_path="ingestion-traces",
                ),
                st.Page(
                    query_traces.render,
                    title="查询追踪",
                    icon="🔎",
                    url_path="query-traces",
                ),
                st.Page(
                    evaluation_panel.render,
                    title="评估面板",
                    icon="🧪",
                    url_path="evaluation",
                ),
            ],
        }
    )
    navigation.run()


if __name__ == "__main__":
    main()
