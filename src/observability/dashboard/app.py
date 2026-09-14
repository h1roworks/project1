"""Streamlit entry point and navigation for the management Dashboard."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

from observability.dashboard.pages import data_browser, overview


def _placeholder(title: str, description: str) -> Callable[[], None]:
    """Create a temporary page while its feature is implemented in later tasks."""

    def render() -> None:
        st.title(title)
        st.info(f"{description}（将在后续 G 阶段任务中实现。）")

    return render


def main() -> None:
    """Configure the app, register all six routes, and run the selected page."""
    st.set_page_config(page_title="Smart Knowledge Hub", page_icon="📚", layout="wide")

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
                    _placeholder("Ingestion 管理", "上传文件、触发摄取和管理文档"),
                    title="Ingestion 管理",
                    icon="📥",
                    url_path="ingestion-manager",
                ),
            ],
            "可观测性": [
                st.Page(
                    _placeholder("Ingestion 追踪", "查看每次文档摄取的处理链路"),
                    title="Ingestion 追踪",
                    icon="📈",
                    url_path="ingestion-traces",
                ),
                st.Page(
                    _placeholder("查询追踪", "查看每次检索查询的处理链路"),
                    title="查询追踪",
                    icon="🔎",
                    url_path="query-traces",
                ),
                st.Page(
                    _placeholder("评估面板", "评估模块尚未启用"),
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
