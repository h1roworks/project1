"""Dashboard page for running and comparing Golden Set evaluations."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from core.settings import EvaluationSettings, Settings
from libs.evaluator.evaluator_factory import EvaluatorFactory
from observability.evaluation.eval_runner import EvalReport, EvalRunner
from observability.dashboard.services.config_service import ConfigService

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_TEST_SET = _PROJECT_ROOT / "tests" / "fixtures" / "golden_test_set.json"
_PROVIDERS = ("custom", "ragas", "composite")


def run_evaluation(settings: Settings, provider: str, test_set_path: str | Path) -> EvalReport:
    """Build the existing query pipeline and run one evaluation.

    Imports the query-engine builder lazily so merely opening other Dashboard
    pages does not initialize vector/BM25 clients.
    """
    from mcp_server.tools.query_knowledge_hub import build_query_engine

    if provider not in _PROVIDERS:
        raise ValueError(f"未知的评估后端: {provider}")
    evaluation = replace(settings.evaluation, provider=provider)
    hybrid_search, _ = build_query_engine(settings)
    evaluator = EvaluatorFactory.create(evaluation)
    return EvalRunner(settings, hybrid_search, evaluator).run(test_set_path)


def _case_rows(report: EvalReport) -> list[dict[str, Any]]:
    """Turn per-query results into a compact table for Streamlit."""
    return [
        {
            "问题": case.query,
            "命中": "是" if case.hit else "否",
            "MRR": round(case.mrr, 4),
            "期望 Chunk": ", ".join(case.expected_chunk_ids) or "-",
            "召回 Chunk": ", ".join(case.retrieved_chunk_ids) or "-",
            "来源": ", ".join(case.retrieved_sources) or "-",
        }
        for case in report.case_results
    ]


def _render_report(report: EvalReport) -> None:
    st.subheader("评估结果")
    metric_items = list(report.metrics.items())
    columns = st.columns(min(4, max(1, len(metric_items))))
    for column, (name, value) in zip(columns, metric_items):
        column.metric(name, f"{value:.4f}")

    if not report.case_results:
        st.info("Golden Test Set 为空，没有可展示的问题。")
        return
    st.markdown("##### 各问题明细")
    st.dataframe(_case_rows(report), hide_index=True, width="stretch")


def _render_history() -> None:
    history = st.session_state.get("evaluation_history", [])
    if not history:
        return
    st.divider()
    st.subheader("本次会话的历史结果")
    st.dataframe(history, hide_index=True, width="stretch")


def render() -> None:
    """Render the H4 evaluation page."""
    st.title("评估面板")
    st.caption("运行 Golden Test Set，量化检索质量，并查看每个问题的命中情况。")

    try:
        settings = ConfigService().get_settings()
    except Exception as exc:
        st.error(f"无法读取评估配置：{exc}")
        return

    current_provider = getattr(settings.evaluation, "provider", "custom")
    provider = st.selectbox(
        "评估后端",
        _PROVIDERS,
        index=_PROVIDERS.index(current_provider) if current_provider in _PROVIDERS else 0,
        help="Custom 计算命中率/MRR；Ragas 评估答案质量；Composite 可组合多个后端。",
    )
    configured_path = getattr(settings.evaluation, "golden_test_set_path", "")
    test_set_path = st.text_input(
        "Golden Test Set 路径",
        value=configured_path or str(_DEFAULT_TEST_SET),
        help="可以填写绝对路径，也可以填写相对于项目根目录的路径。",
    ).strip()

    if st.button("运行评估", type="primary"):
        if not test_set_path:
            st.error("请填写 Golden Test Set 路径。")
            return
        with st.spinner("正在运行评估，请稍候…"):
            try:
                report = run_evaluation(settings, provider, test_set_path)
            except Exception as exc:  # The page should remain usable after a failed run.
                st.error(f"评估失败：{exc}")
                return
        st.session_state.setdefault("evaluation_history", []).append(
            {
                "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "后端": provider,
                **{name: round(value, 4) for name, value in report.metrics.items()},
            }
        )
        st.session_state["evaluation_last_report"] = report

    report = st.session_state.get("evaluation_last_report")
    if isinstance(report, EvalReport):
        _render_report(report)
    _render_history()


if __name__ == "__main__":
    render()
