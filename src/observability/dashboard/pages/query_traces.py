"""Streamlit page for inspecting query retrieval and rerank traces."""

from __future__ import annotations

from typing import Any

import streamlit as st

from observability.dashboard.services.trace_service import QueryTrace, TraceService, TraceStage


def _stage(trace: QueryTrace, name: str) -> TraceStage | None:
    return next((item for item in trace.stages if item.name == name), None)


def _render_result_comparison(stage: TraceStage | None, title: str) -> None:
    st.markdown(f"##### {title}")
    if stage is None:
        st.caption("没有该阶段的追踪数据。")
        return
    results = stage.details.get("results")
    if isinstance(results, list) and results:
        st.dataframe(results, hide_index=True, width="stretch")
    else:
        st.caption(f"召回数量：{stage.details.get('result_count', 0)}（旧日志未保存具体 ID）")


def _render_rerank_comparison(stage: TraceStage | None) -> None:
    st.markdown("##### Rerank 前后排序")
    if stage is None:
        st.caption("没有 Rerank 阶段的追踪数据。")
        return
    before = stage.details.get("before_ids") or []
    after = stage.details.get("after_ids") or []
    if not before and not after:
        st.caption("旧日志未保存 Rerank 前后的 Chunk ID。")
        return
    max_length = max(len(before), len(after))
    st.dataframe(
        [
            {"原始排名": index + 1 if index < len(before) else "-", "原始 Chunk": before[index] if index < len(before) else "-", "Rerank 排名": index + 1 if index < len(after) else "-", "Rerank Chunk": after[index] if index < len(after) else "-"}
            for index in range(max_length)
        ],
        hide_index=True,
        width="stretch",
    )


def _render_detail(trace: QueryTrace) -> None:
    st.divider()
    st.subheader("查询详情")
    st.write(f"**Query:** {trace.query or '(旧日志未记录 Query 文本)'}")
    st.caption(f"Trace ID: {trace.trace_id} · 总耗时：{trace.total_elapsed_ms:.2f} ms")
    if trace.stages:
        st.markdown("##### 阶段耗时（毫秒）")
        st.bar_chart(
            {stage.name: stage.elapsed_ms for stage in trace.stages},
            x_label="阶段",
            y_label="耗时（ms）",
        )
    dense, sparse = st.columns(2)
    with dense:
        _render_result_comparison(_stage(trace, "dense_retrieval"), "Dense 召回")
    with sparse:
        _render_result_comparison(_stage(trace, "sparse_retrieval"), "Sparse / BM25 召回")
    _render_rerank_comparison(_stage(trace, "rerank"))


def render() -> None:
    """Render query history, keyword filtering, and selected trace details."""
    st.title("查询追踪")
    st.caption("对比每次查询的 Dense、Sparse 和 Rerank 处理链路。")
    query_filter = st.text_input("按 Query 关键词筛选")
    if st.button("刷新查询追踪"):
        st.rerun()
    try:
        traces = TraceService().list_query_traces(query_filter)
    except Exception as exc:
        st.error(f"无法读取查询追踪：{exc}")
        return
    if not traces:
        st.info("没有匹配的 Query 追踪记录。")
        return
    st.dataframe(
        [
            {"Query": trace.query or "-", "总耗时（ms）": round(trace.total_elapsed_ms, 2), "开始时间": trace.started_at, "Trace ID": trace.trace_id}
            for trace in traces
        ],
        hide_index=True,
        width="stretch",
    )
    selected_id = st.selectbox(
        "选择一条查询追踪",
        [trace.trace_id for trace in traces],
        format_func=lambda trace_id: next(
            f"{trace.query or '(旧日志)'} · {trace.total_elapsed_ms:.2f} ms"
            for trace in traces
            if trace.trace_id == trace_id
        ),
    )
    _render_detail(next(trace for trace in traces if trace.trace_id == selected_id))


if __name__ == "__main__":
    render()
