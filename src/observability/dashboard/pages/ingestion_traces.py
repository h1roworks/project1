"""Streamlit page showing historical ingestion traces."""

from __future__ import annotations

import streamlit as st

from observability.dashboard.services.trace_service import IngestionTrace, TraceService


def _status_label(status: str) -> str:
    return {"success": "成功", "failed": "失败", "skipped": "跳过"}.get(status, status)


def _render_trace_detail(trace: IngestionTrace) -> None:
    st.divider()
    st.subheader(f"追踪详情：{trace.filename}")
    st.caption(f"Trace ID: {trace.trace_id}")
    if trace.error:
        st.error(trace.error)
    if not trace.stages:
        st.info("该追踪记录没有阶段数据。")
        return

    st.markdown("##### 阶段耗时（毫秒）")
    st.bar_chart(
        {stage.name: stage.elapsed_ms for stage in trace.stages},
        x_label="阶段",
        y_label="耗时（ms）",
    )
    for stage in trace.stages:
        with st.expander(f"{stage.name} · {stage.elapsed_ms:.2f} ms"):
            st.write(f"Method: {stage.method or '-'}")
            st.write(f"Provider: {stage.provider or '-'}")
            if stage.details:
                st.json(stage.details)


def render() -> None:
    """Render ingestion history and the selected trace's stage waterfall."""
    st.title("Ingestion 追踪")
    st.caption("查看文档摄取历史、执行状态和各阶段耗时。")
    if st.button("刷新追踪"):
        st.rerun()

    try:
        traces = TraceService().list_ingestion_traces()
    except Exception as exc:
        st.error(f"无法读取追踪日志：{exc}")
        return
    if not traces:
        st.info("暂时没有 Ingestion 追踪记录。请先摄取一个文档。")
        return

    st.dataframe(
        [
            {
                "文件": trace.filename,
                "Collection": trace.collection or "-",
                "状态": _status_label(trace.status),
                "总耗时（ms）": round(trace.total_elapsed_ms, 2),
                "开始时间": trace.started_at,
                "Trace ID": trace.trace_id,
            }
            for trace in traces
        ],
        hide_index=True,
        width="stretch",
    )
    selected_id = st.selectbox(
        "选择一条追踪记录查看详情",
        [trace.trace_id for trace in traces],
        format_func=lambda trace_id: next(
            f"{trace.filename} · {_status_label(trace.status)} · {trace.total_elapsed_ms:.2f} ms"
            for trace in traces
            if trace.trace_id == trace_id
        ),
    )
    selected = next(trace for trace in traces if trace.trace_id == selected_id)
    _render_trace_detail(selected)


if __name__ == "__main__":
    render()
