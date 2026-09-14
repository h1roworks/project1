"""普通应用日志与 Trace 的 JSON Lines 持久化。"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

_LOGGER_NAME = "smart_knowledge_hub"
_TRACE_LOGGER_PREFIX = f"{_LOGGER_NAME}.trace"
DEFAULT_TRACE_FILE = Path("logs/traces.jsonl")


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """Return a configured logger writing to stderr."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


class JSONFormatter(logging.Formatter):
    """把 trace 字典渲染为一行 JSON。

    传入 ``extra={"trace": trace_dict}`` 时，输出该 trace 本身；普通日志记录
    则包含时间、等级、logger 名称和消息，以便同一 formatter 也能安全使用。
    """

    def format(self, record: logging.LogRecord) -> str:
        trace = getattr(record, "trace", None)
        if isinstance(trace, dict):
            payload: dict[str, Any] = dict(trace)
        elif isinstance(record.msg, dict):
            payload = dict(record.msg)
        else:
            payload = {
                "timestamp": datetime.fromtimestamp(
                    record.created,
                    tz=datetime.now().astimezone().tzinfo,
                ).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
        return json.dumps(payload, ensure_ascii=False, default=_json_default)


def get_trace_logger(trace_file: str | Path = DEFAULT_TRACE_FILE) -> logging.Logger:
    """获取写入指定 JSON Lines 文件的 trace logger。

    同一个文件路径复用同一个 logger，避免重复添加 FileHandler 导致一条 trace
    被写入多次。
    """
    path = Path(trace_file).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
    logger = logging.getLogger(f"{_TRACE_LOGGER_PREFIX}.{digest}")

    if not logger.handlers:
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def write_trace(
    trace_dict: dict[str, Any],
    trace_file: str | Path = DEFAULT_TRACE_FILE,
) -> None:
    """把一条完整 trace 追加到 JSON Lines 文件。

    ``FileHandler`` 每处理一条 logging record 都会添加一个换行符，因此文件中
    每一行都是一个可独立 ``json.loads`` 的请求记录。
    """
    if not isinstance(trace_dict, dict):
        raise TypeError("trace_dict must be a dictionary")
    get_trace_logger(trace_file).info("trace", extra={"trace": trace_dict})


def _json_default(value: Any) -> str:
    """为调用方 details 中的少量非 JSON 值提供稳定的兜底表示。"""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)
