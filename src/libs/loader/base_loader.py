"""Loader 抽象基类：文档加载器的最小接口。

约定（对齐 DEV_SPEC 3.1.1）：Loader 只负责把原始文件解析为统一的
``core.types.Document`` 对象（``text`` + ``metadata``），
**不做切分**（那是 Splitter 的职责）、**不写存储**（那是 Pipeline 的职责）。
``Document.text`` 为规范化 Markdown，``metadata`` 至少包含 ``source_path``。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from core.types import Document


class BaseLoader(ABC):
    """所有文档加载器的抽象基类。

    ``supported_extensions`` 声明本 Loader 能处理的文件扩展名；
    ``can_handle()`` 供 Pipeline / 调度器按文件类型选择对应 Loader，
    新增一种文档格式（如 Markdown）时只需继承本类并声明其扩展名。
    """

    loader_name: str = "base"
    supported_extensions: tuple[str, ...] = ()

    @abstractmethod
    def load(self, path: str) -> Document:
        """解析 ``path`` 指向的文件，返回统一的 ``Document`` 对象。

        Args:
            path: 待解析文件的路径。

        Returns:
            规范化 Markdown 的 Document，metadata 携带定位与来源信息。

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: 文件无法解析出有效文本。
        """
        raise NotImplementedError

    @classmethod
    def can_handle(cls, path: str) -> bool:
        """按扩展名判断本 Loader 是否能处理该文件（大小写不敏感）。"""
        return Path(path).suffix.lower() in cls.supported_extensions
