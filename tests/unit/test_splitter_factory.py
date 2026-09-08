"""B3: Splitter 抽象接口与工厂测试。

Factory 根据配置的 strategy 返回不同类型的 Splitter 实例（测试用 Fake）。
"""

import pytest

from core.settings import SplitterSettings
from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.splitter_factory import SplitterFactory


class FakeSplitter(BaseSplitter):
    strategy = "fake"

    def __init__(self, settings: SplitterSettings) -> None:
        self.settings = settings

    def split_text(self, text, trace=None) -> list[str]:
        size = max(self.settings.chunk_size, 1)
        return [text[i : i + size] for i in range(0, len(text), size) or [0]]


@pytest.fixture(autouse=True)
def _register_fake() -> None:
    SplitterFactory.register("fake", FakeSplitter)
    yield
    SplitterFactory._registry.pop("fake", None)


def make_settings(**overrides) -> SplitterSettings:
    base = {"strategy": "fake", "chunk_size": 4, "chunk_overlap": 1}
    base.update(overrides)
    return SplitterSettings(**base)


def test_factory_returns_impl_instance() -> None:
    s = SplitterFactory.create(make_settings())
    assert isinstance(s, FakeSplitter)
    assert isinstance(s, BaseSplitter)


def test_factory_passes_settings_through() -> None:
    s = SplitterFactory.create(make_settings(chunk_size=8))
    assert s.settings.chunk_size == 8


def test_split_text_respects_chunk_size() -> None:
    s = SplitterFactory.create(make_settings(chunk_size=4))
    chunks = s.split_text("0123456789")
    assert all(len(c) <= 4 for c in chunks)
    assert "".join(chunks) == "0123456789"


def test_empty_strategy_raises() -> None:
    with pytest.raises(ValueError, match="策略未配置"):
        SplitterFactory.create(make_settings(strategy=""))


def test_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="未知的 Splitter 策略"):
        SplitterFactory.create(make_settings(strategy="nonexistent"))
