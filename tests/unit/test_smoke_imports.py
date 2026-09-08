"""Smoke test: verify key top-level packages are importable.

Phase A2 acceptance: at least one import smoke test passes.
"""


def test_top_level_packages_import():
    import mcp_server  # noqa: F401
    import core  # noqa: F401
    import ingestion  # noqa: F401
    import libs  # noqa: F401
    import observability  # noqa: F401


def test_sub_packages_import():
    import mcp_server.tools  # noqa: F401
    import core.query_engine  # noqa: F401
    import core.response  # noqa: F401
    import core.trace  # noqa: F401
    import ingestion.chunking  # noqa: F401
    import ingestion.transform  # noqa: F401
    import ingestion.embedding  # noqa: F401
    import ingestion.storage  # noqa: F401
    import libs.loader  # noqa: F401
    import libs.llm  # noqa: F401
    import libs.embedding  # noqa: F401
    import libs.splitter  # noqa: F401
    import libs.vector_store  # noqa: F401
    import libs.reranker  # noqa: F401
    import libs.evaluator  # noqa: F401
    import observability.dashboard  # noqa: F401
    import observability.evaluation  # noqa: F401
