from unittest.mock import patch
from app.tools import search_knowledge_base


def test_tool_metadata():
    assert search_knowledge_base.name == "search_knowledge_base"
    assert "knowledge base" in search_knowledge_base.description.lower()


def test_search_returns_results():
    with patch("app.tools.retrieval._svc.search", return_value=["restart the service", "check logs"]):
        result = search_knowledge_base.invoke("runaway process")
    assert isinstance(result, str)
    assert len(result) > 0
    assert result != "No relevant content found."


def test_search_unknown_query():
    with patch("app.tools.retrieval._svc.search", return_value=[]):
        result = search_knowledge_base.invoke("quantum entanglement")
    assert result == "No relevant content found."
