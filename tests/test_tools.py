from app.tools import search_knowledge_base


def test_tool_metadata():
    assert search_knowledge_base.name == "search_knowledge_base"
    assert "knowledge base" in search_knowledge_base.description.lower()


def test_search_returns_results():
    result = search_knowledge_base.invoke("runaway process")
    assert isinstance(result, str)
    assert len(result) > 0
    assert result != "No relevant content found."


def test_search_unknown_query():
    result = search_knowledge_base.invoke("quantum entanglement")
    # should still return a string, even if low relevance
    assert isinstance(result, str)
