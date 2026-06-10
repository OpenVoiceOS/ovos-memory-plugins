"""Unit tests for RAGMemory plugin — all HTTP calls are mocked."""
from unittest.mock import MagicMock, patch, call

import pytest

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_memory_plugins.rag import RAGMemory, _cosine, _embed, _upload_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _user(text): return AgentMessage(role=MessageRole.USER, content=text)
def _assistant(text): return AgentMessage(role=MessageRole.ASSISTANT, content=text)


def _make_plugin(**kwargs):
    cfg = {
        "api_url": "http://mock/v1",
        "collection": "test_store",
        "top_k": 3,
        "min_score": 0.0,
        **kwargs,
    }
    return RAGMemory(config=cfg)


# ---------------------------------------------------------------------------
# _cosine
# ---------------------------------------------------------------------------

def test_cosine_identical():
    v = [1.0, 0.0, 0.0]
    assert abs(_cosine(v, v) - 1.0) < 1e-6


def test_cosine_orthogonal():
    assert abs(_cosine([1, 0], [0, 1])) < 1e-6


def test_cosine_zero_vector():
    assert _cosine([0, 0], [1, 1]) == 0.0


# ---------------------------------------------------------------------------
# get_history / update_history
# ---------------------------------------------------------------------------

def test_get_history_empty():
    plugin = _make_plugin()
    assert plugin.get_history("s1") == []


@patch("ovos_memory_plugins.rag._embed", return_value=[0.1, 0.2])
@patch("ovos_memory_plugins.rag._upload_file", return_value="file_001")
@patch("ovos_memory_plugins.rag._attach_to_store")
def test_update_history_stores_exchange(mock_attach, mock_upload, mock_embed):
    plugin = _make_plugin()
    plugin.update_history([_user("hi"), _assistant("hello")], "s1")
    # embed called once for the assistant response
    mock_embed.assert_called_once()
    mock_upload.assert_called_once()
    mock_attach.assert_called_once()


@patch("ovos_memory_plugins.rag._embed", return_value=None)
@patch("ovos_memory_plugins.rag._upload_file", return_value=None)
@patch("ovos_memory_plugins.rag._attach_to_store")
def test_update_history_handles_embed_failure(mock_attach, mock_upload, mock_embed):
    plugin = _make_plugin()
    # Should not crash even when embed + upload return None
    plugin.update_history([_user("x"), _assistant("y")], "s1")


@patch("ovos_memory_plugins.rag._embed", return_value=[1.0, 0.0])
@patch("ovos_memory_plugins.rag._upload_file", return_value="f1")
@patch("ovos_memory_plugins.rag._attach_to_store")
def test_recent_window_capped_at_10(mock_a, mock_u, mock_e):
    plugin = _make_plugin()
    for i in range(7):
        plugin.update_history([_user(f"q{i}"), _assistant(f"a{i}")], "s1")
    assert len(plugin._recent["s1"]) <= 10


# ---------------------------------------------------------------------------
# Retrieval ranking and min_score
# ---------------------------------------------------------------------------

def test_local_search_ranking():
    plugin = _make_plugin(min_score=0.0)
    plugin._local_store = [
        ([1.0, 0.0], "high similarity doc"),
        ([0.0, 1.0], "low similarity doc"),
        ([0.9, 0.1], "medium similarity doc"),
    ]
    query_vec = [1.0, 0.0]
    results = plugin._local_search(query_vec, k=3)
    # Should be sorted by score descending
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0][0] == "high similarity doc"


def test_local_search_min_score_filter():
    plugin = _make_plugin(min_score=0.9)
    plugin._local_store = [
        ([1.0, 0.0], "perfect match"),
        ([0.0, 1.0], "orthogonal"),
    ]
    query_vec = [1.0, 0.0]

    # Mock _vector_search to return nothing (force local fallback)
    with patch("ovos_memory_plugins.rag._vector_search", return_value=[]):
        with patch("ovos_memory_plugins.rag._embed", return_value=query_vec):
            results = plugin._retrieve("test query")

    assert all(s >= 0.9 for _, s in results)
    assert len(results) == 1
    assert results[0][0] == "perfect match"


def test_local_search_top_k_respected():
    plugin = _make_plugin(top_k=2)
    plugin._local_store = [
        ([1.0, 0.0], "doc1"),
        ([0.9, 0.1], "doc2"),
        ([0.8, 0.2], "doc3"),
    ]
    with patch("ovos_memory_plugins.rag._vector_search", return_value=[]):
        with patch("ovos_memory_plugins.rag._embed", return_value=[1.0, 0.0]):
            results = plugin._retrieve("q")
    assert len(results) <= 2


# ---------------------------------------------------------------------------
# build_conversation_context
# ---------------------------------------------------------------------------

@patch("ovos_memory_plugins.rag._vector_search", return_value=[("Q: old\nA: answer", 0.85)])
def test_build_context_injects_retrieved(mock_vs):
    plugin = _make_plugin(system_prompt="Be helpful")
    ctx = plugin.build_conversation_context("new question", "s1")
    assert ctx[0].role == MessageRole.SYSTEM
    assert ctx[-1].role == MessageRole.USER
    assert ctx[-1].content == "new question"
    # Check retrieved doc appears somewhere
    contents = " ".join(m.content for m in ctx)
    assert "old" in contents or "answer" in contents or "Recalled" in contents


@patch("ovos_memory_plugins.rag._vector_search", return_value=[])
@patch("ovos_memory_plugins.rag._embed", return_value=None)
def test_build_context_no_results(mock_embed, mock_vs):
    plugin = _make_plugin()
    ctx = plugin.build_conversation_context("question", "s1")
    assert ctx[-1].content == "question"


@patch("ovos_memory_plugins.rag._vector_search", return_value=[("past exchange", 0.7)])
def test_build_context_inject_as_system(mock_vs):
    plugin = _make_plugin(inject_as_system=True)
    ctx = plugin.build_conversation_context("test", "s1")
    system_msgs = [m for m in ctx if m.role == MessageRole.SYSTEM]
    assert any("Relevant past exchange" in m.content for m in system_msgs)


@patch("ovos_memory_plugins.rag._vector_search", return_value=[("low score doc", 0.1)])
def test_build_context_min_score_filters_retrieved(mock_vs):
    plugin = _make_plugin(min_score=0.5)
    # Even with server returning a result, it should be filtered by min_score
    # vector_search returns it; but _retrieve filters by min_score
    with patch("ovos_memory_plugins.rag._embed", return_value=None):
        ctx = plugin.build_conversation_context("test", "s1")
    contents = " ".join(m.content for m in ctx)
    assert "low score doc" not in contents


# ---------------------------------------------------------------------------
# Endpoint error resilience
# ---------------------------------------------------------------------------

@patch("ovos_memory_plugins.rag._embed", side_effect=Exception("no embed"))
@patch("ovos_memory_plugins.rag._upload_file", side_effect=Exception("no upload"))
@patch("ovos_memory_plugins.rag._attach_to_store", side_effect=Exception("no attach"))
@patch("ovos_memory_plugins.rag._vector_search", side_effect=Exception("no search"))
def test_all_endpoints_failing_no_crash(mock_vs, mock_attach, mock_upload, mock_embed):
    plugin = _make_plugin()
    plugin.update_history([_user("hi"), _assistant("hey")], "s1")
    ctx = plugin.build_conversation_context("next", "s1")
    assert ctx[-1].content == "next"
