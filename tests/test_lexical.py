"""Unit tests for LexicalMemory (real stdlib SQLite FTS5, temp dir — no skips)."""
import sqlite3
import tempfile
from pathlib import Path

import pytest

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_memory_plugins.lexical import LexicalMemory


def _user(text): return AgentMessage(role=MessageRole.USER, content=text)
def _assistant(text): return AgentMessage(role=MessageRole.ASSISTANT, content=text)


def _ingest(plugin, session_id, exchanges):
    for u, a in exchanges:
        plugin.update_history([_user(u), _assistant(a)], session_id)


GEO = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Germany?", "The capital of Germany is Berlin."),
    ("How do you make pasta?", "Boil salted water and cook pasta for ten minutes."),
]


def _fts5_supported() -> bool:
    try:
        con = sqlite3.connect(":memory:")
        con.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        return True
    except sqlite3.OperationalError:
        return False


pytestmark = pytest.mark.skipif(not _fts5_supported(),
                                reason="SQLite build lacks FTS5 (not expected on standard CPython)")


def _make(**kwargs) -> LexicalMemory:
    cfg = {"db_path": ":memory:", **kwargs}
    return LexicalMemory(config=cfg)


def test_keyword_ranking_surfaces_match():
    p = _make()
    _ingest(p, "s1", GEO)
    hits = p.search("France capital")
    assert hits, "expected at least one hit"
    assert "Paris" in hits[0].content


def test_no_match_returns_empty():
    p = _make()
    _ingest(p, "s1", GEO)
    assert p.search("zzzqqx nonexistent") == []


def test_empty_query_returns_empty():
    p = _make()
    _ingest(p, "s1", GEO)
    assert p.search("?!!  ...") == []  # no alnum terms → no MATCH


def test_punctuation_is_sanitized():
    p = _make()
    _ingest(p, "s1", GEO)
    # punctuation/quotes must not raise an FTS5 syntax error
    hits = p.search('France?!! "(capital)"')
    assert any("Paris" in h.content for h in hits)


def test_search_returns_memoryhit_shape():
    p = _make()
    _ingest(p, "s1", GEO)
    hit = p.search("Berlin")[0]
    assert "Berlin" in hit.content
    assert hit.source  # doc id present
    assert isinstance(hit.score, float)


def test_max_num_results_caps():
    p = _make(retrieval={"max_num_results": 1})
    _ingest(p, "s1", GEO)
    assert len(p.search("capital")) == 1


def test_persistence_across_instances():
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "lex.db")
        p1 = LexicalMemory(config={"db_path": path})
        _ingest(p1, "s1", GEO)
        p1.con.close()
        p2 = LexicalMemory(config={"db_path": path})
        hits = p2.search("France")
        assert any("Paris" in h.content for h in hits)


def test_context_contract_last_is_user():
    p = _make(system_prompt="You are helpful.")
    _ingest(p, "s1", GEO)
    ctx = p.build_conversation_context("Tell me about France", "s1")
    assert ctx[-1].role == MessageRole.USER
    assert ctx[-1].content == "Tell me about France"
    assert ctx[0].role == MessageRole.SYSTEM


def test_inject_tool_mode_orders_correctly():
    p = _make(inject_mode="tool", tool_name="search_memory")
    _ingest(p, "s1", GEO)
    ctx = p.build_conversation_context("France capital", "s1")
    idx = next(i for i, m in enumerate(ctx)
               if m.role == MessageRole.ASSISTANT and m.tool_calls)
    assert ctx[idx + 1].role == MessageRole.TOOL
    assert ctx[-1].role == MessageRole.USER


def test_store_failure_does_not_break_history():
    p = _make()
    # force a broken store by closing the connection it will reuse
    p.con.close()
    _ingest(p, "s1", [GEO[0]])  # must not raise
    assert len(p.get_history("s1")) == 2
