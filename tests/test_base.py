"""Unit tests for BaseRetrievalMemory via a trivial in-memory stub subclass.

These lock the storage-agnostic machinery the concrete retrievers inherit:
history, the five inject modes, min_score/top_k, store resilience, and the
trailing-user strip that keeps the AgentContextManager contract.
"""
from typing import Any, Dict, List

import pytest

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_memory_plugins.base import BaseRetrievalMemory
from ovos_memory_plugins.common import MemoryHit


class _ListRetriever(BaseRetrievalMemory):
    """Returns canned hits from config['_hits'] = [(content, score), ...]."""

    def __init__(self, config=None):
        super().__init__(config)
        self.docs: Dict[str, str] = {}

    def _store_document(self, doc_id, text, session_id, metadata):
        self.docs[doc_id] = text

    def _query_backend(self, query, top_k):
        hits = [MemoryHit(content=c, source=f"d{i}", score=s)
                for i, (c, s) in enumerate(self.config.get("_hits", []))]
        return hits[:top_k]


def _user(t): return AgentMessage(role=MessageRole.USER, content=t)
def _assistant(t): return AgentMessage(role=MessageRole.ASSISTANT, content=t)


def test_invalid_inject_mode_raises():
    with pytest.raises(ValueError):
        _ListRetriever(config={"inject_mode": "nope"})


def test_update_history_stores_exchange_and_truncates():
    p = _ListRetriever(config={"max_history": 2})
    p.update_history([_user("q1"), _assistant("a1")], "s1")
    p.update_history([_user("q2"), _assistant("a2")], "s1")
    assert any("q1" in d and "a1" in d for d in p.docs.values())
    assert len(p.get_history("s1")) == 2  # truncated


def test_store_failure_does_not_break_history():
    class Boom(_ListRetriever):
        def _store_document(self, *a, **k):
            raise RuntimeError("disk full")
    p = Boom()
    p.update_history([_user("q"), _assistant("a")], "s1")  # must not raise
    assert len(p.get_history("s1")) == 2


def test_search_applies_min_score_and_top_k():
    p = _ListRetriever(config={"_hits": [("a", 0.9), ("b", 0.5), ("c", 0.1)],
                               "retrieval": {"min_score": 0.4, "max_num_results": 5}})
    hits = p.search("q")
    assert [h.content for h in hits] == ["a", "b"]  # 0.1 dropped
    assert len(p.search("q", top_k=1)) == 1


def test_context_contract_last_is_user():
    p = _ListRetriever(config={"_hits": [("ctx", 1.0)], "system_prompt": "Base."})
    ctx = p.build_conversation_context("hello", "s1")
    assert ctx[0].role == MessageRole.SYSTEM and ctx[0].content == "Base."
    assert ctx[-1].role == MessageRole.USER and ctx[-1].content == "hello"


@pytest.mark.parametrize("mode", ["system", "developer", "system_prompt", "user", "tool"])
def test_all_inject_modes_keep_user_last(mode):
    p = _ListRetriever(config={"_hits": [("recalled", 1.0)], "inject_mode": mode})
    ctx = p.build_conversation_context("question", "s1")
    assert ctx[-1].role == MessageRole.USER
    assert sum(1 for m in ctx if m.role == MessageRole.USER) == 1


def test_no_hits_no_context_block():
    p = _ListRetriever(config={"_hits": [], "system_prompt": "Base."})
    ctx = p.build_conversation_context("x", "s1")
    assert [m.role for m in ctx] == [MessageRole.SYSTEM, MessageRole.USER]


def test_trailing_user_in_history_is_stripped():
    # a dangling user turn (no assistant reply yet) must not produce two trailing users
    p = _ListRetriever(config={"_hits": []})
    p.session2history["s1"] = [_user("old q"), _assistant("old a"), _user("dangling")]
    ctx = p.build_conversation_context("new q", "s1")
    assert all(m.content != "dangling" for m in ctx)   # only the trailing dangling user is dropped
    assert ctx[-1].role == MessageRole.USER and ctx[-1].content == "new q"
    assert any(m.content == "old q" for m in ctx)       # answered earlier turn is kept


def test_tool_mode_with_trailing_user_history_is_well_formed():
    # the critical case: tool inject + a history ending in USER must keep the
    # assistant(tool_call) -> tool(result) -> user(utterance) ordering valid
    p = _ListRetriever(config={"_hits": [("ctx", 1.0)], "inject_mode": "tool"})
    p.session2history["s1"] = [_user("earlier"), _assistant("reply"), _user("dangling")]
    ctx = p.build_conversation_context("now", "s1")
    idx = next(i for i, m in enumerate(ctx) if m.role == MessageRole.ASSISTANT and m.tool_calls)
    assert ctx[idx + 1].role == MessageRole.TOOL
    assert ctx[idx + 1].tool_call_id == ctx[idx].tool_calls[0].id
    assert ctx[-1].role == MessageRole.USER and ctx[-1].content == "now"
    # message immediately before the synthetic tool call is NOT a user turn
    assert ctx[idx - 1].role != MessageRole.USER


def test_extra_system_blocks_are_inserted():
    p = _ListRetriever(config={"_hits": [("ctx", 1.0)], "system_prompt": "Base."})
    extra = [AgentMessage(MessageRole.SYSTEM, "FOLDED SUMMARY")]
    ctx = p.assemble_context("q", "s1", "ctx", [], extra_system=extra)
    sys_text = " ".join(m.content for m in ctx if m.role == MessageRole.SYSTEM)
    assert "Base." in sys_text and "FOLDED SUMMARY" in sys_text
    assert ctx[-1].role == MessageRole.USER
