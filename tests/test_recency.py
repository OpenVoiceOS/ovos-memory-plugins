"""Unit tests for RecencyMemory (in-memory + optional JSON persistence)."""
import time
from pathlib import Path

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_memory_plugins.recency import RecencyMemory


def _user(text): return AgentMessage(role=MessageRole.USER, content=text)
def _assistant(text): return AgentMessage(role=MessageRole.ASSISTANT, content=text)


def _ingest(p, sid, exchanges):
    for u, a in exchanges:
        p.update_history([_user(u), _assistant(a)], sid)


def test_get_history_empty():
    assert RecencyMemory().get_history("s1") == []


def test_window_caps_by_count():
    p = RecencyMemory(config={"max_history": 2})
    _ingest(p, "s1", [("a", "1"), ("b", "2"), ("c", "3")])
    hist = p.get_history("s1")
    assert len(hist) == 2
    assert hist[-1].content == "3"


def test_unbounded_when_zero():
    p = RecencyMemory(config={"max_history": 0})
    _ingest(p, "s1", [("a", "1"), ("b", "2"), ("c", "3")])
    assert len(p.get_history("s1")) == 6


def test_age_prunes_old_messages():
    p = RecencyMemory(config={"max_history": 0, "max_age": 100})
    # inject an old message by back-dating its timestamp
    p._buffers["s1"] = [(time.time() - 1000, _user("ancient"))]
    p.update_history([_user("fresh"), _assistant("now")], "s1")
    contents = [m.content for m in p.get_history("s1")]
    assert "ancient" not in contents
    assert "fresh" in contents and "now" in contents


def test_context_contract_last_is_user():
    p = RecencyMemory(config={"system_prompt": "You are helpful."})
    _ingest(p, "s1", [("hi", "hello")])
    ctx = p.build_conversation_context("next?", "s1")
    assert ctx[0].role == MessageRole.SYSTEM
    assert ctx[-1].role == MessageRole.USER
    assert ctx[-1].content == "next?"


def test_context_prunes_trailing_user():
    p = RecencyMemory()
    p.update_history([_user("dangling")], "s1")
    ctx = p.build_conversation_context("real question", "s1")
    users = [m for m in ctx if m.role == MessageRole.USER]
    assert len(users) == 1
    assert users[0].content == "real question"


def test_sessions_isolated():
    p = RecencyMemory()
    _ingest(p, "A", [("chocolate", "cacao")])
    _ingest(p, "B", [("rockets", "thrust")])
    a = " ".join(m.content for m in p.get_history("A"))
    assert "rockets" not in a


def test_json_persistence_roundtrip(tmp_path):
    path = str(tmp_path / "recency.json")
    p1 = RecencyMemory(config={"db_path": path})
    _ingest(p1, "s1", [("remember me", "ok")])
    p2 = RecencyMemory(config={"db_path": path})
    contents = [m.content for m in p2.get_history("s1")]
    assert "remember me" in contents and "ok" in contents
