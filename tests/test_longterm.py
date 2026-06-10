"""Unit tests for LongTermMemory plugin — all LLM/HTTP calls are mocked."""
import json
import os
import sqlite3
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_memory_plugins.longterm import (
    LongTermMemory, _JsonStore, _SqliteStore, _messages_to_text, _chat_complete,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _user(text): return AgentMessage(role=MessageRole.USER, content=text)
def _assistant(text): return AgentMessage(role=MessageRole.ASSISTANT, content=text)


def _make_plugin(tmp_path, backend="json", summarize_every=3, **kwargs):
    ext = ".db" if backend == "sqlite" else ".json"
    db_path = str(tmp_path / f"mem{ext}")
    cfg = {
        "api_url": "http://mock/v1",
        "model": "mock-model",
        "backend": backend,
        "db_path": db_path,
        "summarize_every": summarize_every,
        "max_summary_tokens": 64,
        "recent_window": 2,
        **kwargs,
    }
    return LongTermMemory(config=cfg)


# ---------------------------------------------------------------------------
# _messages_to_text
# ---------------------------------------------------------------------------

def test_messages_to_text():
    msgs = [_user("hello"), _assistant("hi")]
    text = _messages_to_text(msgs)
    assert "user" in text
    assert "hello" in text
    assert "assistant" in text
    assert "hi" in text


# ---------------------------------------------------------------------------
# _JsonStore
# ---------------------------------------------------------------------------

def test_json_store_roundtrip(tmp_path):
    store = _JsonStore(str(tmp_path / "mem.json"))
    record = {"summary": "test", "recent": [], "exchange_count": 2, "updated_at": 1.0}
    store.save("sess1", record)
    loaded = store.load("sess1")
    assert loaded["summary"] == "test"
    assert loaded["exchange_count"] == 2


def test_json_store_missing_session(tmp_path):
    store = _JsonStore(str(tmp_path / "mem.json"))
    result = store.load("nonexistent")
    assert result == {}


def test_json_store_multiple_sessions(tmp_path):
    store = _JsonStore(str(tmp_path / "mem.json"))
    store.save("s1", {"summary": "A", "recent": [], "exchange_count": 1})
    store.save("s2", {"summary": "B", "recent": [], "exchange_count": 5})
    assert store.load("s1")["summary"] == "A"
    assert store.load("s2")["exchange_count"] == 5


# ---------------------------------------------------------------------------
# _SqliteStore
# ---------------------------------------------------------------------------

def test_sqlite_store_roundtrip(tmp_path):
    store = _SqliteStore(str(tmp_path / "mem.db"))
    record = {
        "summary": "sqlite test",
        "recent": [{"role": "user", "content": "hi"}],
        "exchange_count": 3,
        "updated_at": 100.0,
    }
    store.save("sess1", record)
    loaded = store.load("sess1")
    assert loaded["summary"] == "sqlite test"
    assert loaded["exchange_count"] == 3
    assert loaded["recent"][0]["content"] == "hi"


def test_sqlite_store_upsert(tmp_path):
    store = _SqliteStore(str(tmp_path / "mem.db"))
    store.save("s1", {"summary": "old", "recent": [], "exchange_count": 1})
    store.save("s1", {"summary": "new", "recent": [], "exchange_count": 2})
    assert store.load("s1")["summary"] == "new"


# ---------------------------------------------------------------------------
# LongTermMemory.get_history / update_history
# ---------------------------------------------------------------------------

def test_get_history_empty(tmp_path):
    plugin = _make_plugin(tmp_path)
    assert plugin.get_history("s1") == []


def test_update_history_basic(tmp_path):
    plugin = _make_plugin(tmp_path)
    plugin.update_history([_user("hi"), _assistant("hello")], "s1")
    hist = plugin.get_history("s1")
    assert len(hist) == 2


def test_update_history_drops_hanging_user(tmp_path):
    plugin = _make_plugin(tmp_path)
    plugin.update_history([_user("first")], "s1")
    plugin.update_history([_user("second")], "s1")
    hist = plugin.get_history("s1")
    # The first hanging user message should be dropped
    assert any(m.content == "second" for m in hist)
    assert all(m.content != "first" for m in hist)


def test_update_history_merges_consecutive_assistant(tmp_path):
    plugin = _make_plugin(tmp_path)
    plugin.update_history([_assistant("part1")], "s1")
    plugin.update_history([_assistant("part2")], "s1")
    hist = plugin.get_history("s1")
    assert len(hist) == 1
    assert "part1" in hist[0].content
    assert "part2" in hist[0].content


# ---------------------------------------------------------------------------
# Summarization triggering
# ---------------------------------------------------------------------------

@patch("ovos_memory_plugins.longterm._chat_complete", return_value="Summary text")
def test_summarization_triggered_at_threshold(mock_llm, tmp_path):
    """After summarize_every=2 exchanges the LLM should be called once."""
    plugin = _make_plugin(tmp_path, summarize_every=2, recent_window=1)
    for i in range(2):
        plugin.update_history([_user(f"q{i}"), _assistant(f"a{i}")], "s1")
    assert mock_llm.called
    assert plugin._cache["s1"]["summary"] == "Summary text"
    assert plugin._cache["s1"]["exchange_count"] == 0


@patch("ovos_memory_plugins.longterm._chat_complete", return_value="Summary X")
def test_summarization_not_triggered_below_threshold(mock_llm, tmp_path):
    plugin = _make_plugin(tmp_path, summarize_every=5, recent_window=2)
    for i in range(3):
        plugin.update_history([_user(f"q{i}"), _assistant(f"a{i}")], "s1")
    assert not mock_llm.called


@patch("ovos_memory_plugins.longterm._chat_complete", return_value="Rolling summary")
def test_rolling_summary_accumulates(mock_llm, tmp_path):
    """Second summarization should include prior summary in prompt."""
    plugin = _make_plugin(tmp_path, summarize_every=2, recent_window=0)
    for i in range(4):
        plugin.update_history([_user(f"q{i}"), _assistant(f"a{i}")], "s1")
    # LLM called at 2 exchanges and again at 4 exchanges
    assert mock_llm.call_count == 2
    # Second call should include "Previous summary"
    second_call_prompt = mock_llm.call_args_list[1][1]["messages"][0]["content"]
    assert "Previous summary" in second_call_prompt


# ---------------------------------------------------------------------------
# build_conversation_context
# ---------------------------------------------------------------------------

@patch("ovos_memory_plugins.longterm._chat_complete", return_value="Summ")
def test_build_context_includes_summary(mock_llm, tmp_path):
    plugin = _make_plugin(tmp_path, summarize_every=2, recent_window=1,
                          system_prompt="You are a bot")
    for i in range(2):
        plugin.update_history([_user(f"q{i}"), _assistant(f"a{i}")], "s1")
    ctx = plugin.build_conversation_context("new question", "s1")
    # First message must be system
    assert ctx[0].role == MessageRole.SYSTEM
    assert "You are a bot" in ctx[0].content
    assert "Summ" in ctx[0].content  # summary embedded
    # Last message is the current user utterance
    assert ctx[-1].role == MessageRole.USER
    assert ctx[-1].content == "new question"


def test_build_context_no_history(tmp_path):
    plugin = _make_plugin(tmp_path)
    ctx = plugin.build_conversation_context("hello", "s1")
    assert ctx[-1].role == MessageRole.USER
    assert ctx[-1].content == "hello"


def test_build_context_no_trailing_user_from_history(tmp_path):
    """History ending in USER should be pruned from context."""
    plugin = _make_plugin(tmp_path, summarize_every=10)
    plugin.update_history([_user("q1")], "s1")
    ctx = plugin.build_conversation_context("q2", "s1")
    # Only the new utterance should appear as USER
    user_msgs = [m for m in ctx if m.role == MessageRole.USER]
    assert len(user_msgs) == 1
    assert user_msgs[0].content == "q2"


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------

@patch("ovos_memory_plugins.longterm._chat_complete", return_value="Persisted summary")
def test_persistence_json_roundtrip(mock_llm, tmp_path):
    # recent_window=0 so every message is summarized (nothing kept verbatim)
    plugin = _make_plugin(tmp_path, summarize_every=2, backend="json", recent_window=0)
    for i in range(2):
        plugin.update_history([_user(f"q{i}"), _assistant(f"a{i}")], "s1")
    assert mock_llm.called, "Summarization should have been triggered"
    # Force evict from cache, reload from disk
    del plugin._cache["s1"]
    plugin._load_session("s1")
    assert plugin._cache["s1"]["summary"] == "Persisted summary"


@patch("ovos_memory_plugins.longterm._chat_complete", return_value="Persisted sqlite")
def test_persistence_sqlite_roundtrip(mock_llm, tmp_path):
    plugin = _make_plugin(tmp_path, summarize_every=2, backend="sqlite", recent_window=0)
    for i in range(2):
        plugin.update_history([_user(f"q{i}"), _assistant(f"a{i}")], "s1")
    assert mock_llm.called, "Summarization should have been triggered"
    del plugin._cache["s1"]
    plugin._load_session("s1")
    assert plugin._cache["s1"]["summary"] == "Persisted sqlite"


# ---------------------------------------------------------------------------
# Error resilience
# ---------------------------------------------------------------------------

@patch("ovos_memory_plugins.longterm._chat_complete", side_effect=Exception("LLM down"))
def test_summarization_error_does_not_crash(mock_llm, tmp_path):
    plugin = _make_plugin(tmp_path, summarize_every=2)
    for i in range(2):
        plugin.update_history([_user(f"q{i}"), _assistant(f"a{i}")], "s1")
    # Plugin should survive; summary stays empty
    assert plugin._cache["s1"]["summary"] == ""


# ---------------------------------------------------------------------------
# recent_window enforcement
# ---------------------------------------------------------------------------

@patch("ovos_memory_plugins.longterm._chat_complete", return_value="S")
def test_recent_window_enforced(mock_llm, tmp_path):
    """After summarization only recent_window exchanges should remain."""
    plugin = _make_plugin(tmp_path, summarize_every=3, recent_window=1)
    for i in range(3):
        plugin.update_history([_user(f"q{i}"), _assistant(f"a{i}")], "s1")
    # recent_window=1 → 2 messages kept
    assert len(plugin._cache["s1"]["recent"]) <= 2


# ---------------------------------------------------------------------------
# _chat_complete direct
# ---------------------------------------------------------------------------

@patch("ovos_memory_plugins.longterm.requests.post")
def test_chat_complete_returns_text(mock_post):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"choices": [{"message": {"content": "  answer  "}}]}
    mock_post.return_value = mock_resp
    result = _chat_complete("http://mock/v1", "m", [{"role": "user", "content": "hi"}])
    assert result == "answer"
    mock_resp.raise_for_status.assert_called_once()


# ---------------------------------------------------------------------------
# _JsonStore exception paths
# ---------------------------------------------------------------------------

def test_json_store_corrupted_file(tmp_path):
    """load() on a corrupted JSON file returns empty dict."""
    p = tmp_path / "bad.json"
    p.write_text("NOT JSON")
    store = _JsonStore.__new__(_JsonStore)
    store.path = p
    assert store.load("s1") == {}


def test_json_store_save_corrupted_then_recovers(tmp_path):
    """save() when the file is corrupt creates a fresh store."""
    p = tmp_path / "bad.json"
    p.write_text("NOT JSON")
    store = _JsonStore.__new__(_JsonStore)
    store.path = p
    store.save("s1", {"summary": "x", "recent": [], "exchange_count": 0})
    loaded = store.load("s1")
    assert loaded["summary"] == "x"


# ---------------------------------------------------------------------------
# _resolve_model auto-detect
# ---------------------------------------------------------------------------

@patch("ovos_memory_plugins.longterm.requests.get")
def test_resolve_model_auto_detect(mock_get, tmp_path):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": [{"id": "gemma-2b"}]}
    mock_get.return_value = mock_resp
    plugin = _make_plugin(tmp_path, model="")
    plugin.model = ""
    plugin._resolve_model()
    assert plugin.model == "gemma-2b"


@patch("ovos_memory_plugins.longterm.requests.get", side_effect=Exception("conn refused"))
def test_resolve_model_failure_is_silent(mock_get, tmp_path):
    plugin = _make_plugin(tmp_path, model="fallback")
    plugin.model = ""
    plugin._resolve_model()
    assert plugin.model == ""  # stays empty, no exception raised


# ---------------------------------------------------------------------------
# build_conversation_context — trailing USER pruning
# ---------------------------------------------------------------------------

def test_build_context_prunes_multiple_trailing_users(tmp_path):
    """Consecutive USER messages at end of recent history are all pruned."""
    plugin = _make_plugin(tmp_path, summarize_every=100)
    # Manually populate cache with two trailing USER messages
    plugin._cache["s1"] = {
        "summary": "",
        "recent": [_assistant("hello"), _user("q1"), _user("q2")],
        "exchange_count": 0,
    }
    ctx = plugin.build_conversation_context("final", "s1")
    user_msgs = [m for m in ctx if m.role == MessageRole.USER]
    # Only the new utterance should appear
    assert len(user_msgs) == 1
    assert user_msgs[0].content == "final"
