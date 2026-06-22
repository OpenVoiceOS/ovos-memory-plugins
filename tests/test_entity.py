"""Unit tests for EntityMemory — the LLM client is mocked (no network)."""
from unittest.mock import patch

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_memory_plugins.entity import EntityMemory


def _user(text): return AgentMessage(role=MessageRole.USER, content=text)
def _assistant(text): return AgentMessage(role=MessageRole.ASSISTANT, content=text)


def _make(**kwargs):
    cfg = {"backend": "memory", "model": "m", "api_url": "http://mock/v1", **kwargs}
    return EntityMemory(config=cfg)


def test_get_history_is_empty():
    assert _make().get_history("s1") == []


@patch("ovos_memory_plugins.entity.chat_complete", return_value="- User's name is Alice\n- Likes tea")
def test_extraction_stores_facts(mock_llm):
    p = _make()
    p.update_history([_user("I'm Alice and I love tea"), _assistant("Nice to meet you Alice!")], "s1")
    assert mock_llm.called
    assert p._load_facts("s1") == ["User's name is Alice", "Likes tea"]


@patch("ovos_memory_plugins.entity.chat_complete", return_value="NONE")
def test_none_yields_no_facts(mock_llm):
    p = _make()
    p.update_history([_user("hi"), _assistant("hello")], "s1")
    assert p._load_facts("s1") == []


@patch("ovos_memory_plugins.entity.chat_complete")
def test_facts_dedup_case_insensitive(mock_llm):
    p = _make()
    mock_llm.return_value = "- Likes tea"
    p.update_history([_user("tea?"), _assistant("ok")], "s1")
    mock_llm.return_value = "- likes TEA\n- Has a dog"
    p.update_history([_user("dog?"), _assistant("ok")], "s1")
    assert p._load_facts("s1") == ["Likes tea", "Has a dog"]


@patch("ovos_memory_plugins.entity.chat_complete")
def test_max_facts_caps_oldest_dropped(mock_llm):
    p = _make(max_facts=2)
    mock_llm.return_value = "- fact one\n- fact two\n- fact three"
    p.update_history([_user("x"), _assistant("y")], "s1")
    assert p._load_facts("s1") == ["fact two", "fact three"]


@patch("ovos_memory_plugins.entity.chat_complete", return_value="- User is a pilot")
def test_facts_injected_as_system_block(mock_llm):
    p = _make(system_prompt="You are helpful.")
    p.update_history([_user("I fly planes"), _assistant("cool")], "s1")
    ctx = p.build_conversation_context("what's my job?", "s1")
    sys_text = " ".join(m.content for m in ctx if m.role == MessageRole.SYSTEM)
    assert "User is a pilot" in sys_text
    assert ctx[-1].role == MessageRole.USER
    assert ctx[-1].content == "what's my job?"


def test_no_facts_no_block():
    p = _make(system_prompt="Base.")
    ctx = p.build_conversation_context("anything", "s1")
    assert [m.role for m in ctx] == [MessageRole.SYSTEM, MessageRole.USER]


@patch("ovos_memory_plugins.entity.chat_complete", side_effect=Exception("conn refused"))
def test_offline_extraction_no_op(mock_llm):
    p = _make()
    # extraction must not raise even when the endpoint is down
    p.update_history([_user("I'm Bob"), _assistant("hi Bob")], "s1")
    assert p._load_facts("s1") == []


@patch("ovos_memory_plugins.entity.chat_complete", return_value="- remembered fact")
def test_json_persistence(mock_llm, tmp_path):
    path = str(tmp_path / "entity.json")
    p1 = EntityMemory(config={"backend": "json", "db_path": path, "model": "m",
                              "api_url": "http://mock/v1"})
    p1.update_history([_user("note this"), _assistant("ok")], "s1")
    p2 = EntityMemory(config={"backend": "json", "db_path": path, "model": "m",
                              "api_url": "http://mock/v1"})
    assert "remembered fact" in p2._load_facts("s1")
