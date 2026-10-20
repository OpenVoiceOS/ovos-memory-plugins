"""Unit tests for EntityMemory — the LLM client is mocked (no network)."""
import time

import pytest
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


@patch("ovos_memory_plugins.entity.chat_complete", return_value='["User\'s name is Alice", "Likes tea"]')
def test_extraction_stores_facts(mock_llm):
    p = _make()
    p.update_history([_user("I'm Alice and I love tea"), _assistant("Nice to meet you Alice!")], "s1")
    assert mock_llm.called
    assert p._load_facts("s1") == ["User's name is Alice", "Likes tea"]


@patch("ovos_memory_plugins.entity.chat_complete",
       return_value='```json\n["User\'s name is Alice", "Likes tea", "Has a dog"]\n```')
def test_extraction_parses_fenced_json(mock_llm):
    p = _make()
    p.update_history([_user("hi"), _assistant("hello")], "s1")
    assert p._load_facts("s1") == ["User's name is Alice", "Likes tea", "Has a dog"]


@patch("ovos_memory_plugins.entity.chat_complete", return_value="[]")
def test_empty_array_yields_no_facts(mock_llm):
    p = _make()
    p.update_history([_user("hi"), _assistant("hello")], "s1")
    assert p._load_facts("s1") == []


@patch("ovos_memory_plugins.entity.chat_complete")
def test_facts_dedup_case_insensitive(mock_llm):
    p = _make()
    mock_llm.return_value = '["Likes tea"]'
    p.update_history([_user("tea?"), _assistant("ok")], "s1")
    mock_llm.return_value = '["likes TEA", "Has a dog"]'
    p.update_history([_user("dog?"), _assistant("ok")], "s1")
    assert p._load_facts("s1") == ["Likes tea", "Has a dog"]


@patch("ovos_memory_plugins.entity.chat_complete")
def test_max_facts_caps_oldest_dropped(mock_llm):
    p = _make(max_facts=2)
    mock_llm.return_value = '["fact one", "fact two", "fact three"]'
    p.update_history([_user("x"), _assistant("y")], "s1")
    assert p._load_facts("s1") == ["fact two", "fact three"]


@patch("ovos_memory_plugins.entity.chat_complete", return_value='["User is a pilot"]')
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


@patch("ovos_memory_plugins.entity.chat_complete", return_value='["remembered fact"]')
def test_json_persistence(mock_llm, tmp_path):
    path = str(tmp_path / "entity.json")
    p1 = EntityMemory(config={"backend": "json", "db_path": path, "model": "m",
                              "api_url": "http://mock/v1"})
    p1.update_history([_user("note this"), _assistant("ok")], "s1")
    p2 = EntityMemory(config={"backend": "json", "db_path": path, "model": "m",
                              "api_url": "http://mock/v1"})
    assert "remembered fact" in p2._load_facts("s1")


NO_FACTS_REPLIES = [
    # the first two replies were recorded from qwen2.5:7b to the old prompt;
    # the Portuguese and Spanish replies are written by hand
    "There is no information about the user's preferences, relationships, location, "
    "constraints, or goals that can be extracted from this short exchange.\n\nNONE",
    "NONE\n\nThere are no durable facts about the USER worth remembering in this exchange.",
    "O utilizador não revelou nenhum facto duradouro.",
    "No hay hechos duraderos sobre el usuario en este intercambio.",
]


@pytest.mark.parametrize("reply", NO_FACTS_REPLIES)
def test_no_facts_prose_is_not_stored(reply):
    p = _make()
    with patch("ovos_memory_plugins.entity.chat_complete", return_value=reply):
        p.update_history([_user("What time is it?"), _assistant("It is 3pm.")], "s1")
    assert p._load_facts("s1") == []
    ctx = p.build_conversation_context("What is my cat called?", "s1")
    assert [m.role for m in ctx] == [MessageRole.USER]


def test_facts_mixed_with_commentary_keep_only_the_facts():
    reply = ('Here are the facts:\n["Name is Alice", "Lives in Lisbon"]\n'
             "Note: There are only two durable facts extracted from the conversation.")
    p = _make()
    with patch("ovos_memory_plugins.entity.chat_complete", return_value=reply):
        p.update_history([_user("I'm Alice from Lisbon"), _assistant("Hi Alice")], "s1")
    assert p._load_facts("s1") == ["Name is Alice", "Lives in Lisbon"]


def test_non_string_and_blank_items_are_dropped():
    p = _make()
    with patch("ovos_memory_plugins.entity.chat_complete",
               return_value='["Has a dog", "", "  ", 3, null, ["nested"], {"a": 1}]'):
        p.update_history([_user("x"), _assistant("y")], "s1")
    assert p._load_facts("s1") == ["Has a dog"]


def _stored(reply):
    p = _make()
    with patch("ovos_memory_plugins.entity.chat_complete", return_value=reply):
        p.update_history([_user("x"), _assistant("y")], "s1")
    return p._load_facts("s1")


def test_every_array_in_the_reply_is_read():
    # live qwen2.5:7b reply
    assert _stored('["My cat is called Pixel"]\n\n["I live in Lisbon"]') == [
        "My cat is called Pixel", "I live in Lisbon"]


def test_prose_array_before_the_facts_is_skipped():
    assert _stored('[1] see note ["Lives in Lisbon"]') == ["Lives in Lisbon"]


def test_items_without_letters_or_digits_are_dropped():
    # live qwen2.5:7b reply held "]" as an item
    assert _stored('["My cat is called Pixel", "]"]') == ["My cat is called Pixel"]


def test_repeated_facts_in_one_reply_are_kept_once_in_order():
    assert EntityMemory._parse_facts('["Has a dog", "Likes tea"]\n["Has a dog"]') == [
        "Has a dog", "Likes tea"]


@pytest.mark.parametrize("reply", ["[" * 100000, "[" * 100000 + "]" * 100000],
                         ids=["open-only", "open-and-close"])
def test_deeply_nested_brackets_do_not_raise(reply):
    assert EntityMemory._parse_facts(reply) == []
    assert _stored(reply) == []


def test_reply_without_array_logs_a_warning():
    with patch("ovos_memory_plugins.entity.LOG") as log:
        assert _stored("- Name is Alice\n- Lives in Lisbon") == []
    msg = log.warning.call_args.args[0]
    assert "no JSON array" in msg and "extraction_prompt" in msg
    assert log.warning.call_count == 1


def test_empty_array_and_empty_reply_do_not_warn():
    with patch("ovos_memory_plugins.entity.LOG") as log:
        assert _stored("[]") == []
        assert _stored("") == []
    log.warning.assert_not_called()


def test_array_after_a_long_run_of_brackets_is_read():
    assert EntityMemory._parse_facts("[" * 1000 + '["Has a dog"]') == ["Has a dog"]


def test_long_spaced_brackets_are_bounded_in_time():
    start = time.monotonic()
    assert EntityMemory._parse_facts("[ " * 50000) == []
    assert time.monotonic() - start < 2


def test_many_failed_brackets_stop_the_scan():
    reply = "[x] " * 1000 + '["Has a dog"]'
    assert EntityMemory._parse_facts(reply) == []
