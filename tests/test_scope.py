"""Recall scope: a retrieval backend recalls only the caller's own sessions under
``scope: session`` and every session under ``scope: global`` (the default)."""
import pytest

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_memory_plugins.lexical import LexicalMemory
from ovos_memory_plugins.local_rag import LocalRAGMemory
from ovos_memory_plugins.composite import CompositeMemory

from tests.test_local_rag import FakeEmbedder, FakeEmbeddingsDB


def _lexical(tmp_path, **cfg):
    return LexicalMemory(config={"db_path": str(tmp_path / "lex.sqlite"), **cfg})


def _local_rag(tmp_path, **cfg):
    return LocalRAGMemory(config={"_embedder": FakeEmbedder(), "_db": FakeEmbeddingsDB(),
                                  "collection": "scope", **cfg})


BACKENDS = {"lexical": _lexical, "local-rag": _local_rag}


def _seed(mem):
    mem.update_history([AgentMessage(MessageRole.USER, "the secret word is marmalade"),
                        AgentMessage(MessageRole.ASSISTANT, "noted")], "s1")
    mem.update_history([AgentMessage(MessageRole.USER, "my cat is called biscuit"),
                        AgentMessage(MessageRole.ASSISTANT, "noted")], "s2")


def _recalls(mem, session_id, word="marmalade"):
    ctx = mem.build_conversation_context("what is the secret word", session_id)
    return any(word in (m.content or "") for m in ctx)


@pytest.mark.parametrize("name", sorted(BACKENDS))
def test_session_scope_hides_other_callers(tmp_path, name):
    mem = BACKENDS[name](tmp_path, scope="session")
    _seed(mem)
    assert _recalls(mem, "s1"), "the caller's own turn is recalled"
    assert not _recalls(mem, "s2"), "another caller's turn is not"


@pytest.mark.parametrize("name", sorted(BACKENDS))
def test_global_scope_is_the_default_and_recalls_every_session(tmp_path, name):
    mem = BACKENDS[name](tmp_path)
    assert mem.scope == "global"
    _seed(mem)
    assert _recalls(mem, "s2")


def test_invalid_scope_raises(tmp_path):
    with pytest.raises(ValueError):
        _lexical(tmp_path, scope="everyone")


def test_composite_scope_reaches_members(tmp_path):
    comp = CompositeMemory(config={"scope": "session", "members": [
        {"module": "ovos-memory-plugin-lexical", "config": {"db_path": str(tmp_path / "a.sqlite")}},
        {"module": "ovos-memory-plugin-recency", "config": {"db_path": str(tmp_path / "b.sqlite")}},
    ]})
    _seed(comp)
    assert _recalls(comp, "s1")
    assert not _recalls(comp, "s2")
