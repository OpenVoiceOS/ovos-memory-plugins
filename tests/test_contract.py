"""One script against every memory backend this package registers.

The AgentContextManager contract every backend implements: an exchange
written under one session id is read back in order under that id and
under no other; the context built for the next turn ends with that turn as
a user message; an unknown session yields only the new turn. Each backend
is constructed from its entry point with a configuration that keeps it
off the network and off disk outside the test's own directory.

Two behaviours differ by backend and are pinned rather than judged: the
entity store keeps facts, not turns, so its history is empty; and the
retrieval backends (lexical, local-rag, and a composite of them) recall a
document written under any session, while the turn stores (recency,
long-term) answer only for the session asked.
"""
from importlib.metadata import entry_points
from unittest.mock import patch

import pytest

from ovos_plugin_manager.templates.agents import AgentContextManager, AgentMessage, MessageRole

from tests.test_local_rag import FakeEmbedder, FakeEmbeddingsDB

KEEPS_NO_TURNS = {"ovos-memory-plugin-entity"}
RECALLS_ACROSS_SESSIONS = {"ovos-memory-plugin-lexical", "ovos-memory-plugin-local-rag",
                           "ovos-memory-plugin-composite"}

EXPECTED = {
    "ovos-memory-plugin-composite",
    "ovos-memory-plugin-entity",
    "ovos-memory-plugin-lexical",
    "ovos-memory-plugin-local-rag",
    "ovos-memory-plugin-longterm",
    "ovos-memory-plugin-recency",
}


def _entry_points():
    return {e.name: e for e in entry_points(group="opm.agents.memory")
            if e.dist is not None and e.dist.name == "ovos-memory-plugins"}


def test_every_backend_is_registered():
    assert set(_entry_points()) == EXPECTED


def _config(name, tmp_path):
    db = str(tmp_path / f"{name}.sqlite")
    return {
        "ovos-memory-plugin-recency": {"db_path": db},
        "ovos-memory-plugin-lexical": {"db_path": db},
        "ovos-memory-plugin-local-rag": {"_embedder": FakeEmbedder(), "_db": FakeEmbeddingsDB(),
                                         "collection": "contract"},
        "ovos-memory-plugin-entity": {"backend": "memory", "db_path": str(tmp_path / "entity.json"), "model": "m", "api_url": "http://mock/v1"},
        "ovos-memory-plugin-longterm": {"backend": "json", "db_path": str(tmp_path / "lt.json"),
                                        "model": "m", "api_url": "http://mock/v1",
                                        "summarize_every": 100},
        "ovos-memory-plugin-composite": {"members": [
            {"module": "ovos-memory-plugin-recency", "config": {"db_path": db}},
            {"module": "ovos-memory-plugin-lexical", "config": {"db_path": str(tmp_path / "lex.sqlite")}},
        ]},
    }[name]


@pytest.fixture(params=sorted(EXPECTED))
def backend(request, tmp_path):
    name = request.param
    cls = _entry_points()[name].load()
    with patch("ovos_memory_plugins.entity.chat_complete", return_value="NONE"), \
            patch("ovos_memory_plugins.longterm.chat_complete", return_value="summary"):
        plugin = cls(config=_config(name, tmp_path))
        assert isinstance(plugin, AgentContextManager), name
        yield plugin


def _user(t):
    return AgentMessage(role=MessageRole.USER, content=t)


def _assistant(t):
    return AgentMessage(role=MessageRole.ASSISTANT, content=t)


def test_history_is_per_session_and_ordered(backend, request):
    backend.update_history([_user("my name is Quillanova"), _assistant("hello Quillanova")], "s1")
    backend.update_history([_user("second question"), _assistant("second answer")], "s1")
    contents = [(m.role, m.content) for m in backend.get_history("s1")]
    expected = [] if request.node.callspec.id in KEEPS_NO_TURNS else [
        (MessageRole.USER, "my name is Quillanova"), (MessageRole.ASSISTANT, "hello Quillanova"),
        (MessageRole.USER, "second question"), (MessageRole.ASSISTANT, "second answer"),
    ]
    assert contents == expected
    assert backend.get_history("s2") == []


def test_context_ends_with_the_new_user_turn(backend):
    backend.update_history([_user("my name is Quillanova"), _assistant("hello Quillanova")], "s1")
    context = backend.build_conversation_context("what is my name", "s1")
    assert context[-1].role == MessageRole.USER
    assert context[-1].content == "what is my name"
    assert sum(1 for m in context if m.content == "what is my name") == 1


def test_unknown_session_context_is_only_the_new_turn(backend):
    context = backend.build_conversation_context("first words", "never-seen")
    assert [m.content for m in context if m.role == MessageRole.USER] == ["first words"]
    assert not [m for m in context if m.role == MessageRole.ASSISTANT]


def test_recall_scope_across_sessions(backend, request):
    backend.update_history([_user("the secret word is marmalade"), _assistant("noted")], "s1")
    context = backend.build_conversation_context("what is the secret word", "s2")
    recalled = any("marmalade" in (m.content or "") for m in context)
    if request.node.callspec.id in RECALLS_ACROSS_SESSIONS:
        assert recalled, "retrieval backends recall documents from every session"
    elif request.node.callspec.id in KEEPS_NO_TURNS:
        assert not [m for m in context if m.role == MessageRole.ASSISTANT]
    else:
        assert not recalled, "turn stores answer only for the session asked"
