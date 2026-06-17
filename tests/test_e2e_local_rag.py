"""End-to-end test for LocalRAGMemory against the real local stack.

Exercises the fully-offline RAG path with the actual OVOS plugins:
``ovos-gguf-embeddings-plugin`` (text embeddings) + ``ovos-chromadb-embeddings-plugin``
(EmbeddingsDB). No HTTP, no cloud key — everything runs in-process.

These plugins are declared in the ``test`` extra, so the suite runs them for
real (no skips). The first run downloads the gguf model into the shared HF
cache; subsequent runs are fast.

Run with:  pytest tests/test_e2e_local_rag.py -v -s
"""
from __future__ import annotations

import tempfile

import pytest

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_memory_plugins.local_rag import LocalRAGMemory


def _user(text): return AgentMessage(role=MessageRole.USER, content=text)
def _assistant(text): return AgentMessage(role=MessageRole.ASSISTANT, content=text)


EXCHANGES = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Germany?", "The capital of Germany is Berlin."),
    ("How do you make pasta?", "Boil salted water and cook pasta for 8-10 minutes."),
    ("What is machine learning?", "Machine learning is a subset of AI where models learn from data."),
]


@pytest.fixture(scope="module")
def db_path():
    return tempfile.mkdtemp(prefix="ovos_local_rag_e2e_")


def _make(db_path, **kwargs):
    cfg = {
        "embeddings_plugin": "ovos-gguf-embeddings-plugin",
        "embeddings_config": {"model": "labse"},
        "embeddings_db_plugin": "ovos-chromadb-embeddings-plugin",
        "embeddings_db_config": {"path": db_path},
        "collection": "ovos_local_rag_e2e",
        **kwargs,
    }
    return LocalRAGMemory(config=cfg)


def _ingest(plugin, session_id):
    for u, a in EXCHANGES:
        plugin.update_history([_user(u), _assistant(a)], session_id)


def test_e2e_store_and_retrieve(db_path):
    """Full store → embed → retrieve → inject cycle with real plugins."""
    plugin = _make(db_path, system_prompt="You are a helpful assistant.")
    _ingest(plugin, "e2e-1")

    assert plugin.db.count_embeddings_in_collection("ovos_local_rag_e2e") == len(EXCHANGES)

    ctx = plugin.build_conversation_context("Tell me about European capitals", "e2e-1")
    contents = " ".join(m.content for m in ctx)
    assert "Paris" in contents or "Berlin" in contents, \
        "expected a geography exchange to be recalled by semantic search"
    assert ctx[-1].role == MessageRole.USER
    assert "European capitals" in ctx[-1].content


def test_e2e_semantic_ranking_beats_keyword(db_path):
    """A paraphrase with no shared keywords should still retrieve the right doc."""
    plugin = _make(db_path)
    _ingest(plugin, "e2e-rank")
    hits = plugin._search("Which city is the French capital?")
    assert hits
    assert "Paris" in hits[0][0]


def test_e2e_min_score_filters(db_path):
    """An impossible min_score yields no retrieved context."""
    plugin = _make(db_path, retrieval={"min_score": 0.999})
    _ingest(plugin, "e2e-filter")
    assert plugin._search("quantum chromodynamics") == []


def test_e2e_inject_modes(db_path):
    """Every supported inject_mode produces a valid, contract-conforming context."""
    for mode in ("system", "developer", "system_prompt", "user", "tool"):
        plugin = _make(db_path, inject_mode=mode, system_prompt="Persona prompt.")
        _ingest(plugin, f"e2e-{mode}")
        ctx = plugin.build_conversation_context("capital of France", f"e2e-{mode}")
        assert ctx[-1].role == MessageRole.USER, f"{mode}: utterance must be last"
        if mode == "tool":
            tool_msgs = [m for m in ctx if m.role == MessageRole.TOOL]
            assert tool_msgs and "Paris" in tool_msgs[0].content
        elif mode == "user":
            assert "Context:" in ctx[-1].content
        elif mode == "developer":
            assert any(m.role == MessageRole.DEVELOPER for m in ctx)
        else:  # system / system_prompt
            assert any(m.role == MessageRole.SYSTEM and "Paris" in m.content for m in ctx)


def test_e2e_persistence_across_instances(db_path):
    """A fresh plugin pointed at the same path retrieves previously stored docs."""
    p1 = _make(db_path)
    _ingest(p1, "persist")
    # new instance, same on-disk chromadb path → recall survives the restart
    p2 = _make(db_path)
    hits = p2._search("capital of France")
    assert any("Paris" in c for c, _, _ in hits)
