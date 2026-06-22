"""Unit tests for LocalRAGMemory.

These use deterministic in-memory stub backends (a fake text embedder + a fake
EmbeddingsDB) injected through the ``_embedder`` / ``_db`` config keys, so the
retrieval / injection / persistence logic is asserted without loading any model
or touching disk. Every supported ``inject_mode`` is covered. The real gguf +
chromadb stack is exercised separately in ``test_e2e_local_rag.py``.
"""
from typing import Any, Dict, List, Optional, Tuple

import pytest

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_memory_plugins.local_rag import LocalRAGMemory


# ---------------------------------------------------------------------------
# Deterministic stub backends
# ---------------------------------------------------------------------------

class FakeEmbedder:
    """Bag-of-words 26-dim embedder: vectors of texts sharing letters are close."""

    def __init__(self, config=None):
        self.config = config or {}

    def get_embeddings(self, text: str) -> List[float]:
        vec = [0.0] * 26
        for ch in text.lower():
            if "a" <= ch <= "z":
                vec[ord(ch) - 97] += 1.0
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]


class FakeEmbeddingsDB:
    """In-memory EmbeddingsDB returning cosine *distance* (1 - similarity)."""

    def __init__(self, config=None):
        self.config = config or {}
        self.collections: Dict[str, Dict[str, Tuple[List[float], Dict]]] = {}

    def create_collection(self, name: str, metadata: Optional[Dict] = None) -> Any:
        self.collections.setdefault(name, {})

    def add_embeddings(self, key: str, embedding: List[float],
                       metadata: Optional[Dict] = None,
                       collection_name: Optional[str] = None) -> Any:
        self.collections.setdefault(collection_name, {})[key] = (list(embedding), metadata or {})
        return embedding

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    def query(self, embeddings, top_k=5, return_metadata=False, collection_name=None):
        items = self.collections.get(collection_name, {})
        scored = [(k, 1.0 - self._cosine(embeddings, emb), meta)
                  for k, (emb, meta) in items.items()]
        scored.sort(key=lambda x: x[1])  # ascending distance
        scored = scored[:top_k]
        if return_metadata:
            return scored
        return [(k, d) for k, d, _ in scored]


def _make(**kwargs) -> LocalRAGMemory:
    cfg = {"_embedder": FakeEmbedder(), "_db": FakeEmbeddingsDB(),
           "collection": "ovos_unit_rag", **kwargs}
    return LocalRAGMemory(config=cfg)


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


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------

def test_invalid_inject_mode_raises():
    with pytest.raises(ValueError):
        _make(inject_mode="nope")


def test_defaults():
    p = _make()
    assert p.inject_mode == "system"
    assert p.collection == "ovos_unit_rag"
    assert p.max_num_results == 5


# ---------------------------------------------------------------------------
# history & persistence
# ---------------------------------------------------------------------------

def test_get_history_empty():
    assert _make().get_history("s1") == []


def test_update_history_stores_exchange_in_db():
    p = _make()
    _ingest(p, "s1", [GEO[0]])
    stored = p._db.collections["ovos_unit_rag"]
    assert len(stored) == 1
    (_emb, meta), = stored.values()
    assert "Paris" in meta["content"]
    assert meta["session_id"] == "s1"


def test_doc_ids_are_stable_and_deterministic():
    p = _make()
    _ingest(p, "s1", GEO)
    keys = sorted(p._db.collections["ovos_unit_rag"].keys())
    assert keys == ["s1_0", "s1_1", "s1_2"]


def test_max_history_truncates():
    p = _make(max_history=2)
    _ingest(p, "s1", GEO)
    assert len(p.get_history("s1")) == 2


def test_assistant_without_user_still_stored():
    p = _make()
    p.update_history([_assistant("standalone fact about turbines")], "s1")
    assert len(p._db.collections["ovos_unit_rag"]) == 1


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------

def test_retrieval_ranks_relevant_first():
    p = _make()
    _ingest(p, "s1", GEO)
    hits = p._search("capital of France Paris")
    assert hits, "expected at least one hit"
    assert "Paris" in hits[0][0]


def test_min_score_filters_everything():
    p = _make(retrieval={"min_score": 0.999})
    _ingest(p, "s1", GEO)
    assert p._search("totally unrelated zzz query") == []


def test_max_num_results_caps_hits():
    p = _make(retrieval={"max_num_results": 1})
    _ingest(p, "s1", GEO)
    assert len(p._search("capital city")) == 1


def test_query_mode_history_folds_prior_turns():
    p = _make(retrieval={"query_mode": "history", "query_history_turns": 2})
    p.session2history["s1"] = [_user("tell me about France"), _assistant("ok")]
    q = p._build_query("and its capital?", "s1")
    assert "France" in q and "capital" in q


# ---------------------------------------------------------------------------
# build_conversation_context — contract + every inject_mode
# ---------------------------------------------------------------------------

def test_context_last_message_is_user():
    p = _make(system_prompt="You are helpful.")
    _ingest(p, "s1", GEO)
    ctx = p.build_conversation_context("Tell me about Paris", "s1")
    assert ctx[-1].role == MessageRole.USER
    assert ctx[-1].content == "Tell me about Paris"
    assert ctx[0].role == MessageRole.SYSTEM
    assert ctx[0].content == "You are helpful."


def test_inject_system_adds_separate_system_block():
    p = _make(inject_mode="system", system_prompt="Base prompt.")
    _ingest(p, "s1", GEO)
    ctx = p.build_conversation_context("capital of France", "s1")
    system_msgs = [m for m in ctx if m.role == MessageRole.SYSTEM]
    assert any(m.content == "Base prompt." for m in system_msgs)
    assert any("Context:" in m.content and "Paris" in m.content for m in system_msgs)


def test_inject_developer_uses_developer_role():
    p = _make(inject_mode="developer")
    _ingest(p, "s1", GEO)
    ctx = p.build_conversation_context("capital of France", "s1")
    dev = [m for m in ctx if m.role == MessageRole.DEVELOPER]
    assert dev and "Paris" in dev[0].content


def test_inject_system_prompt_folds_into_one_message():
    p = _make(inject_mode="system_prompt", system_prompt="Persona.")
    _ingest(p, "s1", GEO)
    ctx = p.build_conversation_context("capital of France", "s1")
    system_msgs = [m for m in ctx if m.role == MessageRole.SYSTEM]
    assert len(system_msgs) == 1
    assert "Persona." in system_msgs[0].content
    assert "Paris" in system_msgs[0].content


def test_inject_user_prepends_context_to_utterance():
    p = _make(inject_mode="user")
    _ingest(p, "s1", GEO)
    ctx = p.build_conversation_context("capital of France", "s1")
    last = ctx[-1]
    assert last.role == MessageRole.USER
    assert "Context:" in last.content
    assert "capital of France" in last.content


def test_inject_tool_emits_toolcall_then_tool_result():
    p = _make(inject_mode="tool", tool_name="search_memory")
    _ingest(p, "s1", GEO)
    ctx = p.build_conversation_context("capital of France", "s1")
    # find the assistant tool_calls turn immediately followed by its TOOL result
    idx = next(i for i, m in enumerate(ctx)
               if m.role == MessageRole.ASSISTANT and m.tool_calls)
    call = ctx[idx].tool_calls[0]
    assert call.name == "search_memory"
    assert call.arguments == {"query": "capital of France"}
    tool_msg = ctx[idx + 1]
    assert tool_msg.role == MessageRole.TOOL
    assert tool_msg.tool_call_id == call.id
    assert "Paris" in tool_msg.content
    assert ctx[-1].role == MessageRole.USER


def test_tool_call_ids_are_stable():
    p = _make(inject_mode="tool")
    _ingest(p, "s1", GEO)
    p.build_conversation_context("capital of France", "s1")
    p.build_conversation_context("capital of Germany", "s1")
    assert p._tool_call_counter["s1"] == 2  # 0, then 1 — deterministic


def test_no_hits_no_context_block():
    p = _make(inject_mode="system", system_prompt="Base.")
    # nothing ingested → no retrieval, only the base system prompt + user
    ctx = p.build_conversation_context("anything", "s1")
    assert [m.role for m in ctx] == [MessageRole.SYSTEM, MessageRole.USER]


def test_include_sources_prefixes_chunk_ids():
    p = _make(inject_mode="system", context={"include_sources": True})
    _ingest(p, "s1", GEO)
    ctx = p.build_conversation_context("capital of France", "s1")
    block = " ".join(m.content for m in ctx if m.role == MessageRole.SYSTEM)
    assert "[s1_" in block


def test_sessions_isolated():
    p = _make()
    _ingest(p, "A", [("chocolate factories", "Chocolate is made from cacao beans.")])
    _ingest(p, "B", [("rocket propulsion", "Rockets use combustion for thrust.")])
    ctx_a = p.build_conversation_context("more chocolate", "A")
    recent = " ".join(m.content for m in ctx_a
                      if m.role in (MessageRole.USER, MessageRole.ASSISTANT))
    assert "rocket" not in recent.lower()


# ---------------------------------------------------------------------------
# resilience
# ---------------------------------------------------------------------------

def test_search_failure_degrades_gracefully():
    class Boom(FakeEmbeddingsDB):
        def query(self, *a, **k):
            raise RuntimeError("db down")

    p = _make(_db=Boom(), system_prompt="Base.")
    _ingest(p, "s1", GEO)
    ctx = p.build_conversation_context("capital of France", "s1")
    # no crash; user utterance still last
    assert ctx[-1].role == MessageRole.USER
    assert ctx[-1].content == "capital of France"


def test_store_failure_does_not_break_history():
    class Boom(FakeEmbeddingsDB):
        def add_embeddings(self, *a, **k):
            raise RuntimeError("write failed")

    p = _make(_db=Boom())
    # update_history must not raise even if persistence fails
    _ingest(p, "s1", [GEO[0]])
    assert len(p.get_history("s1")) == 2
