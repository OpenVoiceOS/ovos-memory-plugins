"""Unit + hybrid tests for CompositeMemory (members loaded via a patched OPM loader)."""
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest

from ovos_plugin_manager.templates.agents import AgentContextManager, AgentMessage, MessageRole
from ovos_memory_plugins.common import MemoryHit, fuse_rrf
from ovos_memory_plugins.composite import CompositeMemory
from ovos_memory_plugins.local_rag import LocalRAGMemory
from ovos_memory_plugins.lexical import LexicalMemory
from ovos_memory_plugins.recency import RecencyMemory


def _user(text): return AgentMessage(role=MessageRole.USER, content=text)
def _assistant(text): return AgentMessage(role=MessageRole.ASSISTANT, content=text)


# --------------------------------------------------------------------------- stubs
class _Stub(AgentContextManager):
    def __init__(self, config=None):
        super().__init__(config)
        self.updates = []
        self._hist = list(self.config.get("_history", []))

    def get_history(self, session_id): return list(self._hist)

    def update_history(self, messages, session_id):
        self.updates.append((messages, session_id))

    def build_conversation_context(self, utterance, session_id):
        return [_user(utterance)]


class _StubRetriever(_Stub):
    def search(self, query, session_id=None, top_k=None):
        hits = [MemoryHit(content=c, source=f"d{i}", score=s)
                for i, (c, s) in enumerate(self.config.get("_hits", []))]
        return hits if top_k is None else hits[:top_k]


class _BoomRetriever(_Stub):
    def search(self, *a, **k):
        raise RuntimeError("boom")


class _PlainSummary(_Stub):
    def build_conversation_context(self, utterance, session_id):
        return [AgentMessage(MessageRole.SYSTEM, self.config.get("_summary", "summary")),
                _assistant("prior"), _user(utterance)]


class _FixedRetriever(_Stub):
    """Returns the exact MemoryHit objects from config['_objs'] (to test mutation)."""
    def search(self, query, session_id=None, top_k=None):
        return list(self.config.get("_objs", []))


_REGISTRY = {
    "retr-a": _StubRetriever, "retr-b": _StubRetriever, "retr-c": _StubRetriever,
    "boom": _BoomRetriever, "plain": _PlainSummary, "hist": _Stub,
    "fixed": _FixedRetriever, "recency": RecencyMemory,
    "local-rag": LocalRAGMemory, "lexical": LexicalMemory,
}


def _fake_loader(name):
    return _REGISTRY.get(name)


def _build(members, **kw) -> CompositeMemory:
    with patch("ovos_memory_plugins.composite.load_memory_plugin", side_effect=_fake_loader):
        return CompositeMemory(config={"members": members, **kw})


# --------------------------------------------------------------------------- fusion
def test_rrf_ignores_score_scale():
    # 'berlin' is rank-2 by tiny scores in A and rank-1 by huge scores in B;
    # appearing in both lists should make it win regardless of magnitude.
    members = [
        {"module": "retr-a", "config": {"_hits": [("paris", 0.9), ("berlin", 0.8)]}},
        {"module": "retr-b", "config": {"_hits": [("berlin", 50.0), ("pasta", 10.0)]}},
    ]
    comp = _build(members, fusion="rrf")
    fused = comp._gather("q", "s1")
    contents = [h.content for h in fused]
    assert contents[0] == "berlin"
    assert set(contents) == {"berlin", "paris", "pasta"}


def test_priority_returns_first_nonempty():
    members = [
        {"module": "retr-a", "config": {"_hits": []}},
        {"module": "retr-b", "config": {"_hits": [("only-b", 1.0)]}},
        {"module": "retr-c", "config": {"_hits": [("c", 2.0)]}},
    ]
    comp = _build(members, fusion="priority")
    assert [h.content for h in comp._gather("q", "s1")] == ["only-b"]


def test_merge_dedups_keeping_max_score():
    members = [
        {"module": "retr-a", "config": {"_hits": [("dup", 0.3), ("a", 0.2)]}},
        {"module": "retr-b", "config": {"_hits": [("dup", 0.9)]}},
    ]
    comp = _build(members, fusion="merge")
    fused = comp._gather("q", "s1")
    dup = [h for h in fused if h.content == "dup"][0]
    assert dup.score == 0.9
    assert fused[0].content == "dup"  # highest score first


def test_max_num_results_truncates_fused():
    members = [{"module": "retr-a", "config": {"_hits": [("a", 3), ("b", 2), ("c", 1)]}}]
    comp = _build(members, fusion="merge", max_num_results=2)
    assert len(comp._gather("q", "s1")) == 2


# --------------------------------------------------------------------------- orchestration
def test_update_history_written_through_to_all():
    members = [{"module": "retr-a", "config": {}}, {"module": "plain", "config": {}}]
    comp = _build(members)
    comp.update_history([_user("hi"), _assistant("yo")], "s1")
    for _name, inst, _w in comp.members:
        assert inst.updates, f"{_name} did not receive update_history"


def test_get_history_from_primary():
    members = [
        {"module": "retr-a", "config": {}},
        {"module": "hist", "config": {"_history": [_user("from primary")]}},
    ]
    comp = _build(members, primary="hist")
    hist = comp.get_history("s1")
    assert len(hist) == 1 and hist[0].content == "from primary"


def test_default_primary_is_first_member():
    members = [
        {"module": "hist", "config": {"_history": [_user("first")]}},
        {"module": "retr-a", "config": {}},
    ]
    comp = _build(members)
    assert comp.get_history("s1")[0].content == "first"


def test_unknown_member_is_skipped():
    members = [{"module": "does-not-exist"}, {"module": "retr-a", "config": {"_hits": [("a", 1)]}}]
    comp = _build(members)
    assert len(comp.members) == 1
    assert [h.content for h in comp._gather("q", "s1")] == ["a"]


def test_self_reference_is_skipped():
    members = [{"module": "ovos-memory-plugin-composite"}, {"module": "retr-a"}]
    comp = _build(members)
    assert [n for n, _i, _w in comp.members] == ["retr-a"]


def test_member_search_raising_is_handled():
    members = [
        {"module": "boom"},
        {"module": "retr-a", "config": {"_hits": [("survivor", 1.0)]}},
    ]
    comp = _build(members, fusion="rrf")
    ctx = comp.build_conversation_context("q", "s1")
    assert ctx[-1].role == MessageRole.USER
    assert any("survivor" in m.content for m in ctx if m.role == MessageRole.SYSTEM)


def test_plain_member_system_block_is_folded():
    members = [
        {"module": "retr-a", "config": {"_hits": [("ctx", 1.0)]}},
        {"module": "plain", "config": {"_summary": "ROLLING SUMMARY"}},
    ]
    comp = _build(members, system_prompt="Base.")
    ctx = comp.build_conversation_context("q", "s1")
    sys_text = " ".join(m.content for m in ctx if m.role == MessageRole.SYSTEM)
    assert "ROLLING SUMMARY" in sys_text
    assert "Base." in sys_text
    # the plain member's own history/user turn must NOT leak in
    assert sum(1 for m in ctx if m.role == MessageRole.USER) == 1
    assert ctx[-1].content == "q"


def test_no_members_is_passthrough():
    comp = _build([], system_prompt="Base.")
    ctx = comp.build_conversation_context("hello", "s1")
    assert [m.role for m in ctx] == [MessageRole.SYSTEM, MessageRole.USER]
    assert ctx[-1].content == "hello"


@pytest.mark.parametrize("mode", ["system", "developer", "system_prompt", "user", "tool"])
def test_contract_last_is_user_all_inject_modes(mode):
    members = [{"module": "retr-a", "config": {"_hits": [("recalled fact", 1.0)]}}]
    comp = _build(members, inject_mode=mode)
    ctx = comp.build_conversation_context("question", "s1")
    assert ctx[-1].role == MessageRole.USER


def test_invalid_fusion_raises():
    with pytest.raises(ValueError):
        _build([{"module": "retr-a"}], fusion="nope")


def test_fusion_does_not_mutate_member_hits():
    # a member may cache and reuse MemoryHit objects; fusion/tagging must not touch them
    obj = MemoryHit(content="x", source="d0", score=1.0)
    comp = _build([{"module": "fixed", "config": {"_objs": [obj]}}], fusion="rrf")
    fused = comp._gather("q", "s1")
    assert obj.metadata == {}                       # original untouched
    assert fused[0].metadata.get("retriever") == "fixed"  # the copy is tagged
    assert "fusion_score" in fused[0].metadata


def test_tool_mode_with_dangling_user_primary_is_well_formed():
    # recency primary whose window ends in an unanswered user turn + tool inject:
    # the synthetic tool exchange must stay valid and the utterance stay last
    members = [
        {"module": "recency"},
        {"module": "retr-a", "config": {"_hits": [("ctx", 1.0)]}},
    ]
    comp = _build(members, primary="recency", inject_mode="tool")
    comp.update_history([_user("earlier"), _assistant("reply")], "s1")
    comp.update_history([_user("dangling")], "s1")   # no assistant reply yet
    ctx = comp.build_conversation_context("now", "s1")
    assert all(m.content != "dangling" for m in ctx)
    assert ctx[-1].role == MessageRole.USER and ctx[-1].content == "now"
    idx = next(i for i, m in enumerate(ctx) if m.role == MessageRole.ASSISTANT and m.tool_calls)
    assert ctx[idx + 1].role == MessageRole.TOOL
    assert ctx[idx - 1].role != MessageRole.USER     # tool-call replies to an assistant turn, not a user


# --------------------------------------------------------------------------- hybrid e2e
class FakeEmbedder:
    """Bag-of-words 26-dim embedder (shared-letters → close vectors)."""
    def __init__(self, config=None): self.config = config or {}

    def get_embeddings(self, text):
        vec = [0.0] * 26
        for ch in text.lower():
            if "a" <= ch <= "z":
                vec[ord(ch) - 97] += 1.0
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]


class FakeEmbeddingsDB:
    def __init__(self, config=None):
        self.config = config or {}
        self.collections: Dict[str, Dict] = {}

    def create_collection(self, name, metadata=None): self.collections.setdefault(name, {})

    def add_embeddings(self, key, embedding, metadata=None, collection_name=None):
        self.collections.setdefault(collection_name, {})[key] = (list(embedding), metadata or {})
        return embedding

    @staticmethod
    def _cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    def query(self, embeddings, top_k=5, return_metadata=False, collection_name=None):
        items = self.collections.get(collection_name, {})
        scored = [(k, 1.0 - self._cosine(embeddings, emb), meta) for k, (emb, meta) in items.items()]
        scored.sort(key=lambda x: x[1])
        scored = scored[:top_k]
        return scored if return_metadata else [(k, d) for k, d, _ in scored]


GEO = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Germany?", "The capital of Germany is Berlin."),
    ("How do you make pasta?", "Boil salted water and cook pasta for ten minutes."),
]


def test_hybrid_semantic_plus_lexical_fusion():
    members = [
        {"module": "local-rag", "weight": 1.0,
         "config": {"_embedder": FakeEmbedder(), "_db": FakeEmbeddingsDB(), "collection": "c"}},
        {"module": "lexical", "weight": 1.0, "config": {"db_path": ":memory:"}},
    ]
    comp = _build(members, fusion="rrf")
    for u, a in GEO:
        comp.update_history([_user(u), _assistant(a)], "s1")

    fused = comp._gather("capital of France Paris", "s1")
    assert fused, "hybrid fusion returned nothing"
    assert any("Paris" in h.content for h in fused)

    ctx = comp.build_conversation_context("capital of France", "s1")
    sys_text = " ".join(m.content for m in ctx if m.role == MessageRole.SYSTEM)
    assert "Paris" in sys_text
    assert ctx[-1].role == MessageRole.USER
