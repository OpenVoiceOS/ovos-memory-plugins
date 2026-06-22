#!/usr/bin/env python3
"""Hybrid (semantic + lexical) recall via CompositeMemory — fully offline.

Fuses ``local-rag`` (semantic) and ``lexical`` (keyword/BM25) members and prints
the fused hits under each fusion mode, so you can see how rank fusion combines
retrievers that score on different scales.

This demo injects a toy in-process embedder/vector-store into the ``local-rag``
member (via its config), so it needs **no model download and no network** — it
runs anywhere. A real persona would use the gguf + chromadb stack instead; see
``persona_composite.json``.

    pip install ovos-memory-plugins
    python examples/demo_composite.py
"""
from typing import Dict

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_memory_plugins.composite import CompositeMemory

EXCHANGES = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Germany?", "The capital of Germany is Berlin."),
    ("How do you make pasta?", "Boil salted water and cook pasta for ten minutes."),
]
QUERY = "capital of France"


class ToyEmbedder:
    """Bag-of-words 26-dim embedder — no model, no download."""
    def __init__(self, config=None): self.config = config or {}

    def get_embeddings(self, text):
        vec = [0.0] * 26
        for ch in text.lower():
            if "a" <= ch <= "z":
                vec[ord(ch) - 97] += 1.0
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]


class ToyVectorDB:
    """In-memory EmbeddingsDB returning cosine distance."""
    def __init__(self, config=None):
        self.collections: Dict[str, Dict] = {}

    def create_collection(self, name, metadata=None): self.collections.setdefault(name, {})

    def add_embeddings(self, key, embedding, metadata=None, collection_name=None):
        self.collections.setdefault(collection_name, {})[key] = (list(embedding), metadata or {})
        return embedding

    @staticmethod
    def _cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    def query(self, embeddings, top_k=5, return_metadata=False, collection_name=None):
        items = self.collections.get(collection_name, {})
        scored = [(k, 1.0 - self._cos(embeddings, e), m) for k, (e, m) in items.items()]
        scored.sort(key=lambda x: x[1])
        scored = scored[:top_k]
        return scored if return_metadata else [(k, d) for k, d, _ in scored]


def build(fusion: str) -> CompositeMemory:
    comp = CompositeMemory(config={
        "members": [
            {"module": "ovos-memory-plugin-local-rag", "weight": 1.0,
             "config": {"_embedder": ToyEmbedder(), "_db": ToyVectorDB(),
                        "collection": "demo"}},
            {"module": "ovos-memory-plugin-lexical", "weight": 1.0,
             "config": {"db_path": ":memory:"}},
        ],
        "fusion": fusion,
        "max_num_results": 5,
    })
    for u, a in EXCHANGES:
        comp.update_history(
            [AgentMessage(role=MessageRole.USER, content=u),
             AgentMessage(role=MessageRole.ASSISTANT, content=a)], "demo")
    return comp


def _short(text):
    return text.replace("\n", " ")[:52]


def main():
    print(f"query: {QUERY!r}\n")

    # What each member returns on its own (semantic vs keyword may differ).
    comp = build("rrf")
    print("----- per-member recall -----")
    for name, inst, _w in comp.retrievers:
        hits = inst.search(QUERY, session_id="demo", top_k=5)
        print(f"  {name}:")
        for rank, h in enumerate(hits, 1):
            print(f"    {rank}. score={h.score:+.3f}  {_short(h.content)}")
    print()

    # How each fusion mode consolidates them (fusion_score shown when present).
    for fusion in ("rrf", "weighted", "merge", "priority", "interleave"):
        hits = build(fusion)._gather(QUERY, "demo")
        print(f"===== fusion = {fusion} =====")
        for h in hits:
            meta = h.metadata or {}
            fscore = meta.get("fusion_score")
            tag = f"fused={fscore:.4f}" if fscore is not None else f"score={h.score:+.3f}"
            print(f"  [{meta.get('retriever', '?'):>28}] {tag}  {_short(h.content)}")
        print()


if __name__ == "__main__":
    main()
