# `ovos-memory-plugin-composite`: ensemble memory

`CompositeMemory` is a **pure orchestrator**: it loads several member memory
plugins by name (via `ovos-plugin-manager`) and consolidates them, so a persona's
single `memory_module` slot can combine many memories. It stores and retrieves
nothing itself. Every contribution comes from a member.

```
                       ┌──────────────────────────────────────────┐
 build_conversation_   │ CompositeMemory                          │
 context(utterance) ──▶│  retrievers ─ search() ─┐                │
                       │  (local-rag, lexical)   ├─▶ fuse (RRF) ──┼─▶ one context block
                       │  plain members ─────────┘                │   + folded summaries/facts
                       │  (longterm, entity, recency: system block)│   + primary history
                       └──────────────────────────────────────────┘   + [USER utterance]
```

## How it consolidates

- **Retriever members** (anything exposing `search() -> List[MemoryHit]`, e.g.
  `local-rag`, `lexical`) have their hits **fused** into one ranked, deduplicated
  list, rendered once and injected per the composite's own `inject_mode`.
- **Plain members** (`longterm`, `entity`, `recency`, …) contribute their leading
  `system`/`developer` block(s): a rolling summary, known facts, and so on. Their own
  history and user turn are ignored (the composite owns those).
- **History** comes from the designated `primary` member (default: the first).
- **`update_history` is written through to every member**, so each advances its
  own state.

Members that fail to load, or raise at runtime, are skipped. The composite keeps
working with whatever remains. With zero members it is a harmless passthrough.

## Configuration

```jsonc
{
  "memory_module": "ovos-memory-plugin-composite",
  "ovos-memory-plugin-composite": {
    "members": [
      {"module": "ovos-memory-plugin-local-rag", "weight": 1.0, "config": { /* its block */ }},
      {"module": "ovos-memory-plugin-lexical",   "weight": 1.0, "config": { /* its block */ }},
      {"module": "ovos-memory-plugin-entity",    "config": { /* its block */ }}
    ],
    "primary": "ovos-memory-plugin-local-rag",  // history source (default: first member)
    "fusion": "rrf",                            // rrf | weighted | merge | priority | interleave
    "rrf_k": 60,
    "max_num_results": 5,                        // final fused top-k
    "member_fetch_k": 10,                        // candidates pulled per member (default: 2×top-k)
    "dedup": true,
    "inject_mode": "system",                    // how the fused block is injected
    "system_prompt": "You are a helpful assistant."
  }
}
```

| Key | Default | Description |
|---|---|---|
| `members` | `[]` | Ordered list of `{module, config, weight}`. Each member gets **only** its own `config` block. |
| `primary` | first member | Member whose `get_history` provides the verbatim history. |
| `fusion` | `rrf` | Fusion strategy for retriever hits (see below). |
| `rrf_k` | `60` | RRF constant. |
| `max_num_results` | `5` | Size of the final fused list. |
| `member_fetch_k` | `2×max_num_results` | How many candidates to pull from each member before fusing. |
| `dedup` | `true` | Deduplicate by content (for `merge` / `interleave`). |
| `inject_mode` | `system` | How the fused context is injected (same modes as local-rag). |
| `system_prompt` | `""` | Persona system prompt. |

> Give each retriever member a **distinct** storage path/collection in its config
> so they don't collide, and leave `system_prompt` to the composite (member
> system prompts, if set, are folded in as extra system blocks).

## Fusion modes

| Mode | How | Notes |
|---|---|---|
| `rrf` (default) | `score(d) = Σ weight · 1/(k + rank)` | **Rank-based**, so immune to score-scale mismatch between members. The right default for hybrid recall. |
| `weighted` | weighted sum of per-list min-max-normalized scores | Assumes per-list scores are meaningful. |
| `merge` | union, dedup keeping the max score | Simplest. Scores across members are not strictly comparable. |
| `priority` | first member (in order) with any hits wins | Fallback chain (e.g. prefer semantic, fall back to lexical). |
| `interleave` | round-robin rank-0 of each, then rank-1, … | Rank-based and scale-free, but coarser than RRF. |

### Why RRF is the default

Different retrievers score on **incompatible scales**: semantic cosine similarity
is ~0..1, lexical BM25 is unbounded. Summing raw scores would let whichever scale
has the larger magnitude dominate. RRF throws magnitudes away and fuses on **rank
position** only, so a semantic top-1 and a lexical top-1 count equally. It is the
standard hybrid-search fusion and needs just one parameter (`k`).

## Example: hybrid (semantic + lexical) recall

See [`../examples/persona_composite.json`](../examples/persona_composite.json) and
[`../examples/demo_composite.py`](../examples/demo_composite.py), which fuses
`local-rag` (semantics) with `lexical` (keywords) under RRF, surfacing exchanges
that neither backend ranks first on its own.

---
[← Entity](entity.md) · [Home](README.md) · [Writing a backend →](writing-a-backend.md)
