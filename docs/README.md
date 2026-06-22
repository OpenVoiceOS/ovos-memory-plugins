# OVOS memory plugins — documentation

`ovos-memory-plugins` ships local-first `opm.agents.memory` backends. Each
implements the `AgentContextManager` contract from `ovos-plugin-manager` and is
selected by a persona's `memory_module` key.

- [Overview & how to choose](./overview.md) — the contract, inject modes, and a
  comparison table.
- [`ovos-memory-plugin-longterm`](./longterm.md) — rolling LLM summarization.
- [`ovos-memory-plugin-local-rag`](./local-rag.md) — fully-local in-process semantic RAG.
- [`ovos-memory-plugin-lexical`](./lexical.md) — SQLite FTS5 keyword recall (zero deps).
- [`ovos-memory-plugin-recency`](./recency.md) — sliding short-term buffer (zero deps).
- [`ovos-memory-plugin-entity`](./entity.md) — durable user-fact extraction.
- [`ovos-memory-plugin-composite`](./composite.md) — ensemble orchestrator that
  loads and consolidates the others (hybrid recall via fusion).

Runnable examples and ready persona JSON configs live in [`../examples/`](../examples/).
