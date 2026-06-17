# OVOS memory plugins — documentation

`ovos-memory-plugins` ships three `opm.agents.memory` backends. Each implements
the `AgentContextManager` contract from `ovos-plugin-manager` and is selected by
a persona's `memory_module` key.

- [Overview & how to choose](./overview.md) — the contract, inject modes, and a
  comparison table.
- [`ovos-memory-plugin-longterm`](./longterm.md) — rolling LLM summarization.
- [`ovos-memory-plugin-local-rag`](./local-rag.md) — fully-local in-process RAG.
- [`ovos-memory-plugin-rag`](./http-rag.md) — RAG over an OpenAI-compatible server.

Runnable examples and ready persona JSON configs live in [`../examples/`](../examples/).
