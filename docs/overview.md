# Memory plugins overview

A persona memory plugin is an `AgentContextManager`. The persona owns the chat
engine and tools. The memory plugin owns **conversation state and context
assembly**. It composes with any chat backend instead of generating answers itself.

## The contract

Every backend implements three methods (`ovos_plugin_manager.templates.agents`):

```python
get_history(session_id) -> List[AgentMessage]
update_history(new_messages, session_id) -> None
build_conversation_context(utterance, session_id) -> List[AgentMessage]
```

`build_conversation_context` returns the message list sent to the chat engine.
Two rules hold for every backend here:

- the **first** message MAY be a `system` message carrying `self.system_prompt`
- the **last** message is ALWAYS the current user utterance

The persona calls `build_conversation_context` before each turn and
`update_history` after each exchange.

## Inject modes (retrieval backends)

The retrieval backends (`local-rag`, `lexical`) and the `composite` share a
`BaseRetrievalMemory` that folds recalled context into the conversation via a
configurable `inject_mode`, supporting the full set:

| `inject_mode` | What it does | When to use |
|---|---|---|
| `system` (default) | Retrieved context goes in a **separate** `system` message. The persona's `system_prompt` stays its own message | Keeps the base prompt stable and cacheable. Safe default |
| `developer` | Same, but a `developer`-role message | Providers that distinguish developer from system instructions |
| `system_prompt` | Context folded into the persona's system prompt (one combined `system` message) | Backends that only honour a single system message |
| `user` | Context prepended to the final user message | Backends that ignore system/developer roles |
| `tool` | A synthetic assistant `tool_calls` turn + its `tool` result carry the context, just before the user turn | Tool-calling models. Presents recall as a search-tool result. Needs the `ovos-plugin-manager` TOOL contract |

## Retrieval knobs (retrieval backends)

- `max_num_results`: top-k documents per query.
- `min_score`: drop hits below this score (`null` keeps all). Scale is backend
  specific: `local-rag` is `1 - cosine_distance` (~0..1), `lexical` is `-bm25`.
- `query_mode`: `utterance` (default) or `history` (fold the last N user turns
  into the search query for follow-up questions).

## Choosing a backend

| | longterm | local-rag | lexical | recency | entity | composite |
|---|---|---|---|---|---|---|
| Entry point | `…-longterm` | `…-local-rag` | `…-lexical` | `…-recency` | `…-entity` | `…-composite` |
| Recall style | rolling **summary** | **semantic** top-k | **keyword** (BM25) | recent window | durable **facts** | **ensemble** of members |
| External service | chat endpoint | none | none | none | chat endpoint | members' |
| Extra deps | none | embeddings + vector DB | **none** (stdlib) | **none** | none | members' |
| Runs fully offline | if LLM is local | yes | yes | yes | if LLM is local | if members do |
| Persistence | JSON / SQLite | vector DB | SQLite | optional JSON | JSON | members' |
| Best for | gist of long chats | exact semantic recall | exact-term recall | plain short-term | user preferences | hybrid / combine all |

Rules of thumb:

- Private/offline assistant needing specifics → **local-rag** (semantic) and/or
  **lexical** (keywords). Combine them with **composite** for hybrid recall.
- Remember *who the user is* across sessions → **entity**.
- Just the last few turns → **recency**.
- A running gist of very long chats and you have an LLM endpoint → **longterm**.

A persona's `memory_module` selects exactly one backend, but that one may be
**`ovos-memory-plugin-composite`**, which loads and consolidates several of the
others. See [composite](./composite.md).

---
[Home](README.md) · [Long-term →](longterm.md)
