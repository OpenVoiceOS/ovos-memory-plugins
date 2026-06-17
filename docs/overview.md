# Memory plugins overview

A persona memory plugin is an `AgentContextManager`. The persona owns the chat
engine and tools; the memory plugin owns **conversation state and context
assembly**, composing with any chat backend instead of generating answers itself.

## The contract

Every backend implements three methods (`ovos_plugin_manager.templates.agents`):

```python
get_history(session_id) -> List[AgentMessage]
update_history(new_messages, session_id) -> None
build_conversation_context(utterance, session_id) -> List[AgentMessage]
```

`build_conversation_context` returns the message list sent to the chat engine.
Two rules hold for every backend here:

- the **first** message MAY be a `system` message carrying `self.system_prompt`;
- the **last** message is ALWAYS the current user utterance.

The persona calls `build_conversation_context` before each turn and
`update_history` after each exchange.

## Inject modes (RAG backends)

Both RAG backends fold retrieved context into the conversation via a configurable
`inject_mode`. `LocalRAGMemory` supports the full set:

| `inject_mode` | What it does | When to use |
|---|---|---|
| `system` (default) | Retrieved context goes in a **separate** `system` message; the persona's `system_prompt` stays its own message | Keeps the base prompt stable/cacheable; safe default |
| `developer` | Same, but a `developer`-role message | Providers that distinguish developer from system instructions |
| `system_prompt` | Context folded into the persona's system prompt (one combined `system` message) | Backends that only honour a single system message |
| `user` | Context prepended to the final user message | Backends that ignore system/developer roles |
| `tool` | A synthetic assistant `tool_calls` turn + its `tool` result carry the context, just before the user turn | Tool-calling brains; presents recall as a search-tool result. Needs the `ovos-plugin-manager` TOOL contract |

## Retrieval knobs (RAG backends)

- `max_num_results` — top-k documents per query.
- `min_score` — drop hits below this score (`null` keeps all). For local RAG,
  `score = 1 - cosine_distance`.
- `query_mode` — `utterance` (default) or `history` (fold the last N user turns
  into the search query for follow-up questions).

## Choosing a backend

| | longterm | **local-rag** | http-rag |
|---|---|---|---|
| Entry point | `ovos-memory-plugin-longterm` | `ovos-memory-plugin-local-rag` | `ovos-memory-plugin-rag` |
| Recall style | rolling **summary** of old turns | **semantic** top-k retrieval | semantic top-k retrieval |
| External service | a chat/LLM endpoint | **none** (in-process) | an OpenAI-compatible RAG server |
| Runs fully offline | only if the LLM is local | **yes** | only if the server is local |
| Persistence | JSON / SQLite | local vector DB (e.g. chromadb) | server-side vector store |
| Cost per turn | one LLM call every N exchanges | one local embedding + a vector query | HTTP embed + search |
| Best for | keeping a compact gist of long chats | private/offline assistants needing exact recall | sharing a managed RAG server across clients |

Rule of thumb: for a **private/local/offline** assistant that needs to remember
specifics, use **local-rag**. For long chats where a running gist is enough and
you already have an LLM, use **longterm**. Use **http-rag** only when a
persona-server (or other OpenAI-compatible RAG server) already hosts the store.

The backends are not mutually exclusive at the framework level, but a persona's
`memory_module` selects exactly one.
