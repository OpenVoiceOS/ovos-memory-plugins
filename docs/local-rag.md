# `ovos-memory-plugin-local-rag` — fully-local in-process RAG

`LocalRAGMemory` runs Retrieval-Augmented Generation entirely in-process. It
loads an OVOS text-embeddings plugin and an `EmbeddingsDB` plugin directly — no
HTTP, no cloud key — so a private/offline assistant gets long-term semantic
recall with nothing else running.

```bash
pip install 'ovos-memory-plugins[local-rag]'
```

The `local-rag` extra pulls the default offline stack:
`ovos-gguf-embeddings-plugin` (labse gguf text embeddings) +
`ovos-chromadb-embeddings-plugin` (persistent local vector store).

## How it works

```
update_history(user, assistant)
    └─ embed("Q: <user>\nA: <assistant>")  →  db.add_embeddings(key, vec, {content, session_id})

build_conversation_context(utterance)
    ├─ q = utterance            (or recent user turns folded in, if query_mode="history")
    ├─ hits = db.query(embed(q), top_k=max_num_results)   # cosine distance
    ├─ keep hits with score >= min_score   (score = 1 - distance)
    └─ inject context per inject_mode, then recent history, then [USER: utterance]
```

Document ids are stable and deterministic (`"<session_id>_<n>"`, monotonic per
session) — no randomness, so storage and tests are reproducible. Tool-call ids
in `inject_mode="tool"` are likewise stable (`"memrag_<session_id>_<n>"`).

## Configuration

```json
{
  "memory_module": "ovos-memory-plugin-local-rag",
  "ovos-memory-plugin-local-rag": {
    "embeddings_plugin": "ovos-gguf-embeddings-plugin",
    "embeddings_config": {"model": "labse"},
    "embeddings_db_plugin": "ovos-chromadb-embeddings-plugin",
    "embeddings_db_config": {"path": "~/.local/share/ovos/local_rag_db"},
    "collection": "ovos_local_rag",

    "retrieval": {
      "max_num_results": 5,
      "min_score": null,
      "query_mode": "utterance",
      "query_history_turns": 3
    },

    "context": {
      "header": "Use the following recalled context to answer ...",
      "chunk_prefix": "- ",
      "chunk_separator": "\n\n",
      "include_sources": false,
      "tool_name": "search_memory"
    },

    "inject_mode": "system",
    "system_prompt": "You are a helpful assistant.",
    "max_history": 10
  }
}
```

| Key | Default | Notes |
|---|---|---|
| `embeddings_plugin` | `ovos-gguf-embeddings-plugin` | any `opm.embeddings.text` entry point |
| `embeddings_config` | `{}` | forwarded to the embeddings plugin constructor |
| `embeddings_db_plugin` | `ovos-chromadb-embeddings-plugin` | any `opm.embeddings` (`EmbeddingsDB`) entry point |
| `embeddings_db_config` | `{}` | forwarded to the DB constructor; chromadb takes `path`, qdrant/base take `config` |
| `collection` | `ovos_local_rag` | collection name (chromadb requires 3-512 chars from `[a-zA-Z0-9._-]`) |
| `retrieval.max_num_results` | `5` | top-k |
| `retrieval.min_score` | `null` | drop hits below this score; `null` keeps all |
| `retrieval.query_mode` | `utterance` | `history` folds recent user turns into the query |
| `retrieval.query_history_turns` | `3` | turns folded when `query_mode="history"` |
| `context.*` | — | rendering of the retrieved chunk block |
| `inject_mode` | `system` | see the [inject-modes table](./overview.md#inject-modes-rag-backends) |
| `system_prompt` | `""` | persona base prompt |
| `max_history` | `10` | recent verbatim messages retained per session |

## Inject modes

All five modes from the overview are supported. The `tool` mode emits a
synthetic assistant `tool_calls` turn (a `search_memory` call for the query)
followed by a `MessageRole.TOOL` message carrying the chunks, just before the
user utterance — the assistant-with-`tool_calls` turn precedes its `tool` result
(provider ordering invariant) and the user utterance stays last. It requires a
brain/contract with tool-call support; until that contract ships in a released
`ovos-plugin-manager`, CI installs it from the
`feat/agent-tool-calling` branch via `pre_install_pip`.

## Swapping the stack

Any embeddings + `EmbeddingsDB` pair works. For example, point at a shared
qdrant instance by setting `embeddings_db_plugin` to
`ovos-qdrant-embeddings-plugin` and supplying its `embeddings_db_config`. The
retrieval/inject logic is backend-agnostic.

## Scoring note

`EmbeddingsDB.query` returns a **distance** (lower = closer). For the cosine
space used by the default chromadb stack, the plugin converts it to a similarity
`score = 1 - distance` before applying `min_score`, so `min_score` is a
similarity threshold in `[0, 1]` (higher = stricter).
