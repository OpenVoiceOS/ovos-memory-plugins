# `ovos-memory-plugin-rag` — RAG over an OpenAI-compatible server

`RAGMemory` stores every exchange via an OpenAI-compatible files/embeddings/
vector-stores API and retrieves the top-k most similar prior turns before each
response. It targets a running RAG server (e.g.
[`ovos-persona-server`](https://github.com/OpenVoiceOS/ovos-persona-server)),
falling back to in-process cosine over locally cached embeddings when the
vector-store endpoints are unavailable.

Use it when a server already hosts the vector store — for instance to share one
managed store across multiple clients. For a self-contained, **fully-local**
deployment, prefer [local-rag](./local-rag.md), which does the same retrieval
in-process with no server.

## How it works

```
update_history(exchange)
  ├─ POST /v1/embeddings   (embed doc)
  ├─ POST /v1/files        (upload doc)
  └─ POST /v1/vector_stores/{collection}/files   (attach)

build_conversation_context(utterance)
  ├─ POST /v1/vector_stores/{collection}/search  (top-k by cosine)
  │        fallback: local in-process cosine if the server is unavailable
  └─ inject retrieved docs + recent history + [USER: utterance]
```

The endpoint contract follows
[ovos-persona-server PR #11 (feat/rag)](https://github.com/OpenVoiceOS/ovos-persona-server/pull/11).

## Configuration

| Key | Type | Default | Description |
|---|---|---|---|
| `api_url` | str | `http://192.168.1.200:8000/v1` | OpenAI-compatible server |
| `collection` | str | `ovos_memory` | vector store / collection name |
| `top_k` | int | `3` | max retrieved documents per query |
| `min_score` | float | `0.0` | minimum cosine similarity to inject a doc |
| `inject_as_system` | bool | `false` | bundle retrieved docs into one `system` message instead of individual `assistant` messages |
| `system_prompt` | str | `""` | persona system prompt |
| `request_timeout` | int | `30` | HTTP timeout (s) |

## Persona wiring

```json
{
  "name": "RagBot",
  "handlers": ["ovos-solver-openai-plugin"],
  "memory_module": "ovos-memory-plugin-rag",
  "ovos-memory-plugin-rag": {
    "api_url": "http://localhost:8337/v1",
    "collection": "ragbot_memory",
    "top_k": 4,
    "min_score": 0.3,
    "inject_as_system": true,
    "system_prompt": "You are a knowledgeable assistant."
  }
}
```
