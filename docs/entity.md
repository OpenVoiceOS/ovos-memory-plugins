# `ovos-memory-plugin-entity`: durable user facts

`EntityMemory` distills **durable** facts about the user from the conversation,
such as their name, preferences, relationships, constraints, and goals, and
re-injects these facts into every turn. Unlike a recency window, it recalls
facts no matter how long ago they were said. Unlike RAG, it needs no embeddings
or vector store.

It is "clever by chaining a prompt": after each exchange it asks a **local**
OpenAI-compatible endpoint to extract short fact lines, then merges them
(deduplicated) into a per-session fact store. Local-first, the same kind of endpoint
[`longterm`](./longterm.md) uses, never a hosted provider. If the endpoint is
unreachable, extraction simply no-ops and the turn proceeds.

## How it works

```
  update_history(exchange)
      └─ LLM: "extract durable facts about the USER … else NONE"
              → parse "- fact" lines → dedup/merge into the fact store

  build_conversation_context(utterance)
      → [system?] + [SYSTEM: "Known facts about the user: …"] + [USER: utterance]
```

## Configuration

| Key | Default | Description |
|---|---|---|
| `api_url` | `http://localhost:8000/v1` | OpenAI-compatible endpoint. |
| `model` | auto-detected | Model name. Pulled from `/models` when empty. |
| `max_facts` | `50` | Cap per session (oldest dropped past the cap). |
| `max_extract_tokens` | `128` | `max_tokens` for the extraction call. |
| `request_timeout` | `30` | HTTP timeout (seconds). |
| `backend` | `json` | `json` (persisted) or `memory` (ephemeral). |
| `db_path` | `~/.local/share/ovos/entity_memory.json` | JSON store path. |
| `extraction_prompt` | (built-in) | Override the extraction template (must contain `{exchange}`). |
| `facts_header` | `Known facts about the user:` | Heading for the injected block. |
| `system_prompt` | `""` | Persona system prompt. |

## Example

```json
{
  "memory_module": "ovos-memory-plugin-entity",
  "ovos-memory-plugin-entity": {
    "api_url": "http://localhost:8000/v1",
    "max_facts": 40,
    "system_prompt": "You are a helpful assistant."
  }
}
```

Pair it with a recall backend in a [composite](./composite.md) (e.g. entity facts
+ local-rag) so the model gets both stable user facts and relevant past detail.

---
[← Recency](recency.md) · [Home](README.md) · [Composite →](composite.md)
