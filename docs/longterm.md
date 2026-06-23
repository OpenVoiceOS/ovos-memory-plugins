# `ovos-memory-plugin-longterm` — rolling summarization

`LongTermMemory` keeps a compact running summary of a long conversation. Every
`summarize_every` exchanges it sends the oldest turns to an OpenAI-compatible
chat endpoint, replaces them with the returned summary, and keeps only
`recent_window` exchanges verbatim. The summary plus the recent window are
persisted per session (JSON or SQLite).

Use it when a **gist** of the conversation is enough and you already have an LLM
endpoint — it trades exact recall for a bounded, cheap-to-carry context. For
exact recall of specific facts, prefer [local-rag](./local-rag.md).

## How it works

```
Session store (JSON or SQLite)
  rolling_summary  ← LLM summarize every N exchanges
  recent_window messages (verbatim)

build_conversation_context(utterance)
  [SYSTEM: system_prompt + rolling_summary]
  [... recent verbatim turns ...]
  [USER: utterance]
```

## Configuration

| Key | Type | Default | Description |
|---|---|---|---|
| `api_url` | str | `http://localhost:8000/v1` | OpenAI-compatible chat server |
| `model` | str | auto-detected | model for chat completions |
| `summarize_every` | int | `6` | exchanges accumulated before summarizing |
| `max_summary_tokens` | int | `256` | `max_tokens` for the summary request |
| `recent_window` | int | `4` | exchanges kept verbatim after each summary |
| `backend` | str | `json` | `json` or `sqlite` |
| `db_path` | str | `~/.local/share/ovos/longterm_memory.{json,db}` | persistence path |
| `system_prompt` | str | `""` | persona system prompt |
| `request_timeout` | int | `30` | HTTP timeout (s) |

## Persona wiring

```json
{
  "name": "MyAssistant",
  "handlers": ["ovos-solver-openai-plugin"],
  "memory_module": "ovos-memory-plugin-longterm",
  "ovos-memory-plugin-longterm": {
    "api_url": "http://localhost:8000/v1",
    "model": "mistral",
    "summarize_every": 8,
    "recent_window": 4,
    "backend": "sqlite",
    "db_path": "~/.local/share/ovos/assistant_memory.db",
    "system_prompt": "You are a helpful assistant."
  }
}
```
