# ovos-memory-plugins

Two [OVOS](https://github.com/OpenVoiceOS) `opm.agents.memory` plugins bundled in one package:

| Entry point | Class | Purpose |
|---|---|---|
| `ovos-memory-plugin-longterm` | `LongTermMemory` | Rolling summarization of older turns via an OpenAI-compatible chat endpoint; persists summary + recent window per session |
| `ovos-memory-plugin-rag` | `RAGMemory` | Stores every exchange via a files/embeddings API; retrieves top-k similar past turns at query time and injects them as context |

Both plugins implement `AgentContextManager` from `ovos-plugin-manager` (merged in [OPM PR #363](https://github.com/OpenVoiceOS/ovos-plugin-manager/pull/363)) and are consumed by `ovos-persona` (see [PR #143](https://github.com/OpenVoiceOS/ovos-persona/pull/143)) via the `memory_module` key in persona JSON.

---

## Installation

```bash
pip install ovos-memory-plugins
```

---

## Plugin 1 — `ovos-memory-plugin-longterm`

### How it works

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  Session store (JSON or SQLite)                                  │
  │  ┌──────────────────┐    ┌────────────────────────────────────┐  │
  │  │  rolling_summary │ ←  │  LLM summarize every N exchanges   │  │
  │  └──────────────────┘    └────────────────────────────────────┘  │
  │  ┌────────────────────────────────────────────────────────────┐  │
  │  │  recent_window messages (verbatim)                         │  │
  │  └────────────────────────────────────────────────────────────┘  │
  └─────────────────────────────────────────────────────────────────┘
           │
           ▼  build_conversation_context(utterance, session_id)
  [SYSTEM: system_prompt + rolling_summary]
  [... recent verbatim turns ...]
  [USER: current utterance]
```

Every time `update_history` is called the plugin increments an exchange counter (one assistant message = one exchange).  When the counter reaches `summarize_every` it sends the oldest messages to the configured LLM with a rolling-update prompt and stores the result; only `recent_window` exchanges are kept verbatim.

### Configuration

| Key | Type | Default | Description |
|---|---|---|---|
| `api_url` | str | `http://192.168.1.200:8000/v1` | Base URL of the OpenAI-compatible server |
| `model` | str | auto-detected | Model name for chat completions |
| `summarize_every` | int | `6` | Exchanges to accumulate before summarizing |
| `max_summary_tokens` | int | `256` | `max_tokens` for the summarization request |
| `recent_window` | int | `4` | Exchanges to keep verbatim after each summarization |
| `backend` | str | `"json"` | `"json"` or `"sqlite"` |
| `db_path` | str | `~/.local/share/ovos/longterm_memory.{json,db}` | Path to the persistence file |
| `system_prompt` | str | `""` | Persona system prompt |
| `request_timeout` | int | `30` | HTTP timeout in seconds |

### Persona wiring example

```json
{
  "name": "MyAssistant",
  "handlers": ["ovos-solver-openai-plugin"],
  "memory_module": "ovos-memory-plugin-longterm",
  "ovos-memory-plugin-longterm": {
    "api_url": "http://localhost:8000/v1",
    "model": "mistral",
    "summarize_every": 8,
    "max_summary_tokens": 300,
    "recent_window": 4,
    "backend": "sqlite",
    "db_path": "~/.local/share/ovos/assistant_memory.db",
    "system_prompt": "You are a helpful assistant."
  },
  "ovos-solver-openai-plugin": {
    "api_url": "http://localhost:8000/v1"
  }
}
```

---

## Plugin 2 — `ovos-memory-plugin-rag`

### How it works

```
  update_history(exchange)
      │
      ├─ POST /v1/embeddings  (embed doc text)
      ├─ POST /v1/files       (upload doc text)
      └─ POST /v1/vector_stores/{collection}/files  (attach)

  build_conversation_context(utterance)
      │
      ├─ POST /v1/vector_stores/{collection}/search  (top-k by cosine)
      │        (fallback: local in-process cosine if server unavailable)
      │
      └─ inject retrieved docs + recent history + [USER: utterance]
```

The plugin stores every assistant response (paired with its user prompt) as a document.  At query time it retrieves the `top_k` most similar past exchanges and injects them before the current utterance.  Filtering by `min_score` prevents low-relevance retrievals.

The endpoint contract follows [ovos-persona-server PR #11 (feat/rag)](https://github.com/OpenVoiceOS/ovos-persona-server/pull/11).

### Configuration

| Key | Type | Default | Description |
|---|---|---|---|
| `api_url` | str | `http://192.168.1.200:8000/v1` | Base URL of the OpenAI-compatible server |
| `collection` | str | `"ovos_memory"` | Vector store / collection name |
| `top_k` | int | `3` | Maximum retrieved documents per query |
| `min_score` | float | `0.0` | Minimum cosine similarity for a doc to be injected |
| `system_prompt` | str | `""` | Persona system prompt |
| `inject_as_system` | bool | `False` | Bundle retrieved docs into a single SYSTEM message instead of individual ASSISTANT messages |
| `request_timeout` | int | `30` | HTTP timeout in seconds |

### Persona wiring example

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
    "system_prompt": "You are a knowledgeable assistant.",
    "inject_as_system": true
  },
  "ovos-solver-openai-plugin": {
    "api_url": "http://localhost:8337/v1"
  }
}
```

---

## Architecture

```
ovos-persona (PR #143)
    └── Persona.__init__
            └── load_memory_plugin("ovos-memory-plugin-longterm")
                    └── LongTermMemory(config={...})

ovos-plugin-manager (PR #363)
    └── AgentContextManager
            ├── get_history(session_id)
            ├── update_history(messages, session_id)
            └── build_conversation_context(utterance, session_id) → List[AgentMessage]
```

Personas call `build_conversation_context` before every LLM request.  They call `update_history` after each exchange to keep both plugins' state current.

---

## Running tests

```bash
# Unit tests (mocked — no external services needed)
pip install -e ".[test]"
pytest tests/test_longterm.py tests/test_rag.py -v

# E2E: Long-term memory vs real Gemma at http://192.168.1.200:8000/v1
pytest tests/test_e2e_longterm.py -v -s

# E2E: RAG vs deterministic FastAPI stub (no external LLM needed)
pytest tests/test_e2e_rag.py -v -s
```

---

## Blog draft

### Persistent Memory for OVOS Personas: Long-Term Summarization and RAG

OVOS personas have always maintained a short-term conversation window, but until now there was no standard way to make that memory *persist* across sessions or *retrieve* relevant past exchanges for context.  With the new `opm.agents.memory` plugin contract merged in OPM PR #363, any persona can now declare a `memory_module` and get a fully pluggable memory backend.

This package ships two backends in one installable wheel:

**`ovos-memory-plugin-longterm`** works like a human's episodic memory: it keeps recent turns verbatim, then periodically summarizes everything older into a rolling paragraph.  That paragraph is persisted to disk (JSON or SQLite) and becomes part of the system prompt for every future turn — so even after a fresh process start the persona remembers what was discussed.  Summarization happens via any OpenAI-compatible chat endpoint; the local Gemma 4B model (`http://192.168.1.200:8000/v1`) works well for concise, factual summaries.

**`ovos-memory-plugin-rag`** takes a different angle: instead of compressing, it *retrieves*.  Every assistant response is embedded and uploaded to a vector store.  At query time the plugin finds the `top_k` most similar past exchanges by cosine similarity and injects them as context messages — giving the LLM concrete examples from its own history to reason about.  The endpoint contract mirrors `ovos-persona-server` PR #11 (`/v1/embeddings`, `/v1/files`, `/v1/vector_stores`).

Both plugins fall back gracefully when endpoints are unavailable: LTM silently skips summarization and retains the raw window; RAG falls back to local in-process cosine similarity using the embedding cache.

You can mix and match:  run LTM as the primary memory module and point both the LTM summarizer and the RAG server at the same `ovos-persona-server` instance to get rolling summaries *and* fine-grained retrieval simultaneously.
