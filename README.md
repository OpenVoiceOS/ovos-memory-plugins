# ovos-memory-plugins

[OVOS](https://github.com/OpenVoiceOS) `opm.agents.memory` plugins bundled in one package:

| Entry point | Class | Purpose | Needs |
|---|---|---|---|
| `ovos-memory-plugin-longterm` | `LongTermMemory` | Rolling summarization of older turns via an OpenAI-compatible chat endpoint; persists summary + recent window per session | a chat endpoint |
| `ovos-memory-plugin-local-rag` | `LocalRAGMemory` | **Fully-local in-process RAG** — embeds and stores every exchange in a local vector DB via OVOS embeddings plugins, retrieves top-k relevant prior turns and injects them per `inject_mode`. No network, no cloud key. | local plugins only |

Both plugins implement `AgentContextManager` from `ovos-plugin-manager` (merged in [OPM PR #363](https://github.com/OpenVoiceOS/ovos-plugin-manager/pull/363)) and are consumed by `ovos-persona` (see [PR #143](https://github.com/OpenVoiceOS/ovos-persona/pull/143)) via the `memory_module` key in persona JSON.

`LocalRAGMemory` is the local-first headline: it runs RAG entirely in-process so a private/offline assistant gets long-term recall without standing up any server. `LongTermMemory` carries a compact running summary and needs only a chat endpoint.

Full docs: [`docs/`](./docs/) (overview + a page per backend). Runnable examples and persona JSON: [`examples/`](./examples/).

---

## Installation

```bash
pip install ovos-memory-plugins

# fully-local RAG stack (gguf embeddings + chromadb vector store):
pip install 'ovos-memory-plugins[local-rag]'
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

## Plugin 2 — `ovos-memory-plugin-local-rag`

Fully-local, in-process RAG. Loads an OVOS text-embeddings plugin and an
`EmbeddingsDB` plugin directly (no HTTP), embeds every exchange, and retrieves
the top-k most relevant prior turns before each response.

### How it works

```
  update_history(exchange)
      │
      ├─ embedder.get_embeddings("Q: ...\nA: ...")   (in-process)
      └─ db.add_embeddings(key, vector, metadata)     (local vector store)

  build_conversation_context(utterance)
      │
      ├─ db.query(embed(query), top_k)  → top-k by cosine distance
      ├─ filter by min_score (score = 1 - distance)
      └─ inject per inject_mode + recent history + [USER: utterance]
```

The default stack is `ovos-gguf-embeddings-plugin` (labse gguf embeddings) +
`ovos-chromadb-embeddings-plugin` (persistent vector store) — install the
`local-rag` extra. Any `opm.embeddings.text` + `opm.embeddings` (`EmbeddingsDB`)
pair works.

### Configuration

| Key | Type | Default | Description |
|---|---|---|---|
| `embeddings_plugin` | str | `ovos-gguf-embeddings-plugin` | `opm.embeddings.text` entry point |
| `embeddings_config` | dict | `{}` | Config for the embeddings plugin (e.g. `{"model": "labse"}`) |
| `embeddings_db_plugin` | str | `ovos-chromadb-embeddings-plugin` | `opm.embeddings` (`EmbeddingsDB`) entry point |
| `embeddings_db_config` | dict | `{}` | Config for the DB (e.g. `{"path": "~/.local/share/ovos/local_rag_db"}`) |
| `collection` | str | `ovos_local_rag` | Collection / vector-store name |
| `retrieval.max_num_results` | int | `5` | Max retrieved documents per query |
| `retrieval.min_score` | float\|null | `null` | Drop hits below this score (`score = 1 - cosine_distance`) |
| `retrieval.query_mode` | str | `utterance` | `utterance` or `history` (fold recent user turns into the query) |
| `retrieval.query_history_turns` | int | `3` | Turns folded in when `query_mode="history"` |
| `context.header` | str | (see source) | Header line above the retrieved context |
| `context.chunk_prefix` / `chunk_separator` | str | `"- "` / `"\n\n"` | Rendering of each chunk |
| `context.include_sources` | bool | `false` | Prefix each chunk with its stored id |
| `context.tool_name` | str | `search_memory` | Tool name used by `inject_mode="tool"` |
| `inject_mode` | str | `system` | `system` \| `system_prompt` \| `developer` \| `user` \| `tool` |
| `system_prompt` | str | `""` | Persona system prompt |
| `max_history` | int | `10` | Recent verbatim messages kept per session |

See [`docs/local-rag.md`](./docs/local-rag.md) for the inject-mode details.

### Persona wiring example (fully offline)

```json
{
  "name": "LocalRagBot",
  "handlers": ["ovos-gguf-chat-plugin"],
  "memory_module": "ovos-memory-plugin-local-rag",
  "ovos-memory-plugin-local-rag": {
    "embeddings_plugin": "ovos-gguf-embeddings-plugin",
    "embeddings_config": {"model": "labse"},
    "embeddings_db_plugin": "ovos-chromadb-embeddings-plugin",
    "embeddings_db_config": {"path": "~/.local/share/ovos/local_rag_db"},
    "collection": "localragbot",
    "retrieval": {"max_num_results": 4, "min_score": 0.3},
    "inject_mode": "system",
    "system_prompt": "You are a helpful offline assistant."
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
pip install -e ".[test]"

# everything (unit + e2e), no external services — runs 0-skipped
pytest tests -v

# unit tests only (fast, stub backends)
pytest tests/test_longterm.py tests/test_local_rag.py -v

# e2e: local RAG against the real gguf + chromadb stack (first run downloads the model)
pytest tests/test_e2e_local_rag.py -v -s

# e2e: long-term against an in-process FastAPI chat stub (no external LLM)
pytest tests/test_e2e_longterm.py -v -s
```

---

## Credits

Developed by [TigreGótico](https://tigregotico.pt) for
[OpenVoiceOS](https://openvoiceos.org).

[![NGI0 Commons Fund](./ngi.png)](https://nlnet.nl/project/OpenVoiceOS)

This project was funded through the [NGI0 Commons Fund](https://nlnet.nl/commonsfund),
a fund established by [NLnet](https://nlnet.nl) with financial support from the
European Commission's [Next Generation Internet](https://ngi.eu) programme, under
the aegis of [DG Communications Networks, Content and Technology](https://commission.europa.eu/about-european-commission/departments-and-executive-agencies/communications-networks-content-and-technology_en)
under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429).
