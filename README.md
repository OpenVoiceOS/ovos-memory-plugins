# ovos-memory-plugins

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI](https://img.shields.io/pypi/v/ovos-memory-plugins.svg)](https://pypi.org/project/ovos-memory-plugins/)
[![Build](https://github.com/OpenVoiceOS/ovos-memory-plugins/actions/workflows/build-tests.yml/badge.svg)](https://github.com/OpenVoiceOS/ovos-memory-plugins/actions/workflows/build-tests.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

**Give your [OpenVoiceOS](https://openvoiceos.org) persona a memory.**

By default a chat persona is amnesiac: every turn starts from scratch. A *memory
plugin* fixes that — it remembers what was said and quietly feeds the relevant
bits back into the next prompt, so the assistant can follow up, recall facts, and
stay on topic. This package is a bundle of local-first memory backends plus an
orchestrator that combines them.

"Local-first" means every backend runs on your machine. Some are pure standard
library (zero extra dependencies); the heavier ones use local models or a *local*
LLM endpoint — none of them phone home to a cloud service.

| Backend (`memory_module`) | What it remembers with | Extra setup |
|---|---|---|
| [`ovos-memory-plugin-recency`](docs/recency.md) | the last few turns (sliding window) | none |
| [`ovos-memory-plugin-lexical`](docs/lexical.md) | keyword search over past turns (SQLite FTS5 + BM25) | none |
| [`ovos-memory-plugin-local-rag`](docs/local-rag.md) | semantic search over past turns (embeddings + vector DB) | local model stack |
| [`ovos-memory-plugin-longterm`](docs/longterm.md) | a rolling summary of the whole conversation | a local chat endpoint |
| [`ovos-memory-plugin-entity`](docs/entity.md) | durable facts about the user (name, preferences…) | a local chat endpoint |
| [`ovos-memory-plugin-composite`](docs/composite.md) | **several of the above at once**, results merged | the members' setup |

New here? Jump to [Quick start](#quick-start). Building something? See
[How it works](#how-it-works) and [Write your own backend](#write-your-own-backend).

---

## Install

```bash
pip install ovos-memory-plugins

# add the local semantic-RAG stack (embeddings model + vector store):
pip install 'ovos-memory-plugins[local-rag]'
```

`recency` and `lexical` need nothing beyond the base install — they are pure
Python standard library.

---

## Quick start

A persona is a small JSON file. The `memory_module` key picks a memory backend;
a block with the same name configures it. Drop the file in your `ovos-persona`
personas directory.

### Step 1 — remember the last few turns (no setup)

The simplest memory: a sliding window of recent turns. No models, no endpoints.

```json
{
  "name": "MyAssistant",
  "memory_module": "ovos-memory-plugin-recency",
  "ovos-memory-plugin-recency": {
    "max_history": 10,
    "system_prompt": "You are a helpful assistant."
  }
}
```

The assistant can now handle "and what about tomorrow?" because the previous
turns are still in context. That is all most short conversations need.

### Step 2 — recall things said long ago (semantic)

A window forgets. To recall something from much earlier, search past turns by
meaning:

```json
{
  "name": "MyAssistant",
  "memory_module": "ovos-memory-plugin-local-rag",
  "ovos-memory-plugin-local-rag": {
    "retrieval": {"max_num_results": 4},
    "system_prompt": "You are a helpful assistant."
  }
}
```

Every exchange is embedded and stored in a local vector database; before each
reply the most relevant past exchanges are retrieved and added to the prompt.
Ask "what was that book I mentioned last week?" and it can answer. Needs the
`[local-rag]` extra.

### Step 3 — combine memories (hero mode)

Real assistants want more than one kind of memory. The **composite** backend
loads several members and merges their results, so one `memory_module` gives you
hybrid recall (meaning *and* keywords) plus durable user facts:

```json
{
  "name": "MyAssistant",
  "memory_module": "ovos-memory-plugin-composite",
  "ovos-memory-plugin-composite": {
    "members": [
      {"module": "ovos-memory-plugin-local-rag", "config": {"collection": "kb"}},
      {"module": "ovos-memory-plugin-lexical",   "config": {"db_path": "~/.local/share/ovos/lex.db"}},
      {"module": "ovos-memory-plugin-entity",    "config": {"api_url": "http://localhost:8000/v1"}}
    ],
    "fusion": "rrf",
    "system_prompt": "You are a helpful assistant."
  }
}
```

`local-rag` catches paraphrases, `lexical` catches exact terms (names, codes,
rare words), and `entity` remembers who the user is. Their hits are merged with
Reciprocal Rank Fusion — see [composite](docs/composite.md).

Prefer to see it run before wiring a persona? The [`examples/`](examples/) folder
has ready persona files and offline demo scripts:

```bash
python examples/demo_composite.py     # hybrid recall, fully offline
```

---

## How it works

A memory backend is an `AgentContextManager`. The persona owns the chat model and
tools; the memory owns **conversation state and prompt assembly** — it never
generates the answer itself, it just shapes the messages the model sees. Three
methods make up the whole contract:

```python
get_history(session_id) -> list[AgentMessage]
update_history(new_messages, session_id) -> None
build_conversation_context(utterance, session_id) -> list[AgentMessage]
```

The persona calls `build_conversation_context` before each turn (to assemble the
prompt) and `update_history` after each turn (to record what happened). Two rules
hold for the returned message list:

- the **first** message MAY be a `system` message (the persona prompt);
- the **last** message is ALWAYS the current user utterance.

Everything else — summaries, retrieved snippets, known facts, recent turns — goes
in between. The [overview](docs/overview.md) explains the shared knobs: the five
`inject_mode` strategies (how recalled context is placed in the prompt) and the
retrieval settings (`max_num_results`, `min_score`, `query_mode`).

---

## Choosing a backend

| Want… | Use |
|---|---|
| just the last few turns | `recency` |
| exact-term recall (names, IDs, codes) | `lexical` |
| meaning-based recall of past detail | `local-rag` |
| robust recall (meaning + keywords) | `composite` of `local-rag` + `lexical` |
| to remember who the user is across sessions | `entity` |
| a compact gist of very long chats | `longterm` |
| more than one of the above | `composite` |

The [overview](docs/overview.md#choosing-a-backend) has a full comparison table
(persistence, dependencies, offline behaviour, cost per turn).

---

## The composite

`composite` is a pure orchestrator: it loads member backends by name and
consolidates them.

- **Retriever members** (`local-rag`, `lexical`, or any backend exposing
  `search()`) have their hits **fused** into one ranked, deduplicated list.
  Reciprocal Rank Fusion is the default — it ranks by *position*, not raw score,
  so it combines backends whose scores live on different scales (cosine vs BM25)
  without one drowning out the other. Other modes: `weighted`, `merge`,
  `priority`, `interleave`.
- **Context members** (`longterm`, `entity`, `recency`) contribute their system
  block (a summary, known facts…).
- New turns are recorded in **every** member; recent history comes from a chosen
  `primary` member.

If a member fails to load or errors at runtime, it is skipped and the rest carry
on. Full details and the config schema: [composite](docs/composite.md).

---

## Write your own backend

Two paths, depending on what you are building:

- **A retrieval backend** (stores documents, recalls them by some search) —
  subclass `BaseRetrievalMemory` and implement just two hooks, `_store_document`
  and `_query_backend` (returning `MemoryHit`s). You inherit history handling,
  the five inject modes, the context renderer, and a `search()` that plugs
  straight into the composite.
- **Any other memory** — subclass `AgentContextManager` and implement the three
  contract methods directly.

Register the class under the `opm.agents.memory` entry-point group and it becomes
selectable as a `memory_module`. Step-by-step guide with code:
[write a backend](docs/writing-a-backend.md).

---

## Architecture

```
ovos-persona
  └─ memory_module: "ovos-memory-plugin-composite"
       └─ CompositeMemory
            ├─ ovos-memory-plugin-local-rag   (retriever — semantic)
            ├─ ovos-memory-plugin-lexical     (retriever — keyword)
            └─ ovos-memory-plugin-entity      (context  — user facts)

AgentContextManager  (the contract every backend implements)
  ├─ get_history(session_id)
  ├─ update_history(messages, session_id)
  └─ build_conversation_context(utterance, session_id) -> list[AgentMessage]
```

Retrieval backends and the composite share a `BaseRetrievalMemory` that provides
history, the inject-mode strategies, and the context renderer; a concrete
retriever only implements *store* and *query*. The fusion helpers and the
`MemoryHit` type live in `ovos_memory_plugins.common`.

---

## Running tests

```bash
pip install -e ".[test]"

pytest tests -v                       # everything (unit + end-to-end), no external services
pytest tests/test_composite.py -v     # just the composite (fast)
```

The end-to-end RAG test exercises the real embeddings + vector-store stack; its
first run downloads the embeddings model into the shared cache, then is fast.

---

## License

Apache License 2.0 — see [LICENSE](./LICENSE).

## Credits

Developed by [TigreGótico](https://tigregotico.pt) for
[OpenVoiceOS](https://openvoiceos.org).

[![NGI0 Commons Fund](./ngi.png)](https://nlnet.nl/project/OpenVoiceOS)

This project was funded through the [NGI0 Commons Fund](https://nlnet.nl/commonsfund),
a fund established by [NLnet](https://nlnet.nl) with financial support from the
European Commission's [Next Generation Internet](https://ngi.eu) programme, under
the aegis of [DG Communications Networks, Content and Technology](https://commission.europa.eu/about-european-commission/departments-and-executive-agencies/communications-networks-content-and-technology_en)
under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429).
