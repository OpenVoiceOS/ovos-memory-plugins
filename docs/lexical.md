# `ovos-memory-plugin-lexical`: keyword recall (SQLite FTS5)

`LexicalMemory` recalls prior exchanges by **keyword** match using SQLite's
built-in FTS5 full-text index and its `bm25()` ranking. It needs **no extra
dependencies** (`sqlite3` is stdlib, FTS5 ships with CPython's bundled SQLite),
runs fully offline, and persists to a single `.db` file.

It is a retriever (exposes `search() -> List[MemoryHit]`), so it works standalone
*and* as a member of [`composite`](./composite.md).

## How it works

```
  update_history(exchange) ─▶ INSERT "Q: ...\nA: ..." into an FTS5 table
  build_conversation_context(utterance)
      ├─ SELECT ... WHERE table MATCH <terms> ORDER BY bm25(table) LIMIT k
      ├─ score = -bm25  (higher = better)
      └─ inject per inject_mode + history + [USER: utterance]
```

The query is tokenized to bare alphanumeric terms joined with `OR`, which both
sanitizes punctuation (no FTS5 syntax errors) and treats the query as "any of
these words".

## Why pair it with semantics

Keyword recall catches **exact terms** dense embeddings blur: names, codes, IDs,
and rare words. Semantic RAG catches paraphrases keywords miss. Combining the
two through the [composite](./composite.md) (hybrid search) beats either alone.

## Configuration

| Key | Default | Description |
|---|---|---|
| `db_path` | `~/.local/share/ovos/lexical_memory.db` | SQLite file. Use `":memory:"` for an ephemeral store. |
| `table` | `lexical_memory` | FTS5 virtual-table name. |
| `retrieval.max_num_results` | `5` | Max documents per query. |
| `retrieval.min_score` | `null` | Drop hits below this BM25-derived score (see note). |
| `retrieval.query_mode` / `query_history_turns` | `utterance` / `3` | Fold recent user turns into the query. |
| `context.*` | n/a | Chunk rendering (shared with the other retrievers). |
| `inject_mode` | `system` | `system` \| `developer` \| `system_prompt` \| `user` \| `tool`. |
| `system_prompt` | `""` | Persona system prompt. |
| `max_history` | `10` | Recent verbatim messages kept per session. |

> **Score scale:** lexical `score = -bm25` is **not** on the 0..1 cosine scale of
> `local-rag`. Set `min_score` accordingly, and when combining retrievers prefer
> the composite's rank-based `rrf` fusion rather than comparing raw scores.

Shared `retrieval` / `context` / `inject_mode` semantics are documented in the
[overview](./overview.md).

---
[← Local RAG](local-rag.md) · [Home](README.md) · [Recency →](recency.md)
