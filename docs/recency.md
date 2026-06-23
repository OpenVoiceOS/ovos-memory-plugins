# `ovos-memory-plugin-recency` — short-term buffer

`RecencyMemory` keeps a sliding window of the most recent turns, bounded by a
message count and, optionally, by age (time decay). It is the lightest possible
memory: no LLM, no embeddings, **no extra dependencies**, fully local.

Use it as a persona's `memory_module` for plain short-term context, or as the
`primary` (history-providing) member inside a [composite](./composite.md)
alongside heavier recall backends.

## Configuration

| Key | Default | Description |
|---|---|---|
| `max_history` | `10` | Messages kept verbatim. `0` = unbounded. |
| `max_age` | `null` | If set, drop messages older than this many seconds (decay). |
| `system_prompt` | `""` | Persona system prompt. |
| `db_path` | `null` | Optional JSON file; when set, the window persists across restarts. |

`build_conversation_context` returns `[system?] + recent window + [USER utterance]`,
pruning any trailing user turn so the new utterance is the only tail user message.

JSON persistence (`db_path`) stores message role and content; tool-call structure
(`tool_calls`, `tool_call_id`) is not persisted across restarts.

## Example

```json
{
  "memory_module": "ovos-memory-plugin-recency",
  "ovos-memory-plugin-recency": {
    "max_history": 8,
    "max_age": 1800,
    "system_prompt": "You are a helpful assistant."
  }
}
```
