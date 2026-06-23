# Writing your own memory backend

A memory backend is an `AgentContextManager`: an object the persona asks to
remember turns and to assemble the next prompt. You can write one in a few dozen
lines. There are two starting points — pick by what you are building.

- **A retrieval backend** (stores items, recalls them by some kind of search):
  start from `BaseRetrievalMemory` and implement two small hooks. You inherit
  history handling, the five inject modes, the context renderer, and a `search()`
  that lets the [composite](./composite.md) fuse you with other retrievers.
- **Anything else** (a summary, a fact store, a custom rule): start from
  `AgentContextManager` and implement the three contract methods directly.

## The contract

```python
get_history(session_id) -> list[AgentMessage]
update_history(new_messages, session_id) -> None
build_conversation_context(utterance, session_id) -> list[AgentMessage]
```

`build_conversation_context` returns the messages sent to the chat model. Two
rules: the first message MAY be a `system` message, and the last message is
ALWAYS the current user utterance. `AgentMessage`, `MessageRole`, and `ToolCall`
come from `ovos_plugin_manager.templates.agents`.

## Path A — a retrieval backend

Implement `_store_document` (persist one item) and `_query_backend` (return ranked
[`MemoryHit`](#the-memoryhit-type)s). Everything else is inherited.

```python
from typing import Any, Dict, List
from ovos_memory_plugins.base import BaseRetrievalMemory
from ovos_memory_plugins.common import MemoryHit


class SubstringMemory(BaseRetrievalMemory):
    """Toy retriever: recalls past exchanges by substring count."""

    def __init__(self, config=None):
        super().__init__(config)
        self._docs: Dict[str, str] = {}        # doc_id -> text

    def _store_document(self, doc_id: str, text: str, session_id: str,
                        metadata: Dict[str, Any]) -> None:
        self._docs[doc_id] = text

    def _query_backend(self, query: str, top_k: int) -> List[MemoryHit]:
        q = query.lower()
        hits = [MemoryHit(content=text, source=doc_id, score=float(text.lower().count(q)))
                for doc_id, text in self._docs.items()]
        hits = [h for h in hits if h.score > 0]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]
```

That is a complete, usable backend. From the base you get:

- `update_history` — pairs each user→assistant turn into a `"Q: …\nA: …"`
  document and calls your `_store_document` (errors are caught, so a storage
  hiccup never breaks the conversation);
- `search(query, session_id, top_k)` — calls `_query_backend`, then applies the
  `min_score` filter and `top_k` cap;
- `build_conversation_context` — runs the search and injects the result per the
  configured `inject_mode`, with the user utterance last;
- shared config: `retrieval` (`max_num_results`, `min_score`, `query_mode`,
  `query_history_turns`), `context` (chunk rendering), `inject_mode`,
  `system_prompt`, `max_history`. Add your own keys in `__init__` from
  `self.config`.

Because it exposes `search()` returning `MemoryHit`s, it can be a member of the
composite and be fused with other retrievers for free.

## Path B — any other memory

Subclass `AgentContextManager` and implement the three methods.

```python
from ovos_plugin_manager.templates.agents import (
    AgentContextManager, AgentMessage, MessageRole,
)


class StickyNoteMemory(AgentContextManager):
    """Remembers anything the user says after 'remember', injects it each turn."""

    def __init__(self, config=None):
        super().__init__(config)
        self._notes: dict = {}

    def get_history(self, session_id):
        return []   # this backend isn't a turn buffer

    def update_history(self, new_messages, session_id):
        for m in new_messages:
            if m.role == MessageRole.USER and "remember" in m.content.lower():
                self._notes.setdefault(session_id, []).append(m.content)

    def build_conversation_context(self, utterance, session_id):
        messages = []
        if self.system_prompt:
            messages.append(AgentMessage(MessageRole.SYSTEM, self.system_prompt))
        notes = self._notes.get(session_id, [])
        if notes:
            messages.append(AgentMessage(
                MessageRole.SYSTEM, "Notes:\n" + "\n".join(f"- {n}" for n in notes)))
        messages.append(AgentMessage(MessageRole.USER, utterance))   # always last
        return messages
```

A Path-B backend works standalone and as a *context member* of the composite
(its leading system block is folded in). To also be fused as a retriever, add a
`search(query, session_id=None, top_k=None) -> list[MemoryHit]` method.

## The `MemoryHit` type

```python
from ovos_memory_plugins.common import MemoryHit

MemoryHit(content="...", source="doc-42", score=0.87, metadata={"session_id": "s1"})
```

`content` is the recalled text, `source` an optional id, `score` the
retriever-native relevance (any scale — fusion handles scale differences),
`metadata` is free-form.

## Register it

Expose the class under the `opm.agents.memory` entry-point group in your
package's `pyproject.toml`:

```toml
[project.entry-points."opm.agents.memory"]
my-cool-memory = "my_package.memory:SubstringMemory"
```

After installing, select it from a persona:

```json
{
  "memory_module": "my-cool-memory",
  "my-cool-memory": {"system_prompt": "You are a helpful assistant."}
}
```

## Test it

Backends are plain objects — instantiate, feed turns, assert on the result.
For a Path-A retriever, inject your own storage so tests need no real services
(the bundled backends accept stub backends through their config for exactly this).

```python
def test_recall():
    mem = SubstringMemory(config={"retrieval": {"max_num_results": 3}})
    mem.update_history([
        AgentMessage(MessageRole.USER, "my favourite colour is teal"),
        AgentMessage(MessageRole.ASSISTANT, "noted!"),
    ], "s1")
    ctx = mem.build_conversation_context("what colour did I like?", "s1")
    assert ctx[-1].role == MessageRole.USER          # contract: user last
    assert any("teal" in m.content for m in ctx)      # it was recalled
```
