# Examples

Runnable demos and ready-to-use persona configs for the memory backends.

## Persona JSON configs

Drop one of these into your `ovos-persona` personas directory (or pass it to
persona-server). The `memory_module` key selects the backend; the same-named
block configures it.

| File | Backend | Needs |
|---|---|---|
| [`persona_local_rag.json`](./persona_local_rag.json) | `ovos-memory-plugin-local-rag` | **nothing external** — fully offline (gguf + chromadb) |
| [`persona_lexical.json`](./persona_lexical.json) | `ovos-memory-plugin-lexical` | **nothing** — stdlib SQLite FTS5 |
| [`persona_longterm.json`](./persona_longterm.json) | `ovos-memory-plugin-longterm` | a chat endpoint |
| [`persona_entity.json`](./persona_entity.json) | `ovos-memory-plugin-entity` | a chat endpoint |
| [`persona_composite.json`](./persona_composite.json) | `ovos-memory-plugin-composite` | members' needs (here: gguf + chromadb + a chat endpoint) |

## Runnable scripts

### `demo_local_rag.py` — fully offline

Ingests a few exchanges, then shows how a later utterance retrieves the relevant
prior context. No network, no API key. The first run downloads the gguf
embeddings model into the shared HF cache.

```bash
pip install 'ovos-memory-plugins[local-rag]'
python examples/demo_local_rag.py
# try a different inject mode:
python examples/demo_local_rag.py --inject-mode tool
```

### `demo_inject_modes.py` — compare inject modes offline

Renders the same retrieved context under every supported `inject_mode` so you
can see exactly which messages each strategy produces.

```bash
python examples/demo_inject_modes.py
```

### `demo_composite.py` — hybrid fusion offline

Fuses a semantic (`local-rag`) and a lexical (`lexical`) member and prints the
fused hits under each fusion mode. Uses a toy in-process embedder, so it needs no
model download and no network.

```bash
python examples/demo_composite.py
```
