# Examples

Runnable demos and ready-to-use persona configs for the memory backends.

## Persona JSON configs

Drop one of these into your `ovos-persona` personas directory (or pass it to
persona-server). The `memory_module` key selects the backend; the same-named
block configures it.

| File | Backend | Needs |
|---|---|---|
| [`persona_local_rag.json`](./persona_local_rag.json) | `ovos-memory-plugin-local-rag` | **nothing external** — fully offline (gguf + chromadb) |
| [`persona_longterm.json`](./persona_longterm.json) | `ovos-memory-plugin-longterm` | a chat endpoint |

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
