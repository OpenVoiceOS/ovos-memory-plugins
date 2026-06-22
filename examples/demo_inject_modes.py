#!/usr/bin/env python3
"""Compare every LocalRAGMemory inject_mode side by side — fully offline.

Renders the same retrieved context under each supported ``inject_mode`` so you
can see exactly which messages each strategy produces before wiring a persona.

    pip install 'ovos-memory-plugins[local-rag]'
    python examples/demo_inject_modes.py
"""
import tempfile

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_memory_plugins.local_rag import LocalRAGMemory

EXCHANGES = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Germany?", "The capital of Germany is Berlin."),
]
QUERY = "What is the capital of France?"


def build(mode, db_path):
    plugin = LocalRAGMemory(config={
        "embeddings_plugin": "ovos-gguf-embeddings-plugin",
        "embeddings_config": {"model": "labse"},
        "embeddings_db_plugin": "ovos-chromadb-embeddings-plugin",
        "embeddings_db_config": {"path": db_path},
        "collection": "demo_inject_modes",
        "retrieval": {"max_num_results": 2, "min_score": 0.2},
        "inject_mode": mode,
        "system_prompt": "You are a helpful assistant.",
    })
    for u, a in EXCHANGES:
        plugin.update_history(
            [AgentMessage(role=MessageRole.USER, content=u),
             AgentMessage(role=MessageRole.ASSISTANT, content=a)], mode)
    return plugin.build_conversation_context(QUERY, mode)


def main():
    db_path = tempfile.mkdtemp(prefix="inject_modes_demo_")
    for mode in ("system", "developer", "system_prompt", "user", "tool"):
        print(f"\n===== inject_mode = {mode} =====")
        for m in build(mode, db_path):
            role = m.role.value if hasattr(m.role, "value") else str(m.role)
            if m.tool_calls:
                for tc in m.tool_calls:
                    print(f"  [{role}] tool_call {tc.name}({tc.arguments})")
            else:
                print(f"  [{role}] {m.content[:100]}")


if __name__ == "__main__":
    main()
