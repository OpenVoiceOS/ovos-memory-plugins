#!/usr/bin/env python3
"""Fully-local RAG memory demo — no network, no API key.

Ingests a handful of exchanges into an in-process local vector store
(gguf embeddings + chromadb), then builds the conversation context for a new
question and prints the messages the persona's chat engine would receive,
showing which prior exchange was recalled.

    pip install 'ovos-memory-plugins[local-rag]'
    python examples/demo_local_rag.py [--inject-mode system|developer|system_prompt|user|tool]

The first run downloads the gguf embeddings model into the shared HF cache.
"""
import argparse
import tempfile

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_memory_plugins.local_rag import LocalRAGMemory

EXCHANGES = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Germany?", "The capital of Germany is Berlin."),
    ("How do you make pasta?", "Boil salted water and cook pasta for 8-10 minutes."),
    ("What is machine learning?", "Machine learning is a subset of AI where models learn from data."),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inject-mode", default="system",
                        choices=["system", "developer", "system_prompt", "user", "tool"])
    parser.add_argument("--query", default="Tell me about European capitals")
    args = parser.parse_args()

    plugin = LocalRAGMemory(config={
        "embeddings_plugin": "ovos-gguf-embeddings-plugin",
        "embeddings_config": {"model": "labse"},
        "embeddings_db_plugin": "ovos-chromadb-embeddings-plugin",
        "embeddings_db_config": {"path": tempfile.mkdtemp(prefix="local_rag_demo_")},
        "collection": "demo_local_rag",
        "retrieval": {"max_num_results": 3, "min_score": 0.2},
        "inject_mode": args.inject_mode,
        "system_prompt": "You are a helpful offline assistant.",
    })

    session = "demo"
    print(f"Ingesting {len(EXCHANGES)} exchanges into the local vector store ...")
    for user_txt, asst_txt in EXCHANGES:
        plugin.update_history(
            [AgentMessage(role=MessageRole.USER, content=user_txt),
             AgentMessage(role=MessageRole.ASSISTANT, content=asst_txt)],
            session)

    print(f"\nQuery: {args.query!r}  (inject_mode={args.inject_mode})\n")
    ctx = plugin.build_conversation_context(args.query, session)
    print(f"--- context the chat engine would receive ({len(ctx)} messages) ---")
    for m in ctx:
        role = m.role.value if hasattr(m.role, "value") else str(m.role)
        if m.tool_calls:
            for tc in m.tool_calls:
                print(f"  [{role}] tool_call {tc.name}({tc.arguments})")
        else:
            print(f"  [{role}] {m.content[:120]}")


if __name__ == "__main__":
    main()
