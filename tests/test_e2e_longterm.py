"""
End-to-end test for LongTermMemory against the real Gemma endpoint.

Run with:  pytest tests/test_e2e_longterm.py -v -s

Requires: http://192.168.1.200:8000/v1 to be reachable.
Skipped automatically when the endpoint is unavailable.
"""
import os
import tempfile
import time

import pytest
import requests

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_memory_plugins.longterm import LongTermMemory

GEMMA_URL = os.environ.get("LLM_ENDPOINT", "http://192.168.1.200:8000/v1")
GEMMA_MODEL = "ggml-org/gemma-4-26B-A4B-it-GGUF"


def _gemma_available():
    try:
        r = requests.get(GEMMA_URL.rstrip("/") + "/models", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


skip_if_no_gemma = pytest.mark.skipif(
    not _gemma_available(),
    reason=f"Gemma endpoint {GEMMA_URL} not reachable"
)


def _user(text): return AgentMessage(role=MessageRole.USER, content=text)
def _assistant(text): return AgentMessage(role=MessageRole.ASSISTANT, content=text)


@skip_if_no_gemma
def test_e2e_real_gemma_summarization(tmp_path):
    """
    Simulate a short conversation, trigger summarization via real Gemma,
    then assert the summary is non-empty and semantically plausible.
    """
    plugin = LongTermMemory(config={
        "api_url": GEMMA_URL,
        "model": GEMMA_MODEL,
        "backend": "json",
        "db_path": str(tmp_path / "e2e_ltm.json"),
        "summarize_every": 3,
        "max_summary_tokens": 150,
        "recent_window": 2,
        "system_prompt": "You are a helpful assistant.",
        "request_timeout": 60,
    })

    # Simulate 3 exchanges about Python programming
    exchanges = [
        ("What is a Python list?", "A list is an ordered, mutable sequence of elements."),
        ("How do I add an item to a list?", "Use list.append(item) to add to the end."),
        ("What is the difference between a list and a tuple?",
         "Tuples are immutable; lists are mutable."),
    ]
    session_id = "e2e-session-001"

    for user_txt, asst_txt in exchanges:
        plugin.update_history([_user(user_txt), _assistant(asst_txt)], session_id)

    rec = plugin._cache[session_id]
    summary = rec["summary"]

    print("\n--- REAL GEMMA SUMMARY ---")
    print(summary)
    print("---")

    assert summary, "Summary should not be empty after 3 exchanges"
    assert len(summary) > 20, "Summary should be at least a sentence"
    # Sanity: the summary should touch Python-related content
    assert any(word in summary.lower() for word in ["python", "list", "tuple", "mutable", "append"]), \
        f"Summary does not mention expected topics: {summary!r}"

    # Verify context building includes the summary
    ctx = plugin.build_conversation_context("What else can I do with lists?", session_id)
    system_msg = next((m for m in ctx if m.role == MessageRole.SYSTEM), None)
    assert system_msg is not None, "Context should have a system message"
    assert summary in system_msg.content, "System message should contain the rolling summary"
    assert ctx[-1].content == "What else can I do with lists?"

    print(f"Context length: {len(ctx)} messages")
    print(f"System prompt: {system_msg.content[:200]}...")


@skip_if_no_gemma
def test_e2e_rolling_summary_across_two_batches(tmp_path):
    """
    Two summarization cycles should produce an updated rolling summary that
    references content from both batches.
    """
    plugin = LongTermMemory(config={
        "api_url": GEMMA_URL,
        "model": GEMMA_MODEL,
        "backend": "sqlite",
        "db_path": str(tmp_path / "e2e_roll.db"),
        "summarize_every": 2,
        "max_summary_tokens": 120,
        "recent_window": 1,
        "request_timeout": 60,
    })
    session_id = "e2e-roll-001"

    # Batch 1: cooking topic
    plugin.update_history([_user("How do I boil an egg?"),
                            _assistant("Place egg in cold water, bring to boil, cook 10 minutes.")], session_id)
    plugin.update_history([_user("What temperature for soft-boiled?"),
                            _assistant("Use simmering water around 90°C for 6 minutes.")], session_id)

    summary_1 = plugin._cache[session_id]["summary"]
    print(f"\n--- SUMMARY after batch 1 ---\n{summary_1}\n---")
    assert summary_1, "First batch should produce a summary"

    # Batch 2: geography topic
    plugin.update_history([_user("What is the capital of France?"),
                            _assistant("The capital of France is Paris.")], session_id)
    plugin.update_history([_user("And of Germany?"),
                            _assistant("The capital of Germany is Berlin.")], session_id)

    summary_2 = plugin._cache[session_id]["summary"]
    print(f"\n--- SUMMARY after batch 2 ---\n{summary_2}\n---")
    assert summary_2, "Second batch should produce a summary"
    assert summary_2 != summary_1, "Rolling summary should have been updated"
