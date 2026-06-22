"""End-to-end test for LongTermMemory against a deterministic local stub.

LongTermMemory summarizes older turns through an OpenAI-compatible
``/chat/completions`` endpoint. To exercise the full store → summarize →
persist → rebuild cycle without any external server (local-first, offline,
deterministic in CI), this spins up an in-process FastAPI stub whose
``/chat/completions`` echoes the conversation topics back as a "summary".

Run with:  pytest tests/test_e2e_longterm.py -v -s
"""
from __future__ import annotations

import threading
import time

import pytest
import requests
import uvicorn
from fastapi import FastAPI

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_memory_plugins.longterm import LongTermMemory


def _user(text): return AgentMessage(role=MessageRole.USER, content=text)
def _assistant(text): return AgentMessage(role=MessageRole.ASSISTANT, content=text)


# ---------------------------------------------------------------------------
# Deterministic chat-completions stub
# ---------------------------------------------------------------------------

def build_stub_app() -> FastAPI:
    """A stub that turns the summarization prompt into a deterministic summary.

    The returned "summary" simply restates the salient words from the prompt the
    plugin sent, so topic-based assertions hold without a real LLM. A monotonic
    counter is folded in so two consecutive summarization passes differ.
    """
    app = FastAPI()
    _calls = [0]

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": "stub-summarizer"}]}

    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        _calls[0] += 1
        prompt = " ".join(m.get("content", "") for m in body.get("messages", []))
        topics = [w for w in ("python", "list", "tuple", "mutable", "append",
                              "egg", "boil", "simmering", "france", "paris",
                              "germany", "berlin")
                  if w in prompt.lower()]
        summary = (f"Summary #{_calls[0]}: the conversation covered "
                   + ", ".join(topics or ["various topics"]) + ".")
        return {"choices": [{"message": {"role": "assistant", "content": summary}}]}

    return app


def start_stub(port: int = 18799) -> str:
    app = build_stub_app()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(50):
        try:
            if requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=1).status_code == 200:
                break
        except Exception:
            time.sleep(0.2)
    return f"http://127.0.0.1:{port}/v1"


@pytest.fixture(scope="module")
def stub_url():
    return start_stub()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_e2e_summarization(stub_url, tmp_path):
    """Three exchanges trigger summarization; the summary touches the topics."""
    plugin = LongTermMemory(config={
        "api_url": stub_url,
        "model": "stub-summarizer",
        "backend": "json",
        "db_path": str(tmp_path / "e2e_ltm.json"),
        "summarize_every": 3,
        "max_summary_tokens": 150,
        "recent_window": 2,
        "system_prompt": "You are a helpful assistant.",
        "request_timeout": 30,
    })

    exchanges = [
        ("What is a Python list?", "A list is an ordered, mutable sequence of elements."),
        ("How do I add an item to a list?", "Use list.append(item) to add to the end."),
        ("What is the difference between a list and a tuple?",
         "Tuples are immutable; lists are mutable."),
    ]
    session_id = "e2e-session-001"
    for user_txt, asst_txt in exchanges:
        plugin.update_history([_user(user_txt), _assistant(asst_txt)], session_id)

    summary = plugin._cache[session_id]["summary"]
    assert summary, "Summary should not be empty after 3 exchanges"
    assert len(summary) > 20
    assert any(word in summary.lower()
               for word in ["python", "list", "tuple", "mutable", "append"]), \
        f"Summary does not mention expected topics: {summary!r}"

    ctx = plugin.build_conversation_context("What else can I do with lists?", session_id)
    system_msg = next((m for m in ctx if m.role == MessageRole.SYSTEM), None)
    assert system_msg is not None
    assert summary in system_msg.content
    assert ctx[-1].content == "What else can I do with lists?"


def test_e2e_rolling_summary_across_two_batches(stub_url, tmp_path):
    """Two summarization cycles produce an updated rolling summary."""
    plugin = LongTermMemory(config={
        "api_url": stub_url,
        "model": "stub-summarizer",
        "backend": "sqlite",
        "db_path": str(tmp_path / "e2e_roll.db"),
        "summarize_every": 2,
        "max_summary_tokens": 120,
        "recent_window": 1,
        "request_timeout": 30,
    })
    session_id = "e2e-roll-001"

    plugin.update_history([_user("How do I boil an egg?"),
                           _assistant("Place egg in cold water, bring to boil, cook 10 minutes.")], session_id)
    plugin.update_history([_user("What temperature for soft-boiled?"),
                           _assistant("Use simmering water around 90C for 6 minutes.")], session_id)
    summary_1 = plugin._cache[session_id]["summary"]
    assert summary_1, "First batch should produce a summary"

    plugin.update_history([_user("What is the capital of France?"),
                           _assistant("The capital of France is Paris.")], session_id)
    plugin.update_history([_user("And of Germany?"),
                           _assistant("The capital of Germany is Berlin.")], session_id)
    summary_2 = plugin._cache[session_id]["summary"]
    assert summary_2, "Second batch should produce a summary"
    assert summary_2 != summary_1, "Rolling summary should have been updated"
