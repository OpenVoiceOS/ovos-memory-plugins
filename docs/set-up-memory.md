# Set up persona memory

Each task gives the setup, the steps and the output. The chat model in the
examples is `qwen2.5:7b`, served by a public demo server. The model decides
which facts the `entity` backend stores. The stored facts change from run to
run, and from model to model. Each output below is from one example run.

- [Remember the user after a restart](#remember-the-user-after-a-restart)
- [Give each family member their own memory](#give-each-family-member-their-own-memory)
- [Answer from your own documents, fully local](#answer-from-your-own-documents-fully-local)

All three tasks need an OpenAI-compatible chat server. The examples use
`https://llama.smartgic.io/v1`. Change `api_url` to your own server.

---

## Remember the user after a restart

Use this setup when you run one assistant and want it to keep what you told it
when the service restarts.

The `entity` backend asks the chat model for durable facts
about the user after each exchange, and writes them to a JSON file per user.
After a restart, the facts are back in the prompt.

1. Install the packages:

    ```bash
    pip install --pre ovos-persona-server ovos-memory-plugins ovos-openai-plugin
    ```

2. Save the persona as `assistant.json`:

    ```json
    {
      "name": "assistant",
      "handlers": ["ovos-chat-openai-plugin"],
      "ovos-chat-openai-plugin": {
        "api_url": "https://llama.smartgic.io/v1",
        "key": "sk-none",
        "model": "qwen2.5:7b",
        "allow_system_prompts": true,
        "system_prompt": "You are a home assistant. Answer in at most fifteen words."
      },
      "memory_module": "ovos-memory-plugin-entity",
      "ovos-memory-plugin-entity": {
        "api_url": "https://llama.smartgic.io/v1",
        "model": "qwen2.5:7b",
        "backend": "json",
        "db_path": "~/.local/share/ovos/assistant_facts.json"
      }
    }
    ```

3. Start the server with server-side memory:

    ```bash
    CHAT_MEMORY=transparent ovos-persona-server --persona assistant.json --port 8337
    ```

4. Tell it something. The `user` field selects your memory:

    ```bash
    ask() {
      curl -s localhost:8337/openai/v1/chat/completions \
        -H 'content-type: application/json' \
        -d "{\"model\":\"assistant\",\"user\":\"$1\",\"messages\":[{\"role\":\"user\",\"content\":\"$2\"}]}" |
        python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
    }
    ask alice "My cat is called Pixel and I live in Lisbon."
    ```

5. Stop the server, and start it again with the command of step 3.

The facts file survives the restart. Output of the run:

```json
{
  "alice": {
    "facts": [
      "Lisbon",
      "Pixel (cat)",
      "These are the only two durable facts that can be extracted from the conversation, as they represent the user's location and a specific piece of personal information (a pet). The Assistant's statement about being unable to provide information about Pixel's behavior is an instruction or limitation, rather than a fact worth remembering."
    ]
  }
}
```

This script prints the prompt the persona sends for each user after the
restart:

```python
from ovos_memory_plugins.entity import EntityMemory

mem = EntityMemory(config={"backend": "json",
                           "db_path": "~/.local/share/ovos/assistant_facts.json"})
for user in ("alice", "bob"):
    print(user, [(m.role.value, m.content) for m in mem.build_conversation_context("What is my cat called?", user)])
```

Output:

```
alice [('system', "Known facts about the user:\n- Lisbon\n- Pixel (cat)\n- These are the only two durable facts that can be extracted from the conversation, as they represent the user's location and a specific piece of personal information (a pet). The Assistant's statement about being unable to provide information about Pixel's behavior is an instruction or limitation, rather than a fact worth remembering."), ('user', 'What is my cat called?')]
bob [('user', 'What is my cat called?')]
```

Alice's facts reach the model, and Bob's prompt holds nothing of Alice. The
quality of the facts and of the answer depends on the chat model. In this run
`qwen2.5:7b` stored `Pixel (cat)` and also stored a sentence that explains its
own choices as a third fact. After the restart, it answered a question about
the cat with `Your cat is named Pixel. I don't have information on its
behavior.` A different model, or a custom `extraction_prompt`, changes which
facts are kept.

| Problem | Cause | Fix |
|---|---|---|
| The assistant forgets after a restart | `CHAT_MEMORY` is not `transparent`, so the server keeps no memory | Start the server with `CHAT_MEMORY=transparent` |
| The log says `dropped 1 system message(s)` | `allow_system_prompts` is off, so the facts are removed | Set `allow_system_prompts` to `true` |
| The facts file holds wrong or vague facts | The chat model is too small for the extraction prompt | Use a larger model, or set `extraction_prompt` |

The `longterm` backend keeps a rolling summary instead of facts. A summary is
lossy and can leave out a detail such as the cat's name. Use `entity` for facts
that must survive.

---

## Give each family member their own memory

Use this setup when several people use one assistant, and no one may hear what
another person told it.

Every session has its own history. With `scope: session`,
recalled context comes only from the caller's own past exchanges.

The `lexical` backend stores the exchanges of all sessions in one index. Its
default `scope` is `global`, which recalls documents from every session. Set
`scope` to `session` when several people share the assistant.

This script needs no chat server:

```python
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_memory_plugins.lexical import LexicalMemory

def remember(mem, user, said, answer):
    mem.update_history([AgentMessage(MessageRole.USER, said),
                        AgentMessage(MessageRole.ASSISTANT, answer)], user)

for scope in ("global", "session"):
    mem = LexicalMemory(config={"db_path": ":memory:", "scope": scope})
    remember(mem, "alice", "my door code is 4711", "noted, the door code is 4711")
    remember(mem, "bob", "what time is it", "it is noon")
    print(scope, "bob:", [h.content for h in mem.search("door code", session_id="bob")])
    print(scope, "alice:", [h.content for h in mem.search("door code", session_id="alice")])
    mem.close()
```

Output:

```
global bob: ['Q: my door code is 4711\nA: noted, the door code is 4711']
global alice: ['Q: my door code is 4711\nA: noted, the door code is 4711']
session bob: []
session alice: ['Q: my door code is 4711\nA: noted, the door code is 4711']
```

With `ovos-persona-server`, the OpenAI `user` field selects the session when
`CHAT_MEMORY=transparent`. A run with two users on one persona gave:

```
alice> Remember this: my cat is called Pixel. Reply only with OK.
bob>   Remember this: my favourite colour is green. Reply only with OK.
alice> What is my cat called? Answer with the name only.
       Pixel
bob>   What is my cat called? If you do not know, say UNKNOWN.
       UNKNOWN.
bob>   What is my favourite colour? One word.
       Green.
```

| Problem | Cause | Fix |
|---|---|---|
| One user hears another user's facts | A retrieval backend runs with `scope: global` | Set `scope` to `session` |
| All users share one history | The client sends no `user` field | Send a different `user` value for each person |

---

## Answer from your own documents, fully local

Use this setup when you want answers from your own notes or manuals, with no
cloud service and no vector database.

The `lexical` backend indexes your documents in a local SQLite
file and adds the matching passages to every prompt.

1. Index the documents. Save this script as `index_docs.py`:

    ```python
    import sys
    from ovos_memory_plugins.lexical import LexicalMemory

    mem = LexicalMemory(config={"db_path": sys.argv[1]})
    for path in sys.argv[2:]:
        mem.store(open(path).read(), session_id="docs", metadata={"source": path})
    mem.close()
    print(f"indexed {len(sys.argv) - 2} document(s)")
    ```

    ```bash
    python index_docs.py ~/docs.db bakery.md
    ```

    `bakery.md` in the run:

    ```markdown
    # Pixel Bakery opening hours
    The Pixel Bakery on Rua Augusta opens at 07:15 and closes at 14:30 from Monday to Saturday.
    It is closed on Sundays. The house speciality is the cardamom bun.
    ```

2. Point the persona at the same file. Replace `/home/me` with your home
   directory:

    ```json
    {
      "name": "librarian",
      "handlers": ["ovos-chat-openai-plugin"],
      "ovos-chat-openai-plugin": {
        "api_url": "https://llama.smartgic.io/v1",
        "key": "sk-none",
        "model": "qwen2.5:7b",
        "allow_system_prompts": true,
        "system_prompt": "Answer in at most twenty words."
      },
      "memory_module": "ovos-memory-plugin-lexical",
      "ovos-memory-plugin-lexical": {
        "db_path": "/home/me/docs.db",
        "retrieval": {"max_num_results": 3}
      }
    }
    ```

3. Save the persona as `librarian.json` in a directory of its own, for example
   `~/personas`, and ask it questions:

    ```python
    import os
    from ovos_bus_client.session import Session
    from ovos_persona import PersonaService
    from ovos_utils.fakebus import FakeBus

    svc = PersonaService(bus=FakeBus(), config={"personas_path": os.path.expanduser("~/personas")})
    for q in ("When does the Pixel Bakery close on Saturday?",
              "What is the house speciality at the Pixel Bakery?"):
        print("Q", q)
        print("A", "".join(svc.query(q, "librarian", Session("kitchen"))))
    ```

    Output:

    ```
    Q When does the Pixel Bakery close on Saturday?
    A The Pixel Bakery closes at 14:30 on Saturdays.
    Q What is the house speciality at the Pixel Bakery?
    A The house specialty at the Pixel Bakery is the cardamom bun.
    ```

The `lexical` backend also stores every exchange in the same file. With one
user, that is harmless. With several users, use `scope: session` for the
conversation, and keep shared documents in an `ovos-persona-server` vector
store read through `ovos-openai-rag-memory-plugin`. See
[recall scope](./overview.md#recall-scope-one-user-or-several).

| Problem | Cause | Fix |
|---|---|---|
| The answer ignores the document | `allow_system_prompts` is off | Set it to `true` |
| No passage is found | BM25 matches words, not meaning | Use the words of the document, or add `local-rag` in a [composite](./composite.md) |

---
[Home](README.md) · [Overview →](overview.md)
