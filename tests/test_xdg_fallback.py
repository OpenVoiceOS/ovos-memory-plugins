"""Default store paths fall back to ~/.local/share when XDG_DATA_HOME is unset."""
import pytest
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

from ovos_memory_plugins.entity import EntityMemory
from ovos_memory_plugins.lexical import LexicalMemory
from ovos_memory_plugins.longterm import LongTermMemory


@pytest.mark.parametrize("plugin,config,name", [
    (EntityMemory, {"backend": "json"}, "entity_memory.json"),
    (LexicalMemory, {}, "lexical_memory.db"),
    (LongTermMemory, {"backend": "json"}, "longterm_memory.json"),
    (LongTermMemory, {"backend": "sqlite"}, "longterm_memory.db"),
])
def test_default_db_path_falls_back_to_home(tmp_path, monkeypatch, plugin, config, name):
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    mem = plugin(config={**config, "model": "m"})
    if plugin is EntityMemory:
        mem._merge_facts("s1", ["Likes tea"])
    else:
        mem.update_history([AgentMessage(role=MessageRole.USER, content="hi"),
                            AgentMessage(role=MessageRole.ASSISTANT, content="hello")], "s1")
        mem.close()
    assert (tmp_path / ".local" / "share" / "ovos" / name).is_file()
