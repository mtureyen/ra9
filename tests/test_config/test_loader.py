"""Tests for config loader with hot-reload."""

import json
import time


def test_get_returns_defaults_when_no_file(tmp_path):
    from emotive.config.loader import ConfigManager

    cm = ConfigManager(tmp_path / "nonexistent.json")
    cfg = cm.get()
    assert cfg.phase == 0
    assert cfg.working_memory_capacity == 20


def test_get_loads_from_json_file(tmp_path):
    from emotive.config.loader import ConfigManager

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"phase": 0, "working_memory_capacity": 30}))
    cm = ConfigManager(path)
    cfg = cm.get()
    assert cfg.working_memory_capacity == 30


def test_get_hot_reloads_on_mtime_change(tmp_path):
    from emotive.config.loader import ConfigManager

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"working_memory_capacity": 20}))
    cm = ConfigManager(path)
    assert cm.get().working_memory_capacity == 20

    time.sleep(0.05)
    path.write_text(json.dumps({"working_memory_capacity": 40}))
    assert cm.get().working_memory_capacity == 40


def test_get_caches_when_mtime_unchanged(tmp_path):
    from emotive.config.loader import ConfigManager

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"phase": 0}))
    cm = ConfigManager(path)
    cfg1 = cm.get()
    cfg2 = cm.get()
    assert cfg1 is cfg2


def test_save_writes_and_updates_cache(tmp_path):
    from emotive.config.loader import ConfigManager
    from emotive.config.schema import EmotiveConfig

    path = tmp_path / "config.json"
    cm = ConfigManager(path)
    cfg = EmotiveConfig(working_memory_capacity=50)
    cm.save(cfg)

    loaded = json.loads(path.read_text())
    assert loaded["working_memory_capacity"] == 50


def test_save_then_get_returns_saved_config(tmp_path):
    from emotive.config.loader import ConfigManager
    from emotive.config.schema import EmotiveConfig

    path = tmp_path / "config.json"
    cm = ConfigManager(path)
    cfg = EmotiveConfig(working_memory_capacity=99)
    cm.save(cfg)
    assert cm.get().working_memory_capacity == 99
