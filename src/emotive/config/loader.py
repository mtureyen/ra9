"""Configuration loader with mtime-based hot-reload."""

import json
import threading
from pathlib import Path

from .schema import EmotiveConfig

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config.json"


class ConfigManager:
    """Loads config from JSON, reloads when file changes on disk."""

    def __init__(self, path: Path = DEFAULT_CONFIG_PATH) -> None:
        self._path = path
        self._config: EmotiveConfig | None = None
        self._last_mtime: float = 0.0
        self._lock = threading.Lock()

    def get(self) -> EmotiveConfig:
        """Return current config, reloading from disk if the file changed."""
        if not self._path.exists():
            if self._config is None:
                self._config = EmotiveConfig()
            return self._config

        mtime = self._path.stat().st_mtime
        if mtime != self._last_mtime or self._config is None:
            with self._lock:
                # Double-check after acquiring lock
                mtime = self._path.stat().st_mtime
                if mtime != self._last_mtime or self._config is None:
                    self._config = self._load()
                    self._last_mtime = mtime
        return self._config

    def _load(self) -> EmotiveConfig:
        raw = json.loads(self._path.read_text())
        return EmotiveConfig.model_validate(raw)

    def save(self, config: EmotiveConfig) -> None:
        """Write config to disk."""
        with self._lock:
            self._path.write_text(
                json.dumps(config.model_dump(), indent=2) + "\n"
            )
            self._config = config
            self._last_mtime = self._path.stat().st_mtime

    @property
    def path(self) -> Path:
        return self._path
