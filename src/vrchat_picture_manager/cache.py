from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Cache:
    """JSON形式でStage 1/2の結果をキャッシュする。"""

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self._cache_dir / f"{name}.json"

    def load(self, name: str) -> dict[str, Any]:
        path = self._path(name)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}

    def save(self, name: str, data: dict[str, Any]) -> None:
        path = self._path(name)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
