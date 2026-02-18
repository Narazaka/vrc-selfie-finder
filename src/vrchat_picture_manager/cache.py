from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Cache:
    """JSONL (JSON Lines) 形式でキャッシュする。

    1行につき1エントリ: {"key": "...", "value": ...}
    追記が容易で、途中killしても既存データが失われない。
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self._cache_dir / f"{name}.jsonl"

    def load(self, name: str) -> dict[str, Any]:
        """JSONL ファイルを読み込み、dict として返す。重複キーは後勝ち。"""
        path = self._path(name)
        data: dict[str, Any] = {}
        if not path.exists():
            return data
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                data[entry["key"]] = entry["value"]
        return data

    def append(self, name: str, key: str, value: Any) -> None:
        """1エントリを追記する。"""
        path = self._path(name)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")

    def append_batch(self, name: str, entries: list[tuple[str, Any]]) -> None:
        """複数エントリを一括追記する。"""
        path = self._path(name)
        with open(path, "a", encoding="utf-8") as f:
            for key, value in entries:
                f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")
