from __future__ import annotations

import os
import re
from pathlib import Path

# ファイル名から YYYY-MM-DD を抽出 (名前フォーマットの変遷に対応)
_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _extract_date(fname: str) -> str | None:
    m = _DATE_RE.search(fname)
    return m.group(1) if m else None


def scan_photos(
    photo_dir: Path,
    since: str | None = None,
    until: str | None = None,
) -> list[Path]:
    """VRChat写真フォルダを再帰走査し、VRChat_で始まるPNGファイルを収集する。

    since: "YYYY-MM-DD" 形式。指定時はその日付以降の写真のみ返す。
    until: "YYYY-MM-DD" 形式。指定時はその日付以前の写真のみ返す（指定日を含む）。
    """
    photos: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(photo_dir):
        for fname in filenames:
            if not (fname.startswith("VRChat_") and fname.lower().endswith(".png")):
                continue
            if since or until:
                date = _extract_date(fname)
                if date is None:
                    continue
                if since and date < since:
                    continue
                if until and date > until:
                    continue
            photos.append(Path(dirpath) / fname)
    photos.sort()
    return photos
