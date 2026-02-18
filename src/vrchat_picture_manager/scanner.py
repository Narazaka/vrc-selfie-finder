from __future__ import annotations

import os
from pathlib import Path


def scan_photos(photo_dir: Path) -> list[Path]:
    """VRChat写真フォルダを再帰走査し、VRChat_で始まるPNGファイルを収集する。"""
    photos: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(photo_dir):
        for fname in filenames:
            if fname.startswith("VRChat_") and fname.lower().endswith(".png"):
                photos.append(Path(dirpath) / fname)
    photos.sort()
    return photos
