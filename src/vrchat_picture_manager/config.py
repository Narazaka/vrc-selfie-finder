from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

# デフォルトのVRChat写真フォルダ
DEFAULT_PHOTO_DIR = Path.home() / "Pictures" / "VRChat"


@dataclass
class Config:
    photo_dir: Path = field(default_factory=lambda: DEFAULT_PHOTO_DIR)
    reference_dir: Path = field(default_factory=lambda: Path("reference_images"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    model_path: Path = field(default_factory=lambda: Path("models/yolov8x6_animeface.pt"))
    cache_dir: Path = field(default_factory=lambda: Path(".cache"))

    # Stage 1: 顔検出
    face_confidence_threshold: float = 0.4
    yolo_batch_size: int = 8

    # Stage 2: CLIP類似度
    clip_model_name: str = "ViT-L-14"
    clip_pretrained: str = "laion2b_s32b_b82k"
    similarity_threshold: float = 0.7
    clip_batch_size: int = 32

    # 処理対象アバター (Noneで全アバター)
    target_avatar: str | None = None

    # Stage 1のみ実行
    stage1_only: bool = False

    # デバイス
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
