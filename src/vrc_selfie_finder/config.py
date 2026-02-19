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

    # Stage 2: マッチング
    matcher: str = "ccip"  # "clip" or "ccip"
    clip_model_name: str = "ViT-L-14"
    clip_pretrained: str = "laion2b_s32b_b82k"
    ccip_model: str = "ccip-caformer_b36-24"
    similarity_threshold_min: float = 0.87
    similarity_threshold_max: float = 1.0
    clip_batch_size: int = 32

    # 顔切り抜きモード: "face" (顔のみ), "wide" (顔周辺を広めに), "full" (画像全体)
    crop_mode: str = "wide"
    # "wide" モードの拡大率 (bboxの各辺をこの倍率で拡大)
    crop_padding: float = 0.5

    # 処理対象アバター (Noneで全アバター)
    target_avatar: str | None = None

    # 日付フィルタ (YYYY-MM-DD形式、None で全期間)
    since: str | None = None
    until: str | None = None

    # Stage 1のみ実行
    stage1_only: bool = False

    # 横倒し画像の回転検出を試みる
    try_rotations: bool = True

    # デバイス
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
