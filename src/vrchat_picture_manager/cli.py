from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_PHOTO_DIR, Config
from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VRChat自撮り写真抽出ツール",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--photo-dir",
        type=Path,
        default=DEFAULT_PHOTO_DIR,
        help="VRChat写真フォルダのパス",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("reference_images"),
        help="リファレンス画像フォルダのパス",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="出力先フォルダのパス",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/yolov8x6_animeface.pt"),
        help="YOLOv8モデルファイルのパス",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.7,
        help="CLIP類似度の閾値",
    )
    parser.add_argument(
        "--face-confidence",
        type=float,
        default=0.4,
        help="顔検出の信頼度閾値",
    )
    parser.add_argument(
        "--avatar",
        type=str,
        default=None,
        help="特定アバターのみ処理 (省略時は全アバター)",
    )
    parser.add_argument(
        "--stage1-only",
        action="store_true",
        help="Stage 1 (顔検出) のみ実行",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="推論デバイス (cuda/cpu、省略時は自動検出)",
    )
    parser.add_argument(
        "--yolo-batch-size",
        type=int,
        default=8,
        help="YOLOv8のバッチサイズ",
    )
    parser.add_argument(
        "--clip-batch-size",
        type=int,
        default=32,
        help="CLIPのバッチサイズ",
    )

    args = parser.parse_args()

    config = Config(
        photo_dir=args.photo_dir,
        reference_dir=args.reference_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        face_confidence_threshold=args.face_confidence,
        yolo_batch_size=args.yolo_batch_size,
        similarity_threshold=args.similarity_threshold,
        clip_batch_size=args.clip_batch_size,
        target_avatar=args.avatar,
        stage1_only=args.stage1_only,
    )

    if args.device:
        config.device = args.device

    run_pipeline(config)
