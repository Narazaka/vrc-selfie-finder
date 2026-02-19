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
        "--matcher",
        type=str,
        choices=["clip", "ccip"],
        default="ccip",
        help="マッチングモデル (clip=OpenCLIP, ccip=アニメキャラ特化CCIP)",
    )
    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.87,
        help="類似度の下限閾値",
    )
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=1.0,
        help="類似度の上限閾値",
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
        "--since",
        type=str,
        default=None,
        help="この日付以降の写真のみ処理 (YYYY-MM-DD形式)",
    )
    parser.add_argument(
        "--until",
        type=str,
        default=None,
        help="この日付以前の写真のみ処理 (YYYY-MM-DD形式、指定日を含む)",
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
    parser.add_argument(
        "--crop-mode",
        type=str,
        choices=["face", "wide", "full"],
        default="wide",
        help="CLIP比較時の切り抜きモード (face=顔のみ, wide=顔周辺広め, full=画像全体)",
    )
    parser.add_argument(
        "--crop-padding",
        type=float,
        default=0.5,
        help="wide モードでの切り抜き拡大率",
    )
    parser.add_argument(
        "--try-rotations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="横倒し画像の回転検出を試みる (90°/180°/270°)",
    )

    args = parser.parse_args()

    config = Config(
        photo_dir=args.photo_dir,
        reference_dir=args.reference_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        matcher=args.matcher,
        face_confidence_threshold=args.face_confidence,
        yolo_batch_size=args.yolo_batch_size,
        similarity_threshold_min=args.threshold_min,
        similarity_threshold_max=args.threshold_max,
        clip_batch_size=args.clip_batch_size,
        target_avatar=args.avatar,
        since=args.since,
        until=args.until,
        stage1_only=args.stage1_only,
        crop_mode=args.crop_mode,
        crop_padding=args.crop_padding,
        try_rotations=args.try_rotations,
    )

    if args.device:
        config.device = args.device

    run_pipeline(config)
