from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO


def _load_image(path: Path) -> np.ndarray | None:
    """画像を読み込む。失敗時はNoneを返す。"""
    try:
        with open(path, "rb") as f:
            return np.array(Image.open(f).convert("RGB"))
    except OSError as e:
        print(f"  [WARN] 画像読み込みスキップ: {path} ({e})", file=sys.stderr)
        return None


@dataclass
class FaceDetectionResult:
    face_count: int  # -1 = 読み込み失敗
    bbox: list[float] | None = None  # [x1, y1, x2, y2] 最も信頼度の高い顔


class FaceDetector:
    """YOLOv8でアニメ顔を検出し、顔数とバウンディングボックスを返す。"""

    def __init__(self, model_path: Path, device: str, confidence: float = 0.4) -> None:
        self._model = YOLO(str(model_path))
        self._device = device
        self._confidence = confidence

    def detect_batch(self, image_paths: list[Path], batch_size: int = 8) -> list[FaceDetectionResult]:
        """画像群の顔検出を行い、顔数とバウンディングボックスを返す。"""
        all_results: list[FaceDetectionResult] = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            images = []
            valid_indices = []
            for j, p in enumerate(batch_paths):
                img = _load_image(p)
                if img is not None:
                    images.append(img)
                    valid_indices.append(j)

            batch_results = [FaceDetectionResult(face_count=-1)] * len(batch_paths)

            if images:
                predictions = self._model.predict(
                    images,
                    device=self._device,
                    conf=self._confidence,
                    verbose=False,
                )
                for k, pred in enumerate(predictions):
                    boxes = pred.boxes
                    face_count = len(boxes)
                    bbox = None
                    if face_count > 0:
                        # 最も信頼度の高い顔のバウンディングボックス
                        best_idx = boxes.conf.argmax().item()
                        bbox = boxes.xyxy[best_idx].tolist()
                    batch_results[valid_indices[k]] = FaceDetectionResult(
                        face_count=face_count, bbox=bbox
                    )

            all_results.extend(batch_results)
        return all_results
