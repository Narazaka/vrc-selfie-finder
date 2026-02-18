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
    rotation: int = 0  # 検出時の回転角度 (0, 90, 180, 270) - bboxは回転後の座標系


class FaceDetector:
    """YOLOv8でアニメ顔を検出し、顔数とバウンディングボックスを返す。"""

    def __init__(
        self,
        model_path: Path,
        device: str,
        confidence: float = 0.4,
        try_rotations: bool = False,
    ) -> None:
        self._model = YOLO(str(model_path))
        self._device = device
        self._confidence = confidence
        self._try_rotations = try_rotations

    def _detect_single(self, img: np.ndarray) -> tuple[int, list[float] | None, float]:
        """単一画像の顔検出。(face_count, bbox, best_conf) を返す。"""
        predictions = self._model.predict(
            [img],
            device=self._device,
            conf=self._confidence,
            verbose=False,
        )
        pred = predictions[0]
        boxes = pred.boxes
        face_count = len(boxes)
        bbox = None
        best_conf = 0.0
        if face_count > 0:
            best_idx = boxes.conf.argmax().item()
            bbox = boxes.xyxy[best_idx].tolist()
            best_conf = float(boxes.conf[best_idx])
        return face_count, bbox, best_conf

    def _detect_with_rotations(self, img: np.ndarray) -> FaceDetectionResult:
        """回転を試みて顔検出を行う。顔が見つからない場合に90°/180°/270°で再試行。"""
        # まず元画像で検出
        face_count, bbox, best_conf = self._detect_single(img)
        if face_count > 0:
            return FaceDetectionResult(face_count=face_count, bbox=bbox, rotation=0)

        # 顔が見つからなかった場合、回転して再試行
        best_result = FaceDetectionResult(face_count=0, bbox=None, rotation=0)
        best_found_conf = 0.0
        for k, angle in [(1, 90), (2, 180), (3, 270)]:
            rotated = np.rot90(img, k=k)
            fc, bb, conf = self._detect_single(rotated)
            if fc > 0 and conf > best_found_conf:
                best_found_conf = conf
                best_result = FaceDetectionResult(face_count=fc, bbox=bb, rotation=angle)

        return best_result

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
                if self._try_rotations:
                    # 回転検出モード: 1枚ずつ処理
                    for k, img in enumerate(images):
                        batch_results[valid_indices[k]] = self._detect_with_rotations(img)
                else:
                    # 通常バッチ処理
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
                            best_idx = boxes.conf.argmax().item()
                            bbox = boxes.xyxy[best_idx].tolist()
                        batch_results[valid_indices[k]] = FaceDetectionResult(
                            face_count=face_count, bbox=bbox, rotation=0
                        )

            all_results.extend(batch_results)
        return all_results
