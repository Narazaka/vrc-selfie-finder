from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


class FaceDetector:
    """YOLOv8でアニメ顔を検出し、顔数をカウントする。"""

    def __init__(self, model_path: Path, device: str, confidence: float = 0.4) -> None:
        self._model = YOLO(str(model_path))
        self._device = device
        self._confidence = confidence

    def count_faces_batch(self, image_paths: list[Path], batch_size: int = 8) -> list[int]:
        """画像群の顔数をバッチ推論でカウントする。"""
        face_counts: list[int] = []
        for i in range(0, len(image_paths), batch_size):
            batch = [str(p) for p in image_paths[i : i + batch_size]]
            results = self._model.predict(
                batch,
                device=self._device,
                conf=self._confidence,
                verbose=False,
            )
            for result in results:
                face_counts.append(len(result.boxes))
        return face_counts
