from __future__ import annotations

from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

from .face_detector import FaceDetector


def _load_image(path: Path) -> Image.Image:
    """パスに非ASCII文字が含まれる場合に備えてバイナリ経由で読み込む。"""
    with open(path, "rb") as f:
        return Image.open(f).convert("RGB")


def crop_face(img: Image.Image, bbox: list[float] | None, mode: str, padding: float = 0.5) -> Image.Image:
    """バウンディングボックスに基づいて顔を切り抜く。

    mode:
        "face" - バウンディングボックスそのまま
        "wide" - padding倍率で拡大 (髪型・服装も含む)
        "full" - 切り抜きなし (元画像をそのまま返す)
    """
    if mode == "full" or bbox is None:
        return img

    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    img_w, img_h = img.size

    if mode == "wide":
        x1 = max(0, x1 - w * padding)
        y1 = max(0, y1 - h * padding)
        x2 = min(img_w, x2 + w * padding)
        y2 = min(img_h, y2 + h * padding)

    return img.crop((int(x1), int(y1), int(x2), int(y2)))


class AvatarMatcher:
    """OpenCLIPでリファレンス画像との類似度を計算する。"""

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "laion2b_s32b_b82k",
        device: str = "cuda",
    ) -> None:
        self._device = device
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self._model.eval()

    @torch.no_grad()
    def _encode_pil_images(self, pil_images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        """PIL画像群のCLIP埋め込みベクトルを計算する。"""
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i : i + batch_size]
            tensors = [self._preprocess(img) for img in batch]
            batch_tensor = torch.stack(tensors).to(self._device)
            embeddings = self._model.encode_image(batch_tensor)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu().numpy())
        return np.concatenate(all_embeddings, axis=0)

    def build_reference_embeddings(
        self,
        reference_dir: Path,
        face_detector: FaceDetector | None = None,
        crop_mode: str = "full",
        crop_padding: float = 0.5,
        batch_size: int = 32,
    ) -> dict[str, np.ndarray]:
        """reference_dir配下のサブフォルダごとに埋め込みベクトル群を計算する。

        face_detectorが渡された場合、リファレンス画像にも顔検出+切り抜きを適用する。
        各リファレンス画像の埋め込みを個別に保持する（最大類似度方式用）。
        """
        avatar_embeddings: dict[str, np.ndarray] = {}
        for avatar_dir in sorted(reference_dir.iterdir()):
            if not avatar_dir.is_dir():
                continue
            ref_paths = [
                p for p in sorted(avatar_dir.iterdir())
                if p.suffix.lower() in (".png", ".jpg", ".jpeg")
            ]
            if not ref_paths:
                continue

            # リファレンス画像に顔検出+切り抜きを適用
            pil_images = []
            if face_detector is not None and crop_mode != "full":
                results = face_detector.detect_batch(ref_paths, batch_size=batch_size)
                for path, result in zip(ref_paths, results):
                    img = _load_image(path)
                    if result.bbox is not None:
                        img = crop_face(img, result.bbox, mode=crop_mode, padding=crop_padding)
                    pil_images.append(img)
            else:
                pil_images = [_load_image(p) for p in ref_paths]

            # 各リファレンス画像の埋め込みを個別に保持 (N x D)
            embeddings = self._encode_pil_images(pil_images, batch_size=batch_size)
            avatar_embeddings[avatar_dir.name] = embeddings
        return avatar_embeddings

    def encode_candidates(
        self,
        image_paths: list[Path],
        bboxes: list[list[float] | None],
        crop_mode: str = "face",
        crop_padding: float = 0.5,
        batch_size: int = 32,
    ) -> np.ndarray:
        """候補画像のCLIP埋め込みを計算する（顔切り抜き対応）。"""
        pil_images = []
        for path, bbox in zip(image_paths, bboxes):
            img = _load_image(path)
            img = crop_face(img, bbox, mode=crop_mode, padding=crop_padding)
            pil_images.append(img)
        return self._encode_pil_images(pil_images, batch_size=batch_size)

    def compute_similarities(
        self,
        candidate_embeddings: np.ndarray,
        reference_embeddings: np.ndarray,
    ) -> np.ndarray:
        """候補画像の埋め込みとリファレンス群の最大コサイン類似度を計算する。

        reference_embeddings: (N_ref, D) - 各リファレンス画像の埋め込み
        candidate_embeddings: (N_cand, D) - 各候補画像の埋め込み
        返り値: (N_cand,) - 各候補の最大類似度
        """
        # リファレンスを正規化
        ref_norms = np.linalg.norm(reference_embeddings, axis=1, keepdims=True)
        ref_normed = reference_embeddings / ref_norms
        # 候補×リファレンスの類似度行列 (N_cand, N_ref)
        sim_matrix = candidate_embeddings @ ref_normed.T
        # 各候補について最大類似度を返す
        return sim_matrix.max(axis=1)
