from __future__ import annotations

from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image


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
    def _encode_images(self, image_paths: list[Path], batch_size: int = 32) -> np.ndarray:
        """画像群のCLIP埋め込みベクトルを計算する。"""
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            images = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                images.append(self._preprocess(img))
            batch_tensor = torch.stack(images).to(self._device)
            embeddings = self._model.encode_image(batch_tensor)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu().numpy())
        return np.concatenate(all_embeddings, axis=0)

    def build_reference_embeddings(
        self, reference_dir: Path, batch_size: int = 32
    ) -> dict[str, np.ndarray]:
        """reference_dir配下のサブフォルダごとに平均埋め込みベクトルを計算する。"""
        avatar_embeddings: dict[str, np.ndarray] = {}
        for avatar_dir in sorted(reference_dir.iterdir()):
            if not avatar_dir.is_dir():
                continue
            ref_images = [
                p
                for p in sorted(avatar_dir.iterdir())
                if p.suffix.lower() in (".png", ".jpg", ".jpeg")
            ]
            if not ref_images:
                continue
            embeddings = self._encode_images(ref_images, batch_size=batch_size)
            avatar_embeddings[avatar_dir.name] = embeddings.mean(axis=0)
        return avatar_embeddings

    def compute_similarities(
        self,
        candidate_embeddings: np.ndarray,
        reference_embedding: np.ndarray,
    ) -> np.ndarray:
        """候補画像の埋め込みとリファレンスのコサイン類似度を計算する。"""
        ref = reference_embedding / np.linalg.norm(reference_embedding)
        return candidate_embeddings @ ref

    def encode_candidates(self, image_paths: list[Path], batch_size: int = 32) -> np.ndarray:
        """候補画像のCLIP埋め込みを計算する。"""
        return self._encode_images(image_paths, batch_size=batch_size)
