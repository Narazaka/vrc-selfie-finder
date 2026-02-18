from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .avatar_matcher import crop_face
from .face_detector import FaceDetector


def _load_image(path: Path) -> Image.Image:
    """パスに非ASCII文字が含まれる場合に備えてバイナリ経由で読み込む。"""
    with open(path, "rb") as f:
        return Image.open(f).convert("RGB")


class CCIPMatcher:
    """CCIP (Contrastive Anime Character Image Pre-Training) でキャラクター類似度を計算する。

    差分値ベースのモデルだが、パイプラインとの互換性のため
    similarity = 1 - difference に変換して返す。
    """

    def __init__(self, model: str = "ccip-caformer_b36-24") -> None:
        self._model = model
        # 遅延インポート (onnxruntime初期化が重いため)
        from imgutils.metrics.ccip import ccip_extract_feature, ccip_batch_differences
        self._extract = ccip_extract_feature
        self._batch_diff = ccip_batch_differences

    def _extract_feature(self, img: Image.Image) -> np.ndarray:
        """PIL画像からCCIP特徴ベクトルを抽出する。"""
        return self._extract(img, model=self._model)

    def build_reference_embeddings(
        self,
        reference_dir: Path,
        face_detector: FaceDetector | None = None,
        crop_mode: str = "full",
        crop_padding: float = 0.5,
        batch_size: int = 32,
    ) -> dict[str, np.ndarray]:
        """reference_dir配下のサブフォルダごとに特徴ベクトル群を計算する。"""
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
                        img = crop_face(
                            img, result.bbox, mode=crop_mode,
                            padding=crop_padding, rotation=result.rotation,
                        )
                    pil_images.append(img)
            else:
                pil_images = [_load_image(p) for p in ref_paths]

            # 各リファレンス画像の特徴ベクトルを個別に保持 (N x D)
            features = [self._extract_feature(img) for img in pil_images]
            avatar_embeddings[avatar_dir.name] = np.stack(features)
        return avatar_embeddings

    def encode_candidates(
        self,
        image_paths: list[Path],
        bboxes: list[list[float] | None],
        rotations: list[int] | None = None,
        crop_mode: str = "wide",
        crop_padding: float = 0.5,
        batch_size: int = 32,
    ) -> np.ndarray:
        """候補画像のCCIP特徴ベクトルを計算する。"""
        if rotations is None:
            rotations = [0] * len(image_paths)
        features = []
        for path, bbox, rot in zip(image_paths, bboxes, rotations):
            img = _load_image(path)
            img = crop_face(img, bbox, mode=crop_mode, padding=crop_padding, rotation=rot)
            features.append(self._extract_feature(img))
        return np.stack(features)

    def compute_similarities(
        self,
        candidate_embeddings: np.ndarray,
        reference_embeddings: np.ndarray,
    ) -> np.ndarray:
        """候補と各リファレンスの最小差分値を計算し、similarity = 1 - min_diff で返す。

        ccip_batch_differences を使って効率的にバッチ計算する。
        ref + candidates を結合してペア差分行列を取り、ref-cand部分を抽出する。

        reference_embeddings: (N_ref, D)
        candidate_embeddings: (N_cand, D)
        返り値: (N_cand,) - 各候補のsimilarity (高い = 類似)
        """
        n_ref = reference_embeddings.shape[0]
        n_cand = candidate_embeddings.shape[0]

        # バッチサイズが大きすぎるとOOMになるため、候補をチャンクで処理
        chunk_size = 64
        min_diffs = np.zeros(n_cand, dtype=np.float32)

        for start in range(0, n_cand, chunk_size):
            end = min(start + chunk_size, n_cand)
            chunk = candidate_embeddings[start:end]
            # ref + chunk を結合
            combined = np.concatenate([reference_embeddings, chunk], axis=0)
            # ペア差分行列を計算 (N_ref+chunk_size, N_ref+chunk_size)
            diff_matrix = self._batch_diff(list(combined), model=self._model)
            # ref-cand部分を抽出: 行=ref, 列=cand → 転置して cand x ref
            cross_diffs = diff_matrix[n_ref:, :n_ref]  # (chunk_size, N_ref)
            min_diffs[start:end] = cross_diffs.min(axis=1)

        return 1.0 - min_diffs
