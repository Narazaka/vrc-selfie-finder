from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .avatar_matcher import AvatarMatcher
from .cache import Cache
from .ccip_matcher import CCIPMatcher
from .config import Config
from .face_detector import FaceDetector
from .scanner import scan_photos


def run_stage1(config: Config, cache: Cache) -> tuple[list[Path], dict[str, dict]]:
    """Stage 1: 顔が1つだけの画像をフィルタリングする。

    Returns:
        (顔が1つの画像リスト, {パス文字列: {"bbox": [...], "rotation": 0}})
    """
    print("[Stage 1] 写真スキャン中...")
    all_photos = scan_photos(config.photo_dir, since=config.since)
    since_msg = f" ({config.since} 以降)" if config.since else ""
    print(f"  写真数: {len(all_photos)}{since_msg}")

    cache_name = "face_detections_rot" if config.try_rotations else "face_detections"
    face_cache = cache.load(cache_name)

    # キャッシュ済みをスキップ
    uncached = [p for p in all_photos if str(p) not in face_cache]
    print(f"  未処理: {len(uncached)}, キャッシュ済み: {len(all_photos) - len(uncached)}")

    if uncached:
        rot_msg = " (回転検出有効)" if config.try_rotations else ""
        print(f"[Stage 1] YOLOv8 顔検出中...{rot_msg}")
        detector = FaceDetector(
            model_path=config.model_path,
            device=config.device,
            confidence=config.face_confidence_threshold,
            try_rotations=config.try_rotations,
        )

        # バッチ処理 + tqdm進捗表示 (バッチごとに追記保存)
        batch_size = config.yolo_batch_size
        pbar = tqdm(total=len(uncached), desc="顔検出", unit="枚")
        for i in range(0, len(uncached), batch_size):
            batch = uncached[i : i + batch_size]
            results = detector.detect_batch(batch, batch_size=batch_size)
            entries = []
            for path, result in zip(batch, results):
                value = {
                    "face_count": result.face_count,
                    "bbox": result.bbox,
                    "rotation": result.rotation,
                }
                face_cache[str(path)] = value
                entries.append((str(path), value))
            cache.append_batch(cache_name, entries)
            pbar.update(len(batch))
        pbar.close()

    # 顔が1つだけの画像を抽出
    single_face = []
    face_info_map: dict[str, dict] = {}
    for p in all_photos:
        entry = face_cache.get(str(p))
        if entry is not None and entry["face_count"] == 1:
            single_face.append(p)
            face_info_map[str(p)] = {
                "bbox": entry["bbox"],
                "rotation": entry.get("rotation", 0),
            }

    print(f"  顔が1つの画像: {len(single_face)} / {len(all_photos)}")
    return single_face, face_info_map


def run_stage2(
    config: Config,
    cache: Cache,
    candidates: list[Path],
    face_info_map: dict[str, dict],
) -> None:
    """Stage 2: 類似度で特定アバターを識別し、結果を出力する。"""
    if config.matcher == "ccip":
        print(f"[Stage 2] CCIP モデルロード中... ({config.ccip_model})")
        matcher: AvatarMatcher | CCIPMatcher = CCIPMatcher(model=config.ccip_model)
    else:
        print("[Stage 2] CLIP モデルロード中...")
        matcher = AvatarMatcher(
            model_name=config.clip_model_name,
            pretrained=config.clip_pretrained,
            device=config.device,
        )

    # リファレンス画像にも顔検出+切り抜きを適用するため、検出器を渡す
    face_detector = None
    if config.crop_mode != "full":
        print("[Stage 2] リファレンス画像の顔検出中...")
        face_detector = FaceDetector(
            model_path=config.model_path,
            device=config.device,
            confidence=config.face_confidence_threshold,
        )

    # リファレンス埋め込みを計算
    print("[Stage 2] リファレンス画像の埋め込み計算中...")
    avatar_embeddings = matcher.build_reference_embeddings(
        config.reference_dir,
        face_detector=face_detector,
        crop_mode=config.crop_mode,
        crop_padding=config.crop_padding,
        batch_size=config.clip_batch_size,
    )
    if not avatar_embeddings:
        print("  リファレンス画像が見つかりません。reference_images/ を確認してください。")
        return

    avatar_names = list(avatar_embeddings.keys())
    if config.target_avatar:
        if config.target_avatar not in avatar_names:
            print(f"  アバター '{config.target_avatar}' が見つかりません。")
            print(f"  利用可能: {', '.join(avatar_names)}")
            return
        avatar_names = [config.target_avatar]

    print(f"  対象アバター: {', '.join(avatar_names)}")
    print(f"  切り抜きモード: {config.crop_mode}")

    # 候補画像のCLIP埋め込みを計算 (キャッシュ対応)
    # crop_modeごとに別キャッシュ
    rot_tag = "_rot" if config.try_rotations else ""
    matcher_tag = f"_{config.matcher}" if config.matcher != "clip" else ""
    cache_suffix = f"_{config.crop_mode}{rot_tag}{matcher_tag}"
    embeddings_npy_path = config.cache_dir / f"clip_embeddings{cache_suffix}.npy"
    index_cache_name = f"clip_embedding_index{cache_suffix}"
    embedding_index_cache = cache.load(index_cache_name)

    # 既存の埋め込みキャッシュを読み込み
    cached_matrix: np.ndarray | None = None
    if embeddings_npy_path.exists() and embedding_index_cache:
        cached_matrix = np.load(str(embeddings_npy_path))

    candidate_strs = [str(p) for p in candidates]
    uncached_indices = [i for i, s in enumerate(candidate_strs) if s not in embedding_index_cache]

    print(
        f"  候補画像: {len(candidates)}, "
        f"未処理: {len(uncached_indices)}, キャッシュ済み: {len(candidates) - len(uncached_indices)}"
    )

    # 未処理の画像を処理 (バッチごとにキャッシュ保存)
    if uncached_indices:
        print("[Stage 2] 候補画像のCLIP埋め込み計算中...")
        uncached_paths = [candidates[i] for i in uncached_indices]
        uncached_bboxes = [face_info_map[str(p)]["bbox"] for p in uncached_paths]
        uncached_rotations = [face_info_map[str(p)]["rotation"] for p in uncached_paths]
        batch_size = config.clip_batch_size
        next_idx = cached_matrix.shape[0] if cached_matrix is not None else 0
        pbar2 = tqdm(total=len(uncached_paths), desc="CLIP埋め込み", unit="枚")
        for i in range(0, len(uncached_paths), batch_size):
            batch_paths = uncached_paths[i : i + batch_size]
            batch_bboxes = uncached_bboxes[i : i + batch_size]
            batch_rotations = uncached_rotations[i : i + batch_size]
            embeddings = matcher.encode_candidates(
                batch_paths,
                batch_bboxes,
                rotations=batch_rotations,
                crop_mode=config.crop_mode,
                crop_padding=config.crop_padding,
                batch_size=batch_size,
            )

            # キャッシュ行列に追記
            if cached_matrix is not None:
                cached_matrix = np.concatenate([cached_matrix, embeddings], axis=0)
            else:
                cached_matrix = embeddings

            # インデックスを更新・追記保存
            entries = []
            for path in batch_paths:
                entries.append((str(path), next_idx))
                embedding_index_cache[str(path)] = next_idx
                next_idx += 1

            np.save(str(embeddings_npy_path), cached_matrix)
            cache.append_batch(index_cache_name, entries)

            pbar2.update(len(batch_paths))
        pbar2.close()

    # 全候補の埋め込み行列を構築
    row_indices = [embedding_index_cache[str(p)] for p in candidates]
    all_embeddings = cached_matrix[row_indices]

    # アバターごとに類似度計算・結果出力
    for avatar_name in avatar_names:
        ref_emb = avatar_embeddings[avatar_name]
        similarities = matcher.compute_similarities(all_embeddings, ref_emb)

        # 閾値以上の画像を抽出
        matches = [
            (candidates[i], float(similarities[i]))
            for i in range(len(candidates))
            if similarities[i] >= config.similarity_threshold
        ]
        matches.sort(key=lambda x: x[1], reverse=True)

        print(f"  [{avatar_name}] マッチ: {len(matches)} 枚 (閾値: {config.similarity_threshold})")

        # 出力ディレクトリ作成
        avatar_output = config.output_dir / avatar_name
        avatar_output.mkdir(parents=True, exist_ok=True)

        # 既存のsymlinkを削除
        for existing in avatar_output.iterdir():
            if existing.is_symlink():
                existing.unlink()

        # symlinkと report.tsv を作成
        report_path = avatar_output / "report.tsv"
        with open(report_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["similarity", "path"])
            for photo_path, sim in matches:
                writer.writerow([f"{sim:.4f}", str(photo_path)])
                link_name = avatar_output / photo_path.name
                try:
                    os.symlink(photo_path, link_name)
                except OSError:
                    pass  # 同名ファイルがある場合はスキップ

        print(f"  [{avatar_name}] 出力: {avatar_output}")


def run_pipeline(config: Config) -> None:
    """Stage 1→2 のパイプラインを実行する。"""
    cache = Cache(config.cache_dir)

    single_face_photos, face_info_map = run_stage1(config, cache)

    if config.stage1_only:
        print("\n[完了] Stage 1 のみ実行しました。")
        return

    if not single_face_photos:
        print("\n顔が1つだけの画像が見つかりませんでした。")
        return

    run_stage2(config, cache, single_face_photos, face_info_map)
    print("\n[完了] パイプライン完了。")
