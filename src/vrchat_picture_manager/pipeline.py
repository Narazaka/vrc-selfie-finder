from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .avatar_matcher import AvatarMatcher
from .cache import Cache
from .config import Config
from .face_detector import FaceDetector
from .scanner import scan_photos


def run_stage1(config: Config, cache: Cache) -> list[Path]:
    """Stage 1: 顔が1つだけの画像をフィルタリングする。"""
    print("[Stage 1] 写真スキャン中...")
    all_photos = scan_photos(config.photo_dir)
    print(f"  写真数: {len(all_photos)}")

    face_cache = cache.load("face_counts")

    # キャッシュ済みをスキップ
    uncached = [p for p in all_photos if str(p) not in face_cache]
    print(f"  未処理: {len(uncached)}, キャッシュ済み: {len(all_photos) - len(uncached)}")

    if uncached:
        print("[Stage 1] YOLOv8 顔検出中...")
        detector = FaceDetector(
            model_path=config.model_path,
            device=config.device,
            confidence=config.face_confidence_threshold,
        )

        # バッチ処理 + tqdm進捗表示
        batch_size = config.yolo_batch_size
        for i in tqdm(range(0, len(uncached), batch_size), desc="顔検出"):
            batch = uncached[i : i + batch_size]
            counts = detector.count_faces_batch(batch, batch_size=batch_size)
            for path, count in zip(batch, counts):
                face_cache[str(path)] = count

        cache.save("face_counts", face_cache)

    # 顔が1つだけの画像を抽出
    single_face = [p for p in all_photos if face_cache.get(str(p)) == 1]
    print(f"  顔が1つの画像: {len(single_face)} / {len(all_photos)}")
    return single_face


def run_stage2(
    config: Config,
    cache: Cache,
    candidates: list[Path],
) -> None:
    """Stage 2: CLIP類似度で特定アバターを識別し、結果を出力する。"""
    print("[Stage 2] CLIP モデルロード中...")
    matcher = AvatarMatcher(
        model_name=config.clip_model_name,
        pretrained=config.clip_pretrained,
        device=config.device,
    )

    # リファレンス埋め込みを計算
    print("[Stage 2] リファレンス画像の埋め込み計算中...")
    avatar_embeddings = matcher.build_reference_embeddings(
        config.reference_dir, batch_size=config.clip_batch_size
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

    # 候補画像のCLIP埋め込みを計算 (キャッシュ対応)
    embeddings_cache_path = config.cache_dir / "clip_embeddings.npz"
    embedding_index_cache = cache.load("clip_embedding_index")

    candidate_strs = [str(p) for p in candidates]
    cached_indices = []
    uncached_indices = []
    for i, s in enumerate(candidate_strs):
        if s in embedding_index_cache:
            cached_indices.append(i)
        else:
            uncached_indices.append(i)

    print(
        f"  候補画像: {len(candidates)}, "
        f"未処理: {len(uncached_indices)}, キャッシュ済み: {len(cached_indices)}"
    )

    # 既存の埋め込みキャッシュを読み込み
    cached_embeddings: dict[str, np.ndarray] = {}
    if embeddings_cache_path.exists():
        with np.load(str(embeddings_cache_path)) as data:
            for key in data.files:
                cached_embeddings[key] = data[key]

    # 未処理の画像を処理
    if uncached_indices:
        print("[Stage 2] 候補画像のCLIP埋め込み計算中...")
        uncached_paths = [candidates[i] for i in uncached_indices]
        batch_size = config.clip_batch_size
        for i in tqdm(range(0, len(uncached_paths), batch_size), desc="CLIP埋め込み"):
            batch = uncached_paths[i : i + batch_size]
            embeddings = matcher.encode_candidates(batch, batch_size=batch_size)
            for path, emb in zip(batch, embeddings):
                cached_embeddings[str(path)] = emb
                embedding_index_cache[str(path)] = True

        # 埋め込みキャッシュを保存
        np.savez(str(embeddings_cache_path), **cached_embeddings)
        cache.save("clip_embedding_index", embedding_index_cache)

    # 全候補の埋め込み行列を構築
    all_embeddings = np.stack([cached_embeddings[str(p)] for p in candidates])

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

    single_face_photos = run_stage1(config, cache)

    if config.stage1_only:
        print("\n[完了] Stage 1 のみ実行しました。")
        return

    if not single_face_photos:
        print("\n顔が1つだけの画像が見つかりませんでした。")
        return

    run_stage2(config, cache, single_face_photos)
    print("\n[完了] パイプライン完了。")
