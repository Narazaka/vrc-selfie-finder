# vrchat-picture-manager

VRChatの写真フォルダから、特定アバターが1人だけ映っている「自撮り写真」を自動抽出するツール。

## 仕組み

2段階のパイプラインで処理する。

1. **Stage 1 (顔検出)**: YOLOv8アニメ顔検出モデルで全写真をスキャンし、顔が1つだけ映っている画像を抽出
2. **Stage 2 (アバター識別)**: CCIP (アニメキャラ特化モデル) または OpenCLIP でリファレンス画像との類似度を計算し、特定アバターの写真を識別

結果はアバターごとに `output/アバター名/` へsymlinkとして出力され、`report.tsv` にスコア付きの一覧が生成される。

## 必要環境

- Python 3.12+
- CUDA対応GPU (推奨)
- [uv](https://docs.astral.sh/uv/)

## セットアップ

```bash
uv sync
```

### モデルの準備

YOLOv8アニメ顔検出モデル ([yolov8x6_animeface.pt](https://huggingface.co/Fuyucchi/yolov8_animeface)) を `models/` に配置する。

CCIPモデルは初回実行時に自動ダウンロードされるため、手動準備は不要。

### リファレンス画像の準備

`reference_images/` にアバターごとのサブフォルダを作成し、それぞれに自撮り写真を5-10枚配置する。

```
reference_images/
  avatar_a/
    ref1.png
    ref2.png
  avatar_b/
    ref1.png
    ref2.png
```

## 使い方

```bash
# フルパイプライン (全アバター)
uv run vpm

# 特定アバターのみ
uv run vpm --avatar avatar_a

# Stage 1のみ (顔検出結果の確認)
uv run vpm --stage1-only

# OpenCLIPで実行
uv run vpm --matcher clip --similarity-threshold 0.85

# 類似度閾値を変更
uv run vpm --similarity-threshold 0.80

# 回転検出を無効化
uv run vpm --no-try-rotations

# 写真フォルダを指定
uv run vpm --photo-dir "D:\VRChat\Pictures"
```

## 主なオプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--photo-dir` | `~/Pictures/VRChat` | VRChat写真フォルダ |
| `--reference-dir` | `reference_images` | リファレンス画像フォルダ |
| `--output-dir` | `output` | 結果出力先 |
| `--model-path` | `models/yolov8x6_animeface.pt` | YOLOv8モデルパス |
| `--matcher` | `ccip` | マッチングモデル (`ccip` / `clip`) |
| `--similarity-threshold` | `0.87` | 類似度の閾値 |
| `--face-confidence` | `0.4` | 顔検出の信頼度閾値 |
| `--crop-mode` | `wide` | 切り抜きモード (`face` / `wide` / `full`) |
| `--crop-padding` | `0.5` | `wide` モードの拡大率 |
| `--try-rotations` / `--no-try-rotations` | 有効 | 横倒し画像の回転検出 |
| `--avatar` | 全アバター | 特定アバターのみ処理 |
| `--since` | 全期間 | この日付以降の写真のみ処理 (YYYY-MM-DD) |
| `--stage1-only` | `false` | Stage 1のみ実行 |
| `--device` | 自動検出 | `cuda` / `cpu` |

## キャッシュ

処理結果は `.cache/` にキャッシュされる。閾値を変更して再実行する場合、顔検出や特徴抽出の再計算はスキップされ数秒で完了する。
