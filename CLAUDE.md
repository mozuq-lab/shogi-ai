# 将棋AI開発ガイド（蒸留Transformer）

## プロジェクト概要

水匠5の評価値を教師として蒸留した軽量Transformer評価ネットワークを構築する。

## ディレクトリ構成

```
shogi-ai/
├── external/shogi-cli/     # 水匠5 (git submodule)
├── shogi/                  # 将棋関連ユーティリティ
│   └── encoding.py         # SFEN → Tensor変換
├── models/                 # モデル定義
│   └── value_transformer.py
├── data/raw/               # 生成データ
├── tools/                  # データ生成スクリプト
├── train/                  # 学習スクリプト
├── engine/                 # USIエンジン
├── scripts/                # ユーティリティスクリプト
├── env/                    # 環境構築ドキュメント
└── reports/                # 評価レポート
```

## 開発フェーズ

- Phase 0: 環境構築 ✓
- Phase 1: 教師データ生成
- Phase 2: モデル設計・実装
- Phase 3: 学習・オフライン評価
- Phase 4: エンジン組み込み・対局テスト

## 技術スタック

- Python 3.10+
- PyTorch (CUDA)
- cshogi (将棋ライブラリ)
- 水匠5 (やねうら王ベース、USIプロトコル)

## 水匠5の使い方

```bash
cd external/shogi-cli/suisho5
echo -e "usi\nisready\nposition startpos\ngo movetime 1000\nquit" | ./YaneuraOu-mac
```

### USIコマンド

- `position sfen <SFEN>`: 局面設定
- `go movetime <ms>`: 思考（ミリ秒指定）
- `go depth <n>`: 思考（深さ指定）
- 出力: `info ... score cp <評価値> ...` → centipawn単位

## モデル仕様

### 入力
- 81マス × 駒種埋め込み + 手番

### 出力
- スカラー値 [-1, 1] (勝率近似)

### アーキテクチャ（初期案）
- d_model: 256
- n_heads: 4
- n_layers: 4
- ffn_dim: 512

## 評価値の正規化

```python
def normalize_cp(cp: int, scale: float = 1200.0) -> float:
    """centipawnを[-1, 1]に正規化（tanh的なスケーリング）"""
    return math.tanh(cp / scale)
```

## コーディング規約

- 型ヒント必須
- docstring: Google style
- テスト: pytest
