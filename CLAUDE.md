# 将棋AI開発ガイド（蒸留Transformer）

## プロジェクト概要

水匠5の評価値を教師として蒸留した軽量Transformer評価ネットワークを構築する。
Value Network（局面→評価値）のみを実装対象とし、ポリシーネットは将来の拡張とする。

## 開発環境

| 環境 | 用途 | 備考 |
|------|------|------|
| Mac (Apple Silicon) | 開発、デバッグ、データ生成 | 現在の開発環境 |
| Windows (RTX 4070 Super) | 本格学習 (Phase 3) | CUDA使用 |

データの受け渡しはクラウドストレージを使用する。

## 開発フェーズと進捗

- [x] Phase 0: 環境構築
- [x] Phase 1: 教師データ生成
- [ ] Phase 2: モデル設計・実装
- [ ] Phase 3: 学習・オフライン評価（Windows）
- [ ] Phase 4: エンジン組み込み・対局テスト

## ディレクトリ構成

```
shogi-ai/
├── external/shogi-cli/     # 水匠5 (git submodule)
│   └── suisho5/
│       ├── YaneuraOu-mac   # エンジン本体
│       └── eval/nn.bin     # 評価関数
├── shogi/                  # 将棋関連ユーティリティ
│   ├── __init__.py
│   └── usi_engine.py       # USIエンジンラッパー ✓
├── models/                 # モデル定義
│   └── value_transformer.py  # (Phase 2で実装)
├── data/raw/               # 生成データ (.gitignore対象)
├── tools/
│   └── gen_dataset.py      # データ生成スクリプト ✓
├── train/                  # 学習スクリプト (Phase 3で実装)
├── engine/                 # USIエンジン (Phase 4で実装)
├── scripts/
│   └── usi_test.py         # USI疎通確認スクリプト ✓
├── env/
│   └── README.md           # セットアップ手順
└── reports/                # 評価レポート
```

## 実装済み機能

### USIエンジンラッパー (`shogi/usi_engine.py`)

```python
from shogi import USIEngine, get_default_engine_path

with USIEngine(get_default_engine_path()) as engine:
    engine.init_usi()
    engine.set_option("USI_OwnBook", False)
    engine.is_ready()
    engine.set_position(moves=["7g7f", "3c3d"])
    result = engine.go(depth=10)  # or movetime=500
    print(result.bestmove, result.score_cp)
```

### データ生成スクリプト (`tools/gen_dataset.py`)

水匠5同士の自己対局で教師データを生成する。

```bash
# 深さ10で1対局（デフォルト）
python tools/gen_dataset.py -n 1

# 深さ15で10対局
python tools/gen_dataset.py -n 10 --depth 15 -o data/raw/depth15.jsonl

# 時間指定（500ms）で生成
python tools/gen_dataset.py -n 10 --movetime 500
```

#### 出力形式 (JSONL)

```json
{"sfen": "startpos", "score_cp": 37, "ply": 0, "game_id": 0, "result": "white_win"}
{"sfen": "startpos moves 2g2f", "score_cp": -136, "ply": 1, "game_id": 0, "result": "white_win"}
```

| フィールド | 説明 |
|-----------|------|
| `sfen` | 局面（startpos + 指し手列） |
| `score_cp` | 評価値（centipawn、手番側視点） |
| `ply` | 手数 |
| `game_id` | 対局ID |
| `result` | 対局結果 (`black_win`, `white_win`, `draw`) |

## 水匠5の使い方

```bash
cd external/shogi-cli/suisho5
echo -e "usi\nisready\nposition startpos\ngo depth 10\nquit" | ./YaneuraOu-mac
```

### パフォーマンス (Mac Apple Silicon)

| スレッド数 | nps |
|-----------|-----|
| 1 | 約 1.0M |
| 8 | 約 4.9M |

## Phase 2: モデル設計（次のタスク）

### 入力表現

- 81マスをトークンとして扱う
- 各トークン: 駒種（空、歩〜玉、成駒）× 先後 の埋め込み
- 手番埋め込みを全トークンに加算

### 出力

- スカラー値 [-1, 1]（勝率近似）

### アーキテクチャ（初期案）

```
d_model: 256
n_heads: 4
n_layers: 4
ffn_dim: 512
```

### 評価値の正規化

```python
import math

def normalize_cp(cp: int, scale: float = 1200.0) -> float:
    """centipawnを[-1, 1]に正規化"""
    return math.tanh(cp / scale)
```

## Phase 3: 学習（Windows環境）

### ハイパーパラメータ（初期案）

- データセット: 10万〜100万局面
- 訓練:バリデーション = 9:1
- Optimizer: AdamW
- 学習率: 1e-4〜3e-4
- バッチサイズ: 512〜1024
- 損失関数: MSE

## コーディング規約

- 型ヒント必須
- docstring: Google style
- テスト: pytest
- Python 3.10+互換（`from __future__ import annotations`使用）
