# 将棋AI（蒸留Transformer）

水匠5の評価値を教師として蒸留した軽量Transformer評価ネットワーク。

## 特徴

- **Transformer ベースの評価関数**: 81マス + 持ち駒をトークンとして処理
- **拡張特徴量**: 利きマップ、玉の安全度、駒価値などを追加入力として使用可能
- **勝敗補助損失**: 評価値予測と勝率予測のマルチタスク学習
- **USIプロトコル対応**: 将棋GUIで対局可能

## セットアップ

```bash
# 依存関係のインストール
pip install torch python-shogi

# サブモジュールの初期化（水匠5エンジン）
git submodule update --init --recursive
```

## クイックスタート

### 学習

```bash
# 基本的な学習
PYTHONPATH=. python train/train.py \
    --data data/raw/dataset.jsonl \
    --epochs 100 \
    --batch-size 512

# 全ての改良を有効にした学習
PYTHONPATH=. python train/train.py \
    --data data/raw/dataset.jsonl \
    --use-features \
    --aux-loss-weight 0.1 \
    --epochs 100 \
    --batch-size 512
```

### データ生成

```bash
# 水匠5で自己対局データを生成
python tools/gen_dataset.py -n 100 --depth 10 -o data/raw/dataset.jsonl
```

### USIエンジンとして使用

```bash
# 起動
./shogi-ai-engine.sh

# または直接実行
PYTHONPATH=. python engine/usi_server.py --model checkpoints/best.pt
```

将棋所やShogiGUIなどのUSI対応GUIで `shogi-ai-engine.sh` をエンジンとして登録すると対局できます。

## 主なオプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--data` | 必須 | データファイルパス |
| `--epochs` | 100 | エポック数 |
| `--batch-size` | 512 | バッチサイズ |
| `--use-features` | - | 拡張特徴量を使用 |
| `--aux-loss-weight` | 0.1 | 勝敗補助損失の重み |
| `--device` | auto | デバイス（auto/cuda/mps/cpu） |

## ディレクトリ構成

```
shogi-ai/
├── models/           # モデル定義
├── train/            # 学習スクリプト
├── engine/           # USIエンジン
├── tools/            # データ生成ツール
├── external/         # 外部エンジン（git submodule）
└── checkpoints/      # 学習済みモデル
```

## 開発ドキュメント

詳細な開発ガイドは [CLAUDE.md](CLAUDE.md) を参照してください。

## ライセンス

MIT License
