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
    --normalize-turn \
    --augment-flip \
    --cp-noise 7.5 \
    --cp-filter-threshold 1500 \
    --num-workers 4 \
    --epochs 100 \
    --batch-size 512
```

### データ生成

```bash
# 水匠5で自己対局データを生成
python tools/gen_dataset.py -n 100 --depth 10 -o data/raw/dataset.jsonl

# 弱いAIとの対局データ生成
python tools/gen_dataset.py -n 100 --weak-side white --weak-prob 0.3 --random-opening 0
```

### 棋譜確認（KIF形式変換）

```bash
# 対局をKIF形式で出力（ShogiGUIで評価値グラフ表示可能）
python scripts/to_kif.py data/raw/dataset.jsonl 0 -o game0.kif
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
| `--lr` | 3e-4 | 学習率 |
| `--use-features` | - | 拡張特徴量を使用 |
| `--aux-loss-weight` | 0.1 | 勝敗補助損失の重み |
| `--device` | auto | デバイス（auto/cuda/mps/cpu） |
| `--normalize-turn` | - | 後手番を先手視点に正規化 |
| `--augment-flip` | - | 左右反転でデータ2倍化 |
| `--cp-noise` | 0 | 評価値ノイズの標準偏差（cp） |
| `--cp-filter-threshold` | - | 極端な評価値を除外する閾値 |
| `--grad-clip-norm` | 1.0 | 勾配クリッピングのmax_norm |
| `--label-smoothing` | 0.05 | Label Smoothingの強度 |
| `--num-workers` | 0 | データローダーの並列ワーカー数 |

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
