# 環境構築ガイド

## 動作確認済み環境

- macOS (Apple Silicon M1/M2/M3)
- Python 3.10+
- PyTorch 2.0+ (CUDA / MPS)

## セットアップ手順

### 1. リポジトリのクローン

```bash
git clone --recursive <repository-url>
cd shogi-ai
```

既にクローン済みの場合、サブモジュールを取得:

```bash
git submodule update --init --recursive
```

### 2. Python仮想環境の作成

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

### 3. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

#### PyTorch (CUDA版) のインストール

NVIDIA GPU を使用する場合:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### PyTorch (Apple Silicon MPS版)

```bash
pip install torch torchvision
```

### 4. 水匠5の動作確認

```bash
python scripts/usi_test.py
```

期待される出力:

```
水匠5 USI通信テスト

=== USI初期化テスト ===
[OK] USI初期化成功

=== 局面設定・思考テスト ===
[OK] 最善手: 7g7f
[OK] 評価値: 85 cp

=== SFEN局面テスト ===
[OK] SFEN局面評価: 0 cp

=== 結果サマリー ===
  USI初期化: [OK]
  局面設定・思考: [OK]
  SFEN局面: [OK]
```

### 5. Gatekeeper警告の解除 (macOS)

初回実行時に「開発元を検証できない」警告が出る場合:

```bash
xattr -d com.apple.quarantine external/shogi-cli/suisho5/YaneuraOu-mac
```

または、システム環境設定 → セキュリティとプライバシー → 「このまま許可」

## トラブルシューティング

### Error! : failed to read nn.bin

評価関数ファイルが見つからない。サブモジュールが正しく取得されているか確認:

```bash
ls external/shogi-cli/suisho5/eval/nn.bin
```

### Permission denied

実行権限を付与:

```bash
chmod +x external/shogi-cli/suisho5/YaneuraOu-mac
```

### cshogi のインストールエラー

```bash
pip install cshogi --no-build-isolation
```

## ハードウェア要件

### 最小構成
- CPU: 4コア以上
- RAM: 8GB以上
- ストレージ: 10GB以上

### 推奨構成（学習用）
- GPU: NVIDIA RTX 4070 Super (VRAM 12GB)
- RAM: 32GB以上
- ストレージ: SSD 100GB以上
