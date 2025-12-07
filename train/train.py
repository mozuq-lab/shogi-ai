"""Value Network学習スクリプト.

使用例:
    # Mac環境での動作確認（小規模）
    python train/train.py --data data/raw/hao_trial_500_depth10.jsonl --epochs 5 --batch-size 64

    # Windows環境での本格学習
    python train/train.py --data data/raw/large_dataset.jsonl --epochs 100 --batch-size 512 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

from models import ShogiValueDataset, ValueTransformer, collate_fn

if TYPE_CHECKING:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """学習設定."""

    # データ
    data_path: str = "data/raw/hao_trial_500_depth10.jsonl"
    val_split: float = 0.1

    # モデル
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1

    # 学習
    epochs: int = 100
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5

    # 評価値正規化
    cp_scale: float = 1200.0

    # デバイス
    device: str = "auto"

    # 保存
    output_dir: str = "checkpoints"
    save_every: int = 10
    log_every: int = 100

    # 再開
    resume: str | None = None

    # 拡張特徴量
    use_features: bool = False

    # 勝敗補助損失
    aux_loss_weight: float = 0.1

    # 学習安定化
    grad_clip_norm: float = 1.0
    label_smoothing: float = 0.05

    # データ前処理
    cp_noise: float = 0.0
    cp_filter_threshold: float | None = None


@dataclass
class TrainState:
    """学習状態."""

    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)


def get_device(device_str: str) -> torch.device:
    """デバイスを取得."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: TrainConfig,
    state: TrainState,
) -> None:
    """チェックポイントを保存."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": vars(config),
        "state": vars(state),
    }
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved: {path}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[TrainConfig, TrainState]:
    """チェックポイントを読み込み."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    config = TrainConfig(**checkpoint["config"])
    state = TrainState(**checkpoint["state"])

    logger.info(f"Checkpoint loaded: {path} (epoch {state.epoch})")
    return config, state


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    state: TrainState,
    log_every: int,
    use_features: bool = False,
    aux_loss_weight: float = 0.1,
    grad_clip_norm: float = 1.0,
    label_smoothing: float = 0.05,
) -> float:
    """1エポック学習."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        board = batch["board"].to(device)
        hand = batch["hand"].to(device)
        turn = batch["turn"].to(device)
        target_value = batch["value"].to(device).unsqueeze(1)
        target_outcome = batch["outcome"].to(device).unsqueeze(1)
        features = batch.get("features")
        if features is not None:
            features = features.to(device)

        # Label Smoothing: [0, 1] → [smoothing, 1 - smoothing]
        smoothed_outcome = target_outcome * (1 - 2 * label_smoothing) + label_smoothing

        optimizer.zero_grad()
        value, outcome = model(board, hand, turn, features)

        # 評価値損失（主タスク）
        value_loss = nn.functional.mse_loss(value, target_value)

        # 勝敗損失（補助タスク、Label Smoothing適用）
        outcome_loss = nn.functional.binary_cross_entropy(outcome, smoothed_outcome)

        # 合計損失
        loss = value_loss + aux_loss_weight * outcome_loss
        loss.backward()

        # Gradient Clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        state.global_step += 1

        if state.global_step % log_every == 0:
            logger.info(
                f"Step {state.global_step}: loss={loss.item():.6f} "
                f"(value={value_loss.item():.6f}, outcome={outcome_loss.item():.6f})"
            )

    return total_loss / num_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_features: bool = False,
    aux_loss_weight: float = 0.1,
) -> float:
    """バリデーション."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        board = batch["board"].to(device)
        hand = batch["hand"].to(device)
        turn = batch["turn"].to(device)
        target_value = batch["value"].to(device).unsqueeze(1)
        target_outcome = batch["outcome"].to(device).unsqueeze(1)
        features = batch.get("features")
        if features is not None:
            features = features.to(device)

        value, outcome = model(board, hand, turn, features)

        # 評価値損失（主タスク）
        value_loss = nn.functional.mse_loss(value, target_value)

        # 勝敗損失（補助タスク）
        outcome_loss = nn.functional.binary_cross_entropy(outcome, target_outcome)

        # 合計損失
        loss = value_loss + aux_loss_weight * outcome_loss

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def train(config: TrainConfig) -> None:
    """学習メイン処理."""
    # デバイス
    device = get_device(config.device)
    logger.info(f"Using device: {device}")

    # データセット
    logger.info(f"Loading dataset: {config.data_path}")
    logger.info(f"Use features: {config.use_features}")
    dataset = ShogiValueDataset(
        config.data_path,
        cp_scale=config.cp_scale,
        use_features=config.use_features,
        cp_noise=config.cp_noise,
        cp_filter_threshold=config.cp_filter_threshold,
    )
    logger.info(f"Dataset size: {len(dataset)}")

    # 訓練/検証分割
    val_size = int(len(dataset) * config.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    logger.info(f"Train: {train_size}, Val: {val_size}")

    # データローダー
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Windows互換性のため
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    # モデル
    model = ValueTransformer(
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        ffn_dim=config.ffn_dim,
        dropout=config.dropout,
        use_features=config.use_features,
    ).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # オプティマイザ・スケジューラ
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.epochs - config.warmup_epochs,
        eta_min=config.lr * 0.01,
    )

    # 状態
    state = TrainState()

    # 再開
    if config.resume:
        resume_path = Path(config.resume)
        if resume_path.exists():
            _, state = load_checkpoint(resume_path, model, optimizer, scheduler)

    # 出力ディレクトリ
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 学習ログ
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"log_{run_name}.json"

    # 学習ループ
    logger.info("Starting training...")
    start_time = time.time()

    for epoch in range(state.epoch, config.epochs):
        state.epoch = epoch
        epoch_start = time.time()

        # Warmup
        if epoch < config.warmup_epochs:
            warmup_lr = config.lr * (epoch + 1) / config.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr

        # 学習
        train_loss = train_epoch(
            model, train_loader, optimizer, device, state, config.log_every,
            use_features=config.use_features,
            aux_loss_weight=config.aux_loss_weight,
            grad_clip_norm=config.grad_clip_norm,
            label_smoothing=config.label_smoothing,
        )
        state.train_losses.append(train_loss)

        # バリデーション
        val_loss = validate(
            model, val_loader, device,
            use_features=config.use_features,
            aux_loss_weight=config.aux_loss_weight,
        )
        state.val_losses.append(val_loss)

        # スケジューラ更新（warmup後）
        if epoch >= config.warmup_epochs:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch + 1}/{config.epochs}: "
            f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
            f"lr={current_lr:.2e}, time={epoch_time:.1f}s"
        )

        # ベストモデル保存
        if val_loss < state.best_val_loss:
            state.best_val_loss = val_loss
            save_checkpoint(
                output_dir / "best.pt",
                model, optimizer, scheduler, config, state,
            )

        # 定期保存
        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch + 1:04d}.pt",
                model, optimizer, scheduler, config, state,
            )

        # ログ保存
        log_data = {
            "config": vars(config),
            "train_losses": state.train_losses,
            "val_losses": state.val_losses,
            "best_val_loss": state.best_val_loss,
        }
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

    # 最終保存
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time / 60:.1f} minutes")
    save_checkpoint(
        output_dir / "final.pt",
        model, optimizer, scheduler, config, state,
    )


def main() -> None:
    """エントリーポイント."""
    parser = argparse.ArgumentParser(description="Value Network学習")
    parser.add_argument("--data", type=str, required=True, help="データファイルパス")
    parser.add_argument("--epochs", type=int, default=100, help="エポック数")
    parser.add_argument("--batch-size", type=int, default=512, help="バッチサイズ")
    parser.add_argument("--lr", type=float, default=3e-4, help="学習率")
    parser.add_argument("--device", type=str, default="auto", help="デバイス")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="出力ディレクトリ")
    parser.add_argument("--resume", type=str, default=None, help="再開するチェックポイント")
    parser.add_argument("--val-split", type=float, default=0.1, help="検証データ割合")
    parser.add_argument("--save-every", type=int, default=10, help="保存間隔（エポック）")
    parser.add_argument("--log-every", type=int, default=100, help="ログ間隔（ステップ）")

    # モデルパラメータ
    parser.add_argument("--d-model", type=int, default=256, help="埋め込み次元")
    parser.add_argument("--n-heads", type=int, default=4, help="アテンションヘッド数")
    parser.add_argument("--n-layers", type=int, default=4, help="レイヤー数")
    parser.add_argument("--ffn-dim", type=int, default=512, help="FFN次元")
    parser.add_argument("--dropout", type=float, default=0.1, help="ドロップアウト率")

    # 拡張特徴量
    parser.add_argument("--use-features", action="store_true", help="拡張特徴量を使用")

    # 補助損失
    parser.add_argument("--aux-loss-weight", type=float, default=0.1, help="勝敗補助損失の重み")

    # 学習安定化
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="勾配クリッピングのmax_norm")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label Smoothingの強度")

    # データ前処理
    parser.add_argument("--cp-noise", type=float, default=0.0, help="評価値ノイズの標準偏差（cp）")
    parser.add_argument("--cp-filter-threshold", type=float, default=None, help="評価値フィルタの閾値（cp）")

    args = parser.parse_args()

    config = TrainConfig(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        output_dir=args.output_dir,
        resume=args.resume,
        val_split=args.val_split,
        save_every=args.save_every,
        log_every=args.log_every,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        use_features=args.use_features,
        aux_loss_weight=args.aux_loss_weight,
        grad_clip_norm=args.grad_clip_norm,
        label_smoothing=args.label_smoothing,
        cp_noise=args.cp_noise,
        cp_filter_threshold=args.cp_filter_threshold,
    )

    train(config)


if __name__ == "__main__":
    main()
