"""将棋局面データセット."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from models.sfen_parser import parse_sfen
from models.value_transformer import normalize_cp
from models.features import compute_all_features

if TYPE_CHECKING:
    pass


class ShogiValueDataset(Dataset):
    """将棋局面評価値データセット.

    JONLファイルから局面と評価値を読み込み、モデル入力形式に変換する。

    Args:
        data_path: JONLデータファイルのパス
        cp_scale: centipawn正規化のスケール（デフォルト: 1200）
        use_features: 拡張特徴量を使用するかどうか（デフォルト: False）
        cp_noise: 評価値に加えるノイズの標準偏差（デフォルト: 0、無効）
        cp_filter_threshold: 評価値フィルタの閾値（デフォルト: None、無効）
    """

    def __init__(
        self,
        data_path: str | Path,
        cp_scale: float = 1200.0,
        use_features: bool = False,
        cp_noise: float = 0.0,
        cp_filter_threshold: float | None = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.cp_scale = cp_scale
        self.use_features = use_features
        self.cp_noise = cp_noise
        self.cp_filter_threshold = cp_filter_threshold
        self.samples: list[dict] = []

        self._load_data()

    def _load_data(self) -> None:
        """データファイルを読み込む."""
        with open(self.data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)

                # 評価値フィルタ: 極端な評価値を除外
                if self.cp_filter_threshold is not None:
                    score_cp = sample.get("score_cp", 0)
                    if abs(score_cp) > self.cp_filter_threshold:
                        continue

                self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """指定インデックスのサンプルを取得.

        Args:
            idx: サンプルインデックス

        Returns:
            dict with keys:
                - board: 盤面テンソル (81,)
                - hand: 持ち駒テンソル (14,)
                - turn: 手番テンソル ()
                - value: 正規化評価値テンソル ()
                - outcome: 勝敗ラベル (1.0: 手番側勝ち, 0.0: 手番側負け, 0.5: 引き分け)
                - features: 拡張特徴量テンソル (81, 10) [use_features=True時のみ]
        """
        sample = self.samples[idx]
        sfen = sample["sfen"]
        score_cp = sample["score_cp"]

        # 評価値ノイズ付与（学習時の過学習抑制）
        if self.cp_noise > 0:
            score_cp = score_cp + random.gauss(0, self.cp_noise)

        # SFENをパース
        parsed = parse_sfen(sfen)

        # 評価値を正規化
        value = normalize_cp(score_cp, self.cp_scale)

        # 勝敗ラベルを手番視点に変換
        result_str = sample.get("result", "draw")
        ply = sample.get("ply", 0)
        is_black_turn = (ply % 2 == 0)  # 偶数手目は先手番

        if result_str == "black_win":
            outcome = 1.0 if is_black_turn else 0.0
        elif result_str == "white_win":
            outcome = 0.0 if is_black_turn else 1.0
        else:  # draw
            outcome = 0.5

        result = {
            "board": parsed.board,
            "hand": parsed.hand,
            "turn": parsed.turn,
            "value": torch.tensor(value, dtype=torch.float32),
            "outcome": torch.tensor(outcome, dtype=torch.float32),
        }

        # 拡張特徴量を追加
        if self.use_features:
            features = compute_all_features(parsed.board)
            # (81, 10) のテンソルにまとめる
            # [attack(2), king_dist(2), piece_value(1), control(1), king_safety(4)]
            result["features"] = torch.cat([
                features["attack_map"],        # (81, 2)
                features["king_distance"],     # (81, 2)
                features["piece_value"].unsqueeze(1),  # (81, 1)
                features["control"].unsqueeze(1),      # (81, 1)
                features["king_safety"],       # (81, 4)
            ], dim=1)

        return result


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """バッチをまとめる関数.

    Args:
        batch: サンプルのリスト

    Returns:
        バッチ化されたテンソルの辞書
    """
    result = {
        "board": torch.stack([s["board"] for s in batch]),
        "hand": torch.stack([s["hand"] for s in batch]),
        "turn": torch.stack([s["turn"] for s in batch]),
        "value": torch.stack([s["value"] for s in batch]),
        "outcome": torch.stack([s["outcome"] for s in batch]),
    }

    # 拡張特徴量がある場合
    if "features" in batch[0]:
        result["features"] = torch.stack([s["features"] for s in batch])

    return result
