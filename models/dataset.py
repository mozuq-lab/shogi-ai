"""将棋局面データセット."""

from __future__ import annotations

import json
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
    """

    def __init__(
        self,
        data_path: str | Path,
        cp_scale: float = 1200.0,
        use_features: bool = False,
    ) -> None:
        self.data_path = Path(data_path)
        self.cp_scale = cp_scale
        self.use_features = use_features
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
                - features: 拡張特徴量テンソル (81, 6) [use_features=True時のみ]
        """
        sample = self.samples[idx]
        sfen = sample["sfen"]
        score_cp = sample["score_cp"]

        # SFENをパース
        parsed = parse_sfen(sfen)

        # 評価値を正規化
        value = normalize_cp(score_cp, self.cp_scale)

        result = {
            "board": parsed.board,
            "hand": parsed.hand,
            "turn": parsed.turn,
            "value": torch.tensor(value, dtype=torch.float32),
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
    }

    # 拡張特徴量がある場合
    if "features" in batch[0]:
        result["features"] = torch.stack([s["features"] for s in batch])

    return result
