"""モデル関連のテスト."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from models import (
    ParsedPosition,
    ShogiValueDataset,
    ValueTransformer,
    collate_fn,
    denormalize_cp,
    normalize_cp,
    parse_sfen,
)


class TestNormalizeCp:
    """評価値正規化のテスト."""

    def test_zero(self) -> None:
        assert normalize_cp(0) == 0.0

    def test_positive(self) -> None:
        result = normalize_cp(1200)
        assert 0.7 < result < 0.8  # tanh(1) ≈ 0.76

    def test_negative(self) -> None:
        result = normalize_cp(-1200)
        assert -0.8 < result < -0.7

    def test_round_trip(self) -> None:
        for cp in [-2000, -500, 0, 500, 2000]:
            normalized = normalize_cp(cp)
            denormalized = denormalize_cp(normalized)
            assert abs(denormalized - cp) < 1.0  # 誤差1cp以内


class TestSfenParser:
    """SFENパーサーのテスト."""

    def test_startpos(self) -> None:
        result = parse_sfen("startpos")
        assert isinstance(result, ParsedPosition)
        assert result.board.shape == (81,)
        assert result.hand.shape == (14,)
        assert result.turn.item() == 0  # 先手

    def test_startpos_initial_board(self) -> None:
        result = parse_sfen("startpos")
        # 1段目（後手側）: 香桂銀金王金銀桂香
        assert result.board[0].item() == 16  # 後手香 (l)
        assert result.board[1].item() == 17  # 後手桂 (n)
        assert result.board[4].item() == 22  # 後手王 (k)
        # 9段目（先手側）: 香桂銀金王金銀桂香
        assert result.board[72].item() == 2  # 先手香 (L)
        assert result.board[76].item() == 8  # 先手王 (K)

    def test_startpos_with_moves(self) -> None:
        result = parse_sfen("startpos moves 7g7f")
        assert result.turn.item() == 1  # 後手番

        # 7七の歩がなくなり、7六に移動
        # idx = (rank - 1) * 9 + (9 - file)
        # 7g: file=7, rank=7 -> (7-1)*9 + (9-7) = 54 + 2 = 56
        # 7f: file=7, rank=6 -> (6-1)*9 + (9-7) = 45 + 2 = 47
        assert result.board[56].item() == 0  # 7gは空
        assert result.board[47].item() == 1  # 7fに先手歩

    def test_startpos_two_moves(self) -> None:
        result = parse_sfen("startpos moves 7g7f 3c3d")
        assert result.turn.item() == 0  # 先手番（2手後）

    def test_empty_hand(self) -> None:
        result = parse_sfen("startpos")
        assert result.hand.sum().item() == 0


class TestValueTransformer:
    """ValueTransformerモデルのテスト."""

    @pytest.fixture
    def model(self) -> ValueTransformer:
        return ValueTransformer(
            d_model=64,
            n_heads=2,
            n_layers=2,
            ffn_dim=128,
            dropout=0.0,
        )

    def test_output_shape(self, model: ValueTransformer) -> None:
        batch_size = 4
        board = torch.zeros(batch_size, 81, dtype=torch.long)
        hand = torch.zeros(batch_size, 14, dtype=torch.long)
        turn = torch.zeros(batch_size, dtype=torch.long)

        value, outcome = model(board, hand, turn)
        assert value.shape == (batch_size, 1)
        assert outcome.shape == (batch_size, 1)

    def test_output_range(self, model: ValueTransformer) -> None:
        batch_size = 4
        board = torch.randint(0, 29, (batch_size, 81))
        hand = torch.randint(0, 5, (batch_size, 14))
        turn = torch.randint(0, 2, (batch_size,))

        value, outcome = model(board, hand, turn)
        # 評価値は [-1, 1]
        assert torch.all(value >= -1.0)
        assert torch.all(value <= 1.0)
        # 勝率は [0, 1]
        assert torch.all(outcome >= 0.0)
        assert torch.all(outcome <= 1.0)

    def test_gradient_flow(self, model: ValueTransformer) -> None:
        board = torch.randint(0, 29, (2, 81))
        hand = torch.randint(0, 5, (2, 14))
        turn = torch.randint(0, 2, (2,))

        value, outcome = model(board, hand, turn)
        loss = value.sum() + outcome.sum()
        loss.backward()

        # 全パラメータに勾配が流れていることを確認
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestShogiValueDataset:
    """データセットのテスト."""

    @pytest.fixture
    def sample_data_path(self, tmp_path: Path) -> Path:
        data = [
            '{"sfen": "startpos", "score_cp": 0, "ply": 0, "game_id": 0, "result": "draw"}',
            '{"sfen": "startpos moves 7g7f", "score_cp": 50, "ply": 1, "game_id": 0, "result": "draw"}',
            '{"sfen": "startpos moves 7g7f 3c3d", "score_cp": -30, "ply": 2, "game_id": 0, "result": "draw"}',
        ]
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(data))
        return path

    def test_load(self, sample_data_path: Path) -> None:
        dataset = ShogiValueDataset(sample_data_path)
        assert len(dataset) == 3

    def test_getitem(self, sample_data_path: Path) -> None:
        dataset = ShogiValueDataset(sample_data_path)
        sample = dataset[0]

        assert "board" in sample
        assert "hand" in sample
        assert "turn" in sample
        assert "value" in sample

        assert sample["board"].shape == (81,)
        assert sample["hand"].shape == (14,)
        assert sample["value"].shape == ()

    def test_collate(self, sample_data_path: Path) -> None:
        dataset = ShogiValueDataset(sample_data_path)
        batch = [dataset[i] for i in range(3)]
        collated = collate_fn(batch)

        assert collated["board"].shape == (3, 81)
        assert collated["hand"].shape == (3, 14)
        assert collated["turn"].shape == (3,)
        assert collated["value"].shape == (3,)


class TestIntegration:
    """統合テスト."""

    def test_model_with_dataset(self, tmp_path: Path) -> None:
        # データ準備
        data = [
            '{"sfen": "startpos", "score_cp": 0, "ply": 0, "game_id": 0, "result": "draw"}',
            '{"sfen": "startpos moves 7g7f", "score_cp": 50, "ply": 1, "game_id": 0, "result": "draw"}',
        ]
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(data))

        # データセット
        dataset = ShogiValueDataset(path)
        batch = collate_fn([dataset[0], dataset[1]])

        # モデル
        model = ValueTransformer(d_model=64, n_heads=2, n_layers=2, ffn_dim=128)

        # 推論
        value, outcome = model(batch["board"], batch["hand"], batch["turn"])
        assert value.shape == (2, 1)
        assert outcome.shape == (2, 1)

        # 損失計算
        target_value = batch["value"].unsqueeze(1)
        target_outcome = batch["outcome"].unsqueeze(1)
        value_loss = torch.nn.functional.mse_loss(value, target_value)
        outcome_loss = torch.nn.functional.binary_cross_entropy(outcome, target_outcome)
        assert value_loss.item() >= 0
        assert outcome_loss.item() >= 0
