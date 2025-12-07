"""局面評価器.

学習済みValue Networkを使用して局面を評価する。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import shogi
import torch

from models import ValueTransformer, denormalize_cp, parse_sfen, compute_all_features

if TYPE_CHECKING:
    pass


class Evaluator:
    """局面評価器."""

    def __init__(
        self,
        model_path: Path | str,
        device: str = "auto",
    ) -> None:
        """初期化.

        Args:
            model_path: チェックポイントファイルのパス
            device: 推論デバイス（auto/cuda/mps/cpu）
        """
        self.device = self._get_device(device)
        self.model, self.use_features = self._load_model(Path(model_path))

    def _get_device(self, device_str: str) -> torch.device:
        """デバイスを取得."""
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device_str)

    def _load_model(self, model_path: Path) -> tuple[ValueTransformer, bool]:
        """モデルを読み込み."""
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        # チェックポイントからモデル設定を取得
        config = checkpoint.get("config", {})
        use_features = config.get("use_features", False)

        model = ValueTransformer(
            d_model=config.get("d_model", 256),
            n_heads=config.get("n_heads", 4),
            n_layers=config.get("n_layers", 4),
            ffn_dim=config.get("ffn_dim", 512),
            dropout=0.0,  # 推論時はドロップアウト無効
            use_features=use_features,
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        return model, use_features

    def evaluate_sfen(self, sfen: str) -> int:
        """SFEN文字列で表された局面を評価.

        Args:
            sfen: SFEN文字列（"startpos"または"startpos moves ..."形式）

        Returns:
            評価値（centipawn、手番側視点）
        """
        # SFENをパース
        parsed = parse_sfen(sfen)

        # バッチ次元を追加
        board = parsed.board.unsqueeze(0).to(self.device)
        hand = parsed.hand.unsqueeze(0).to(self.device)
        turn = parsed.turn.unsqueeze(0).to(self.device)

        # 拡張特徴量
        features = None
        if self.use_features:
            feat_dict = compute_all_features(parsed.board)
            features = torch.cat([
                feat_dict["attack_map"],
                feat_dict["king_distance"],
                feat_dict["piece_value"].unsqueeze(1),
                feat_dict["control"].unsqueeze(1),
                feat_dict["king_safety"],
            ], dim=1).unsqueeze(0).to(self.device)

        # 推論
        with torch.no_grad():
            output = self.model(board, hand, turn, features)
            value = output.item()

        # centipawnに変換
        return int(denormalize_cp(value))

    def evaluate_board(self, board: shogi.Board) -> int:
        """python-shogiのBoardオブジェクトを評価.

        Args:
            board: shogi.Boardオブジェクト

        Returns:
            評価値（centipawn、手番側視点）
        """
        sfen = self._board_to_sfen(board)
        return self.evaluate_sfen(sfen)

    def _board_to_sfen(self, board: shogi.Board) -> str:
        """BoardオブジェクトをSFEN文字列に変換."""
        # python-shogiのsfen()は"sfen ..."形式を返す
        return "sfen " + board.sfen()

    def find_best_move(self, board: shogi.Board) -> tuple[str, int]:
        """最善手を探索（1手読み）.

        Args:
            board: 現在の局面

        Returns:
            (最善手のUSI表記, 評価値)
        """
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return "resign", -30000

        best_move = None
        best_score = -100000

        for move in legal_moves:
            # 手を適用
            board.push(move)

            # 相手視点の評価値を取得して符号反転
            score = -self.evaluate_board(board)

            # 手を戻す
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

        return best_move.usi(), best_score
