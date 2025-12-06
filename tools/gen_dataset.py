#!/usr/bin/env python3
"""教師データ生成スクリプト

水匠5同士の自己対局を行い、各局面の評価値を収集する。
"""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from shogi.usi_engine import USIEngine, get_default_engine_path


@dataclass
class PositionRecord:
    """局面レコード"""
    sfen: str           # 局面（SFEN形式）
    score_cp: int       # 評価値（centipawn、手番側から見た値）
    ply: int            # 手数
    game_id: int        # 対局ID
    result: Optional[str] = None  # 対局結果（"black_win", "white_win", "draw"）


class SelfPlayGenerator:
    """自己対局によるデータ生成"""

    def __init__(
        self,
        engine_path: Path,
        movetime: int = 500,
        max_ply: int = 256,
        use_book: bool = False,
    ):
        """
        Args:
            engine_path: エンジンパス
            movetime: 1手あたりの思考時間（ミリ秒）
            max_ply: 最大手数
            use_book: 定跡使用
        """
        self.engine_path = engine_path
        self.movetime = movetime
        self.max_ply = max_ply
        self.use_book = use_book

    def play_game(self, game_id: int) -> list[PositionRecord]:
        """1対局を行い、局面データを収集

        Args:
            game_id: 対局ID

        Returns:
            局面レコードのリスト
        """
        records: list[PositionRecord] = []
        moves: list[str] = []
        result: Optional[str] = None

        with USIEngine(self.engine_path) as engine:
            engine.init_usi()
            engine.set_option("USI_OwnBook", self.use_book)
            engine.set_option("Threads", 1)  # 再現性のため1スレッド
            engine.is_ready()
            engine.new_game()

            for ply in range(self.max_ply):
                # 局面を設定
                engine.set_position(moves=moves if moves else None)

                # 探索
                search_result = engine.go(movetime=self.movetime)

                # 投了チェック
                if search_result.bestmove in ("resign", "win"):
                    if search_result.bestmove == "resign":
                        # 手番側が投了 = 相手の勝ち
                        result = "white_win" if ply % 2 == 0 else "black_win"
                    else:
                        # 入玉宣言勝ち
                        result = "black_win" if ply % 2 == 0 else "white_win"
                    break

                # 評価値を取得（手番側から見た値）
                score_cp = search_result.score_cp
                if score_cp is None:
                    if search_result.score_mate is not None:
                        # 詰みの場合は大きな値に変換
                        score_cp = 30000 if search_result.score_mate > 0 else -30000
                    else:
                        score_cp = 0  # フォールバック

                # SFEN取得のため一時的に局面を設定して'd'コマンドを使う代わりに、
                # movesから構築したSFENを保存（簡易版）
                # 注: 正確なSFENを取得するにはcshogiを使用するのがベスト
                sfen = self._moves_to_sfen_approx(moves, ply)

                records.append(PositionRecord(
                    sfen=sfen,
                    score_cp=score_cp,
                    ply=ply,
                    game_id=game_id,
                ))

                # 指し手を追加
                moves.append(search_result.bestmove)

                # 詰みの場合は終了
                if search_result.score_mate is not None:
                    if search_result.score_mate > 0:
                        result = "black_win" if ply % 2 == 0 else "white_win"
                    else:
                        result = "white_win" if ply % 2 == 0 else "black_win"
                    break

            # 最大手数到達
            if result is None:
                result = "draw"

        # 結果を各レコードに設定
        for record in records:
            record.result = result

        return records

    def _moves_to_sfen_approx(self, moves: list[str], ply: int) -> str:
        """指し手リストから近似SFENを生成（簡易版）

        正確なSFENを得るにはcshogiを使用すべき。
        ここでは「startpos moves ...」形式で保存。
        """
        if not moves:
            return "startpos"
        return "startpos moves " + " ".join(moves)


def generate_dataset(
    num_games: int,
    output_path: Path,
    engine_path: Path,
    movetime: int = 500,
    max_ply: int = 256,
    use_book: bool = False,
) -> int:
    """データセットを生成

    Args:
        num_games: 対局数
        output_path: 出力ファイルパス
        engine_path: エンジンパス
        movetime: 思考時間（ミリ秒）
        max_ply: 最大手数
        use_book: 定跡使用

    Returns:
        生成した局面数
    """
    generator = SelfPlayGenerator(
        engine_path=engine_path,
        movetime=movetime,
        max_ply=max_ply,
        use_book=use_book,
    )

    total_positions = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for game_id in range(num_games):
            print(f"Game {game_id + 1}/{num_games}...", end=" ", flush=True)

            try:
                records = generator.play_game(game_id)
                for record in records:
                    f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

                total_positions += len(records)
                print(f"{len(records)} positions, result: {records[0].result if records else 'N/A'}")

            except Exception as e:
                print(f"Error: {e}")
                continue

    return total_positions


def main():
    parser = argparse.ArgumentParser(description="教師データ生成")
    parser.add_argument("--num-games", "-n", type=int, default=1, help="対局数")
    parser.add_argument("--output", "-o", type=str, default=None, help="出力ファイル")
    parser.add_argument("--movetime", "-t", type=int, default=500, help="思考時間(ms)")
    parser.add_argument("--max-ply", type=int, default=256, help="最大手数")
    parser.add_argument("--use-book", action="store_true", help="定跡使用")
    parser.add_argument("--engine", type=str, default=None, help="エンジンパス")

    args = parser.parse_args()

    # デフォルト出力ファイル名
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"data/raw/positions_{timestamp}.jsonl"

    # エンジンパス
    engine_path = Path(args.engine) if args.engine else get_default_engine_path()

    if not engine_path.exists():
        print(f"Error: Engine not found: {engine_path}")
        return 1

    output_path = Path(args.output)

    print(f"Settings:")
    print(f"  Engine: {engine_path}")
    print(f"  Games: {args.num_games}")
    print(f"  Movetime: {args.movetime}ms")
    print(f"  Max ply: {args.max_ply}")
    print(f"  Use book: {args.use_book}")
    print(f"  Output: {output_path}")
    print()

    total = generate_dataset(
        num_games=args.num_games,
        output_path=output_path,
        engine_path=engine_path,
        movetime=args.movetime,
        max_ply=args.max_ply,
        use_book=args.use_book,
    )

    print(f"\nDone! Total positions: {total}")
    print(f"Output: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
