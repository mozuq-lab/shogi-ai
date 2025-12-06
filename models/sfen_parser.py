"""SFENパーサー: SFEN文字列を盤面テンソルに変換.

SFENフォーマット:
- position sfen <board> <turn> <hand> <move_count>
- position startpos [moves <move1> <move2> ...]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# 駒種マッピング（SFEN記号 → 駒ID）
# 先手の駒: 1-14
# 後手の駒: 15-28
PIECE_TO_ID: dict[str, int] = {
    # 先手（大文字）
    "P": 1,   # 歩
    "L": 2,   # 香
    "N": 3,   # 桂
    "S": 4,   # 銀
    "G": 5,   # 金
    "B": 6,   # 角
    "R": 7,   # 飛
    "K": 8,   # 王
    "+P": 9,  # と
    "+L": 10,  # 成香
    "+N": 11,  # 成桂
    "+S": 12,  # 成銀
    "+B": 13,  # 馬
    "+R": 14,  # 龍
    # 後手（小文字）
    "p": 15,
    "l": 16,
    "n": 17,
    "s": 18,
    "g": 19,
    "b": 20,
    "r": 21,
    "k": 22,
    "+p": 23,
    "+l": 24,
    "+n": 25,
    "+s": 26,
    "+b": 27,
    "+r": 28,
}

# 持ち駒のインデックス（先手0-6, 後手7-13）
HAND_PIECE_TO_IDX: dict[str, int] = {
    # 先手
    "P": 0, "L": 1, "N": 2, "S": 3, "G": 4, "B": 5, "R": 6,
    # 後手
    "p": 7, "l": 8, "n": 9, "s": 10, "g": 11, "b": 12, "r": 13,
}

# 初期配置のSFEN盤面部分
STARTPOS_BOARD = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL"


@dataclass
class ParsedPosition:
    """パース済み局面データ."""

    board: torch.Tensor  # (81,) 駒種ID
    hand: torch.Tensor   # (14,) 持ち駒枚数
    turn: torch.Tensor   # () 手番（0:先手, 1:後手）


def parse_board_sfen(board_sfen: str) -> torch.Tensor:
    """SFEN盤面部分をパースして81要素のテンソルに変換.

    Args:
        board_sfen: SFEN盤面文字列（例: "lnsgkgsnl/1r5b1/..."）

    Returns:
        盤面テンソル (81,) - 各マスの駒種ID（0=空）
    """
    board = torch.zeros(81, dtype=torch.long)
    rows = board_sfen.split("/")

    if len(rows) != 9:
        raise ValueError(f"Invalid board SFEN: expected 9 rows, got {len(rows)}")

    idx = 0
    for row in rows:
        col = 0
        i = 0
        while i < len(row):
            char = row[i]

            if char.isdigit():
                # 空マスの数
                col += int(char)
            elif char == "+":
                # 成駒
                if i + 1 >= len(row):
                    raise ValueError(f"Invalid board SFEN: '+' at end of row")
                piece = char + row[i + 1]
                if piece not in PIECE_TO_ID:
                    raise ValueError(f"Unknown piece: {piece}")
                board[idx + col] = PIECE_TO_ID[piece]
                col += 1
                i += 1
            else:
                # 通常の駒
                if char not in PIECE_TO_ID:
                    raise ValueError(f"Unknown piece: {char}")
                board[idx + col] = PIECE_TO_ID[char]
                col += 1

            i += 1

        if col != 9:
            raise ValueError(f"Invalid board SFEN: row has {col} squares, expected 9")
        idx += 9

    return board


def parse_hand_sfen(hand_sfen: str) -> torch.Tensor:
    """SFEN持ち駒部分をパースして14要素のテンソルに変換.

    Args:
        hand_sfen: SFEN持ち駒文字列（例: "2P3p" or "-"）

    Returns:
        持ち駒テンソル (14,) - 各持ち駒種の枚数
    """
    hand = torch.zeros(14, dtype=torch.long)

    if hand_sfen == "-":
        return hand

    i = 0
    while i < len(hand_sfen):
        # 枚数を読み取る
        count = 0
        while i < len(hand_sfen) and hand_sfen[i].isdigit():
            count = count * 10 + int(hand_sfen[i])
            i += 1

        if count == 0:
            count = 1

        # 駒種を読み取る
        if i >= len(hand_sfen):
            break

        piece = hand_sfen[i]
        if piece not in HAND_PIECE_TO_IDX:
            raise ValueError(f"Unknown hand piece: {piece}")

        hand[HAND_PIECE_TO_IDX[piece]] = count
        i += 1

    return hand


def apply_move(board: torch.Tensor, hand: torch.Tensor, move: str, turn: int) -> None:
    """指し手を盤面に適用（インプレース）.

    Args:
        board: 盤面テンソル (81,)
        hand: 持ち駒テンソル (14,)
        move: USI形式の指し手（例: "7g7f", "P*5e"）
        turn: 手番（0:先手, 1:後手）
    """
    if len(move) < 4:
        raise ValueError(f"Invalid move: {move}")

    # 駒打ち
    if move[1] == "*":
        piece_char = move[0]
        to_file = int(move[2])  # 筋（1-9）
        to_rank = ord(move[3]) - ord("a") + 1  # 段（1-9）
        to_idx = (to_rank - 1) * 9 + (9 - to_file)

        # 持ち駒から減らす
        if turn == 0:  # 先手
            hand_idx = HAND_PIECE_TO_IDX[piece_char.upper()]
            piece_id = PIECE_TO_ID[piece_char.upper()]
        else:  # 後手
            hand_idx = HAND_PIECE_TO_IDX[piece_char.lower()]
            piece_id = PIECE_TO_ID[piece_char.lower()]

        if hand[hand_idx] <= 0:
            raise ValueError(f"No {piece_char} in hand")

        hand[hand_idx] -= 1
        board[to_idx] = piece_id
        return

    # 通常の指し手
    from_file = int(move[0])  # 筋（1-9）
    from_rank = ord(move[1]) - ord("a") + 1  # 段（1-9）
    to_file = int(move[2])  # 筋（1-9）
    to_rank = ord(move[3]) - ord("a") + 1  # 段（1-9）

    from_idx = (from_rank - 1) * 9 + (9 - from_file)
    to_idx = (to_rank - 1) * 9 + (9 - to_file)

    # 成り判定
    promote = len(move) == 5 and move[4] == "+"

    # 駒を取る場合
    captured = board[to_idx].item()
    if captured != 0 and captured != 8 and captured != 22:
        # 王(K=8, k=22)は持ち駒にならない

        # 成駒は元の駒に戻す
        # 駒ID: P=1, L=2, N=3, S=4, G=5, B=6, R=7, K=8
        #       +P=9, +L=10, +N=11, +S=12, +B=13, +R=14
        # 後手は +14
        # と金〜成銀(9-12, 23-26): -8で生駒に戻る
        # 馬・龍(13-14, 27-28): -7で生駒に戻る（金をスキップするため）
        if captured in (9, 10, 11, 12):  # 先手のと金〜成銀
            captured = captured - 8  # → 歩〜銀 (1-4)
        elif captured in (13, 14):  # 先手の馬・龍
            captured = captured - 7  # → 角・飛 (6-7)
        elif captured in (23, 24, 25, 26):  # 後手のと金〜成銀
            captured = captured - 8  # → 歩〜銀 (15-18)
        elif captured in (27, 28):  # 後手の馬・龍
            captured = captured - 7  # → 角・飛 (20-21)

        # 持ち駒に追加（相手の駒を自分の駒として）
        # 持ち駒インデックス: 歩=0, 香=1, 桂=2, 銀=3, 金=4, 角=5, 飛=6
        # 先手駒ID: 歩=1, 香=2, 桂=3, 銀=4, 金=5, 角=6, 飛=7
        # 後手駒ID: 歩=15, 香=16, 桂=17, 銀=18, 金=19, 角=20, 飛=21
        if turn == 0:  # 先手が取った（後手の駒を取る）
            # 後手の駒（15-21）→ 先手の持ち駒インデックス（0-6）
            # 注: 金(g=19)は特別処理が必要
            if captured == 19:  # 後手の金
                hand_idx = 4  # 金のインデックス
            elif captured == 20:  # 後手の角
                hand_idx = 5
            elif captured == 21:  # 後手の飛
                hand_idx = 6
            else:
                hand_idx = captured - 15  # 歩〜銀 (15-18 → 0-3)
            hand[hand_idx] += 1
        else:  # 後手が取った（先手の駒を取る）
            # 先手の駒（1-7）→ 後手の持ち駒インデックス（7-13）
            if captured == 5:  # 先手の金
                hand_idx = 11  # 7 + 4
            elif captured == 6:  # 先手の角
                hand_idx = 12  # 7 + 5
            elif captured == 7:  # 先手の飛
                hand_idx = 13  # 7 + 6
            else:
                hand_idx = captured - 1 + 7  # 歩〜銀 (1-4 → 7-10)
            hand[hand_idx] += 1

    # 駒を移動
    piece = board[from_idx].item()
    if promote:
        # 成る
        # 駒ID: P=1, L=2, N=3, S=4, G=5, B=6, R=7, K=8
        #       +P=9, +L=10, +N=11, +S=12, +B=13, +R=14
        # 後手: p=15, l=16, n=17, s=18, g=19, b=20, r=21, k=22
        #       +p=23, +l=24, +n=25, +s=26, +b=27, +r=28
        # 歩〜銀(1-4, 15-18): +8で成駒に
        # 角・飛(6-7, 20-21): +7で成駒に（金をスキップするため）
        if piece in (1, 2, 3, 4):  # 先手の歩〜銀
            piece = piece + 8
        elif piece in (6, 7):  # 先手の角・飛
            piece = piece + 7
        elif piece in (15, 16, 17, 18):  # 後手の歩〜銀
            piece = piece + 8
        elif piece in (20, 21):  # 後手の角・飛
            piece = piece + 7

    board[from_idx] = 0
    board[to_idx] = piece


def parse_sfen(sfen: str) -> ParsedPosition:
    """SFEN文字列をパースして局面データに変換.

    Args:
        sfen: SFEN文字列
            - "startpos" または "startpos moves ..."
            - "sfen <board> <turn> <hand> <move_count>"

    Returns:
        パース済み局面データ
    """
    parts = sfen.strip().split()

    if not parts:
        raise ValueError("Empty SFEN string")

    # startpos形式
    if parts[0] == "startpos":
        board = parse_board_sfen(STARTPOS_BOARD)
        hand = torch.zeros(14, dtype=torch.long)
        turn = 0  # 先手

        # moves がある場合
        if len(parts) > 1 and parts[1] == "moves":
            for i, move in enumerate(parts[2:]):
                current_turn = i % 2
                apply_move(board, hand, move, current_turn)
            # 最終手番を計算
            turn = len(parts[2:]) % 2

        return ParsedPosition(
            board=board,
            hand=hand,
            turn=torch.tensor(turn, dtype=torch.long),
        )

    # sfen形式
    if parts[0] == "sfen" or parts[0] == "position":
        if parts[0] == "position" and len(parts) > 1 and parts[1] == "sfen":
            parts = parts[1:]  # "position sfen ..." → "sfen ..."

        if parts[0] == "sfen" and len(parts) >= 4:
            board = parse_board_sfen(parts[1])
            turn = 0 if parts[2] == "b" else 1
            hand = parse_hand_sfen(parts[3])

            # moves がある場合
            moves_idx = None
            for i, part in enumerate(parts):
                if part == "moves":
                    moves_idx = i
                    break

            if moves_idx is not None:
                for j, move in enumerate(parts[moves_idx + 1 :]):
                    current_turn = (turn + j) % 2
                    apply_move(board, hand, move, current_turn)
                turn = (turn + len(parts[moves_idx + 1 :])) % 2

            return ParsedPosition(
                board=board,
                hand=hand,
                turn=torch.tensor(turn, dtype=torch.long),
            )

    # フォールバック: 盤面SFENとして解釈を試みる
    raise ValueError(f"Cannot parse SFEN: {sfen}")
