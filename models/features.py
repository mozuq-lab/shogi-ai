"""拡張特徴量生成モジュール.

局面から利きマップ、王との距離などの特徴量を生成する。
"""

from __future__ import annotations

import torch

# 駒の動きの定義（dx, dy）
# 正の値は先手から見た動き
PIECE_MOVES: dict[int, list[tuple[int, int, bool]]] = {
    # (dx, dy, is_sliding) - is_sliding=Trueは飛び駒
    # 先手の駒
    1: [(0, -1, False)],  # 歩: 前に1マス
    2: [(0, -1, True)],   # 香: 前に何マスでも
    3: [(-1, -2, False), (1, -2, False)],  # 桂: 前に2、左右に1
    4: [(-1, -1, False), (0, -1, False), (1, -1, False),  # 銀
        (-1, 1, False), (1, 1, False)],
    5: [(-1, -1, False), (0, -1, False), (1, -1, False),  # 金
        (-1, 0, False), (1, 0, False), (0, 1, False)],
    6: [(-1, -1, True), (1, -1, True), (-1, 1, True), (1, 1, True)],  # 角
    7: [(0, -1, True), (0, 1, True), (-1, 0, True), (1, 0, True)],    # 飛
    8: [(-1, -1, False), (0, -1, False), (1, -1, False),  # 王
        (-1, 0, False), (1, 0, False),
        (-1, 1, False), (0, 1, False), (1, 1, False)],
    9: [(-1, -1, False), (0, -1, False), (1, -1, False),  # と金（金と同じ）
        (-1, 0, False), (1, 0, False), (0, 1, False)],
    10: [(-1, -1, False), (0, -1, False), (1, -1, False),  # 成香
         (-1, 0, False), (1, 0, False), (0, 1, False)],
    11: [(-1, -1, False), (0, -1, False), (1, -1, False),  # 成桂
         (-1, 0, False), (1, 0, False), (0, 1, False)],
    12: [(-1, -1, False), (0, -1, False), (1, -1, False),  # 成銀
         (-1, 0, False), (1, 0, False), (0, 1, False)],
    13: [(-1, -1, True), (1, -1, True), (-1, 1, True), (1, 1, True),  # 馬
         (0, -1, False), (0, 1, False), (-1, 0, False), (1, 0, False)],
    14: [(0, -1, True), (0, 1, True), (-1, 0, True), (1, 0, True),  # 龍
         (-1, -1, False), (1, -1, False), (-1, 1, False), (1, 1, False)],
}

# 後手の駒（15-28）は先手の駒（1-14）のy方向を反転
for piece_id in range(1, 15):
    if piece_id in PIECE_MOVES:
        PIECE_MOVES[piece_id + 14] = [
            (dx, -dy, sliding) for dx, dy, sliding in PIECE_MOVES[piece_id]
        ]


def compute_attack_map(board: torch.Tensor) -> torch.Tensor:
    """盤面から利きマップを計算.

    Args:
        board: 盤面テンソル (81,) 駒種ID

    Returns:
        利きマップ (81, 2) - [先手の利き数, 後手の利き数]
    """
    attack_map = torch.zeros(81, 2, dtype=torch.float32)
    board_2d = board.view(9, 9)

    for idx in range(81):
        piece_id = board[idx].item()
        if piece_id == 0:
            continue

        row, col = idx // 9, idx % 9
        is_black = piece_id <= 14  # 先手の駒
        attack_idx = 0 if is_black else 1

        if piece_id not in PIECE_MOVES:
            continue

        for dx, dy, is_sliding in PIECE_MOVES[piece_id]:
            nx, ny = col + dx, row + dy

            if is_sliding:
                # 飛び駒: 障害物に当たるまで進む
                while 0 <= nx < 9 and 0 <= ny < 9:
                    target_idx = ny * 9 + nx
                    attack_map[target_idx, attack_idx] += 1
                    # 駒があったら止まる
                    if board_2d[ny, nx] != 0:
                        break
                    nx += dx
                    ny += dy
            else:
                # 歩行駒: 1マスだけ
                if 0 <= nx < 9 and 0 <= ny < 9:
                    target_idx = ny * 9 + nx
                    attack_map[target_idx, attack_idx] += 1

    return attack_map


def compute_king_distance(board: torch.Tensor) -> torch.Tensor:
    """各マスから両王までの距離を計算.

    Args:
        board: 盤面テンソル (81,) 駒種ID

    Returns:
        王距離マップ (81, 2) - [先手王までの距離, 後手王までの距離]
                              王が盤上にない場合は10.0
    """
    distance_map = torch.full((81, 2), 10.0, dtype=torch.float32)

    # 王の位置を探す
    black_king_idx = None
    white_king_idx = None

    for idx in range(81):
        piece_id = board[idx].item()
        if piece_id == 8:  # 先手王
            black_king_idx = idx
        elif piece_id == 22:  # 後手王
            white_king_idx = idx

    # 距離を計算（チェビシェフ距離）
    for idx in range(81):
        row, col = idx // 9, idx % 9

        if black_king_idx is not None:
            king_row, king_col = black_king_idx // 9, black_king_idx % 9
            dist = max(abs(row - king_row), abs(col - king_col))
            distance_map[idx, 0] = dist

        if white_king_idx is not None:
            king_row, king_col = white_king_idx // 9, white_king_idx % 9
            dist = max(abs(row - king_row), abs(col - king_col))
            distance_map[idx, 1] = dist

    # 正規化（0-1の範囲に）
    distance_map = distance_map / 8.0

    return distance_map


def compute_piece_value_map(board: torch.Tensor) -> torch.Tensor:
    """各マスの駒価値マップを計算.

    Args:
        board: 盤面テンソル (81,) 駒種ID

    Returns:
        駒価値マップ (81,) - 先手の駒は正、後手の駒は負
    """
    # 駒の価値（歩を100として）
    PIECE_VALUES = {
        1: 100, 2: 300, 3: 350, 4: 450, 5: 500, 6: 700, 7: 900, 8: 10000,
        9: 500, 10: 450, 11: 450, 12: 500, 13: 1000, 14: 1200,
        15: -100, 16: -300, 17: -350, 18: -450, 19: -500, 20: -700, 21: -900, 22: -10000,
        23: -500, 24: -450, 25: -450, 26: -500, 27: -1000, 28: -1200,
    }

    value_map = torch.zeros(81, dtype=torch.float32)

    for idx in range(81):
        piece_id = board[idx].item()
        if piece_id in PIECE_VALUES:
            value_map[idx] = PIECE_VALUES[piece_id]

    # 正規化（-1〜1の範囲に、王を除く）
    value_map = torch.clamp(value_map / 1200.0, -1.0, 1.0)

    return value_map


def compute_control_map(board: torch.Tensor) -> torch.Tensor:
    """利き差分（支配度）マップを計算.

    Args:
        board: 盤面テンソル (81,) 駒種ID

    Returns:
        支配度マップ (81,) - 正なら先手有利、負なら後手有利
    """
    attack_map = compute_attack_map(board)
    # 先手の利き - 後手の利き
    control = attack_map[:, 0] - attack_map[:, 1]
    # tanh で正規化
    return torch.tanh(control / 3.0)


def compute_all_features(board: torch.Tensor) -> dict[str, torch.Tensor]:
    """全ての拡張特徴量を計算.

    Args:
        board: 盤面テンソル (81,) 駒種ID

    Returns:
        特徴量辞書:
            - attack_map: (81, 2) 利きマップ
            - king_distance: (81, 2) 王距離マップ
            - piece_value: (81,) 駒価値マップ
            - control: (81,) 支配度マップ
    """
    return {
        "attack_map": compute_attack_map(board),
        "king_distance": compute_king_distance(board),
        "piece_value": compute_piece_value_map(board),
        "control": compute_control_map(board),
    }
