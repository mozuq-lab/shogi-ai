# -*- coding: utf-8 -*-
"""Convert SFEN moves from JSONL to KIF format.

Usage:
    python scripts/to_kif.py <jsonl_file> <game_id> [--output <output_file>]

Examples:
    python scripts/to_kif.py data/raw/hao_depth10_100k.jsonl 1
    python scripts/to_kif.py data/raw/hao_depth10_100k.jsonl 1 --output game1.kif
    python scripts/to_kif.py data/raw/hao_depth10_100k.jsonl 1 -o game1.kif
"""
import argparse
import json
import shogi

PIECE_JAPANESE = {
    shogi.PAWN: '歩', shogi.LANCE: '香', shogi.KNIGHT: '桂', shogi.SILVER: '銀',
    shogi.GOLD: '金', shogi.BISHOP: '角', shogi.ROOK: '飛', shogi.KING: '玉',
    shogi.PROM_PAWN: 'と', shogi.PROM_LANCE: '成香', shogi.PROM_KNIGHT: '成桂',
    shogi.PROM_SILVER: '成銀', shogi.PROM_BISHOP: '馬', shogi.PROM_ROOK: '龍',
}

FILE_JAPANESE = ['１', '２', '３', '４', '５', '６', '７', '８', '９']
RANK_JAPANESE = ['一', '二', '三', '四', '五', '六', '七', '八', '九']


def square_to_kif(square: int) -> str:
    """Convert square index to KIF notation like (75)."""
    file_num = 9 - (square % 9)
    rank_num = (square // 9) + 1
    return f'({file_num}{rank_num})'


def move_to_kif(board: shogi.Board, move: shogi.Move) -> str:
    """Convert a move to KIF notation."""
    if move.from_square is not None:
        piece_type = board.piece_type_at(move.from_square)
    else:
        piece_type = move.drop_piece_type

    # python-shogi座標系: square = (9 - file) * 9 + rank
    # file: 1-9 (1筋-9筋), rank: 0-8 (一段-九段)
    to_file = 9 - (move.to_square % 9)  # 1-9
    to_rank = move.to_square // 9       # 0-8

    result = FILE_JAPANESE[to_file - 1] + RANK_JAPANESE[to_rank]
    result += PIECE_JAPANESE.get(piece_type, '?')

    if move.promotion:
        result += '成'
    if move.from_square is None:
        result += '打'
    else:
        # 同じ駒種で同じマスに移動できる駒が複数あるか確認
        dominated_piece_type = piece_type
        if piece_type >= shogi.PROM_PAWN:
            # 成駒の場合は元の駒種で判定
            dominated_piece_type = piece_type - 8  # 成駒 -> 元駒

        same_piece_moves = []
        for legal_move in board.legal_moves:
            if legal_move.to_square == move.to_square and legal_move.from_square is not None:
                lm_piece = board.piece_type_at(legal_move.from_square)
                lm_base = lm_piece - 8 if lm_piece >= shogi.PROM_PAWN else lm_piece
                if lm_base == dominated_piece_type or lm_piece == piece_type:
                    same_piece_moves.append(legal_move)

        if len(same_piece_moves) > 1:
            result += square_to_kif(move.from_square)

    return result


def load_game_from_jsonl(jsonl_path: str, game_id: int) -> dict | None:
    """Load a game from JSONL file.

    Returns the entry with the highest ply for the given game_id.
    """
    best_entry = None
    best_ply = -1

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get('game_id') == game_id:
                ply = entry.get('ply', 0)
                if ply > best_ply:
                    best_ply = ply
                    best_entry = entry

    return best_entry


def list_games(jsonl_path: str, limit: int = 20) -> list[dict]:
    """List available games in JSONL file."""
    games = {}

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            gid = entry.get('game_id')
            ply = entry.get('ply', 0)

            if gid not in games or ply > games[gid]['ply']:
                games[gid] = {
                    'game_id': gid,
                    'ply': ply,
                    'result': entry.get('result'),
                    'score_cp': entry.get('score_cp'),
                }

    # Sort by game_id and return first 'limit' entries
    sorted_games = sorted(games.values(), key=lambda x: x['game_id'])
    return sorted_games[:limit]


def convert_to_kif(entry: dict) -> str:
    """Convert a game entry to KIF format."""
    sfen = entry['sfen']
    result = entry['result']
    score_cp = entry['score_cp']
    game_id = entry['game_id']

    # Extract moves from sfen
    if sfen.startswith('startpos moves '):
        moves_str = sfen[len('startpos moves '):]
        moves = moves_str.split()
    elif sfen == 'startpos':
        moves = []
    else:
        # Custom starting position - not supported yet
        return f"# Error: Custom starting position not supported: {sfen}"

    ply = len(moves)

    # score_cp=30000 の場合（手番側が詰みを検出）
    # → 負けた側の「余計な一手」が含まれているので除外
    if score_cp == 30000:
        moves = moves[:-1]
        ply = ply - 1

    result_ja = '後手勝ち' if result == 'white_win' else '先手勝ち' if result == 'black_win' else '引き分け'

    lines = []
    lines.append(f'# ---- 棋譜 (game_id={game_id}) ----')
    lines.append('手合割：平手')
    lines.append('先手：Hao')
    lines.append('後手：Hao')
    lines.append(f'結果：{result_ja} ({ply}手)')
    lines.append('')
    lines.append('手数----指手----')

    board = shogi.Board()
    for i, move_usi in enumerate(moves):
        move = shogi.Move.from_usi(move_usi)
        turn = '▲' if board.turn == shogi.BLACK else '△'
        kif = move_to_kif(board, move)
        lines.append(f'{i+1:4d} {turn}{kif}')
        board.push(move)

    lines.append(f'{len(moves)+1:4d} 投了')
    lines.append('')
    lines.append(f'まで{len(moves)}手で{result_ja}')

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Convert SFEN moves from JSONL to KIF format.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/raw/hao_depth10_100k.jsonl 1
  %(prog)s data/raw/hao_depth10_100k.jsonl 1 --output game1.kif
  %(prog)s data/raw/hao_depth10_100k.jsonl --list
"""
    )
    parser.add_argument('jsonl_file', help='Path to JSONL file')
    parser.add_argument('game_id', type=int, nargs='?', default=None,
                        help='Game ID to convert')
    parser.add_argument('-o', '--output', help='Output file path (default: stdout)')
    parser.add_argument('-l', '--list', action='store_true',
                        help='List available games')
    parser.add_argument('--list-limit', type=int, default=20,
                        help='Number of games to list (default: 20)')

    args = parser.parse_args()

    if args.list:
        games = list_games(args.jsonl_file, args.list_limit)
        print(f"{'game_id':>8}  {'ply':>4}  {'result':>12}  {'score_cp':>10}")
        print('-' * 42)
        for g in games:
            print(f"{g['game_id']:>8}  {g['ply']:>4}  {g['result']:>12}  {g['score_cp']:>10}")
        return

    if args.game_id is None:
        parser.error("game_id is required (or use --list to see available games)")

    entry = load_game_from_jsonl(args.jsonl_file, args.game_id)

    if entry is None:
        print(f"Error: game_id={args.game_id} not found in {args.jsonl_file}")
        return

    kif_content = convert_to_kif(entry)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(kif_content)
        print(f"Written to {args.output}")
    else:
        print(kif_content)


if __name__ == '__main__':
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    main()
