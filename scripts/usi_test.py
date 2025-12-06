#!/usr/bin/env python3
"""水匠5 USI通信テストスクリプト

水匠5エンジンとのUSI通信が正常に動作するか確認する。
"""

from __future__ import annotations

import subprocess
import re
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from shogi_utils import get_default_engine_path


def get_engine_path() -> Path:
    """エンジンのパスを取得"""
    return get_default_engine_path()


def run_usi_command(commands: list[str], timeout: int = 30) -> str:
    """USIコマンドを実行して出力を取得"""
    engine_path = get_engine_path()
    engine_dir = engine_path.parent

    if not engine_path.exists():
        raise FileNotFoundError(f"Engine not found: {engine_path}")

    input_text = "\n".join(commands) + "\n"

    result = subprocess.run(
        [str(engine_path)],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(engine_dir),
    )

    return result.stdout + result.stderr


def test_usi_init() -> bool:
    """USI初期化テスト"""
    print("=== USI初期化テスト ===")

    output = run_usi_command(["usi", "isready", "quit"])

    if "usiok" in output and "readyok" in output:
        print("[OK] USI初期化成功")
        return True
    else:
        print("[NG] USI初期化失敗")
        print(output)
        return False


def test_position_and_go() -> bool:
    """局面設定と思考テスト"""
    print("\n=== 局面設定・思考テスト ===")

    commands = [
        "usi",
        "setoption name USI_OwnBook value false",
        "isready",
        "position startpos",
        "go movetime 1000",
        "quit",
    ]

    output = run_usi_command(commands, timeout=60)

    # bestmoveが返ってくるか確認
    if "bestmove" in output:
        bestmove_match = re.search(r"bestmove (\S+)", output)
        if bestmove_match:
            print(f"[OK] 最善手: {bestmove_match.group(1)}")

        # 評価値を抽出
        cp_match = re.search(r"score cp (-?\d+)", output)
        if cp_match:
            print(f"[OK] 評価値: {cp_match.group(1)} cp")

        return True
    else:
        print("[NG] bestmoveが返ってこない")
        print(output)
        return False


def test_sfen_position() -> bool:
    """SFEN局面での評価テスト"""
    print("\n=== SFEN局面テスト ===")

    # 中盤の局面例
    sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"

    commands = [
        "usi",
        "setoption name USI_OwnBook value false",
        "isready",
        f"position sfen {sfen}",
        "go movetime 500",
        "quit",
    ]

    output = run_usi_command(commands, timeout=30)

    if "bestmove" in output:
        cp_match = re.search(r"score cp (-?\d+)", output)
        if cp_match:
            print(f"[OK] SFEN局面評価: {cp_match.group(1)} cp")
            return True

    print("[NG] SFEN局面評価失敗")
    return False


def extract_evaluation(output: str) -> int | None:
    """出力から評価値(cp)を抽出"""
    # 最後のinfo行からscoreを取得
    lines = output.strip().split("\n")
    for line in reversed(lines):
        if "score cp" in line:
            match = re.search(r"score cp (-?\d+)", line)
            if match:
                return int(match.group(1))
        elif "score mate" in line:
            match = re.search(r"score mate (-?\d+)", line)
            if match:
                mate_in = int(match.group(1))
                # 詰みの場合は大きな値を返す
                return 30000 if mate_in > 0 else -30000
    return None


def main():
    """メイン実行"""
    print("水匠5 USI通信テスト\n")

    results = []

    try:
        results.append(("USI初期化", test_usi_init()))
        results.append(("局面設定・思考", test_position_and_go()))
        results.append(("SFEN局面", test_sfen_position()))
    except FileNotFoundError as e:
        print(f"エラー: {e}")
        return 1
    except subprocess.TimeoutExpired:
        print("エラー: タイムアウト")
        return 1

    print("\n=== 結果サマリー ===")
    all_passed = True
    for name, passed in results:
        status = "OK" if passed else "NG"
        print(f"  {name}: [{status}]")
        if not passed:
            all_passed = False

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
