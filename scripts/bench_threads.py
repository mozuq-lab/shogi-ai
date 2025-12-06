#!/usr/bin/env python3
"""マルチスレッドNPS測定スクリプト"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shogi_utils import USIEngine, get_default_engine_path

# SFEN: 中盤の複雑な局面
SFEN = "ln1gkg1nl/6sb1/2pp1p1pp/pr2p4/1p5P1/2P1P1P2/PPBP1P2P/2G1S2R1/LN2KG1NL b - 1"


def main():
    print("マルチスレッドNPS測定\n")
    print(f"Engine: {get_default_engine_path().name}")
    print(f"Position: {SFEN}\n")

    results = []

    for threads in [1, 4, 8, 16]:
        print(f"Testing Threads={threads}...", end=" ", flush=True)

        with USIEngine(get_default_engine_path()) as engine:
            engine.init_usi()
            engine.set_option("Threads", threads)
            engine.set_option("USI_OwnBook", False)
            engine.set_option("BookFile", "no_book")
            engine.is_ready()
            engine.set_position(sfen=SFEN)

            import time
            start = time.time()
            result = engine.go(depth=18)
            elapsed = time.time() - start

            nps = result.nodes / elapsed if elapsed > 0 else 0
            results.append((threads, result.nodes, elapsed, nps))
            print(f"nodes={result.nodes:,}, time={elapsed:.2f}s, nps={nps/1e6:.2f}M")

    print("\n=== 結果サマリー ===")
    print(f"{'Threads':>8} | {'Nodes':>12} | {'Time (s)':>10} | {'NPS':>10}")
    print("-" * 50)
    for threads, nodes, elapsed, nps in results:
        print(f"{threads:>8} | {nodes:>12,} | {elapsed:>10.2f} | {nps/1e6:>9.2f}M")


if __name__ == "__main__":
    main()
