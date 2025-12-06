#!/bin/bash
# Shogi-AI USIエンジン起動スクリプト
cd "$(dirname "$0")"
PYTHONPATH=. .venv/bin/python engine/usi_server.py --model checkpoints/best.pt "$@"
