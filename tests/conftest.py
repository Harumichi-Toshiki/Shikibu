# 変更禁止（PROTECTED）。Shikibu M1 の受け入れテスト共通部品。
import os
import random
import struct
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))

import raypair_ref as ref  # noqa: E402

# 実装側の Python モジュール名（pyo3 + maturin でビルドする）
MODULE_NAME = os.environ.get("SHIKIBU_FEAT_MODULE", "shikibu_feat")

# ランダム局面テストの局面数（環境変数で増やせる。既定 20,000）
N_RANDOM = int(os.environ.get("SHIKIBU_N_RANDOM", "20000"))


@pytest.fixture(scope="session")
def feat():
    import importlib
    return importlib.import_module(MODULE_NAME)


def flip_sfen(s):
    """盤を180°回転し、先後を入れ替えた局面の SFEN を返す。"""
    b, stm, h = ref.parse_sfen(s)
    rows = []
    for y in range(9):
        r, e = "", 0
        for x in range(9):
            p = b[8 - y][8 - x]
            if p is None:
                e += 1
                continue
            if e:
                r += str(e)
                e = 0
            c, pt = p
            ch = pt[-1]
            ch = ch.lower() if c == 0 else ch.upper()
            r += ("+" if pt.startswith("+") else "") + ch
        if e:
            r += str(e)
        rows.append(r)
    hs = ""
    for c in (1, 0):
        for i in range(6, -1, -1):
            n = h[c][i]
            if n:
                ch = ref.HAND_ORDER[i]
                ch = ch.upper() if c == 1 else ch.lower()
                hs += (str(n) if n > 1 else "") + ch
    return "/".join(rows) + (" w " if stm == 0 else " b ") + (hs or "-") + " 1"


def random_boards(n, seed=20261020):
    """cshogi でランダムに指し進めた局面（cshogi.Board）を n 個返す。"""
    import cshogi
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b = cshogi.Board()
        plies = rng.randrange(0, 200)
        ok = True
        for _ in range(plies):
            moves = list(b.legal_moves)
            if not moves:
                ok = False
                break
            b.push(rng.choice(moves))
        if ok and list(b.legal_moves):
            out.append(b)
    return out


def make_psv(board, score, move, ply, result):
    """cshogi の局面から PSV（40バイト）を作る。"""
    buf = np.zeros(32, dtype=np.uint8)
    board.to_psfen(buf)
    return bytes(buf) + struct.pack("<hHHbB", score, move, ply, result, 0)


def normalize(sfen):
    """SFEN を (盤面, 手番, 持ち駒) に正規化して比較できる形にする。"""
    b, stm, h = ref.parse_sfen(sfen)
    return (tuple(tuple(r) for r in b), stm, tuple(tuple(x) for x in h))
