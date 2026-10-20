# 変更禁止（PROTECTED）。Shikibu M3 受け入れテスト用のデータ生成。
import os
import random
import struct

from conftest import make_psv, random_boards, ref
from m2_data import write_records

PIECE_VALUE = {"P": 1, "L": 3, "N": 4, "S": 5, "G": 6, "B": 8, "R": 10, "K": 0}


def material_score(sfen):
    """手番側から見た駒得（盤上＋持ち駒、成駒は元の駒の価値）× 150。"""
    board, stm, hands = ref.parse_sfen(sfen)
    tot = [0, 0]
    for row in board:
        for p in row:
            if p:
                tot[p[0]] += PIECE_VALUE[p[1].lstrip("+")]
    for c in (0, 1):
        for i, h in enumerate(ref.HAND_ORDER):
            tot[c] += PIECE_VALUE[h] * hands[c][i]
    return max(-30000 + 1, min(30000 - 1, 150 * (tot[stm] - tot[1 - stm])))


def make_material_dataset(dirpath, n_files=2, per_file=3000, seed=31):
    """評価値＝駒得（学習で必ず当てられる）の PSV を作る。勝敗は評価値の符号。"""
    os.makedirs(dirpath, exist_ok=True)
    boards = random_boards(n_files * per_file, seed=seed)
    rng = random.Random(seed)
    paths, uid = [], 0
    for fi in range(n_files):
        recs = []
        for _ in range(per_file):
            b = boards[uid]
            s = material_score(b.sfen())
            res = 1 if s > 0 else (-1 if s < 0 else 0)
            recs.append(make_psv(b, s, uid, rng.randrange(0, 300), res))
            uid += 1
        p = os.path.join(dirpath, f"mat_{fi:02d}.psv")
        write_records(p, recs)
        paths.append(p)
    return paths
