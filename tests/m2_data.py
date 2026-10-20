# 変更禁止（PROTECTED）。Shikibu M2 受け入れテスト用のデータ生成。
import math
import os
import random
import struct

import numpy as np

from conftest import make_psv, random_boards

REC = 40


def write_records(path, recs):
    with open(path, "wb") as f:
        for r in recs:
            assert len(r) == REC
            f.write(r)


def read_records(path):
    data = open(path, "rb").read()
    assert len(data) % REC == 0
    return [data[i:i + REC] for i in range(0, len(data), REC)]


def rec_id(rec):
    """テスト用データでは move 欄に通し番号を入れている。"""
    return struct.unpack("<H", rec[34:36])[0]


def make_dataset(dirpath, n_files=3, per_file=3000, seed=7):
    """ランダム局面の PSV ファイルを n_files 個作る。move 欄 = 通し番号（一意）。"""
    os.makedirs(dirpath, exist_ok=True)
    boards = random_boards(n_files * per_file, seed=seed)
    rng = random.Random(seed)
    paths, all_recs, uid = [], [], 0
    for fi in range(n_files):
        recs = []
        for _ in range(per_file):
            b = boards[uid]
            recs.append(make_psv(b, rng.randrange(-5000, 5001), uid, rng.randrange(0, 300),
                                 rng.choice([-1, 0, 1])))
            uid += 1
        p = os.path.join(dirpath, f"data_{fi:02d}.psv")
        write_records(p, recs)
        paths.append(p)
        all_recs += recs
    return paths, all_recs


def make_scale_dataset(path, true_scale, n=200_000, seed=11, draw_rate=0.0, n_mate=2000):
    """sigmoid(score/true_scale) の確率で勝敗を振った人工データ。
    |score|>=30000 の局面（勝敗を逆にしてある）も混ぜ、除外されることを確かめる。"""
    import cshogi
    rng = random.Random(seed)
    buf = np.zeros(32, dtype=np.uint8)
    cshogi.Board().to_psfen(buf)
    head = bytes(buf)
    with open(path, "wb") as f:
        for i in range(n):
            score = rng.randrange(-4000, 4001)
            if rng.random() < draw_rate:
                res = 0
            else:
                res = 1 if rng.random() < 1.0 / (1.0 + math.exp(-score / true_scale)) else -1
            f.write(head + struct.pack("<hHHbB", score, i & 0xFFFF, 0, res, 0))
        for i in range(n_mate):
            score = 32000 if i % 2 == 0 else -32000
            res = -1 if score > 0 else 1
            f.write(head + struct.pack("<hHHbB", score, 0, 0, res, 0))
