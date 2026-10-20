#!/usr/bin/env python
"""BatchLoader の速度（毎秒の局面数）を計測する（§S9）。

  python bench/bench_loader.py [--records 1000000] [--batch 16384] [--workdir DIR]

num_threads = 1, 2, 4, 8 と version = 0, 1 の組み合わせで、全データを1周する時間を測る。
あわせて、読み込みと PSV の解読だけの速さ（psv_stats）を参考値として出す。
データは乱数で作った局面（重複あり。ユニークな局面を循環させて件数を揃える）。
"""
import argparse
import os
import platform
import random
import struct
import tempfile
import time

import cshogi
import numpy as np
import shikibu_feat as feat

UNIQUE_BOARDS = 5_000  # 乱数で指し進めた局面の数（生成時間を抑える）
MAX_PLIES = 200


def random_board(rng):
    b = cshogi.Board()
    for _ in range(rng.randrange(0, MAX_PLIES)):
        moves = list(b.legal_moves)
        if not moves:
            break
        b.push(rng.choice(moves))
    return b


def make_data(dirpath, n_records, n_files=4, seed=1):
    """ユニークな局面を循環させて n_records 件の PSV を n_files 個のファイルに書く。"""
    rng = random.Random(seed)
    boards = []
    while len(boards) < UNIQUE_BOARDS:
        b = random_board(rng)
        if list(b.legal_moves):
            buf = np.zeros(32, dtype=np.uint8)
            b.to_psfen(buf)
            boards.append(bytes(buf))
    paths = []
    per_file = n_records // n_files
    for fi in range(n_files):
        p = os.path.join(dirpath, f"bench_{fi:02d}.psv")
        with open(p, "wb") as f:
            for i in range(per_file):
                k = (fi * per_file + i) % UNIQUE_BOARDS
                score = rng.randrange(-3000, 3001)
                result = rng.choice([-1, 0, 1])
                f.write(boards[k] + struct.pack("<hHHbB", score, i & 0xFFFF, i & 0xFFFF, result, 0))
        paths.append(p)
    return paths, per_file * n_files


def timed(fn):
    t = time.perf_counter()
    n = fn()
    return n, time.perf_counter() - t


def run_loader(paths, batch, version, threads):
    def go():
        n = 0
        for b in feat.BatchLoader(paths, batch, version, num_threads=threads):
            n += len(b["score"])
        return n
    return timed(go)


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "不明"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=int, default=1_000_000)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--workdir", default=None, help="データを置くフォルダ（既定は一時フォルダ）")
    args = ap.parse_args()

    print("## 計測環境")
    print(f"- CPU: {cpu_model()}（論理コア数 {os.cpu_count()}）")
    print(f"- OS: {platform.platform()}")
    print(f"- Python {platform.python_version()}, numpy {np.__version__}")
    print(f"- バッチサイズ {args.batch}、シャッフルなし")
    print()

    with tempfile.TemporaryDirectory(dir=args.workdir) as d:
        print(f"データを生成中（{args.records:,} 局面）...", flush=True)
        paths, n = make_data(d, args.records)

        # 参考値：読み込みと PSV の解読だけ（特徴抽出なし）
        _, sec = timed(lambda: feat.psv_stats(paths) and n)
        print("## 参考：読み込みと解読だけ（psv_stats）")
        print(f"- {n / sec:,.0f} 局面/秒（{sec:.2f} 秒）")
        print()

        print("## BatchLoader（局面/秒）")
        print("| version | num_threads | 局面数 | 秒 | 局面/秒 |")
        print("|---|---|---|---|---|")
        for version in (0, 1):
            for threads in (1, 2, 4, 8):
                got, sec = run_loader(paths, args.batch, version, threads)
                assert got == n, f"件数が違う: {got} != {n}"
                print(f"| {version} | {threads} | {got:,} | {sec:.2f} | {got / sec:,.0f} |", flush=True)


if __name__ == "__main__":
    main()
