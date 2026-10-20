"""変更禁止（PROTECTED）。Shikibu の速度の合格判定。

使い方：
    python perf/perf_check.py --profile dev     # 開発環境（Claude Code のクラウド、4論理コア程度）
    python perf/perf_check.py --profile colab   # Google Colab の本番環境（Michi が実行）

前提とする API（shikibu_feat）：
    bench_extract(paths: list[str], n: int, version: int, num_threads: int) -> dict
        Rust の中だけで、PSV n 局面について「解読 → 両視点の特徴抽出（昇順・重複込み）」を行い、
        {"positions": int, "seconds": float} を返す。Python を介さない純粋な計算速度。
    BatchLoader（M2 で実装済み）

判定は「合格（PASS）／不合格（FAIL）」を表示し、1つでも不合格なら終了コード 1 を返す。
"""
import argparse
import json
import os
import platform
import random
import struct
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import numpy as np  # noqa: E402

# 合格基準（局面/秒）
PROFILES = {
    # 開発環境：1スレッドの純粋計算と、4スレッドのローダー
    "dev": {"core_1t": 200_000, "loader_threads": 4, "loader": 500_000},
    # Colab 本番：論理コア数 − 1 スレッドのローダーで 100 万局面/秒
    "colab": {"core_1t": 200_000, "loader_threads": None, "loader": 1_000_000},
}

N_DISTINCT = 20_000      # 異なる局面の数（ランダムに指し進めて作る）
N_TOTAL = 1_000_000      # 計測に使う局面数（異なる局面を評価値を変えて繰り返す）
BATCH = 16_384


def make_bench_file(path):
    from conftest import make_psv, random_boards
    boards = random_boards(N_DISTINCT, seed=424242)
    heads = [make_psv(b, 0, 0, 0, 0)[:32] for b in boards]
    rng = random.Random(1)
    with open(path, "wb") as f:
        for i in range(N_TOTAL):
            f.write(heads[i % N_DISTINCT] + struct.pack("<hHHbB", rng.randrange(-3000, 3001), 0,
                                                          rng.randrange(0, 300), rng.choice([-1, 0, 1]), 0))


def run_loader(feat, path, version, threads):
    t = time.perf_counter()
    n = 0
    checksum = 0
    for b in feat.BatchLoader([path], BATCH, version, shuffle_buffer=0, num_threads=threads):
        n += len(b["score"])
        # 学習ループと同じく、配列の中身に実際に触れる
        checksum += int(b["stm_indices"][-1]) + int(b["nstm_indices"][-1])
    return n / (time.perf_counter() - t), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=PROFILES, required=True)
    ap.add_argument("--module", default=os.environ.get("SHIKIBU_FEAT_MODULE", "shikibu_feat"))
    args = ap.parse_args()
    import importlib
    feat = importlib.import_module(args.module)
    prof = PROFILES[args.profile]
    cpus = os.cpu_count() or 1
    loader_threads = prof["loader_threads"] or max(1, cpus - 1)

    print(f"環境: {platform.platform()} / Python {platform.python_version()} / 論理コア {cpus}")
    results, ok = [], True
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bench.psv")
        make_bench_file(path)
        for version in (0, 1):
            r = feat.bench_extract([path], N_TOTAL, version, 1)
            assert r["positions"] == N_TOTAL, r
            core = r["positions"] / r["seconds"]
            pass_core = core >= prof["core_1t"]
            lps, n = run_loader(feat, path, version, loader_threads)
            assert n == N_TOTAL
            pass_loader = lps >= prof["loader"]
            ok &= pass_core and pass_loader
            results.append({"version": version, "core_1thread": round(core), "loader": round(lps),
                            "loader_threads": loader_threads})
            print(f"v{version}: 純粋計算 1スレッド {core:,.0f} 局面/秒 (基準 {prof['core_1t']:,}) "
                  f"{'PASS' if pass_core else 'FAIL'} / ローダー {loader_threads}スレッド {lps:,.0f} 局面/秒 "
                  f"(基準 {prof['loader']:,}) {'PASS' if pass_loader else 'FAIL'}")
    print(json.dumps({"profile": args.profile, "cpus": cpus, "results": results, "pass": ok}, ensure_ascii=False))
    print("総合:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
