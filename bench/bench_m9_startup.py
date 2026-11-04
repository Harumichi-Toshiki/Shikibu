"""M9：学習の開始前の段階の時間が、データ量に対してどう増えるかを測る（ローカル・CPU。比例を見るためのもの）。

使い方:  python bench/bench_m9_startup.py --dir <作業フォルダ> --sizes 2000000,8000000,32000000,128000000

同じ 6,000 局面（有効な局面）を繰り返した PSV を作り、局面数（--sizes）ごとに次を測る。
  S  データの確認 sample（既定）：抜き出し（最大 scale_sample 局面）＋ psv_stats ＋ estimate_eval_scale
  P  データの確認 probe（probe_records = 3000、全ファイル）
  F  データの確認 full（全件の psv_stats ＋ 推定）
  B  シャッフルの充填：最初のバッチまでの時間（shuffle_buffer = --buffer）
  W  シャッフルの充填 shuffle_warm（--warm 件から開始）
Drive の代わりにローカルのディスクで測るので、絶対値は Colab と違う。見るのは「増え方」。
作ったファイルは最後に消す（--keep で残す）。
"""
import argparse
import glob
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from m2_data import make_dataset  # noqa: E402  （有効な局面を作る既存のテスト用データ）
import shikibu_feat  # noqa: E402
from shikibu import train  # noqa: E402

REC = 40
BASE_RECORDS = 6000
FILE_RECORDS = 1_000_000  # 1 ファイルの局面数（40 MB）


def make_files(work, n_records, base_bytes):
    d = os.path.join(work, f"n{n_records}")
    os.makedirs(d, exist_ok=True)
    n_files = max(1, n_records // FILE_RECORDS)
    per = n_records // n_files
    reps, rest = divmod(per, BASE_RECORDS)
    chunk = base_bytes * reps + base_bytes[: rest * REC]
    paths = []
    for i in range(n_files):
        p = os.path.join(d, f"shard_{i:04d}.psv")
        if not os.path.exists(p) or os.path.getsize(p) != len(chunk):
            with open(p, "wb") as f:
                f.write(chunk)
        paths.append(p)
    return paths


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return time.perf_counter() - t, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(tempfile.gettempdir(), "shk_m9_bench"))
    ap.add_argument("--sizes", default="2000000,8000000,32000000,128000000")
    ap.add_argument("--buffer", type=int, default=2_000_000)
    ap.add_argument("--warm", type=int, default=200_000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.dir, exist_ok=True)

    seed_paths, _ = make_dataset(os.path.join(args.dir, "base"), n_files=1, per_file=BASE_RECORDS, seed=7)
    with open(seed_paths[0], "rb") as f:
        base_bytes = f.read()
    tmp = os.path.join(args.dir, "check_sample.psv")
    rows = []
    for n in [int(x) for x in args.sizes.split(",")]:
        paths = make_files(args.dir, n, base_bytes)
        total = sum(os.path.getsize(p) for p in paths) // REC
        # S：sample（既定）
        t_pick, (files, count) = timed(lambda: train.sample_scale_positions(paths, 10_000_000, 0, tmp))
        t_stat, _ = timed(lambda: (shikibu_feat.psv_stats(files),
                                   shikibu_feat.estimate_eval_scale(files, sample=min(count, 10_000_000), seed=0)))
        if files != paths and os.path.exists(tmp):
            os.remove(tmp)
        s_total = t_pick + t_stat
        # P：probe
        t_pick, (files, count, _) = timed(lambda: train.probe_positions(paths, 3000, tmp))
        t_stat, _ = timed(lambda: (shikibu_feat.psv_stats(files),
                                   shikibu_feat.estimate_eval_scale(files, sample=count, seed=0)))
        if os.path.exists(tmp):
            os.remove(tmp)
        p_total = t_pick + t_stat
        # F：full（全件）
        t_full, _ = timed(lambda: (shikibu_feat.psv_stats(paths),
                                   shikibu_feat.estimate_eval_scale(paths, sample=10_000_000, seed=0)))
        # B / W：最初のバッチまで
        def first_batch(**kw):
            it = shikibu_feat.BatchLoader(paths, 65536, 2, seed=0, num_threads=args.threads, **kw)
            next(it)
            del it
        t_b, _ = timed(lambda: first_batch(shuffle_buffer=args.buffer))
        t_w, _ = timed(lambda: first_batch(shuffle_buffer=args.buffer, shuffle_warm=args.warm))
        rows.append((total, s_total, p_total, t_full, t_b, t_w))
        print(f"{total:>12,} 局面  S {s_total:7.1f}s  P {p_total:7.1f}s  F {t_full:7.1f}s  "
              f"B {t_b:6.2f}s  W {t_w:6.2f}s", flush=True)

    print("\n| 局面数 | S（既定） | P（probe） | F（全件） | B（バッファ充填） | W（warm） |")
    print("|---:|---:|---:|---:|---:|---:|")
    for total, s, p, f, b, w in rows:
        print(f"| {total:,} | {s:.1f} s | {p:.1f} s | {f:.1f} s | {b:.2f} s | {w:.2f} s |")
    if not args.keep:
        shutil.rmtree(args.dir, ignore_errors=True)


if __name__ == "__main__":
    main()
