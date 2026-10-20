#!/usr/bin/env python
"""Shikibu のデータ用コマンド（§S3.1・§S4）。

  python tools/shk_tools.py stats DATA...
  python tools/shk_tools.py estimate-scale DATA... [--sample N] [--seed S]
  python tools/shk_tools.py shuffle DATA... --out DIR [--records-per-shard N] [--seed S] [--mem-limit-mb M]

DATA は PSV ファイル（1局面40バイト）。
"""
import argparse
import json
import sys

import shikibu_feat as feat

# 1シャードは 1GB（25,000,000 局面、§S3.1）
DEFAULT_RECORDS_PER_SHARD = 25_000_000


def cmd_stats(args):
    st = feat.psv_stats(args.paths)
    print(json.dumps(st, ensure_ascii=False, indent=2))
    if st["result_missing"]:
        print(
            "警告: 勝敗が記録されていないとみなされる（game_result == 0 が95%超、§S4-2）。"
            "eval_scale は推定できない。手動指定と λ = 1.0 で学習する。",
            file=sys.stderr,
        )
    return 0


def cmd_estimate_scale(args):
    try:
        r = feat.estimate_eval_scale(args.paths, args.sample, args.seed)
    except ValueError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    print(json.dumps(r, ensure_ascii=False, indent=2))
    if r["eval_scale"] >= 0.99 * 5000 or r["eval_scale"] <= 1.01 * 50:
        print(
            "警告: 推定値が探索範囲（[50, 5000]）の端に張り付いている。"
            "勝敗と評価値の対応が弱い可能性がある（§S4-3）。",
            file=sys.stderr,
        )
    return 0


def cmd_shuffle(args):
    shards = feat.shuffle_shards(
        args.paths, args.out, args.records_per_shard, args.seed, args.mem_limit_mb
    )
    for s in shards:
        print(s)
    print(f"{len(shards)} シャードを書き出した", file=sys.stderr)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Shikibu のデータ用コマンド")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("stats", help="データの統計（§S4-1・§S4-2）")
    p.add_argument("paths", nargs="+")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("estimate-scale", help="eval_scale の推定（§S4-3）")
    p.add_argument("paths", nargs="+")
    p.add_argument("--sample", type=int, default=10_000_000, help="抽出する局面数（既定 1,000 万）")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_estimate_scale)

    p = sub.add_parser("shuffle", help="事前シャッフルとシャード分割（§S3.1）")
    p.add_argument("paths", nargs="+")
    p.add_argument("--out", required=True, help="出力フォルダ（空であること）")
    p.add_argument("--records-per-shard", type=int, default=DEFAULT_RECORDS_PER_SHARD)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--mem-limit-mb",
        type=int,
        default=4096,
        help="使うメモリの目安（MB）。足りなければバケツに分けて2パスで処理する",
    )
    p.set_defaults(func=cmd_shuffle)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
