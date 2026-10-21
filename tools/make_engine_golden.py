"""エンジン（Nuenagon）用の正解データを作る（Nuenagon の N3 で使う）。

使い方：
    python tools/make_engine_golden.py --shk <file.shk> --out <golden.json> --n <局面数> --seed <整数>

局面はランダムに指し進めて作る（seed で決まる）。先頭は平手初期局面。
各局面について、特徴（昇順・重複込み）、量子化した推論の途中の値（§E4）、評価値 cp を書く。
JSON の形式は「shikibu-engine-golden-v1」（tests/test_m4_acceptance.py の冒頭を参照）。
"""
import argparse
import hashlib
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import cshogi  # noqa: E402
import shikibu_feat  # noqa: E402
from shikibu import export, quantsim  # noqa: E402

FORMAT = "shikibu-engine-golden-v1"
MAX_PLY = 200  # この手数で打ち切り、初期局面から指し直す


def random_sfens(n, seed):
    """初期局面から乱数で指し進めた局面の SFEN を n 個返す。打ち切り・詰みのあとは指し直す。"""
    rng = random.Random(seed)
    out = [cshogi.Board().sfen()]
    board, ply = cshogi.Board(), 0
    while len(out) < n:
        moves = list(board.legal_moves)
        if not moves or ply >= MAX_PLY:
            board, ply = cshogi.Board(), 0
            continue
        board.push(rng.choice(moves))
        ply += 1
        out.append(board.sfen())
    return out


def build(shk_path, n, seed):
    header, tensors = export.read_shk(shk_path)
    version = quantsim.FEATURE_VERSIONS[header["feature_set"]]
    q = quantsim.quantize(tensors)
    es = header["eval_scale"]
    positions = []
    for sfen in random_sfens(n, seed):
        stm = 0 if sfen.split()[1] == "b" else 1
        stm_f = shikibu_feat.extract_sfen(sfen, stm, version)
        nstm_f = shikibu_feat.extract_sfen(sfen, 1 - stm, version)
        a, b, raw = quantsim.qforward(q, stm_f, nstm_f)
        positions.append({
            "sfen": sfen,
            "stm_features": stm_f,
            "nstm_features": nstm_f,
            "acc_stm": [int(v) for v in a],
            "acc_nstm": [int(v) for v in b],
            "raw": raw,
            "cp": quantsim.cp_of(raw, es),
        })
    with open(shk_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return {"format": FORMAT, "shk_sha256": sha, "header": header, "positions": positions}


def main(argv=None):
    p = argparse.ArgumentParser(prog="python tools/make_engine_golden.py")
    p.add_argument("--shk", required=True, help="対象の .shk")
    p.add_argument("--out", required=True, help="書き出す JSON")
    p.add_argument("--n", type=int, required=True, help="局面数")
    p.add_argument("--seed", type=int, required=True, help="局面を作る乱数の種")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    if args.n < 1:
        print("エラー: --n は1以上", file=sys.stderr)
        return 1
    try:
        golden = build(args.shk, args.n, args.seed)
    except (ValueError, OSError, RuntimeError) as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(golden, f, ensure_ascii=False, indent=1)
    print(f"正解データを書きました: {args.out}（{len(golden['positions'])} 局面）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
