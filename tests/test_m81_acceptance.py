# 変更禁止（PROTECTED）。Shikibu M8.1（壊れたレコードを飛ばして学習を続ける）の受け入れテスト。
#
# 背景：100億局面の学習データのどこかに、読み解けないレコード（"PackedSfen のビット列が途中で終わった" など）が混ざっていて、
# 学習が毎回同じところ（約 6,000 ステップ）で止まった。1つの壊れたレコードで数時間の学習を止めないようにする。
#
# ■ 仕様
#   BatchLoader：読み解けないレコード（decode_psv がエラーになるもの）と、game_result が −1〜1 の外のレコードは、
#     エラーにせず飛ばす（そのバッチの局面数がその分だけ減る）。飛ばした数を数える。
#     BatchLoader.skipped() -> int：このローダーで飛ばしたレコードの数（作ってからの合計）。
#     読み込みの位置（state()・resume_state）は今までどおり「ファイルの中のレコードの位置」で数える（飛ばしたレコードも1つと数える）。
#   学習（shikibu.train）：ログの "train" の行に "bad_records"（この学習で飛ばしたレコードの合計。周回をまたいで足す。再開したら
#     checkpoint に保存した値から続ける）を書く。飛ばしたレコードが1つ以上あったら、標準出力に1回だけ
#     「壊れたレコードを飛ばしました」を含む行を出す。検証データ（val_path）の読み込みも同じく飛ばす。
#   eval_scale の確認・推定（psv_stats・estimate_eval_scale）も、読み解けないレコードで止まらないこと。
#   decode_psv・extract_psv は今までどおり（壊れたレコードではエラー）。
import json
import os
import random
import struct
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))

import cshogi  # noqa: E402
from conftest import make_psv, random_boards  # noqa: E402
from m2_data import write_records  # noqa: E402
from m3_data import material_score  # noqa: E402
from test_m3_acceptance import load_ckpt, make_config, read_log, run  # noqa: E402


def bad_records(feat, n, seed):
    """読み解けないレコードを n 個作る（半分はビット列が途中で終わるもの、半分は乱数）。"""
    rng = random.Random(seed)
    good = make_psv(random_boards(1, seed=seed)[0], 0, 0, 10, 1)
    out = []
    while len(out) < n:
        if len(out) % 2 == 0:
            r = good[:2] + b"\xff" * 30 + good[32:]
        else:
            r = rng.randbytes(32) + struct.pack("<hHHbB", 0, 0, 0, 1, 0)
        try:
            feat.decode_psv(r)
        except Exception:
            out.append(r)
    return out


def mixed_file(feat, path, boards, n_bad, seed, bad_result=0):
    rng = random.Random(seed)
    recs = []
    for i, b in enumerate(boards):
        s = material_score(b.sfen())
        recs.append(make_psv(b, s, i, 50, 1 if s > 0 else (-1 if s < 0 else 0)))
    bads = bad_records(feat, n_bad, seed)
    for _ in range(bad_result):  # 局面は正しいが game_result が範囲の外
        bads.append(make_psv(boards[0], 0, 0, 10, 5))
    for r in bads:
        recs.insert(rng.randrange(len(recs) + 1), r)
    write_records(path, recs)
    return len(bads)


def test_loader_skips_bad_records(feat, tmp_path):
    boards = random_boards(500, seed=8101)
    p = str(tmp_path / "mix.psv")
    nbad = mixed_file(feat, p, boards, 7, 1, bad_result=2)
    with pytest.raises(Exception):
        feat.decode_psv(bad_records(feat, 1, 2)[0])  # decode_psv は今までどおり
    ld = feat.BatchLoader([p], 128, 0, shuffle_buffer=0, num_threads=2)
    got = 0
    for batch in ld:
        got += len(batch["score"])
    assert got == len(boards), f"正しいレコードを全部読めていない（{got} / {len(boards)}）"
    assert ld.skipped() == nbad, f"飛ばした数が違う（{ld.skipped()} / {nbad}）"


def test_loader_resume_counts_bad_records_as_positions(feat, tmp_path):
    boards = random_boards(400, seed=8102)
    p = str(tmp_path / "mix.psv")
    mixed_file(feat, p, boards, 5, 3)
    full = [s for b in feat.BatchLoader([p], 64, 0, shuffle_buffer=0, num_threads=1) for s in b["score"].tolist()]
    ld = feat.BatchLoader([p], 64, 0, shuffle_buffer=0, num_threads=1)
    first = [s for _, b in zip(range(3), ld) for s in b["score"].tolist()]
    st = dict(ld.state())
    rest = [s for b in feat.BatchLoader([p], 64, 0, shuffle_buffer=0, num_threads=1, resume_state=st)
            for s in b["score"].tolist()]
    assert first + rest == full, "途中から再開したときの局面の並びが、続けて読んだときと違う"


def test_stats_tolerate_bad_records(feat, tmp_path):
    boards = random_boards(2000, seed=8103)
    p = str(tmp_path / "mix.psv")
    mixed_file(feat, p, boards, 9, 4)
    feat.psv_stats([p])
    feat.estimate_eval_scale([p], sample=2000, seed=0)


def test_training_survives_bad_records(feat, tmp_path):
    paths, total_bad = [], 0
    for i in range(2):
        p = str(tmp_path / f"t{i}.psv")
        total_bad += mixed_file(feat, p, random_boards(1500, seed=8110 + i), 6, 10 + i)
        paths.append(p)
    val = str(tmp_path / "val.psv")
    mixed_file(feat, val, random_boards(600, seed=8120), 3, 20)
    out = tmp_path / "o"
    # 2周以上回す（1周 = 3000 局面 = 約 12 ステップ）
    cfg = make_config(tmp_path / "c.toml", paths, out, data__val_path=val, train__max_steps=30, train__total_steps=30,
                      train__val_every_steps=10, train__val_positions=500, train__eval_scale="auto",
                      train__lam=0.75, train__save_every_steps=15)
    assert run(["--config", cfg]) == 0, "壊れたレコードのあるデータで学習が止まった"
    log = read_log(str(out))
    tr = [r for r in log if r.get("type") == "train"]
    assert tr and "bad_records" in tr[-1], "ログの train の行に bad_records が無い"
    assert tr[-1]["bad_records"] >= total_bad, f"bad_records が少ない（{tr[-1]['bad_records']} / 1周で {total_bad}）"
    assert any(r.get("type") == "val" for r in log)
    assert load_ckpt(str(out))["step"] == 30
    # 再開しても続きから数える
    cfg2 = make_config(tmp_path / "c2.toml", paths, out, data__val_path=val, train__max_steps=40, train__total_steps=30,
                       train__val_every_steps=10, train__val_positions=500, train__eval_scale="auto",
                       train__lam=0.75, train__save_every_steps=15)
    assert run(["--config", cfg2, "--resume"]) == 0
    tr2 = [r for r in read_log(str(out)) if r.get("type") == "train"]
    assert tr2[-1]["bad_records"] >= tr[-1]["bad_records"], "再開したら bad_records が減った"
