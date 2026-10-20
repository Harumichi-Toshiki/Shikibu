# 変更禁止（PROTECTED）。Shikibu M2 の受け入れテスト。
# 実装（shikibu_feat モジュール、M1 と同じモジュールに追加する）が満たすべき API：
#
#   psv_stats(paths: list[str]) -> dict
#       {"records": int, "result_counts": {"-1": int, "0": int, "1": int},
#        "score_min": int, "score_max": int, "score_mean": float, "result_missing": bool}
#       result_missing は game_result == 0 の割合が 0.95 を超えるとき True（§S4-2）
#
#   estimate_eval_scale(paths: list[str], sample: int = 10_000_000, seed: int = 0) -> dict
#       {"eval_scale": float, "bce": float, "used": int}
#       §S4-3 の方法（|score|>=30000 を除外、勝=1/引分=0.5/負=0 との BCE を
#       log S について [50, 5000] で黄金分割探索）。result_missing なら ValueError を送出する。
#
#   shuffle_shards(inputs: list[str], out_dir: str, records_per_shard: int,
#                  seed: int, mem_limit_mb: int) -> list[str]
#       全局面をシャッフルし out_dir/shard_00000.psv, shard_00001.psv, ... に書き出し、
#       パスの昇順リストを返す。最後以外のシャードはちょうど records_per_shard 局面。
#       メモリ使用は mem_limit_mb 程度に抑える（足りなければバケツに分ける2パス方式、§S3.1）。
#
#   class BatchLoader(paths: list[str], batch_size: int, version: int,
#                     shuffle_buffer: int = 0, seed: int = 0, num_threads: int = 1,
#                     cache_dir: str | None = None, max_cached: int = 3,
#                     resume_state: dict | None = None)
#       イテレータ。全ファイルを1周したら終わる（最後の端数バッチも返す）。各要素は dict：
#         "stm_indices", "stm_offsets", "nstm_indices", "nstm_offsets": numpy int64
#         "score": numpy float32, "result": numpy float32（勝=1 / 引分=0.5 / 負=0）
#       stm = 手番側視点、nstm = 非手番側視点。indices/offsets は nn.EmbeddingBag の形式で、
#       各局面の特徴は extract_psv(rec, 視点, version) と同じ（昇順・重複込み）。
#       shuffle_buffer = 0 のときはファイルの順・ファイル内の順のまま返す。
#       並びは seed と shuffle_buffer だけで決まり、num_threads や cache_dir によらない。
#       state() -> dict：JSON にできる再開用の状態（直前に返したバッチの直後の位置）。
#       resume_state を渡すとその位置から再開する。
#         shuffle_buffer = 0 なら、止めずに回した場合と完全に同じ続きを返すこと。
#         shuffle_buffer > 0 なら、バッファに残っていた局面は失ってよいが、重複して返してはいけない。
#       cache_dir を指定したら、読む前にファイルを cache_dir へコピーしてから読み（先読み可）、
#       読み終えたファイルは削除する。cache_dir 内の .psv は常に max_cached + 1 個以下。
import json
import os
import struct
from collections import Counter

import numpy as np
import pytest

from m2_data import (make_dataset, make_scale_dataset, read_records, rec_id,
                     write_records)

BATCH = 512


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    d = tmp_path_factory.mktemp("m2data")
    paths, recs = make_dataset(str(d))
    return [str(p) for p in paths], recs


def unpack_batch(b):
    """バッチを (局面ごとの stm 特徴, nstm 特徴, score, result) のリストにする。"""
    out = []
    for key in ("stm", "nstm"):
        idx, off = np.asarray(b[f"{key}_indices"]), np.asarray(b[f"{key}_offsets"])
        assert idx.dtype == np.int64 and off.dtype == np.int64
        bounds = list(off) + [len(idx)]
        out.append([idx[bounds[i]:bounds[i + 1]].tolist() for i in range(len(off))])
    score = np.asarray(b["score"])
    result = np.asarray(b["result"])
    assert score.dtype == np.float32 and result.dtype == np.float32
    assert len(score) == len(result) == len(out[0]) == len(out[1])
    return list(zip(out[0], out[1], score.tolist(), result.tolist()))


def expected_sample(feat, rec, version):
    sfen = feat.decode_psv(rec)[0]
    stm = 0 if sfen.split()[1] == "b" else 1
    score, res = struct.unpack("<h", rec[32:34])[0], struct.unpack("<b", rec[38:39])[0]
    return (list(feat.extract_psv(rec, stm, version)), list(feat.extract_psv(rec, 1 - stm, version)),
            float(score), (res + 1) / 2.0)


def collect(loader, limit=None):
    """ローダーから最大 limit バッチ取り出し、局面ごとのサンプルの一覧を返す。"""
    samples = []
    for i, b in enumerate(loader):
        samples += unpack_batch(b)
        if limit is not None and i + 1 >= limit:
            break
    return samples


# ---------- 統計 ----------

def test_psv_stats(feat, dataset):
    paths, recs = dataset
    st = feat.psv_stats(paths)
    assert st["records"] == len(recs)
    cnt = Counter(struct.unpack("<b", r[38:39])[0] for r in recs)
    assert {k: st["result_counts"][k] for k in ("-1", "0", "1")} == {"-1": cnt[-1], "0": cnt[0], "1": cnt[1]}
    scores = [struct.unpack("<h", r[32:34])[0] for r in recs]
    assert st["score_min"] == min(scores) and st["score_max"] == max(scores)
    assert abs(st["score_mean"] - sum(scores) / len(scores)) < 1e-3
    assert st["result_missing"] is False


# ---------- eval_scale の推定 ----------

@pytest.mark.parametrize("true_scale", [500.0, 1200.0])
def test_estimate_eval_scale(feat, tmp_path, true_scale):
    p = str(tmp_path / "scale.psv")
    make_scale_dataset(p, true_scale)
    r = feat.estimate_eval_scale([p])
    assert abs(r["eval_scale"] - true_scale) / true_scale < 0.05, r
    assert r["used"] == 200_000  # |score|>=30000 の 2,000 局面は除外される
    assert 0.0 < r["bce"] < 0.7


def test_estimate_eval_scale_rejects_missing_results(feat, tmp_path):
    p = str(tmp_path / "noresult.psv")
    make_scale_dataset(p, 600.0, n=20_000, draw_rate=1.0, n_mate=0)
    assert feat.psv_stats([p])["result_missing"] is True
    with pytest.raises(ValueError):
        feat.estimate_eval_scale([p])


# ---------- シャッフルとシャード分割 ----------

@pytest.mark.parametrize("mem_limit_mb", [64, 1])
def test_shuffle_shards(feat, dataset, tmp_path, mem_limit_mb):
    paths, recs = dataset
    out = tmp_path / f"shards_{mem_limit_mb}"
    shards = feat.shuffle_shards(paths, str(out), 2000, 123, mem_limit_mb)
    assert shards == sorted(shards)
    assert [os.path.basename(s) for s in shards] == [f"shard_{i:05d}.psv" for i in range(len(shards))]
    got = [read_records(s) for s in shards]
    assert all(len(g) == 2000 for g in got[:-1]) and 0 < len(got[-1]) <= 2000
    flat = [r for g in got for r in g]
    assert Counter(flat) == Counter(recs), "局面の集合が元データと一致しない"
    # 混ざっているか：最初のシャードに、最初の入力ファイルの局面ばかりが入っていないこと
    first_ids = [rec_id(r) for r in got[0]]
    assert sum(i < 3000 for i in first_ids) / len(first_ids) < 0.5
    # 同じ seed なら同じ結果
    shards2 = feat.shuffle_shards(paths, str(tmp_path / f"again_{mem_limit_mb}"), 2000, 123, mem_limit_mb)
    assert [read_records(s) for s in shards2] == got


# ---------- ローダー ----------

@pytest.mark.parametrize("version", [0, 1])
@pytest.mark.parametrize("threads", [1, 4])
def test_loader_in_order(feat, dataset, version, threads):
    paths, recs = dataset
    loader = feat.BatchLoader(paths, BATCH, version, shuffle_buffer=0, num_threads=threads)
    samples = collect(loader)
    assert len(samples) == len(recs)
    for s, r in zip(samples, recs):
        assert s == expected_sample(feat, r, version)


def test_loader_batch_sizes(feat, dataset):
    paths, recs = dataset
    sizes = [len(b["score"]) for b in feat.BatchLoader(paths, BATCH, 0)]
    assert all(s == BATCH for s in sizes[:-1]) and sum(sizes) == len(recs)


def test_loader_shuffle(feat, dataset):
    paths, recs = dataset
    def run(seed, threads):
        return collect(feat.BatchLoader(paths, BATCH, 0, shuffle_buffer=2000, seed=seed, num_threads=threads))
    a = run(5, 1)
    assert sorted(map(repr, a)) == sorted(repr(expected_sample(feat, r, 0)) for r in recs), "1周で全局面がちょうど1回ずつ出ていない"
    assert a != [expected_sample(feat, r, 0) for r in recs], "シャッフルされていない"
    assert run(5, 4) == a, "num_threads で並びが変わった"
    assert run(6, 1) != a, "seed を変えても並びが変わらない"


def test_loader_resume_exact(feat, dataset):
    paths, recs = dataset
    full = collect(feat.BatchLoader(paths, BATCH, 0, num_threads=4))
    l1 = feat.BatchLoader(paths, BATCH, 0, num_threads=4)
    head = collect(l1, limit=7)
    st = json.loads(json.dumps(l1.state()))
    tail = collect(feat.BatchLoader(paths, BATCH, 0, num_threads=4, resume_state=st))
    assert head + tail == full


def test_loader_resume_shuffle_no_duplicates(feat, dataset):
    paths, recs = dataset
    kw = dict(shuffle_buffer=2000, seed=9, num_threads=4)
    l1 = feat.BatchLoader(paths, BATCH, 0, **kw)
    head = collect(l1, limit=5)
    st = json.loads(json.dumps(l1.state()))
    tail = collect(feat.BatchLoader(paths, BATCH, 0, resume_state=st, **kw))
    keys = [repr(s) for s in head + tail]
    assert len(keys) == len(set(keys)), "再開後に同じ局面が重複して出た"
    assert len(keys) >= len(recs) - 2000 - 8 * BATCH, "再開で失った局面が多すぎる"


def test_loader_cache_dir(feat, dataset, tmp_path):
    paths, recs = dataset
    remote = tmp_path / "remote"
    remote.mkdir()
    rpaths = []
    for p in paths:  # 小さなファイルに分けて、キャッシュの出入りを多くする
        rs = read_records(p)
        for k in range(0, len(rs), 1000):
            q = str(remote / f"{os.path.basename(p)}.{k:05d}.psv")
            write_records(q, rs[k:k + 1000])
            rpaths.append(q)
    rpaths.sort()
    cache = tmp_path / "cache"
    cache.mkdir()
    expect = collect(feat.BatchLoader(rpaths, BATCH, 0, num_threads=2))
    got, peak = [], 0
    for b in feat.BatchLoader(rpaths, BATCH, 0, num_threads=2, cache_dir=str(cache), max_cached=2):
        got += unpack_batch(b)
        peak = max(peak, len([f for f in os.listdir(cache) if f.endswith(".psv")]))
    assert got == expect, "cache_dir を使うと結果が変わった"
    assert peak <= 3, f"cache_dir に .psv が {peak} 個あった（上限 max_cached+1 = 3）"
    assert not [f for f in os.listdir(cache) if f.endswith(".psv")], "読み終えたファイルが残っている"
    assert all(os.path.exists(p) for p in rpaths), "元のファイルが消えた"
