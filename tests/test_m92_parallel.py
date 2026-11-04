# Shikibu M9.2 の確かめ（新規）：データの確認の読みを並列にしても、結果は同じで、遅い読みでは速くなる。
import os
import struct
import time

import pytest

from shikibu import train

REC = 40


def _write(path, n, seed, bad_every=0):
    """n 局面の PSV を書く。move 欄に通し番号（ファイルごとに違う）。bad_every ごとに勝敗を範囲外にする。"""
    with open(path, "wb") as f:
        for i in range(n):
            r = bytearray(REC)
            struct.pack_into("<h", r, 32, (i * 7 + seed) % 3000 - 1500)
            r[38] = ((i + seed) % 3 - 1) & 0xFF  # 勝敗は -1・0・1（1 バイトの 2 の補数）
            struct.pack_into("<H", r, 34, (i + seed * 97) % 65536)
            if bad_every and i % bad_every == 0 and i:
                r[38] = 5
            f.write(bytes(r))
    return str(path)


@pytest.fixture
def files(tmp_path):
    paths = []
    for k in range(12):
        n = 200 + 37 * k
        paths.append(_write(tmp_path / f"f{k:02d}.psv", n, seed=k, bad_every=53))
    return paths


def test_parallel_sample_is_the_same_as_serial(files, tmp_path):
    out1, out16 = str(tmp_path / "s1.psv"), str(tmp_path / "s16.psv")
    f1, c1 = train.sample_scale_positions(files, 1500, 3, out1, threads=1)
    f16, c16 = train.sample_scale_positions(files, 1500, 3, out16, threads=16)
    assert c1 == c16 > 0 and f1 == [out1] and f16 == [out16]
    assert open(out1, "rb").read() == open(out16, "rb").read()


def test_parallel_probe_is_the_same_as_serial(files, tmp_path):
    out1, out16 = str(tmp_path / "p1.psv"), str(tmp_path / "p16.psv")
    _, c1, s1 = train.probe_positions(files, 90, out1, threads=1)
    _, c16, s16 = train.probe_positions(files, 90, out16, threads=16)
    assert (c1, s1) == (c16, s16)
    assert s1 > 0  # 範囲外の勝敗は飛ばされている
    assert open(out1, "rb").read() == open(out16, "rb").read()


def test_parallel_probe_with_file_subset_is_the_same(files, tmp_path):
    out1, out4 = str(tmp_path / "q1.psv"), str(tmp_path / "q4.psv")
    a = train.probe_positions(files, 90, out1, max_files=5, seed=9, threads=1)
    b = train.probe_positions(files, 90, out4, max_files=5, seed=9, threads=4)
    assert a[1:] == b[1:]
    assert open(out1, "rb").read() == open(out4, "rb").read()


def test_parallel_read_is_faster_when_each_read_waits(files, tmp_path, monkeypatch):
    # 読みのたびに 20 ms 待つ（Drive の遅さを模す）。並列 16 本のほうが、直列よりずっと短い。
    orig = train.read_file_blocks

    def slow(path, spans):
        time.sleep(0.02)
        return orig(path, spans)

    monkeypatch.setattr(train, "read_file_blocks", slow)
    many = files * 3  # 36 ファイル（重複してよい。読む順番の検査は上のテストで行った）
    t = time.perf_counter()
    train.sample_scale_positions(many, 3000, 1, str(tmp_path / "a.psv"), threads=1)
    t_serial = time.perf_counter() - t
    t = time.perf_counter()
    train.sample_scale_positions(many, 3000, 1, str(tmp_path / "b.psv"), threads=16)
    t_par = time.perf_counter() - t
    assert t_serial >= 0.5, t_serial          # 直列は待ちの合計（36 × 20 ms 以上）
    assert t_par < 0.5 * t_serial, (t_serial, t_par)
    assert open(tmp_path / "a.psv", "rb").read() == open(tmp_path / "b.psv", "rb").read()


def test_check_threads_option_is_validated(tmp_path):
    cfg_path = tmp_path / "c.toml"
    cfg_path.write_text(
        '[data]\ntrain_glob = "/x/*.psv"\ncheck_threads = 0\n'
        '[model]\nnetwork = "halfkp_32x2-8-8_f"\n'
        '[train]\ntotal_steps = 10\nmax_steps = 10\n'
        f'[output]\nout_dir = "{tmp_path}/o"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="check_threads"):
        train.validate(train.load_config(str(cfg_path)))


def test_unwritable_cache_warns_and_training_continues(tmp_path):
    # キャッシュのファイル名にディレクトリを置き、書けない状態を作る（root では権限で再現できないため）。
    import json
    from m2_data import make_dataset
    data_dir = str(tmp_path / "d")
    make_dataset(data_dir, n_files=2, per_file=600, seed=21)
    os.makedirs(os.path.join(data_dir, train.CHECK_CACHE_NAME))
    out = tmp_path / "o"
    os.makedirs(out)
    cfg = tmp_path / "c.toml"
    cfg.write_text(
        f'[data]\ntrain_glob = "{data_dir}/*.psv"\ncheck_cache = true\nnum_threads = 2\n'
        '[model]\nnetwork = "halfkp_32x2-8-8_f"\n'
        '[train]\nbatch_size = 64\ntotal_steps = 2\nmax_steps = 2\nlam = 0.75\neval_scale = "auto"\n'
        'seed = 0\ndevice = "cpu"\nprogress_every_sec = 1000.0\nlog_every_steps = 1\n'
        f'[output]\nout_dir = "{out}"\n', encoding="utf-8")
    assert train.main(["--config", str(cfg)]) == 0
    log = [json.loads(x) for x in open(out / "log.jsonl", encoding="utf-8") if x.strip()]
    st = next(r for r in log if r["type"] == "startup")
    assert st["cache"] == "miss" and st["cache_saved"] is False
    assert st["check_threads"] == 16
