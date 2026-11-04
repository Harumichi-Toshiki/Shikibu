# Shikibu M9 の確かめ（新規。既存のテストは変えない）。
#   1. 開始前の確認（check_mode）・結果のキャッシュ・シャッフルの開始（shuffle_warm）：1 エポックで全局面が 1 回ずつ使われる
#   2. 第1層の疎な計算（ft_forward・ft_grad）と遅延型の Adam（ft_optimizer）：今の計算と一致する（許容誤差つき）・損失が同程度に下がる
#   3. fast（bf16）：CPU では無視される（結果が変わらない）
import json
import os
import struct

import numpy as np
import pytest
import torch

from m2_data import REC, make_dataset, read_records
from shikibu import networks, optim, train

ARCH = "halfkp_32x2-8-8_f"   # 因数分解つきの小さい構造（テスト用。W = 32、L1 = L2 = 8）
N_FILES, PER_FILE = 3, 2000  # 1 エポックに 6000 局面


def _sig_batch(b):
    """局面の鍵：得点・勝敗・玉の位置・手番側の特徴の並び。1 エポックの中で一意になることを、テストの前提にする。"""
    out = []
    so, si = b["stm_offsets"], b["stm_indices"]
    ends = list(so[1:]) + [len(si)]
    for i in range(len(b["score"])):
        feats = tuple(int(x) for x in si[so[i]:ends[i]])
        out.append((float(b["score"][i]), float(b["result"][i]), int(b["kings"][i, 0]), int(b["kings"][i, 1]), feats))
    return out


def _epoch_sigs(feat, paths, batch, **kw):
    loader = feat.BatchLoader(paths, batch, 2, num_threads=2, **kw)
    sigs = []
    for b in loader:
        sigs += _sig_batch(b)
    return sigs


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    d = tmp_path_factory.mktemp("m9data")
    paths, recs = make_dataset(str(d), n_files=N_FILES, per_file=PER_FILE, seed=91)
    return paths


# ---------- 1. 開始前の確認・シャッフル ----------

def test_loader_one_epoch_uses_every_record_once(feat, dataset, tmp_path):
    ref = sorted(_epoch_sigs(feat, dataset, 512))
    assert len(ref) == N_FILES * PER_FILE
    assert len(set(ref)) == len(ref)  # 比べる鍵が一意であること（得点・勝敗・玉の位置）
    for kw in ({"shuffle_buffer": 1500},
               {"shuffle_buffer": 1500, "shuffle_warm": 300},
               {"shuffle_buffer": 1500, "shuffle_warm": 1500},
               {"shuffle_buffer": 1500, "shuffle_warm": 300, "cache_dir": str(tmp_path / "c"), "max_cached": 2}):
        got = _epoch_sigs(feat, dataset, 512, **kw)
        assert len(got) == len(ref), kw
        assert sorted(got) == ref, kw


def test_loader_warm_start_changes_the_order_only(feat, dataset):
    a = _epoch_sigs(feat, dataset, 256, shuffle_buffer=1500, seed=3)
    b = _epoch_sigs(feat, dataset, 256, shuffle_buffer=1500, shuffle_warm=200, seed=3)
    assert sorted(a) == sorted(b)
    assert a != b  # 立ち上がりの件数が違えば、並びは変わる


def test_loader_rejects_warm_larger_than_buffer(feat, dataset):
    with pytest.raises(ValueError):
        feat.BatchLoader(dataset, 64, 2, shuffle_buffer=100, shuffle_warm=200)


def test_probe_covers_head_middle_tail_and_skips_bad_results(tmp_path):
    n = 3000
    recs = [bytearray(REC) for _ in range(n)]
    for i, r in enumerate(recs):
        struct.pack_into("<h", r, 32, i % 1000)  # score
        r[38] = 1 if i % 3 == 0 else 0           # 勝敗（1 か 0）
        struct.pack_into("<H", r, 34, i)         # 通し番号
    recs[1500][38] = 5                           # 範囲外の勝敗（飛ばす）
    p = tmp_path / "a.psv"
    p.write_bytes(b"".join(bytes(r) for r in recs))
    out = tmp_path / "probe.psv"
    files, count, skipped = train.probe_positions([str(p)], 600, str(out))  # 1 ブロック 200 件
    assert files == [str(out)]
    # 3 ブロックは先頭（0）・中央（(3000−200)//2 = 1400）・末尾（2800）。範囲外の 1 件は数えない
    ids = set()
    for st in (0, 1400, 2800):
        ids |= set(range(st, st + 200))
    ids.discard(1500)
    assert count == len(ids) and skipped == 1
    got = read_records(str(out))
    assert sorted(struct.unpack("<H", r[34:36])[0] for r in got) == sorted(ids)


def test_probe_max_files_reads_a_subset_only(tmp_path):
    paths = []
    for i in range(6):
        p = tmp_path / f"f{i}.psv"
        p.write_bytes(bytes(REC * 300))  # 全部 0 の局面（勝敗 0・得点 0）
        paths.append(str(p))
    _, count, skipped = train.probe_positions(paths, 300, str(tmp_path / "o.psv"), max_files=2, seed=4)
    assert count == 2 * 300 and skipped == 0
    _, count_all, _ = train.probe_positions(paths, 300, str(tmp_path / "o2.psv"))
    assert count_all == 6 * 300


def test_probe_rejects_truncated_file(tmp_path):
    p = tmp_path / "bad.psv"
    p.write_bytes(b"\0" * 41)
    with pytest.raises(RuntimeError):
        train.probe_positions([str(p)], 300, str(tmp_path / "x.psv"))


def test_check_cache_roundtrip_and_key_changes(dataset, tmp_path):
    d = {"check_mode": "sample", "probe_records": 3000, "probe_files": 0}
    t = {"scale_sample": 10_000_000, "seed": 0}
    k1 = train.check_key(dataset, d, t)
    assert k1 == train.check_key(dataset, d, t)                       # 同じデータ・設定なら同じ鍵
    assert k1 != train.check_key(dataset, {**d, "check_mode": "probe"}, t)
    assert k1 != train.check_key(dataset, d, {**t, "seed": 1})
    path = str(tmp_path / train.CHECK_CACHE_NAME)
    assert train.read_check_cache(path) == {}                          # 無いときは空
    assert train.write_check_cache(path, k1, {"count": 5, "result_missing": False, "eval_scale": 600.0})
    assert train.read_check_cache(path)[k1]["eval_scale"] == 600.0
    with open(path, "w") as f:
        f.write("{壊れた")
    assert train.read_check_cache(path) == {}                          # 壊れていても学習は止めない


def _write_cfg(path, data_dir, out_dir, data=None, train_=None, network=ARCH, steps=4):
    data_cfg = {"train_glob": f"{data_dir}/*.psv", "val_path": "", "cache_dir": "", "max_cached": 3,
                "shuffle_buffer": 0, "num_threads": 2, **(data or {})}
    train_cfg = {"batch_size": 64, "lr": 1e-3, "weight_decay": 1e-4, "warmup_steps": 0, "total_steps": steps,
                 "max_steps": steps, "min_lr_ratio": 0.0, "lam": 0.75, "eval_scale": "auto",
                 "scale_sample": 10_000_000, "allow_missing_results": False, "seed": 0, "device": "cpu",
                 "torch_threads": 1, "deterministic": False, "save_every_steps": 0, "save_every_minutes": 30.0,
                 "val_every_steps": 0, "log_every_steps": 1, "progress_every_sec": 1000.0, **(train_ or {})}
    lines = []
    for sec, body in (("data", data_cfg), ("model", {"network": network, "ft_width": 0}),
                      ("train", train_cfg), ("output", {"out_dir": str(out_dir)})):
        lines.append(f"[{sec}]")
        lines += [f"{k} = {json.dumps(v)}" for k, v in body.items()]
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return str(path)


def _log(out_dir):
    with open(os.path.join(out_dir, "log.jsonl"), encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def test_startup_cache_hit_gives_same_scale_and_is_logged(tmp_path):
    data_dir = str(tmp_path / "d")  # 確認の結果はデータのフォルダに書くので、このテスト専用のデータを使う
    make_dataset(data_dir, n_files=N_FILES, per_file=PER_FILE, seed=92)
    o1, o2 = tmp_path / "run1", tmp_path / "run2"
    os.makedirs(o1)
    os.makedirs(o2)
    c1 = _write_cfg(tmp_path / "c1.toml", data_dir, o1, data={"check_cache": True})
    c2 = _write_cfg(tmp_path / "c2.toml", data_dir, o2, data={"check_cache": True})
    assert train.main(["--config", c1]) == 0
    assert train.main(["--config", c2]) == 0
    st1 = next(r for r in _log(o1) if r["type"] == "startup")
    st2 = next(r for r in _log(o2) if r["type"] == "startup")
    assert st1["cache"] == "miss" and st2["cache"] == "hit"
    assert "data_check" in st1["seconds"]
    s1 = next(r for r in _log(o1) if r["type"] == "start")
    s2 = next(r for r in _log(o2) if r["type"] == "start")
    assert s1["eval_scale"] == s2["eval_scale"] and s2["eval_scale_source"] == "auto"
    assert os.path.exists(os.path.join(data_dir, train.CHECK_CACHE_NAME))


def test_startup_default_check_is_unchanged_when_cache_off(dataset, tmp_path):
    data_dir = os.path.dirname(dataset[0])
    assert not os.path.exists(os.path.join(data_dir, train.CHECK_CACHE_NAME))
    o = tmp_path / "run"
    os.makedirs(o)
    c = _write_cfg(tmp_path / "c.toml", data_dir, o)
    assert train.main(["--config", c]) == 0
    st = next(r for r in _log(o) if r["type"] == "startup")
    assert st["cache"] == "off" and st["check_mode"] == "sample" and st["shuffle_warm"] == 0
    assert not os.path.exists(os.path.join(data_dir, train.CHECK_CACHE_NAME))


@pytest.mark.parametrize("data, train_, msg", [
    ({"check_mode": "bogus"}, {}, "check_mode"),
    ({"shuffle_buffer": 100, "shuffle_warm": 200}, {}, "shuffle_warm"),
    ({}, {"ft_grad": "sparse"}, "lazy_adam"),
    ({}, {"ft_optimizer": "bogus"}, "ft_optimizer"),
    ({}, {"ft_forward": "bogus"}, "ft_forward"),
    ({}, {"fast": "yes"}, "fast"),
])
def test_invalid_m9_options_are_refused_before_training(dataset, tmp_path, data, train_, msg):
    data_dir = os.path.dirname(dataset[0])
    o = tmp_path / "run"
    c = _write_cfg(tmp_path / "c.toml", data_dir, o, data=data, train_=train_)
    cfg = train.load_config(c)
    with pytest.raises(ValueError, match=msg):
        train.validate(cfg)


def test_sparse_grad_on_factorized_needs_sparse_forward(dataset, tmp_path):
    data_dir = os.path.dirname(dataset[0])
    c = _write_cfg(tmp_path / "c.toml", data_dir, tmp_path / "o",
                   train_={"ft_grad": "sparse", "ft_optimizer": "lazy_adam", "ft_forward": "dense"})
    with pytest.raises(ValueError, match="ft_forward"):
        train.validate(train.load_config(c))


# ---------- 2. 第1層 ----------

def _batch(n_pos=48, per=30, seed=5, nf=125388, plane=1548):
    g = torch.Generator().manual_seed(seed)
    def side():
        counts = torch.randint(1, per, (n_pos,), generator=g)
        offs = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)[:-1]])
        idx = torch.randint(0, nf, (int(counts.sum()),), generator=g)
        return idx, offs
    si, so = side()
    ni, no = side()
    kings = torch.randint(0, 81, (n_pos, 2), generator=g)
    return si, so, ni, no, kings


def _model(seed=0):
    torch.manual_seed(seed)
    return networks.build(ARCH)


def test_sparse_forward_matches_dense_factorized_table():
    si, so, ni, no, kings = _batch()
    dense = _model(0)
    sparse = _model(0)
    sparse.load_state_dict(dense.state_dict())
    sparse.FT_FORWARD_SPARSE = True
    with torch.no_grad():
        z1 = dense(si, so, ni, no, kings=kings, base=True)
        z2 = sparse(si, so, ni, no, kings=kings, base=True)
    assert torch.allclose(z1, z2, atol=1e-5, rtol=1e-5), float((z1 - z2).abs().max())


def test_sparse_grad_matches_dense_grad_after_densifying():
    si, so, ni, no, kings = _batch(seed=6)
    a = _model(1)
    b = _model(1)
    b.load_state_dict(a.state_dict())
    b.FT_FORWARD_SPARSE = True
    b.FT_GRAD_SPARSE = True
    a(si, so, ni, no, kings=kings, base=True).sum().backward()
    b(si, so, ni, no, kings=kings, base=True).sum().backward()
    g_dense = a.ft.weight.grad
    g_sparse = b.ft.weight.grad
    assert g_sparse.is_sparse
    assert torch.allclose(g_dense, g_sparse.to_dense(), atol=1e-5, rtol=1e-5)


def test_lazy_adam_first_step_equals_adamw_exactly():
    # 1 ステップ目は、使われなかった行の m が 0 なので、密な AdamW も動かさない。よって全体が一致する。
    si, so, ni, no, kings = _batch(seed=7)
    score = torch.randn(48) * 300
    result = torch.randint(0, 2, (48,)).float()
    a = _model(2)
    b = _model(2)
    b.load_state_dict(a.state_dict())
    b.FT_FORWARD_SPARSE = True
    b.FT_GRAD_SPARSE = True
    oa = train.build_optimizer(a, 1e-3, 1e-4)
    ob = train.build_optimizer(b, 1e-3, 1e-4, ft_optimizer="lazy_adam")
    for m, o in ((a, oa), (b, ob)):
        o.zero_grad(set_to_none=True)
        z = m(si, so, ni, no, kings=kings, base=True)
        train.compute_loss(z, score, result, 600.0, 0.75).backward()
        o.step()
        train.clip_weights_(m, ft=not isinstance(o, optim.CombinedOptimizer))
    assert torch.allclose(a.ft.weight, b.ft.weight, atol=1e-7)
    assert torch.allclose(a.l1.weight, b.l1.weight, atol=1e-7)


def test_lazy_adam_moves_the_same_rows_as_dense_adamw_on_step_one():
    # 使われなかった行は動かない。動く行の集合は、密な AdamW（1 ステップ目）と一致する。
    # （勾配がちょうど 0 の行は、どちらでも動かない。だから「使った行は全部動く」とは言わない）
    si, so, ni, no, kings = _batch(seed=8)
    dense = _model(3)
    lazy = _model(3)
    lazy.load_state_dict(dense.state_dict())
    lazy.FT_FORWARD_SPARSE = True
    lazy.FT_GRAD_SPARSE = True
    od = train.build_optimizer(dense, 1e-2, 0.0)
    ol = train.build_optimizer(lazy, 1e-2, 0.0, ft_optimizer="lazy_adam")
    before = dense.ft.weight.detach().clone()
    used = torch.unique(torch.cat([si, ni, (si % 1548) + 125388, (ni % 1548) + 125388]))
    for m, o in ((dense, od), (lazy, ol)):
        o.zero_grad(set_to_none=True)
        z = m(si, so, ni, no, kings=kings, base=True)
        train.compute_loss(z, torch.zeros(48), torch.ones(48), 600.0, 0.75).backward()
        o.step()
    moved_dense = (dense.ft.weight.detach() != before).any(dim=1)
    moved_lazy = (lazy.ft.weight.detach() != before).any(dim=1)
    assert bool(moved_lazy.any())
    assert torch.equal(moved_lazy, moved_dense)
    untouched = torch.ones(lazy.ft.weight.shape[0], dtype=torch.bool)
    untouched[used] = False
    assert not bool(moved_lazy[untouched].any())


def test_lazy_adam_loss_drops_like_adamw_over_steps():
    # 目標は、同じ構造の「教師」の出力（学べる問題にする）。1 層目の重みを大きくして、教師の出力を広げる。
    si, so, ni, no, kings = _batch(seed=9)
    teacher = _model(99)
    with torch.no_grad():  # 各層を大きくして、教師の出力が 0 付近に潰れないようにする
        teacher.ft.weight.mul_(50.0)
        teacher.l1.weight.mul_(4.0)
        teacher.l2.weight.mul_(4.0)
        teacher.out.weight.mul_(8.0)
        target_z = teacher(si, so, ni, no, kings=kings, base=True)
    result = torch.zeros(48)
    finals = {}
    for name in ("adamw", "lazy"):
        m = _model(4)
        if name == "lazy":
            m.FT_FORWARD_SPARSE = True
            m.FT_GRAD_SPARSE = True
        o = train.build_optimizer(m, 1e-2, 1e-4, ft_optimizer="lazy_adam" if name == "lazy" else "adamw")
        losses = []
        for _ in range(200):
            o.zero_grad(set_to_none=True)
            z = m(si, so, ni, no, kings=kings, base=True)
            loss = train.compute_loss(z, target_z, result, 1.0, 1.0)  # t = sigmoid(教師の出力)
            loss.backward()
            o.step()
            train.clip_weights_(m, ft=name != "adamw")
            losses.append(float(loss))
        finals[name] = (losses[0], losses[-1])
        assert float(m.ft.weight.abs().max()) <= 1.0 + 1e-6
    for name, (first, last) in finals.items():
        assert last < 0.5 * first, (name, first, last)  # どちらも半分以下に下がる
    assert abs(finals["lazy"][1] - finals["adamw"][1]) < 0.1 * finals["adamw"][0]


def test_lazy_state_roundtrips_and_mismatch_is_refused():
    m = _model(5)
    o = train.build_optimizer(m, 1e-3, 1e-4, ft_optimizer="lazy_adam")
    sd = o.state_dict()
    assert optim.is_lazy_state(sd)
    o2 = train.build_optimizer(_model(5), 1e-3, 1e-4, ft_optimizer="lazy_adam")
    o2.load_state_dict(sd)
    plain_sd = train.build_optimizer(_model(5), 1e-3, 1e-4).state_dict()  # 通常の AdamW の形（kind なし）
    with pytest.raises(ValueError):
        o.load_state_dict(plain_sd)


def test_fast_flag_is_ignored_on_cpu(dataset, tmp_path):
    data_dir = os.path.dirname(dataset[0])
    outs = {}
    for fast in (False, True):
        o = tmp_path / f"f{int(fast)}"
        os.makedirs(o)
        c = _write_cfg(tmp_path / f"c{int(fast)}.toml", data_dir, o, train_={"fast": fast})
        assert train.main(["--config", c]) == 0
        outs[fast] = torch.load(os.path.join(o, "checkpoint.pt"), map_location="cpu", weights_only=False)["model"]
    for k in outs[False]:
        assert torch.equal(outs[False][k], outs[True][k]), k


def test_sparse_training_run_end_to_end(dataset, tmp_path):
    data_dir = os.path.dirname(dataset[0])
    o = tmp_path / "sparse"
    os.makedirs(o)
    c = _write_cfg(tmp_path / "s.toml", data_dir, o, data={"shuffle_buffer": 1000, "shuffle_warm": 200},
                   train_={"ft_forward": "sparse", "ft_grad": "sparse", "ft_optimizer": "lazy_adam"}, steps=6)
    assert train.main(["--config", c]) == 0
    log = _log(o)
    assert [r["type"] for r in log if r["type"] == "end"] == ["end"]
    st = next(r for r in log if r["type"] == "startup")
    assert st["shuffle_warm"] == 200 and st["ft_optimizer"] == "lazy_adam"
    assert "buffer_fill" not in [r.get("name") for r in log if r.get("type") == "phase"]
    losses = [r["loss"] for r in log if r.get("type") == "train"]
    assert len(losses) == 6 and all(np.isfinite(losses))
