# M11.1 の確認（新しいファイル）：hcpe の読み込み（Rust）、指し手の無い局面を残す・飛ばす、件数、
# 使えないレコードが続いたときに止まる、組の混ぜ方と損失、組の設定の検査、shift9 の一致。
import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest

cshogi = pytest.importorskip("cshogi")
torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
sys.path.insert(0, os.path.join(ROOT, "python"))

import dainagon_ref as R  # noqa: E402

MODULE_NAME = os.environ.get("SHIKIBU_FEAT_MODULE", "shikibu_feat")


@pytest.fixture(scope="module")
def feat():
    import importlib
    return importlib.import_module(MODULE_NAME)


def _hcpe_rows(n, seed, with_moves=True, result_range=(0, 2)):
    """乱数で指し進めた局面の hcpe（レコード 38 バイトの列）と、その指し手（cshogi の move）。"""
    rng = random.Random(seed)
    arr = np.zeros(n, cshogi.HuffmanCodedPosAndEval)
    moves = []
    i = 0
    while i < n:
        b = cshogi.Board()
        for _ in range(rng.randint(1, 200)):
            if b.is_game_over() or i >= n:
                break
            m = rng.choice(list(b.legal_moves))
            b.to_hcp(arr["hcp"][i])
            arr["bestMove16"][i] = cshogi.move16(m) if with_moves else 0
            arr["eval"][i] = rng.randint(-3000, 3000)
            arr["gameResult"][i] = rng.randint(*result_range)
            moves.append(m)
            i += 1
            b.push(m)
    return arr, moves


def _write(path, arr):
    with open(path, "wb") as f:
        f.write(arr.tobytes())


def test_hcpe_record_matches_python_reference_on_thousands(feat):
    """Rust の hcpe の 1 レコード（盤・持ち駒・王手・指し手・評価値・勝敗）が、dl_features.from_hcpe と一致する。"""
    from shikibu import dl_features as F
    arr, _ = _hcpe_rows(3000, 21)
    ref = F.from_hcpe(arr)
    out = [feat.dl_record_hcpe(arr[i].tobytes()) for i in range(len(arr))]
    move = np.array([o[3] for o in out])
    keep = move >= 0
    assert int(keep.sum()) == len(ref["move"])
    board = np.stack([np.frombuffer(o[0], dtype=np.uint8) for o in out])[keep]
    hands = np.stack([np.frombuffer(o[1], dtype=np.uint8) for o in out])[keep]
    check = np.array([o[2] for o in out], dtype=np.uint8)[keep]
    assert np.array_equal(move[keep], ref["move"])
    assert np.array_equal(board, ref["board"])
    assert np.array_equal(hands, ref["hands"])
    assert np.array_equal(check, ref["check"])
    assert np.array_equal(np.array([o[4] for o in out], dtype=np.float32)[keep], ref["score"])
    assert np.array_equal(np.array([o[5] for o in out], dtype=np.float32)[keep], ref["result"])


def test_hcpe_move_labels_all_legal_moves(feat):
    """全部の合法手（成り・打つ・王手の局面を含む）について、指し手の番号が参照（cshogi の USI）と一致する。"""
    rng = random.Random(4)
    checked = promo = drop = 0
    for g in range(120):
        b = cshogi.Board()
        for _ in range(rng.randint(1, 120)):
            if b.is_game_over():
                break
            hcp = np.zeros(1, cshogi.HuffmanCodedPosAndEval)
            b.to_hcp(hcp["hcp"][0])
            for m in list(b.legal_moves)[:12]:
                hcp["bestMove16"][0] = cshogi.move16(m)
                rec = hcp[0].tobytes()
                got = feat.dl_record_hcpe(rec)[3]
                want = R.label_of_usi(b, cshogi.move_to_usi(m))
                assert got == (-1 if want is None else want), (cshogi.move_to_usi(m), b.sfen())
                checked += 1
                promo += cshogi.move_is_promotion(m)
                drop += cshogi.move_is_drop(m)
            b.push(rng.choice(list(b.legal_moves)))
    assert checked > 2000 and promo > 50 and drop > 50


def test_hcpe_loader_matches_from_hcpe_and_threads(feat, tmp_path):
    from shikibu import dl_features as F
    arr, _ = _hcpe_rows(5000, 33)
    p = str(tmp_path / "a.hcpe")
    _write(p, arr)
    ref = F.from_hcpe(arr)

    def collect(**kw):
        got = {k: [] for k in ("board", "hands", "check", "move", "score", "result")}
        ld = feat.BatchLoader([p], 777, 0, output="dl", fmt="hcpe", **kw)
        for b in ld:
            for k in got:
                got[k].append(b[k])
        return {k: np.concatenate(v) for k, v in got.items()}, ld

    one, ld1 = collect(num_threads=1)
    four, _ = collect(num_threads=4)
    for k in one:
        assert np.array_equal(one[k], four[k]), k
    for k in one:
        assert np.array_equal(one[k], ref[k]), k
    assert ld1.skipped() == 0
    snap = ld1.stats().snapshot()
    assert snap["processed"] == 5000 and snap["kept"] == len(ref["move"])


def test_hcpe_loader_resume_is_exact(feat, tmp_path):
    arr, _ = _hcpe_rows(4000, 34)
    p = str(tmp_path / "r.hcpe")
    _write(p, arr)
    full = [b["move"] for b in feat.BatchLoader([p], 300, 0, output="dl", fmt="hcpe")]
    ld = feat.BatchLoader([p], 300, 0, output="dl", fmt="hcpe")
    first = [next(ld)["move"] for _ in range(4)]
    st = ld.state()
    rest = [b["move"] for b in feat.BatchLoader([p], 300, 0, output="dl", fmt="hcpe", resume_state=st)]
    assert np.array_equal(np.concatenate(first + rest), np.concatenate(full))


def test_require_move_false_keeps_positions_without_moves(feat, tmp_path):
    arr, _ = _hcpe_rows(1000, 35, with_moves=False)
    p = str(tmp_path / "nomove.hcpe")
    _write(p, arr)
    # 既定（require_move=True）は、使える局面が 1 つも無いまま終わる（この短さでは 100 万件の上限には届かない）
    ld = feat.BatchLoader([p], 100, 0, output="dl", fmt="hcpe")
    with pytest.raises(StopIteration):
        next(ld)
    assert ld.skip_reasons()["no_move"] == 1000
    # require_move=False なら、指し手 −1 のまま残る（飛ばさない）
    ld = feat.BatchLoader([p], 100, 0, output="dl", fmt="hcpe", require_move=False)
    got = [b["move"] for b in ld]
    allm = np.concatenate(got)
    assert len(allm) == 1000 and np.all(allm == -1)
    assert ld.skipped() == 0


def test_skip_reasons_are_counted(feat, tmp_path):
    arr, _ = _hcpe_rows(600, 36)
    arr = arr.copy()
    arr["gameResult"][3] = 7           # 勝敗の範囲外
    arr["bestMove16"][5] = 0           # 指し手なし
    p = str(tmp_path / "mix.hcpe")
    _write(p, arr)
    ld = feat.BatchLoader([p], 50, 0, output="dl", fmt="hcpe")
    n = sum(len(b["move"]) for b in ld)
    reasons = ld.skip_reasons()
    assert n == 598
    assert reasons == {"unreadable": 0, "result_range": 1, "no_move": 1}
    assert ld.skipped() == 2


def test_dry_run_stops_with_reasons(feat, tmp_path):
    """使えるレコードが 100 万件続けて無ければ、全部捨てて待つのではなく、理由つきのエラーで止まる。"""
    arr = np.zeros(1_100_000, cshogi.HuffmanCodedPosAndEval)
    arr["bestMove16"] = 0
    p = str(tmp_path / "dry.hcpe")
    _write(p, arr)
    ld = feat.BatchLoader([p], 4096, 0, output="dl", fmt="hcpe")
    with pytest.raises(RuntimeError, match="指し手なし"):
        for _ in ld:
            pass


def test_psv_moves_zero_mode(feat, tmp_path):
    """PSV でも同じ。指し手 0 の局面は、require_move=False で −1 のまま残る。"""
    from test_m11_acceptance import _random_positions, _psv
    pos = _random_positions(200, 9)
    arr = _psv(pos, 10)
    arr["move"][:] = 0
    p = str(tmp_path / "nomove.psv")
    arr.tofile(p)
    got = np.concatenate([b["move"] for b in feat.BatchLoader([p], 64, 0, output="dl", require_move=False)])
    assert len(got) == 200 and np.all(got == -1)


def test_features_path_rejects_hcpe(feat, tmp_path):
    p = tmp_path / "x.hcpe"
    p.write_bytes(bytes(38))
    with pytest.raises(ValueError):
        feat.BatchLoader([str(p)], 1, 0, output="features", fmt="hcpe")


def test_shift9_equals_conv_version():
    from shikibu import dl_networks as N
    import torch.nn.functional as F
    torch.manual_seed(0)
    for C in (1, 8, 9, 18, 19, 192):
        x = torch.randn(3, C, 9, 9)
        assert torch.equal(N.shift9(x), F.conv2d(x, N._shift_kernel(C), padding=1, groups=C))


def test_split_counts_and_threads():
    from shikibu import train_dl as T
    assert T.split_counts([0.5, 0.5], 4096) == [2048, 2048]
    c = T.split_counts([1.0, 2.0, 0.5], 1000)
    assert sum(c) == 1000 and c[0] == 285
    with pytest.raises(ValueError):
        T.split_counts([1.0, 1000.0], 500)
    assert T.split_threads([1.0, 1.0], 11) == [6, 6] or sum(T.split_threads([1.0, 1.0], 11)) >= 2


def test_masked_terms_match_manual_and_zero_count_is_zero():
    from shikibu import train_dl as T
    torch.manual_seed(1)
    B = 6
    p = torch.randn(B, 2187)
    vl = torch.randn(B, 128)
    rl = torch.randn(B, 1)
    x = {
        "move": torch.tensor([3, -1, 7, 9, 2, 5]),
        "score": torch.tensor([100.0, -200.0, 0.0, 300.0, -50.0, 10.0]),
        "result": torch.tensor([1.0, 0.5, 0.0, 1.0, 0.0, 0.5]),
        "pol": torch.tensor([1, 0, 1, 1, 0, 0], dtype=torch.uint8),
        "val": torch.tensor([0, 1, 1, 1, 0, 0], dtype=torch.uint8),
        "res": torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.uint8),
        "scale": torch.full((B,), 600.0),
        "sid": torch.zeros(B, dtype=torch.uint8),
    }
    loss_p, loss_v, loss_r, per = T.masked_terms(p, vl, rl, x, 0.0)
    # 手で：方策は pol・move>=0 の 3 局面（0, 2, 3）の平均
    ce = torch.nn.functional.cross_entropy(p, x["move"].clamp(min=0), reduction="none")
    want_p = (ce[0] + ce[2] + ce[3]) / 3
    assert torch.allclose(loss_p, want_p, atol=1e-5)
    # 価値は val の 1,2,3 の平均
    from shikibu import dl_features as F
    hl = F.hl_gauss(torch.sigmoid(x["score"] / 600.0))
    cev = -(hl * torch.log_softmax(vl, dim=1)).sum(1)
    assert torch.allclose(loss_v, (cev[1] + cev[2] + cev[3]) / 3, atol=1e-5)
    # 勝敗は res の 0,1
    bce = torch.nn.functional.binary_cross_entropy_with_logits(rl.view(-1), x["result"], reduction="none")
    assert torch.allclose(loss_r, (bce[0] + bce[1]) / 2, atol=1e-5)
    # 対象が 0 の項は 0（NaN にならない）
    x2 = dict(x, res=torch.zeros(B, dtype=torch.uint8), pol=torch.zeros(B, dtype=torch.uint8))
    lp, _, lr, _ = T.masked_terms(p, vl, rl, x2, 0.0)
    assert float(lp) == 0.0 and float(lr) == 0.0


def test_scale_estimate_recovers_known_scale():
    from shikibu import train_dl as T
    rng = np.random.default_rng(0)
    score = rng.integers(-2500, 2500, size=200000).astype(np.float64)
    p = 1 / (1 + np.exp(-score / 600.0))
    result = (rng.random(len(p)) < p).astype(np.float32)
    est = T.estimate_scale_from_samples(score.astype(np.float32), result)
    assert 0.85 * 600 <= est <= 1.15 * 600, est


# ---------- 学習（組の混ぜ方） ----------

def _mixed_config(path, out, sources_toml, max_steps, extra_data=""):
    path.write_text(f"""
[data]
cache_dir = ""
max_cached = 3
shuffle_buffer = 2000
num_threads = 2
{extra_data}
{sources_toml}

[model]
network = "dl_32_S_0_0"

[train]
batch_size = 64
lr = 1e-3
total_steps = 100
max_steps = {max_steps}
eval_scale = "auto"
scale_sample = 20000
compile = false
channels_last = false
amp = "off"
device = "cpu"
log_every_steps = 1
progress_every_sec = 60.0
save_every_minutes = 30.0

[output]
out_dir = "{out}"
""", encoding="utf-8")


def _env():
    return dict(os.environ, PYTHONPATH=os.path.join(ROOT, "python"))


def test_mixed_sources_train_and_resume(tmp_path):
    """方策だけの hcpe の組と、価値だけの PSV（指し手 0）の組を混ぜて学習し、再開できる。"""
    from test_m11_acceptance import _psv, _random_positions
    hd = tmp_path / "hcpe"
    hd.mkdir()
    arr, _ = _hcpe_rows(3000, 41)
    _write(hd / "a.hcpe", arr)
    pd = tmp_path / "psv"
    pd.mkdir()
    pos = _random_positions(3000, 42)
    parr = _psv(pos, 43)
    parr["move"][:] = 0
    parr.tofile(pd / "v.psv")
    out = tmp_path / "out"
    sources = f"""
[[data.sources]]
name = "moves_hcpe"
glob = "{hd}/*.hcpe"
format = "hcpe"
weight = 0.5
policy = true
value = false
result = true

[[data.sources]]
name = "big_psv"
glob = "{pd}/*.psv"
format = "psv"
weight = 0.5
policy = false
value = true
result = true
eval_scale = 600
"""
    cfg = tmp_path / "c.toml"
    _mixed_config(cfg, out, sources, 3)
    env = _env()
    r = subprocess.run([sys.executable, "-m", "shikibu.train_dl", "--config", str(cfg)], env=env,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr
    rows = [json.loads(s) for s in open(out / "log.jsonl", encoding="utf-8") if s.strip()]
    tr = [x for x in rows if x.get("kind") == "train"]
    assert tr and set(tr[-1]["sources"]) == {"moves_hcpe", "big_psv"}
    assert tr[-1]["sources"]["moves_hcpe"]["n"] + tr[-1]["sources"]["big_psv"]["n"] == 64
    assert all(np.isfinite(x["loss"]) for x in tr)
    # 値は、値の組（big_psv）の局面でだけ出る：hcpe 側の loss_v は 0
    assert tr[-1]["sources"]["moves_hcpe"]["loss_v"] == 0.0
    start = [x for x in rows if x.get("kind") == "start"][-1]
    assert start["eval_scales"]["moves_hcpe"] is None and start["eval_scales"]["big_psv"] == 600.0
    ck = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 3 and ck["source_names"] == ["moves_hcpe", "big_psv"]
    # 再開
    cfg.write_text(cfg.read_text(encoding="utf-8").replace("max_steps = 3", "max_steps = 5"), encoding="utf-8")
    r = subprocess.run([sys.executable, "-m", "shikibu.train_dl", "--config", str(cfg), "--resume"], env=env,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr
    ck = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 5


def test_policy_true_without_moves_stops_before_training(tmp_path):
    """指し手の無い PSV を policy = true にした設定は、学習を始める前に止まる（永遠に待たない）。"""
    from test_m11_acceptance import _psv, _random_positions
    pd = tmp_path / "psv"
    pd.mkdir()
    arr = _psv(_random_positions(500, 51), 52)
    arr["move"][:] = 0
    arr.tofile(pd / "nomove.psv")
    out = tmp_path / "out"
    sources = f"""
[[data.sources]]
name = "nomove"
glob = "{pd}/*.psv"
format = "psv"
weight = 1.0
policy = true
value = true
result = true
eval_scale = 600
"""
    cfg = tmp_path / "c.toml"
    _mixed_config(cfg, out, sources, 2)
    r = subprocess.run([sys.executable, "-m", "shikibu.train_dl", "--config", str(cfg)], env=_env(),
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 1 and "指し手のある局面が1つも無い" in r.stderr
    assert "Traceback" not in r.stderr and not (out / "checkpoint.pt").exists()


def test_source_glob_placeholder_is_refused(tmp_path):
    cfg = tmp_path / "c.toml"
    _mixed_config(cfg, tmp_path / "o", """
[[data.sources]]
name = "x"
glob = "/content/<学習データ>/*.hcpe"
format = "hcpe"
policy = true
""", 2)
    r = subprocess.run([sys.executable, "-m", "shikibu.train_dl", "--config", str(cfg)], env=_env(),
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 1 and "設定のひな形のまま" in r.stderr and "Traceback" not in r.stderr
    assert not (tmp_path / "o").exists()


def test_source_key_and_format_are_checked(tmp_path):
    from shikibu import train_dl as T
    with pytest.raises(ValueError):
        T.validate_sources([{"glob": "x", "format": "csv"}])
    with pytest.raises(ValueError):
        T.validate_sources([{"glob": "x", "wieght": 1.0}])
    with pytest.raises(ValueError):
        T.validate_sources([{"glob": "x", "policy": False, "value": False, "result": False}])
