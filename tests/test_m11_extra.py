# M11 の追加の確認（変更禁止ではない）。受け入れテスト test_m11_acceptance.py が覆わない点を確かめる。
#   - PSV の指し手の 16 ビットを、全部の符号について cshogi（参照）と比べる（変換できる符号の範囲）
#   - 最大時間（max_minutes）で正常に止まる、fp16 を CPU で断る、BatchLoader の output の検査
#   - 重み減衰の対象（BN・LN・相対位置バイアス・ゲート）、シフトの Module と関数の一致、HL-Gauss の正規化
import json
import os
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
START_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"


@pytest.fixture(scope="module")
def feat():
    import importlib
    return importlib.import_module(MODULE_NAME)


def _rec(sfen, mv16):
    """PSV 1 レコード（40 バイト）：局面・評価値 0・指し手・手数 1・勝敗 0。"""
    b = cshogi.Board()
    b.set_sfen(sfen)
    arr = np.zeros(1, cshogi.PackedSfenValue)
    b.to_psfen(arr["sfen"][0])
    arr["move"][0] = mv16
    arr["gamePly"][0] = 1
    return arr[0].tobytes(), b


def _psv_move_domain(v):
    """cshogi（参照）が正しく扱える符号か：移動先が盤内で、移動元が盤内か、打つ駒の種類が 1〜7。"""
    if v & 0x7f >= 81:
        return False
    if (v >> 14) & 1:
        return 1 <= (v >> 7) & 0x7f <= 7
    return ((v >> 7) & 0x7f) < 81


@pytest.mark.parametrize("sfen", [
    START_SFEN,
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2",   # 後手番（座標の回転）
])
def test_psv_move_codes_match_reference_exhaustively(feat, sfen):
    """全部の 16 ビットの符号について、Rust の指し手の番号が参照（cshogi 経由）と一致する。"""
    rec0, board = _rec(sfen, 0)
    rec = bytearray(rec0)
    checked = 0
    for v in range(1 << 16):
        if not _psv_move_domain(v):
            continue
        rec[34:36] = v.to_bytes(2, "little")   # PSV の指し手は 34・35 バイト目（リトルエンディアン）
        _, _, _, label = feat.dl_record(bytes(rec))
        ref = R.label_from_psv_move(board, v)
        assert label == (-1 if ref is None else ref), (hex(v), sfen)
        checked += 1
    assert checked > 10000


def test_psv_move_outside_domain_is_minus_one(feat):
    """cshogi が扱えない符号（移動先が盤外、打つ駒の種類が不正など）は −1（飛ばす対象）。"""
    rec, _ = _rec(START_SFEN, 81)              # 移動先 81 は盤外
    assert feat.dl_record(rec)[3] == -1
    rec, _ = _rec(START_SFEN, (1 << 14) | (0 << 7) | 40)   # 打つ駒の種類 0
    assert feat.dl_record(rec)[3] == -1
    rec, _ = _rec(START_SFEN, (1 << 14) | (8 << 7) | 40)   # 打つ駒の種類 8（成駒は打てない）
    assert feat.dl_record(rec)[3] == -1


def test_dl_record_bad_length_raises(feat):
    with pytest.raises(ValueError):
        feat.dl_record(bytes(39))


def test_loader_rejects_unknown_output(feat, tmp_path):
    p = tmp_path / "x.psv"
    p.write_bytes(bytes(40))
    with pytest.raises(ValueError):
        feat.BatchLoader([str(p)], 1, 0, output="nope")


def test_shift9_module_matches_function():
    from shikibu import dl_networks as N
    x = torch.randn(2, 27, 9, 9)
    assert torch.equal(N.Shift9(27)(x), N.shift9(x))
    m = N.Shift9(27)
    assert len(list(m.parameters())) == 0 and len(list(m.buffers())) == 1


def test_hl_gauss_normalized_and_centered():
    from shikibu import dl_features as F
    t = torch.linspace(0, 1, 37)
    g = F.hl_gauss(t)
    assert torch.allclose(g.sum(1), torch.ones(37), atol=1e-6)
    # 平均が t に近い（0.5 のときは対称）
    centers = (torch.arange(128) + 0.5) / 128
    mean = (g * centers).sum(1)
    assert float((mean - t).abs().max()) < 0.01
    assert torch.allclose(F.hl_gauss(torch.tensor([0.5]))[0], F.hl_gauss(torch.tensor([0.5]))[0].flip(0), atol=1e-6)


def test_no_decay_groups():
    from shikibu import dl_networks as N
    net = N.build_dl("dl_64_SFTH_HL_H")
    groups = N.param_groups(net, 1e-3)
    decay_ids = {id(p) for p in groups[0]["params"]}
    nodecay_ids = {id(p) for p in groups[1]["params"]}
    assert not decay_ids & nodecay_ids
    assert len(decay_ids | nodecay_ids) == len([p for p in net.parameters() if p.requires_grad])
    for name, p in net.named_parameters():
        if name.endswith("relpos") or name.endswith("bias") or ".bn" in name or ".norm" in name:
            assert id(p) in nodecay_ids, name
        if name.endswith("weight") and ("conv" in name or "fc" in name or "out.weight" in name or "qkv" in name):
            assert id(p) in decay_ids, name
    assert groups[0]["weight_decay"] == 1e-3 and groups[1]["weight_decay"] == 0.0


def test_unknown_optimizer_name_is_rejected():
    from shikibu import train_dl as T
    with pytest.raises(ValueError):
        T.make_optimizer("sgd", torch.nn.Linear(2, 2), 0.1, 0.0)


def _write_cfg(path, glob_pat, out, amp="off", max_minutes=0.02):
    path.write_text(f"""
[data]
train_glob = "{glob_pat}"
cache_dir = ""
shuffle_buffer = 0
num_threads = 1

[model]
network = "dl_32_S_0_0"

[train]
batch_size = 256
lr = 1e-3
total_steps = 100000
max_steps = 100000
eval_scale = 600
compile = false
channels_last = false
amp = "{amp}"
device = "cpu"
max_minutes = {max_minutes}
save_every_minutes = 30.0
log_every_steps = 1
progress_every_sec = 60.0

[output]
out_dir = "{out}"
""", encoding="utf-8")


def _psv_file(path, n, seed=0):
    import random
    rng = random.Random(seed)
    arr = np.zeros(n, cshogi.PackedSfenValue)
    i = 0
    while i < n:
        b = cshogi.Board()
        for _ in range(rng.randint(1, 60)):
            if b.is_game_over() or i >= n:
                break
            m = rng.choice(list(b.legal_moves))
            b.to_psfen(arr["sfen"][i])
            arr["move"][i] = cshogi.move16_to_psv(cshogi.move16(m))
            arr["score"][i] = rng.randint(-2000, 2000)
            arr["gamePly"][i] = b.move_number
            arr["game_result"][i] = rng.choice([-1, 0, 1])
            i += 1
            b.push(m)
    arr.tofile(path)


def test_max_minutes_stops_cleanly(tmp_path):
    """max_minutes を過ぎたら、保存して終了コード 0 で終わる（Colab の 1 回 8 時間の運用）。"""
    d = tmp_path / "tr"
    d.mkdir()
    _psv_file(d / "t.psv", 3000)
    out = tmp_path / "o"
    cfg = tmp_path / "c.toml"
    _write_cfg(cfg, f"{d}/*.psv", out)
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "python"))
    r = subprocess.run([sys.executable, "-m", "shikibu.train_dl", "--config", str(cfg)], env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    ck = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert 0 < ck["step"] < 100000
    rows = [json.loads(s) for s in open(out / "log.jsonl", encoding="utf-8") if s.strip()]
    end = [x for x in rows if x.get("kind") == "end"][-1]
    assert end["stopped_by"] == "max_minutes"
    assert all(x["type"] == x["kind"] for x in rows if "kind" in x)


def test_fp16_on_cpu_is_refused(tmp_path):
    d = tmp_path / "tr"
    d.mkdir()
    _psv_file(d / "t.psv", 600)
    out = tmp_path / "o"
    cfg = tmp_path / "c.toml"
    _write_cfg(cfg, f"{d}/*.psv", out, amp="fp16")
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "python"))
    r = subprocess.run([sys.executable, "-m", "shikibu.train_dl", "--config", str(cfg)], env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode != 0 and not (out / "checkpoint.pt").exists()


def test_attention_matches_explicit_formula():
    """注意（SDPA）の結果が、softmax(q·kᵀ/√hd + bias)·v の式と一致する。"""
    from shikibu import dl_networks as N
    import math
    torch.manual_seed(0)
    att = N._Attention(64, 16)
    x = torch.randn(2, 81, 64)
    got = att(x)
    h = att.norm(x)
    q, k, v = att.qkv(h).view(2, 81, 3, 4, 16).permute(2, 0, 3, 1, 4)
    bias = att.relpos[:, att.rel_index]
    w = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(16) + bias, dim=-1)
    o = (w @ v).transpose(1, 2).reshape(2, 81, 64)
    want = att.out(o * torch.sigmoid(att.gate(h)))
    assert torch.allclose(got, want, atol=1e-5)
