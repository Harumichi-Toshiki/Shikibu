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
    no_decay_names = ("file_emb", "rank_emb", "gate_attn", "gate_mlp", "relpos_bias", "norm1", "norm2", "bias")
    for name, p in net.named_parameters():
        if any(k in name for k in no_decay_names) or ".bn" in name or "bn" in name.split(".")[-2:][0]:
            assert id(p) in nodecay_ids, name
        if name.endswith("weight") and any(k in name for k in ("conv", "fc", "qkv", "proj", "ffn", "up", "down", "policy", "value_")):
            if "bn" not in name and "norm" not in name:
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




# ---------- F・T は Dainagon と同じ計算（M11.0.1） ----------

def _dainagon_relpos_index():
    """Dainagon の _build_relpos_index と同じ（index[i, j] = (yi − yj + 8) × 17 + (xi − xj + 8)）。"""
    idx = torch.zeros(81, 81, dtype=torch.long)
    for i in range(81):
        yi, xi = divmod(i, 9)
        for j in range(81):
            yj, xj = divmod(j, 9)
            idx[i, j] = (yi - yj + 8) * 17 + (xi - xj + 8)
    return idx


class _DainagonF(torch.nn.Module):
    """Dainagon の network_chunagon_v35.py の GatedTransformerBlockRelPos を、そのまま写したもの（テストの参照）。"""

    def __init__(self, channels, num_heads, mlp_ratio=4):
        super().__init__()
        nn = torch.nn
        self.channels = channels
        self.num_heads = num_heads
        self.file_emb = nn.Parameter(torch.randn(1, channels, 1, 9) * 0.02)
        self.rank_emb = nn.Parameter(torch.randn(1, channels, 9, 1) * 0.02)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels)
        mlp_hidden = channels * mlp_ratio
        self.ffn = nn.Sequential(nn.Linear(channels, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, channels))
        self.gate_attn = nn.Parameter(torch.tensor(0.1))
        self.gate_mlp = nn.Parameter(torch.tensor(0.1))
        self.relpos_bias = nn.Parameter(torch.zeros(num_heads, 17 * 17))
        self.register_buffer("relpos_index", _dainagon_relpos_index())

    def forward(self, x):
        B, C, H, W = x.shape
        x_pos = x + self.file_emb + self.rank_emb
        tokens = x_pos.flatten(2).transpose(1, 2)
        normed = self.norm1(tokens)
        qkv = self.qkv(normed).reshape(B, H * W, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = (q @ k.transpose(-2, -1)) * (C // self.num_heads) ** -0.5
        attn = attn + self.relpos_bias[:, self.relpos_index].unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn_out = self.proj((attn @ v).transpose(1, 2).reshape(B, H * W, C))
        tokens = tokens + self.gate_attn * attn_out
        tokens = tokens + self.gate_mlp * self.ffn(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(B, C, H, W)


class _RefGLUT(torch.nn.Module):
    """T の参照：同じ注意・位置の埋め込み・ゲートで、FFN だけ GLU（Linear(C,2·ffn·C) を 2 つに分け、GELU(ゲート)×値、Linear(ffn·C,C)）。"""

    def __init__(self, channels, num_heads, ffn):
        super().__init__()
        self.base = _DainagonF(channels, num_heads)
        self.base.ffn = torch.nn.Module()
        self.base.ffn.up = torch.nn.Linear(channels, 2 * ffn * channels)
        self.base.ffn.down = torch.nn.Linear(ffn * channels, channels)
        self.ffn_mul = ffn

    def forward(self, x):
        b = self.base
        B, C, H, W = x.shape
        x_pos = x + b.file_emb + b.rank_emb
        tokens = x_pos.flatten(2).transpose(1, 2)
        normed = b.norm1(tokens)
        qkv = b.qkv(normed).reshape(B, H * W, 3, b.num_heads, C // b.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = (q @ k.transpose(-2, -1)) * (C // b.num_heads) ** -0.5
        attn = attn + b.relpos_bias[:, b.relpos_index].unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn_out = b.proj((attn @ v).transpose(1, 2).reshape(B, H * W, C))
        tokens = tokens + b.gate_attn * attn_out
        gate, value = b.ffn.up(b.norm2(tokens)).chunk(2, dim=-1)
        tokens = tokens + b.gate_mlp * b.ffn.down(torch.nn.functional.gelu(gate) * value)
        return tokens.transpose(1, 2).reshape(B, C, H, W)


def _copy_matching(src, dst):
    """同じ名前の重み（と buffer 以外の全部）を src から dst へ写す。dst の名前が全部そろうことを確かめる。"""
    sd = src.state_dict()
    own = dst.state_dict()
    missing = [k for k in own if k not in sd and k != "relpos_index"]
    assert not missing, missing
    dst.load_state_dict({k: sd[k] for k in own if k in sd}, strict=False)


@pytest.mark.parametrize("channels,heads", [(64, 2), (320, 10)])
def test_F_matches_dainagon(channels, heads):
    from shikibu import dl_networks as N
    torch.manual_seed(1)
    ref = _DainagonF(channels, heads)
    with torch.no_grad():
        ref.relpos_bias.normal_(0, 0.5)   # 0 初期化のままだと相対位置を確かめられないので、確かめるときだけ乱数にする
        ref.gate_attn.fill_(0.7)
        ref.gate_mlp.fill_(-0.4)
    ours = N.GatedTransformerBlockRelPos(channels, torch.nn.ReLU, {"hd": channels // heads, "ffn": 2})
    _copy_matching(ref, ours)
    ref.eval()
    ours.eval()
    x = torch.randn(3, channels, 9, 9)
    with torch.no_grad():
        want = ref(x)
        got = ours(x)
    assert float((got - want).abs().max()) < 1e-5


@pytest.mark.parametrize("channels,heads", [(64, 2), (320, 10)])
def test_T_matches_glu_reference(channels, heads):
    from shikibu import dl_networks as N
    torch.manual_seed(2)
    ref = _RefGLUT(channels, heads, ffn=2)
    with torch.no_grad():
        ref.base.relpos_bias.normal_(0, 0.5)
        ref.base.gate_attn.fill_(0.3)
        ref.base.gate_mlp.fill_(0.2)
    ours = N.LightTransformerBlock(channels, torch.nn.ReLU, {"hd": channels // heads, "ffn": 2})
    # ours の名前：file_emb・rank_emb・norm1・norm2・qkv・proj・ffn.up・ffn.down・gate_attn・gate_mlp・relpos_bias
    sd = ref.base.state_dict()
    sd.update({"ffn.up.weight": ref.base.ffn.up.weight, "ffn.up.bias": ref.base.ffn.up.bias,
               "ffn.down.weight": ref.base.ffn.down.weight, "ffn.down.bias": ref.base.ffn.down.bias})
    ours.load_state_dict({k: sd[k] for k in ours.state_dict() if k in sd}, strict=False)
    ref.eval()
    ours.eval()
    x = torch.randn(3, channels, 9, 9)
    with torch.no_grad():
        want = ref(x)
        got = ours(x)
    assert float((got - want).abs().max()) < 1e-5


def test_F_T_init_values():
    from shikibu import dl_networks as N
    f = N.GatedTransformerBlockRelPos(64, torch.nn.ReLU, {"hd": 32, "ffn": 2})
    t = N.LightTransformerBlock(64, torch.nn.ReLU, {"hd": 32, "ffn": 2})
    for blk in (f, t):
        assert float(blk.relpos_bias.detach().abs().max()) == 0.0
        assert abs(float(blk.gate_attn.detach()) - 0.1) < 1e-7 and abs(float(blk.gate_mlp.detach()) - 0.1) < 1e-7
        assert blk.qkv.bias is None
    assert tuple(f.file_emb.shape) == (1, 64, 1, 9) and tuple(f.rank_emb.shape) == (1, 64, 9, 1)


def test_F_T_macs_formula():
    from shikibu import dl_networks as N
    C, ffn = 96, 2
    assert N.GatedTransformerBlockRelPos.macs(C, {"hd": 32, "ffn": 2}) == 81 * (4 * C * C + 8 * C * C) + 2 * 81 * 81 * C
    assert N.LightTransformerBlock.macs(C, {"hd": 32, "ffn": ffn}) == 81 * (4 * C * C + 3 * ffn * C * C) + 2 * 81 * 81 * C
