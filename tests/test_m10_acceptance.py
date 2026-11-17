# 変更禁止（PROTECTED）。Shikibu M10（やねうら王の SFNN・レイヤースタック）の受け入れテスト。設計書 §S13。
#
# 目的：やねうら王の SFNN（レイヤースタックつき）と同じ構造を学習し、やねうら王がそのまま読める nn.bin に書き出す。
# 最初の対象は SFNN_halfkahm2_512_15_32_k3k3。正解は tests/sfnn_ref.py（やねうら王 2bc64b4 で確かめた参照実装）と
# tests/golden/sfnn_golden.json。
#
# ■ 構造の名前（model.network に書く）
#     sfnn_<特徴>_<W>-<H1>-<H2>[_f][_k3k3]
#   特徴：halfk…（halfkp・halfka1・halfka2・halfkahm1・halfkahm2）。W は 64 の倍数（64〜8192）。
#   H1 は 8n か 8n−1（7〜255）。8n−1 のときは「近道」あり（fc_0 の出力が H1+1 個）。H2 は 8 の倍数（8〜1024）。
#   オプション：f（因数分解、M7 と同じ）、k3k3（9 組のレイヤースタック。無ければ 1 組）。psqt は使えない（ValueError）。
#   networks.parse_arch は大文字・小文字・オプションの順を問わず受け付け、正規の形（すべて小文字、f → k3k3 の順）を
#   "canonical" で返す。ほかに "sfnn": True, "feature", "W", "H1", "H2", "factorize", "stacks"（1 か 9）,
#   "shortcut"（bool）, "version", "num_features" を返す。従来の構造の名前の parse_arch は "sfnn": False を返す。
#
# ■ モデル（networks.build(名前)）
#   属性：SFNN=True、FEATURE_VERSION、NUM_FEATURES、ft_width（W）、H1、H2、SHORTCUT、STACKS、NEEDS_KINGS（k3k3 なら True）。
#   export_tensors() / set_from_tensors(d)：float32 のテンソル。S = STACKS、O0 = H1 + (1 if SHORTCUT else 0)。
#     ft_weight (NUM_FEATURES, W)、ft_bias (W,)、
#     l0_weight (S, O0, W)、l0_bias (S, O0)、l1_weight (S, H2, 2×H1)、l1_bias (S, H2)、l2_weight (S, 1, H2)、l2_bias (S, 1)
#   forward(stm_indices, stm_offsets, nstm_indices, nstm_offsets, kings=None) の式（局面ごと）：
#     a_p = ft_bias + Σ ft_weight[特徴]（p は手番側・相手側。M7 と同じく、因数分解つきなら因数分解の特徴も含めた番号）
#     c_p = clamp(a_p, 0, 1)
#     x = [c_手番[:W/2] × c_手番[W/2:] × 127/128, c_相手[:W/2] × c_相手[W/2:] × 127/128]（W 個）
#     b = k3k3 なら M7 と同じバケツ（kingzone_ref.k3k3_bucket と同じ。kings が無ければ例外）、無ければ 0
#     y0 = l0_weight[b] · x + l0_bias[b]
#     h = [clamp(y0[:H1]² × 127/128, 0, 1), clamp(y0[:H1], 0, 1)]（2×H1 個）
#     a1 = clamp(l1_weight[b] · h + l1_bias[b], 0, 1)
#     z = l2_weight[b] · a1 + l2_bias[b] ＋（近道ありなら）y0[H1]
#   学習（損失・データ・eval_scale など）は従来の構造と同じ。train.clip_weights_ は ft_weight・ft_bias を [−1, 1]、
#   l0_weight・l1_weight・l2_weight を [−127/64, 127/64] に収める（バイアスは収めない）。
#
# ■ 書き出し（export.export_nnbin(checkpoint, out, fv_scale=None) が SFNN の構造なら SFNN の nn.bin を書く）
#   ファイルの並びは tests/sfnn_ref.py。es = eval_scale、fv = fv_scale（省略時 max(1, floor(8128 / es))）、
#   s = es × fv / 8128（s > 1 なら ValueError）。q(x, k, 型) = round_away(x × k) を型の範囲に収めたもの（M7 と同じ）。
#     FT：ft_b = q(ft_bias, 127, int16)、ft_w = q(因数分解を畳んだ重み, 127, int16)（ファイルにはこの値をそのまま書く）
#     ファイルのスタック j（0〜8）には、Shikibu のバケツ i = 8 − j の重みを書く（やねうら王のバケツの番号は
#       Shikibu の 8 − バケツ。k3k3 なしなら j = i = 0）。
#     fc_0：行ごとの倍率 r（行 < H1 は 1、近道の行 H1 は s）。b0 = q(l0_bias × r, 8128, int32)、w0 = q(l0_weight × r, 64, int8)
#     fc_1：b1 = q(l1_bias, 8128, int32)、w1 = q(l1_weight, 64, int8)
#     fc_2：b2 = q(l2_bias × s, 8128, int32)、w2 = q(l2_weight × s, 64, int8)
#   戻り値の dict：network（正規の名前）, arch（"SFNN_<特徴>_<W>_<H1>_<H2>" ＋ k3k3 なら "_k3k3"）, eval_scale, fv_scale,
#     stacks, yo_edition（"YANEURAOU_ENGINE_" ＋ arch）。
#   CLI（python -m shikibu.export）は従来と同じく "FV_SCALE=<整数>" と "EDITION=<yo_edition>" を表示する。
#   従来の構造（halfk… の nn.bin、raypair の .shk）の書き出しは変えない。
#
# ■ 実験室：lab.PRESETS に "sfnn_halfkahm2_512-15-32_f_k3k3" を足す。lab.describe はこの構造で export_format "nn.bin"、
#   yo_compatible True、yo_arch・yo_edition を返す。
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))

import cshogi  # noqa: E402
import kingzone_ref as K  # noqa: E402
import nnue_ref as N  # noqa: E402
import sfnn_ref as SF  # noqa: E402
from conftest import random_boards  # noqa: E402
from m3_data import make_material_dataset  # noqa: E402
from test_m3_acceptance import load_ckpt, make_config, read_log, run  # noqa: E402
from test_m7_acceptance import LIMIT, SPARSE, q, ref_features  # noqa: E402

from shikibu import networks, train  # noqa: E402

GOLDEN = os.path.join(HERE, "golden", "sfnn_golden.json")


# ================================================================ 参照実装（やねうら王との一致）

def test_reference_matches_yaneuraou(tmp_path):
    g = json.load(open(GOLDEN, encoding="utf-8"))
    assert g["arch"] == "SFNN_halfkahm2_512_15_32_k3k3"
    seen = set()
    for f in g["files"]:
        p = str(tmp_path / f"s{f['seed']}.bin")
        SF.make(p, g["arch"], f["seed"], f["scale"])
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == f["sha256"], "参照実装の nn.bin の作り方が変わっている"
        net = SF.Net(p, g["arch"])
        for c in f["cases"]:
            b = cshogi.Board(c["sfen"])
            k, out, ev = net.forward(b, 16)
            assert k == c["bucket"] and ev == c["eval16"], c["sfen"]
            if c["raw"] is not None:
                assert out == c["raw"], c["sfen"]
            seen.add((k, b.turn))
    assert len(seen) == 18  # 全部のバケツ × 両方の手番


def test_bucket_numbering():
    for b in random_boards(60, seed=10001) + [cshogi.Board(s) for s in SPARSE]:
        assert SF.k3k3_bucket(b) == 8 - K.k3k3_bucket(b)


# ================================================================ 構造の名前とモデル

def test_parse_arch_sfnn():
    p = networks.parse_arch("SFNN_HalfKAhm2_512-15-32_K3K3_F")
    assert p["canonical"] == "sfnn_halfkahm2_512-15-32_f_k3k3"
    assert p["sfnn"] is True and p["feature"] == "halfkahm2" and (p["W"], p["H1"], p["H2"]) == (512, 15, 32)
    assert p["factorize"] is True and p["stacks"] == 9 and p["shortcut"] is True
    assert p["num_features"] == N.num_features("halfkahm2") + N.plane_size("halfkahm2")
    q2 = networks.parse_arch("sfnn_halfkp_128-16-32")
    assert q2["stacks"] == 1 and q2["shortcut"] is False and q2["factorize"] is False
    assert networks.parse_arch("halfkahm2_512x2-16-32_f")["sfnn"] is False
    for bad in ("sfnn_halfkahm2_512-14-32_k3k3", "sfnn_halfkahm2_96-15-32", "sfnn_halfkahm2_512-15-30",
                "sfnn_raypair0_512-15-32", "sfnn_halfkahm2_512-15-32_psqt", "sfnn_halfkahm2_512-15-32_k3k3_k3k3"):
        with pytest.raises(ValueError):
            networks.parse_arch(bad)


SHAPES = ["sfnn_halfkahm2_64-7-8_k3k3", "sfnn_halfkp_64-8-16", "sfnn_halfka2_128-15-8_f_k3k3"]


def _random_tensors(m, seed):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, v in m.export_tensors().items():
        if k == "ft_weight":
            out[k] = (torch.rand(v.shape, generator=g) * 2 - 1) * 0.05
        elif k == "ft_bias":
            out[k] = torch.rand(v.shape, generator=g) * 0.5 + 0.1
        elif k.endswith("weight"):
            out[k] = (torch.rand(v.shape, generator=g) * 2 - 1) * LIMIT * 0.5
        else:
            out[k] = (torch.rand(v.shape, generator=g) * 2 - 1) * 0.3
    return out


def float_formula(p, w, b):
    stm = 0 if b.turn == cshogi.BLACK else 1
    W, H1 = p["W"], p["H1"]
    c = {}
    for persp in (stm, 1 - stm):
        f = ref_features(p["feature"], p["factorize"], b, persp)
        c[persp] = np.clip(w["ft_bias"] + w["ft_weight"][f].sum(axis=0), 0, 1)
    x = np.concatenate([c[pp][:W // 2] * c[pp][W // 2:] * 127 / 128 for pp in (stm, 1 - stm)])
    bk = K.k3k3_bucket(b) if p["stacks"] == 9 else 0
    y0 = w["l0_weight"][bk] @ x + w["l0_bias"][bk]
    h = np.concatenate([np.clip(y0[:H1] ** 2 * 127 / 128, 0, 1), np.clip(y0[:H1], 0, 1)])
    a1 = np.clip(w["l1_weight"][bk] @ h + w["l1_bias"][bk], 0, 1)
    z = float((w["l2_weight"][bk] @ a1 + w["l2_bias"][bk])[0])
    if p["shortcut"]:
        z += float(y0[H1])
    return z, bk


@pytest.mark.parametrize("arch", SHAPES)
def test_build_and_forward(arch):
    p = networks.parse_arch(arch)
    m = networks.build(arch)
    S, O0 = p["stacks"], p["H1"] + (1 if p["shortcut"] else 0)
    assert m.SFNN and (m.FEATURE_VERSION, m.NUM_FEATURES, m.ft_width, m.H1, m.H2) == \
        (p["version"], p["num_features"], p["W"], p["H1"], p["H2"])
    assert (bool(m.SHORTCUT), m.STACKS, bool(m.NEEDS_KINGS)) == (p["shortcut"], S, S == 9)
    want = {"ft_weight": (p["num_features"], p["W"]), "ft_bias": (p["W"],),
            "l0_weight": (S, O0, p["W"]), "l0_bias": (S, O0), "l1_weight": (S, p["H2"], 2 * p["H1"]),
            "l1_bias": (S, p["H2"]), "l2_weight": (S, 1, p["H2"]), "l2_bias": (S, 1)}
    assert {k: tuple(v.shape) for k, v in m.export_tensors().items()} == want
    w = _random_tensors(m, 23)
    m.set_from_tensors(w)
    m.eval()
    npw = {k: v.double().numpy() for k, v in w.items()}
    boards = random_boards(24, seed=10101) + [cshogi.Board(s) for s in SPARSE]
    si, so, ni, no, kings, expect, buckets = [], [], [], [], [], [], set()
    for b in boards:
        stm = 0 if b.turn == cshogi.BLACK else 1
        fs = ref_features(p["feature"], p["factorize"], b, stm)
        fn = ref_features(p["feature"], p["factorize"], b, 1 - stm)
        so.append(len(si)); si += fs
        no.append(len(ni)); ni += fn
        kings.append(list(K.king_squares(b)))
        z, bk = float_formula(p, npw, b)
        expect.append(z)
        buckets.add(bk)
    if S == 9:
        assert len(buckets) >= 3
    t = lambda a: torch.tensor(a, dtype=torch.int64)
    with torch.no_grad():
        z = m(t(si), t(so), t(ni), t(no), kings=t(kings))
    assert tuple(z.shape) == (len(boards),)
    assert np.allclose(z.double().numpy(), expect, atol=1e-4), f"{arch}: forward が式と違う"
    if S == 9:
        with pytest.raises(Exception):
            m(t(si), t(so), t(ni), t(no))
    back = m.export_tensors()
    for k in w:
        assert torch.equal(back[k], w[k].float()), k


def test_clip_sfnn():
    m = networks.build("sfnn_halfkahm2_64-7-8_k3k3")
    m.set_from_tensors({k: torch.full_like(v, 5.0) for k, v in m.export_tensors().items()})
    train.clip_weights_(m)
    t = m.export_tensors()
    assert t["ft_weight"].max() <= 1.0 and t["ft_bias"].max() <= 1.0
    for k in ("l0_weight", "l1_weight", "l2_weight"):
        assert t[k].max() <= LIMIT + 1e-6, k
    for k in ("l0_bias", "l1_bias", "l2_bias"):
        assert float(t[k].min()) == 5.0, f"{k} はクリップしない"


# ================================================================ 学習と書き出し

TRAIN_ARCHS = ["sfnn_halfkahm2_64-7-8_f_k3k3", "sfnn_halfkp_64-8-16"]


@pytest.fixture(scope="module")
def mat(tmp_path_factory):
    d = tmp_path_factory.mktemp("m10mat")
    paths = make_material_dataset(str(d / "train"), n_files=2, per_file=1500, seed=1077)
    val = make_material_dataset(str(d / "val"), n_files=1, per_file=1000, seed=1078)[0]
    return paths, val


@pytest.fixture(scope="module")
def runs(mat, tmp_path_factory):
    paths, val = mat
    out = {}
    for arch in TRAIN_ARCHS:
        d = tmp_path_factory.mktemp(arch.replace("-", "_"))
        W = networks.parse_arch(arch)["W"]
        cfg = make_config(d / "c.toml", paths, d / "out", model__network=arch, model__ft_width=W,
                          data__val_path=val, train__max_steps=60, train__total_steps=60, train__lr=3e-3,
                          train__eval_scale=600.0, train__save_every_steps=60, train__val_every_steps=20,
                          train__val_positions=1000)
        assert run(["--config", cfg]) == 0, f"{arch} の学習が失敗した"
        out[arch] = str(d / "out")
    return out


def test_training_runs(runs):
    for arch, out in runs.items():
        ck = load_ckpt(out)
        assert ck["network"] == arch and ck["step"] == 60
        losses = [r["loss"] for r in read_log(out) if r.get("type") == "train"]
        assert losses and all(math.isfinite(x) for x in losses), f"{arch} の損失が有限でない"
        assert np.mean(losses[-10:]) < np.mean(losses[:10]), f"{arch} の損失が下がらない"


def expected_sfnn(p, t, eval_scale, fv):
    d = lambda k: t[k].double().numpy()
    w = d("ft_weight")
    nf = N.num_features(p["feature"])
    if p["factorize"]:
        P = N.plane_size(p["feature"])
        w = w[:nf] + np.tile(w[nf:], (nf // P, 1))
    s = eval_scale * fv / 8128.0
    S, H1 = p["stacks"], p["H1"]
    O0 = H1 + (1 if p["shortcut"] else 0)
    r = np.ones((O0,))
    if p["shortcut"]:
        r[H1] = s
    out = {"ft_b": q(d("ft_bias"), 127, np.int16), "ft_w": q(w, 127, np.int16)}
    for k in ("b0", "w0", "b1", "w1", "b2", "w2"):
        out[k] = []
    for j in range(S):
        i = 8 - j if S == 9 else 0
        out["b0"].append(q(d("l0_bias")[i] * r, 8128, np.int32))
        out["w0"].append(q(d("l0_weight")[i] * r[:, None], 64, np.int8))
        out["b1"].append(q(d("l1_bias")[i], 8128, np.int32))
        out["w1"].append(q(d("l1_weight")[i], 64, np.int8))
        out["b2"].append(q(d("l2_bias")[i] * s, 8128, np.int32))
        out["w2"].append(q(d("l2_weight")[i] * s, 64, np.int8))
    for k in ("b0", "w0", "b1", "w1", "b2", "w2"):
        out[k] = np.stack(out[k])
    return out


def yo_arch(p):
    return f"SFNN_{p['feature']}_{p['W']}_{p['H1']}_{p['H2']}" + ("_k3k3" if p["stacks"] == 9 else "")


@pytest.mark.parametrize("arch", TRAIN_ARCHS)
def test_export_sfnn_exact(runs, tmp_path, arch):
    from shikibu import export
    ckp = os.path.join(runs[arch], "checkpoint.pt")
    path = str(tmp_path / "nn.bin")
    info = export.export_nnbin(ckp, path)
    ck = torch.load(ckp, map_location="cpu", weights_only=False)
    es = float(ck["eval_scale"])
    fv = max(1, math.floor(8128 / es))
    p = networks.parse_arch(arch)
    ya = yo_arch(p)
    assert info["network"] == arch and info["arch"] == ya and info["fv_scale"] == fv
    assert info["stacks"] == p["stacks"] and info["yo_edition"] == "YANEURAOU_ENGINE_" + ya
    m = networks.build(arch)
    m.load_state_dict(ck["model"])
    exp = expected_sfnn(p, m.export_tensors(), es, fv)
    # 仕様どおりの値を仕様どおりの並びで書いたバイト列と一致すること（構造の文字列の部分は自由）
    raw = open(path, "rb").read()
    n = int.from_bytes(raw[8:12], "little")
    assert int.from_bytes(raw[0:4], "little") == SF.VERSION and int.from_bytes(raw[4:8], "little") == SF.SFNN_HASH
    body = raw[12 + n:]
    want = bytearray()
    want += SF.FT_HASH.to_bytes(4, "little")
    want += SF.leb128_block(exp["ft_b"].reshape(-1)) + SF.leb128_block(exp["ft_w"].reshape(-1))
    W, H1, H2 = p["W"], p["H1"], p["H2"]

    def padded(a, cols):
        o = np.zeros((a.shape[0], SF.pad32(cols)), dtype=np.int8)
        o[:, :cols] = a
        return o.tobytes()
    for j in range(p["stacks"]):
        want += SF.NET_HASH.to_bytes(4, "little")
        want += exp["b0"][j].astype("<i4").tobytes() + padded(exp["w0"][j], W)
        want += exp["b1"][j].astype("<i4").tobytes() + padded(exp["w1"][j], 2 * H1)
        want += exp["b2"][j].astype("<i4").tobytes() + padded(exp["w2"][j], H2)
    assert bytes(body) == bytes(want), f"{arch}: nn.bin の中身が仕様と違う"
    if p["stacks"] == 9:
        assert SF.Net(path, ya).S == 9  # 参照実装で読めること


def test_exported_eval_matches_float(runs, tmp_path):
    from shikibu import export
    arch = "sfnn_halfkahm2_64-7-8_f_k3k3"
    ckp = os.path.join(runs[arch], "checkpoint.pt")
    path = str(tmp_path / "nn.bin")
    info = export.export_nnbin(ckp, path)
    ck = torch.load(ckp, map_location="cpu", weights_only=False)
    p = networks.parse_arch(arch)
    m = networks.build(arch)
    m.load_state_dict(ck["model"])
    m.eval()
    net = SF.Net(path, info["arch"])
    Z, E = [], []
    for b in random_boards(200, seed=10501):
        stm = 0 if b.turn == cshogi.BLACK else 1
        fs = ref_features(p["feature"], p["factorize"], b, stm)
        fn = ref_features(p["feature"], p["factorize"], b, 1 - stm)
        t = lambda a: torch.tensor(a, dtype=torch.int64)
        with torch.no_grad():
            z = m(t(fs), t([0]), t(fn), t([0]), kings=t([list(K.king_squares(b))])).item()
        k, _, cp = net.forward(b, info["fv_scale"])
        assert k == 8 - K.k3k3_bucket(b)
        Z.append(z * float(ck["eval_scale"]))
        E.append(cp)
    Z, E = np.array(Z), np.array(E)
    d = np.abs(E - Z)
    assert d.mean() <= 50 and d.max() <= 300, f"nn.bin の評価値が float のモデルとずれすぎる（平均 {d.mean():.1f}、最大 {d.max():.1f}）"
    if Z.std() > 30:
        slope = np.polyfit(Z, E, 1)[0]
        assert 0.9 <= slope <= 1.1, f"nn.bin の評価値の尺度が float のモデルと違う（傾き {slope:.3f}）"


def test_cli_edition(runs, tmp_path):
    ckp = os.path.join(runs["sfnn_halfkahm2_64-7-8_f_k3k3"], "checkpoint.pt")
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "python"))
    r = subprocess.run([sys.executable, "-m", "shikibu.export", "--checkpoint", ckp, "--out", str(tmp_path / "nn.bin")],
                       capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "FV_SCALE=13" in r.stdout and "EDITION=YANEURAOU_ENGINE_SFNN_halfkahm2_64_7_8_k3k3" in r.stdout, r.stdout


def test_lab_describe_sfnn():
    from shikibu import lab
    assert "sfnn_halfkahm2_512-15-32_f_k3k3" in lab.PRESETS
    d = lab.describe("sfnn_halfkahm2_512-15-32_f_k3k3")
    assert d["export_format"] == "nn.bin" and d["yo_compatible"] is True
    assert d["yo_arch"] == "SFNN_halfkahm2_512_15_32_k3k3"
    assert d["yo_edition"] == "YANEURAOU_ENGINE_SFNN_halfkahm2_512_15_32_k3k3"
