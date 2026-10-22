# 変更禁止（PROTECTED）。Shikibu M6（HalfKP の学習と、やねうら王の nn.bin の書き出し）の受け入れテスト。
#
# 目的：同じ学習器・同じデータ・同じ損失で RayPair と HalfKP を学習し、評価関数の構造の差を測れるようにする（設計書 §S11）。
# HalfKP の特徴量と推論の定義は tests/halfkp_ref.py（やねうら王の静的評価値と一致することを確かめた参照実装）、
# nn.bin のファイルの並びは tests/halfkp_nnbin.py。
#
# ■ 特徴の版（shikibu_feat）
#   版 2 = HalfKP：num_features(2) = 125388。extract_sfen(sfen, 視点, 2) は halfkp_ref.features(cshogi.Board(sfen), 視点) と同じ
#     （昇順・重複込み。盤上にも持ち駒にも無い駒は、駒の番号 0 として数える）。
#   版 3 = HalfKP の因数分解つき（HalfKP^）：num_features(3) = 125388 + 1548 = 126936。
#     版 2 の各特徴 f に加えて、玉の位置に関係ない特徴 125388 + (f mod 1548) を足したもの（昇順・重複込み）。
#   extract_psv・BatchLoader も版 2・3 に対応する（各局面の特徴は extract_psv と同じ）。
#
# ■ ネットワーク（shikibu.networks）
#   "HalfKP_256Network"：版 2、NUM_FEATURES 125388、W 256、L1 32、L2 32。
#   "HalfKPf_256Network"：版 3（因数分解つき）、NUM_FEATURES 126936、W 256、L1 32、L2 32。
#   計算式・重みの名前と形・重みのクリップは RayPair と同じ（§F3.2・§S5）。model.ft_width は 256 で使う。
#   学習（shikibu.train）は、model.network にこの名前を書けば、RayPair と同じ設定・同じ手順で学習できる。
#
# ■ 書き出し（shikibu.export）
#   export_nnbin(checkpoint_path, out_path, fv_scale=None) -> dict（キー：network, eval_scale, fv_scale）
#   CLI：python -m shikibu.export --checkpoint <checkpoint.pt> --out <file> で、HalfKP のネットワークなら nn.bin を書き出し、
#        標準出力に "FV_SCALE=<整数>" を含む1行を出す（RayPair なら従来どおり .shk）。--fv-scale <整数> で指定もできる。
#   やり方（すべて倍精度、丸めは「0 から遠い方への四捨五入」）：
#     1. 因数分解つきなら、畳み込む：W[k×1548 + p] ← W[k×1548 + p] + W[125388 + p]（k = 0〜80、p = 0〜1547）。
#     2. fv_scale を決める：指定が無ければ max(1, floor(8128 / eval_scale))。s = eval_scale × fv_scale / 8128（≦ 1）。
#     3. 量子化：ft_bias・ft_weight → round(×127)（int16）、l1・l2 の weight → round(×64)（int8）、bias → round(×8128)（int32）、
#        out_weight → round(×s×64)（int8）、out_bias → round(×s×8128)（int32）。範囲を超えたらその型の範囲にクリップする。
#     4. halfkp_nnbin.py の並び（版 0x7AF32F16、ハッシュ、構造の文字列 ARCH、…）で書く。
#   こうすると、やねうら王・Nuenagon で FV_SCALE=<fv_scale> にしたとき、cp ≒ 学習の z × eval_scale になる。
import json
import math
import os
import struct
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
import halfkp_nnbin as NB  # noqa: E402
import halfkp_ref as R  # noqa: E402
from conftest import make_psv, random_boards  # noqa: E402
from m2_data import write_records  # noqa: E402
from m3_data import make_material_dataset  # noqa: E402
from test_m3_acceptance import load_ckpt, make_config, run  # noqa: E402

from shikibu import networks  # noqa: E402

FE = 1548
NK = R.N_FEATURES  # 125388
# 駒の足りない局面（詰将棋のような。駒箱の駒が特徴に入る）
SPARSE = ["4k4/9/4g4/9/9/9/9/4R4/4K4 b - 1", "4k4/9/9/9/9/9/4g4/9/4K4 b g 1", "4k4/9/9/9/9/9/9/4r4/4K4 b - 1",
          "1+Psnkp3/9/4P2+P1/4L4/9/9/6K2/9/9 b 2G 1", "9/9/k8/N1B6/R+P7/3G5/4K4/9/9 b 2GB 1"]


def factorized(feats):
    return sorted(list(feats) + [NK + (f % FE) for f in feats])


def round_away(x):
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def q(x, scale, dtype):
    info = np.iinfo(dtype)
    return np.clip(round_away(np.asarray(x, dtype=np.float64) * scale), info.min, info.max).astype(dtype)


# ---------------------------------------------------------------- 特徴

def test_num_features(feat):
    assert feat.num_features(2) == 125388
    assert feat.num_features(3) == 126936


def test_extract_matches_reference(feat):
    boards = random_boards(400, seed=6006) + [cshogi.Board(s) for s in SPARSE]
    for b in boards:
        s = b.sfen()
        for p in (0, 1):
            want = R.features(b, p)
            got = list(feat.extract_sfen(s, p, 2))
            assert got == want, f"版 2 の特徴が参照実装と違う（視点 {p}）: {s}"
            got3 = list(feat.extract_sfen(s, p, 3))
            assert got3 == factorized(want), f"版 3 の特徴が違う（視点 {p}）: {s}"


@pytest.fixture(scope="module")
def mat_paths(tmp_path_factory):
    return make_material_dataset(str(tmp_path_factory.mktemp("m6mat")), n_files=2, per_file=1500, seed=66)


@pytest.mark.parametrize("version", [2, 3])
def test_loader(feat, tmp_path, version):
    boards = random_boards(300, seed=6106) + [cshogi.Board(s) for s in SPARSE]
    recs = [make_psv(b, 10 * i, i, 50, 1 if i % 2 else -1) for i, b in enumerate(boards)]
    p = str(tmp_path / "h.psv")
    write_records(p, recs)
    got = []
    for batch in feat.BatchLoader([p], 64, version, shuffle_buffer=0, num_threads=2):
        n = len(batch["score"])
        for side in ("stm", "nstm"):
            idx, off = batch[f"{side}_indices"], batch[f"{side}_offsets"]
            ends = list(off[1:]) + [len(idx)]
            for i in range(n):
                got.append((side, list(idx[off[i]:ends[i]])))
    want = []
    for b in boards:
        stm = 0 if b.turn == cshogi.BLACK else 1
        f = {s: R.features(b, s) for s in (0, 1)}
        if version == 3:
            f = {s: factorized(v) for s, v in f.items()}
        want.append(("stm", f[stm]))
        want.append(("nstm", f[1 - stm]))
    # バッチの中では stm の全局面 → nstm の全局面の順に並ぶので、局面ごとに並べ替えて比べる
    by = {"stm": [g for s, g in got if s == "stm"], "nstm": [g for s, g in got if s == "nstm"]}
    wby = {"stm": [g for s, g in want if s == "stm"], "nstm": [g for s, g in want if s == "nstm"]}
    for side in ("stm", "nstm"):
        assert len(by[side]) == len(wby[side])
        for i, (a, w) in enumerate(zip(by[side], wby[side])):
            assert a == w, f"BatchLoader（版 {version}）の {side} の特徴が違う: {boards[i].sfen()}"


# ---------------------------------------------------------------- 学習と書き出し

@pytest.fixture(scope="module")
def trained(mat_paths, tmp_path_factory):
    out = {}
    for name in ("HalfKP_256Network", "HalfKPf_256Network"):
        d = tmp_path_factory.mktemp(name)
        cfg = make_config(d / "c.toml", mat_paths, d / "out", model__network=name, model__ft_width=256,
                          train__max_steps=60, train__total_steps=60, train__lr=3e-3, train__eval_scale=600.0,
                          train__save_every_steps=60, train__val_every_steps=60)
        assert run(["--config", cfg]) == 0, f"{name} の学習が失敗した"
        out[name] = str(d / "out")
    return out


def test_networks_registered():
    for name, nf, ver in (("HalfKP_256Network", 125388, 2), ("HalfKPf_256Network", 126936, 3)):
        m = networks.build(name, ft_width=256)
        assert m.NUM_FEATURES == nf and m.FEATURE_VERSION == ver and m.L1 == 32 and m.L2 == 32 and m.ft_width == 256


def test_training_runs(trained):
    for name, out in trained.items():
        ck = load_ckpt(out)
        assert ck["network"] == name and ck["step"] == 60
        log = [json.loads(l) for l in open(os.path.join(out, "log.jsonl"), encoding="utf-8") if l.strip()]
        losses = [r["loss"] for r in log if r.get("type") == "train"]
        assert losses and all(math.isfinite(x) for x in losses), f"{name} の損失が有限でない"
        assert np.mean(losses[-10:]) < np.mean(losses[:10]), f"{name} の損失が下がらない: {losses[:3]} … {losses[-3:]}"


def expected_quantized(ck, fv):
    sd = {k: v.detach().double().numpy() for k, v in ck["model"].items()}
    w = sd["ft.weight"].copy()
    if w.shape[0] == NK + FE:
        w = w[:NK] + np.tile(w[NK:], (81, 1))
    es = float(ck["eval_scale"])
    s = es * fv / 8128.0
    return {
        "ft_b": q(sd["ft_bias"], 127, np.int16), "ft_w": q(w, 127, np.int16),
        "b1": q(sd["l1.bias"], 8128, np.int32), "w1": q(sd["l1.weight"], 64, np.int8),
        "b2": q(sd["l2.bias"], 8128, np.int32), "w2": q(sd["l2.weight"], 64, np.int8),
        "bo": q(sd["out.bias"] * s, 8128, np.int32), "wo": q(sd["out.weight"] * s, 64, np.int8),
    }


@pytest.mark.parametrize("name", ["HalfKP_256Network", "HalfKPf_256Network"])
def test_export_exact(trained, tmp_path, name):
    from shikibu import export
    ckp = os.path.join(trained[name], "checkpoint.pt")
    path = str(tmp_path / "nn.bin")
    info = export.export_nnbin(ckp, path)
    ck = torch.load(ckp, map_location="cpu", weights_only=False)
    fv = max(1, math.floor(8128 / float(ck["eval_scale"])))
    assert info["fv_scale"] == fv, info
    data = open(path, "rb").read()
    ver, h, n = struct.unpack_from("<III", data, 0)
    ft_hash, net_hash, file_hash = NB.hashes()
    assert ver == NB.VERSION and h == file_hash and data[12:12 + n].decode() == NB.ARCH
    assert struct.unpack_from("<I", data, 12 + n)[0] == ft_hash
    net = R.Net(path)  # 並びと長さの確認を兼ねる
    exp = expected_quantized(ck, fv)
    got = {"ft_b": net.ft_b, "ft_w": net.ft_w, "b1": net.b1, "w1": net.w1, "b2": net.b2, "w2": net.w2,
           "bo": net.bo, "wo": net.wo}
    for k, e in exp.items():
        g = np.asarray(got[k]).reshape(e.shape).astype(np.int64)
        assert np.array_equal(g, e.astype(np.int64)), f"{name}: {k} の量子化が仕様と違う（違う要素 {int((g != e).sum())} 個）"


@pytest.mark.parametrize("name", ["HalfKP_256Network", "HalfKPf_256Network"])
def test_exported_eval_matches_float(trained, tmp_path, name):
    from shikibu import export
    ckp = os.path.join(trained[name], "checkpoint.pt")
    path = str(tmp_path / "nn.bin")
    info = export.export_nnbin(ckp, path)
    ck = torch.load(ckp, map_location="cpu", weights_only=False)
    m = networks.build(name, ft_width=256)
    m.load_state_dict(ck["model"])
    m.eval()
    net = R.Net(path)
    ver = m.FEATURE_VERSION
    diffs = []
    for b in random_boards(200, seed=6206):
        stm = 0 if b.turn == cshogi.BLACK else 1
        fs, fn = R.features(b, stm), R.features(b, 1 - stm)
        if ver == 3:
            fs, fn = factorized(fs), factorized(fn)
        with torch.no_grad():
            z = m(torch.tensor(fs), torch.tensor([0]), torch.tensor(fn), torch.tensor([0])).item()
        cp_float = z * float(ck["eval_scale"])
        _, _, cp = net.forward(b, info["fv_scale"])
        diffs.append(abs(cp - cp_float))
    mean, mx = float(np.mean(diffs)), float(np.max(diffs))
    assert mean <= 20 and mx <= 150, f"{name}: 書き出した nn.bin の評価値が float のモデルとずれすぎる（平均 {mean:.1f}、最大 {mx:.1f} cp）"


def test_cli_prints_fv_scale(trained, tmp_path):
    ckp = os.path.join(trained["HalfKP_256Network"], "checkpoint.pt")
    out = str(tmp_path / "cli.bin")
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "python"))
    r = subprocess.run([sys.executable, "-m", "shikibu.export", "--checkpoint", ckp, "--out", out, "--fv-scale", "12"],
                       capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "FV_SCALE=12" in r.stdout, r.stdout
    assert open(out, "rb").read(4) == struct.pack("<I", NB.VERSION)


def test_raypair_unchanged(trained, mat_paths, tmp_path):
    """RayPair の学習と .shk の書き出しは、従来どおり動く。"""
    from shikibu import export
    cfg = make_config(tmp_path / "rp.toml", mat_paths, tmp_path / "rp", train__max_steps=10, train__total_steps=10)
    assert run(["--config", cfg]) == 0
    h = export.export_shk(str(tmp_path / "rp" / "checkpoint.pt"), str(tmp_path / "rp.shk"))
    assert h["feature_set"] == "RayPair_v0"
