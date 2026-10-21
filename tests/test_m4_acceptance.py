# 変更禁止（PROTECTED）。Shikibu M4 の受け入れテスト。
#
# ■ shikibu.export（python/shikibu/export.py）
#   export_shk(checkpoint_path: str, out_path: str) -> dict
#       チェックポイントから §F4 の .shk を書き出し、ヘッダの dict を返す。
#       ヘッダの値：feature_set は "RayPair_v0" / "RayPair_v1"、network はネットワーク名、num_features・W・L1・L2 はモデルの値、
#       eval_scale・lam はチェックポイントの値、positions は「step × batch_size」（学習した局面数）。
#       書き出した後に読み直し、ヘッダ・CRC32・全テンソルが一致することを確かめる（一致しなければ例外）。
#       CLI：python -m shikibu.export --checkpoint <checkpoint.pt> --out <file.shk>
#   read_shk(path: str) -> tuple[dict, dict[str, numpy.ndarray]]
#       .shk を読み、(ヘッダ, テンソル) を返す。ヘッダのキーは
#       format_version, feature_set, network, num_features, W, L1, L2, eval_scale, lam, positions。
#       テンソルは §F4.2 の名前と形の float32。マジック・形式バージョン・予約領域・長さ・CRC32 のどれかが不正なら ValueError。
#
# ■ shikibu.quantsim（python/shikibu/quantsim.py）…設計書 §E4 の量子化をそのまま再現する
#   quantize(tensors: dict) -> dict          … 入力は float32 の numpy 配列（read_shk の戻り値と同じ形）。名前は §F4.2 と同じ。dtype：ft_* は int16、*_weight は int8、
#                                              l1/l2/out の bias は int32。丸めは「0から遠い方への四捨五入」。
#   qforward(q, stm_features, nstm_features) -> (acc_stm, acc_nstm, raw)
#       acc_*：第1層の蓄積値（int、長さ W）、raw：最終出力の int32 の値（z = raw / 8128）。
#   compare(shk_path, psv_paths, n) -> dict  … 先頭 n 局面で float 版と量子化版を比べる。キー：
#       n, mean_abs_z, max_abs_z, mean_abs_cp, max_abs_cp, loss_float, loss_quant
#       （loss は §S6 の損失、lam・eval_scale はヘッダの値）
#   CLI：python -m shikibu.quantsim --shk <file.shk> --psv <file.psv> [--n 100000]
#
# ■ tools/make_engine_golden.py（Nuenagon の N3 で使う正解データを作る）
#   python tools/make_engine_golden.py --shk <file.shk> --out <golden.json> --n <局面数> --seed <整数>
#   JSON：{"format": "shikibu-engine-golden-v1", "shk_sha256": <.shk の SHA-256（16進）>, "header": <read_shk のヘッダ>,
#          "positions": [{"sfen", "stm_features", "nstm_features", "acc_stm", "acc_nstm", "raw", "cp"}, ...]}
#   features は昇順・重複込み。cp は 倍精度で round(raw × eval_scale / 8128)（0から遠い方への四捨五入、クリップなし）。
#   positions の先頭は平手初期局面。後手番の局面と、持ち駒のある局面をそれぞれ1つ以上含む。局面はランダムに指し進めて作る（seed で決まる）。
#
# ■ 学習の進捗表示（shikibu.train の追加仕様）
#   設定キー train.progress_every_sec（既定 30）：この秒数ごと（0 なら log_every_steps ごと）に、画面（標準出力）へ
#     「12.3%」のような小数1桁の進捗と、「残り」の文字を含む1行を出す（例：[  12.3%] step 1,234 / 10,000  残り 1時間02分  103.1万局面/秒）。
#   log.jsonl の train 行に "progress"（= step / max_steps）と "eta_sec"（残り秒数の見込み、0以上）を追加する。
#   log.jsonl の1行目は従来どおり "start" 行のままにする（M3 のテストとの互換）。
#   段階の名前は順に "data_check"（勝敗欠損の確認）→ "eval_scale"（推定するときだけ）→ "prefetch"（cache_dir を使うときだけ）
#     → "buffer_fill"（shuffle_buffer > 0 のときだけ）→ "train" → 最後に "done"（正常終了）または "error"（エラーで終了）。
#   出力フォルダの status.json を、段階が変わるたびと、画面に進捗を出すたびに上書きする（一時ファイル→改名）。中身：
#     {"phase": <今の段階>, "history": [<入った段階の名前を順に>], "step": int, "max_steps": int,
#      "progress": float, "eta_sec": float | null, "updated": <UNIX 時刻>}
#   start 行より後の段階（"prefetch"・"buffer_fill"・"train"）は、log.jsonl にも {"type": "phase", "name": <名前>} の行を書く。
#   設定キー train.scale_sample：勝敗欠損の確認と eval_scale の推定で読む局面数の上限。
#     全学習ファイルからランダムな位置の連続ブロックを取り出して使う（ファイルの先頭だけにしない）。
#     start 行に "scale_positions_used"（実際に使った局面数）を書く。データがこれより少なければ全部使う。
#   既存のチェックポイント（M3 の形式）からの --resume は引き続きできること。
import hashlib
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest
import torch

import m4_ref
from conftest import ref
from m2_data import make_scale_dataset
from m3_data import make_material_dataset
from test_m3_acceptance import load_ckpt, make_config, read_log, run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from shikibu import export, networks, quantsim  # noqa: E402


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("m4")
    paths = make_material_dataset(str(d / "data"))
    out = {}
    for net in ("RayPair_v0Network", "RayPair_v1Network"):
        o = d / net
        cfg = make_config(d / f"{net}.toml", paths, o, model__network=net, train__lr=3e-3,
                          train__max_steps=30, train__total_steps=30, train__eval_scale=543.21, train__lam=0.75)
        assert run(["--config", cfg]) == 0
        out[net] = str(o / "checkpoint.pt")
    return {"paths": paths, "ck": out, "dir": d}


def model_tensors(ck_path):
    ck = load_ckpt(os.path.dirname(ck_path))
    m = networks.build(ck["network"])
    m.load_state_dict(ck["model"])
    return ck, {k: v.numpy() for k, v in m.export_tensors().items()}


# ---------- 書き出しと読み込み ----------

@pytest.mark.parametrize("net,feat,n,l", [("RayPair_v0Network", "RayPair_v0", 15000, 16),
                                          ("RayPair_v1Network", "RayPair_v1", 115352, 32)])
def test_export_matches_spec(trained, tmp_path, net, feat, n, l):
    ck_path = trained["ck"][net]
    p = str(tmp_path / "x.shk")
    hdr = export.export_shk(ck_path, p)
    h, t = m4_ref.parse_shk(open(p, "rb").read())       # 設計書から書いた独立の読み取り
    ck, mt = model_tensors(ck_path)
    assert h["feature_set"] == feat and h["network"] == net
    assert (h["num_features"], h["W"], h["L1"], h["L2"], h["format_version"]) == (n, 128, l, l, 1)
    assert h["eval_scale"] == np.float32(543.21) and h["lam"] == np.float32(0.75)
    assert h["positions"] == ck["step"] * ck["config"]["train"]["batch_size"]
    for k in m4_ref.ORDER:
        assert np.array_equal(t[k], mt[k]), k
    assert hdr["feature_set"] == feat and hdr["positions"] == h["positions"]


def test_read_shk_and_corruption(trained, tmp_path):
    p = str(tmp_path / "v0.shk")
    export.export_shk(trained["ck"]["RayPair_v0Network"], p)
    data = open(p, "rb").read()
    h, t = export.read_shk(p)
    rh, rt = m4_ref.parse_shk(data)
    for k in rh:
        assert h[k] == rh[k] or (isinstance(rh[k], float) and abs(h[k] - rh[k]) < 1e-6), k
    for k in m4_ref.ORDER:
        assert np.array_equal(np.asarray(t[k]), rt[k]), k
    bad = {
        "crc": data[:1000] + bytes([data[1000] ^ 1]) + data[1001:],
        "magic": b"SHIKIBX\0" + data[8:],
        "version": data[:8] + (2).to_bytes(4, "little") + data[12:],
        "truncated": data[:-10],
        "reserved": data[:120] + b"\x01" + data[121:],
    }
    for name, b in bad.items():
        q = tmp_path / f"bad_{name}.shk"
        q.write_bytes(b)
        with pytest.raises(ValueError):
            export.read_shk(str(q))


# ---------- 量子化 ----------

def test_quantize_matches_spec(trained):
    _, t = model_tensors(trained["ck"]["RayPair_v0Network"])
    t = {k: v.copy() for k, v in t.items()}
    # 丸めの方向を確かめる値（×64 でちょうど .5 になる、2進で正確な値）
    t["l1_weight"][0, :6] = np.array([2.5, -2.5, 0.5, -0.5, 1.5, -1.5], dtype=np.float32) / 64
    t["l1_weight"][1, :2] = np.array([127 / 64, -127 / 64], dtype=np.float32)
    q = quantsim.quantize(t)
    r = m4_ref.quantize(t)
    for k in m4_ref.ORDER:
        assert np.asarray(q[k]).dtype == r[k].dtype, k
        assert np.array_equal(np.asarray(q[k]), r[k]), k
    assert list(np.asarray(q["l1_weight"])[0, :6]) == [3, -3, 1, -1, 2, -2]


@pytest.mark.parametrize("net", ["RayPair_v0Network", "RayPair_v1Network"])
def test_qforward_matches_spec(trained, net):
    from conftest import random_boards
    ck, t = model_tensors(trained["ck"][net])
    v = 0 if net.endswith("v0Network") else 1
    q = quantsim.quantize(t)
    r = m4_ref.quantize(t)
    m = networks.build(net)
    m.load_state_dict(ck["model"])
    m.eval()
    diffs = []
    for b in random_boards(60, seed=77):
        s = b.sfen()
        stm = 0 if s.split()[1] == "b" else 1
        fs, fn = ref.features(s, stm, v), ref.features(s, 1 - stm, v)
        a1, b1, raw1 = quantsim.qforward(q, fs, fn)
        a2, b2, raw2 = m4_ref.qforward(r, fs, fn)
        assert list(np.asarray(a1)) == list(a2) and list(np.asarray(b1)) == list(b2)
        assert int(raw1) == raw2
        tt = lambda a: torch.tensor(a, dtype=torch.int64)
        with torch.no_grad():
            zf = m(tt(fs), tt([0]), tt(fn), tt([0])).item()
        diffs.append(abs(zf - raw2 / 8128))
    assert np.mean(diffs) < 0.05, np.mean(diffs)


def test_compare(trained, tmp_path):
    p = str(tmp_path / "v0.shk")
    export.export_shk(trained["ck"]["RayPair_v0Network"], p)
    r = quantsim.compare(p, trained["paths"][:1], 2000)
    for k in ("n", "mean_abs_z", "max_abs_z", "mean_abs_cp", "max_abs_cp", "loss_float", "loss_quant"):
        assert k in r, k
    assert r["n"] == 2000
    assert 0 <= r["mean_abs_z"] < 0.05 and r["max_abs_z"] >= r["mean_abs_z"]
    assert abs(r["loss_quant"] - r["loss_float"]) < 0.01


# ---------- エンジン用の正解データ ----------

def test_engine_golden(trained, tmp_path):
    p = str(tmp_path / "v0.shk")
    export.export_shk(trained["ck"]["RayPair_v0Network"], p)
    out = str(tmp_path / "golden.json")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(ROOT, "python") + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "make_engine_golden.py"), "--shk", p,
                        "--out", out, "--n", "40", "--seed", "3"], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    g = json.load(open(out, encoding="utf-8"))
    assert g["format"] == "shikibu-engine-golden-v1"
    assert g["shk_sha256"] == hashlib.sha256(open(p, "rb").read()).hexdigest()
    h, t = m4_ref.parse_shk(open(p, "rb").read())
    assert g["header"]["feature_set"] == "RayPair_v0"
    q = m4_ref.quantize(t)
    pos = g["positions"]
    assert len(pos) == 40
    assert pos[0]["sfen"].split()[:3] == ["lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL", "b", "-"]
    assert any(x["sfen"].split()[1] == "w" for x in pos)
    assert any(x["sfen"].split()[2] != "-" for x in pos)
    for x in pos:
        s = x["sfen"]
        stm = 0 if s.split()[1] == "b" else 1
        assert x["stm_features"] == ref.features(s, stm, 0)
        assert x["nstm_features"] == ref.features(s, 1 - stm, 0)
        a, b, raw = m4_ref.qforward(q, x["stm_features"], x["nstm_features"])
        assert x["acc_stm"] == [int(v) for v in a] and x["acc_nstm"] == [int(v) for v in b]
        assert x["raw"] == raw
        assert x["cp"] == m4_ref.cp_of(raw, h["eval_scale"])


# ---------- 進捗表示 ----------

def test_progress_reporting(trained, tmp_path, capsys):
    out = tmp_path / "prog"
    cfg = make_config(tmp_path / "p.toml", trained["paths"], out, train__max_steps=12, train__total_steps=12,
                      train__log_every_steps=1, train__progress_every_sec=0, data__shuffle_buffer=1000)
    assert run(["--config", cfg]) == 0
    text = capsys.readouterr().out
    assert re.search(r"\d+\.\d%", text), "進捗（小数1桁の%）が画面に出ていない"
    assert "残り" in text, "残り時間が画面に出ていない"
    log = read_log(str(out))
    assert log[0]["type"] == "start"
    names = [r["name"] for r in log if r.get("type") == "phase"]
    assert "buffer_fill" in names and "train" in names
    assert names.index("buffer_fill") < names.index("train")
    st = json.load(open(out / "status.json", encoding="utf-8"))
    h = st["history"]
    assert h[0] == "data_check" and "eval_scale" not in h, h
    assert h.index("buffer_fill") < h.index("train") and h[-1] == "done" and st["phase"] == "done", h
    assert st["step"] == 12 and st["max_steps"] == 12 and abs(st["progress"] - 1.0) < 1e-9
    first_train = next(i for i, r in enumerate(log) if r.get("type") == "train")
    train_phase = next(i for i, r in enumerate(log) if r.get("type") == "phase" and r["name"] == "train")
    assert train_phase < first_train
    trains = [r for r in log if r.get("type") == "train"]
    for r in trains:
        assert abs(r["progress"] - r["step"] / 12) < 1e-9
        assert r["eta_sec"] >= 0
    assert trains[-1]["eta_sec"] < 1e-6 + trains[0]["eta_sec"]


def test_scale_sample_limit(tmp_path):
    p = str(tmp_path / "scale.psv")
    make_scale_dataset(p, 800.0)
    out = tmp_path / "lim"
    cfg = make_config(tmp_path / "l.toml", [p], out, train__eval_scale="auto", train__lam=0.75,
                      train__scale_sample=50_000, train__max_steps=2, train__total_steps=2)
    assert run(["--config", cfg]) == 0
    log = read_log(str(out))
    start = next(r for r in log if r.get("type") == "start")
    assert 40_000 <= start["scale_positions_used"] <= 50_000, start
    assert abs(start["eval_scale"] - 800.0) / 800.0 < 0.10, start
    assert log[0]["type"] == "start"
    st = json.load(open(out / "status.json", encoding="utf-8"))
    assert st["history"][:2] == ["data_check", "eval_scale"] and st["history"][-1] == "done", st["history"]


def test_status_on_error(tmp_path):
    p = str(tmp_path / "nores.psv")
    make_scale_dataset(p, 600.0, n=20_000, draw_rate=1.0, n_mate=0)
    out = tmp_path / "err"
    rc = run(["--config", make_config(tmp_path / "e.toml", [p], out, train__lam=0.75,
                                      train__max_steps=3, train__total_steps=3)])
    assert rc != 0
    st = json.load(open(out / "status.json", encoding="utf-8"))
    assert st["phase"] == "error" and st["history"][0] == "data_check", st
