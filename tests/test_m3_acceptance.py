# 変更禁止（PROTECTED）。Shikibu M3（学習ループ）の受け入れテスト。
#
# 実装は Python パッケージ `shikibu`（置き場所：python/shikibu/）。テストは python/ を import パスに加える。
# データの読み込みには M1/M2 の shikibu_feat（BatchLoader・psv_stats・estimate_eval_scale）を使うこと。
#
# ■ shikibu.networks
#   NETWORKS: dict[str, type]  … "RayPair_v0Network", "RayPair_v1Network" を登録（§F3.1）
#   build(name: str, ft_width: int = 128) -> nn.Module
#   ネットワークのクラス属性：FEATURE_VERSION（0/1）、NUM_FEATURES、L1、L2。インスタンス属性：ft_width
#   forward(stm_indices, stm_offsets, nstm_indices, nstm_offsets) -> Tensor[B]
#       手番側から見た勝率のロジット z。計算は §F3.2 のとおり（重複したインデックスは回数ぶん加算）。
#   export_tensors() -> dict[str, Tensor]
#       §F4.2 の名前と形：ft_weight [N][W], ft_bias [W], l1_weight [L1][2W], l1_bias [L1],
#       l2_weight [L2][L1], l2_bias [L2], out_weight [1][L2], out_bias [1]（float32、CPU 上のコピー）
#   set_from_tensors(d: dict[str, Tensor]) -> None  … export_tensors と同じ名前・形で重みを設定する
#
# ■ shikibu.train
#   clip_weights_(model) -> None        … §F3.3 のクリップ（ft_weight/ft_bias は [-1,1]、
#                                          l1/l2/out の weight は [-127/64, 127/64]、それ以外の bias はそのまま）
#   compute_loss(z, score, result, eval_scale, lam) -> Tensor（スカラー）
#       t = lam * sigmoid(score / eval_scale) + (1 - lam) * result、損失 = mean(BCEWithLogits(z, t))
#   build_optimizer(model, lr, weight_decay) -> torch.optim.AdamW
#       weight decay は l1/l2/out の weight だけ。ft_weight・ft_bias・すべての bias は 0。
#   lr_factor(step, warmup_steps, total_steps, min_lr_ratio) -> float
#       step < warmup_steps なら (step + 1) / warmup_steps。それ以外は
#       p = min(1, (step - warmup_steps) / max(1, total_steps - warmup_steps)) として
#       min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + cos(pi * p))。step は 0 始まりの更新回数。
#   main(argv: list[str]) -> int
#       python -m shikibu.train --config <TOML> [--resume] と同じ。正常終了で 0 を返す。
#       設定エラー・勝敗欠損では 0 以外を返すか、例外（SystemExit / ValueError / RuntimeError）を送出する。
#
# ■ 設定ファイル（TOML）のキー：下の make_config を参照（すべてのキーに対応すること）。
#   train.max_steps：この実行で、通算の更新回数がこの値に達したら保存して終了する（Colab の区切り用）。
#   train.total_steps：学習率スケジュールの長さ。
#   train.seed：重みの初期値・データの並び・乱数のすべてをこの値で決める（同じ設定なら同じ結果になること）。
#   train.eval_scale："auto" なら学習開始時に estimate_eval_scale で推定、数値ならそれを使う。
#   勝敗欠損の判定（psv_stats）と eval_scale の推定（estimate_eval_scale）は、train_paths の「先頭の1ファイル」だけで行う
#   （シャードは事前シャッフル済みなので全体を代表する。全データを読むと Colab では Drive 経由で30分以上かかるため）。
#   --resume のときは推定をやり直さず、チェックポイントの eval_scale を使う。
#   データを1周したら、次の周は seed + 周回番号 で新しい BatchLoader を作って続ける（最後の端数バッチも使う）。
#   勝敗欠損（psv_stats の result_missing）のとき：allow_missing_results が false なら学習を始めずにエラー。
#     true のときは lam == 1.0 かつ eval_scale が数値であることを要求し、満たさなければエラー。
#
# ■ 出力（output.out_dir）
#   checkpoint.pt：最新のチェックポイント（torch.save の dict）。少なくとも次のキーを持つ：
#     "model", "optimizer", "step", "epoch", "loader_state", "eval_scale", "lam", "network", "config", "rng"
#     save_every_steps / save_every_minutes ごと、および終了時に保存。書き込みは一時ファイル→改名で行う。
#   log.jsonl：1行1 JSON。
#     {"type": "start", "eval_scale": float, "eval_scale_source": "auto" | "config", ...}
#     {"type": "train", "step": int, "loss": float, "lr": float, ...}   … log_every_steps ごと
#     {"type": "val", "step": int, "val_loss": float, "val_mse": float, ...}  … val_every_steps ごと
#       val_mse = mean((sigmoid(z) - sigmoid(score / eval_scale))^2)。val_path の先頭 val_positions 局面で計算。
#   --resume のときは checkpoint.pt から、モデル・オプティマイザ・step・周回・ローダーの位置・乱数状態を復元して続ける。
#   train.deterministic = true（CPU、torch_threads = 1、shuffle_buffer = 0）なら、
#   「通しで N ステップ」と「途中で止めて --resume で N ステップまで」の最終の重みはビット単位で一致すること。
import json
import math
import os
import sys

import numpy as np
import pytest
import torch

from conftest import random_boards, ref
from m2_data import make_scale_dataset
from m3_data import make_material_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from shikibu import networks, train  # noqa: E402

LIMIT = 127.0 / 64.0
NAMES = ["ft_weight", "ft_bias", "l1_weight", "l1_bias", "l2_weight", "l2_bias", "out_weight", "out_bias"]


def make_config(path, train_paths, out_dir, **over):
    cfg = {
        "data": {"train_paths": list(train_paths), "val_path": "", "cache_dir": "", "max_cached": 3,
                 "shuffle_buffer": 0, "num_threads": 2},
        "model": {"network": "RayPair_v0Network", "ft_width": 128},
        "train": {"batch_size": 256, "lr": 1e-3, "weight_decay": 0.0, "warmup_steps": 5, "total_steps": 40,
                  "max_steps": 40, "min_lr_ratio": 0.01, "lam": 1.0, "eval_scale": 600.0, "scale_sample": 1_000_000,
                  "allow_missing_results": False, "seed": 0, "device": "cpu", "torch_threads": 1,
                  "deterministic": True, "save_every_steps": 10, "save_every_minutes": 30.0,
                  "val_every_steps": 10, "val_positions": 2000, "log_every_steps": 1},
        "output": {"out_dir": str(out_dir)},
    }
    for k, v in over.items():
        sec, key = k.split("__")
        cfg[sec][key] = v

    def val(x):
        if isinstance(x, bool):
            return "true" if x else "false"
        if isinstance(x, str):
            return json.dumps(x)
        if isinstance(x, list):
            return "[" + ", ".join(val(e) for e in x) + "]"
        return repr(x)

    with open(path, "w", encoding="utf-8") as f:
        for sec, kv in cfg.items():
            f.write(f"[{sec}]\n")
            for k, v in kv.items():
                f.write(f"{k} = {val(v)}\n")
            f.write("\n")
    return str(path)


def run(argv):
    try:
        rc = train.main(argv)
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else 1
    return rc


def read_log(out_dir):
    with open(os.path.join(out_dir, "log.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_ckpt(out_dir):
    return torch.load(os.path.join(out_dir, "checkpoint.pt"), map_location="cpu", weights_only=False)


@pytest.fixture(scope="module")
def mat_paths(tmp_path_factory):
    return make_material_dataset(str(tmp_path_factory.mktemp("mat")))


# ---------- ネットワーク ----------

@pytest.mark.parametrize("name,ver,nf,l1", [("RayPair_v0Network", 0, 15000, 16), ("RayPair_v1Network", 1, 115352, 32)])
def test_registry_and_shapes(name, ver, nf, l1):
    assert name in networks.NETWORKS
    m = networks.build(name)
    assert (m.FEATURE_VERSION, m.NUM_FEATURES, m.L1, m.L2, m.ft_width) == (ver, nf, l1, l1, 128)
    t = m.export_tensors()
    W = 128
    shapes = {"ft_weight": (nf, W), "ft_bias": (W,), "l1_weight": (l1, 2 * W), "l1_bias": (l1,),
              "l2_weight": (l1, l1), "l2_bias": (l1,), "out_weight": (1, l1), "out_bias": (1,)}
    assert set(t) == set(NAMES)
    for k, s in shapes.items():
        assert tuple(t[k].shape) == s, k
        assert t[k].dtype == torch.float32
    assert tuple(networks.build(name, ft_width=64).export_tensors()["l1_weight"].shape) == (l1, 128)


def random_tensors(m, seed):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, v in m.export_tensors().items():
        lim = 1.0 if k.startswith("ft") else (LIMIT if k.endswith("weight") else 0.5)
        out[k] = (torch.rand(v.shape, generator=g) * 2 - 1) * lim * (0.05 if k == "ft_weight" else 1.0)
    return out


@pytest.mark.parametrize("name", ["RayPair_v0Network", "RayPair_v1Network"])
def test_forward_matches_formula(name):
    m = networks.build(name)
    w = random_tensors(m, 3)
    m.set_from_tensors(w)
    m.eval()
    v = m.FEATURE_VERSION
    sfens = [b.sfen() for b in random_boards(6, seed=99)] + [
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        "4k4/9/9/9/9/9/3p5/9/2N1K4 w 2Pg 1"]
    si, so, ni, no, expect = [], [], [], [], []
    np_w = {k: t.double().numpy() for k, t in w.items()}
    for s in sfens:
        stm = 0 if s.split()[1] == "b" else 1
        fs, fn = ref.features(s, stm, v), ref.features(s, 1 - stm, v)
        so.append(len(si)); si += fs
        no.append(len(ni)); ni += fn
        acc = lambda f: np_w["ft_bias"] + np_w["ft_weight"][f].sum(axis=0)  # 重複は回数ぶん加算される
        x = np.clip(np.concatenate([acc(fs), acc(fn)]), 0, 1)
        h1 = np.clip(np_w["l1_weight"] @ x + np_w["l1_bias"], 0, 1)
        h2 = np.clip(np_w["l2_weight"] @ h1 + np_w["l2_bias"], 0, 1)
        expect.append(float((np_w["out_weight"] @ h2 + np_w["out_bias"])[0]))
    t = lambda a: torch.tensor(a, dtype=torch.int64)
    with torch.no_grad():
        z = m(t(si), t(so), t(ni), t(no))
    assert tuple(z.shape) == (len(sfens),)
    assert np.allclose(z.double().numpy(), expect, atol=1e-4), (z, expect)
    # export → set の往復で値が変わらない
    back = m.export_tensors()
    for k in NAMES:
        assert torch.equal(back[k], w[k].float()), k


# ---------- 部品 ----------

def test_clip_weights():
    m = networks.build("RayPair_v0Network")
    m.set_from_tensors({k: torch.full_like(v, 5.0) for k, v in m.export_tensors().items()})
    train.clip_weights_(m)
    t = m.export_tensors()
    assert t["ft_weight"].max() == 1.0 and t["ft_bias"].max() == 1.0
    for k in ("l1_weight", "l2_weight", "out_weight"):
        assert abs(t[k].max().item() - LIMIT) < 1e-6, k
    for k in ("l1_bias", "l2_bias", "out_bias"):
        assert t[k].max().item() == 5.0, k
    m.set_from_tensors({k: torch.full_like(v, -5.0) for k, v in m.export_tensors().items()})
    train.clip_weights_(m)
    t = m.export_tensors()
    assert t["ft_weight"].min() == -1.0 and abs(t["l1_weight"].min().item() + LIMIT) < 1e-6


def test_compute_loss():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(100, generator=g)
    score = torch.randint(-3000, 3000, (100,), generator=g).float()
    result = torch.randint(0, 3, (100,), generator=g).float() / 2
    for lam in (1.0, 0.75, 0.0):
        t = lam * torch.sigmoid(score / 700.0) + (1 - lam) * result
        exp = torch.nn.functional.binary_cross_entropy_with_logits(z, t)
        got = train.compute_loss(z, score, result, 700.0, lam)
        assert got.dim() == 0 and abs(got.item() - exp.item()) < 1e-6


def test_optimizer_groups():
    m = networks.build("RayPair_v0Network")
    opt = train.build_optimizer(m, 1e-3, 0.01)
    assert isinstance(opt, torch.optim.AdamW)
    decay_shapes = sorted(tuple(p.shape) for g in opt.param_groups if g["weight_decay"] > 0 for p in g["params"])
    assert decay_shapes == sorted([(16, 256), (16, 16), (1, 16)])
    n_all = sum(len(g["params"]) for g in opt.param_groups)
    assert n_all == len(list(m.parameters()))
    assert all(g["lr"] == 1e-3 for g in opt.param_groups)


def test_lr_factor():
    f = train.lr_factor
    assert abs(f(0, 10, 110, 0.01) - 0.1) < 1e-12
    assert abs(f(9, 10, 110, 0.01) - 1.0) < 1e-12
    assert abs(f(10, 10, 110, 0.01) - 1.0) < 1e-12
    assert abs(f(60, 10, 110, 0.01) - (0.01 + 0.99 * 0.5)) < 1e-12
    assert abs(f(110, 10, 110, 0.01) - 0.01) < 1e-12
    assert abs(f(500, 10, 110, 0.01) - 0.01) < 1e-12
    assert abs(f(0, 0, 100, 0.0) - 1.0) < 1e-12


# ---------- 学習の通し ----------

def test_training_learns_material(mat_paths, tmp_path):
    out = tmp_path / "out"
    cfg = make_config(tmp_path / "c.toml", mat_paths, out, train__lr=3e-3, train__max_steps=120,
                      train__total_steps=120, train__warmup_steps=5, data__val_path=mat_paths[1],
                      train__val_every_steps=40, train__save_every_steps=40)
    assert run(["--config", cfg]) == 0
    log = read_log(str(out))
    assert log[0]["type"] == "start" and log[0]["eval_scale"] == 600.0 and log[0]["eval_scale_source"] == "config"
    tr = [r for r in log if r["type"] == "train"]
    assert tr[-1]["step"] == 120
    first = sum(r["loss"] for r in tr[:10]) / 10
    last = sum(r["loss"] for r in tr[-10:]) / 10
    assert last < first * 0.9, (first, last)
    vals = [r for r in log if r["type"] == "val"]
    assert [r["step"] for r in vals] == [40, 80, 120]
    assert vals[-1]["val_mse"] < vals[0]["val_mse"]
    ck = load_ckpt(str(out))
    for k in ("model", "optimizer", "step", "epoch", "loader_state", "eval_scale", "lam", "network", "config", "rng"):
        assert k in ck, k
    assert ck["step"] == 120 and ck["epoch"] >= 4 and ck["network"] == "RayPair_v0Network"
    # 学習後の重みはクリップの範囲内
    m = networks.build("RayPair_v0Network")
    m.load_state_dict(ck["model"])
    t = m.export_tensors()
    assert t["ft_weight"].abs().max() <= 1.0 and t["l1_weight"].abs().max() <= LIMIT + 1e-6


def test_resume_is_bitwise_identical(mat_paths, tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    common = dict(train__max_steps=40, train__total_steps=40, train__save_every_steps=7)
    assert run(["--config", make_config(tmp_path / "a.toml", mat_paths, a, **common)]) == 0
    cfg_b1 = make_config(tmp_path / "b1.toml", mat_paths, b, **{**common, "train__max_steps": 23})
    assert run(["--config", cfg_b1]) == 0
    assert load_ckpt(str(b))["step"] == 23
    cfg_b2 = make_config(tmp_path / "b2.toml", mat_paths, b, **common)
    assert run(["--config", cfg_b2, "--resume"]) == 0
    ca, cb = load_ckpt(str(a)), load_ckpt(str(b))
    assert cb["step"] == 40
    for k in ca["model"]:
        assert torch.equal(ca["model"][k], cb["model"][k]), f"再開後の重みが一致しない: {k}"


def test_v1_smoke(mat_paths, tmp_path):
    out = tmp_path / "v1"
    cfg = make_config(tmp_path / "v1.toml", mat_paths, out, model__network="RayPair_v1Network",
                      train__max_steps=8, train__total_steps=8)
    assert run(["--config", cfg]) == 0
    assert load_ckpt(str(out))["network"] == "RayPair_v1Network"


# ---------- eval_scale と勝敗欠損 ----------

def test_eval_scale_auto(tmp_path):
    p = str(tmp_path / "scale.psv")
    make_scale_dataset(p, 800.0)
    out = tmp_path / "auto"
    cfg = make_config(tmp_path / "auto.toml", [p], out, train__eval_scale="auto", train__lam=0.75,
                      train__max_steps=3, train__total_steps=3)
    assert run(["--config", cfg]) == 0
    start = read_log(str(out))[0]
    assert start["eval_scale_source"] == "auto"
    assert abs(start["eval_scale"] - 800.0) / 800.0 < 0.05
    assert abs(load_ckpt(str(out))["eval_scale"] - start["eval_scale"]) < 1e-6


def test_missing_results(tmp_path):
    p = str(tmp_path / "nores.psv")
    make_scale_dataset(p, 600.0, n=20_000, draw_rate=1.0, n_mate=0)
    out1 = tmp_path / "m1"
    rc = run(["--config", make_config(tmp_path / "m1.toml", [p], out1, train__lam=0.75,
                                      train__max_steps=3, train__total_steps=3)])
    assert rc != 0, "勝敗が欠損しているのに学習が始まった"
    assert not os.path.exists(os.path.join(str(out1), "checkpoint.pt"))
    out2 = tmp_path / "m2"
    rc = run(["--config", make_config(tmp_path / "m2.toml", [p], out2, train__lam=0.75,
                                      train__allow_missing_results=True, train__max_steps=3, train__total_steps=3)])
    assert rc != 0, "勝敗欠損で lam != 1.0 なのに学習が始まった"
    out3 = tmp_path / "m3"
    rc = run(["--config", make_config(tmp_path / "m3.toml", [p], out3, train__lam=1.0,
                                      train__allow_missing_results=True, train__max_steps=3, train__total_steps=3)])
    assert rc == 0
    assert load_ckpt(str(out3))["step"] == 3
