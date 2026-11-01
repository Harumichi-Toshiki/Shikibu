# 変更禁止（PROTECTED）。Shikibu M8（学習の高速化）の受け入れテスト。
#
# Colab（L4）での測定（設計担当の診断、2026-11-01）：
#   halfkp_256x2-32-32_f：GPU の計算だけ 85万局面/秒、全部込み 78万。因数分解なしなら 139万 / 112万。
#   raypair0_128x2-16-16：読み込みだけ 123万、GPU の計算だけ 125万、全部込み 82万（読み込みと計算が重なっていない）。
#   fused AdamW で GPU の計算が 1.1 倍前後。
# M8 でやること：
#   1. 因数分解（_f）を「入力」ではなく「重み」の側で行う。
#      今：入力に「玉に関係ない特徴」N + (f mod P) を足す（特徴の数が倍、その行への足し込みが集中して遅い）。
#      M8：入力は因数分解なしの版（INPUT_VERSION）のまま。重みを W_eff[k×P + p] = W[k×P + p] + W[N + p] として計算し、
#          その W_eff から読む。数学的に今と同じ（出力も勾配も一致する）。checkpoint の形（ft.weight は N + P 行）と書き出しは変えない。
#      psqt つきの構造では、psqt_weight も同じように扱う。
#   2. 読み込み（Rust）と GPU の計算を重ねる（次のバッチの読み込みと GPU への転送を、今のステップの計算と並行に行う）。
#   3. CUDA のときは fused AdamW を使う（使えなければ従来どおり）。
#
# ■ API（shikibu.networks）
#   すべてのネットワークに属性 INPUT_VERSION を持たせる：学習のときにローダーに渡す特徴の版。
#     因数分解つき（版 3・5・7・9・11）なら 1 つ小さい版（2・4・6・8・10）、それ以外は FEATURE_VERSION と同じ。
#     従来の名前の HalfKPf_256Network も同じ（INPUT_VERSION = 2）。
#   forward(stm_indices, stm_offsets, nstm_indices, nstm_offsets, kings=None, base=False)
#     base=False：従来どおり、FEATURE_VERSION の特徴（因数分解の特徴を含む）を受け取る（M6・M7 のテストはこの形）。
#     base=True ：INPUT_VERSION の特徴を受け取る。出力は、同じ局面の FEATURE_VERSION の特徴で base=False としたときと一致する。
#   学習（shikibu.train）は、ローダーに INPUT_VERSION を渡し、学習と検証で base=True を使う。
#
# ■ 速さは Colab で持ち主が測る（tools/shk_speedcheck.py は設計担当の診断の道具。M8 で train.py の実際の1ステップを使う形に直してよい）。
import os
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
from conftest import random_boards  # noqa: E402
from m3_data import make_material_dataset  # noqa: E402
from test_m3_acceptance import load_ckpt, make_config, run  # noqa: E402

from shikibu import networks  # noqa: E402

ARCHS = ["halfkp_64x2-8-16_f", "halfkahm2_32x2-8-8_f", "halfka1_32x2-8-8_f_psqt", "halfka2_32x2-8-8_f_psqt_k3k3",
         "halfkahm1_64x2-16-8_f_k3k3"]


def feats(feature, board, persp, factorize):
    f = N.features(feature, board, persp)
    if factorize:
        nf, p = N.num_features(feature), N.plane_size(feature)
        f = sorted(f + [nf + (x % p) for x in f])
    return f


def batch(feature, boards, factorize):
    si, so, ni, no, kings = [], [], [], [], []
    for b in boards:
        stm = 0 if b.turn == cshogi.BLACK else 1
        so.append(len(si)); si += feats(feature, b, stm, factorize)
        no.append(len(ni)); ni += feats(feature, b, 1 - stm, factorize)
        kings.append(list(K.king_squares(b)))
    t = lambda a: torch.tensor(a, dtype=torch.int64)
    return t(si), t(so), t(ni), t(no), t(kings)


def test_input_version():
    for name, want in (("halfkp_256x2-32-32_f", 2), ("halfkp_256x2-32-32", 2), ("halfkahm2_512x2-16-32_f", 10),
                       ("halfka1_32x2-8-8_f_psqt", 4), ("raypair0_128x2-16-16", 0), ("raypair1k_32x2-8-8_psqt", 13)):
        assert networks.build(name).INPUT_VERSION == want, name
    assert networks.build("HalfKPf_256Network", ft_width=256).INPUT_VERSION == 2
    assert networks.build("RayPair_v0Network").INPUT_VERSION == 0


@pytest.mark.parametrize("arch", ARCHS)
def test_base_input_matches_factorized(arch):
    p = networks.parse_arch(arch)
    m = networks.build(arch)
    g = torch.Generator().manual_seed(3)
    w = {}
    for k, v in m.export_tensors().items():
        sc = 0.05 if k == "ft_weight" else (0.2 if k == "psqt_weight" else 0.5)
        w[k] = (torch.rand(v.shape, generator=g) * 2 - 1) * sc
    m.set_from_tensors(w)
    boards = random_boards(40, seed=8801) + [cshogi.Board("8k/9/9/9/9/9/9/9/K8 w - 1"),
                                             cshogi.Board("4k4/9/9/9/9/9/4g4/9/4K4 b g 1")]
    full = batch(p["feature"], boards, True)
    base = batch(p["feature"], boards, False)
    target = torch.rand(len(boards), generator=g)
    outs, grads = [], []
    for args, is_base in ((full, False), (base, True)):
        m.zero_grad(set_to_none=True)
        z = m(*args[:4], kings=args[4], base=is_base)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, target)
        loss.backward()
        outs.append(z.detach().double())
        grads.append({n: q.grad.detach().double().clone() for n, q in m.named_parameters() if q.grad is not None})
    assert torch.allclose(outs[0], outs[1], atol=1e-5), f"{arch}: base=True の出力が因数分解の入力と違う"
    assert set(grads[0]) == set(grads[1]), f"{arch}: 勾配のあるパラメータが違う"
    for n in grads[0]:
        assert torch.allclose(grads[0][n], grads[1][n], atol=1e-6, rtol=1e-4), f"{arch}: {n} の勾配が違う"


def test_legacy_halfkpf_base_input():
    m = networks.build("HalfKPf_256Network", ft_width=256)
    with torch.no_grad():
        m.ft.weight.uniform_(-0.05, 0.05)
    boards = random_boards(10, seed=8802)
    full, base = batch("halfkp", boards, True), batch("halfkp", boards, False)
    with torch.no_grad():
        a = m(*full[:4])
        b = m(*base[:4], base=True)
    assert torch.allclose(a, b, atol=1e-5)


@pytest.fixture(scope="module")
def mat(tmp_path_factory):
    d = tmp_path_factory.mktemp("m8mat")
    return make_material_dataset(str(d), n_files=2, per_file=1200, seed=88)


def test_training_uses_base_input_and_learns(mat, tmp_path):
    arch = "halfkahm2_32x2-8-8_f"
    cfg = make_config(tmp_path / "c.toml", mat, tmp_path / "o", model__network=arch, model__ft_width=0,
                      train__max_steps=40, train__total_steps=40, train__lr=3e-3)
    assert run(["--config", cfg]) == 0
    ck = load_ckpt(str(tmp_path / "o"))
    assert ck["network"] == arch and ck["step"] == 40
    assert tuple(ck["model"]["ft.weight"].shape) == (73305 + 1629, 32), "checkpoint の形を変えてはいけない"
    import json
    log = [json.loads(x) for x in open(tmp_path / "o" / "log.jsonl", encoding="utf-8") if x.strip()]
    losses = [r["loss"] for r in log if r.get("type") == "train"]
    assert np.mean(losses[-10:]) < np.mean(losses[:10])


def test_deterministic_runs_identical(mat, tmp_path):
    """読み込みと計算を重ねても、決定的モードでは同じ設定の2回の学習が完全に一致する。"""
    arch = "halfkp_32x2-8-8_f_psqt"
    res = []
    for tag in ("a", "b"):
        cfg = make_config(tmp_path / f"{tag}.toml", mat, tmp_path / tag, model__network=arch, model__ft_width=0,
                          train__max_steps=25, train__total_steps=25)
        assert run(["--config", cfg]) == 0
        res.append(load_ckpt(str(tmp_path / tag)))
    for k in res[0]["model"]:
        assert torch.equal(res[0]["model"][k], res[1]["model"][k]), f"同じ設定の学習の結果が違う: {k}"
