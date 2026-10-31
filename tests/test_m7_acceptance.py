# 変更禁止（PROTECTED）。Shikibu M7（評価関数の構造の実験室）の受け入れテスト。設計書 §S12。
#
# 目的：評価関数の構造（特徴・幅・層）を、設定の1行で切り替えて学習し、検証の損失（val_loss）で比べられるようにする。
# やねうら王と同じ構造は nn.bin に書き出して、やねうら王・Nuenagon でそのまま使えるようにする。
#
# ■ 構造の名前（model.network に書く。従来の名前 RayPair_v0Network・HalfKPf_256Network なども引き続き使える）
#     <特徴>_<W>x2-<L1>-<L2>[_<オプション>…]
#   特徴：raypair0・raypair1（RayPair_v0・v1）、raypair0k・raypair1k（RayPair ＋ 玉の周りのブロック、tests/kingzone_ref.py）、
#         halfkp・halfka1・halfka2・halfkahm1・halfkahm2（やねうら王と同じ。定義は tests/nnue_ref.py）
#   W は 32 の倍数（32〜8192）、L1・L2 は 8 の倍数（8〜1024）。
#   オプション（どの順に書いてもよい。同じものを2回書くのは誤り）：
#     f    ：因数分解つき（halfk… だけ）。学習のときだけ「玉に関係ない特徴」を足し、書き出すときに畳み込む。
#     psqt ：駒の点数を直接足す項（PSQT）。
#     k3k3 ：層を 9 組持ち、両方の玉の段（自陣・中段・敵陣）で使う組を選ぶ（layer stack）。
#   正規の形：すべて小文字、オプションは f → psqt → k3k3 の順。networks.parse_arch がこれを返し、
#   checkpoint の network にはこの正規の形を書く。誤った名前は ValueError（日本語の説明つき）。
#
# ■ 特徴の版（shikibu_feat の version）と次元
#   0 raypair0 15000、1 raypair1 115352、2 halfkp 125388、3 halfkp+f 126936、
#   4 halfka1 138510、5 halfka1+f 140220、6 halfka2 131949、7 halfka2+f 133578、
#   8 halfkahm1 76950、9 halfkahm1+f 78660、10 halfkahm2 73305、11 halfkahm2+f 74934、
#   12 raypair0k 16392、13 raypair1k 116744。
#   因数分解つき（+f）は、元の各特徴 f に「N + (f mod P)」を足したもの（N は元の次元、P は 1つの玉の位置あたりの駒の番号の数：
#   halfkp 1548、halfka1・halfkahm1 1710、halfka2・halfkahm2 1629）。いずれも昇順・重複込み。
#   shikibu_feat.king_squares_sfen(sfen) -> (手番側の玉のマス（手番側の視点）, 相手の玉のマス（相手の視点）)。
#     マスは RayPair の視点座標の sq = y × 9 + x（設計書 §F1.2）。kingzone_ref.king_squares と同じ。
#   BatchLoader のバッチには、どの版でも "kings"（int64、形 (局面数, 2)、king_squares_sfen と同じ値）を加える。
#
# ■ ネットワーク（shikibu.networks）
#   build(名前, ft_width=128)：従来の名前ならそのクラス、構造の名前なら構造に従う（W は名前の値。ft_width は使わない）。
#   parse_arch(名前) -> dict（キー：canonical, feature, W, L1, L2, factorize, psqt, stacks（1 か 9）, version, num_features）。
#   arch_of(名前, ft_width=128) -> 正規の構造の名前（従来の名前も変換する。例：HalfKPf_256Network → halfkp_256x2-32-32_f）。
#   モデルの属性：FEATURE_VERSION、NUM_FEATURES、ft_width、L1、L2、PSQT（bool）、STACKS（1 か 9）、NEEDS_KINGS（k3k3 なら True）。
#   forward(stm_indices, stm_offsets, nstm_indices, nstm_offsets, kings=None)。kings は (局面数, 2) の int64。
#   export_tensors() の名前と形（S = STACKS、N = NUM_FEATURES）：
#     ft_weight (N, W)、ft_bias (W)
#     S = 1：l1_weight (L1, 2W)、l1_bias (L1)、l2_weight (L2, L1)、l2_bias (L2)、out_weight (1, L2)、out_bias (1)（従来と同じ）
#     S = 9：l1_weight (9, L1, 2W)、l1_bias (9, L1)、l2_weight (9, L2, L1)、l2_bias (9, L2)、out_weight (9, 1, L2)、out_bias (9, 1)
#     psqt のとき：psqt_weight (N, S)
#   計算式：従来の §F3.2 の式を、組 b の重みで行う。b = 0（S = 1）、または kingzone_ref.k3k3_bucket と同じ
#     b = zone(手番側の玉) × 3 + zone(相手の玉)（zone：その玉の側の視点で y ≧ 6 なら 0、3〜5 なら 1、≦ 2 なら 2）。
#     psqt のとき z に (P_手番側[b] − P_相手側[b]) / 2 を足す。P_q[b] = Σ_{i ∈ F_q（重複込み）} psqt_weight[i, b]。
#   重みのクリップ（§F3.3）はすべての組に行う。psqt_weight はクリップしない。
#
# ■ 書き出し（shikibu.export）
#   export_nnbin：halfk… で psqt・k3k3 なしの構造なら、どの幅でも nn.bin を書く（tests/nnue_nnbin.py の並び。
#     切り上げで増えた列は 0）。量子化の式は M6 と同じ（因数分解は全部の玉の位置へ畳み込む）。
#     戻り値の dict に network（正規の名前）, arch（やねうら王の構造の名前。例 halfkahm2_512x2-16-32）, eval_scale, fv_scale,
#     yo_edition（"YANEURAOU_ENGINE_NNUE_" + arch を大文字にして "-" を "_" にしたもの）を入れる。
#     CLI は "FV_SCALE=<整数>" と "EDITION=<yo_edition>" をそれぞれ含む行を出す。
#   export_shk：raypair0・raypair1 で psqt・k3k3 なしの構造なら、どの幅でも .shk を書く（ヘッダの W・L1・L2 はその値、
#     network は正規の名前、feature_set は従来どおり RayPair_v0 / RayPair_v1）。
#   それ以外（psqt・k3k3・raypair…k、形式の合わない組み合わせ）は、どちらも ValueError（「書き出せない」理由を日本語で）。
#     CLI は終了コード 1 で、標準エラーに理由を出す。
#
# ■ 実験室（shikibu.lab）
#   PRESETS：{構造の名前: 日本語の説明} の dict（ノートブックの選択肢）。
#   describe(名前) -> dict：canonical, feature, W, L1, L2, train_features（学習のときの次元）, infer_features（因数分解を畳んだ後）,
#     params（学習する重みの数）, max_active（1局面1視点の特徴の最大数：halfkp 38、halfka… 40、raypair 208、raypair…k 256）,
#     export_format（"nn.bin" / ".shk" / None）, yo_compatible（bool）, yo_arch・yo_edition（やねうら王互換なら文字列、違えば None）,
#     nnbin_bytes（nn.bin なら書き出したときの大きさ、違えば None）。
#   make_config(名前, train_glob, out_dir, val_path="", overrides=None) -> 設定の dict（train.load_config と同じ形）。
#     overrides は {"train.max_steps": 1000} のような「セクション.キー」の dict。model.network は正規の名前、
#     model.ft_width は W にする。他は既定値（train.DEFAULTS）。
#   write_config(cfg, path)：TOML に書く（train.load_config で読めること）。
#   compare_runs(run_dirs) -> list[dict]：各 run の log.jsonl と checkpoint.pt から、キー run_dir, network, step, positions,
#     eval_scale, val_loss（最後の検証）, best_val_loss, common_step（全 run で検証がある最大の step）, val_loss_at_common。
#     val_loss_at_common の小さい順。
#   export_run(run_dir) -> dict：run_dir/export/ に nn.bin（halfk…）か nuenagon.shk（raypair）を書き、
#     {"format", "path", …（export_nnbin・export_shk の戻り値）} を返す。書き出せない構造なら例外にせず
#     {"format": None, "path": None, "reason": 日本語の理由} を返す。
#
# ■ 検証データ：data.val_path に指定したファイルは、train_glob・train_paths に含まれていても学習には使わない
#   （train.train_paths が除く。比べるのは os.path.realpath）。
#
# ■ ノートブック notebooks/shikibu_lab.ipynb と、設定のひな形 configs/colab_lab.toml（§S12.6）。
import ast
import glob
import hashlib
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
import kingzone_ref as K  # noqa: E402
import nnue_nnbin as NB  # noqa: E402
import nnue_ref as N  # noqa: E402
from conftest import make_psv, random_boards, ref  # noqa: E402
from m2_data import write_records  # noqa: E402
from m3_data import make_material_dataset  # noqa: E402
from test_m3_acceptance import load_ckpt, make_config, read_log, run  # noqa: E402

from shikibu import networks, train  # noqa: E402

LIMIT = 127.0 / 64.0
SPARSE = ["4k4/9/4g4/9/9/9/9/4R4/4K4 b - 1", "4k4/9/9/9/9/9/4g4/9/4K4 b g 1", "4k4/9/9/9/9/9/9/4r4/4K4 b - 1",
          "1+Psnkp3/9/4P2+P1/4L4/9/9/6K2/9/9 b 2G 1", "9/9/k8/N1B6/R+P7/3G5/4K4/9/9 b 2GB 1",
          "8k/9/9/9/9/9/9/9/K8 w - 1", "k8/9/9/9/9/9/9/9/8K b - 1", "4k4/9/9/9/9/9/9/9/3K5 b - 1",
          "3k5/9/9/9/9/9/9/9/5K3 w 2r2b4g4s4n4l18p 1"]
# (特徴, 因数分解) → 版
VERSIONS = {("raypair0", False): 0, ("raypair1", False): 1, ("halfkp", False): 2, ("halfkp", True): 3,
            ("halfka1", False): 4, ("halfka1", True): 5, ("halfka2", False): 6, ("halfka2", True): 7,
            ("halfkahm1", False): 8, ("halfkahm1", True): 9, ("halfkahm2", False): 10, ("halfkahm2", True): 11,
            ("raypair0k", False): 12, ("raypair1k", False): 13}
DIMS = {0: 15000, 1: 115352, 2: 125388, 3: 126936, 4: 138510, 5: 140220, 6: 131949, 7: 133578,
        8: 76950, 9: 78660, 10: 73305, 11: 74934, 12: 16392, 13: 116744}
HALFK = ["halfkp", "halfka1", "halfka2", "halfkahm1", "halfkahm2"]


def ref_features(feature, factorize, board, persp):
    """参照実装による特徴（昇順・重複込み）。"""
    s = board.sfen()
    if feature in ("raypair0", "raypair1"):
        return sorted(ref.features(s, persp, int(feature[-1])))
    if feature in ("raypair0k", "raypair1k"):
        v = int(feature[7])
        return sorted(list(ref.features(s, persp, v)) + K.zone_features(board, persp, K.BASE[12 + v]))
    f = N.features(feature, board, persp)
    if factorize:
        nf, p = N.num_features(feature), N.plane_size(feature)
        f = sorted(f + [nf + (x % p) for x in f])
    return f


def boards_for_features(n, seed):
    return random_boards(n, seed=seed) + [cshogi.Board(s) for s in SPARSE]


# ================================================================ 特徴

def test_num_features(feat):
    for v, d in DIMS.items():
        assert feat.num_features(v) == d, f"版 {v} の次元が違う"
    with pytest.raises(ValueError):
        feat.num_features(14)


@pytest.mark.parametrize("feature", HALFK[1:])
def test_extract_halfka(feat, feature):
    boards = boards_for_features(300, seed=7001 + HALFK.index(feature))
    v0, v1 = VERSIONS[(feature, False)], VERSIONS[(feature, True)]
    mirrored = 0
    for b in boards:
        s = b.sfen()
        for p in (0, 1):
            want = ref_features(feature, False, b, p)
            assert list(feat.extract_sfen(s, p, v0)) == want, f"{feature}（版 {v0}）の特徴が参照実装と違う（視点 {p}）: {s}"
            assert list(feat.extract_sfen(s, p, v1)) == ref_features(feature, True, b, p), \
                f"{feature}（版 {v1}、因数分解つき）の特徴が違う（視点 {p}）: {s}"
            _, k = N.bona_pieces(b, p)
            mirrored += (k // 9 >= 5)
    if "hm" in feature:
        assert mirrored > 50  # 左右反転する局面を十分に含んでいること（テスト自体の確認）


@pytest.mark.parametrize("version", [12, 13])
def test_extract_kingzone(feat, version):
    feature = "raypair0k" if version == 12 else "raypair1k"
    for b in boards_for_features(300, seed=7101 + version):
        s = b.sfen()
        for p in (0, 1):
            want = ref_features(feature, False, b, p)
            got = list(feat.extract_sfen(s, p, version))
            assert got == want, f"{feature} の特徴が参照実装と違う（視点 {p}）: {s}"
            assert len(got) <= 256 and max(got) < DIMS[version]
    # 手計算の例：両方の玉が中央の筋の端の段にいて、ほかに駒がない。区間 C は盤の外の 10 マスずつ。
    s = "4k4/9/9/9/9/9/9/9/4K4 b - 1"
    base = K.BASE[version]
    c = sorted([base + 0 * 696 + o * 29 + 28 for o in range(14, 24)] + [base + 1 * 696 + o * 29 + 28 for o in range(0, 10)])
    v = version - 12
    assert list(feat.extract_sfen(s, 0, version)) == sorted(list(ref.features(s, 0, v)) + c)


def test_king_squares(feat):
    for b in boards_for_features(300, seed=7201):
        got = tuple(feat.king_squares_sfen(b.sfen()))
        assert got == K.king_squares(b), f"king_squares_sfen が違う: {b.sfen()}"


@pytest.mark.parametrize("version", [7, 10, 12])
def test_loader_new_versions(feat, tmp_path, version):
    feature, factorize = [k for k, v in VERSIONS.items() if v == version][0]
    boards = boards_for_features(250, seed=7301 + version)
    recs = [make_psv(b, 10 * i, i, 50, 1 if i % 2 else -1) for i, b in enumerate(boards)]
    p = str(tmp_path / "x.psv")
    write_records(p, recs)
    got = {"stm": [], "nstm": [], "kings": []}
    for batch in feat.BatchLoader([p], 64, version, shuffle_buffer=0, num_threads=2):
        n = len(batch["score"])
        kings = np.asarray(batch["kings"])
        assert kings.shape == (n, 2) and kings.dtype == np.int64, "バッチの kings の形・型が違う"
        got["kings"] += [tuple(int(x) for x in r) for r in kings]
        for side in ("stm", "nstm"):
            idx, off = batch[f"{side}_indices"], batch[f"{side}_offsets"]
            ends = list(off[1:]) + [len(idx)]
            got[side] += [list(idx[off[i]:ends[i]]) for i in range(n)]
    assert len(got["stm"]) == len(boards)
    for i, b in enumerate(boards):
        stm = 0 if b.turn == cshogi.BLACK else 1
        assert got["stm"][i] == ref_features(feature, factorize, b, stm), f"版 {version} の stm の特徴が違う: {b.sfen()}"
        assert got["nstm"][i] == ref_features(feature, factorize, b, 1 - stm), f"版 {version} の nstm の特徴が違う"
        assert got["kings"][i] == K.king_squares(b), f"バッチの kings が違う: {b.sfen()}"


def test_reference_matches_yaneuraou(feat, tmp_path):
    """やねうら王の静的評価値（正解データ）と、Rust の特徴 ＋ 参照実装の推論が1点も違わないこと。"""
    g = json.load(open(os.path.join(HERE, "golden", "nnue_golden.json"), encoding="utf-8"))
    sfens = g["positions"]
    for item in g["files"]:
        arch = item["arch"]
        path = str(tmp_path / (arch + ".bin"))
        NB.make(path, arch, item["seed"], item["scale"])
        assert hashlib.sha256(open(path, "rb").read()).hexdigest() == item["sha256"], "nnue_nnbin.make の出力が変わった"
        net = N.Net(path, arch)
        version = VERSIONS[(net.feature, False)]
        for s, want in zip(sfens, item["cp"]):
            b = cshogi.Board(s)
            stm = 0 if b.turn == cshogi.BLACK else 1
            acc = {p: net.accumulate(list(feat.extract_sfen(s, p, version))) for p in (0, 1)}
            x = np.concatenate([np.clip(acc[stm], 0, 127), np.clip(acc[1 - stm], 0, 127)])
            h1 = np.clip((net.b1 + net.w1 @ x) >> 6, 0, 127)
            h2 = np.clip((net.b2 + net.w2 @ h1) >> 6, 0, 127)
            raw = int((net.bo + net.wo @ h2)[0])
            cp = abs(raw) // 16 * (1 if raw >= 0 else -1)
            assert cp == want, f"{arch}: やねうら王の評価値と違う（{cp} / {want}）: {s}"
        os.remove(path)


# ================================================================ ネットワーク

def test_parse_arch():
    p = networks.parse_arch("HalfKAhm2_512X2-16-32_k3k3_F")
    assert p["canonical"] == "halfkahm2_512x2-16-32_f_k3k3"
    assert (p["feature"], p["W"], p["L1"], p["L2"], p["factorize"], p["psqt"], p["stacks"]) == \
        ("halfkahm2", 512, 16, 32, True, False, 9)
    assert p["version"] == 11 and p["num_features"] == 74934
    p = networks.parse_arch("raypair0k_256x2-32-32_psqt")
    assert (p["version"], p["num_features"], p["psqt"], p["stacks"], p["factorize"]) == (12, 16392, True, 1, False)
    assert networks.parse_arch("halfkp_256x2-32-32")["canonical"] == "halfkp_256x2-32-32"
    for bad in ["halfkp_250x2-32-32", "halfkp_256x2-30-32", "halfkp_256x2-32-12", "raypair0_128x2-16-16_f",
                "foo_256x2-32-32", "halfkp_256x3-32-32", "halfkp_256x2-32-32_xyz", "halfkp_256x2-32-32_f_f",
                "halfkp_256-32-32", "halfkp_0x2-32-32", "halfkp_16384x2-32-32", "HalfKP_256Network"]:
        with pytest.raises(ValueError):
            networks.parse_arch(bad)


def test_arch_of_and_legacy():
    assert networks.arch_of("RayPair_v0Network", 128) == "raypair0_128x2-16-16"
    assert networks.arch_of("RayPair_v1Network", 256) == "raypair1_256x2-32-32"
    assert networks.arch_of("HalfKP_256Network", 256) == "halfkp_256x2-32-32"
    assert networks.arch_of("HalfKPf_256Network", 256) == "halfkp_256x2-32-32_f"
    assert networks.arch_of("HALFKP_256x2-32-32_F") == "halfkp_256x2-32-32_f"
    # 従来のクラスと同じ構造なら、重みをそのまま移して同じ値を出す
    old = networks.build("HalfKPf_256Network", ft_width=256)
    new = networks.build("halfkp_256x2-32-32_f")
    assert type(new) is not None and new.NUM_FEATURES == old.NUM_FEATURES and new.ft_width == 256
    g = torch.Generator().manual_seed(5)
    w = {k: (torch.rand(v.shape, generator=g) * 2 - 1) * (0.05 if k == "ft_weight" else 0.5)
         for k, v in old.export_tensors().items()}
    old.set_from_tensors(w)
    new.set_from_tensors(old.export_tensors())
    b = cshogi.Board("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
    fs = torch.tensor(ref_features("halfkp", True, b, 0))
    fn = torch.tensor(ref_features("halfkp", True, b, 1))
    o = torch.tensor([0])
    with torch.no_grad():
        assert torch.allclose(old(fs, o, fn, o), new(fs, o, fn, o), atol=1e-6)


ARCH_SHAPES = ["raypair0_64x2-16-8", "halfkp_64x2-8-16_f_psqt", "halfkahm2_32x2-8-8_k3k3",
               "halfka2_64x2-16-8_f_psqt_k3k3", "raypair1k_32x2-8-8_psqt"]


@pytest.mark.parametrize("arch", ARCH_SHAPES)
def test_build_attributes_and_shapes(arch):
    p = networks.parse_arch(arch)
    m = networks.build(arch)
    assert (m.FEATURE_VERSION, m.NUM_FEATURES, m.ft_width, m.L1, m.L2) == \
        (p["version"], p["num_features"], p["W"], p["L1"], p["L2"])
    assert (bool(m.PSQT), m.STACKS, bool(m.NEEDS_KINGS)) == (p["psqt"], p["stacks"], p["stacks"] == 9)
    t = m.export_tensors()
    Nf, W, L1, L2, S = p["num_features"], p["W"], p["L1"], p["L2"], p["stacks"]
    want = {"ft_weight": (Nf, W), "ft_bias": (W,)}
    if S == 1:
        want.update({"l1_weight": (L1, 2 * W), "l1_bias": (L1,), "l2_weight": (L2, L1), "l2_bias": (L2,),
                     "out_weight": (1, L2), "out_bias": (1,)})
    else:
        want.update({"l1_weight": (S, L1, 2 * W), "l1_bias": (S, L1), "l2_weight": (S, L2, L1), "l2_bias": (S, L2),
                     "out_weight": (S, 1, L2), "out_bias": (S, 1)})
    if p["psqt"]:
        want["psqt_weight"] = (Nf, S)
    assert {k: tuple(v.shape) for k, v in t.items()} == want
    assert all(v.dtype == torch.float32 for v in t.values())


def _random_tensors(m, seed):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, v in m.export_tensors().items():
        lim = 1.0 if k.startswith("ft") else (LIMIT if k.endswith("weight") and k != "psqt_weight" else 0.5)
        sc = 0.05 if k == "ft_weight" else (0.2 if k == "psqt_weight" else 1.0)
        out[k] = (torch.rand(v.shape, generator=g) * 2 - 1) * lim * sc
    return out


@pytest.mark.parametrize("arch", ARCH_SHAPES)
def test_forward_matches_formula(arch):
    p = networks.parse_arch(arch)
    m = networks.build(arch)
    w = _random_tensors(m, 17)
    m.set_from_tensors(w)
    m.eval()
    npw = {k: v.double().numpy() for k, v in w.items()}
    boards = random_boards(24, seed=7401) + [cshogi.Board(s) for s in SPARSE]
    si, so, ni, no, kings, expect = [], [], [], [], [], []
    buckets = set()
    for b in boards:
        stm = 0 if b.turn == cshogi.BLACK else 1
        fs = ref_features(p["feature"], p["factorize"], b, stm)
        fn = ref_features(p["feature"], p["factorize"], b, 1 - stm)
        so.append(len(si)); si += fs
        no.append(len(ni)); ni += fn
        kings.append(list(K.king_squares(b)))
        bk = K.k3k3_bucket(b) if p["stacks"] == 9 else 0
        buckets.add(bk)
        pick = (lambda a: a[bk]) if p["stacks"] == 9 else (lambda a: a)
        acc = lambda f: npw["ft_bias"] + npw["ft_weight"][f].sum(axis=0)
        x = np.clip(np.concatenate([acc(fs), acc(fn)]), 0, 1)
        h1 = np.clip(pick(npw["l1_weight"]) @ x + pick(npw["l1_bias"]), 0, 1)
        h2 = np.clip(pick(npw["l2_weight"]) @ h1 + pick(npw["l2_bias"]), 0, 1)
        z = float((pick(npw["out_weight"]) @ h2 + pick(npw["out_bias"]))[0])
        if p["psqt"]:
            z += (npw["psqt_weight"][fs, bk].sum() - npw["psqt_weight"][fn, bk].sum()) / 2
        expect.append(z)
    if p["stacks"] == 9:
        assert len(buckets) >= 3  # テスト自体の確認
    t = lambda a: torch.tensor(a, dtype=torch.int64)
    with torch.no_grad():
        z = m(t(si), t(so), t(ni), t(no), kings=t(kings))
    assert tuple(z.shape) == (len(boards),)
    assert np.allclose(z.double().numpy(), expect, atol=1e-4), f"{arch}: forward が式と違う"
    if p["stacks"] == 9:
        with pytest.raises(Exception):
            m(t(si), t(so), t(ni), t(no))  # kings が無いと計算できない
    back = m.export_tensors()
    for k in w:
        assert torch.equal(back[k], w[k].float()), k


def test_clip_all_stacks():
    m = networks.build("halfkp_32x2-8-8_psqt_k3k3")
    m.set_from_tensors({k: torch.full_like(v, 5.0) for k, v in m.export_tensors().items()})
    train.clip_weights_(m)
    t = m.export_tensors()
    assert t["ft_weight"].max() <= 1.0 and t["ft_bias"].max() <= 1.0
    for k in ("l1_weight", "l2_weight", "out_weight"):
        assert t[k].max() <= LIMIT + 1e-6, k
    for k in ("l1_bias", "out_bias", "psqt_weight"):
        assert float(t[k].min()) == 5.0, f"{k} はクリップしない"


# ================================================================ 学習

TRAIN_ARCHS = ["halfkahm2_64x2-16-32_f", "halfkp_64x2-8-32", "raypair0k_64x2-16-16_psqt_k3k3", "raypair0_64x2-8-8",
               "halfka1_32x2-8-8_f_psqt"]


@pytest.fixture(scope="module")
def mat(tmp_path_factory):
    d = tmp_path_factory.mktemp("m7mat")
    paths = make_material_dataset(str(d / "train"), n_files=2, per_file=1500, seed=77)
    val = make_material_dataset(str(d / "val"), n_files=1, per_file=1000, seed=78)[0]
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
        log = read_log(out)
        losses = [r["loss"] for r in log if r.get("type") == "train"]
        assert losses and all(math.isfinite(x) for x in losses), f"{arch} の損失が有限でない"
        assert np.mean(losses[-10:]) < np.mean(losses[:10]), f"{arch} の損失が下がらない"
        vals = [r for r in log if r.get("type") == "val"]
        assert [r["step"] for r in vals] == [20, 40, 60] and all(math.isfinite(r["val_loss"]) for r in vals)
        m = networks.build(arch)
        m.load_state_dict(ck["model"])
        t = m.export_tensors()
        assert t["ft_weight"].abs().max() <= 1.0
        for k in ("l1_weight", "l2_weight", "out_weight"):
            assert t[k].abs().max() <= LIMIT + 1e-6


def test_canonical_name_saved(mat, tmp_path):
    paths, _ = mat
    cfg = make_config(tmp_path / "c.toml", paths, tmp_path / "o", model__network="HalfKP_32X2-8-8_PSQT_F",
                      model__ft_width=32, train__max_steps=3, train__total_steps=3)
    assert run(["--config", cfg]) == 0
    assert load_ckpt(str(tmp_path / "o"))["network"] == "halfkp_32x2-8-8_f_psqt"
    # model.ft_width = 0 は「構造の名前に合わせる」
    cfg = make_config(tmp_path / "z.toml", paths, tmp_path / "z", model__network="halfkp_32x2-8-8",
                      model__ft_width=0, train__max_steps=3, train__total_steps=3)
    assert run(["--config", cfg]) == 0
    assert networks.build(load_ckpt(str(tmp_path / "z"))["network"]).ft_width == 32


def test_bad_arch_and_width_rejected(mat, tmp_path):
    paths, _ = mat
    cfg = make_config(tmp_path / "a.toml", paths, tmp_path / "a", model__network="halfkp_250x2-32-32",
                      train__max_steps=3, train__total_steps=3)
    assert run(["--config", cfg]) != 0
    assert not os.path.exists(tmp_path / "a" / "checkpoint.pt")
    cfg = make_config(tmp_path / "b.toml", paths, tmp_path / "b", model__network="halfkp_64x2-8-8",
                      model__ft_width=128, train__max_steps=3, train__total_steps=3)
    assert run(["--config", cfg]) != 0, "構造の名前の W と model.ft_width が違うときは止まること"


def test_resume_bitwise_stacks(mat, tmp_path):
    paths, _ = mat
    arch = "halfkahm2_32x2-8-8_f_psqt_k3k3"
    common = dict(model__network=arch, model__ft_width=32, train__max_steps=30, train__total_steps=30,
                  train__save_every_steps=7)
    assert run(["--config", make_config(tmp_path / "a.toml", paths, tmp_path / "a", **common)]) == 0
    assert run(["--config", make_config(tmp_path / "b1.toml", paths, tmp_path / "b",
                                        **{**common, "train__max_steps": 13})]) == 0
    assert run(["--config", make_config(tmp_path / "b2.toml", paths, tmp_path / "b", **common), "--resume"]) == 0
    ca, cb = load_ckpt(str(tmp_path / "a")), load_ckpt(str(tmp_path / "b"))
    assert cb["step"] == 30
    for k in ca["model"]:
        assert torch.equal(ca["model"][k], cb["model"][k]), f"再開後の重みが一致しない: {k}"


def test_val_holdout_excluded(mat, tmp_path):
    paths, val = mat
    d = {"train_glob": os.path.join(os.path.dirname(val), "*.psv"), "train_paths": [], "val_path": val}
    with pytest.raises(ValueError):
        train.train_paths(d)  # 検証のファイルしか無ければ、学習データが無い
    d = {"train_glob": "", "train_paths": list(paths) + [val], "val_path": val}
    assert [os.path.realpath(x) for x in train.train_paths(d)] == [os.path.realpath(x) for x in paths]
    both = tmp_path / "both"
    both.mkdir()
    for x in paths:
        os.symlink(x, both / os.path.basename(x))
    os.symlink(val, both / "val.psv")
    d = {"train_glob": str(both / "*.psv"), "train_paths": [], "val_path": val}  # 別の名前でも同じファイルなら除く
    got = [os.path.basename(x) for x in train.train_paths(d)]
    assert got == sorted(os.path.basename(x) for x in paths)


# ================================================================ 書き出し

def round_away(x):
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def q(x, scale, dtype):
    info = np.iinfo(dtype)
    return np.clip(round_away(np.asarray(x, dtype=np.float64) * scale), info.min, info.max).astype(np.int64)


def expected_nnbin(arch, t, eval_scale, fv):
    p = networks.parse_arch(arch)
    w = t["ft_weight"].double().numpy()
    nf = N.num_features(p["feature"])
    if p["factorize"]:
        P = N.plane_size(p["feature"])
        w = w[:nf] + np.tile(w[nf:], (nf // P, 1))
    s = eval_scale * fv / 8128.0
    d = lambda k: t[k].double().numpy()
    return {"ft_b": q(d("ft_bias"), 127, np.int16), "ft_w": q(w, 127, np.int16),
            "b1": q(d("l1_bias"), 8128, np.int32), "w1": q(d("l1_weight"), 64, np.int8),
            "b2": q(d("l2_bias"), 8128, np.int32), "w2": q(d("l2_weight"), 64, np.int8),
            "bo": q(d("out_bias") * s, 8128, np.int32), "wo": q(d("out_weight") * s, 64, np.int8)}


def yo_arch(arch):
    p = networks.parse_arch(arch)
    return f"{p['feature']}_{p['W']}x2-{p['L1']}-{p['L2']}"


@pytest.mark.parametrize("arch", ["halfkahm2_64x2-16-32_f", "halfkp_64x2-8-32"])
def test_export_nnbin_exact(runs, tmp_path, arch):
    from shikibu import export
    ckp = os.path.join(runs[arch], "checkpoint.pt")
    path = str(tmp_path / "nn.bin")
    info = export.export_nnbin(ckp, path)
    ck = torch.load(ckp, map_location="cpu", weights_only=False)
    fv = max(1, math.floor(8128 / float(ck["eval_scale"])))
    ya = yo_arch(arch)
    assert info["fv_scale"] == fv and info["network"] == arch and info["arch"] == ya
    assert info["yo_edition"] == "YANEURAOU_ENGINE_NNUE_" + ya.upper().replace("-", "_")
    assert os.path.getsize(path) == NB.file_size(ya)
    ver, h, s, fth = NB.read_header(path)
    ft_hash, net_hash, file_hash = NB.hashes(ya)
    assert (ver, h, s, fth) == (NB.VERSION, file_hash, NB.arch_string(ya), ft_hash)
    raw = open(path, "rb").read()
    n = len(s.encode())
    nf = N.num_features(networks.parse_arch(arch)["feature"])
    W = networks.parse_arch(arch)["W"]
    assert struct.unpack_from("<I", raw, 12 + n + 4 + 2 * W + 2 * W * nf)[0] == net_hash
    net = N.Net(path, ya)
    m = networks.build(arch)
    m.load_state_dict(ck["model"])
    exp = expected_nnbin(arch, m.export_tensors(), float(ck["eval_scale"]), fv)
    got = {"ft_b": net.ft_b, "ft_w": net.ft_w, "b1": net.b1, "w1": net.w1, "b2": net.b2, "w2": net.w2,
           "bo": net.bo, "wo": net.wo}
    for k, e in exp.items():
        g = np.asarray(got[k]).reshape(e.shape).astype(np.int64)
        assert np.array_equal(g, e), f"{arch}: {k} の量子化が仕様と違う（違う要素 {int((g != e).sum())} 個）"
    # 切り上げで増えた列が 0 であること（ファイルの全体を作り直して比べる）
    p = networks.parse_arch(arch)
    L1, L2 = p["L1"], p["L2"]

    def padded(a, cols):
        out = np.zeros((a.shape[0], N.pad32(cols)), dtype=np.int8)
        out[:, :cols] = a
        return out
    body = (struct.pack("<I", net_hash) + exp["b1"].astype("<i4").tobytes() + padded(exp["w1"], 2 * W).tobytes()
            + exp["b2"].astype("<i4").tobytes() + padded(exp["w2"], L1).tobytes()
            + exp["bo"].astype("<i4").tobytes() + padded(exp["wo"].reshape(1, L2), L2).tobytes())
    assert raw.endswith(body), f"{arch}: ネットワークの部分のバイト列が仕様と違う（切り上げの列は 0）"


@pytest.mark.parametrize("arch", ["halfkahm2_64x2-16-32_f", "halfkp_64x2-8-32"])
def test_exported_eval_matches_float(runs, tmp_path, arch):
    from shikibu import export
    ckp = os.path.join(runs[arch], "checkpoint.pt")
    path = str(tmp_path / "nn.bin")
    info = export.export_nnbin(ckp, path)
    ck = torch.load(ckp, map_location="cpu", weights_only=False)
    p = networks.parse_arch(arch)
    m = networks.build(arch)
    m.load_state_dict(ck["model"])
    m.eval()
    net = N.Net(path, yo_arch(arch))
    diffs = []
    for b in random_boards(150, seed=7501):
        stm = 0 if b.turn == cshogi.BLACK else 1
        fs = ref_features(p["feature"], p["factorize"], b, stm)
        fn = ref_features(p["feature"], p["factorize"], b, 1 - stm)
        with torch.no_grad():
            z = m(torch.tensor(fs), torch.tensor([0]), torch.tensor(fn), torch.tensor([0])).item()
        _, _, cp = net.forward(b, info["fv_scale"])
        diffs.append(abs(cp - z * float(ck["eval_scale"])))
    mean, mx = float(np.mean(diffs)), float(np.max(diffs))
    assert mean <= 20 and mx <= 150, f"{arch}: nn.bin の評価値が float のモデルとずれすぎる（平均 {mean:.1f}、最大 {mx:.1f}）"


def test_export_shk_wide(runs, tmp_path):
    from shikibu import export
    h = export.export_shk(os.path.join(runs["raypair0_64x2-8-8"], "checkpoint.pt"), str(tmp_path / "w.shk"))
    assert (h["feature_set"], h["network"], h["W"], h["L1"], h["L2"], h["num_features"]) == \
        ("RayPair_v0", "raypair0_64x2-8-8", 64, 8, 8, 15000)
    rh, rt = export.read_shk(str(tmp_path / "w.shk"))
    assert rt["l1_weight"].shape == (8, 128) and rh["network"] == "raypair0_64x2-8-8"


def test_export_unsupported(runs, tmp_path):
    from shikibu import export
    for arch in ("raypair0k_64x2-16-16_psqt_k3k3", "halfka1_32x2-8-8_f_psqt"):
        ckp = os.path.join(runs[arch], "checkpoint.pt")
        with pytest.raises(ValueError):
            export.export_nnbin(ckp, str(tmp_path / "x.bin"))
        with pytest.raises(ValueError):
            export.export_shk(ckp, str(tmp_path / "x.shk"))
        env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "python"))
        r = subprocess.run([sys.executable, "-m", "shikibu.export", "--checkpoint", ckp, "--out", str(tmp_path / "y")],
                           capture_output=True, text=True, env=env, cwd=ROOT)
        assert r.returncode == 1 and "書き出せ" in r.stderr, (r.returncode, r.stderr)
    with pytest.raises(ValueError):
        export.export_shk(os.path.join(runs["halfkp_64x2-8-32"], "checkpoint.pt"), str(tmp_path / "z.shk"))


def test_cli_edition(runs, tmp_path):
    ckp = os.path.join(runs["halfkahm2_64x2-16-32_f"], "checkpoint.pt")
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "python"))
    r = subprocess.run([sys.executable, "-m", "shikibu.export", "--checkpoint", ckp, "--out", str(tmp_path / "nn.bin")],
                       capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "FV_SCALE=13" in r.stdout and "EDITION=YANEURAOU_ENGINE_NNUE_HALFKAHM2_64X2_16_32" in r.stdout, r.stdout


# ================================================================ 実験室

def test_lab_describe():
    from shikibu import lab
    d = lab.describe("halfkahm2_512x2-16-32_f")
    assert (d["canonical"], d["feature"], d["W"], d["L1"], d["L2"]) == ("halfkahm2_512x2-16-32_f", "halfkahm2", 512, 16, 32)
    assert (d["train_features"], d["infer_features"], d["max_active"]) == (74934, 73305, 40)
    assert d["params"] == 74934 * 512 + 512 + 16 * 1024 + 16 + 32 * 16 + 32 + 32 + 1
    assert d["export_format"] == "nn.bin" and d["yo_compatible"] is True
    assert d["yo_arch"] == "halfkahm2_512x2-16-32" and d["yo_edition"] == "YANEURAOU_ENGINE_NNUE_HALFKAHM2_512X2_16_32"
    assert d["nnbin_bytes"] == NB.file_size("halfkahm2_512x2-16-32")
    d = lab.describe("raypair0_256x2-32-32")
    assert (d["export_format"], d["yo_compatible"], d["yo_edition"], d["max_active"], d["nnbin_bytes"]) == \
        (".shk", False, None, 208, None)
    d = lab.describe("raypair1k_256x2-32-32_k3k3")
    assert (d["export_format"], d["max_active"], d["infer_features"]) == (None, 256, 116744)
    assert d["params"] == 116744 * 256 + 256 + 9 * (32 * 512 + 32 + 32 * 32 + 32 + 32 + 1)
    d = lab.describe("halfkp_256x2-32-32_psqt")
    assert (d["export_format"], d["yo_compatible"], d["max_active"]) == (None, False, 38)
    assert d["params"] == 125388 * 256 + 256 + 32 * 512 + 32 + 32 * 32 + 32 + 32 + 1 + 125388
    assert isinstance(lab.PRESETS, dict) and len(lab.PRESETS) >= 8
    for name, desc in lab.PRESETS.items():
        assert networks.parse_arch(name)["canonical"] == name and isinstance(desc, str) and desc
    for must in ("raypair0_128x2-16-16", "halfkp_256x2-32-32_f"):
        assert must in lab.PRESETS
    assert any(n.startswith("halfkahm2_") for n in lab.PRESETS)
    with pytest.raises(ValueError):
        lab.describe("halfkp_250x2-32-32")


def test_lab_make_config_and_train(mat, tmp_path):
    from shikibu import lab
    paths, val = mat
    glob_ = os.path.join(os.path.dirname(paths[0]), "*.psv")
    cfg = lab.make_config("HalfKP_32x2-8-8_F", train_glob=glob_, out_dir=str(tmp_path / "run"), val_path=val,
                          overrides={"train.max_steps": 12, "train.total_steps": 12, "train.batch_size": 256,
                                     "train.device": "cpu", "train.eval_scale": 600.0, "train.val_every_steps": 6,
                                     "train.val_positions": 500, "data.cache_dir": "", "data.shuffle_buffer": 0,
                                     "data.num_threads": 2, "train.deterministic": True})
    assert cfg["model"]["network"] == "halfkp_32x2-8-8_f" and cfg["model"]["ft_width"] == 32
    assert cfg["data"]["train_glob"] == glob_ and cfg["data"]["val_path"] == val
    assert set(cfg) == set(train.DEFAULTS)
    path = str(tmp_path / "lab.toml")
    lab.write_config(cfg, path)
    back = train.load_config(path)
    train.validate(back)
    assert back == cfg
    with pytest.raises(ValueError):
        lab.make_config("halfkp_32x2-8-8", train_glob=glob_, out_dir="x", overrides={"train.no_such_key": 1})
    assert run(["--config", path]) == 0
    assert load_ckpt(str(tmp_path / "run"))["network"] == "halfkp_32x2-8-8_f"


def test_lab_compare_runs(runs):
    from shikibu import lab
    dirs = [runs[a] for a in TRAIN_ARCHS]
    rows = lab.compare_runs(dirs)
    assert len(rows) == len(dirs)
    keys = {"run_dir", "network", "step", "positions", "eval_scale", "val_loss", "best_val_loss", "common_step",
            "val_loss_at_common"}
    for r in rows:
        assert keys <= set(r), r
        log = read_log(r["run_dir"])
        vals = [x for x in log if x.get("type") == "val"]
        assert r["val_loss"] == vals[-1]["val_loss"] and r["best_val_loss"] == min(x["val_loss"] for x in vals)
        assert r["common_step"] == 60 and r["val_loss_at_common"] == [x for x in vals if x["step"] == 60][0]["val_loss"]
        assert r["step"] == 60 and r["positions"] == 60 * 256 and r["eval_scale"] == 600.0
        assert r["network"] == load_ckpt(r["run_dir"])["network"]
    assert [r["val_loss_at_common"] for r in rows] == sorted(r["val_loss_at_common"] for r in rows)


def test_lab_export_run(runs):
    from shikibu import lab
    r = lab.export_run(runs["halfkahm2_64x2-16-32_f"])
    assert r["format"] == "nn.bin" and r["path"] == os.path.join(runs["halfkahm2_64x2-16-32_f"], "export", "nn.bin")
    assert os.path.exists(r["path"]) and r["fv_scale"] == 13 and r["yo_edition"].endswith("HALFKAHM2_64X2_16_32")
    r = lab.export_run(runs["raypair0_64x2-8-8"])
    assert r["format"] == ".shk" and r["path"] == os.path.join(runs["raypair0_64x2-8-8"], "export", "nuenagon.shk")
    assert os.path.exists(r["path"])
    r = lab.export_run(runs["raypair0k_64x2-16-16_psqt_k3k3"])
    assert r["format"] is None and r["path"] is None and isinstance(r["reason"], str) and r["reason"]


def test_notebook_and_template():
    nb_path = os.path.join(ROOT, "notebooks", "shikibu_lab.ipynb")
    nb = json.load(open(nb_path, encoding="utf-8"))
    assert nb["nbformat"] == 4
    cells = nb["cells"]
    assert len(cells) >= 8
    md = "".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")
    code = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    assert len(md) >= 800 and any("぀" <= ch <= "ヿ" for ch in md), "説明（日本語）が足りない"
    for c in cells:
        if c["cell_type"] == "code":
            assert not c.get("outputs") and c.get("execution_count") is None, "出力を消してから保存すること"
    for src in code:
        lines = [ln for ln in src.splitlines() if not ln.lstrip().startswith(("!", "%"))]
        ast.parse("\n".join(lines))
    allcode = "\n".join(code)
    for must in ("ARCH", "lab.PRESETS", "lab.describe", "lab.make_config", "lab.write_config", "lab.compare_runs",
                 "lab.export_run"):
        assert must in allcode, f"ノートブックに {must} が無い"
    assert "mon.start" in allcode or "shikibu.train" in allcode, "ノートブックに学習を始めるセルが無い"
    text = open(nb_path, encoding="utf-8").read()
    for drive_path in __import__("re").findall(r"/content/drive/MyDrive/[^\"'\s\\]*", text):
        assert "<" in drive_path, f"個人の Drive のパスが書いてある: {drive_path}"
    tpl = os.path.join(ROOT, "configs", "colab_lab.toml")
    cfg = train.load_config(tpl)
    with pytest.raises(ValueError):
        train.check_placeholders(cfg)  # ひな形のまま（<…> がある）
    networks.parse_arch(cfg["model"]["network"])
    assert cfg["data"]["val_path"], "実験室のひな形は検証データを使う"
