# 変更禁止（PROTECTED）。Shikibu M11（DL の学習：NNUE の教師データ（PSV）で軽い DL を学習する）の受け入れテスト。
# 設計書 v1.9 §S14（DL1）の前半。正解は reference/dainagon_ref.py（Dainagon の学習器の写し）と cshogi。
#
# 目的：Colab の L4 で 1 局面あたり約 5 万局面/秒の DL の学習。CPU は「詰めた局面」（約 100 バイト）だけを作り、
# 55 チャンネルへの展開は GPU で行う。展開の結果は Dainagon の features.py とビット単位で同じでなければならない
# （学習したモデルを、今の Dainagon のエンジンに改造なしで載せるため）。
#
# ■ Rust（shikibu_feat）
#   dl_record(rec: bytes(40)) -> (board: list[int] 81, hands: list[int] 14, check: int, label: int)
#     PSV 1 局面の「詰めた局面」。board・hands・check の意味は dainagon_ref.compact_ref と同じ：
#       board[ty*9+tx] = 手番側から見た座標（後手番なら 180° 回す）の駒のチャンネル番号（手番側 0〜13、相手 14〜27）、空き 28
#       hands = [手番側の 歩 香 桂 銀 金 角 飛, 相手の 歩 香 桂 銀 金 角 飛]、check = 手番側が王手されていれば 1
#     label = PSV の指し手（やねうら王の 16 ビット）の 2187 の番号（dainagon_ref.label_from_psv_move と同じ）。変換できなければ −1。
#     読み解けないレコードは ValueError。
#   BatchLoader(..., output="features")：新しい引数 output。"features"（既定）は今までと完全に同じ。
#     output="dl" のとき、version は使わない（検査もしない）。各バッチの dict：
#       "board" uint8 (n, 81)、"hands" uint8 (n, 14)、"check" uint8 (n,)、"move" int64 (n,)（2187 の番号）、
#       "score" float32 (n,)（PSV の評価値そのまま）、"result" float32 (n,)（(game_result + 1) / 2、今までと同じ）
#     飛ばすレコード（skipped() に数える）：読み解けない／game_result が −1〜1 の外／label が −1。
#     state()・resume_state・シャッフル・キャッシュ・num_threads による並びの不変は、output="features" と同じ。
#
# ■ Python
#   shikibu.dl_features
#     expand(board, hands, check) -> torch.float32 (B, 55, 9, 9)。入力は torch のテンソル（uint8 か整数型）、
#       出力は入力と同じ device。dainagon_ref.make_features_v16_fast とビット単位で同じ（torch.equal）。
#       注意：持ち駒は「枚数 / 10.0 を float64 で計算して float32 にした値」（numpy と同じ）。表を引く形にすること。
#     hl_gauss(t) -> (B, 128) float32。dainagon_ref.make_hl_gauss_target と差 1e-6 以内。
#     from_hcpe(arr) -> dict（numpy）。cshogi.HuffmanCodedPosAndEval の配列を、上と同じ形の dict にする
#       （"board" "hands" "check" "move" "score" "result"）。指し手は bestMove16（cshogi の 16 ビット）、
#       result は gameResult（0 引き分け・1 先手勝ち・2 後手勝ち）を手番から見た 0 / 0.5 / 1 に。
#       指し手が変換できない局面は除く。検証用（Dainagon のテストセットは hcpe）。cshogi を使ってよい（遅くてよい）。
#   shikibu.dl_networks（いろいろな構造を試せることが目的。ブロックの種類は登録表で増やせる）
#     構造の名前：dl_<C>_<trunk>_<policy>_<value>[_<オプション>…]
#       C：32 の倍数（32〜1024）。trunk・policy・value：ブロックの並び（登録表の 1 文字を並べた文字列）。
#       trunk は 1 文字以上。policy・value は "0" ならブロックなし。
#       オプション（どれも省略できる。省略したときの値を「既定」とする）：
#         stem<N>：stem の 3×3 conv・BN・ReLU の層の数（1〜4、既定 1。最初の層は 55→C、残りは C→C）
#         ffn<N>：T の FFN の中間の幅の倍率（1〜8、既定 2）。F は Dainagon と同じ 4 倍のまま変えない
#         hd<N>：T・F の注意のヘッドあたりの次元（16・32・64、既定 32。C を割り切れないときは ValueError）
#         vh<N>：value_head の中間の幅（8 の倍数、8〜2048、既定 256）
#       正規の形：すべて大文字のブロック、オプションは stem → ffn → hd → vh の順に、既定と違うものだけ付ける。
#       小文字・オプションの順の違いも受け付ける。同じオプションを 2 回、知らないオプション・文字は ValueError。
#     BLOCKS：dict（1 文字の大文字 → ブロックのクラス）。最初から登録されているもの：
#         S＝StandardBlock（3×3 conv × 2、Dainagon と同じ）、L＝LargeKernelBlock（7×7 depthwise、Dainagon と同じ）、
#         F＝GatedTransformerBlockRelPos（Dainagon と同じ、FFN 4 倍）、
#         T＝軽い Transformer（F と同じ注意と相対位置バイアス。FFN をゲート付き GELU（GLU）、中間の幅は ffn × C）、
#         H＝シフトブロック（shift9 → 1×1 conv → BN → 活性化 → shift9 → 1×1 conv → BN → 残差を足して活性化）。
#       ブロックのクラスの約束：__init__(self, channels, act, opts)（act は活性化のクラス、opts は parse_dl の "options"）、
#         入出力は (B, C, 9, 9)、クラスメソッド macs(channels, opts) は 1 局面の順伝播の掛け算の回数を返す。
#     register_block(letter, cls)：登録表に足す（1 文字の大文字、"0" 以外。すでにあれば ValueError）。
#       足したブロックは、parse_dl・build_dl・describe_dl でそのまま使える。
#     policy の分岐の活性化は ReLU、value の分岐は Swish（Dainagon と同じ）。trunk は ReLU。
#     頭は Dainagon v35 と同じ形（policy 1×1 conv → 27ch → 2187、value_conv C→64・BN・Swish、
#       value_head 64×81→vh→128、result_head 64×81→128→1、bin_centers）。
#     parse_dl(name) -> dict："canonical", "channels", "trunk", "policy", "value",
#       "options"（{"stem", "ffn", "hd", "vh"} を全部、省略されたものは既定で埋めて）
#     build_dl(name) -> nn.Module。forward(x) は ChunagonNetV3 と同じ：train() なら (p (B,2187), value_logits (B,128),
#       result_logit (B,1))、eval() なら (p, v (B,1)（bin_centers による期待値、0〜1）, result_prob (B,1))。
#     shift9(x)：(B, C, 9, 9) → 同じ形。チャンネル c は d = c % 9、dy = d // 3 − 1、dx = d % 3 − 1 として
#       out[:, c, y, x] = x[:, c, y + dy, x + dx]（盤の外は 0）。学習するパラメータを持たない（ONNX に書き出せる形で）。
#     describe_dl(name) -> dict："canonical", "params"（学習するパラメータの数）, "macs"（1 局面の順伝播の掛け算の回数＝
#       stem ＋ 各ブロックの macs ＋ 頭。conv・linear・注意の行列積を数える。BN・活性化・shift9 は数えなくてよい）。
#     PRESETS：{"A": "dl_192_SHHHHT_H_HH", "B": "dl_128_SSHHHHHHHHTT_H_HH", "X2": "dl_192_SSHHHHHHTT_H_HH"}
#   shikibu.train_dl（python -m shikibu.train_dl --config <toml> [--resume]）
#     設定（TOML）：
#       [data]   train_glob, val_path（PSV、空でよい）, val_hcpe_glob（hcpe、空でよい）, cache_dir, max_cached,
#                shuffle_buffer, num_threads
#       [model]  network（dl_… の名前）
#       [train]  optimizer（既定 "adamw"。今は "adamw" だけ。ほかの値は設定の誤りとして止まる。後で Muon などを足す口）,
#                batch_size, lr, weight_decay, warmup_steps, total_steps, max_steps, min_lr_ratio,
#                eval_scale（"auto" か数）, value_mix_lambda（既定 0.5）, value_loss_scale（既定 1.5）,
#                label_smoothing（既定 0.03）, grad_clip（既定 1.0）, amp（"bf16"・"fp16"・"off"）, compile（真偽）,
#                channels_last（真偽）, device, seed, save_every_minutes, max_minutes（0 なら無制限）,
#                val_every_steps, val_positions, log_every_steps, progress_every_sec
#       [output] out_dir
#     損失（Dainagon の train_rocm.py と同じ。ただし教師の勝率は sigmoid(score / eval_scale)）：
#       loss = CE(p, move; label_smoothing) + value_loss_scale × (λ × CE(value_logits, hl_gauss(勝率)) + (1−λ) × BCE(result_logit, result))
#     出力（out_dir）：checkpoint.pt（"model" "optimizer" "step" "network" "eval_scale" "config" "loader_state" を含む）、
#       log.jsonl（kind が "train" の行に step・loss・loss_p・loss_v・positions_per_sec、
#       kind が "val" の行に source（"psv" か "hcpe"）・policy_acc・loss_p・value_mse・result_acc）、status.json、
#       train.lock（従来の lock.py を使う）。max_steps に達したら保存して終わる。--resume は checkpoint.pt から続ける。
#     value_mse は Dainagon の evaluate と同じ定義（v と sigmoid(score / eval_scale) の MSE。hcpe の検証は score / 600）。
#
# 既存のテスト（M1〜M10）はすべて通ること。既定の設定の結果を変えないこと（output の既定は "features"）。
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
N_POS = int(os.environ.get("SHIKIBU_M11_N", "4000"))


@pytest.fixture(scope="module")
def feat():
    import importlib
    return importlib.import_module(MODULE_NAME)


def _random_positions(n, seed):
    """乱数で指し進めた局面と、その局面で指した手（打つ手・成る手・王手の局面を多めに）。"""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b = cshogi.Board()
        for _ in range(rng.randint(1, 200)):
            if b.is_game_over():
                break
            ms = list(b.legal_moves)
            drops = [m for m in ms if cshogi.move_is_drop(m)]
            promos = [m for m in ms if cshogi.move_is_promotion(m)]
            r = rng.random()
            if drops and r < 0.25:
                m = rng.choice(drops)
            elif promos and r < 0.45:
                m = rng.choice(promos)
            else:
                m = rng.choice(ms)
            out.append((b.copy(), m))
            if len(out) >= n:
                break
            b.push(m)
    return out


def _psv(positions, seed):
    rng = random.Random(seed)
    arr = np.zeros(len(positions), cshogi.PackedSfenValue)
    for i, (b, m) in enumerate(positions):
        b.to_psfen(arr["sfen"][i])
        arr["move"][i] = cshogi.move16_to_psv(cshogi.move16(m))
        arr["score"][i] = rng.randint(-5000, 5000)
        arr["gamePly"][i] = b.move_number
        arr["game_result"][i] = rng.choice([-1, 0, 1])
    return arr


@pytest.fixture(scope="module")
def data():
    pos = _random_positions(N_POS, 11)
    arr = _psv(pos, 12)
    boards = []
    for i in range(len(arr)):
        c = cshogi.Board()
        c.set_psfen(arr["sfen"][i])
        boards.append(c)
    return pos, arr, boards


# ---------- 1. Rust：1 局面の詰めた局面と指し手の番号 ----------

def test_dl_record_matches_reference(feat, data):
    _, arr, boards = data
    for i in range(len(arr)):
        board, hands, check, label = feat.dl_record(arr[i].tobytes())
        rb, rh, rc = R.compact_ref(boards[i])
        assert list(board) == rb.tolist(), (i, boards[i].sfen())
        assert list(hands) == rh.tolist(), (i, boards[i].sfen())
        assert check == rc, (i, boards[i].sfen())
        ref_label = R.label_from_psv_move(boards[i], arr["move"][i])
        assert label == (-1 if ref_label is None else ref_label), (i, boards[i].sfen())


def test_dl_record_covers_hard_cases(data):
    pos, _, boards = data
    white = sum(b.turn for b in boards)
    checks = sum(b.is_check() for b in boards)
    drops = sum(cshogi.move_is_drop(m) for _, m in pos)
    promos = sum(cshogi.move_is_promotion(m) for _, m in pos)
    assert white > len(pos) // 4 and checks > 20 and drops > 100 and promos > 100


def test_dl_record_invalid_move_is_minus_one(feat):
    arr = _psv([(cshogi.Board(), cshogi.Board().move_from_usi("7g7f"))], 0)
    arr["move"][0] = 0
    assert feat.dl_record(arr[0].tobytes())[3] == -1


# ---------- 2. Rust：BatchLoader(output="dl") ----------

def _write(path, arr):
    with open(path, "wb") as f:
        f.write(arr.tobytes())


def _collect(loader):
    keys = ["board", "hands", "check", "move", "score", "result"]
    got = {k: [] for k in keys}
    for b in loader:
        for k in keys:
            got[k].append(b[k])
    return {k: np.concatenate(v) for k, v in got.items()}


def test_loader_dl_output(feat, data, tmp_path):
    _, arr, boards = data
    arr = arr.copy()
    arr["move"][5] = 0                 # 指し手なし → 飛ばす
    arr["game_result"][9] = 5          # 範囲外 → 飛ばす
    p = str(tmp_path / "a.psv")
    _write(p, arr)
    ld = feat.BatchLoader([p], 777, 0, output="dl", num_threads=3)
    got = _collect(ld)
    assert ld.skipped() == 2
    keep = [i for i in range(len(arr)) if i not in (5, 9)]
    assert got["board"].dtype == np.uint8 and got["board"].shape == (len(keep), 81)
    assert got["hands"].dtype == np.uint8 and got["hands"].shape == (len(keep), 14)
    assert got["check"].dtype == np.uint8 and got["move"].dtype == np.int64
    assert got["score"].dtype == np.float32 and got["result"].dtype == np.float32
    for j, i in enumerate(keep[:1500]):
        rb, rh, rc = R.compact_ref(boards[i])
        assert got["board"][j].tolist() == rb.tolist()
        assert got["hands"][j].tolist() == rh.tolist()
        assert int(got["check"][j]) == rc
        assert int(got["move"][j]) == R.label_from_psv_move(boards[i], arr["move"][i])
    assert np.array_equal(got["score"], arr["score"][keep].astype(np.float32))
    assert np.array_equal(got["result"], ((arr["game_result"][keep].astype(np.float32) + 1) * 0.5))


def test_loader_dl_resume_and_threads(feat, data, tmp_path):
    _, arr, _ = data
    p = str(tmp_path / "b.psv")
    _write(p, arr)
    full = _collect(feat.BatchLoader([p], 500, 0, output="dl", num_threads=1))
    full4 = _collect(feat.BatchLoader([p], 500, 0, output="dl", num_threads=4))
    for k in full:
        assert np.array_equal(full[k], full4[k])
    ld = feat.BatchLoader([p], 500, 0, output="dl")
    first = [next(ld) for _ in range(3)]
    st = ld.state()
    rest = _collect(feat.BatchLoader([p], 500, 0, output="dl", resume_state=st))
    joined = np.concatenate([b["board"] for b in first] + [rest["board"]])
    assert np.array_equal(joined, full["board"])


def test_loader_features_default_unchanged(feat, data, tmp_path):
    _, arr, _ = data
    p = str(tmp_path / "c.psv")
    _write(p, arr[:300])
    b = next(feat.BatchLoader([p], 300, 0))
    assert "stm_indices" in b and "board" not in b


# ---------- 3. Python：GPU で 55 チャンネルに展開 ----------

def test_expand_bit_exact(data):
    from shikibu import dl_features as F
    _, _, boards = data
    rb, rh, rc, ref = [], [], [], []
    for c in boards:
        b, h, k = R.compact_ref(c)
        rb.append(b); rh.append(h); rc.append(k)
        ref.append(R.make_features_v16_fast(c, c.turn == cshogi.WHITE))
    x = F.expand(torch.from_numpy(np.stack(rb)), torch.from_numpy(np.stack(rh)),
                 torch.tensor(rc, dtype=torch.uint8))
    assert x.dtype == torch.float32 and tuple(x.shape) == (len(boards), 55, 9, 9)
    want = torch.from_numpy(np.stack(ref))
    bad = (x != want).flatten(1).any(1).nonzero().flatten().tolist()
    assert not bad, [boards[i].sfen() for i in bad[:5]]


def test_expand_device_follows_input():
    from shikibu import dl_features as F
    b, h, k = R.compact_ref(cshogi.Board())
    x = F.expand(torch.from_numpy(b[None]), torch.from_numpy(h[None]), torch.tensor([k], dtype=torch.uint8))
    assert x.device.type == "cpu"


def test_hl_gauss():
    from shikibu import dl_features as F
    t = torch.tensor([0.0, 0.001, 0.25, 0.5, 0.731, 0.999, 1.0])
    got = F.hl_gauss(t)
    assert got.dtype == torch.float32 and tuple(got.shape) == (7, 128)
    for i, v in enumerate(t.tolist()):
        assert np.abs(got[i].numpy() - R.make_hl_gauss_target(v)).max() < 1e-6


def test_from_hcpe(data):
    from shikibu import dl_features as F
    pos, _, _ = data
    pos = pos[:500]
    arr = np.zeros(len(pos), cshogi.HuffmanCodedPosAndEval)
    for i, (b, m) in enumerate(pos):
        b.to_hcp(arr["hcp"][i])
        arr["bestMove16"][i] = cshogi.move16(m)
        arr["eval"][i] = i - 250
        arr["gameResult"][i] = i % 3
    d = F.from_hcpe(arr)
    assert len(d["move"]) == len(pos)
    for i, (b, m) in enumerate(pos):
        rb, rh, rc = R.compact_ref(b)
        assert d["board"][i].tolist() == rb.tolist() and d["hands"][i].tolist() == rh.tolist()
        assert int(d["check"][i]) == rc
        assert int(d["move"][i]) == R.label_of_usi(b, cshogi.move_to_usi(m))
        assert float(d["score"][i]) == i - 250
        gr = i % 3
        want = 0.5 if gr == 0 else (1.0 if (gr == 1) == (b.turn == cshogi.BLACK) else 0.0)
        assert float(d["result"][i]) == want


# ---------- 4. ネットワーク ----------

def test_parse_dl():
    from shikibu import dl_networks as N
    d = N.parse_dl("dl_192_shhhht_h_hh")
    assert d["canonical"] == "dl_192_SHHHHT_H_HH" and d["channels"] == 192
    assert d["trunk"] == "SHHHHT" and d["policy"] == "H" and d["value"] == "HH"
    assert d["options"] == {"stem": 1, "ffn": 2, "hd": 32, "vh": 256}
    assert N.parse_dl("dl_64_S_0_0")["policy"] == "0"
    e = N.parse_dl("dl_192_SHT_H_HH_VH512_stem2_hd64")
    assert e["canonical"] == "dl_192_SHT_H_HH_stem2_hd64_vh512"
    assert e["options"] == {"stem": 2, "ffn": 2, "hd": 64, "vh": 512}
    assert N.parse_dl("dl_192_SHT_H_HH_stem1_ffn2_hd32_vh256")["canonical"] == "dl_192_SHT_H_HH"
    for bad in ["dl_100_S_0_0", "dl_192_SX_0_0", "dl_192__0_0", "dl_192_S_0", "halfkp_256x2-32-32", "dl_0_S_0_0",
                "dl_192_S_0_0_stem5", "dl_192_S_0_0_ffn0", "dl_96_T_0_0_hd64", "dl_192_S_0_0_vh12",
                "dl_192_S_0_0_stem2_stem3", "dl_192_S_0_0_foo1"]:
        with pytest.raises(ValueError):
            N.parse_dl(bad)


def test_options_change_model():
    from shikibu import dl_networks as N
    base = N.describe_dl("dl_96_ST_0_0")
    s2 = N.describe_dl("dl_96_ST_0_0_stem2")
    assert s2["macs"] - base["macs"] == 9 * 96 * 96 * 81                 # stem の 2 層目（3×3、C→C）
    f4 = N.describe_dl("dl_96_ST_0_0_ffn4")
    assert f4["macs"] - base["macs"] == 81 * (96 * 96 * 2 * 2 + 96 * 96 * 2)  # GLU：入 C→2·ffn·C、出 ffn·C→C
    v = N.describe_dl("dl_96_ST_0_0_vh512")
    assert v["macs"] - base["macs"] == 64 * 81 * 256 + 256 * 128
    net = N.build_dl("dl_96_ST_0_0_stem2_ffn4_hd16_vh512")
    net.train()
    p, vl, rl = net(torch.randn(2, 55, 9, 9))
    assert p.shape == (2, 2187) and vl.shape == (2, 128) and rl.shape == (2, 1)


def test_register_block():
    from shikibu import dl_networks as N
    import torch.nn as nn

    class Z(nn.Module):
        def __init__(self, channels, act, opts):
            super().__init__()
            self.conv = nn.Conv2d(channels, channels, 1, bias=False)
            self.act = act()

        def forward(self, x):
            return self.act(self.conv(x) + x)

        @classmethod
        def macs(cls, channels, opts):
            return channels * channels * 81

    with pytest.raises(ValueError):
        N.register_block("S", Z)
    for bad in ["0", "zz", "z"]:
        with pytest.raises(ValueError):
            N.register_block(bad, Z)
    N.register_block("Z", Z)
    try:
        assert N.parse_dl("dl_64_SZ_Z_0")["canonical"] == "dl_64_SZ_Z_0"
        net = N.build_dl("dl_64_SZ_Z_0")
        net.train()
        p, vl, rl = net(torch.randn(2, 55, 9, 9))
        (p.sum() + vl.sum()).backward()
        d0 = N.describe_dl("dl_64_S_0_0")["macs"]
        assert N.describe_dl("dl_64_SZ_Z_0")["macs"] - d0 == 2 * 64 * 64 * 81
        assert N.describe_dl("dl_64_SZ_Z_0")["params"] == sum(p.numel() for p in net.parameters() if p.requires_grad)
    finally:
        del N.BLOCKS["Z"]


def test_shift9():
    from shikibu import dl_networks as N
    x = torch.randn(2, 18, 9, 9)
    y = N.shift9(x)
    for c in range(18):
        d = c % 9
        dy, dx = d // 3 - 1, d % 3 - 1
        for yy in range(9):
            for xx in range(9):
                sy, sx = yy + dy, xx + dx
                want = x[:, c, sy, sx] if (0 <= sy < 9 and 0 <= sx < 9) else torch.zeros(2)
                assert torch.equal(y[:, c, yy, xx], want)


@pytest.mark.parametrize("key", ["A", "B", "X2"])
def test_presets_forward(key):
    from shikibu import dl_networks as N
    name = N.PRESETS[key]
    net = N.build_dl(name)
    x = torch.randn(3, 55, 9, 9)
    net.train()
    p, vl, rl = net(x)
    assert p.shape == (3, 2187) and vl.shape == (3, 128) and rl.shape == (3, 1)
    net.eval()
    with torch.no_grad():
        p, v, r = net(x)
    assert p.shape == (3, 2187) and v.shape == (3, 1) and r.shape == (3, 1)
    assert float(v.min()) >= 0 and float(v.max()) <= 1 and float(r.min()) >= 0 and float(r.max()) <= 1
    d = N.describe_dl(name)
    assert d["params"] == sum(p.numel() for p in net.parameters() if p.requires_grad)


def test_presets_macs():
    from shikibu import dl_networks as N
    # 設計の見積もり（順伝播、1 局面）：A 約 1.35 億、B 約 1.1 億、X2 約 2.3 億。±25% に収まること。
    for key, est in [("A", 1.35e8), ("B", 1.1e8), ("X2", 2.3e8)]:
        m = N.describe_dl(N.PRESETS[key])["macs"]
        assert 0.75 * est <= m <= 1.25 * est, (key, m)


def test_all_block_kinds_build():
    from shikibu import dl_networks as N
    net = N.build_dl("dl_64_SLHTF_SH_LT_stem3_ffn3_hd16_vh64")
    net.train()
    p, vl, rl = net(torch.randn(2, 55, 9, 9))
    (p.sum() + vl.sum() + rl.sum()).backward()


# ---------- 5. 学習（CPU で数歩） ----------

def _hcpe_file(path, pos):
    arr = np.zeros(len(pos), cshogi.HuffmanCodedPosAndEval)
    for i, (b, m) in enumerate(pos):
        b.to_hcp(arr["hcp"][i])
        arr["bestMove16"][i] = cshogi.move16(m)
        arr["eval"][i] = 0
        arr["gameResult"][i] = 1
    with open(path, "wb") as f:
        f.write(arr.tobytes())


def _config(tmp_path, train_glob, val_psv, val_hcpe, out, max_steps, optimizer="adamw"):
    return f"""
[data]
train_glob = "{train_glob}"
val_path = "{val_psv}"
val_hcpe_glob = "{val_hcpe}"
cache_dir = ""
max_cached = 3
shuffle_buffer = 1000
num_threads = 2

[model]
network = "dl_32_HT_H_H"

[train]
optimizer = "{optimizer}"
batch_size = 64
lr = 1e-3
weight_decay = 1e-3
warmup_steps = 2
total_steps = 10
max_steps = {max_steps}
min_lr_ratio = 0.01
eval_scale = 600
value_mix_lambda = 0.5
value_loss_scale = 1.5
label_smoothing = 0.03
grad_clip = 1.0
amp = "off"
compile = false
channels_last = false
device = "cpu"
seed = 0
save_every_minutes = 30.0
max_minutes = 0
val_every_steps = 3
val_positions = 200
log_every_steps = 1
progress_every_sec = 1.0

[output]
out_dir = "{out}"
"""


def test_train_and_resume(data, tmp_path):
    pos, arr, _ = data
    d = tmp_path / "train"
    d.mkdir()
    _write(str(d / "t0.psv"), arr[:2000])
    _write(str(d / "t1.psv"), arr[2000:3500])
    vdir = tmp_path / "val"
    vdir.mkdir()
    _write(str(vdir / "v.psv"), arr[3500:])
    _hcpe_file(str(vdir / "v.hcpe"), pos[3500:])
    out = tmp_path / "out"
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "python"))
    cfg = tmp_path / "c.toml"
    cfg.write_text(_config(tmp_path, f"{d}/*.psv", vdir / "v.psv", f"{vdir}/*.hcpe", out, 6), encoding="utf-8")
    r = subprocess.run([sys.executable, "-m", "shikibu.train_dl", "--config", str(cfg)], env=env,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr
    ck = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 6 and ck["network"] == "dl_32_HT_H_H" and float(ck["eval_scale"]) == 600.0
    for k in ("model", "optimizer", "config", "loader_state"):
        assert k in ck
    cfg.write_text(_config(tmp_path, f"{d}/*.psv", vdir / "v.psv", f"{vdir}/*.hcpe", out, 10), encoding="utf-8")
    r = subprocess.run([sys.executable, "-m", "shikibu.train_dl", "--config", str(cfg), "--resume"], env=env,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr
    ck = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 10
    rows = [json.loads(s) for s in open(out / "log.jsonl", encoding="utf-8") if s.strip()]
    tr = [x for x in rows if x.get("kind") == "train"]
    assert tr and all(np.isfinite(x["loss"]) for x in tr)
    for k in ("step", "loss", "loss_p", "loss_v", "positions_per_sec"):
        assert k in tr[-1]
    va = [x for x in rows if x.get("kind") == "val"]
    assert {x["source"] for x in va} == {"psv", "hcpe"}
    for x in va:
        assert 0.0 <= x["policy_acc"] <= 1.0 and x["loss_p"] > 0 and x["value_mse"] >= 0 and 0.0 <= x["result_acc"] <= 1.0
    assert (out / "status.json").exists()


def test_train_refuses_without_resume(data, tmp_path):
    _, arr, _ = data
    d = tmp_path / "tr"
    d.mkdir()
    _write(str(d / "t.psv"), arr[:500])
    out = tmp_path / "o"
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "python"))
    cfg = tmp_path / "c.toml"
    cfg.write_text(_config(tmp_path, f"{d}/*.psv", "", "", out, 2), encoding="utf-8")
    r = subprocess.run([sys.executable, "-m", "shikibu.train_dl", "--config", str(cfg)], env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    r = subprocess.run([sys.executable, "-m", "shikibu.train_dl", "--config", str(cfg)], env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode != 0


def test_train_unknown_optimizer_stops(data, tmp_path):
    _, arr, _ = data
    d = tmp_path / "tr"
    d.mkdir()
    _write(str(d / "t.psv"), arr[:500])
    out = tmp_path / "o"
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "python"))
    cfg = tmp_path / "c.toml"
    cfg.write_text(_config(tmp_path, f"{d}/*.psv", "", "", out, 2, optimizer="sgd"), encoding="utf-8")
    r = subprocess.run([sys.executable, "-m", "shikibu.train_dl", "--config", str(cfg)], env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode != 0 and not (out / "checkpoint.pt").exists()
