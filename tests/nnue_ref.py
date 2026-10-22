# 変更禁止（PROTECTED）。やねうら王の「古典的な」NNUE（<特徴>_<W>x2-<L1>-<L2>）の参照実装。
# 設計担当が仕様から書いたもので、やねうら王（コミット 32165a8）を各構造でビルドし、乱数の重みの nn.bin で
# 静的評価値が1点も違わないことを確かめてある（tests/golden/nnue_golden.json）。HalfKP・256x2-32-32 のときは
# tests/halfkp_ref.py と同じ結果になる。
#
# ■ マスの番号：sq = 筋の番号（1筋=0 … 9筋=8）× 9 ＋ 段の番号（一段=0 … 九段=8）。cshogi と同じ（1一=0、9九=80）。
#   後手の視点では、盤を180度回したマス Inv(sq) = 80 − sq を使う。左右の反転は Mir(sq) = (8 − 筋) × 9 ＋ 段。
#
# ■ 駒の番号（BonaPiece）。f は「その視点の側の駒」、e は「相手の駒」。マスはその視点のマス（後手の視点なら Inv）。
#   持ち駒（0〜89）：その種類を n 枚持っているとき、1枚目〜n枚目のそれぞれを1つの特徴とする。番号は「基準 ＋ (枚目 − 1)」。
#     基準（f／e）：歩 1／20、香 39／44、桂 49／54、銀 59／64、金 69／74、角 79／82、飛 85／88。
#   盤上（90〜1547）：f_X = 90 + 162×k、e_X = f_X + 81、番号は f_X（または e_X）＋ マス。
#     k：歩 0、香 1、桂 2、銀 3、金 4（成歩・成香・成桂・成銀も金と同じ）、角 5、馬 6、飛 7、龍 8。
#   玉（HalfKA の系統だけ）：f_king = 1548 ＋ マス、e_king = 1629 ＋ マス。
#   盤上にも持ち駒にも無い駒（「駒箱」の駒。玉以外は全部で38枚）は、1枚ごとに駒の番号 0 として数える。
#
# ■ 特徴の種類（K は「その視点の側の玉」のマス。後手の視点では Inv をとったもの）
#   halfkp   ：K × 1548 ＋ 駒（玉以外の 38 枚）。次元 81 × 1548 = 125388。
#   halfka1  ：K × 1710 ＋ 駒（両方の玉を含む 40 枚）。次元 81 × 1710 = 138510。
#   halfka2  ：K × 1629 ＋ 駒'（40 枚。駒' は e_king を f_king と同じ番号にしたもの：駒 ≧ 1629 なら 駒 − 81）。次元 81 × 1629 = 131949。
#   halfkahm1：halfka1 の左右反転つき（hm = half mirror）。K の筋が 6〜9筋（筋の番号 5〜8）なら、K を Mir(K) にし、
#              盤上の駒（玉を含む。番号 ≧ 90）のマスも Mir にする（持ち駒は変えない）。次元 45 × 1710 = 76950。
#   halfkahm2：halfka2 の左右反転つき（反転してから 駒' にする）。次元 45 × 1629 = 73305。
#   特徴の番号は重複してよい（駒箱の駒）。ここでは昇順に並べた一覧（多重集合）を返す。
#
# ■ 推論（すべて整数。W・L1・L2 は構造の文字列の値）
#   蓄積値 acc[p] = バイアス(int16, W) ＋ Σ 重みの行(int16, W)（int16 の2の補数で丸める）。
#   入力 x = [clamp(acc[手番側], 0, 127), clamp(acc[相手側], 0, 127)]（2W 個）
#   隠れ層1：h1 = clamp((b1 + Σ w1 × x) >> 6, 0, 127)（L1 個。積和は int32、>> は算術右シフト）
#   隠れ層2：h2 = clamp((b2 + Σ w2 × h1) >> 6, 0, 127)（L2 個）
#   出力：raw = bo + Σ wo × h2（int32）
#   評価値：cp = raw / FV_SCALE を 0 の方向に切り捨て。ここではクリップしない。
#
# ■ nn.bin のファイルの並び（すべてリトルエンディアン）：tests/nnue_nnbin.py を見ること。
#   アフィン層の重みは「出力 × 入力（32 の倍数に切り上げた幅）」の行優先。切り上げで増えた列は 0。
import struct

import numpy as np

FE_END = 1548
F_KING = 1548
E_KING = 1629
FE_END2 = 1710
FE_HAND_END = 90

# 特徴の種類 → (やねうら王の名前, 特徴のハッシュ, 1つの玉の位置あたりの駒の番号の数, 玉の位置の数, 玉を含むか, 両玉を同じ番号にするか, 左右反転するか)
FEATURES = {
    "halfkp": ("HalfKP(Friend)", 0x5D69D5B8, FE_END, 81, False, False, False),
    "halfka1": ("HalfKA1(Friend)", 0x5F134CB8, FE_END2, 81, True, False, False),
    "halfka2": ("HalfKA2(Friend)", 0x5F234CB8, E_KING, 81, True, True, False),
    "halfkahm1": ("HalfKA_hm1(Friend)", 0x7F134CB8, FE_END2, 45, True, False, True),
    "halfkahm2": ("HalfKA_hm2(Friend)", 0x7F234CB8, E_KING, 45, True, True, True),
}

BOARD_K = {1: 0, 2: 1, 3: 2, 4: 3, 7: 4, 9: 4, 10: 4, 11: 4, 12: 4, 5: 5, 13: 6, 6: 7, 14: 8}  # cshogi の駒種 → k
HAND_BASE = [(1, 20), (39, 44), (49, 54), (59, 64), (69, 74), (79, 82), (85, 88)]  # 歩香桂銀金角飛（cshogi の持ち駒の順）


def num_features(feature):
    _, _, plane, nk, _, _, _ = FEATURES[feature]
    return plane * nk


def plane_size(feature):
    """1つの玉の位置あたりの駒の番号の数（因数分解の「玉に関係ない特徴」の数）。"""
    return FEATURES[feature][2]


def mir(sq):
    return (8 - sq // 9) * 9 + sq % 9


def bona_pieces(board, persp):
    """視点 persp（0 = 先手、1 = 後手）から見た駒の番号の一覧（玉を含む 40 個。駒箱の駒は 0）と、その視点の玉のマス。"""
    pieces = board.pieces
    out = []
    ksq = None
    for sq, p in enumerate(pieces):
        if p == 0:
            continue
        color = 0 if p < 16 else 1
        vsq = sq if persp == 0 else 80 - sq
        mine = (color == persp)
        if (p & 15) == 8:
            out.append((F_KING if mine else E_KING) + vsq)
            if mine:
                ksq = vsq
            continue
        f = 90 + 162 * BOARD_K[p & 15]
        out.append((f if mine else f + 81) + vsq)
    for color in (0, 1):
        for h, n in enumerate(board.pieces_in_hand[color]):
            fb, eb = HAND_BASE[h]
            base = fb if color == persp else eb
            for i in range(n):
                out.append(base + i)
    out += [0] * (40 - len(out))
    return out, ksq


def features(feature, board, persp):
    """特徴の番号の一覧（昇順・重複込み）。"""
    _, _, plane, _, with_king, merge, mirror = FEATURES[feature]
    bps, k = bona_pieces(board, persp)
    if not with_king:
        bps = [b for b in bps if b < F_KING]
        assert len(bps) == 38
    if mirror and k // 9 >= 5:
        k = mir(k)
        bps = [FE_HAND_END + ((b - FE_HAND_END) // 81) * 81 + mir((b - FE_HAND_END) % 81) if b >= FE_HAND_END else b
               for b in bps]
    if merge:
        bps = [b - 81 if b >= E_KING else b for b in bps]
    return sorted(k * plane + b for b in bps)


def parse_arch(arch):
    """'halfkp_256x2-32-32' → (特徴, W, L1, L2)。"""
    feat, rest = arch.split("_", 1)
    first, l1, l2 = rest.split("-")
    w, two = first.lower().split("x")
    assert two == "2"
    return feat.lower(), int(w), int(l1), int(l2)


def pad32(n):
    return (n + 31) // 32 * 32


class Net:
    def __init__(self, path, arch):
        self.feature, W, L1, L2 = parse_arch(arch)
        self.W, self.L1, self.L2 = W, L1, L2
        n_feat = num_features(self.feature)
        data = open(path, "rb").read()
        ver, h, n = struct.unpack_from("<III", data, 0)
        assert ver == 0x7AF32F16
        off = 12 + n
        self.arch_string = data[12:12 + n].decode()
        off += 4
        self.ft_b = np.frombuffer(data, "<i2", W, off).astype(np.int64); off += W * 2
        self.ft_w = np.frombuffer(data, "<i2", n_feat * W, off).reshape(n_feat, W); off += n_feat * W * 2
        off += 4

        def take(dt, cnt):
            nonlocal off
            a = np.frombuffer(data, dt, cnt, off)
            off += cnt * np.dtype(dt).itemsize
            return a.astype(np.int64)
        self.b1 = take("<i4", L1); self.w1 = take("i1", L1 * pad32(2 * W)).reshape(L1, pad32(2 * W))[:, :2 * W]
        self.b2 = take("<i4", L2); self.w2 = take("i1", L2 * pad32(L1)).reshape(L2, pad32(L1))[:, :L1]
        self.bo = take("<i4", 1); self.wo = take("i1", pad32(L2)).reshape(1, pad32(L2))[:, :L2]
        assert off == len(data), f"ファイルの長さが合わない（{off} / {len(data)}）"

    def accumulate(self, feats):
        s = self.ft_b + self.ft_w[feats].astype(np.int64).sum(axis=0)
        return ((s + 32768) % 65536 - 32768).astype(np.int64)

    def forward(self, board, fv_scale=16):
        stm = 0 if board.turn == 0 else 1
        acc = {p: self.accumulate(features(self.feature, board, p)) for p in (0, 1)}
        x = np.concatenate([np.clip(acc[stm], 0, 127), np.clip(acc[1 - stm], 0, 127)])
        h1 = np.clip((self.b1 + self.w1 @ x) >> 6, 0, 127)
        h2 = np.clip((self.b2 + self.w2 @ h1) >> 6, 0, 127)
        raw = int((self.bo + self.wo @ h2)[0])
        q = abs(raw) // fv_scale
        return acc, raw, (q if raw >= 0 else -q)
