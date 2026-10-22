# 変更禁止（PROTECTED）。HalfKP（やねうら王の halfkp_256x2-32-32 と同じ）の参照実装。設計担当が仕様から書いたもので、
# やねうら王（コミット 32165a8）の静的評価値と、乱数の重みの nn.bin で一致することを確かめてある。
#
# ■ マスの番号：sq = 筋の番号（1筋=0 … 9筋=8）× 9 ＋ 段の番号（一段=0 … 九段=8）。cshogi と同じ（1一=0、9九=80）。
#   後手の視点では、盤を180度回したマス Inv(sq) = 80 − sq を使う。
#
# ■ 駒の番号（BonaPiece）。f は「その視点の側の駒」、e は「相手の駒」。
#   盤上：f_X = 90 + 162×k、e_X = f_X + 81、番号は f_X（または e_X）＋ マス。
#     k：歩 0、香 1、桂 2、銀 3、金 4（成歩・成香・成桂・成銀も金と同じ）、角 5、馬 6、飛 7、龍 8。玉は特徴に入れない。
#   持ち駒：その種類を n 枚持っているとき、1枚目〜n枚目のそれぞれを1つの特徴とする。番号は「基準 ＋ (枚目 − 1)」。
#     基準（f／e）：歩 1／20、香 39／44、桂 49／54、銀 59／64、金 69／74、角 79／82、飛 85／88。
#   合計 1548 種類（fe_end）。
#   盤上にも持ち駒にも無い駒（詰将棋の「駒箱」の駒。玉以外は全部で38枚）は、1枚ごとに駒の番号 0 として数える（同じ特徴が重複してよい）。
#
# ■ 特徴の番号：視点の側の玉のマス（後手の視点では Inv）× 1548 ＋ 駒の番号。全部で 81 × 1548 = 125388。
#
# ■ 推論（すべて整数）
#   蓄積値 acc[p] = バイアス(int16) ＋ Σ 重みの行(int16)（int16 の範囲で計算する。あふれは int16 の2の補数で丸める）。
#   入力 x = [clamp(acc[手番側], 0, 127)（256個）, clamp(acc[相手側], 0, 127)（256個）]（uint8）
#   隠れ層1：h1 = clamp((b1 + Σ w1 × x) >> 6, 0, 127)（32個。積和は int32、>> は算術右シフト）
#   隠れ層2：h2 = clamp((b2 + Σ w2 × h1) >> 6, 0, 127)（32個）
#   出力：raw = bo + Σ wo × h2（int32）
#   評価値：cp = raw / FV_SCALE を 0 の方向に切り捨て（C++ の整数の割り算）。FV_SCALE の既定は 16。
#   ここではクリップしない（やねうら王は ±29999 でクリップする。探索で使うときの扱いは Nuenagon の決まりに従う）。
import struct

import numpy as np

FE_END = 1548
N_FEATURES = 81 * FE_END
HALF = 256

BOARD_K = {1: 0, 2: 1, 3: 2, 4: 3, 7: 4, 9: 4, 10: 4, 11: 4, 12: 4, 5: 5, 13: 6, 6: 7, 14: 8}  # cshogi の駒種 → k
HAND_BASE = [(1, 20), (39, 44), (49, 54), (59, 64), (69, 74), (79, 82), (85, 88)]  # 歩香桂銀金角飛（cshogi の持ち駒の順）


def board_bp(color, ptype, sq, persp):
    k = BOARD_K[ptype]
    f = 90 + 162 * k
    if persp == 0:
        return (f if color == 0 else f + 81) + sq
    return (f if color == 1 else f + 81) + (80 - sq)


def features(board, persp):
    """cshogi.Board と視点（0 = 先手、1 = 後手）から、HalfKP の特徴の番号の一覧（重複なし、並べ替え済み）。"""
    pieces = board.pieces
    kings = [None, None]
    for sq, p in enumerate(pieces):
        if p == 8:
            kings[0] = sq
        elif p == 24:
            kings[1] = sq
    ksq = kings[0] if persp == 0 else 80 - kings[1]
    out = []
    for sq, p in enumerate(pieces):
        if p == 0 or (p & 15) == 8:
            continue
        color = 0 if p < 16 else 1
        out.append(ksq * FE_END + board_bp(color, p & 15, sq, persp))
    for color in (0, 1):
        for h, n in enumerate(board.pieces_in_hand[color]):
            fb, eb = HAND_BASE[h]
            mine = (color == persp)
            base = fb if mine else eb
            for i in range(n):
                out.append(ksq * FE_END + base + i)
    # 盤上にも持ち駒にも無い駒（詰将棋などの「駒箱」の駒）は、駒の番号 0 の特徴として数える（38 − 盤上と持ち駒の玉以外の駒の数だけ）
    out += [ksq * FE_END + 0] * (38 - len(out))
    return sorted(out)


class Net:
    def __init__(self, path):
        data = open(path, "rb").read()
        ver, h, n = struct.unpack_from("<III", data, 0)
        assert ver == 0x7AF32F16
        off = 12 + n
        self.arch = data[12:12 + n].decode()
        off += 4
        self.ft_b = np.frombuffer(data, "<i2", HALF, off).astype(np.int64); off += HALF * 2
        self.ft_w = np.frombuffer(data, "<i2", N_FEATURES * HALF, off).reshape(N_FEATURES, HALF); off += N_FEATURES * HALF * 2
        off += 4
        def take(dt, cnt):
            nonlocal off
            a = np.frombuffer(data, dt, cnt, off)
            off += cnt * np.dtype(dt).itemsize
            return a.astype(np.int64)
        self.b1 = take("<i4", 32); self.w1 = take("i1", 32 * 512).reshape(32, 512)
        self.b2 = take("<i4", 32); self.w2 = take("i1", 32 * 32).reshape(32, 32)
        self.bo = take("<i4", 1); self.wo = take("i1", 32).reshape(1, 32)
        assert off == len(data), f"ファイルの長さが合わない（{off} / {len(data)}）"

    def accumulate(self, feats):
        s = self.ft_b + self.ft_w[feats].astype(np.int64).sum(axis=0)
        return ((s + 32768) % 65536 - 32768).astype(np.int64)  # int16 の2の補数

    def forward(self, board, fv_scale=16):
        stm = 0 if board.turn == 0 else 1
        acc = {p: self.accumulate(features(board, p)) for p in (0, 1)}
        x = np.concatenate([np.clip(acc[stm], 0, 127), np.clip(acc[1 - stm], 0, 127)])
        h1 = np.clip((self.b1 + self.w1 @ x) >> 6, 0, 127)
        h2 = np.clip((self.b2 + self.w2 @ h1) >> 6, 0, 127)
        raw = int((self.bo + self.wo @ h2)[0])
        q = abs(raw) // fv_scale
        cp = q if raw >= 0 else -q
        return acc, raw, cp
