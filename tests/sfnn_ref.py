# 変更禁止（PROTECTED）。やねうら王の SFNN（レイヤースタックつき）の参照実装と、乱数の重みの nn.bin を作る関数。
# 設計担当が仕様から書いたもので、やねうら王（コミット 2bc64b4）を SFNN_halfkahm2_512_15_32_k3k3 でビルドし、
# 乱数の重みの nn.bin で静的評価値が1点も違わないことを確かめてある（tests/golden/sfnn_golden.json）。
# 特徴量（HalfKA_hm2 など）の番号の付け方は tests/nnue_ref.py と同じ（features 関数をそのまま使う）。
#
# ■ 構造の名前：SFNN_<特徴>_<W>_<H1>_<H2>_k3k3（例 SFNN_halfkahm2_512_15_32_k3k3）。
#   W は FT の幅（片側）。H1 は 8n か 8n−1。8n−1 のときは fc_0 の出力が H1+1 個で、最後の1個が「近道」。
#   レイヤースタックは 9 組（k3k3）。
#
# ■ nn.bin のファイルの並び（すべてリトルエンディアン）
#   u32 版 0x7AF32F16、u32 ハッシュ 0x3C203B32、u32 文字列の長さ n、構造の文字列（n バイト。中身は読み込みで使われない）
#   u32 FT のハッシュ 0x5F134AB8
#   FT のバイアス（int16 × W）を LEB128 で圧縮した塊
#   FT の重み（int16 × 特徴の次元 × W、特徴ごとに W 個ずつ）を LEB128 で圧縮した塊
#     塊 = 文字列 "COMPRESSED_LEB128"（17 バイト）＋ u32 バイト数 ＋ 符号つき LEB128 の列
#     ファイルの値は、推論で使う値の半分（読み込むときに 2 倍する）。
#   スタック 0〜8 のそれぞれ：
#     u32 ハッシュ 0x6333718A
#     fc_0：バイアス int32 × O0、重み int8 × O0 × pad32(W)（O0 = H1+1（近道あり）または H1）
#     fc_1：バイアス int32 × H2、重み int8 × H2 × pad32(2×H1)
#     fc_2：バイアス int32 × 1、重み int8 × 1 × pad32(H2)
#     重みは「出力 × 入力（32 の倍数に切り上げた幅）」の行優先。切り上げで増えた列は 0。
#   ファイルはここで終わる（余りがあると読み込みに失敗する）。
#
# ■ 推論（すべて整数）
#   蓄積値 acc[p] = 2×バイアス ＋ Σ 2×重みの行（int16 の2の補数で丸める）。p は視点（先手 0・後手 1）。
#   FT の出力 x（W 個、0〜255 の整数）：手番側の視点 → 相手側の視点 の順に、それぞれ j = 0 … W/2−1 について
#     s0 = clamp(acc[p][j], 0, 254)、s1 = clamp(acc[p][j + W/2], 0, 254)、x = clamp(((s0 << 7) × s1) >> 16, 0, 255)
#   バケツ（k3k3）：f = 手番側の玉の段、e = 相手の玉の段（段は 一段=0 … 九段=8、マス % 9）。
#     手番が先手なら e = 8 − e、後手なら f = 8 − f。バケツ = (f // 3) × 3 + e // 3。
#   そのバケツのスタックで：
#     y0 = b0 + w0 · x（int32、O0 個）
#     隠れ層の入力 h（pad32(2×H1) 個）：h[i] = min(127, (y0[i]² ) >> 19)（i < H1）、
#                                         h[H1 + i] = clamp(y0[i] >> 6, 0, 127)（i < H1）、残りは 0
#     y1 = b1 + w1 · h、a1 = clamp(y1 >> 6, 0, 127)（H2 個）
#     out = b2 + w2 · a1（int32）＋（近道ありなら）y0[H1]
#   評価値 = out / FV_SCALE を 0 の方向に切り捨て（クリップしない）。
import struct

import numpy as np

import nnue_ref

VERSION = 0x7AF32F16
SFNN_HASH = 0x3C203B32
FT_HASH = 0x5F134AB8
NET_HASH = 0x6333718A
LEB128_MAGIC = b"COMPRESSED_LEB128"
STACKS = 9


def parse_arch(arch):
    """'SFNN_halfkahm2_512_15_32_k3k3' → (特徴, W, H1, H2, スタック数)。"""
    parts = arch.split("_")
    assert parts[0].upper() == "SFNN" and parts[-1].lower() == "k3k3", arch
    feat, w, h1, h2 = parts[1].lower(), int(parts[2]), int(parts[3]), int(parts[4])
    assert h1 % 8 in (0, 7) and w % 64 == 0
    return feat, w, h1, h2, STACKS


def pad32(n):
    return (n + 31) // 32 * 32


def shortcut(h1):
    return h1 % 8 == 7


def k3k3_bucket(board):
    """cshogi の Board から k3k3 のバケツ（0〜8）。"""
    stm = board.turn  # 0 先手、1 後手
    f = board.king_square(stm) % 9
    e = board.king_square(1 - stm) % 9
    if stm == 0:
        e = 8 - e
    else:
        f = 8 - f
    return (f // 3) * 3 + e // 3


# ---------------------------------------------------------------- LEB128

def leb128_encode(values):
    out = bytearray()
    for v in values:
        v = int(v)
        while True:
            byte = v & 0x7F
            v >>= 7
            if (v == 0 and not (byte & 0x40)) or (v == -1 and (byte & 0x40)):
                out.append(byte)
                break
            out.append(byte | 0x80)
    return bytes(out)


def leb128_block(values):
    body = leb128_encode(values)
    return LEB128_MAGIC + struct.pack("<I", len(body)) + body


def leb128_read(data, off, count):
    """塊を読んで (値の配列 int64, 次の位置)。"""
    assert data[off:off + 17] == LEB128_MAGIC, "LEB128 の目印が無い"
    off += 17
    nbytes = struct.unpack_from("<I", data, off)[0]
    off += 4
    end = off + nbytes
    out = np.empty(count, dtype=np.int64)
    pos = off
    for i in range(count):
        result, shift = 0, 0
        while True:
            byte = data[pos]
            pos += 1
            result |= (byte & 0x7F) << shift
            shift += 7
            if not (byte & 0x80):
                if byte & 0x40:
                    result -= 1 << shift
                break
        out[i] = result
    assert pos == end, f"LEB128 のバイト数が合わない（{pos - off} / {nbytes}）"
    return out, end


# ---------------------------------------------------------------- 書き出し・読み込み

def write(path, arch, q):
    """q：ft_b (W,)、ft_w (次元, W)（ファイルに書く値＝推論の値の半分）、
    b0 (S, O0) w0 (S, O0, W)、b1 (S, H2) w1 (S, H2, 2H1)、b2 (S, 1) w2 (S, 1, H2)。"""
    feat, W, H1, H2, S = parse_arch(arch)
    s = arch.encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<III", VERSION, SFNN_HASH, len(s)))
        f.write(s)
        f.write(struct.pack("<I", FT_HASH))
        f.write(leb128_block(np.asarray(q["ft_b"]).reshape(-1)))
        f.write(leb128_block(np.asarray(q["ft_w"]).reshape(-1)))
        for i in range(S):
            f.write(struct.pack("<I", NET_HASH))
            for b, w, cols in (("b0", "w0", W), ("b1", "w1", 2 * H1), ("b2", "w2", H2)):
                f.write(np.ascontiguousarray(q[b][i], dtype="<i4").tobytes())
                m = np.zeros((q[w][i].shape[0], pad32(cols)), dtype=np.int8)
                m[:, :cols] = q[w][i]
                f.write(m.tobytes())


RANGES = {  # 乱数の重みの範囲（ファイルに書く値）
    "normal": dict(ft_b=(0, 40), ft_w=(-24, 24), b0=(-3000, 3000), w0=(-6, 6),
                   b1=(-2000, 2000), w1=(-40, 40), b2=(-3000, 3000), w2=(-80, 80)),
    "wide": dict(ft_b=(-60, 100), ft_w=(-60, 60), b0=(-20000, 20000), w0=(-30, 30),
                 b1=(-8000, 8000), w1=(-127, 127), b2=(-30000, 30000), w2=(-127, 127)),
}


def make(path, arch, seed, scale="normal"):
    feat, W, H1, H2, S = parse_arch(arch)
    O0 = H1 + 1 if shortcut(H1) else H1
    r = RANGES[scale]
    rng = np.random.default_rng(seed)
    n = nnue_ref.num_features(feat)

    def ints(key, shape):
        lo, hi = r[key]
        return rng.integers(lo, hi + 1, size=shape)
    q = {"ft_b": ints("ft_b", (W,)), "ft_w": ints("ft_w", (n, W)),
         "b0": ints("b0", (S, O0)), "w0": ints("w0", (S, O0, W)),
         "b1": ints("b1", (S, H2)), "w1": ints("w1", (S, H2, 2 * H1)),
         "b2": ints("b2", (S, 1)), "w2": ints("w2", (S, 1, H2))}
    write(path, arch, q)
    return q


class Net:
    def __init__(self, path, arch):
        self.feature, W, H1, H2, S = parse_arch(arch)
        self.W, self.H1, self.H2, self.S = W, H1, H2, S
        self.O0 = H1 + 1 if shortcut(H1) else H1
        n = nnue_ref.num_features(self.feature)
        data = open(path, "rb").read()
        ver, h, ln = struct.unpack_from("<III", data, 0)
        assert ver == VERSION and h == SFNN_HASH
        off = 12 + ln
        assert struct.unpack_from("<I", data, off)[0] == FT_HASH
        off += 4
        self.ft_b, off = leb128_read(data, off, W)
        w, off = leb128_read(data, off, n * W)
        self.ft_w = w.reshape(n, W)
        self.st = []
        for _ in range(S):
            assert struct.unpack_from("<I", data, off)[0] == NET_HASH
            off += 4
            layer = {}
            for b, wn, rows, cols in (("b0", "w0", self.O0, W), ("b1", "w1", H2, 2 * H1), ("b2", "w2", 1, H2)):
                layer[b] = np.frombuffer(data, "<i4", rows, off).astype(np.int64)
                off += rows * 4
                pc = pad32(cols)
                layer[wn] = np.frombuffer(data, "i1", rows * pc, off).reshape(rows, pc).astype(np.int64)
                off += rows * pc
            self.st.append(layer)
        assert off == len(data), f"ファイルの長さが合わない（{off} / {len(data)}）"

    def accumulate(self, feats):
        b = (2 * self.ft_b + 32768) % 65536 - 32768
        w = (2 * self.ft_w[feats] + 32768) % 65536 - 32768
        s = b + w.sum(axis=0)
        return (s + 32768) % 65536 - 32768

    def transform(self, acc, stm):
        half = self.W // 2
        out = []
        for p in (stm, 1 - stm):
            s0 = np.clip(acc[p][:half], 0, 254)
            s1 = np.clip(acc[p][half:], 0, 254)
            out.append(np.clip(((s0 << 7) * s1) >> 16, 0, 255))
        return np.concatenate(out)

    def forward(self, board, fv_scale=16):
        stm = board.turn
        acc = {p: self.accumulate(nnue_ref.features(self.feature, board, p)) for p in (0, 1)}
        x = self.transform(acc, stm)
        k = k3k3_bucket(board)
        L = self.st[k]
        H1 = self.H1
        y0 = L["b0"] + L["w0"][:, :self.W] @ x
        h = np.zeros(pad32(2 * H1), dtype=np.int64)
        h[:H1] = np.minimum(127, (y0[:H1] * y0[:H1]) >> 19)
        h[H1:2 * H1] = np.clip(y0[:H1] >> 6, 0, 127)
        y1 = L["b1"] + L["w1"] @ h
        a1 = np.clip(y1 >> 6, 0, 127)
        out = int((L["b2"] + L["w2"][:, :self.H2] @ a1)[0])
        if shortcut(H1):
            out += int(y0[H1])
        q = abs(out) // fv_scale
        return k, out, (q if out >= 0 else -q)
