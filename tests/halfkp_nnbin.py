# 変更禁止（PROTECTED）。やねうら王の NNUE（halfkp_256x2-32-32）形式の評価関数ファイル（nn.bin）を、乱数の重みで作る。
# ファイルの並び（すべてリトルエンディアン）：
#   ヘッダ：u32 版（0x7AF32F16）、u32 ハッシュ、u32 構造の文字列の長さ、構造の文字列
#   特徴変換：u32 ハッシュ、int16 バイアス[256]、int16 重み[125388][256]
#   ネットワーク：u32 ハッシュ、
#     隠れ層1：int32 バイアス[32]、int8 重み[32][512]
#     隠れ層2：int32 バイアス[32]、int8 重み[32][32]
#     出力層 ：int32 バイアス[1]、 int8 重み[1][32]
# 乱数は Python の random.Random（版が変わっても同じ値を出す）だけを使う。
import random
import struct

import numpy as np

U32 = 0xFFFFFFFF
VERSION = 0x7AF32F16
FE_END = 1548
SQ_NB = 81
N_FEATURES = SQ_NB * FE_END  # 125388
HALF = 256
ARCH = ("Features=HalfKP(Friend)[125388->256x2],Network=AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-32]"
        "(ClippedReLU[32](AffineTransformSparseInput[32<-512](InputSlice[512(0:512)])))))")
FEATURE_HASH = 0x5D69D5B8


def _u32(x):
    return x & U32


def _affine_hash(prev, out_dims):
    return _u32((0xCC03DAE4 + out_dims) ^ (prev >> 1) ^ _u32(prev << 31))


def _relu_hash(prev):
    return _u32(0x538D24C7 + prev)


def hashes():
    ft = _u32(FEATURE_HASH ^ (HALF * 2))
    h = _u32(0xEC42E90D ^ (HALF * 2) ^ (0 << 10))
    h = _relu_hash(_affine_hash(h, 32))
    h = _relu_hash(_affine_hash(h, 32))
    net = _affine_hash(h, 1)
    return ft, net, _u32(ft ^ net)


def _ints(rng, n, lo, hi, dtype):
    """[lo, hi] の一様な整数を n 個（randbytes から作る）。"""
    span = hi - lo + 1
    raw = np.frombuffer(rng.randbytes(n * 4), dtype="<u4").astype(np.int64)
    return (lo + raw % span).astype(dtype)


RANGES = {
    # (特徴変換の重み, そのバイアス, 隠れ層1の重み, そのバイアス, 隠れ層2の重み, そのバイアス, 出力層の重み, そのバイアス)
    "normal": ((-12, 12), (-30, 90), (-40, 40), (-3000, 3000), (-50, 50), (-2000, 2000), (-80, 80), (-500, 500)),
}
_WIDE = ((-120, 120), (-300, 300), (-127, 127), (-30000, 30000), (-127, 127), (-30000, 30000), (-127, 127), (-5000, 5000))
for _name, _idx in (("wide_ft", (0, 1)), ("wide_l1", (2, 3)), ("wide_l2", (4, 5)), ("wide_out", (6, 7))):
    _r = list(RANGES["normal"])
    for _i in _idx:
        _r[_i] = _WIDE[_i]
    RANGES[_name] = tuple(_r)


def make(path, seed, scale="normal"):
    """乱数の重みの nn.bin を path に書く。scale：'normal'（ふつうの幅）、'wide_ft'・'wide_l1'・'wide_l2'・'wide_out'
    （その層だけ、飽和しやすい広い幅）。"""
    rng = random.Random(seed)
    ft_hash, net_hash, file_hash = hashes()
    ftw, ftb, l1w, l1b, l2w, l2b, ow, ob = RANGES[scale]
    with open(path, "wb") as f:
        a = ARCH.encode()
        f.write(struct.pack("<III", VERSION, file_hash, len(a)))
        f.write(a)
        f.write(struct.pack("<I", ft_hash))
        f.write(_ints(rng, HALF, *ftb, "<i2").tobytes())
        chunk = 1 << 22
        left = N_FEATURES * HALF
        while left:
            n = min(left, chunk)
            f.write(_ints(rng, n, *ftw, "<i2").tobytes())
            left -= n
        f.write(struct.pack("<I", net_hash))
        f.write(_ints(rng, 32, *l1b, "<i4").tobytes())
        f.write(_ints(rng, 32 * 512, *l1w, "i1").tobytes())
        f.write(_ints(rng, 32, *l2b, "<i4").tobytes())
        f.write(_ints(rng, 32 * 32, *l2w, "i1").tobytes())
        f.write(_ints(rng, 1, *ob, "<i4").tobytes())
        f.write(_ints(rng, 32, *ow, "i1").tobytes())
    return path


if __name__ == "__main__":
    import sys
    make(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1, sys.argv[3] if len(sys.argv) > 3 else "normal")
