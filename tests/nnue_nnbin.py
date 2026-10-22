# 変更禁止（PROTECTED）。やねうら王の古典的な NNUE（<特徴>_<W>x2-<L1>-<L2>）の nn.bin を作る・調べる道具。
# ファイルの並び（すべてリトルエンディアン）：
#   ヘッダ：u32 版（0x7AF32F16）、u32 ファイルのハッシュ（= 特徴変換のハッシュ ^ ネットワークのハッシュ）、
#           u32 構造の文字列の長さ、構造の文字列（arch_string() のとおり。読む側は中身を確かめなくてよい）
#   特徴変換：u32 ハッシュ（= 特徴のハッシュ ^ 2W）、int16 バイアス[W]、int16 重み[特徴の次元][W]
#   ネットワーク：u32 ハッシュ（hashes() のとおり）、
#     隠れ層1：int32 バイアス[L1]、int8 重み[L1][pad32(2W)]
#     隠れ層2：int32 バイアス[L2]、int8 重み[L2][pad32(L1)]
#     出力層 ：int32 バイアス[1]、 int8 重み[1][pad32(L2)]
#   pad32(n) は n を 32 の倍数に切り上げたもの。切り上げで増えた列は 0 を書く。
# 乱数は Python の random.Random（版が変わっても同じ値を出す）だけを使う。
import random
import struct

import numpy as np

from nnue_ref import FEATURES, num_features, pad32, parse_arch

U32 = 0xFFFFFFFF
VERSION = 0x7AF32F16


def _u32(x):
    return x & U32


def _affine_hash(prev, out_dims):
    return _u32((0xCC03DAE4 + out_dims) ^ (prev >> 1) ^ _u32(prev << 31))


def _relu_hash(prev):
    return _u32(0x538D24C7 + prev)


def hashes(arch):
    """(特徴変換のハッシュ, ネットワークのハッシュ, ファイルのハッシュ)。"""
    feat, W, L1, L2 = parse_arch(arch)
    ft = _u32(FEATURES[feat][1] ^ (2 * W))
    h = _u32(0xEC42E90D ^ (2 * W) ^ (0 << 10))
    h = _relu_hash(_affine_hash(h, L1))
    h = _relu_hash(_affine_hash(h, L2))
    net = _affine_hash(h, 1)
    return ft, net, _u32(ft ^ net)


def arch_string(arch):
    feat, W, L1, L2 = parse_arch(arch)
    n = 2 * W
    s = f"InputSlice[{n}(0:{n})]"
    s = f"AffineTransformSparseInput[{L1}<-{n}]({s})"
    s = f"ClippedReLU[{L1}]({s})"
    s = f"AffineTransform[{L2}<-{L1}]({s})"
    s = f"ClippedReLU[{L2}]({s})"
    s = f"AffineTransform[1<-{L2}]({s})"
    return f"Features={FEATURES[feat][0]}[{num_features(feat)}->{W}x2],Network={s}"


def file_size(arch):
    feat, W, L1, L2 = parse_arch(arch)
    return (12 + len(arch_string(arch).encode()) + 4 + 2 * W + 2 * W * num_features(feat) + 4
            + 4 * L1 + L1 * pad32(2 * W) + 4 * L2 + L2 * pad32(L1) + 4 + pad32(L2))


def _ints(rng, n, lo, hi, dtype):
    span = hi - lo + 1
    raw = np.frombuffer(rng.randbytes(n * 4), dtype="<u4").astype(np.int64)
    return (lo + raw % span).astype(dtype)


def _padded(rng, rows, cols, lo, hi):
    a = np.zeros((rows, pad32(cols)), dtype="i1")
    a[:, :cols] = _ints(rng, rows * cols, lo, hi, "i1").reshape(rows, cols)
    return a


RANGES = {
    # (特徴変換の重み, そのバイアス, 隠れ層1の重み, そのバイアス, 隠れ層2の重み, そのバイアス, 出力層の重み, そのバイアス)
    "normal": ((-12, 12), (-30, 90), (-40, 40), (-3000, 3000), (-50, 50), (-2000, 2000), (-80, 80), (-500, 500)),
    "wide": ((-120, 120), (-300, 300), (-127, 127), (-30000, 30000), (-127, 127), (-30000, 30000), (-127, 127), (-5000, 5000)),
}


def make(path, arch, seed, scale="normal"):
    """乱数の重みの nn.bin を path に書く。"""
    feat, W, L1, L2 = parse_arch(arch)
    rng = random.Random(seed)
    ft_hash, net_hash, file_hash = hashes(arch)
    ftw, ftb, l1w, l1b, l2w, l2b, ow, ob = RANGES[scale]
    with open(path, "wb") as f:
        a = arch_string(arch).encode()
        f.write(struct.pack("<III", VERSION, file_hash, len(a)))
        f.write(a)
        f.write(struct.pack("<I", ft_hash))
        f.write(_ints(rng, W, *ftb, "<i2").tobytes())
        chunk = 1 << 22
        left = num_features(feat) * W
        while left:
            n = min(left, chunk)
            f.write(_ints(rng, n, *ftw, "<i2").tobytes())
            left -= n
        f.write(struct.pack("<I", net_hash))
        f.write(_ints(rng, L1, *l1b, "<i4").tobytes())
        f.write(_padded(rng, L1, 2 * W, *l1w).tobytes())
        f.write(_ints(rng, L2, *l2b, "<i4").tobytes())
        f.write(_padded(rng, L2, L1, *l2w).tobytes())
        f.write(_ints(rng, 1, *ob, "<i4").tobytes())
        f.write(_padded(rng, 1, L2, *ow).tobytes())
    return path


def read_header(path):
    """(版, ファイルのハッシュ, 構造の文字列, 特徴変換のハッシュ, ネットワークのハッシュの位置のハッシュ値)。"""
    with open(path, "rb") as f:
        data = f.read()
    ver, h, n = struct.unpack_from("<III", data, 0)
    s = data[12:12 + n].decode()
    (fth,) = struct.unpack_from("<I", data, 12 + n)
    return ver, h, s, fth


if __name__ == "__main__":
    import sys
    make(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1,
         sys.argv[4] if len(sys.argv) > 4 else "normal")
