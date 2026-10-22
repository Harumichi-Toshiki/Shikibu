"""やねうら王の NNUE 形式（halfkp_256x2-32-32）の nn.bin を読み書きする（設計書 §S11）。

ファイルの並び（すべてリトルエンディアン）：
  ヘッダ：u32 版（0x7AF32F16）、u32 ハッシュ、u32 構造の文字列の長さ、構造の文字列
  特徴変換：u32 ハッシュ、int16 バイアス[256]、int16 重み[125388][256]
  ネットワーク：u32 ハッシュ、
    隠れ層1：int32 バイアス[32]、int8 重み[32][512]
    隠れ層2：int32 バイアス[32]、int8 重み[32][32]
    出力層 ：int32 バイアス[1]、 int8 重み[1][32]
"""
import struct

import numpy as np

VERSION = 0x7AF32F16
N_FEATURES = 125388  # 81 × 1548
HALF = 256
ARCH = ("Features=HalfKP(Friend)[125388->256x2],Network=AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-32]"
        "(ClippedReLU[32](AffineTransformSparseInput[32<-512](InputSlice[512(0:512)])))))")
FEATURE_HASH = 0x5D69D5B8
U32 = 0xFFFFFFFF


def _u32(x):
    return x & U32


def _affine_hash(prev, out_dims):
    return _u32((0xCC03DAE4 + out_dims) ^ (prev >> 1) ^ _u32(prev << 31))


def _relu_hash(prev):
    return _u32(0x538D24C7 + prev)


def hashes():
    """(特徴変換のハッシュ, ネットワークのハッシュ, ファイルのハッシュ)。"""
    ft = _u32(FEATURE_HASH ^ (HALF * 2))
    h = _u32(0xEC42E90D ^ (HALF * 2) ^ (0 << 10))
    h = _relu_hash(_affine_hash(h, 32))
    h = _relu_hash(_affine_hash(h, 32))
    net = _affine_hash(h, 1)
    return ft, net, _u32(ft ^ net)


def write(path, ft_b, ft_w, b1, w1, b2, w2, bo, wo):
    """量子化済みの配列から nn.bin を書く。形は ft_b (256,) int16、ft_w (125388, 256) int16、
    b1 (32,) int32、w1 (32, 512) int8、b2 (32,) int32、w2 (32, 32) int8、bo (1,) int32、wo (1, 32) int8。"""
    ft_hash, net_hash, file_hash = hashes()
    arch = ARCH.encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<III", VERSION, file_hash, len(arch)))
        f.write(arch)
        f.write(struct.pack("<I", ft_hash))
        f.write(np.ascontiguousarray(ft_b, dtype="<i2").tobytes())
        f.write(np.ascontiguousarray(ft_w, dtype="<i2").tobytes())
        f.write(struct.pack("<I", net_hash))
        for arr, dt in ((b1, "<i4"), (w1, "i1"), (b2, "<i4"), (w2, "i1"), (bo, "<i4"), (wo, "i1")):
            f.write(np.ascontiguousarray(arr, dtype=dt).tobytes())


def read(path):
    """nn.bin を読んで、ヘッダの確認をしたうえで配列の dict を返す（書き出しの確認に使う）。"""
    with open(path, "rb") as f:
        data = f.read()
    ver, file_hash, n = struct.unpack_from("<III", data, 0)
    if ver != VERSION:
        raise ValueError(f"nn.bin の版が違う: {ver:#x}")
    if data[12:12 + n].decode() != ARCH:
        raise ValueError("nn.bin の構造の文字列が違う")
    ft_hash, net_hash, want_file_hash = hashes()
    if file_hash != want_file_hash:
        raise ValueError("nn.bin のファイルのハッシュが違う")
    off = 12 + n
    if struct.unpack_from("<I", data, off)[0] != ft_hash:
        raise ValueError("nn.bin の特徴変換のハッシュが違う")
    off += 4

    def take(dt, shape):
        nonlocal off
        cnt = int(np.prod(shape))
        a = np.frombuffer(data, dt, cnt, off).reshape(shape)
        off += cnt * np.dtype(dt).itemsize
        return a

    out = {"ft_b": take("<i2", (HALF,)), "ft_w": take("<i2", (N_FEATURES, HALF))}
    if struct.unpack_from("<I", data, off)[0] != net_hash:
        raise ValueError("nn.bin のネットワークのハッシュが違う")
    off += 4
    out["b1"], out["w1"] = take("<i4", (32,)), take("i1", (32, 512))
    out["b2"], out["w2"] = take("<i4", (32,)), take("i1", (32, 32))
    out["bo"], out["wo"] = take("<i4", (1,)), take("i1", (1, 32))
    if off != len(data):
        raise ValueError(f"nn.bin の長さが合わない（{off} / {len(data)}）")
    return out
