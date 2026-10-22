"""やねうら王の NNUE 形式（<特徴>_<W>x2-<L1>-<L2>）の nn.bin を読み書きする（設計書 §S11・§S12）。

対象は halfk… の特徴（HalfKP・HalfKA1・HalfKA2・HalfKA_hm1・HalfKA_hm2）で、psqt・k3k3 なしの構造。
ファイルの並び（すべてリトルエンディアン。tests/nnue_nnbin.py と同じ）：
  ヘッダ：u32 版（0x7AF32F16）、u32 ハッシュ（特徴変換 ^ ネットワーク）、u32 構造の文字列の長さ、構造の文字列
  特徴変換：u32 ハッシュ、int16 バイアス[W]、int16 重み[次元][W]
  ネットワーク：u32 ハッシュ、
    隠れ層1：int32 バイアス[L1]、int8 重み[L1][pad32(2W)]
    隠れ層2：int32 バイアス[L2]、int8 重み[L2][pad32(L1)]
    出力層 ：int32 バイアス[1]、 int8 重み[1][pad32(L2)]
  pad32 は 32 の倍数への切り上げ。切り上げで増えた列は 0。
"""
import struct

import numpy as np

VERSION = 0x7AF32F16
# 特徴 → (やねうら王の名前, 特徴のハッシュ, 玉の位置あたりの駒の番号の数（P）, 玉の位置の数)
FEATURES = {
    "halfkp": ("HalfKP(Friend)", 0x5D69D5B8, 1548, 81),
    "halfka1": ("HalfKA1(Friend)", 0x5F134CB8, 1710, 81),
    "halfka2": ("HalfKA2(Friend)", 0x5F234CB8, 1629, 81),
    "halfkahm1": ("HalfKA_hm1(Friend)", 0x7F134CB8, 1710, 45),
    "halfkahm2": ("HalfKA_hm2(Friend)", 0x7F234CB8, 1629, 45),
}
U32 = 0xFFFFFFFF


def pad32(n):
    return (n + 31) // 32 * 32


def num_features(feature):
    """nn.bin の特徴の次元（玉の位置 × 駒の番号）。因数分解は含まない。"""
    _, _, plane, squares = FEATURES[feature]
    return plane * squares


def arch_string(feature, W, L1, L2):
    """やねうら王の構造の文字列（ヘッダに入る）。"""
    name = FEATURES[feature][0]
    n = 2 * W
    st = f"InputSlice[{n}(0:{n})]"
    st = f"AffineTransformSparseInput[{L1}<-{n}]({st})"
    st = f"ClippedReLU[{L1}]({st})"
    st = f"AffineTransform[{L2}<-{L1}]({st})"
    st = f"ClippedReLU[{L2}]({st})"
    st = f"AffineTransform[1<-{L2}]({st})"
    return f"Features={name}[{num_features(feature)}->{W}x2],Network={st}"


def _affine_hash(prev, out_dims):
    return ((0xCC03DAE4 + out_dims) ^ (prev >> 1) ^ ((prev << 31) & U32)) & U32


def hashes(feature, W, L1, L2):
    """(特徴変換のハッシュ, ネットワークのハッシュ, ファイルのハッシュ)。"""
    ft = (FEATURES[feature][1] ^ (2 * W)) & U32
    h = (0xEC42E90D ^ (2 * W)) & U32
    h = (0x538D24C7 + _affine_hash(h, L1)) & U32
    h = (0x538D24C7 + _affine_hash(h, L2)) & U32
    net = _affine_hash(h, 1)
    return ft, net, ft ^ net


def file_size(feature, W, L1, L2):
    """書き出したときの大きさ（バイト）。"""
    arch = arch_string(feature, W, L1, L2).encode()
    return (12 + len(arch) + 4 + 2 * W + 2 * W * num_features(feature)
            + 4 + 4 * L1 + L1 * pad32(2 * W) + 4 * L2 + L2 * pad32(L1) + 4 + pad32(L2))


def write(path, feature, W, L1, L2, q):
    """量子化済みの配列から nn.bin を書く。q は次の形の dict（列は切り上げる前の数）：
    ft_b (W,) int16、ft_w (次元, W) int16、b1 (L1,) int32、w1 (L1, 2W) int8、
    b2 (L2,) int32、w2 (L2, L1) int8、bo (1,) int32、wo (1, L2) int8。"""
    ft_hash, net_hash, file_hash = hashes(feature, W, L1, L2)
    arch = arch_string(feature, W, L1, L2).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<III", VERSION, file_hash, len(arch)))
        f.write(arch)
        f.write(struct.pack("<I", ft_hash))
        f.write(np.ascontiguousarray(q["ft_b"], dtype="<i2").tobytes())
        f.write(np.ascontiguousarray(q["ft_w"], dtype="<i2").tobytes())
        f.write(struct.pack("<I", net_hash))
        f.write(np.ascontiguousarray(q["b1"], dtype="<i4").tobytes())
        f.write(_padded(q["w1"], pad32(2 * W)))
        f.write(np.ascontiguousarray(q["b2"], dtype="<i4").tobytes())
        f.write(_padded(q["w2"], pad32(L1)))
        f.write(np.ascontiguousarray(q["bo"], dtype="<i4").tobytes())
        f.write(_padded(q["wo"], pad32(L2)))


def _padded(a, cols):
    """行を cols 列に広げる（足りない列は 0）。int8 のバイト列を返す。"""
    a = np.asarray(a, dtype=np.int8)
    out = np.zeros((a.shape[0], cols), dtype=np.int8)
    out[:, :a.shape[1]] = a
    return out.tobytes()


def read(path, feature, W, L1, L2):
    """nn.bin を読んで、ヘッダとハッシュを確かめたうえで配列の dict（write と同じ形・列は切り上げる前）を返す。"""
    with open(path, "rb") as f:
        data = f.read()
    ver, file_hash, n = struct.unpack_from("<III", data, 0)
    if ver != VERSION:
        raise ValueError(f"nn.bin の版が違う: {ver:#x}")
    arch = arch_string(feature, W, L1, L2)
    if data[12:12 + n].decode() != arch:
        raise ValueError("nn.bin の構造の文字列が違う")
    ft_hash, net_hash, want_file_hash = hashes(feature, W, L1, L2)
    if file_hash != want_file_hash:
        raise ValueError("nn.bin のファイルのハッシュが違う")
    off = 12 + n
    if struct.unpack_from("<I", data, off)[0] != ft_hash:
        raise ValueError("nn.bin の特徴変換のハッシュが違う")
    off += 4

    def take(dt, shape, cols=None):
        """shape（行, 列）の配列を読む。cols は行の実際の列数（切り上げ前）。"""
        nonlocal off
        rows = shape[0]
        width = shape[1] if len(shape) > 1 else 1
        full = pad32(width) if cols is not None else width
        cnt = rows * full
        a = np.frombuffer(data, dt, cnt, off).reshape(rows, full)[:, :width]
        off += cnt * np.dtype(dt).itemsize
        return a.reshape(shape)

    out = {"ft_b": take("<i2", (W,)), "ft_w": take("<i2", (num_features(feature), W))}
    if struct.unpack_from("<I", data, off)[0] != net_hash:
        raise ValueError("nn.bin のネットワークのハッシュが違う")
    off += 4
    out["b1"] = take("<i4", (L1,))
    out["w1"] = take("i1", (L1, 2 * W), cols=True)
    out["b2"] = take("<i4", (L2,))
    out["w2"] = take("i1", (L2, L1), cols=True)
    out["bo"] = take("<i4", (1,))
    out["wo"] = take("i1", (1, L2), cols=True)
    if off != len(data):
        raise ValueError(f"nn.bin の長さが合わない（{off} / {len(data)}）")
    return out
