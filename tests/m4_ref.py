# 変更禁止（PROTECTED）。M4 の受け入れテスト用の、設計書から直接書いた独立の参照実装。
# .shk の読み書き（§F4）と、量子化した推論（§E4）。遅いが、仕様をそのまま書いたもの。
import struct
import zlib

import numpy as np

MAGIC = b"SHIKIBU\0"
HEADER_SIZE = 180
ORDER = ["ft_weight", "ft_bias", "l1_weight", "l1_bias", "l2_weight", "l2_bias", "out_weight", "out_bias"]


def shapes(n, w, l1, l2):
    return {"ft_weight": (n, w), "ft_bias": (w,), "l1_weight": (l1, 2 * w), "l1_bias": (l1,),
            "l2_weight": (l2, l1), "l2_bias": (l2,), "out_weight": (1, l2), "out_bias": (1,)}


def _cstr(b):
    return b.split(b"\0", 1)[0].decode("ascii")


def parse_shk(data):
    """bytes → (header dict, tensors dict[name → float32 ndarray])。不正なら ValueError。"""
    if len(data) < HEADER_SIZE + 4:
        raise ValueError("短すぎる")
    if data[0:8] != MAGIC:
        raise ValueError("マジックが違う")
    (ver,) = struct.unpack_from("<I", data, 8)
    if ver != 1:
        raise ValueError("形式バージョンが違う")
    feat = _cstr(data[12:44])
    net = _cstr(data[44:76])
    n, w, l1, l2 = struct.unpack_from("<4I", data, 76)
    (scale, lam) = struct.unpack_from("<2f", data, 92)
    (positions,) = struct.unpack_from("<Q", data, 100)
    reserved = data[108:172]
    (body_len,) = struct.unpack_from("<Q", data, 172)
    if any(reserved):
        raise ValueError("予約領域が0でない")
    if len(data) != HEADER_SIZE + body_len + 4:
        raise ValueError("ファイルの長さが合わない")
    body = data[HEADER_SIZE:HEADER_SIZE + body_len]
    (crc,) = struct.unpack_from("<I", data, HEADER_SIZE + body_len)
    if zlib.crc32(body) & 0xFFFFFFFF != crc:
        raise ValueError("CRC32 が合わない")
    sh = shapes(n, w, l1, l2)
    expect = sum(int(np.prod(s)) for s in sh.values()) * 4
    if body_len != expect:
        raise ValueError("本体の長さが形と合わない")
    arr = np.frombuffer(body, dtype="<f4")
    out, pos = {}, 0
    for k in ORDER:
        c = int(np.prod(sh[k]))
        out[k] = arr[pos:pos + c].reshape(sh[k]).astype(np.float32)
        pos += c
    header = {"format_version": ver, "feature_set": feat, "network": net, "num_features": n, "W": w,
              "L1": l1, "L2": l2, "eval_scale": scale, "lam": lam, "positions": positions}
    return header, out


def build_shk(header, tensors):
    """header dict と tensors から .shk の bytes を作る（テストで壊れたファイルを作るのにも使う）。"""
    sh = shapes(header["num_features"], header["W"], header["L1"], header["L2"])
    body = b"".join(np.ascontiguousarray(tensors[k], dtype="<f4").reshape(sh[k]).tobytes() for k in ORDER)
    h = bytearray(HEADER_SIZE)
    h[0:8] = MAGIC
    struct.pack_into("<I", h, 8, 1)
    h[12:12 + len(header["feature_set"])] = header["feature_set"].encode("ascii")
    h[44:44 + len(header["network"])] = header["network"].encode("ascii")
    struct.pack_into("<4I", h, 76, header["num_features"], header["W"], header["L1"], header["L2"])
    struct.pack_into("<2f", h, 92, header["eval_scale"], header["lam"])
    struct.pack_into("<Q", h, 100, header["positions"])
    struct.pack_into("<Q", h, 172, len(body))
    return bytes(h) + body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


# ---------------- 量子化（§E4） ----------------

def rnd(x):
    """0から遠い方への四捨五入（C++ の std::lround と同じ）。"""
    x = np.asarray(x, dtype=np.float64)
    return (np.sign(x) * np.floor(np.abs(x) + 0.5)).astype(np.int64)


def quantize(t):
    q = {"ft_weight": rnd(t["ft_weight"].astype(np.float64) * 127).astype(np.int16),
         "ft_bias": rnd(t["ft_bias"].astype(np.float64) * 127).astype(np.int16)}
    for k in ("l1", "l2", "out"):
        q[f"{k}_weight"] = np.clip(rnd(t[f"{k}_weight"].astype(np.float64) * 64), -127, 127).astype(np.int8)
        q[f"{k}_bias"] = rnd(t[f"{k}_bias"].astype(np.float64) * 127 * 64).astype(np.int32)
    return q


def accumulate(q, feats):
    acc = q["ft_bias"].astype(np.int64).copy()
    for i in feats:                       # 重複は回数ぶん加算
        acc += q["ft_weight"][i]
    assert np.all(np.abs(acc) <= 32767), "int16 の範囲を超えた"
    return acc


def qforward(q, stm_feats, nstm_feats):
    """量子化した推論。返り値：(手番側の蓄積値, 非手番側の蓄積値, 最終出力 raw の int)。"""
    a, b = accumulate(q, stm_feats), accumulate(q, nstm_feats)
    x = np.clip(np.concatenate([a, b]), 0, 127)
    h = x
    for k in ("l1", "l2"):
        s = q[f"{k}_weight"].astype(np.int64) @ h + q[f"{k}_bias"].astype(np.int64)
        h = np.clip(s >> 6, 0, 127)       # 算術右シフト（負は直後に0になる）
    raw = int((q["out_weight"].astype(np.int64) @ h)[0] + int(q["out_bias"][0]))
    return a, b, raw


def cp_of(raw, eval_scale):
    return int(rnd(raw * float(np.float32(eval_scale)) / 8128.0))
