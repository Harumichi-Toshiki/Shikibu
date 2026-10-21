"""重みファイル（.shk）の書き出しと読み込み（設計書 §F4）。

使い方：python -m shikibu.export --checkpoint <checkpoint.pt> --out <file.shk>

- 書き出しのあとで読み直し、ヘッダ・CRC32・全テンソルが一致することを確かめる。
- すべてリトルエンディアン。本体は §F4.2 の順に float32 を隙間なく並べる。
"""
import argparse
import struct
import sys
import zlib

import numpy as np
import torch

from shikibu import networks

MAGIC = b"SHIKIBU\0"
FORMAT_VERSION = 1
HEADER_SIZE = 180
ORDER = ["ft_weight", "ft_bias", "l1_weight", "l1_bias", "l2_weight", "l2_bias", "out_weight", "out_bias"]
FEATURE_SETS = {0: "RayPair_v0", 1: "RayPair_v1"}


def _pack_header(h):
    b = bytearray(HEADER_SIZE)
    b[0:8] = MAGIC
    struct.pack_into("<I", b, 8, FORMAT_VERSION)
    b[12:44] = h["feature_set"].encode("ascii").ljust(32, b"\0")
    b[44:76] = h["network"].encode("ascii").ljust(32, b"\0")
    struct.pack_into("<4I", b, 76, h["num_features"], h["W"], h["L1"], h["L2"])
    struct.pack_into("<2f", b, 92, h["eval_scale"], h["lam"])
    struct.pack_into("<Q", b, 100, h["positions"])
    # 108〜171 は予約（0 のまま）
    struct.pack_into("<Q", b, 172, h["body_len"])
    return bytes(b)


def _cstr(b):
    return b.split(b"\0", 1)[0].decode("ascii")


def _shapes(h):
    n, w, l1, l2 = h["num_features"], h["W"], h["L1"], h["L2"]
    return {"ft_weight": (n, w), "ft_bias": (w,), "l1_weight": (l1, 2 * w), "l1_bias": (l1,),
            "l2_weight": (l2, l1), "l2_bias": (l2,), "out_weight": (1, l2), "out_bias": (1,)}


def read_shk(path):
    """.shk を読んで (ヘッダ, テンソル) を返す。形式が不正なら ValueError。"""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < HEADER_SIZE + 4:
        raise ValueError("ファイルが短すぎる")
    if data[0:8] != MAGIC:
        raise ValueError("マジックが違う（SHIKIBU ではない）")
    (ver,) = struct.unpack_from("<I", data, 8)
    if ver != FORMAT_VERSION:
        raise ValueError(f"形式バージョンが未対応: {ver}")
    if any(data[108:172]):
        raise ValueError("予約領域が 0 でない")
    n, w, l1, l2 = struct.unpack_from("<4I", data, 76)
    scale, lam = struct.unpack_from("<2f", data, 92)
    (positions,) = struct.unpack_from("<Q", data, 100)
    (body_len,) = struct.unpack_from("<Q", data, 172)
    if len(data) != HEADER_SIZE + body_len + 4:
        raise ValueError("ファイルの長さが本体の長さと合わない（切れている可能性）")
    body = data[HEADER_SIZE:HEADER_SIZE + body_len]
    (crc,) = struct.unpack_from("<I", data, HEADER_SIZE + body_len)
    if zlib.crc32(body) & 0xFFFFFFFF != crc:
        raise ValueError("CRC32 が一致しない")
    h = {"format_version": ver, "feature_set": _cstr(data[12:44]), "network": _cstr(data[44:76]),
         "num_features": n, "W": w, "L1": l1, "L2": l2, "eval_scale": float(scale), "lam": float(lam),
         "positions": positions}
    shapes = _shapes(h)
    if body_len != sum(int(np.prod(s)) for s in shapes.values()) * 4:
        raise ValueError("本体の長さが形と合わない")
    arr = np.frombuffer(body, dtype="<f4")
    tensors, pos = {}, 0
    for k in ORDER:
        c = int(np.prod(shapes[k]))
        tensors[k] = arr[pos:pos + c].reshape(shapes[k]).astype(np.float32)
        pos += c
    return h, tensors


def export_shk(checkpoint_path, out_path):
    """checkpoint.pt から .shk を書き出し、読み直して確かめる。ヘッダの dict を返す。"""
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    m = networks.build(ck["network"], ft_width=ck["config"]["model"]["ft_width"])
    m.load_state_dict(ck["model"])
    tensors = {k: v.numpy() for k, v in m.export_tensors().items()}
    body = b"".join(np.ascontiguousarray(tensors[k], dtype="<f4").tobytes() for k in ORDER)
    h = {"feature_set": FEATURE_SETS[m.FEATURE_VERSION], "network": ck["network"],
         "num_features": m.NUM_FEATURES, "W": m.ft_width, "L1": m.L1, "L2": m.L2,
         "eval_scale": float(ck["eval_scale"]), "lam": float(ck["lam"]),
         "positions": int(ck["step"]) * int(ck["config"]["train"]["batch_size"]), "body_len": len(body)}
    blob = _pack_header(h) + body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)
    with open(out_path, "wb") as f:
        f.write(blob)

    # 読み直して確かめる
    rh, rt = read_shk(out_path)
    for k in ("feature_set", "network", "num_features", "W", "L1", "L2", "positions"):
        if rh[k] != h[k]:
            raise ValueError(f"書き出しの確認に失敗（ヘッダ {k}）")
    if rh["eval_scale"] != float(np.float32(h["eval_scale"])) or rh["lam"] != float(np.float32(h["lam"])):
        raise ValueError("書き出しの確認に失敗（eval_scale / lam）")
    for k in ORDER:
        if not np.array_equal(rt[k], tensors[k]):
            raise ValueError(f"書き出しの確認に失敗（テンソル {k}）")
    return {k: rh[k] for k in ("format_version", "feature_set", "network", "num_features", "W", "L1", "L2",
                               "eval_scale", "lam", "positions")}


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m shikibu.export")
    p.add_argument("--checkpoint", required=True, help="学習の checkpoint.pt")
    p.add_argument("--out", required=True, help="書き出す .shk のパス")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        h = export_shk(args.checkpoint, args.out)
    except (ValueError, OSError, RuntimeError) as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    print(f"書き出しました: {args.out}")
    for k, v in h.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
