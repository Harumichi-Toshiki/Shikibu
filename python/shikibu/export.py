"""重みファイルの書き出し（設計書 §F4・§S11・§S12）。

使い方：python -m shikibu.export --checkpoint <checkpoint.pt> --out <file>
  - raypair0・raypair1（psqt・k3k3 なし）なら .shk を書く（§F4）。どの幅でもよい。
  - halfk…（psqt・k3k3 なし）なら、やねうら王の nn.bin を書き、"FV_SCALE=<整数>" と
    "EDITION=<やねうら王の版の名前>" を表示する（§S11）。--fv-scale <整数> で FV_SCALE を指定できる。
  - それ以外（psqt・k3k3・raypair…k）は書き出せない（ValueError。CLI は終了コード 1）。

- 書き出しのあとで読み直し、ヘッダ・CRC32・全テンソル（nn.bin はハッシュ）が一致することを確かめる。
- .shk はすべてリトルエンディアン。本体は §F4.2 の順に float32 を隙間なく並べる。
"""
import argparse
import math
import struct
import sys
import zlib

import numpy as np
import torch

from shikibu import networks, nnbin

MAGIC = b"SHIKIBU\0"
FORMAT_VERSION = 1
HEADER_SIZE = 180
ORDER = ["ft_weight", "ft_bias", "l1_weight", "l1_bias", "l2_weight", "l2_bias", "out_weight", "out_bias"]
FEATURE_SETS = {0: "RayPair_v0", 1: "RayPair_v1"}
# やねうら王の出力の範囲の定数（Shikibu の量子化の式、§S11）
FV_NUMER = 8128
W_FT_SCALE, W_L_SCALE, B_SCALE = 127, 64, 8128


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


# ---------------------------------------------------------------- .shk

def _load(checkpoint_path):
    """checkpoint.pt を読み、(checkpoint, 重みを読み込んだネットワーク, 正規の構造の名前) を返す。"""
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    m = networks.build(ck["network"], ft_width=ck["config"]["model"]["ft_width"] or 128)
    m.load_state_dict(ck["model"])
    return ck, m, networks.arch_of(ck["network"], m.ft_width)


def arch_of_checkpoint(checkpoint_path):
    """checkpoint.pt の正規の構造の名前だけを返す（重みは読まない）。"""
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return networks.arch_of(ck["network"], ck["config"]["model"]["ft_width"] or 128)


def export_shk(checkpoint_path, out_path):
    """checkpoint.pt から .shk を書き出し、読み直して確かめる。ヘッダの dict を返す。
    書き出せるのは raypair0・raypair1 で psqt・k3k3 なしの構造だけ（それ以外は ValueError）。"""
    ck, m, arch = _load(checkpoint_path)
    p = networks.parse_arch(arch)
    if p["feature"] not in ("raypair0", "raypair1") or p["psqt"] or p["stacks"] != 1:
        raise ValueError(f"この構造は .shk に書き出せない（.shk は raypair0・raypair1 の psqt・k3k3 なしだけ）: {arch}")
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


# ---------------------------------------------------------------- やねうら王の nn.bin

def _round_away(x):
    """0 から遠い方への四捨五入（§S11）。"""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _quant(x, scale, dtype):
    info = np.iinfo(dtype)
    return np.clip(_round_away(np.asarray(x, dtype=np.float64) * scale), info.min, info.max).astype(dtype)


def nnbin_support(arch):
    """nn.bin に書き出せるなら None、書き出せないなら理由（日本語）。arch は正規の構造の名前。"""
    p = networks.parse_arch(arch)
    if p["feature"] not in nnbin.FEATURES:
        return f"nn.bin（やねうら王の形式）に書き出せるのは halfk… の特徴だけ（{p['feature']} は書き出せない）"
    if p["psqt"] or p["stacks"] != 1:
        return "psqt・k3k3 の構造は、まだ書き出せない（学習と val_loss の比較だけできる）"
    return None


def yo_arch(arch):
    """やねうら王の構造の名前（例 halfkahm2_512x2-16-32）。"""
    p = networks.parse_arch(arch)
    return f"{p['feature']}_{p['W']}x2-{p['L1']}-{p['L2']}"


def yo_edition(arch):
    """やねうら王のビルドの YANEURAOU_EDITION に入れる名前（例 YANEURAOU_ENGINE_NNUE_HALFKAHM2_512X2_16_32）。"""
    return "YANEURAOU_ENGINE_NNUE_" + yo_arch(arch).upper().replace("-", "_")


def export_nnbin(checkpoint_path, out_path, fv_scale=None):
    """checkpoint.pt からやねうら王の nn.bin を書き出し、読み直して確かめる（§S11・§S12）。
    戻り値は {"network", "arch", "eval_scale", "fv_scale", "yo_edition"}。
    fv_scale が None なら floor(8128 / eval_scale)。書き出せない構造は ValueError（理由つき）。"""
    ck, m, arch = _load(checkpoint_path)
    why = nnbin_support(arch)
    if why:
        raise ValueError(f"この構造は nn.bin に書き出せない: {why}")
    p = networks.parse_arch(arch)
    feature, W, L1, L2 = p["feature"], p["W"], p["L1"], p["L2"]
    es = float(ck["eval_scale"])
    if fv_scale is None:
        fv_scale = max(1, math.floor(FV_NUMER / es))
    if fv_scale < 1:
        raise ValueError(f"FV_SCALE は1以上: {fv_scale}")
    s = es * fv_scale / FV_NUMER
    if s > 1:
        raise ValueError(f"FV_SCALE が大きすぎる（s = {s:.3f} > 1 になると、出力が範囲を超える）")

    t = {k: v.double().numpy() for k, v in m.export_tensors().items()}
    w = t["ft_weight"]
    if p["factorize"]:
        # 因数分解つき：玉の位置に関係ない特徴を、各玉の位置の特徴へ足す（§S11）
        n_half = nnbin.num_features(feature)
        w = w[:n_half] + np.tile(w[n_half:], (nnbin.FEATURES[feature][3], 1))
    q = {"ft_b": _quant(t["ft_bias"], W_FT_SCALE, np.int16), "ft_w": _quant(w, W_FT_SCALE, np.int16),
         "b1": _quant(t["l1_bias"], B_SCALE, np.int32), "w1": _quant(t["l1_weight"], W_L_SCALE, np.int8),
         "b2": _quant(t["l2_bias"], B_SCALE, np.int32), "w2": _quant(t["l2_weight"], W_L_SCALE, np.int8),
         "bo": _quant(t["out_bias"] * s, B_SCALE, np.int32),
         "wo": _quant(t["out_weight"].reshape(1, L2) * s, W_L_SCALE, np.int8)}
    nnbin.write(out_path, feature, W, L1, L2, q)

    # 読み直して確かめる
    back = nnbin.read(out_path, feature, W, L1, L2)
    for k, v in q.items():
        if not np.array_equal(back[k].reshape(v.shape), v):
            raise ValueError(f"書き出しの確認に失敗（{k}）")
    return {"network": ck["network"], "arch": yo_arch(arch), "eval_scale": es, "fv_scale": fv_scale,
            "yo_edition": yo_edition(arch)}


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m shikibu.export")
    p.add_argument("--checkpoint", required=True, help="学習の checkpoint.pt")
    p.add_argument("--out", required=True, help="書き出すファイル（raypair は .shk、halfk… は nn.bin）")
    p.add_argument("--fv-scale", type=int, default=None,
                   help="nn.bin の FV_SCALE を指定する（省略時は floor(8128 / eval_scale)）")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if networks.parse_arch(arch_of_checkpoint(args.checkpoint))["feature"] in nnbin.FEATURES:
            h = export_nnbin(args.checkpoint, args.out, args.fv_scale)
            print(f"書き出しました（nn.bin）: {args.out}")
            print(f"  network: {h['network']}")
            print(f"  arch: {h['arch']}")
            print(f"  eval_scale: {h['eval_scale']}")
            print(f"FV_SCALE={h['fv_scale']}")
            print(f"EDITION={h['yo_edition']}")
            return 0
        if args.fv_scale is not None:
            raise ValueError("--fv-scale は halfk… の nn.bin を書くときだけ使う")
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
