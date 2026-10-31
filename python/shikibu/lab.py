"""評価関数の構造の実験室（設計書 §S12）。

同じデータ・同じ学習量で、構造（特徴・幅・層）だけを変えて学習し、検証の損失（val_loss）で比べる。
ノートブック notebooks/shikibu_lab.ipynb から使う。注意：val_loss は「局面をどれだけ当てられるか」の目安で、
強さは対局で確かめる。比べるときは構造以外（データ・学習量・eval_scale・lam・乱数の種）を全部そろえる。
"""
import copy
import json
import os

import torch

from shikibu import export, networks, nnbin, train

# ノートブックの選択肢：構造の名前 → 日本語の説明（何が違うか）
PRESETS = {
    "raypair0_128x2-16-16": "今の RayPair_v0（基準。Nuenagon でそのまま使える）",
    "raypair0_256x2-32-32": "RayPair_v0 の幅を広げたもの（賢くなるが重くなる）",
    "raypair1_256x2-32-32": "RayPair_v1（領域つき）の幅を広げたもの",
    "raypair0k_256x2-32-32": "RayPair_v0 ＋ 玉の周りのブロック（独自。書き出しはまだできない）",
    "halfkp_256x2-32-32_f": "やねうら王の標準の HalfKP（広く使われている構造。nn.bin に書ける）",
    "halfkp_512x2-16-32_f": "HalfKP の幅 512（重くなるが、やねうら王でそのまま使える）",
    "halfkp_1024x2-8-32_f": "HalfKP の幅 1024（とても重い）",
    "halfkahm2_256x2-32-32_f": "HalfKA_hm2（玉も特徴に入れ、左右を畳む。nn.bin に書ける）",
    "halfkahm2_512x2-16-32_f": "HalfKA_hm2 の幅 512（nn.bin に書ける）",
    "halfkahm2_1024x2-8-32_f": "HalfKA_hm2 の幅 1024（重いが正確。nn.bin に書ける）",
    "halfkahm2_512x2-16-32_f_psqt": "HalfKA_hm2 ＋ 駒の点数の直接の項（独自。まだ書き出せない）",
    "halfkahm2_512x2-16-32_f_k3k3": "HalfKA_hm2 ＋ 玉の段で層を切り替える（独自。まだ書き出せない）",
}

# 特徴ごとの 1 局面 1 視点の特徴の最大数（推論のとき 1 回に入る要素の上限の目安）
MAX_ACTIVE = {"halfkp": 38, "raypair0": 208, "raypair1": 208, "raypair0k": 256, "raypair1k": 256}


def describe(name):
    """構造の性質を dict で返す。キー：canonical, feature, W, L1, L2, train_features（学習のときの次元）,
    infer_features（因数分解を畳んだ後）, params（学習する重みの数）, max_active, export_format（"nn.bin" / ".shk" / None）,
    yo_compatible, yo_arch, yo_edition, nnbin_bytes。"""
    p = networks.parse_arch(name)
    feat, W, L1, L2, S = p["feature"], p["W"], p["L1"], p["L2"], p["stacks"]
    infer = networks.FEATURES[feat][2]
    params = p["num_features"] * W + W + S * (L1 * 2 * W + L1 + L2 * L1 + L2 + L2 + 1)
    if p["psqt"]:
        params += p["num_features"] * S
    why = export.nnbin_support(p["canonical"])
    yo = why is None
    if yo:
        fmt = "nn.bin"
    elif feat in ("raypair0", "raypair1") and not p["psqt"] and S == 1:
        fmt = ".shk"
    else:
        fmt = None
    return {"canonical": p["canonical"], "feature": feat, "W": W, "L1": L1, "L2": L2,
            "train_features": p["num_features"], "infer_features": infer, "params": params,
            "max_active": MAX_ACTIVE.get(feat, 40), "export_format": fmt, "yo_compatible": yo,
            "yo_arch": export.yo_arch(name) if yo else None,
            "yo_edition": export.yo_edition(name) if yo else None,
            "nnbin_bytes": nnbin.file_size(feat, W, L1, L2) if yo else None}


def make_config(name, train_glob, out_dir, val_path="", overrides=None):
    """構造の名前から設定の dict を作る（train.load_config と同じ形）。

    overrides は {"train.max_steps": 1000} のような「セクション.キー」の dict。
    model.network は正規の名前、model.ft_width は構造の W にする。他は train.DEFAULTS の既定値。
    """
    p = networks.parse_arch(name)
    cfg = copy.deepcopy(train.DEFAULTS)
    cfg["data"]["train_glob"] = train_glob
    cfg["data"]["val_path"] = val_path
    cfg["output"]["out_dir"] = out_dir
    cfg["model"]["network"] = p["canonical"]
    cfg["model"]["ft_width"] = p["W"]
    for k, v in (overrides or {}).items():
        sec, _, key = k.partition(".")
        if sec not in cfg or key not in cfg[sec]:
            raise ValueError(f"未知の設定: {k}")
        cfg[sec][key] = v
    return cfg


def _toml(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return "[" + ", ".join(_toml(e) for e in v) + "]"
    return repr(v)


def write_config(cfg, path):
    """設定を TOML に書く（train.load_config で読める形）。"""
    with open(path, "w", encoding="utf-8") as f:
        for sec, kv in cfg.items():
            f.write(f"[{sec}]\n")
            for k, v in kv.items():
                f.write(f"{k} = {_toml(v)}\n")
            f.write("\n")
    return path


def _log(run_dir):
    with open(os.path.join(run_dir, "log.jsonl"), encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def compare_runs(run_dirs):
    """各 run の検証の損失を比べる。全部の run で検証がある最大の step（common_step）の val_loss で並べる。"""
    info = []
    for d in run_dirs:
        ck = torch.load(os.path.join(d, "checkpoint.pt"), map_location="cpu", weights_only=False)
        vals = [r for r in _log(d) if r.get("type") == "val"]
        info.append((d, ck, vals))
    common = None
    for _, _, vals in info:
        steps = {r["step"] for r in vals}
        common = steps if common is None else common & steps
    cs = max(common) if common else None
    rows = []
    for d, ck, vals in info:
        rows.append({"run_dir": d, "network": ck["network"], "step": ck["step"],
                     "positions": ck["step"] * ck["config"]["train"]["batch_size"],
                     "eval_scale": float(ck["eval_scale"]),
                     "val_loss": vals[-1]["val_loss"] if vals else None,
                     "best_val_loss": min(r["val_loss"] for r in vals) if vals else None,
                     "common_step": cs,
                     "val_loss_at_common": next((r["val_loss"] for r in vals if r["step"] == cs), None)})
    rows.sort(key=lambda r: (r["val_loss_at_common"] is None, r["val_loss_at_common"] or 0.0))
    return rows


def export_run(run_dir):
    """run の checkpoint を run_dir/export/ に書き出す。書き出せない構造は例外にせず、理由を返す。"""
    ckp = os.path.join(run_dir, "checkpoint.pt")
    d = describe(export.arch_of_checkpoint(ckp))
    if d["export_format"] is None:
        if d["feature"] in nnbin.FEATURES:
            reason = export.nnbin_support(d["canonical"])
        else:
            reason = (f"{d['feature']} は書き出せない（書き出せるのは、psqt・k3k3 なしの halfk…（nn.bin）と"
                      "raypair0・raypair1（.shk））")
        return {"format": None, "path": None, "reason": reason}
    out = os.path.join(run_dir, "export")
    os.makedirs(out, exist_ok=True)
    if d["export_format"] == "nn.bin":
        path = os.path.join(out, "nn.bin")
        r = export.export_nnbin(ckp, path)
    else:
        path = os.path.join(out, "nuenagon.shk")
        r = export.export_shk(ckp, path)
    return {"format": d["export_format"], "path": path, **r}
