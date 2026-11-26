"""DL の学習ループ（M11・M11.1）：詰めた局面から、方策と価値の 2 つの頭を持つネットワークを学習する。

使い方：python -m shikibu.train_dl --config <TOML> [--resume] [--force-unlock]

- 局面は shikibu_feat.BatchLoader(output="dl") から読む。55 チャンネルへの展開は dl_features.expand（GPU）。
- データは「組（source）」ごとに読む（M11.1）。組ごとに、方策を学ぶか（policy）・価値を学ぶか（value）・
  勝敗を学ぶか（result）を決め、1 バッチに weight の割合で混ぜる。設定に [[data.sources]] が無ければ、
  今の train_glob を 1 つの組（PSV、3 つとも学ぶ）として扱うので、M11 と同じ結果になる。
- 損失（教師の勝率は sigmoid(score / eval_scale)。項ごとに、その項を学ぶ局面だけで平均する）：
    loss = CE(p, move; label_smoothing) + value_loss_scale × (λ × CE(value_logits, hl_gauss(勝率)) + (1 − λ) × BCE(result_logit, result))
- 出力（out_dir）：checkpoint.pt・log.jsonl・status.json・train.lock。log の各行は "type" と "kind" に同じ値を持つ。
"""
import argparse
import glob
import os
import random
import signal
import sys
import tempfile
import threading
import time
import tomllib

import numpy as np
import torch
import torch.nn.functional as F

import shikibu_feat
from shikibu import dl_features, dl_networks
from shikibu.lock import TrainLock
from shikibu.train import (Status, JsonlLog, epoch_perm, eval_scale_setting, lr_factor, sample_scale_positions,
                           pick_device, seed_everything, rng_state, restore_rng, fmt_sec,
                           check_placeholders, PLACEHOLDER_KEYS, _on_term)

PSV_BYTES = 40
KEYS = ("board", "hands", "check", "move", "score", "result")          # 検証と、組の混ぜ方によらない部分
BATCH_KEYS = KEYS + ("sid", "pol", "val", "res", "scale")              # 学習のバッチ（組の情報を足す）
VAL_BATCH = 4096  # 検証の展開をまとめる大きさ（メモリを抑える）
OPTIMIZERS = ("adamw",)  # 今は AdamW だけ。後でオプティマイザを足すときは make_optimizer と、この表を直す
SOURCE_KEYS = ("name", "glob", "format", "weight", "policy", "value", "result", "eval_scale")
DL_PLACEHOLDER_KEYS = PLACEHOLDER_KEYS + [("data", "val_hcpe_glob")]
DRY_LOG_SEC = 30.0   # buffer_fill の間、読んだ件数を出す間隔（秒）

DEFAULTS = {
    "data": {
        "train_paths": [],
        "train_glob": "",
        "sources": [],            # [[data.sources]]（M11.1）。空なら train_glob を 1 つの組として扱う
        "val_path": "",
        "val_hcpe_glob": "",
        "cache_dir": "",
        "max_cached": 3,
        "shuffle_buffer": 0,
        "num_threads": 1,
    },
    "model": {"network": ""},
    "train": {
        "optimizer": "adamw",
        "batch_size": 4096,
        "lr": 4e-4,
        "weight_decay": 1e-3,
        "warmup_steps": 0,
        "total_steps": 0,
        "max_steps": 0,
        "min_lr_ratio": 0.0,
        "eval_scale": "auto",
        "scale_sample": 10_000_000,
        "value_mix_lambda": 0.5,
        "value_loss_scale": 1.5,
        "label_smoothing": 0.03,
        "grad_clip": 1.0,
        "amp": "bf16",
        "compile": True,
        "channels_last": True,
        "device": "auto",
        "seed": 0,
        "save_every_minutes": 30.0,
        "max_minutes": 0,
        "val_every_steps": 0,
        "val_positions": 100_000,
        "log_every_steps": 100,
        "progress_every_sec": 30.0,
    },
    "output": {"out_dir": ""},
}


# ---------- 設定 ----------

def load_config(path):
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"未知のセクション: {sorted(unknown)}")
    cfg = {}
    for sec, defaults in DEFAULTS.items():
        extra = set(raw.get(sec, {})) - set(defaults)
        if extra:
            raise ValueError(f"[{sec}] に未知のキー: {sorted(extra)}")
        cfg[sec] = {**defaults, **raw.get(sec, {})}
    return cfg


def _check_num(section, key, minimum):
    v = section[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v < minimum:
        raise ValueError(f"{key} は {minimum} 以上の数: {v!r}")


def validate_sources(sources):
    names = set()
    for i, src in enumerate(sources):
        if not isinstance(src, dict):
            raise ValueError(f"[[data.sources]] の {i + 1} 番目は表で書く")
        extra = set(src) - set(SOURCE_KEYS)
        if extra:
            raise ValueError(f"[[data.sources]] に未知のキー: {sorted(extra)}")
        name = src.get("name", f"source{i}")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"組の名前は空でない文字列で、重複しないこと: {name!r}")
        names.add(name)
        if not isinstance(src.get("glob", ""), str) or not src.get("glob"):
            raise ValueError(f"組 {name}：glob を指定する")
        if src.get("format", "psv") not in ("psv", "hcpe"):
            raise ValueError(f'組 {name}：format は "psv" か "hcpe"')
        w = src.get("weight", 1.0)
        if isinstance(w, bool) or not isinstance(w, (int, float)) or w <= 0:
            raise ValueError(f"組 {name}：weight は正の数")
        for key in ("policy", "value", "result"):
            if not isinstance(src.get(key, True), bool):
                raise ValueError(f"組 {name}：{key} は true か false")
        if not (src.get("policy", True) or src.get("value", True) or src.get("result", True)):
            raise ValueError(f"組 {name}：policy・value・result のどれも false にできない（学ぶものが無い）")
        eval_scale_setting(src.get("eval_scale", "auto"))


def validate(cfg):
    d, t = cfg["data"], cfg["train"]
    if not d["sources"] and not d["train_paths"] and not d["train_glob"]:
        raise ValueError("data.sources（[[data.sources]]）か data.train_glob を指定する")
    if d["sources"]:
        validate_sources(d["sources"])
    if not cfg["output"]["out_dir"]:
        raise ValueError("output.out_dir を指定する")
    if not cfg["model"]["network"]:
        raise ValueError("model.network（dl_… の名前）を指定する")
    dl_networks.parse_dl(cfg["model"]["network"])
    if t["optimizer"] not in OPTIMIZERS:
        raise ValueError(f"train.optimizer は {', '.join(OPTIMIZERS)} のどれか（今は adamw だけ）: {t['optimizer']!r}")
    if t["total_steps"] <= 0 or t["max_steps"] <= 0:
        raise ValueError("train.total_steps と train.max_steps は1以上")
    if t["batch_size"] <= 0 or d["num_threads"] <= 0:
        raise ValueError("batch_size と num_threads は1以上")
    if d["cache_dir"] and d["max_cached"] <= 0:
        raise ValueError("max_cached は1以上")
    eval_scale_setting(t["eval_scale"])
    if not 0.0 <= float(t["value_mix_lambda"]) <= 1.0:
        raise ValueError("value_mix_lambda は 0〜1")
    _check_num(t, "value_loss_scale", 0.0)
    if not 0.0 <= float(t["label_smoothing"]) < 1.0:
        raise ValueError("label_smoothing は 0 以上 1 未満")
    _check_num(t, "grad_clip", 0.0)
    if t["amp"] not in ("bf16", "fp16", "off"):
        raise ValueError('train.amp は "bf16"・"fp16"・"off" のどれか')
    for key in ("compile", "channels_last"):
        if not isinstance(t[key], bool):
            raise ValueError(f"train.{key} は true か false")
    _check_num(t, "log_every_steps", 1)
    _check_num(t, "save_every_minutes", 0.0)
    _check_num(t, "max_minutes", 0.0)
    _check_num(t, "val_every_steps", 0)
    _check_num(t, "progress_every_sec", 0.0)
    if d["val_path"] and t["val_positions"] <= 0:
        raise ValueError("val_positions は1以上")
    if t["scale_sample"] <= 0:
        raise ValueError("scale_sample は1以上")


def source_list(cfg):
    """学習に使う組の一覧（正規化済み）。[[data.sources]] が無ければ train_glob を 1 つの組にする（M11 と同じ）。"""
    d, t = cfg["data"], cfg["train"]
    if not d["sources"]:
        listed = list(d["train_paths"]) if (not d["train_glob"] and d["train_paths"]) else None
        return [{"name": "train", "glob": d["train_glob"], "paths": listed, "format": "psv", "weight": 1.0,
                 "policy": True, "value": True, "result": True, "eval_scale": t["eval_scale"]}]
    out = []
    for i, src in enumerate(d["sources"]):
        s = {"name": src.get("name", f"source{i}"), "glob": src["glob"], "paths": None,
             "format": src.get("format", "psv"), "weight": float(src.get("weight", 1.0)),
             "policy": bool(src.get("policy", True)), "value": bool(src.get("value", True)),
             "result": bool(src.get("result", True)), "eval_scale": src.get("eval_scale", "auto")}
        out.append(s)
    return out


def split_counts(weights, batch):
    """組ごとの 1 バッチの局面数。割合どおりに切り捨て、端数は最後の組が受け持つ（合計は batch）。"""
    tot = sum(weights)
    counts = [int(batch * w / tot) for w in weights[:-1]]
    counts.append(batch - sum(counts))
    if any(c <= 0 for c in counts):
        raise ValueError(f"batch_size（{batch}）が小さすぎて、組に 1 局面も割り当てられない（weight を見直す）")
    return counts


def split_threads(weights, threads):
    tot = sum(weights)
    return [max(1, round(threads * w / tot)) for w in weights]


# ---------- 部品 ----------

def make_optimizer(name, model, lr, weight_decay, fused=False):
    """オプティマイザを作る。名前から選ぶ口はここだけ（後で Muon などを足すときは、ここを直す）。"""
    if name == "adamw":
        kw = {"fused": True} if fused else {}
        return torch.optim.AdamW(dl_networks.param_groups(model, weight_decay), lr=lr, **kw)
    raise ValueError(f"未対応のオプティマイザ: {name}")


def loss_terms(p, value_logits, result_logit, move, score, result, eval_scale, lam, label_smoothing):
    """損失の各項（すべての局面を使う。M11 の形。速さの測定で使う）。返り値は (loss_p, loss_v, loss_r)。"""
    p = p.float()
    loss_p = F.cross_entropy(p, move, label_smoothing=label_smoothing)
    target_win = torch.sigmoid(score / eval_scale)
    hl = dl_features.hl_gauss(target_win).to(value_logits.device)
    loss_v = -(hl * F.log_softmax(value_logits.float(), dim=1)).sum(dim=1).mean()
    loss_r = F.binary_cross_entropy_with_logits(result_logit.float().view(-1), result)
    return loss_p, loss_v, loss_r


def masked_terms(p, value_logits, result_logit, x, label_smoothing):
    """組の混ぜ方の損失（M11.1）。各項は、その項を学ぶ局面だけで平均する（対象が 0 なら 0）。

    x の "pol"・"val"・"res" は局面ごとの 0/1 のマスク、"sid" は組の番号、"scale" は局面の eval_scale。
    返り値：(loss_p, loss_v, loss_r, per_sample)。per_sample は組ごとの集計に使う (ce_p, ce_v, bce)。
    """
    p = p.float()
    move = x["move"].clamp(min=0)
    ce_p = F.cross_entropy(p, move, reduction="none", label_smoothing=label_smoothing)
    pol = x["pol"].float()
    loss_p = (ce_p * pol).sum() / pol.sum().clamp(min=1.0)
    target_win = torch.sigmoid(x["score"] / x["scale"])
    hl = dl_features.hl_gauss(target_win).to(value_logits.device)
    ce_v = -(hl * F.log_softmax(value_logits.float(), dim=1)).sum(dim=1)
    val = x["val"].float()
    loss_v = (ce_v * val).sum() / val.sum().clamp(min=1.0)
    bce = F.binary_cross_entropy_with_logits(result_logit.float().view(-1), x["result"], reduction="none")
    res = x["res"].float()
    loss_r = (bce * res).sum() / res.sum().clamp(min=1.0)
    return loss_p, loss_v, loss_r, (ce_p, ce_v, bce)


def total_loss(loss_p, loss_v, loss_r, lam, value_loss_scale):
    return loss_p + value_loss_scale * (lam * loss_v + (1.0 - lam) * loss_r)


class DLPrefetcher:
    """次のバッチの転送を、今のステップの計算と重ねる（M8 と同じ考え方）。CUDA では固定メモリ 2 面と転送のストリーム。"""

    def __init__(self, device, keys=BATCH_KEYS):
        self.device = device
        self.cuda = device.type == "cuda"
        self.keys = keys
        self.stream = torch.cuda.Stream(device) if self.cuda else None
        self.slots = [{}, {}]
        self.events = [None, None]
        self.n = 0

    def send(self, batch):
        if not self.cuda:
            return {k: torch.from_numpy(batch[k]).to(self.device) for k in self.keys}, None
        face = self.n % 2
        self.n += 1
        if self.events[face] is not None:
            self.events[face].synchronize()
        out = {}
        with torch.cuda.stream(self.stream):
            for k in self.keys:
                a = torch.from_numpy(batch[k])
                buf = self.slots[face].get(k)
                if buf is None or buf.numel() < a.numel() or buf.dtype != a.dtype:
                    buf = torch.empty(a.numel(), dtype=a.dtype, pin_memory=True)
                    self.slots[face][k] = buf
                view = buf[: a.numel()].view(a.shape)
                view.copy_(a)
                out[k] = view.to(self.device, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self.stream)
        self.events[face] = ev
        return out, ev

    def ready(self, x, ev):
        if ev is None:
            return x
        cur = torch.cuda.current_stream(self.device)
        cur.wait_event(ev)
        for t in x.values():
            t.record_stream(cur)
        return x


def autocast(device, amp):
    """amp："bf16"（CPU・CUDA）・"fp16"（CUDA のみ）・"off"。"""
    if amp == "off":
        return torch.autocast(device.type, enabled=False)
    return torch.autocast(device.type, dtype=torch.bfloat16 if amp == "bf16" else torch.float16)


@torch.no_grad()
def evaluate(model, val, eval_scale, device, amp, channels_last):
    """検証（psv・hcpe の 1 つ）。返り値：policy_acc・loss_p・value_mse・result_acc（引き分けを除く）・n。"""
    model.eval()
    n = val["move"].shape[0]
    correct = 0
    lp_sum = 0.0
    se_sum = 0.0
    res_ok = res_n = 0
    for i in range(0, n, VAL_BATCH):
        sl = slice(i, min(n, i + VAL_BATCH))
        x = dl_features.expand(val["board"][sl], val["hands"][sl], val["check"][sl])
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with autocast(device, amp):
            p, v, r = model(x)
        p, v, r = p.float(), v.float().view(-1), r.float().view(-1)
        move = val["move"][sl]
        correct += int((p.argmax(dim=1) == move).sum())
        lp_sum += float(F.cross_entropy(p, move, reduction="sum"))
        target = torch.sigmoid(val["score"][sl] / eval_scale)
        se_sum += float(((v - target) ** 2).sum())
        res = val["result"][sl]
        keep = res != 0.5
        res_ok += int(((r[keep] > 0.5) == (res[keep] > 0.5)).sum())
        res_n += int(keep.sum())
    model.train()
    return {
        "n": n,
        "policy_acc": correct / n,
        "loss_p": lp_sum / n,
        "value_mse": se_sum / n,
        "result_acc": res_ok / res_n if res_n else 0.0,
    }


def load_hcpe_validation(glob_pat, n):
    """hcpe の検証データの先頭 n 局面を、詰めた局面の dict にする（Dainagon のテストセット。cshogi で読む）。"""
    import cshogi
    files = sorted(glob.glob(glob_pat))
    if not files:
        return None
    parts = []
    got = 0
    for p in files:
        take = np.fromfile(p, dtype=cshogi.HuffmanCodedPosAndEval, count=max(0, n - got))
        if len(take):
            parts.append(dl_features.from_hcpe(take))
            got += len(take)
        if got >= n:
            break
    if not parts:
        return None
    return {k: np.concatenate([d[k] for d in parts]) for k in KEYS} | {"n": got}


def to_tensors(d, device):
    """検証データ（numpy の dict）を device のテンソルにする。"""
    out = {k: torch.from_numpy(np.ascontiguousarray(d[k])).to(device) for k in KEYS}
    out["score"] = out["score"].float()
    out["result"] = out["result"].float()
    return out


def estimate_scale_from_samples(score, result):
    """評価値の尺度の推定（hcpe の組）。Rust の estimate_eval_scale と同じ目的：|score| ≥ 30000 を除き、
    BCE(sigmoid(score / S), r) を log S について黄金分割探索（範囲 [50, 5000]）で最小化する（r は勝 1・引分 0.5・負 0）。"""
    keep = np.abs(score) < 30000
    s = score[keep].astype(np.float64)
    r = result[keep].astype(np.float64)
    if len(s) == 0:
        raise RuntimeError("尺度の推定に使える局面が無い")

    def loss(log_scale):
        p = 1.0 / (1.0 + np.exp(-s / np.exp(log_scale)))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-np.mean(r * np.log(p) + (1 - r) * np.log(1 - p)))

    lo, hi = np.log(50.0), np.log(5000.0)
    g = (np.sqrt(5.0) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = loss(a), loss(b)
    while hi - lo > 1e-4:
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = loss(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = loss(b)
    return float(np.exp((lo + hi) / 2))


# ---------- 学習 ----------

class DLTrainer:
    def __init__(self, cfg, resume, status):
        self.cfg = cfg
        self.status = status
        d, m, t, o = cfg["data"], cfg["model"], cfg["train"], cfg["output"]
        self.d, self.t = d, t
        self.out_dir = o["out_dir"]
        self.ckpt_path = os.path.join(self.out_dir, "checkpoint.pt")
        self.network = dl_networks.parse_dl(m["network"])["canonical"]
        self.lam = float(t["value_mix_lambda"])
        self.vls = float(t["value_loss_scale"])
        self.saved_step = -1
        self.timing = {}
        self.sources = source_list(cfg)
        for src in self.sources:
            src["paths"] = self._source_paths(src)
        self.names = [s["name"] for s in self.sources]
        self.counts = split_counts([s["weight"] for s in self.sources], t["batch_size"])
        self.threads = split_threads([s["weight"] for s in self.sources], d["num_threads"])
        self.shuffles = [int(d["shuffle_buffer"] * s["weight"] / sum(x["weight"] for x in self.sources))
                         for s in self.sources]

        ck = None
        if os.path.exists(self.ckpt_path):
            if not resume:
                raise FileExistsError(f"{self.ckpt_path} がある。続けるなら --resume を付ける（上書きはしない）")
            ck = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
            if ck["network"] != self.network or ck.get("optimizer_name") != t["optimizer"]:
                raise ValueError("checkpoint の network・optimizer が設定と違う")
            if "source_names" in ck and ck["source_names"] != self.names:
                raise ValueError(f"checkpoint の組（{ck['source_names']}）が設定（{self.names}）と違う")
        elif resume:
            print("checkpoint.pt が無いので最初から学習します", file=sys.stderr)

        self.device = pick_device(t["device"])
        if t["amp"] == "fp16" and self.device.type != "cuda":
            raise ValueError('train.amp = "fp16" は CUDA のときだけ使える（CPU なら "bf16" か "off"）')
        seed_everything(t["seed"])
        self.model = dl_networks.build_dl(self.network).to(self.device)
        if t["channels_last"]:
            self.model = self.model.to(memory_format=torch.channels_last)
        self.fwd = torch.compile(self.model) if t["compile"] else self.model
        self.optimizer = make_optimizer(t["optimizer"], self.model, t["lr"], t["weight_decay"],
                                        fused=self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda") if (t["amp"] == "fp16" and self.device.type == "cuda") else None
        self.prefetch = DLPrefetcher(self.device, BATCH_KEYS)

        if ck is None:
            t0 = time.monotonic()
            self.eval_scales = self._decide_eval_scales()
            self.timing["data_check"] = time.monotonic() - t0
            self.step = 0
            self.states = [{"epoch": 0, "perm": epoch_perm(len(s["paths"]), self._seed(i, 0), 0), "loader_state": None}
                           for i, s in enumerate(self.sources)]
            self.bad_base = 0
        else:
            self.model.load_state_dict(ck["model"])
            self.optimizer.load_state_dict(ck["optimizer"])
            if self.scaler is not None and ck.get("scaler") is not None:
                self.scaler.load_state_dict(ck["scaler"])
            # M11 の checkpoint には eval_scales が無い（単独の組だった）
            self.eval_scales = ck.get("eval_scales") or {self.names[0]: ck["eval_scale"]}
            self.step = ck["step"]
            self.states = ck["sources_state"] if "sources_state" in ck else [
                {"epoch": ck["epoch"], "perm": ck.get("file_perm"), "loader_state": ck["loader_state"]}]
            for i, s in enumerate(self.sources):
                st = self.states[i]
                if st["perm"] is None:
                    st["perm"] = list(range(len(s["paths"])))
                if sorted(st["perm"]) != list(range(len(s["paths"]))):
                    raise ValueError("checkpoint の file_perm が学習データのファイル数と合わない")
            self.bad_base = int(ck.get("bad_records", 0))
            restore_rng(ck["rng"])

        self.eval_scale = self._first_value_scale()
        self.val = self._load_validation()
        self.log = JsonlLog(os.path.join(self.out_dir, "log.jsonl"))
        self.t0 = time.monotonic()
        self.last_save = self.t0
        self.last_log = self.t0
        self.pos_since_log = 0
        self.pos_since_print = 0
        self.loaders = [None] * len(self.sources)
        self.bad_closed = 0
        self.bad_records = self.bad_base
        self.bad_announced = False
        self.cache_seen = [0] * len(self.sources)
        self._write_start()
        for i in range(len(self.sources)):
            self._open_loader(i, self.states[i]["loader_state"], self.states[i]["epoch"], self.states[i]["perm"])
        self.loader_states = [dict(ld.state()) for ld in self.loaders]

    def _seed(self, i, epoch):
        return self.t["seed"] + 1000 * i + epoch

    def _source_paths(self, src):
        """組のファイル（glob）。検証用の PSV（val_path）は学習から外す（M4 と同じ）。"""
        paths = list(src["paths"]) if src.get("paths") else sorted(glob.glob(src["glob"]))
        vp = self.d["val_path"]
        if vp:
            val = os.path.realpath(vp)
            paths = [x for x in paths if os.path.realpath(x) != val]
        if not paths:
            raise ValueError(f"組 {src['name']} の学習データが1つも無い（glob: {src['glob']}）")
        return paths

    def _first_value_scale(self):
        for s in self.sources:
            if s["value"] and self.eval_scales.get(s["name"]) is not None:
                return float(self.eval_scales[s["name"]])
        return 600.0

    # 開始前
    def _decide_eval_scales(self):
        """組ごとに、勝敗の確認と eval_scale の決定（M11 の _decide_eval_scale を組ごとに）。"""
        t = self.t
        self.status.enter("data_check")
        scales = {}
        for s in self.sources:
            setting = eval_scale_setting(s["eval_scale"])
            if s["format"] == "hcpe":
                scales[s["name"]] = self._hcpe_scale(s, setting) if (s["value"] and setting is None) else (
                    setting if setting is not None else None)
                self._check_moves(s)
                continue
            scales[s["name"]] = self._psv_scale(s, setting)
            self._check_moves(s)
        return scales

    def _psv_scale(self, s, setting):
        t = self.t
        sample_path = os.path.join(tempfile.gettempdir(), f"shikibu_dl_scale_{s['name']}.psv")
        files = None
        try:
            files, count = sample_scale_positions(s["paths"], t["scale_sample"], t["seed"], sample_path)
            if count == 0:
                raise RuntimeError(f"組 {s['name']} に局面が1つも無い")
            if s["result"] or s["value"]:
                stats = shikibu_feat.psv_stats(files)
                if stats["result_missing"]:
                    raise RuntimeError(f"組 {s['name']}：勝敗が欠損している（psv_stats）。DL の学習は勝敗を使うので、データを確かめる")
            if not s["value"]:
                return None
            if setting is not None:
                return setting
            self.status.enter("eval_scale")
            return float(shikibu_feat.estimate_eval_scale(files, sample=min(count, t["scale_sample"]),
                                                         seed=t["seed"])["eval_scale"])
        finally:
            if files is not None and files != s["paths"] and os.path.exists(sample_path):
                os.remove(sample_path)

    def _hcpe_scale(self, s, setting):
        """hcpe の組の尺度：先頭の数十万局面から推定する（Python の実装。PSV の Rust の推定とは標本が違う）。"""
        if setting is not None:
            return setting
        self.status.enter("eval_scale")
        n = min(self.t["scale_sample"], 2_000_000)
        ld = shikibu_feat.BatchLoader(s["paths"], n, 0, 0, self.t["seed"], self.d["num_threads"],
                                      output="dl", fmt="hcpe", require_move=False)
        b = next(ld, None)
        if b is None:
            raise RuntimeError(f"組 {s['name']} の局面が読めない")
        return estimate_scale_from_samples(b["score"], b["result"])

    def _check_moves(self, s):
        """policy = true の組は、指し手のある局面が実際にあるか、学習の前に確かめる（M11.1）。
        指し手の無いデータを policy = true にすると、全部捨てて永遠に待つので、ここで止める。"""
        if not s["policy"]:
            return
        n = 20000
        ld = shikibu_feat.BatchLoader(s["paths"], n, 0, 0, self.t["seed"], 1,
                                      output="dl", fmt=s["format"], require_move=False)
        b = next(ld, None)
        if b is None:
            raise RuntimeError(f"組 {s['name']} の局面が読めない（形式 {s['format']} を確かめる）")
        with_move = int((b["move"] >= 0).sum())
        if with_move == 0:
            raise RuntimeError(f"組 {s['name']}：指し手のある局面が1つも無い（policy = true は指し手が要る。"
                               f"指し手の無いデータは policy = false にする）")

    def _load_validation(self):
        """検証データ：psv（val_path）と hcpe（val_hcpe_glob）。先頭 val_positions 局面を device に置く。"""
        t, d = self.t, self.d
        vals = {}
        if d["val_path"]:
            loader = shikibu_feat.BatchLoader([d["val_path"]], t["val_positions"], 0, 0, 0,
                                              d["num_threads"], output="dl")
            batch = next(loader, None)
            n_bad = loader.skipped()
            del loader
            if n_bad > 0:
                print(f"検証データで壊れたレコードを飛ばしました（{n_bad} 件）", flush=True)
            if batch is None:
                raise RuntimeError("検証データ（val_path）が空")
            vals["psv"] = (to_tensors(batch, self.device), None)  # None なら学習の eval_scale（学習と同じ尺度）
        if d["val_hcpe_glob"]:
            hv = load_hcpe_validation(d["val_hcpe_glob"], t["val_positions"])
            if hv is not None:
                vals["hcpe"] = (to_tensors(hv, self.device), 600.0)
        return vals or None

    def _open_loader(self, i, state, epoch, perm):
        """組 i の読み込み。キャッシュは組ごとのフォルダ（BatchLoader は作るとき、フォルダの shk_cache_ を消すため）。"""
        d, t, s = self.d, self.t, self.sources[i]
        if self.loaders[i] is not None:
            self.bad_closed += self.loaders[i].skipped()
        self.loaders[i] = None
        self.states[i]["epoch"], self.states[i]["perm"] = epoch, perm
        cache = os.path.join(d["cache_dir"], f"src{i}") if d["cache_dir"] else None
        if cache:
            os.makedirs(cache, exist_ok=True)
        paths = [s["paths"][j] for j in perm]
        self.loaders[i] = shikibu_feat.BatchLoader(
            paths, self.counts[i], 0, shuffle_buffer=self.shuffles[i], seed=self._seed(i, epoch),
            num_threads=self.threads[i], cache_dir=cache, max_cached=d["max_cached"],
            resume_state=state, output="dl", fmt=s["format"], require_move=s["policy"],
        )

    def _fetch(self):
        """各組から 1 バッチずつ読み、1 つにまとめて device へ送り始める。組ごとの情報（マスク）も付ける。"""
        parts = []
        for i, s in enumerate(self.sources):
            b = next(self.loaders[i], None)
            if b is None:  # この組は 1 周終わった。次の周回は、この組だけ並べ替えてから
                epoch = self.states[i]["epoch"] + 1
                self._open_loader(i, None, epoch, epoch_perm(len(s["paths"]), self._seed(i, epoch), epoch))
                b = next(self.loaders[i], None)
                if b is None:
                    raise RuntimeError(f"組 {s['name']} の局面が1つも読めなかった")
            parts.append((i, b))
        self._log_cache_events()
        batch = self._combine(parts)
        bad = self.bad_base + self.bad_closed + sum(ld.skipped() for ld in self.loaders)
        if bad > 0 and not self.bad_announced:
            print(f"壊れたレコードを飛ばしました（ここまでに {bad} 件）。学習は続けます", flush=True)
            self.bad_announced = True
        meta = {"states": [{"epoch": self.states[i]["epoch"], "perm": self.states[i]["perm"],
                            "loader_state": dict(self.loaders[i].state())} for i in range(len(self.sources))],
                "bad": bad, "counts": {self.names[i]: len(b["move"]) for i, b in parts}}
        x, ev = self.prefetch.send(batch)
        return x, ev, meta

    def _combine(self, parts):
        cols = {k: [] for k in KEYS + ("sid", "pol", "val", "res", "scale")}
        for i, b in parts:
            s = self.sources[i]
            n = len(b["move"])
            for k in KEYS:
                cols[k].append(b[k])
            cols["sid"].append(np.full(n, i, dtype=np.uint8))
            cols["pol"].append(((b["move"] >= 0) & s["policy"]).astype(np.uint8))
            cols["val"].append(np.full(n, 1 if s["value"] else 0, dtype=np.uint8))
            cols["res"].append(np.full(n, 1 if s["result"] else 0, dtype=np.uint8))
            scale = self.eval_scales.get(s["name"]) or 1.0
            cols["scale"].append(np.full(n, float(scale), dtype=np.float32))
        return {k: np.concatenate(v) for k, v in cols.items()}

    def _log_cache_events(self):
        if not self.d["cache_dir"]:
            return
        for i, ld in enumerate(self.loaders):
            evs = ld.cache_events()
            for e in evs[self.cache_seen[i]:]:
                msg = f"キャッシュのコピーが失敗（組 {self.names[i]}・{e['file']}・{e['attempt']} 回目）：{e['error']}"
                print(msg, file=sys.stderr, flush=True)
                self._log({"kind": "cache_event", "step": self.step, "source": self.names[i], **e})
            self.cache_seen[i] = len(evs)

    def _write_start(self):
        self._log({
            "kind": "start", "step": self.step, "resumed": self.step > 0,
            "eval_scales": self.eval_scales, "eval_scale": self.eval_scale,
            "network": self.network, "device": str(self.device),
            "sources": [{"name": s["name"], "format": s["format"], "weight": s["weight"], "policy": s["policy"],
                         "value": s["value"], "result": s["result"], "n_files": len(s["paths"]),
                         "batch_per_step": self.counts[i], "threads": self.threads[i]}
                        for i, s in enumerate(self.sources)],
            "config": self.cfg,
        })

    def _log(self, rec):
        """1 行を書く。"type" と "kind" は同じ値（shk_monitor と受け入れテストのため）。"""
        rec = {"type": rec["kind"], **rec}
        self.log.write(rec)

    def save(self, reason):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_name": self.t["optimizer"],
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "step": self.step,
            "epoch": self.states[0]["epoch"],
            "loader_state": self.states[0]["loader_state"],
            "sources_state": self.states,
            "source_names": self.names,
            "eval_scales": self.eval_scales,
            "eval_scale": self.eval_scale,
            "network": self.network,
            "config": self.cfg,
            "rng": rng_state(),
            "file_perm": self.states[0]["perm"],
            "bad_records": self.bad_records,
        }
        tmp = self.ckpt_path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.ckpt_path)
        self.last_save = time.monotonic()
        self.saved_step = self.step
        self._log({"kind": "save", "step": self.step, "epoch": self.states[0]["epoch"], "reason": reason})

    def _train_step(self, x, ev, meta):
        x = self.prefetch.ready(x, ev)
        t = self.t
        lr = t["lr"] * lr_factor(self.step, t["warmup_steps"], t["total_steps"], t["min_lr_ratio"])
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        feats = dl_features.expand(x["board"], x["hands"], x["check"])
        if t["channels_last"]:
            feats = feats.contiguous(memory_format=torch.channels_last)
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device, t["amp"]):
            p, vl, rl = self.fwd(feats)
        loss_p, loss_v, loss_r, per = masked_terms(p, vl, rl, x, float(t["label_smoothing"]))
        loss = total_loss(loss_p, loss_v, loss_r, self.lam, self.vls)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(t["grad_clip"]))
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(t["grad_clip"]))
            self.optimizer.step()
        self.states = [{"epoch": st["epoch"], "perm": st["perm"], "loader_state": st["loader_state"]}
                       for st in meta["states"]]
        self.bad_records = meta["bad"]
        self.step += 1
        n = int(x["move"].shape[0])
        self.pos_since_log += n
        self.pos_since_print += n
        self._last_per = (x["sid"], per, x["pol"], x["val"], x["res"])
        return loss, loss_p, loss_v, loss_r, lr

    def _per_source_row(self):
        """組ごとの局面数と、各項の損失（ログ用。学習の損失には使わない）。"""
        sid, (ce_p, ce_v, bce), pol, val, res = self._last_per
        out = {}
        for i, s in enumerate(self.sources):
            m = (sid == i).float()
            pm = pol.float() * m
            vm = val.float() * m
            rm = res.float() * m
            out[s["name"]] = {
                "n": int(m.sum().item()),
                "loss_p": float(((ce_p * pm).sum() / pm.sum().clamp(min=1.0)).item()),
                "loss_v": float(((ce_v * vm).sum() / vm.sum().clamp(min=1.0)).item()),
                "loss_r": float(((bce * rm).sum() / rm.sum().clamp(min=1.0)).item()),
            }
        return out

    def _time_up(self):
        mm = float(self.t["max_minutes"])
        return mm > 0 and time.monotonic() - self.t0 >= mm * 60.0

    def _report(self, now):
        t = self.t
        done = self.step - self.step_train0
        eta = max(0.0, (t["max_steps"] - self.step) * (now - self.t_train0) / done) if done > 0 else None
        pps = self.pos_since_print / max(now - self.last_print, 1e-9)
        pct = 100.0 * self.step / t["max_steps"]
        print(f"[{pct:5.1f}%] step {self.step:,} / {t['max_steps']:,}  残り {fmt_sec(eta)}  {pps / 1e4:.1f}万局面/秒",
              flush=True)
        self.status.progress(self.step, t["max_steps"], eta)
        self.last_print, self.pos_since_print, self.printed_step = now, 0, self.step

    def _fill_monitor(self, stop):
        """buffer_fill の間、読んだ件数（理由別）を 30 秒ごとに標準エラーと log.jsonl に出す（M11.1）。"""
        handles = [ld.stats() for ld in self.loaders]
        while not stop.wait(DRY_LOG_SEC):
            snap = [h.snapshot() for h in handles]
            msg = "; ".join(f"{n}: 読んだ {s['processed']:,}・使えた {s['kept']:,}"
                            f"（読み解けない {s['unreadable']:,}・勝敗の範囲外 {s['result_range']:,}・指し手なし {s['no_move']:,}）"
                            for n, s in zip(self.names, snap))
            print("バッファを充填中：" + msg, file=sys.stderr, flush=True)
            self._log({"kind": "fill", "step": self.step, "stats": dict(zip(self.names, snap))})

    def _validate(self):
        for source, (val, scale) in self.val.items():
            s = self.eval_scale if scale is None else scale
            r = evaluate(self.model, val, s, self.device, self.t["amp"], self.t["channels_last"])
            self._log({"kind": "val", "step": self.step, "source": source, **r})

    def run(self):
        t = self.t
        save_interval = t["save_every_minutes"] * 60.0
        if self.d["shuffle_buffer"] > 0:
            self.status.enter("buffer_fill", self.log)
        stop = threading.Event()
        mon = threading.Thread(target=self._fill_monitor, args=(stop,), daemon=True)
        mon.start()
        try:
            self.pending = self._fetch()
        finally:
            stop.set()
            mon.join(timeout=1.0)
        self._log({"kind": "startup", "seconds": {k: round(v, 3) for k, v in self.timing.items()},
                   "amp": t["amp"], "compile": t["compile"], "channels_last": t["channels_last"],
                   "optimizer": t["optimizer"]})
        self.status.enter("train", self.log)
        self.t_train0, self.step_train0 = time.monotonic(), self.step
        self.last_print, self.printed_step = self.t_train0, None
        while self.step < t["max_steps"] and not self._time_up():
            cur, self.pending = self.pending, None
            if self.step + 1 < t["max_steps"]:
                self.pending = self._fetch()
            loss, loss_p, loss_v, loss_r, lr = self._train_step(*cur)
            now = time.monotonic()
            if self.step % t["log_every_steps"] == 0:
                self._log({
                    "kind": "train", "step": self.step, "epoch": self.states[0]["epoch"],
                    "loss": float(loss.item()), "loss_p": float(loss_p.item()), "loss_v": float(loss_v.item()),
                    "loss_r": float(loss_r.item()), "lr": lr, "bad_records": self.bad_records,
                    "progress": self.step / t["max_steps"],
                    "positions_per_sec": self.pos_since_log / max(now - self.last_log, 1e-9),
                    "elapsed_sec": now - self.t0,
                    "sources": self._per_source_row(),
                })
                self.last_log = now
                self.pos_since_log = 0
            if now - self.last_print >= float(t["progress_every_sec"]):
                self._report(now)
            if self.val is not None and t["val_every_steps"] > 0 and self.step % t["val_every_steps"] == 0:
                self._validate()
            if time.monotonic() - self.last_save >= save_interval:
                self.save("periodic")
        if self.printed_step != self.step:
            self._report(time.monotonic())
        if self.val is not None and (t["val_every_steps"] == 0 or self.step % t["val_every_steps"] != 0):
            self._validate()
        if self.saved_step != self.step:
            self.save("final")
        self._log({"kind": "end", "step": self.step, "epoch": self.states[0]["epoch"],
                   "elapsed_sec": time.monotonic() - self.t0,
                   "stopped_by": "max_steps" if self.step >= t["max_steps"] else "max_minutes"})
        self.log.close()
        self.loaders = [None] * len(self.sources)
        self.status.progress(self.step, t["max_steps"], 0.0)
        self.status.enter("done")


def parse_args(argv):
    p = argparse.ArgumentParser(prog="python -m shikibu.train_dl")
    p.add_argument("--config", required=True, help="設定ファイル（TOML）")
    p.add_argument("--resume", action="store_true", help="checkpoint.pt から続ける")
    p.add_argument("--force-unlock", action="store_true", help="実行中かもしれないロック（train.lock）も引き継ぐ")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    status, lock = None, None
    signal.signal(signal.SIGTERM, _on_term)
    try:
        cfg = load_config(args.config)
        check_placeholders(cfg, DL_PLACEHOLDER_KEYS)
        for src in cfg["data"]["sources"]:  # 組の glob に残った目印も止める（M11.1）
            check_placeholders({"data": {"glob": src.get("glob", "")}}, [("data", "glob")])
        validate(cfg)
        out_dir = cfg["output"]["out_dir"]
        os.makedirs(out_dir, exist_ok=True)
        lock = TrainLock.acquire(out_dir, force=args.force_unlock)
        status = Status(out_dir)
        DLTrainer(cfg, args.resume, status).run()
    except (ValueError, RuntimeError, FileExistsError) as e:
        if status is not None and status.history:
            status.enter("error")
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    finally:
        if lock is not None:
            lock.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
