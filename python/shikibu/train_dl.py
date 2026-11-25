"""DL の学習ループ（M11）：詰めた局面（PSV）から、方策と価値の 2 つの頭を持つネットワークを学習する。

使い方：python -m shikibu.train_dl --config <TOML> [--resume] [--force-unlock]

- 局面は shikibu_feat.BatchLoader(output="dl") から読む（詰めた局面。CPU は軽い）。55 チャンネルへの展開は
  dl_features.expand で GPU（入力と同じ device）で行う。展開は Dainagon の特徴量とビット単位で同じ。
- 損失（Dainagon の train_rocm.py と同じ。教師の勝率は sigmoid(score / eval_scale)）：
    loss = CE(p, move; label_smoothing) + value_loss_scale × (λ × CE(value_logits, hl_gauss(勝率)) + (1 − λ) × BCE(result_logit, result))
- 出力（out_dir）：checkpoint.pt・log.jsonl・status.json・train.lock。log の各行は "type" と "kind" の両方に同じ値を持つ
  （"type" は shk_monitor と今の学習器（train.py）と同じ。"kind" は受け入れテストの名前）。
- 部品は train.py から使い回す（設定の読み込みは DL の既定値で行う）。train.py の動きは変えない。
"""
import argparse
import glob
import os
import random
import signal
import sys
import tempfile
import time
import tomllib

import numpy as np
import torch
import torch.nn.functional as F

import shikibu_feat
from shikibu import dl_features, dl_networks
from shikibu.lock import TrainLock
from shikibu.train import (Status, JsonlLog, epoch_perm, eval_scale_setting, lr_factor, sample_scale_positions,
                           train_paths, pick_device, seed_everything, rng_state, restore_rng, fmt_sec,
                           check_placeholders, PLACEHOLDER_KEYS, _on_term)

# 目印 <…> を調べるキー（NNUE の学習器の一覧に、DL の検証の PSV 以外の hcpe を足したもの）
DL_PLACEHOLDER_KEYS = PLACEHOLDER_KEYS + [("data", "val_hcpe_glob")]

PSV_BYTES = 40
KEYS = ("board", "hands", "check", "move", "score", "result")
VAL_BATCH = 4096  # 検証の展開をまとめる大きさ（メモリを抑える）
OPTIMIZERS = ("adamw",)  # 今は AdamW だけ。後でオプティマイザを足すときは make_optimizer と、この表を直す

DEFAULTS = {
    "data": {
        "train_paths": [],
        "train_glob": "",
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


def validate(cfg):
    d, t = cfg["data"], cfg["train"]
    if not d["train_paths"] and not d["train_glob"]:
        raise ValueError("data.train_paths か data.train_glob を指定する")
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


# ---------- 部品 ----------

def make_optimizer(name, model, lr, weight_decay, fused=False):
    """オプティマイザを作る。名前から選ぶ口はここだけ（後で Muon などを足すときは、ここを直す）。"""
    if name == "adamw":
        kw = {"fused": True} if fused else {}
        return torch.optim.AdamW(dl_networks.param_groups(model, weight_decay), lr=lr, **kw)
    raise ValueError(f"未対応のオプティマイザ: {name}")


def loss_terms(p, value_logits, result_logit, move, score, result, eval_scale, lam, label_smoothing):
    """損失の各項（float32）。返り値は (loss_p, loss_v, loss_r)。全体は呼び出し側で組み立てる。"""
    p = p.float()
    loss_p = F.cross_entropy(p, move, label_smoothing=label_smoothing)
    target_win = torch.sigmoid(score / eval_scale)
    hl = dl_features.hl_gauss(target_win).to(value_logits.device)
    loss_v = -(hl * F.log_softmax(value_logits.float(), dim=1)).sum(dim=1).mean()
    loss_r = F.binary_cross_entropy_with_logits(result_logit.float().view(-1), result)
    return loss_p, loss_v, loss_r


def total_loss(loss_p, loss_v, loss_r, lam, value_loss_scale):
    return loss_p + value_loss_scale * (lam * loss_v + (1.0 - lam) * loss_r)


class DLPrefetcher:
    """次のバッチの転送を、今のステップの計算と重ねる（M8 と同じ考え方）。CUDA では固定メモリ 2 面と転送のストリーム。"""

    def __init__(self, device):
        self.device = device
        self.cuda = device.type == "cuda"
        self.stream = torch.cuda.Stream(device) if self.cuda else None
        self.slots = [{}, {}]
        self.events = [None, None]
        self.n = 0

    def send(self, batch):
        if not self.cuda:
            return {k: torch.from_numpy(batch[k]).to(self.device) for k in KEYS}, None
        face = self.n % 2
        self.n += 1
        if self.events[face] is not None:
            self.events[face].synchronize()
        out = {}
        with torch.cuda.stream(self.stream):
            for k in KEYS:
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
        self.paths = train_paths(d)
        self.lam = float(t["value_mix_lambda"])
        self.vls = float(t["value_loss_scale"])
        self.saved_step = -1
        self.timing = {}

        ck = None
        if os.path.exists(self.ckpt_path):
            if not resume:
                raise FileExistsError(f"{self.ckpt_path} がある。続けるなら --resume を付ける（上書きはしない）")
            ck = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
            if ck["network"] != self.network or ck.get("optimizer_name") != t["optimizer"]:
                raise ValueError("checkpoint の network・optimizer が設定と違う")
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
        self.prefetch = DLPrefetcher(self.device)

        if ck is None:
            t0 = time.monotonic()
            self.eval_scale, self.eval_scale_source = self._decide_eval_scale()
            self.timing["data_check"] = time.monotonic() - t0
            self.step, self.epoch, loader_state = 0, 0, None
            self.bad_base = 0
            self.perm = epoch_perm(len(self.paths), t["seed"], 0)
        else:
            self.model.load_state_dict(ck["model"])
            self.optimizer.load_state_dict(ck["optimizer"])
            if self.scaler is not None and ck.get("scaler") is not None:
                self.scaler.load_state_dict(ck["scaler"])
            self.eval_scale, self.eval_scale_source = ck["eval_scale"], ck["eval_scale_source"]
            self.step, self.epoch, loader_state = ck["step"], ck["epoch"], ck["loader_state"]
            self.bad_base = int(ck.get("bad_records", 0))
            self.perm = ck.get("file_perm", list(range(len(self.paths))))
            if sorted(self.perm) != list(range(len(self.paths))):
                raise ValueError("checkpoint の file_perm が学習データのファイル数と合わない")
            restore_rng(ck["rng"])

        self.val = self._load_validation()
        self.log = JsonlLog(os.path.join(self.out_dir, "log.jsonl"))
        self.t0 = time.monotonic()
        self.last_save = self.t0
        self.last_log = self.t0
        self.pos_since_log = 0
        self.pos_since_print = 0
        self._write_start()
        self.bad_closed = 0
        self.bad_records = self.bad_base
        self.bad_announced = False
        self._open_loader(loader_state, self.epoch, self.perm)
        self.loader_state = dict(self.loader.state())

    # 開始前
    def _decide_eval_scale(self):
        t = self.t
        setting = eval_scale_setting(t["eval_scale"])
        self.status.enter("data_check")
        sample_path = os.path.join(tempfile.gettempdir(), "shikibu_dl_scale_sample.psv")
        files = None
        try:
            files, count = sample_scale_positions(self.paths, t["scale_sample"], t["seed"], sample_path)
            if count == 0:
                raise RuntimeError("学習データに局面が1つも無い")
            stats = shikibu_feat.psv_stats(files)
            if stats["result_missing"]:
                raise RuntimeError("勝敗が欠損している（psv_stats）。DL の学習は勝敗を使うので、データを確かめる")
            if setting is not None:
                return setting, "config"
            self.status.enter("eval_scale")
            est = float(shikibu_feat.estimate_eval_scale(files, sample=min(count, t["scale_sample"]),
                                                         seed=t["seed"])["eval_scale"])
            return est, "auto"
        finally:
            if files is not None and files != list(self.paths) and os.path.exists(sample_path):
                os.remove(sample_path)

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
            vals["psv"] = (to_tensors(batch, self.device), None)  # None なら eval_scale（学習と同じ尺度）
        if d["val_hcpe_glob"]:
            hv = load_hcpe_validation(d["val_hcpe_glob"], t["val_positions"])
            if hv is not None:
                vals["hcpe"] = (to_tensors(hv, self.device), 600.0)
        return vals or None

    def _open_loader(self, state, epoch, perm):
        d, t = self.d, self.t
        if getattr(self, "loader", None) is not None:
            self.bad_closed += self.loader.skipped()
        self.loader = None
        self.rd_epoch, self.rd_perm = epoch, perm
        paths = [self.paths[i] for i in perm]
        self.loader = shikibu_feat.BatchLoader(
            paths, t["batch_size"], 0, shuffle_buffer=d["shuffle_buffer"], seed=t["seed"] + epoch,
            num_threads=d["num_threads"], cache_dir=d["cache_dir"] or None, max_cached=d["max_cached"],
            resume_state=state, output="dl",
        )

    def _fetch(self):
        batch = next(self.loader, None)
        if batch is None:
            self.rd_epoch += 1
            self.rd_perm = epoch_perm(len(self.paths), self.t["seed"], self.rd_epoch)
            self._open_loader(None, self.rd_epoch, self.rd_perm)
            batch = next(self.loader, None)
            if batch is None:
                raise RuntimeError("局面が1つも読めなかった")
        bad = self.bad_base + self.bad_closed + self.loader.skipped()
        if bad > 0 and not self.bad_announced:
            print(f"壊れたレコードを飛ばしました（ここまでに {bad} 件）。学習は続けます", flush=True)
            self.bad_announced = True
        meta = {"epoch": self.rd_epoch, "perm": self.rd_perm, "loader_state": dict(self.loader.state()), "bad": bad}
        x, ev = self.prefetch.send(batch)
        return x, ev, meta

    def _write_start(self):
        self._log({"kind": "start", "step": self.step, "epoch": self.epoch, "resumed": self.step > 0,
                   "eval_scale": self.eval_scale, "eval_scale_source": self.eval_scale_source,
                   "network": self.network, "device": str(self.device), "n_files": len(self.paths),
                   "config": self.cfg})

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
            "epoch": self.epoch,
            "loader_state": self.loader_state,
            "eval_scale": self.eval_scale,
            "eval_scale_source": self.eval_scale_source,
            "network": self.network,
            "config": self.cfg,
            "rng": rng_state(),
            "file_perm": self.perm,
            "bad_records": self.bad_records,
        }
        tmp = self.ckpt_path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.ckpt_path)
        self.last_save = time.monotonic()
        self.saved_step = self.step
        self._log({"kind": "save", "step": self.step, "epoch": self.epoch, "reason": reason})

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
        loss_p, loss_v, loss_r = loss_terms(p, vl, rl, x["move"], x["score"], x["result"],
                                            self.eval_scale, self.lam, float(t["label_smoothing"]))
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
        self.epoch, self.perm, self.loader_state = meta["epoch"], meta["perm"], meta["loader_state"]
        self.bad_records = meta["bad"]
        self.step += 1
        n = int(x["move"].shape[0])
        self.pos_since_log += n
        self.pos_since_print += n
        return loss, loss_p, loss_v, loss_r, lr

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
        self.pending = self._fetch()
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
                    "kind": "train", "step": self.step, "epoch": self.epoch,
                    "loss": float(loss.item()), "loss_p": float(loss_p.item()), "loss_v": float(loss_v.item()),
                    "loss_r": float(loss_r.item()), "lr": lr, "bad_records": self.bad_records,
                    "progress": self.step / t["max_steps"],
                    "positions_per_sec": self.pos_since_log / max(now - self.last_log, 1e-9),
                    "elapsed_sec": now - self.t0,
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
            self._validate()  # 終わりの検証（周期の検証と重なるときは、もう一度しない）
        if self.saved_step != self.step:
            self.save("final")
        self._log({"kind": "end", "step": self.step, "epoch": self.epoch,
                   "elapsed_sec": time.monotonic() - self.t0,
                   "stopped_by": "max_steps" if self.step >= t["max_steps"] else "max_minutes"})
        self.log.close()
        self.loader = None
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
