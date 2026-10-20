"""学習ループ（設計書 §S6・§F3.3）。

使い方：python -m shikibu.train --config <TOML> [--resume]

- 重みの初期化より前に乱数の種を決める。乱数の状態・ローダーの位置・オプティマイザは checkpoint.pt に保存する。
- 局面は shikibu_feat.BatchLoader から読む。1周したら、次の周は seed + 周回番号 の新しいローダーで続ける。
- ログは <out_dir>/log.jsonl（1行1 JSON）。
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time
import tomllib

import numpy as np
import torch
import torch.nn.functional as F

import shikibu_feat
from shikibu import networks

# CUDA で決定的な計算（use_deterministic_algorithms）を使うときに必要。torch の CUDA 初期化より前に設定する。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

LIMIT = 127.0 / 64.0  # §F3.3 の Linear の重みの範囲
DECAY_PARAMS = ("l1.weight", "l2.weight", "out.weight")  # weight decay の対象（§S6）
BATCH_KEYS = ("stm_indices", "stm_offsets", "nstm_indices", "nstm_offsets", "score", "result")

DEFAULTS = {
    "data": {
        "train_paths": [],
        "train_glob": "",
        "val_path": "",
        "cache_dir": "",
        "max_cached": 3,
        "shuffle_buffer": 0,
        "num_threads": 1,
    },
    "model": {"network": "RayPair_v0Network", "ft_width": 128},
    "train": {
        "batch_size": 16384,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "warmup_steps": 0,
        "total_steps": 0,
        "max_steps": 0,
        "min_lr_ratio": 0.0,
        "lam": 0.75,
        "eval_scale": "auto",
        "scale_sample": 10_000_000,
        "allow_missing_results": False,
        "seed": 0,
        "device": "auto",
        "torch_threads": 1,
        "deterministic": False,
        "save_every_steps": 0,
        "save_every_minutes": 30.0,
        "val_every_steps": 0,
        "val_positions": 100_000,
        "log_every_steps": 100,
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


def validate(cfg):
    d, t = cfg["data"], cfg["train"]
    if not d["train_paths"] and not d["train_glob"]:
        raise ValueError("data.train_paths か data.train_glob を指定する")
    if not cfg["output"]["out_dir"]:
        raise ValueError("output.out_dir を指定する")
    if t["total_steps"] <= 0 or t["max_steps"] <= 0:
        raise ValueError("train.total_steps と train.max_steps は1以上")
    if t["batch_size"] <= 0 or d["num_threads"] <= 0:
        raise ValueError("batch_size と num_threads は1以上")
    if d["cache_dir"] and d["max_cached"] <= 0:
        raise ValueError("max_cached は1以上")
    if not 0.0 <= float(t["lam"]) <= 1.0:
        raise ValueError("lam は 0〜1")
    if d["val_path"] and t["val_positions"] <= 0:
        raise ValueError("val_positions は1以上")
    eval_scale_setting(t["eval_scale"])


def eval_scale_setting(v):
    """設定の eval_scale を解釈する。"auto" なら None、数値なら float。"""
    if v == "auto":
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
        return float(v)
    raise ValueError(f'eval_scale は "auto" か正の数: {v!r}')


def train_paths(d):
    if d["train_glob"]:
        paths = sorted(glob.glob(d["train_glob"]))
    else:
        paths = list(d["train_paths"])
    if not paths:
        raise ValueError("学習データが1つも無い（data.train_paths / data.train_glob）")
    return paths


# ---------- 部品 ----------

def pick_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def configure_runtime(t):
    torch.set_num_threads(int(t["torch_threads"]))
    torch.use_deterministic_algorithms(bool(t["deterministic"]), warn_only=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def rng_state():
    st = {"torch": torch.get_rng_state(), "python": random.getstate(), "numpy": np.random.get_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def restore_rng(st):
    torch.set_rng_state(st["torch"])
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


@torch.no_grad()
def clip_weights_(model):
    """§F3.3：各ステップの後に範囲へ収める。bias は対象外。"""
    model.ft.weight.clamp_(-1.0, 1.0)
    model.ft_bias.clamp_(-1.0, 1.0)
    for lin in (model.l1, model.l2, model.out):
        lin.weight.clamp_(-LIMIT, LIMIT)


def compute_loss(z, score, result, eval_scale, lam):
    """t = λ·sigmoid(score / eval_scale) + (1 − λ)·result、損失 = mean(BCEWithLogits(z, t))（§S6）。"""
    t = lam * torch.sigmoid(score / eval_scale) + (1.0 - lam) * result
    return F.binary_cross_entropy_with_logits(z, t)


def build_optimizer(model, lr, weight_decay):
    named = dict(model.named_parameters())
    decay = [named[k] for k in DECAY_PARAMS]
    no_decay = [p for k, p in named.items() if k not in DECAY_PARAMS]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def lr_factor(step, warmup_steps, total_steps, min_lr_ratio):
    """ウォームアップ＋コサイン減衰の倍率。step は 0 始まりの更新回数。"""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    p = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * p))


def to_device(batch, device):
    """ローダーの numpy 配列を torch テンソルにして device へ送る。CUDA では固定メモリ経由で非同期転送する。"""
    out = {}
    for k in BATCH_KEYS:
        x = torch.from_numpy(batch[k])
        if device.type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
        out[k] = x
    out["score"] = out["score"].float()
    out["result"] = out["result"].float()
    return out


@torch.no_grad()
def evaluate(model, val, eval_scale, lam):
    """検証：(val_loss, val_mse)。val_mse = mean((sigmoid(z) − sigmoid(score / eval_scale))²)。"""
    model.eval()
    z = model(val["stm_indices"], val["stm_offsets"], val["nstm_indices"], val["nstm_offsets"])
    loss = compute_loss(z, val["score"], val["result"], eval_scale, lam)
    target = torch.sigmoid(val["score"] / eval_scale)
    mse = ((torch.sigmoid(z) - target) ** 2).mean()
    model.train()
    return float(loss.item()), float(mse.item())


class JsonlLog:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def write(self, rec):
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


# ---------- 学習 ----------

class Trainer:
    def __init__(self, cfg, resume):
        self.cfg = cfg
        d, m, t, o = cfg["data"], cfg["model"], cfg["train"], cfg["output"]
        self.d, self.t = d, t
        self.out_dir = o["out_dir"]
        self.ckpt_path = os.path.join(self.out_dir, "checkpoint.pt")
        self.lam = float(t["lam"])
        self.paths = train_paths(d)

        ck = None
        if os.path.exists(self.ckpt_path):
            if not resume:
                raise FileExistsError(f"{self.ckpt_path} がある。続けるなら --resume を付ける（上書きはしない）")
            ck = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
            if ck["network"] != m["network"] or ck["lam"] != self.lam:
                raise ValueError("checkpoint の network / lam が設定と違う")
        elif resume:
            print("checkpoint.pt が無いので最初から学習します", file=sys.stderr)

        self.device = pick_device(t["device"])
        configure_runtime(t)
        seed_everything(t["seed"])
        self.model = networks.build(m["network"], ft_width=m["ft_width"])
        self.version = self.model.FEATURE_VERSION
        if shikibu_feat.num_features(self.version) != self.model.NUM_FEATURES:
            raise RuntimeError("ネットワークの特徴数が shikibu_feat と一致しない")
        self.model.to(self.device)
        self.optimizer = build_optimizer(self.model, t["lr"], t["weight_decay"])

        if ck is None:
            self.eval_scale, self.eval_scale_source = self._decide_eval_scale()
            self.step, self.epoch, loader_state = 0, 0, None
        else:
            self.model.load_state_dict(ck["model"])
            self.optimizer.load_state_dict(ck["optimizer"])
            self.eval_scale, self.eval_scale_source = ck["eval_scale"], ck["eval_scale_source"]
            self.step, self.epoch, loader_state = ck["step"], ck["epoch"], ck["loader_state"]
            restore_rng(ck["rng"])

        self.val = self._load_validation()
        self.loader = None
        self.cache_seen = 0
        self._open_loader(loader_state)
        self.log = JsonlLog(os.path.join(self.out_dir, "log.jsonl"))
        self.t0 = time.monotonic()
        self.last_save = self.t0
        self.last_log = self.t0
        self.pos_since_log = 0

    def _decide_eval_scale(self):
        """勝敗の欠損を確かめ、eval_scale を決める。判定は先頭の1ファイルだけで行う（§S4）。"""
        t, lam = self.t, self.lam
        setting = eval_scale_setting(t["eval_scale"])
        stats = shikibu_feat.psv_stats([self.paths[0]])
        if stats["result_missing"]:
            if not t["allow_missing_results"]:
                raise RuntimeError("勝敗が欠損している（psv_stats）。データを確認するか allow_missing_results = true にする")
            if lam != 1.0:
                raise RuntimeError("勝敗が欠損のときは lam = 1.0 が必要")
            if setting is None:
                raise RuntimeError("勝敗が欠損のときは eval_scale を数値で指定する")
        if setting is not None:
            return setting, "config"
        est = shikibu_feat.estimate_eval_scale([self.paths[0]], sample=t["scale_sample"], seed=t["seed"])
        return float(est["eval_scale"]), "auto"

    def _load_validation(self):
        if not self.d["val_path"]:
            return None
        loader = shikibu_feat.BatchLoader([self.d["val_path"]], self.t["val_positions"], self.version,
                                          0, 0, self.d["num_threads"])
        batch = next(loader, None)
        del loader
        if batch is None:
            raise RuntimeError("検証データが空")
        return to_device(batch, self.device)

    def _open_loader(self, state):
        d, t = self.d, self.t
        self.loader = None  # 前のローダーを止めてからでないと、キャッシュ先を作り直せない
        self.cache_seen = 0
        self.loader = shikibu_feat.BatchLoader(
            self.paths, t["batch_size"], self.version,
            shuffle_buffer=d["shuffle_buffer"], seed=t["seed"] + self.epoch,
            num_threads=d["num_threads"], cache_dir=d["cache_dir"] or None,
            max_cached=d["max_cached"], resume_state=state,
        )

    def _next_batch(self):
        batch = next(self.loader, None)
        if batch is None:  # 1周終わった。次の周は seed + 周回番号
            self.epoch += 1
            self._open_loader(None)
            batch = next(self.loader, None)
            if batch is None:
                raise RuntimeError("局面が1つも読めなかった")
        return batch

    def _emit_cache(self):
        if not self.d["cache_dir"]:
            return
        stats = self.loader.cache_stats()
        new = list(stats[self.cache_seen:])
        self.cache_seen = len(stats)
        self.log.write({"type": "cache", "step": self.step, "epoch": self.epoch,
                        "copied": new, "copied_total": len(stats)})

    def save(self, reason):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "loader_state": dict(self.loader.state()),
            "eval_scale": self.eval_scale,
            "eval_scale_source": self.eval_scale_source,
            "lam": self.lam,
            "network": self.cfg["model"]["network"],
            "config": self.cfg,
            "rng": rng_state(),
        }
        tmp = self.ckpt_path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.ckpt_path)  # 途中で切れても古い checkpoint.pt が残るように改名で置き換える
        self.last_save = time.monotonic()
        self.log.write({"type": "save", "step": self.step, "epoch": self.epoch, "reason": reason})
        self._emit_cache()

    def _train_step(self, batch):
        x = to_device(batch, self.device)
        z = self.model(x["stm_indices"], x["stm_offsets"], x["nstm_indices"], x["nstm_offsets"])
        loss = compute_loss(z, x["score"], x["result"], self.eval_scale, self.lam)
        lr = self.t["lr"] * lr_factor(self.step, self.t["warmup_steps"], self.t["total_steps"],
                                      self.t["min_lr_ratio"])
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        clip_weights_(self.model)
        self.step += 1
        self.pos_since_log += int(x["score"].shape[0])
        return loss, lr

    def run(self):
        t = self.t
        self.log.write({
            "type": "start", "step": self.step, "epoch": self.epoch, "resumed": self.step > 0,
            "eval_scale": self.eval_scale, "eval_scale_source": self.eval_scale_source,
            "lam": self.lam, "network": self.cfg["model"]["network"], "device": str(self.device),
            "n_files": len(self.paths), "config": self.cfg,
        })
        save_interval = t["save_every_minutes"] * 60.0
        while self.step < t["max_steps"]:
            batch = self._next_batch()
            loss, lr = self._train_step(batch)

            if self.step % t["log_every_steps"] == 0:
                now = time.monotonic()
                self.log.write({
                    "type": "train", "step": self.step, "epoch": self.epoch,
                    "loss": float(loss.item()), "lr": lr,
                    "positions_per_sec": self.pos_since_log / max(now - self.last_log, 1e-9),
                    "elapsed_sec": now - self.t0,
                })
                self.last_log = now
                self.pos_since_log = 0

            if self.val is not None and t["val_every_steps"] > 0 and self.step % t["val_every_steps"] == 0:
                val_loss, val_mse = evaluate(self.model, self.val, self.eval_scale, self.lam)
                self.log.write({"type": "val", "step": self.step, "val_loss": val_loss, "val_mse": val_mse})

            periodic = t["save_every_steps"] > 0 and self.step % t["save_every_steps"] == 0
            if periodic or time.monotonic() - self.last_save >= save_interval:
                self.save("periodic")

        self.save("final")
        self.log.write({"type": "end", "step": self.step, "epoch": self.epoch,
                        "elapsed_sec": time.monotonic() - self.t0})
        self.log.close()
        self.loader = None


def parse_args(argv):
    p = argparse.ArgumentParser(prog="python -m shikibu.train")
    p.add_argument("--config", required=True, help="設定ファイル（TOML）")
    p.add_argument("--resume", action="store_true", help="checkpoint.pt から続ける")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    cfg = load_config(args.config)
    validate(cfg)
    os.makedirs(cfg["output"]["out_dir"], exist_ok=True)
    Trainer(cfg, args.resume).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
