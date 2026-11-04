"""学習ループ（設計書 §S6・§F3.3）。

使い方：python -m shikibu.train --config <TOML> [--resume]

- 重みの初期化より前に乱数の種を決める。乱数の状態・ローダーの位置・オプティマイザは checkpoint.pt に保存する。
- 局面は shikibu_feat.BatchLoader から読む。各周回の始めに、学習ファイルの順番を seed + 周回番号 で並べ替えてから渡す。
  順番は checkpoint.pt の file_perm に保存する（無ければ M3 と同じ並び順）。
- ログは <out_dir>/log.jsonl（1行1 JSON）。
"""
import argparse
import glob
import hashlib
import json
import math
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
from shikibu import networks
from shikibu.lock import TrainLock
from shikibu.optim import CombinedOptimizer, LazyAdamRows, is_lazy_state

# CUDA で決定的な計算（use_deterministic_algorithms）を使うときに必要。torch の CUDA 初期化より前に設定する。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

LIMIT = 127.0 / 64.0  # §F3.3 の Linear の重みの範囲
PSV_BYTES = 40
SCALE_BLOCK = 10_000  # eval_scale の推定に使う局面を取り出す連続ブロックの長さ（局面数）
BATCH_KEYS = ("stm_indices", "stm_offsets", "nstm_indices", "nstm_offsets", "score", "result", "kings")

DEFAULTS = {
    "data": {
        "train_paths": [],
        "train_glob": "",
        "val_path": "",
        "cache_dir": "",
        "max_cached": 3,
        "shuffle_buffer": 0,
        "num_threads": 1,
        "check_mode": "sample",   # 勝敗の確認の抜き取り："sample"（従来）・"probe"（各ファイルの先頭・中央・末尾）・"full"（全件）
        "probe_records": 3000,    # probe のとき、1 ファイルから読む局面数（3 ブロックに分ける）
        "probe_files": 0,         # probe のとき、読むファイル数の上限（0 なら全ファイル。読む回数は 3 × この数）
        "check_cache": False,     # 確認の結果を、データのフォルダに保存して、次回から読むだけにする（M9）
        "shuffle_warm": 0,        # 0 は従来どおり（バッファを満たしてから学習）。>0 なら、この件数から学習を始める
    },
    "model": {"network": "RayPair_v0Network", "ft_width": 0},  # ft_width 0 = 名前に合わせる（従来の名前なら 128）
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
        "progress_every_sec": 30.0,
        "ft_forward": "dense",    # 第1層の順伝播："dense"（従来）・"sparse"（有効な行だけ。因数分解つきでも密な表を作らない）
        "ft_grad": "dense",       # 第1層の勾配："dense"（従来）・"sparse"（疎。ft_optimizer = "lazy_adam" が必要）
        "ft_optimizer": "adamw",  # "adamw"（従来）・"lazy_adam"（第1層は使った行だけ更新。数学的に AdamW と違う）
        "fast": False,            # CUDA のとき、2 層目以降を bf16 の自動混合精度で計算する（GPU が無ければ無視）
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


PLACEHOLDER_KEYS = [("data", "train_paths"), ("data", "train_glob"), ("data", "val_path"),
                    ("data", "cache_dir"), ("output", "out_dir")]


def check_placeholders(cfg):
    """パスに "<…>" の目印が残っていたら、ひな形のまま使おうとしているので止める（何も書かない）。"""
    found = []
    for sec, key in PLACEHOLDER_KEYS:
        v = cfg[sec][key]
        for s in (v if isinstance(v, list) else [v]):
            if isinstance(s, str) and "<" in s and ">" in s:
                found.append(f"{sec}.{key} = {s}")
    if found:
        raise ValueError("設定のひな形のまま（<…> の目印が残っている）です。自分の環境に合わせて書き換えてから使ってください: "
                         + "; ".join(found))


def _check_min(section, key, minimum):
    v = section[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v < minimum:
        raise ValueError(f"train.{key} は {minimum} 以上の数: {v!r}")


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
    network_name(cfg["model"])
    _check_min(t, "log_every_steps", 1)
    _check_min(t, "save_every_steps", 0)  # 0 は無効
    _check_min(t, "val_every_steps", 0)   # 0 は無効
    _check_min(t, "progress_every_sec", 0.0)
    if d["check_mode"] not in ("sample", "probe", "full"):
        raise ValueError('data.check_mode は "sample"・"probe"・"full" のどれか')
    _check_min(d, "probe_records", 3)
    if not isinstance(d["probe_files"], int) or d["probe_files"] < 0:
        raise ValueError("data.probe_files は 0 以上の整数（0 なら全ファイル）")
    if not isinstance(d["shuffle_warm"], int) or not 0 <= d["shuffle_warm"] <= d["shuffle_buffer"]:
        raise ValueError("data.shuffle_warm は 0 以上で、shuffle_buffer 以下の整数")
    if t["ft_forward"] not in ("dense", "sparse") or t["ft_grad"] not in ("dense", "sparse"):
        raise ValueError('train.ft_forward・ft_grad は "dense" か "sparse"')
    if t["ft_optimizer"] not in ("adamw", "lazy_adam"):
        raise ValueError('train.ft_optimizer は "adamw" か "lazy_adam"')
    if t["ft_grad"] == "sparse" and t["ft_optimizer"] != "lazy_adam":
        raise ValueError('train.ft_grad = "sparse" には ft_optimizer = "lazy_adam" が必要（AdamW は疎な勾配を扱えない）')
    if t["ft_grad"] == "sparse" and t["ft_forward"] != "sparse" and networks.parse_arch(networks.arch_of(
            network_name(cfg["model"]), cfg["model"]["ft_width"] or 128))["factorize"]:
        raise ValueError('因数分解つきの網で ft_grad = "sparse" には ft_forward = "sparse" が必要')
    if not isinstance(t["fast"], bool):
        raise ValueError("train.fast は true か false")


def network_name(m):
    """model.network を確かめ、checkpoint に書く名前を返す（従来の名前はそのまま、構造の名前は正規の形）。"""
    name = m["network"]
    if name in networks.NETWORKS:
        return name
    p = networks.parse_arch(name)
    if m.get("ft_width", 0) not in (0, p["W"]):
        raise ValueError(f"model.ft_width（{m['ft_width']}）が構造の名前の W（{p['W']}）と違う: {name}")
    return p["canonical"]


def eval_scale_setting(v):
    """設定の eval_scale を解釈する。"auto" なら None、数値なら float。"""
    if v == "auto":
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
        return float(v)
    raise ValueError(f'eval_scale は "auto" か正の数: {v!r}')


def epoch_perm(n, seed, epoch):
    """周回ごとのファイルの並べ方（0〜n−1 の順列）。seed + 周回番号 で決まる。"""
    perm = list(range(n))
    random.Random(seed + epoch).shuffle(perm)
    return perm


def train_paths(d):
    if d["train_glob"]:
        paths = sorted(glob.glob(d["train_glob"]))
    else:
        paths = list(d["train_paths"])
    if d.get("val_path"):
        # 検証のファイルは学習に使わない（§S12.5）
        val = os.path.realpath(d["val_path"])
        paths = [x for x in paths if os.path.realpath(x) != val]
    if not paths:
        raise ValueError("学習データが1つも無い（data.train_paths / data.train_glob）")
    return paths


# ---------- 部品 ----------

def sample_scale_positions(paths, limit, seed, out_path):
    """勝敗の確認と eval_scale の推定に使う局面を取り出す（§S4）。

    全ファイルから、ファイルの大きさに比例した数の「ランダムな位置の連続ブロック」を取る。
    局面数が limit 以下なら全部使う（ファイルをそのまま渡す）。
    返り値：(使うファイルのリスト, 局面数)。抜き出したときは out_path に書く。
    """
    sizes = [os.path.getsize(p) // PSV_BYTES for p in paths]
    total = sum(sizes)
    if total <= limit:
        return list(paths), total
    rng = np.random.default_rng(seed)
    count = 0
    with open(out_path, "wb") as out:
        for p, n in zip(paths, sizes):
            quota = limit * n // total
            if quota == 0:
                continue
            nb = -(-quota // SCALE_BLOCK)  # ブロック数（切り上げ）
            lens = [quota // nb] * nb
            lens[-1] += quota - sum(lens)
            with open(p, "rb") as f:
                for ln in lens:
                    start = int(rng.integers(0, n - ln + 1))
                    f.seek(start * PSV_BYTES)
                    out.write(f.read(ln * PSV_BYTES))
                    count += ln
    return [out_path], count


def probe_positions(paths, per_file, out_path, max_files=0, seed=0):
    """抜き取り（data.check_mode = "probe"，M9）：各ファイルの先頭・中央・末尾から per_file/3 件ずつ読む。

    ファイルの大きさは全ファイルについて先に検査する（stat だけ。40 の倍数でなければ止める）。
    max_files > 0 なら、seed で決まる無作為な max_files 個のファイルだけを読む（読む回数を抑える）。
    勝敗が 1・0・−1 でない局面は飛ばして数える。返り値：(使うファイルのリスト, 局面数, 飛ばした数)。
    局面は out_path に書く（ローカル）。
    """
    n_of = {}
    for p in paths:
        nbytes = os.path.getsize(p)
        if nbytes % PSV_BYTES:
            raise RuntimeError(f"{p}: サイズが40バイトの倍数でない（壊れたファイル）")
        n_of[p] = nbytes // PSV_BYTES
    read_paths = list(paths)
    if max_files > 0 and max_files < len(read_paths):
        pick = sorted(random.Random(seed).sample(range(len(read_paths)), max_files))
        read_paths = [read_paths[i] for i in pick]
    per_block = max(1, per_file // 3)
    count = skipped = 0
    with open(out_path, "wb") as out:
        for p in read_paths:
            n = n_of[p]
            if n == 0:
                continue
            length = min(per_block, n)
            starts = sorted({0, (n - length) // 2, n - length})
            with open(p, "rb") as f:
                for st in starts:
                    f.seek(st * PSV_BYTES)
                    buf = np.frombuffer(f.read(length * PSV_BYTES), dtype=np.uint8).reshape(-1, PSV_BYTES)
                    res = buf[:, 38].astype(np.int8)
                    keep = buf[(res >= -1) & (res <= 1)]
                    out.write(keep.tobytes())
                    count += len(keep)
                    skipped += len(buf) - len(keep)
    return [out_path], count, skipped


CHECK_CACHE_NAME = "shikibu_check_cache.json"  # データのフォルダに置く（*.psv には当たらない名前）
CHECK_CACHE_KEEP = 8


def check_key(paths, d, t):
    """確認の結果の鍵：ファイルの名前・大きさ・更新時刻と、確認の設定。どれかが変われば作り直す（M9）。"""
    sig = {"v": 1,
           "files": [[os.path.realpath(p), os.path.getsize(p), os.stat(p).st_mtime_ns] for p in paths],
           "mode": d["check_mode"], "probe_records": d["probe_records"], "probe_files": d["probe_files"],
           "scale_sample": t["scale_sample"], "seed": t["seed"]}
    return hashlib.sha256(json.dumps(sig, sort_keys=True).encode("utf-8")).hexdigest()


def check_cache_path(paths):
    return os.path.join(os.path.dirname(os.path.realpath(paths[0])), CHECK_CACHE_NAME)


def read_check_cache(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}  # 無い・壊れている・読めないときは、無かったものとして作り直す


def write_check_cache(path, key, entry):
    """確認の結果を書く（一時ファイル→改名）。書けなくても学習は続ける。"""
    data = read_check_cache(path)
    data.pop(key, None)
    data[key] = entry
    while len(data) > CHECK_CACHE_KEEP:
        data.pop(next(iter(data)))
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)
        return True
    except OSError as e:
        print(f"確認の結果を保存できなかった（学習は続ける）: {e}", file=sys.stderr, flush=True)
        return False


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
def clip_weights_(model, ft=True):
    """§F3.3：各ステップの後に範囲へ収める。bias・psqt は対象外。ft=False なら第1層の重みは触らない（M9）。"""
    model.clip_(LIMIT, ft=ft)


def compute_loss(z, score, result, eval_scale, lam):
    """t = λ·sigmoid(score / eval_scale) + (1 − λ)·result、損失 = mean(BCEWithLogits(z, t))（§S6）。"""
    t = lam * torch.sigmoid(score / eval_scale) + (1.0 - lam) * result
    return F.binary_cross_entropy_with_logits(z, t)


def build_optimizer(model, lr, weight_decay, fused=False, ft_optimizer="adamw"):
    """AdamW。fused は CUDA で使えるときだけ True にする（M8）。False なら従来どおり。

    ft_optimizer = "lazy_adam" のときは、第1層の重み（ft）だけ遅延型の Adam にする（M9）。
    """
    decay = model.decay_parameters()
    ids = {id(p) for p in decay}
    lazy = ft_optimizer == "lazy_adam"
    ft_id = id(model.ft.weight) if lazy else None
    no_decay = [p for p in model.parameters() if id(p) not in ids and id(p) != ft_id]
    kw = {"fused": True} if fused else {}
    adamw = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr, **kw,
    )
    if lazy:
        return CombinedOptimizer(adamw, LazyAdamRows(model.ft.weight, lr))
    return adamw


def autocast_bf16(device, fast):
    """fast のとき、CUDA で bf16 の自動混合精度を使う文脈（M9）。CPU では何もしない。"""
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(fast) and device.type == "cuda")


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


class Prefetcher:
    """次のバッチの転送を、今のステップの計算と重ねる（M8）。

    CPU では従来どおり（to_device を同期で呼ぶ）。CUDA では、転送専用のストリームと
    固定メモリの作業領域（2 面を交互に使う）を使い、非同期に送る。
    送り終わりは event で示し、計算を始める前に ready() で待つ。
    """

    def __init__(self, device):
        self.device = device
        self.cuda = device.type == "cuda"
        self.stream = torch.cuda.Stream(device) if self.cuda else None
        self.slots = [{}, {}]   # 面 → {キー: 固定メモリのテンソル}
        self.events = [None, None]
        self.n = 0

    def send(self, batch):
        """ローダーのバッチを device へ送り始める。返り値は (テンソルの dict, event または None)。"""
        if not self.cuda:
            return to_device(batch, self.device), None
        face = self.n % 2
        self.n += 1
        if self.events[face] is not None:
            self.events[face].synchronize()  # この面を前回の転送が読み終わるまで待つ
        out = {}
        with torch.cuda.stream(self.stream):
            for k in BATCH_KEYS:
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
        """転送が終わってから計算に使う。CPU では何もしない（to_device で済んでいる）。"""
        if ev is None:
            return x
        cur = torch.cuda.current_stream(self.device)
        cur.wait_event(ev)
        for t in x.values():
            t.record_stream(cur)  # 転送のストリームで作ったテンソルを、計算のストリームで使い終わるまで解放しない
        x["score"] = x["score"].float()
        x["result"] = x["result"].float()
        return x


@torch.no_grad()
def evaluate(model, val, eval_scale, lam, device=None, fast=False):
    """検証：(val_loss, val_mse)。val_mse = mean((sigmoid(z) − sigmoid(score / eval_scale))²)。"""
    model.eval()
    with autocast_bf16(device or torch.device("cpu"), fast):
        z = model(val["stm_indices"], val["stm_offsets"], val["nstm_indices"], val["nstm_offsets"], kings=val["kings"], base=True)
    z = z.float()
    loss = compute_loss(z, val["score"], val["result"], eval_scale, lam)
    target = torch.sigmoid(val["score"] / eval_scale)
    mse = ((torch.sigmoid(z) - target) ** 2).mean()
    model.train()
    return float(loss.item()), float(mse.item())


def fmt_sec(s):
    """残り秒数を「1時間02分」「2分05秒」の形にする。不明なら「--」。"""
    if s is None:
        return "--"
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    return f"{h}時間{m:02d}分" if h else f"{m}分{sec:02d}秒"


class Status:
    """out_dir/status.json に、今の段階と進み具合を書く（一時ファイル→改名）。監視ツールが読む。"""

    def __init__(self, out_dir):
        self.path = os.path.join(out_dir, "status.json")
        self.phase = None
        self.history = []
        self.step, self.max_steps, self.eta_sec = 0, 0, None

    def enter(self, name, log=None):
        self.phase = name
        self.history.append(name)
        if log is not None:
            log.write({"type": "phase", "name": name})
        self.write()

    def progress(self, step, max_steps, eta_sec):
        self.step, self.max_steps, self.eta_sec = step, max_steps, eta_sec
        self.write()

    def write(self):
        data = {"phase": self.phase, "history": self.history, "step": self.step, "max_steps": self.max_steps,
                "progress": self.step / self.max_steps if self.max_steps else 0.0,
                "eta_sec": self.eta_sec, "updated": time.time()}
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, self.path)


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
    def __init__(self, cfg, resume, status=None):
        self.cfg = cfg
        self.status = status or Status(cfg["output"]["out_dir"])
        d, m, t, o = cfg["data"], cfg["model"], cfg["train"], cfg["output"]
        self.d, self.t = d, t
        self.out_dir = o["out_dir"]
        self.ckpt_path = os.path.join(self.out_dir, "checkpoint.pt")
        self.lam = float(t["lam"])
        self.paths = train_paths(d)
        self.saved_step = -1
        self.network = network_name(m)
        self.scale_used = None  # 勝敗の確認と推定に使った局面数（新規の学習でだけ決まる）
        self.timing = {}        # 開始前の段階の時間（秒）。startup の行に書く（M9）
        self.cache_state = "off"  # 確認の結果のキャッシュ："off"・"hit"・"miss"
        self.lazy = t["ft_optimizer"] == "lazy_adam"

        ck = None
        if os.path.exists(self.ckpt_path):
            if not resume:
                raise FileExistsError(f"{self.ckpt_path} がある。続けるなら --resume を付ける（上書きはしない）")
            ck = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
            if ck["network"] != self.network or ck["lam"] != self.lam:
                raise ValueError("checkpoint の network / lam が設定と違う")
        elif resume:
            print("checkpoint.pt が無いので最初から学習します", file=sys.stderr)

        self.device = pick_device(t["device"])
        configure_runtime(t)
        seed_everything(t["seed"])
        self.model = networks.build(self.network, ft_width=m["ft_width"] or 128)
        self.version = self.model.INPUT_VERSION  # 因数分解つきでも、ローダーは因数分解なしの版（M8）
        if shikibu_feat.num_features(self.model.FEATURE_VERSION) != self.model.NUM_FEATURES:
            raise RuntimeError("ネットワークの特徴数が shikibu_feat と一致しない")
        self.model.FT_FORWARD_SPARSE = t["ft_forward"] == "sparse"  # M9（既定 False = 従来の計算）
        self.model.FT_GRAD_SPARSE = t["ft_grad"] == "sparse"
        self.model.to(self.device)
        self.optimizer = build_optimizer(self.model, t["lr"], t["weight_decay"],
                                         fused=self.device.type == "cuda", ft_optimizer=t["ft_optimizer"])
        self.prefetch = Prefetcher(self.device)

        if ck is None:
            t0 = time.monotonic()
            self.eval_scale, self.eval_scale_source = self._decide_eval_scale()
            self.timing["data_check"] = time.monotonic() - t0
            self.step, self.epoch, loader_state = 0, 0, None
            self.perm = epoch_perm(len(self.paths), t["seed"], 0)
        else:
            self.model.load_state_dict(ck["model"])
            if is_lazy_state(ck["optimizer"]) != self.lazy:
                raise ValueError("checkpoint の optimizer（train.ft_optimizer）が設定と違う")
            self.optimizer.load_state_dict(ck["optimizer"])
            self.eval_scale, self.eval_scale_source = ck["eval_scale"], ck["eval_scale_source"]
            self.step, self.epoch, loader_state = ck["step"], ck["epoch"], ck["loader_state"]
            # M3 の checkpoint には file_perm が無い。その場合は M3 と同じ並び（設定の順）で続ける
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
        self.loader = None
        self.cache_seen = 0
        if d["cache_dir"]:
            self.status.enter("prefetch", self.log)
        # 壊れたレコードの数（M8.1）。学習済みの位置までの合計を checkpoint に書く。再開したら、その値から続ける。
        self.bad_base = int(ck["bad_records"]) if ck is not None and "bad_records" in ck else 0
        self.bad_closed = 0      # 閉じたローダーで飛ばした数（周回をまたいで足す）
        self.bad_records = self.bad_base
        self.bad_announced = False
        self._open_loader(loader_state, self.epoch, self.perm)
        # 学習済みの位置（checkpoint に書く）。先読みの位置（rd_*）とは分けて持つ（M8）
        self.loader_state = dict(self.loader.state())

    def _check_missing(self, missing, setting):
        """勝敗の欠損の扱い（§S4）。欠損のときに使えない組み合わせは止める。"""
        t, lam = self.t, self.lam
        if missing:
            if not t["allow_missing_results"]:
                raise RuntimeError("勝敗が欠損している（psv_stats）。データを確認するか allow_missing_results = true にする")
            if lam != 1.0:
                raise RuntimeError("勝敗が欠損のときは lam = 1.0 が必要")
            if setting is None:
                raise RuntimeError("勝敗が欠損のときは eval_scale を数値で指定する")

    def _check_files(self, sample_path):
        """確認に使う局面（check_mode）。返り値：(使うファイルのリスト, 局面数)。"""
        d, t = self.d, self.t
        mode = d["check_mode"]
        if mode == "full":
            total = 0
            for p in self.paths:
                nbytes = os.path.getsize(p)
                if nbytes % PSV_BYTES:
                    raise RuntimeError(f"{p}: サイズが40バイトの倍数でない（壊れたファイル）")
                total += nbytes // PSV_BYTES
            return list(self.paths), total  # 全件を読む（psv_stats）。時間は全データの大きさに比例する
        if mode == "probe":
            files, count, skipped = probe_positions(self.paths, d["probe_records"], sample_path,
                                                    max_files=d["probe_files"], seed=t["seed"])
            if skipped:
                print(f"抜き取りで、勝敗が 1・0・−1 でない局面を {skipped} 件飛ばした", flush=True)
            return files, count
        return sample_scale_positions(self.paths, t["scale_sample"], t["seed"], sample_path)

    def _decide_eval_scale(self):
        """勝敗の欠損を確かめ、eval_scale を決める（§S4）。確認は check_mode の抜き取りで行う。

        check_cache = true なら、結果をデータのフォルダに保存し、同じデータ・同じ設定なら読むだけにする（M9）。
        """
        d, t, lam = self.d, self.t, self.lam
        setting = eval_scale_setting(t["eval_scale"])
        self.status.enter("data_check")
        cache_path = key = None
        if d["check_cache"]:
            cache_path, key = check_cache_path(self.paths), check_key(self.paths, d, t)
            hit = read_check_cache(cache_path).get(key)
            if hit is not None and (setting is not None or hit.get("eval_scale") is not None):
                self.cache_state = "hit"
                self.scale_used = hit["count"]
                self._check_missing(bool(hit["result_missing"]), setting)
                if setting is not None:
                    return setting, "config"
                return float(hit["eval_scale"]), "auto"
            self.cache_state = "miss"
        sample_path = os.path.join(tempfile.gettempdir(), "shikibu_scale_sample.psv")  # ローカルに書く（Drive には書かない）
        files = None
        try:
            files, count = self._check_files(sample_path)
            self.scale_used = count
            if count == 0:
                raise RuntimeError("学習データに局面が1つも無い")
            stats = shikibu_feat.psv_stats(files)
            missing = bool(stats["result_missing"])
            self._check_missing(missing, setting)
            est = None
            if setting is None:
                self.status.enter("eval_scale")
                # 推定に使う局面は scale_sample 以下（全件モードでも、全件を標本にしない）
                est = float(shikibu_feat.estimate_eval_scale(files, sample=min(count, t["scale_sample"]),
                                                             seed=t["seed"])["eval_scale"])
            if cache_path is not None:
                write_check_cache(cache_path, key, {"count": count, "result_missing": missing,
                                                    "eval_scale": est, "saved": time.time()})
            if setting is not None:
                return setting, "config"
            return est, "auto"
        finally:
            if files is not None and files != list(self.paths) and os.path.exists(sample_path):
                os.remove(sample_path)

    def _load_validation(self):
        if not self.d["val_path"]:
            return None
        loader = shikibu_feat.BatchLoader([self.d["val_path"]], self.t["val_positions"], self.version,
                                          0, 0, self.d["num_threads"])
        batch = next(loader, None)
        n_bad = loader.skipped()
        del loader
        if n_bad > 0:
            print(f"検証データで壊れたレコードを飛ばしました（{n_bad} 件）", flush=True)
        if batch is None:
            raise RuntimeError("検証データが空")
        return to_device(batch, self.device)

    def _open_loader(self, state, epoch, perm):
        d, t = self.d, self.t
        if self.loader is not None:
            self.bad_closed += self.loader.skipped()  # 閉じるローダーで飛ばした数を残す（M8.1）
        self.loader = None  # 前のローダーを止めてからでないと、キャッシュ先を作り直せない
        self.cache_seen = 0
        self.rd_epoch, self.rd_perm = epoch, perm  # ローダーが今いる周回と順番（先読みの位置）
        paths = [self.paths[i] for i in perm]  # 周回ごとの順番
        self.loader = shikibu_feat.BatchLoader(
            paths, t["batch_size"], self.version,
            shuffle_buffer=d["shuffle_buffer"], seed=t["seed"] + epoch,
            num_threads=d["num_threads"], cache_dir=d["cache_dir"] or None,
            max_cached=d["max_cached"], resume_state=state, shuffle_warm=d["shuffle_warm"],
        )

    def _fetch(self):
        """次のバッチを読み、device へ送り始める。読んだ位置（周回・順番・loader_state）も一緒に返す（M8）。"""
        batch = next(self.loader, None)
        if batch is None:  # 1周終わった。次の周は seed + 周回番号 の順番で
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
        meta = {"epoch": self.rd_epoch, "perm": self.rd_perm, "loader_state": dict(self.loader.state()),
                "bad": bad}
        x, ev = self.prefetch.send(batch)
        return x, ev, meta

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
            "loader_state": self.loader_state,
            "eval_scale": self.eval_scale,
            "eval_scale_source": self.eval_scale_source,
            "lam": self.lam,
            "network": self.network,
            "config": self.cfg,
            "rng": rng_state(),
            "file_perm": self.perm,
            "bad_records": self.bad_records,
        }
        tmp = self.ckpt_path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.ckpt_path)  # 途中で切れても古い checkpoint.pt が残るように改名で置き換える
        self.last_save = time.monotonic()
        self.saved_step = self.step
        self.log.write({"type": "save", "step": self.step, "epoch": self.epoch, "reason": reason})
        self._emit_cache()

    def _train_step(self, x, ev, meta):
        x = self.prefetch.ready(x, ev)
        with autocast_bf16(self.device, self.t["fast"]):
            z = self.model(x["stm_indices"], x["stm_offsets"], x["nstm_indices"], x["nstm_offsets"], kings=x["kings"], base=True)
        z = z.float()
        loss = compute_loss(z, x["score"], x["result"], self.eval_scale, self.lam)
        lr = self.t["lr"] * lr_factor(self.step, self.t["warmup_steps"], self.t["total_steps"],
                                      self.t["min_lr_ratio"])
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        clip_weights_(self.model, ft=not self.lazy)  # 遅延型は第1層の行を optimizer の中で収める（M9）
        # このバッチまで学習した。checkpoint に書く位置を、このバッチの位置に進める
        self.epoch, self.perm, self.loader_state = meta["epoch"], meta["perm"], meta["loader_state"]
        self.bad_records = meta["bad"]
        self.step += 1
        self.pos_since_log += int(x["score"].shape[0])
        self.pos_since_print += int(x["score"].shape[0])
        return loss, lr

    def _write_start(self):
        self.log.write({
            "type": "start", "step": self.step, "epoch": self.epoch, "resumed": self.step > 0,
            "eval_scale": self.eval_scale, "eval_scale_source": self.eval_scale_source,
            "lam": self.lam, "network": self.network, "device": str(self.device),
            "n_files": len(self.paths), "scale_positions_used": self.scale_used, "config": self.cfg,
        })

    def _eta(self, now):
        """残り秒数の見込み。この実行の train 段階の平均速度から求める（0 以上）。"""
        done = self.step - self.step_train0
        if done <= 0:
            return None
        return max(0.0, (self.t["max_steps"] - self.step) * (now - self.t_train0) / done)

    def _progress_due(self, now):
        pe = self.t["progress_every_sec"]
        if pe > 0:
            return now - self.last_print >= pe
        return self.step % self.t["log_every_steps"] == 0

    def _report(self, now):
        """画面に進捗を1行出し、status.json を書く。"""
        t = self.t
        eta = self._eta(now)
        pps = self.pos_since_print / max(now - self.last_print, 1e-9)
        pct = 100.0 * self.step / t["max_steps"]
        print(f"[{pct:5.1f}%] step {self.step:,} / {t['max_steps']:,}  残り {fmt_sec(eta)}  {pps / 1e4:.1f}万局面/秒",
              flush=True)
        self.status.progress(self.step, t["max_steps"], eta)
        self.last_print, self.pos_since_print, self.printed_step = now, 0, self.step

    def run(self):
        t = self.t
        save_interval = t["save_every_minutes"] * 60.0
        self.pending = None
        t_fill = time.monotonic()
        if self.d["shuffle_buffer"] > 0 and self.d["shuffle_warm"] == 0:
            self.status.enter("buffer_fill", self.log)
            self.pending = self._fetch()  # 最初の1回でシャッフルバッファを満たす
        if self.pending is None and self.step < t["max_steps"]:
            self.pending = self._fetch()  # shuffle_warm > 0 のときは、少ない件数から始める（M9）
        self.timing["first_batch"] = time.monotonic() - t_fill
        self.log.write({"type": "startup", "seconds": {k: round(v, 3) for k, v in self.timing.items()},
                        "cache": self.cache_state, "check_mode": self.d["check_mode"],
                        "scale_positions_used": self.scale_used, "shuffle_warm": self.d["shuffle_warm"],
                        "ft_forward": t["ft_forward"], "ft_grad": t["ft_grad"],
                        "ft_optimizer": t["ft_optimizer"], "fast": t["fast"]})
        self.status.enter("train", self.log)
        self.t_train0, self.step_train0 = time.monotonic(), self.step
        self.last_print, self.printed_step = self.t_train0, None
        while self.step < t["max_steps"]:
            cur, self.pending = self.pending, None
            if self.step + 1 < t["max_steps"]:
                # 次のバッチを、今のステップの計算を積む前に読んで送り始める（読み込みと計算を重ねる）
                self.pending = self._fetch()
            loss, lr = self._train_step(*cur)

            now = time.monotonic()
            if self.step % t["log_every_steps"] == 0:
                self.log.write({
                    "type": "train", "step": self.step, "epoch": self.epoch,
                    "loss": float(loss.item()), "lr": lr, "bad_records": self.bad_records,
                    "progress": self.step / t["max_steps"], "eta_sec": self._eta(now),
                    "positions_per_sec": self.pos_since_log / max(now - self.last_log, 1e-9),
                    "elapsed_sec": now - self.t0,
                })
                self.last_log = now
                self.pos_since_log = 0
            if self._progress_due(now):
                self._report(now)

            if self.val is not None and t["val_every_steps"] > 0 and self.step % t["val_every_steps"] == 0:
                val_loss, val_mse = evaluate(self.model, self.val, self.eval_scale, self.lam,
                                             device=self.device, fast=t["fast"])
                self.log.write({"type": "val", "step": self.step, "val_loss": val_loss, "val_mse": val_mse})

            periodic = t["save_every_steps"] > 0 and self.step % t["save_every_steps"] == 0
            if periodic or time.monotonic() - self.last_save >= save_interval:
                self.save("periodic")

        if self.printed_step != self.step:
            self._report(time.monotonic())
        if self.saved_step != self.step:
            self.save("final")
        self.log.write({"type": "end", "step": self.step, "epoch": self.epoch,
                        "elapsed_sec": time.monotonic() - self.t0})
        self.log.close()
        self.loader = None
        self.status.progress(self.step, t["max_steps"], 0.0)
        self.status.enter("done")


def parse_args(argv):
    p = argparse.ArgumentParser(prog="python -m shikibu.train")
    p.add_argument("--config", required=True, help="設定ファイル（TOML）")
    p.add_argument("--resume", action="store_true", help="checkpoint.pt から続ける")
    p.add_argument("--force-unlock", action="store_true",
                   help="実行中かもしれないロック（train.lock）も引き継ぐ。切断直後などに使う")
    return p.parse_args(argv)


def _on_term(signum, frame):
    raise SystemExit(128 + signum)  # 後始末（ロックの解放）を通すため、例外として抜ける


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    status, lock = None, None
    signal.signal(signal.SIGTERM, _on_term)
    try:
        cfg = load_config(args.config)
        check_placeholders(cfg)
        validate(cfg)
        out_dir = cfg["output"]["out_dir"]
        lock = TrainLock.acquire(out_dir, force=args.force_unlock)  # 何かを書く前に、二重起動を確かめる
        status = Status(out_dir)
        Trainer(cfg, args.resume, status).run()
    except (ValueError, RuntimeError, FileExistsError) as e:
        # 設定やデータの問題は、学習を始める前に分かるものだけ。エラーの内容を表示して 1 で終わる。
        if status is not None and status.history:  # 段階に入っていたら status.json を error にする
            status.enter("error")
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    finally:
        if lock is not None:
            lock.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
