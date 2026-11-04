"""第1層（ft の重み）の遅延型 Adam と、AdamW とのまとめ役（M9）。

設定 train.ft_optimizer = "lazy_adam" のときだけ使う（既定は "adamw"＝今の動作）。

LazyAdamRows（第1層の重み・行ごと）：
- そのステップの勾配（疎）に現れた行（特徴）だけ、m・v・重みを更新する。数式は Adam と同じで、
  バイアス補正だけは全体のステップ数 t で行う（TensorFlow の LazyAdam と同じ考え方）。
- 使われなかった行は、m・v・重みのどれも動かさない。
  【数学的な違い】密な AdamW は、使われなかった行も m の慣性で重みが動き、v は減衰する。
  遅延型はそれをしない。よって同じ設定でも、損失の推移は完全には一致しない（M9 の確かめは「同程度に下がる」まで）。
- 重みの減衰（weight decay）はかけない。今の設定でも第1層の重みは減衰の対象外（decay_parameters に入らない）なので、
  今の AdamW と減衰の扱いは同じ。
- クリップ（±1）は、更新した行だけに行う。使われていない行は、もともと範囲内にあるので触らなくてよい。

CombinedOptimizer：第1層以外は AdamW（今と同じ）、第1層は LazyAdamRows を 1 つにまとめる。
学習の run() から見たら今の AdamW と同じ使い方（param_groups の lr・step・zero_grad・state_dict）。
"""
import torch


class LazyAdamRows:
    """第1層の重み（行 × 幅）を、使った行だけ更新する Adam（M9）。"""

    def __init__(self, param, lr, betas=(0.9, 0.999), eps=1e-8):
        self.param = param
        self.param_groups = [{"params": [param], "lr": lr, "betas": betas, "eps": eps, "weight_decay": 0.0}]
        self.m = torch.zeros_like(param)  # 1 次モーメント（密。AdamW と同じ大きさ）
        self.v = torch.zeros_like(param)  # 2 次モーメント
        self.t = 0                        # 全体の更新回数（バイアス補正に使う）

    def zero_grad(self, set_to_none=True):
        if set_to_none:
            self.param.grad = None
        elif self.param.grad is not None:
            self.param.grad.zero_()

    @torch.no_grad()
    def step(self):
        self.t += 1
        g = self.param.grad
        if g is None:
            return
        grp = self.param_groups[0]
        lr, (b1, b2), eps = grp["lr"], grp["betas"], grp["eps"]
        if g.is_sparse:
            g = g.coalesce()              # 同じ行の勾配を足し合わせ、行の番号を重複なしにする
            rows = g.indices()[0]
            vals = g.values()
        else:                             # 密な勾配（ft_grad = "dense"）のときは、全部の行を使ったものとして扱う
            rows = torch.arange(g.shape[0], device=g.device)
            vals = g
        if rows.numel() == 0:
            return
        m = self.m.index_select(0, rows).mul_(b1).add_(vals, alpha=1.0 - b1)
        v = self.v.index_select(0, rows).mul_(b2).addcmul_(vals, vals, value=1.0 - b2)
        self.m.index_copy_(0, rows, m)
        self.v.index_copy_(0, rows, v)
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t
        denom = (v / bc2).sqrt_().add_(eps)
        p = self.param.index_select(0, rows).addcdiv_(m, denom, value=-lr / bc1)
        p.clamp_(-1.0, 1.0)               # §F3.3：第1層の重みは ±1 に収める（使った行だけ）
        self.param.index_copy_(0, rows, p)

    def state_dict(self):
        return {"kind": "lazy_rows", "t": self.t, "m": self.m, "v": self.v,
                "lr": self.param_groups[0]["lr"]}

    def load_state_dict(self, d):
        if d.get("kind") != "lazy_rows" or d["m"].shape != self.m.shape:
            raise ValueError("checkpoint の第1層の optimizer が設定と合わない（ft_optimizer・次元を確かめる）")
        self.t = int(d["t"])
        self.m.copy_(d["m"])
        self.v.copy_(d["v"])
        self.param_groups[0]["lr"] = d["lr"]


class CombinedOptimizer:
    """第1層以外の AdamW ＋ 第1層の LazyAdamRows を 1 つにまとめる。"""

    def __init__(self, adamw, lazy):
        self.adamw = adamw
        self.lazy = lazy

    @property
    def param_groups(self):
        return self.adamw.param_groups + self.lazy.param_groups

    def zero_grad(self, set_to_none=True):
        self.adamw.zero_grad(set_to_none=set_to_none)
        self.lazy.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.adamw.step()
        self.lazy.step()

    def state_dict(self):
        return {"kind": "lazy_adam", "adamw": self.adamw.state_dict(), "lazy": self.lazy.state_dict()}

    def load_state_dict(self, d):
        if d.get("kind") != "lazy_adam":
            raise ValueError("checkpoint の optimizer が設定（ft_optimizer）と違う")
        self.adamw.load_state_dict(d["adamw"])
        self.lazy.load_state_dict(d["lazy"])


def is_lazy_state(d):
    """checkpoint の optimizer の状態が遅延型の形か。"""
    return isinstance(d, dict) and d.get("kind") == "lazy_adam"
