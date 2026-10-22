"""ネットワークの定義（設計書 §F3.1・§F3.2・§S5）。

名前からクラスを引けるように `NETWORKS` に登録する。計算式は §F3.2 のとおり。
"""
import torch
from torch import nn


class RayPairNetwork(nn.Module):
    """共通部分。サブクラスで特徴数・第2層以降の幅を決める。"""

    FEATURE_VERSION = None  # 0・1 = RayPair、2 = HalfKP、3 = HalfKP（因数分解つき）
    NUM_FEATURES = None     # 特徴の次元数
    L1 = None
    L2 = None

    def __init__(self, ft_width=128):
        super().__init__()
        W = ft_width
        self.ft_width = W
        # 第1層：視点で共有する。重複したインデックスは回数ぶん加算される（mode="sum"）
        self.ft = nn.EmbeddingBag(self.NUM_FEATURES, W, mode="sum")
        self.ft_bias = nn.Parameter(torch.zeros(W))
        self.l1 = nn.Linear(2 * W, self.L1)
        self.l2 = nn.Linear(self.L1, self.L2)
        self.out = nn.Linear(self.L2, 1)
        nn.init.normal_(self.ft.weight, 0.0, 0.01)

    def _accumulate(self, indices, offsets):
        return self.ft_bias + self.ft(indices, offsets)

    def forward(self, stm_indices, stm_offsets, nstm_indices, nstm_offsets):
        # 手番側を先に連結する（§F3.2）
        x = torch.cat(
            [self._accumulate(stm_indices, stm_offsets), self._accumulate(nstm_indices, nstm_offsets)],
            dim=1,
        )
        h = torch.clamp(x, 0.0, 1.0)
        h = torch.clamp(self.l1(h), 0.0, 1.0)
        h = torch.clamp(self.l2(h), 0.0, 1.0)
        return self.out(h).squeeze(1)

    def _named_tensors(self):
        # §F4.2 の名前。書き出し（export.py）と同じ名前を使う
        return {
            "ft_weight": self.ft.weight,
            "ft_bias": self.ft_bias,
            "l1_weight": self.l1.weight,
            "l1_bias": self.l1.bias,
            "l2_weight": self.l2.weight,
            "l2_bias": self.l2.bias,
            "out_weight": self.out.weight,
            "out_bias": self.out.bias,
        }

    def export_tensors(self):
        """重みを float32 の CPU テンソルの dict で返す（§F4.2 の名前と形）。"""
        return {k: v.detach().float().cpu().clone() for k, v in self._named_tensors().items()}

    @torch.no_grad()
    def set_from_tensors(self, d):
        for k, v in self._named_tensors().items():
            v.copy_(d[k].to(device=v.device, dtype=v.dtype))


class RayPair_v0Network(RayPairNetwork):
    FEATURE_VERSION = 0
    NUM_FEATURES = 15000
    L1 = 16
    L2 = 16


class RayPair_v1Network(RayPairNetwork):
    FEATURE_VERSION = 1
    NUM_FEATURES = 115352
    L1 = 32
    L2 = 32


class HalfKP_256Network(RayPairNetwork):
    """HalfKP（やねうら王の halfkp_256x2-32-32）。計算式は RayPair と同じで、特徴と幅だけが違う（§S11）。"""
    FEATURE_VERSION = 2
    NUM_FEATURES = 125388
    L1 = 32
    L2 = 32


class HalfKPf_256Network(RayPairNetwork):
    """HalfKP の因数分解つき（HalfKP^）。学習のときだけ玉の位置に関係ない特徴を足す（§S11）。"""
    FEATURE_VERSION = 3
    NUM_FEATURES = 126936
    L1 = 32
    L2 = 32


NETWORKS = {
    "RayPair_v0Network": RayPair_v0Network,
    "RayPair_v1Network": RayPair_v1Network,
    "HalfKP_256Network": HalfKP_256Network,
    "HalfKPf_256Network": HalfKPf_256Network,
}


def build(name, ft_width=128):
    if name not in NETWORKS:
        raise ValueError(f"未登録のネットワーク: {name}（登録済み: {sorted(NETWORKS)}）")
    return NETWORKS[name](ft_width=ft_width)
