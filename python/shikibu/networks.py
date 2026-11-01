"""ネットワークの定義（設計書 §F3.1・§F3.2・§S5・§S11・§S12）。

- 従来の名前（RayPair_v0Network・HalfKPf_256Network など）は `NETWORKS` に登録したクラス。
- 構造の名前（例 "halfkahm2_512x2-16-32_f"）は `parse_arch` で読み、`ArchNetwork` を作る（§S12.1）。
計算式は §F3.2 のとおり。psqt・k3k3 は構造の名前でだけ使える（§S12.2・§S12.4）。
"""
import re

import torch
from torch import nn

# 特徴 → (版（因数分解なし）, 版（因数分解つき）, 次元, 1つの玉の位置あたりの駒の番号の数)
# 版の番号は shikibu_feat の version と同じ（§S11・§S12）。
FEATURES = {
    "raypair0": (0, None, 15000, None),
    "raypair1": (1, None, 115352, None),
    "halfkp": (2, 3, 125388, 1548),
    "halfka1": (4, 5, 138510, 1710),
    "halfka2": (6, 7, 131949, 1629),
    "halfkahm1": (8, 9, 76950, 1710),
    "halfkahm2": (10, 11, 73305, 1629),
    "raypair0k": (12, None, 16392, None),
    "raypair1k": (13, None, 116744, None),
}
# オプションの正規の順（f → psqt → k3k3）
OPTIONS = ("f", "psqt", "k3k3")
_ARCH_RE = re.compile(r"^([a-z0-9]+)_(\d+)x2-(\d+)-(\d+)((?:_[a-z0-9]+)*)$")


def parse_arch(name):
    """構造の名前を読んで、構造の dict を返す。誤りは日本語の ValueError。

    キー：canonical（正規の名前）, feature, W, L1, L2, factorize, psqt, stacks（1 か 9）, version, num_features。
    """
    if not isinstance(name, str):
        raise ValueError(f"構造の名前は文字列: {name!r}")
    s = name.strip().lower()
    m = _ARCH_RE.match(s)
    if not m:
        raise ValueError(f"構造の名前の形が違う（例：halfkp_256x2-32-32_f）: {name}")
    feat, W, L1, L2, opts = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(5)
    if feat not in FEATURES:
        raise ValueError(f"未知の特徴 {feat}（使えるもの: {', '.join(FEATURES)}）: {name}")
    if W % 32 or not 32 <= W <= 8192:
        raise ValueError(f"W は 32 の倍数で 32〜8192: {W}（{name}）")
    for nm, v in (("L1", L1), ("L2", L2)):
        if v % 8 or not 8 <= v <= 1024:
            raise ValueError(f"{nm} は 8 の倍数で 8〜1024: {v}（{name}）")
    toks = [t for t in opts.split("_") if t]
    for t in toks:
        if t not in OPTIONS:
            raise ValueError(f"未知のオプション {t}（使えるもの: f, psqt, k3k3）: {name}")
    if len(set(toks)) != len(toks):
        raise ValueError(f"オプションが重複している: {name}")
    factorize = "f" in toks
    v0, v1, nf, plane = FEATURES[feat]
    if factorize and v1 is None:
        raise ValueError(f"因数分解（f）は halfk… の特徴でだけ使える: {name}")
    canon = f"{feat}_{W}x2-{L1}-{L2}" + "".join("_" + o for o in OPTIONS if o in toks)
    return {"canonical": canon, "feature": feat, "W": W, "L1": L1, "L2": L2, "factorize": factorize,
            "psqt": "psqt" in toks, "stacks": 9 if "k3k3" in toks else 1,
            "version": v1 if factorize else v0, "num_features": nf + (plane if factorize else 0)}


import torch.nn.functional as _F

_PLANE = {3: (1548, 81), 5: (1710, 81), 7: (1629, 81), 9: (1710, 45), 11: (1629, 45)}


def _eff(weight, version):
    """因数分解を重みの側で行う：W_eff[k×P + p] = W[k×P + p] + W[N + p]（M8）。"""
    P, nk = _PLANE[version]
    n = P * nk
    return (weight[:n].view(nk, P, -1) + weight[n:].unsqueeze(0)).reshape(n, -1)


def _table(module, version, base):
    """第1層の表。base=True で因数分解つきの版なら、W_eff を 1 回だけ作る（M8）。None は従来どおり module を使う。"""
    if base and version in _PLANE:
        return _eff(module.weight, version)
    return None


def _bag(module, table, idx, off):
    if table is None:
        return module(idx, off)
    return _F.embedding_bag(idx, table, off, mode="sum")


class RayPairNetwork(nn.Module):
    """従来の名前のネットワークの共通部分。サブクラスで特徴数・第2層以降の幅を決める。"""

    FEATURE_VERSION = None  # 0 = RayPair_v0、1 = RayPair_v1、2・3 = HalfKP、…（§S11）
    NUM_FEATURES = None     # 特徴の次元数
    L1 = None
    L2 = None
    PSQT = False            # 従来の名前は psqt なし
    STACKS = 1              # 従来の名前は層を 1 組だけ持つ
    NEEDS_KINGS = False

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

    @property
    def INPUT_VERSION(self):
        return self.FEATURE_VERSION - 1 if self.FEATURE_VERSION in _PLANE else self.FEATURE_VERSION

    def forward(self, stm_indices, stm_offsets, nstm_indices, nstm_offsets, kings=None, base=False):
        # 手番側を先に連結する（§F3.2）。kings は従来の名前では使わない
        table = _table(self.ft, self.FEATURE_VERSION, base)
        a = self.ft_bias + _bag(self.ft, table, stm_indices, stm_offsets)
        b = self.ft_bias + _bag(self.ft, table, nstm_indices, nstm_offsets)
        x = torch.cat([a, b], dim=1)
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

    @torch.no_grad()
    def clip_(self, limit):
        """§F3.3 のクリップ。ft の重みと bias は ±1、Linear の重みは ±limit。psqt はクリップしない。"""
        self.ft.weight.clamp_(-1.0, 1.0)
        self.ft_bias.clamp_(-1.0, 1.0)
        t = self._named_tensors()
        for k in ("l1_weight", "l2_weight", "out_weight"):
            t[k].clamp_(-limit, limit)

    def decay_parameters(self):
        """weight decay の対象（§S6）：Linear の重み。"""
        t = self._named_tensors()
        return [t["l1_weight"], t["l2_weight"], t["out_weight"]]


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
# 従来の名前 → (特徴, L1, L2, 因数分解の接尾辞)。構造の名前への変換に使う
_LEGACY_ARCH = {
    "RayPair_v0Network": ("raypair0", 16, 16, ""),
    "RayPair_v1Network": ("raypair1", 32, 32, ""),
    "HalfKP_256Network": ("halfkp", 32, 32, ""),
    "HalfKPf_256Network": ("halfkp", 32, 32, "_f"),
}


def arch_of(name, ft_width=128):
    """正規の構造の名前。従来の名前は W を ft_width にして変換する（例：HalfKPf_256Network → halfkp_256x2-32-32_f）。"""
    if name in _LEGACY_ARCH:
        f, l1, l2, o = _LEGACY_ARCH[name]
        return parse_arch(f"{f}_{ft_width}x2-{l1}-{l2}{o}")["canonical"]
    return parse_arch(name)["canonical"]


def zone_of(sq):
    """玉の段の区分（その玉の側の視点）：y ≧ 6 → 0、3〜5 → 1、≦ 2 → 2（§S12.4）。sq は y × 9 + x。"""
    y = torch.div(sq, 9, rounding_mode="floor")
    return (y < 6).long() + (y < 3).long()


class ArchNetwork(nn.Module):
    """構造の名前から作るネットワーク（§S12）。S = 1 のときは従来と同じ名前・形で、S = 9 は層を 9 組持つ。

    k3k3 の計算は、全部の組で計算してから局面ごとに組を選ぶ（重みを局面ごとに複製しない。
    バッチ 65536 でも GPU のメモリに収まる）。
    """

    def __init__(self, name):
        super().__init__()
        p = parse_arch(name)
        self.arch = p
        self.ARCH = p["canonical"]
        self.FEATURE_VERSION = p["version"]
        self.NUM_FEATURES = p["num_features"]
        self.ft_width = W = p["W"]
        self.L1, self.L2 = p["L1"], p["L2"]
        self.PSQT = p["psqt"]
        self.STACKS = S = p["stacks"]
        self.NEEDS_KINGS = S > 1
        self.ft = nn.EmbeddingBag(self.NUM_FEATURES, W, mode="sum")
        self.ft_bias = nn.Parameter(torch.zeros(W))
        nn.init.normal_(self.ft.weight, 0.0, 0.01)
        if S == 1:
            self.l1 = nn.Linear(2 * W, self.L1)
            self.l2 = nn.Linear(self.L1, self.L2)
            self.out = nn.Linear(self.L2, 1)
        else:
            def stacked(i, o):
                lin = [nn.Linear(i, o) for _ in range(S)]
                w = nn.Parameter(torch.stack([x.weight.detach() for x in lin]))
                b = nn.Parameter(torch.stack([x.bias.detach() for x in lin]))
                return w, b
            self.l1_weight, self.l1_bias = stacked(2 * W, self.L1)
            self.l2_weight, self.l2_bias = stacked(self.L1, self.L2)
            self.out_weight, self.out_bias = stacked(self.L2, 1)
        if self.PSQT:
            # 駒の点数の直接の項（§S12.2）。組ごとに 1 個（S 列）
            self.psqt = nn.EmbeddingBag(self.NUM_FEATURES, S, mode="sum")
            nn.init.zeros_(self.psqt.weight)

    def _named_tensors(self):
        d = {"ft_weight": self.ft.weight, "ft_bias": self.ft_bias}
        if self.STACKS == 1:
            d.update({"l1_weight": self.l1.weight, "l1_bias": self.l1.bias, "l2_weight": self.l2.weight,
                      "l2_bias": self.l2.bias, "out_weight": self.out.weight, "out_bias": self.out.bias})
        else:
            d.update({"l1_weight": self.l1_weight, "l1_bias": self.l1_bias, "l2_weight": self.l2_weight,
                      "l2_bias": self.l2_bias, "out_weight": self.out_weight, "out_bias": self.out_bias})
        if self.PSQT:
            d["psqt_weight"] = self.psqt.weight
        return d

    export_tensors = RayPairNetwork.export_tensors
    set_from_tensors = RayPairNetwork.set_from_tensors
    clip_ = RayPairNetwork.clip_
    decay_parameters = RayPairNetwork.decay_parameters

    INPUT_VERSION = RayPairNetwork.INPUT_VERSION

    def forward(self, stm_indices, stm_offsets, nstm_indices, nstm_offsets, kings=None, base=False):
        v = self.FEATURE_VERSION
        table = _table(self.ft, v, base)
        a = self.ft_bias + _bag(self.ft, table, stm_indices, stm_offsets)
        b = self.ft_bias + _bag(self.ft, table, nstm_indices, nstm_offsets)
        h0 = torch.clamp(torch.cat([a, b], dim=1), 0.0, 1.0)
        if self.STACKS == 1:
            h = torch.clamp(self.l1(h0), 0.0, 1.0)
            h = torch.clamp(self.l2(h), 0.0, 1.0)
            z = self.out(h).squeeze(1)
            bucket = None
        else:
            if kings is None:
                raise ValueError("k3k3 の構造には kings（玉のマス）が必要")
            bucket = zone_of(kings[:, 0]) * 3 + zone_of(kings[:, 1])
            idx = bucket[:, None, None]

            def layer(h, w, bias):
                # 全部の組の出力を作ってから、局面ごとに組を選ぶ：形は（局面、組、出力）
                allh = torch.einsum("ni,soi->nso", h, w) + bias[None]
                return allh.gather(1, idx.expand(-1, 1, w.shape[1]))[:, 0]
            h = torch.clamp(layer(h0, self.l1_weight, self.l1_bias), 0.0, 1.0)
            h = torch.clamp(layer(h, self.l2_weight, self.l2_bias), 0.0, 1.0)
            z = layer(h, self.out_weight, self.out_bias)[:, 0]
        if self.PSQT:
            # (P_手番側 − P_相手側) / 2（§S12.2）。P は局面の特徴の psqt の和
            ptable = _table(self.psqt, v, base)
            ps = _bag(self.psqt, ptable, stm_indices, stm_offsets)
            pn = _bag(self.psqt, ptable, nstm_indices, nstm_offsets)
            if bucket is None:
                z = z + (ps[:, 0] - pn[:, 0]) * 0.5
            else:
                z = z + (ps.gather(1, bucket[:, None])[:, 0] - pn.gather(1, bucket[:, None])[:, 0]) * 0.5
        return z


def build(name, ft_width=128):
    """名前からネットワークを作る。従来の名前は ft_width を使い、構造の名前は名前の W を使う（ft_width は見ない）。"""
    if name in NETWORKS:
        return NETWORKS[name](ft_width=ft_width)
    try:
        return ArchNetwork(name)
    except ValueError as e:
        raise ValueError(f"未登録のネットワーク: {name}（登録済み: {sorted(NETWORKS)}、または構造の名前。{e}）") from None
