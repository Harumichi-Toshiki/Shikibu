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
    if s.startswith("sfnn_"):
        return _parse_sfnn(s, name)
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
            "version": v1 if factorize else v0, "num_features": nf + (plane if factorize else 0),
            "sfnn": False}


# SFNN（M10）の特徴は halfk… だけ。構造の名前：sfnn_<特徴>_<W>-<H1>-<H2>[_f][_k3k3]
SFNN_FEATURES = ("halfkp", "halfka1", "halfka2", "halfkahm1", "halfkahm2")
_SFNN_RE = re.compile(r"^sfnn_([a-z0-9]+)_(\d+)-(\d+)-(\d+)((?:_[a-z0-9]+)*)$")


def _parse_sfnn(s, name):
    """SFNN の構造の名前（小文字にしたもの s）を読む。返す dict のキーは parse_arch と同じ（sfnn は True）。
    従来の構造の L1・L2 の代わりに H1（fc_0 の出力の数、8n か 8n−1）・H2 を持つ。"""
    m = _SFNN_RE.match(s)
    if not m:
        raise ValueError(f"構造の名前の形が違う（例：sfnn_halfkahm2_512-15-32_f_k3k3）: {name}")
    feat, W, H1, H2 = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    if feat not in SFNN_FEATURES:
        raise ValueError(f"SFNN の特徴は halfk… だけ（{', '.join(SFNN_FEATURES)}）: {name}")
    if W % 64 or not 64 <= W <= 8192:
        raise ValueError(f"W は 64 の倍数で 64〜8192: {W}（{name}）")
    if not (7 <= H1 <= 255 and H1 % 8 in (0, 7)):
        raise ValueError(f"H1 は 8n か 8n−1 で 7〜255: {H1}（{name}）")
    if H2 % 8 or not 8 <= H2 <= 1024:
        raise ValueError(f"H2 は 8 の倍数で 8〜1024: {H2}（{name}）")
    toks = [t for t in m.group(5).split("_") if t]
    for t in toks:
        if t == "psqt":
            raise ValueError(f"SFNN に psqt は無い: {name}")
        if t not in OPTIONS:
            raise ValueError(f"未知のオプション {t}（使えるもの: f, k3k3）: {name}")
    if len(set(toks)) != len(toks):
        raise ValueError(f"オプションが重複している: {name}")
    factorize, k3k3 = "f" in toks, "k3k3" in toks
    v0, v1, nf, plane = FEATURES[feat]
    canon = f"sfnn_{feat}_{W}-{H1}-{H2}" + ("_f" if factorize else "") + ("_k3k3" if k3k3 else "")
    return {"canonical": canon, "sfnn": True, "feature": feat, "W": W, "H1": H1, "H2": H2,
            "factorize": factorize, "psqt": False, "stacks": 9 if k3k3 else 1, "shortcut": H1 % 8 == 7,
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


def _ft_table(net, module, base):
    """dense（既定）の経路の表。sparse の経路（FT_FORWARD_SPARSE）では W_eff を作らないので None（M9）。"""
    if net.FT_FORWARD_SPARSE and base:
        return None
    return _table(module, net.FEATURE_VERSION, base)


def _ft_pair_triton(net, si, so, ni, no):
    """FT_KERNEL = "triton" の第1層（バイアス込み）。因数分解つきなら、本体の行と共通の行を足す（M9.1）。"""
    from shikibu import ft_triton
    v = net.FEATURE_VERSION
    if v in _PLANE:
        P, nk = _PLANE[v]
        return ft_triton.ft_pair(net, si, so, ni, no, P * nk, P, True)
    return ft_triton.ft_pair(net, si, so, ni, no, net.NUM_FEATURES, 1, False)


def _bag(module, table, idx, off):
    if table is None:
        return module(idx, off)
    return _F.embedding_bag(idx, table, off, mode="sum")


def _bag_ft(net, module, table, base, idx, off, sparse):
    """第1層（と psqt）の有効な行の和（§F3.2）。

    - 因数分解つきで FT_FORWARD_SPARSE のとき：W_eff[k×P+p] = W[k×P+p] + W[N+p] の和を、
      「本体の行の和」＋「共通部分の行（idx mod P）の和」で作る。W_eff を作らないので、密な計算がない（M9）。
    - それ以外は従来どおり（table があれば密な W_eff の表。なければ module）。sparse は勾配を疎にする（FT_GRAD_SPARSE）。
    """
    v = net.FEATURE_VERSION
    if base and v in _PLANE and net.FT_FORWARD_SPARSE:
        P, nk = _PLANE[v]
        n = P * nk
        return (_F.embedding_bag(idx, module.weight, off, mode="sum", sparse=sparse)
                + _F.embedding_bag(idx % P + n, module.weight, off, mode="sum", sparse=sparse))
    if table is not None:
        if sparse:
            raise ValueError("因数分解つきの疎な勾配には ft_forward = \"sparse\" が必要")
        return _F.embedding_bag(idx, table, off, mode="sum")
    if sparse:
        return _F.embedding_bag(idx, module.weight, off, mode="sum", sparse=True)
    return module(idx, off)


class RayPairNetwork(nn.Module):
    """従来の名前のネットワークの共通部分。サブクラスで特徴数・第2層以降の幅を決める。"""

    FEATURE_VERSION = None  # 0 = RayPair_v0、1 = RayPair_v1、2・3 = HalfKP、…（§S11）
    NUM_FEATURES = None     # 特徴の次元数
    L1 = None
    L2 = None
    PSQT = False            # 従来の名前は psqt なし
    STACKS = 1              # 従来の名前は層を 1 組だけ持つ
    NEEDS_KINGS = False
    FT_FORWARD_SPARSE = False  # M9：第1層の順伝播を有効な行だけで行う（既定 False = 従来の計算）
    FT_GRAD_SPARSE = False     # M9：第1層の勾配を疎にする（オプティマイザは遅延型）
    FT_KERNEL = "torch"        # M9.1：第1層の計算（"torch" = 従来、"triton" = ft_triton の専用カーネル）
    SFNN = False               # M10：従来の構造は SFNN ではない

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
        if self.FT_KERNEL == "triton" and base:
            a, b = _ft_pair_triton(self, stm_indices, stm_offsets, nstm_indices, nstm_offsets)
        else:
            table = _ft_table(self, self.ft, base)
            sp = self.FT_GRAD_SPARSE
            a = self.ft_bias + _bag_ft(self, self.ft, table, base, stm_indices, stm_offsets, sp)
            b = self.ft_bias + _bag_ft(self, self.ft, table, base, nstm_indices, nstm_offsets, sp)
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
    def clip_(self, limit, ft=True):
        """§F3.3 のクリップ。ft の重みと bias は ±1、Linear の重みは ±limit。psqt はクリップしない。

        ft=False なら ft の重みは触らない（遅延型のオプティマイザが、使った行だけを先に収める。M9）。
        """
        if ft:
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

    SFNN = False  # M10：SFNN は SFNNNetwork

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
        self.FT_FORWARD_SPARSE = False  # M9（学習の設定で変える）
        self.FT_GRAD_SPARSE = False
        self.FT_KERNEL = "torch"
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
        if self.FT_KERNEL == "triton" and base:
            a, b = _ft_pair_triton(self, stm_indices, stm_offsets, nstm_indices, nstm_offsets)
        else:
            table = _ft_table(self, self.ft, base)
            sp = self.FT_GRAD_SPARSE
            a = self.ft_bias + _bag_ft(self, self.ft, table, base, stm_indices, stm_offsets, sp)
            b = self.ft_bias + _bag_ft(self, self.ft, table, base, nstm_indices, nstm_offsets, sp)
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
            ptable = _ft_table(self, self.psqt, base)
            ps = _bag_ft(self, self.psqt, ptable, base, stm_indices, stm_offsets, False)
            pn = _bag_ft(self, self.psqt, ptable, base, nstm_indices, nstm_offsets, False)
            if bucket is None:
                z = z + (ps[:, 0] - pn[:, 0]) * 0.5
            else:
                z = z + (ps.gather(1, bucket[:, None])[:, 0] - pn.gather(1, bucket[:, None])[:, 0]) * 0.5
        return z


_SFNN_SCALE = 127 / 128  # 2 つの値の積（FT の出力・fc_0 の 2 乗の ClippedReLU）の倍率（§M10）


def _uniform(shape, fan_in):
    """nn.Linear と同じ初期値（一様分布 ±1/√fan_in）。"""
    return (torch.rand(shape) * 2 - 1) / fan_in ** 0.5


def _stack_linear(h, w, b, bucket):
    """組ごとの全結合：局面 n の入力 h[n] に、その局面のバケツ bucket[n] の組の w・b を掛ける（形は (局面, 出力)）。

    組ごとに局面をまとめて、その組の行列だけを掛ける（計算は 1 組ぶん。全部の組で計算してから選ぶ方法の 1/S）。
    bucket が None（組が 1 つ）なら、ふつうの全結合。
    """
    if bucket is None:
        return _F.linear(h, w[0], b[0])
    S = w.shape[0]
    order = torch.argsort(bucket, stable=True)
    counts = torch.bincount(bucket, minlength=S).tolist()
    parts = h[order].split(counts)
    y = torch.cat([_F.linear(parts[s], w[s], b[s]) for s in range(S)], dim=0)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=order.device)
    return y[inv]


class SFNNNetwork(nn.Module):
    """SFNN（やねうら王の SFNN・レイヤースタック。M10）。式は tests/test_m10_acceptance.py の冒頭のとおり。

    FT（第1層）は ArchNetwork と同じ（因数分解・M9 の疎な計算・M9.1 の triton も使える）。
    そのあとが SFNN：FT の出力を 2 つに分けて掛け合わせ、fc_0 の出力の 2 乗の ClippedReLU と ClippedReLU の両方を
    fc_1 に入れ、fc_0 の余りの 1 個（近道、H1 が 8n−1 のとき）を出力に足す。k3k3 なら 9 組の層から玉の段で 1 組を選ぶ。
    """

    SFNN = True
    PSQT = False

    def __init__(self, name):
        super().__init__()
        p = parse_arch(name)
        if not p["sfnn"]:
            raise ValueError(f"SFNN の構造の名前ではない: {name}")
        self.arch = p
        self.ARCH = p["canonical"]
        self.FEATURE_VERSION = p["version"]
        self.NUM_FEATURES = p["num_features"]
        self.ft_width = W = p["W"]
        self.H1, self.H2 = p["H1"], p["H2"]
        self.SHORTCUT = p["shortcut"]
        self.O0 = O0 = self.H1 + (1 if self.SHORTCUT else 0)
        self.STACKS = S = p["stacks"]
        self.NEEDS_KINGS = S > 1
        self.FT_FORWARD_SPARSE = False  # M9（学習の設定で変える）
        self.FT_GRAD_SPARSE = False
        self.FT_KERNEL = "torch"
        self.ft = nn.EmbeddingBag(self.NUM_FEATURES, W, mode="sum")
        self.ft_bias = nn.Parameter(torch.zeros(W))
        nn.init.normal_(self.ft.weight, 0.0, 0.01)
        # 層は組ごとに S 組。fc_0 → fc_1 → fc_2 の順
        self.l0_weight = nn.Parameter(_uniform((S, O0, W), W))
        self.l0_bias = nn.Parameter(_uniform((S, O0), W))
        self.l1_weight = nn.Parameter(_uniform((S, self.H2, 2 * self.H1), 2 * self.H1))
        self.l1_bias = nn.Parameter(_uniform((S, self.H2), 2 * self.H1))
        self.l2_weight = nn.Parameter(_uniform((S, 1, self.H2), self.H2))
        self.l2_bias = nn.Parameter(_uniform((S, 1), self.H2))

    @property
    def INPUT_VERSION(self):
        return self.FEATURE_VERSION - 1 if self.FEATURE_VERSION in _PLANE else self.FEATURE_VERSION

    def _named_tensors(self):
        # tests/test_m10_acceptance.py の名前と形（S = 組の数、O0 = fc_0 の出力の数）
        return {"ft_weight": self.ft.weight, "ft_bias": self.ft_bias,
                "l0_weight": self.l0_weight, "l0_bias": self.l0_bias,
                "l1_weight": self.l1_weight, "l1_bias": self.l1_bias,
                "l2_weight": self.l2_weight, "l2_bias": self.l2_bias}

    export_tensors = RayPairNetwork.export_tensors
    set_from_tensors = RayPairNetwork.set_from_tensors

    @torch.no_grad()
    def clip_(self, limit, ft=True):
        """§M10：ft の重みと bias は ±1、層の重みは ±limit。層のバイアスはクリップしない。ft=False なら ft の重みは触らない。"""
        if ft:
            self.ft.weight.clamp_(-1.0, 1.0)
        self.ft_bias.clamp_(-1.0, 1.0)
        for w in (self.l0_weight, self.l1_weight, self.l2_weight):
            w.clamp_(-limit, limit)

    def decay_parameters(self):
        """weight decay の対象：層の重み（従来と同じく全結合の重み）。"""
        return [self.l0_weight, self.l1_weight, self.l2_weight]

    def forward(self, stm_indices, stm_offsets, nstm_indices, nstm_offsets, kings=None, base=False):
        S = self.STACKS
        if self.FT_KERNEL == "triton" and base:
            a, b = _ft_pair_triton(self, stm_indices, stm_offsets, nstm_indices, nstm_offsets)
        else:
            table = _ft_table(self, self.ft, base)
            sp = self.FT_GRAD_SPARSE
            a = self.ft_bias + _bag_ft(self, self.ft, table, base, stm_indices, stm_offsets, sp)
            b = self.ft_bias + _bag_ft(self, self.ft, table, base, nstm_indices, nstm_offsets, sp)
        # 手番側の視点の FT の出力を 2 つに分けて掛け合わせ、相手側も同じ。x は W 個（§M10）
        ca, cb = torch.clamp(a, 0.0, 1.0), torch.clamp(b, 0.0, 1.0)
        half = self.ft_width // 2
        x = torch.cat([ca[:, :half] * ca[:, half:] * _SFNN_SCALE, cb[:, :half] * cb[:, half:] * _SFNN_SCALE], dim=1)
        if S == 1:
            bucket = None
        else:
            if kings is None:
                raise ValueError("k3k3 の構造には kings（玉のマス）が必要")
            bucket = zone_of(kings[:, 0]) * 3 + zone_of(kings[:, 1])
        H1 = self.H1
        y0 = _stack_linear(x, self.l0_weight, self.l0_bias, bucket)
        h = torch.cat([torch.clamp(y0[:, :H1] * y0[:, :H1] * _SFNN_SCALE, 0.0, 1.0),
                       torch.clamp(y0[:, :H1], 0.0, 1.0)], dim=1)
        a1 = torch.clamp(_stack_linear(h, self.l1_weight, self.l1_bias, bucket), 0.0, 1.0)
        z = _stack_linear(a1, self.l2_weight, self.l2_bias, bucket)[:, 0]
        if self.SHORTCUT:
            z = z + y0[:, H1]
        return z


def build(name, ft_width=128):
    """名前からネットワークを作る。従来の名前は ft_width を使い、構造の名前は名前の W を使う（ft_width は見ない）。"""
    if name in NETWORKS:
        return NETWORKS[name](ft_width=ft_width)
    try:
        return SFNNNetwork(name) if parse_arch(name)["sfnn"] else ArchNetwork(name)
    except ValueError as e:
        raise ValueError(f"未登録のネットワーク: {name}（登録済み: {sorted(NETWORKS)}、または構造の名前。{e}）") from None
