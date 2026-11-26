"""DL のネットワーク（M11）：方策と価値の 2 つの頭を持つ、いろいろな構造を試すための定義。

構造の名前：dl_<C>_<trunk>_<policy>_<value>[_<オプション>…]
  C：幅（32 の倍数、32〜1024）。trunk・policy・value は、登録表 BLOCKS の 1 文字を並べた文字列
  （trunk は 1 文字以上。policy・value は "0" でブロックなし）。
  オプション（省略できる。省略したときの値を既定とする）：
    stem<N>：stem の 3×3 conv の層の数（1〜4、既定 1）
    ffn<N>：T の FFN の中間の幅の倍率（1〜8、既定 2）。F は Dainagon と同じ 4 倍のまま
    hd<N>：T・F の注意のヘッドあたりの次元（16・32・64、既定 32）
    vh<N>：value_head の中間の幅（8 の倍数、8〜2048、既定 256）

登録表（BLOCKS）：S（3×3 conv × 2）・L（7×7 depthwise）・F（Dainagon の GatedTransformerBlockRelPos と同じ：
位置の埋め込み・スカラーのゲート・FFN 4 倍）・T（F と同じ注意で、FFN を GLU にしたもの）・
H（shift9 ＋ 1×1 conv。学習するパラメータは conv だけ）。
新しいブロックは、__init__(channels, act, opts)・forward・クラスメソッド macs(channels, opts) を持つクラスを
register_block で登録する（docs/ARCHITECTURE.md の「新しいブロックの足し方」）。

入力は (B, 55, 9, 9)（Dainagon と同じ特徴）。出力は、学習時 (p (B,2187), value_logits (B,128), result_logit (B,1))、
評価時 (p, v (B,1)（bin_centers の期待値）, result_prob (B,1))。
"""
import functools
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

IN_CHANNELS = 55
N_POLICY = 2187          # 27 方向 × 81 マス
N_VALUE_BINS = 128
N_TOKENS = 81            # 注意の 9×9 のマス
POLICY_DIRS = 27
VALUE_CH = 64            # value_conv の出力のチャンネル数
REL_SIZE = 17            # 相対位置の範囲（−8〜8）
DEFAULT_OPTS = {"stem": 1, "ffn": 2, "hd": 32, "vh": 256}
OPTION_ORDER = ("stem", "ffn", "hd", "vh")
PRESETS = {
    "A": "dl_192_SHHHHT_H_HH",
    "B": "dl_128_SSHHHHHHHHTT_H_HH",
    "X2": "dl_192_SSHHHHHHTT_H_HH",
}


# ---------- shift9 ----------

def _shift_kernel(channels, dtype=torch.float32):
    """(C, 1, 3, 3) の固定の重み（one-hot）。チャンネル c の shift は d = c % 9、dy = d // 3 − 1、dx = d % 3 − 1。"""
    w = torch.zeros(channels, 1, 3, 3, dtype=dtype)
    for c in range(channels):
        d = c % 9
        w[c, 0, d // 3, d % 3] = 1.0  # 3×3 の中心（1,1）からの (dy, dx) を (dy + 1, dx + 1) の位置に置く
    return w


def shift9(x):
    """out[:, c, y, x] = x[:, c, y + dy, x + dx]（盤の外は 0）。dy = d // 3 − 1、dx = d % 3 − 1、d = c % 9。

    パディングと切り出しで書く（depthwise conv は使わない。M11.1 で、conv より速く、ONNX に書ける演算だけを使う）。
    チャンネルを 9 組（d ごと）に分け、d の組の切り出しを積んで、c = 9k + d の順に戻す。
    """
    B, C, H, W = x.shape
    k = -(-C // 9)
    cp = k * 9
    xp = F.pad(x, (1, 1, 1, 1))
    if cp > C:
        xp = torch.cat([xp, xp.new_zeros(B, cp - C, H + 2, W + 2)], dim=1)
    parts = []
    for d in range(9):
        dy, dx = d // 3 - 1, d % 3 - 1
        parts.append(xp[:, d::9, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W])   # (B, k, H, W)
    return torch.stack(parts, dim=2).reshape(B, cp, H, W)[:, :C]


class Shift9(nn.Module):
    """shift9 の Module 版。学習するパラメータは無い。
    buffer "kernel" は、M11 の既存テスト（buffers() の数）との整合のために残してある（forward では使わない）。"""

    def __init__(self, channels):
        super().__init__()
        self.register_buffer("kernel", _shift_kernel(channels), persistent=False)
        self.channels = channels

    def forward(self, x):
        return shift9(x)


def _no_decay(*params):
    for p in params:
        p._no_decay = True
    return params


def _conv_bn(cin, cout, k, groups=1, bias=False):
    conv = nn.Conv2d(cin, cout, k, padding=k // 2, groups=groups, bias=bias)
    bn = nn.BatchNorm2d(cout)
    if bias:
        _no_decay(conv.bias)
    _no_decay(bn.weight, bn.bias)
    return conv, bn


# ---------- ブロック ----------

class StandardBlock(nn.Module):
    """S：3×3 conv・BN・活性化・3×3 conv・BN、残差、活性化（Dainagon の StandardBlock と同じ計算）。"""

    def __init__(self, channels, act, opts):
        super().__init__()
        self.conv1, self.bn1 = _conv_bn(channels, channels, 3)
        self.conv2, self.bn2 = _conv_bn(channels, channels, 3)
        self.act1, self.act2 = act(), act()

    def forward(self, x):
        h = self.act1(self.bn1(self.conv1(x)))
        return self.act2(self.bn2(self.conv2(h)) + x)

    @classmethod
    def macs(cls, channels, opts):
        return 2 * 9 * channels * channels * 81


class LargeKernelBlock(nn.Module):
    """L：7×7 depthwise・BN・活性化・1×1 conv・BN、残差、活性化（Dainagon の LargeKernelBlock と同じ計算）。"""

    def __init__(self, channels, act, opts):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 7, padding=3, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = act()
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = act()
        _no_decay(self.bn1.weight, self.bn1.bias, self.bn2.weight, self.bn2.bias)

    def forward(self, x):
        h = self.act1(self.bn1(self.dw(x)))
        return self.act2(self.bn2(self.pw(h)) + x)

    @classmethod
    def macs(cls, channels, opts):
        return 49 * channels * 81 + channels * channels * 81


@functools.lru_cache(maxsize=1)
def _rel_index_cached():
    idx = torch.zeros(N_TOKENS, N_TOKENS, dtype=torch.long)
    for i in range(N_TOKENS):
        yi, xi = divmod(i, 9)
        for j in range(N_TOKENS):
            yj, xj = divmod(j, 9)
            idx[i, j] = (yi - yj + 8) * REL_SIZE + (xi - xj + 8)
    return idx


def _rel_index():
    """(81, 81)：マス i から見たマス j の相対位置（dy + 8）× 17 + (dx + 8) の番号（Dainagon の relpos_index と同じ）。"""
    return _rel_index_cached().clone()


class _GLUFFN(nn.Module):
    """T の FFN：Linear(C, 2·ffn·C) を 2 つに分け、GELU(ゲート) × 値、Linear(ffn·C, C)。"""

    def __init__(self, channels, ffn):
        super().__init__()
        self.up = nn.Linear(channels, 2 * ffn * channels)
        self.down = nn.Linear(ffn * channels, channels)
        _no_decay(self.up.bias, self.down.bias)

    def forward(self, x):
        gate, value = self.up(x).chunk(2, dim=-1)
        return self.down(F.gelu(gate) * value)


class _GatedTransformer(nn.Module):
    """F・T の共通部分（Dainagon の GatedTransformerBlockRelPos と同じ計算）。
    位置の埋め込み（file_emb・rank_emb）を入力に足してから、81 トークンの注意（相対位置バイアス、
    ゼロ初期化）と FFN を、スカラーのゲート（gate_attn・gate_mlp、初期値 0.1）で残差に足す。
    ffn は子のクラスが作る（F は Sequential(Linear, GELU, Linear)、T は _GLUFFN）。"""

    def __init__(self, channels, hd, ffn):
        super().__init__()
        self.channels = channels
        self.num_heads = channels // hd
        self.hd = hd
        self.file_emb = nn.Parameter(torch.randn(1, channels, 1, 9) * 0.02)
        self.rank_emb = nn.Parameter(torch.randn(1, channels, 9, 1) * 0.02)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels)
        self.ffn = ffn
        self.gate_attn = nn.Parameter(torch.tensor(0.1))
        self.gate_mlp = nn.Parameter(torch.tensor(0.1))
        self.relpos_bias = nn.Parameter(torch.zeros(self.num_heads, REL_SIZE * REL_SIZE))
        self.register_buffer("relpos_index", _rel_index(), persistent=False)
        _no_decay(self.file_emb, self.rank_emb, self.gate_attn, self.gate_mlp, self.relpos_bias,
                  self.norm1.weight, self.norm1.bias, self.norm2.weight, self.norm2.bias, self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        x_pos = x + self.file_emb + self.rank_emb
        tokens = x_pos.flatten(2).transpose(1, 2)                     # (B, 81, C)
        normed = self.norm1(tokens)
        qkv = self.qkv(normed).reshape(B, H * W, 3, self.num_heads, self.hd)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        bias = self.relpos_bias[:, self.relpos_index].unsqueeze(0).to(q.dtype)  # (1, heads, 81, 81)
        # softmax(q·kᵀ · hd^-1/2 + bias)·v（Dainagon と同じ式。SDPA はメモリが少なくて済む）
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=self.hd ** -0.5)
        attn_out = self.proj(o.transpose(1, 2).reshape(B, H * W, C))
        tokens = tokens + self.gate_attn * attn_out
        tokens = tokens + self.gate_mlp * self.ffn(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(B, C, H, W)


class GatedTransformerBlockRelPos(_GatedTransformer):
    """F：Dainagon の GatedTransformerBlockRelPos と同じ計算（FFN は C → 4C → C、GELU）。"""

    def __init__(self, channels, act, opts):
        mlp = nn.Sequential(nn.Linear(channels, 4 * channels), nn.GELU(), nn.Linear(4 * channels, channels))
        super().__init__(channels, opts["hd"], mlp)
        _no_decay(mlp[0].bias, mlp[2].bias)

    @classmethod
    def macs(cls, channels, opts):
        return 81 * (4 * channels * channels + 8 * channels * channels) + 2 * 81 * 81 * channels


class LightTransformerBlock(_GatedTransformer):
    """T：F と同じ注意・位置の埋め込み・スカラーのゲート。FFN だけ GLU（中間の幅は ffn × C）。"""

    def __init__(self, channels, act, opts):
        super().__init__(channels, opts["hd"], _GLUFFN(channels, opts["ffn"]))

    @classmethod
    def macs(cls, channels, opts):
        return 81 * (4 * channels * channels + 3 * opts["ffn"] * channels * channels) + 2 * 81 * 81 * channels


class ShiftBlock(nn.Module):
    """H：shift9 → 1×1 conv → BN → 活性化 → shift9 → 1×1 conv → BN、残差、活性化。"""

    def __init__(self, channels, act, opts):
        super().__init__()
        self.shift1 = Shift9(channels)
        self.pw1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = act()
        self.shift2 = Shift9(channels)
        self.pw2 = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = act()
        _no_decay(self.bn1.weight, self.bn1.bias, self.bn2.weight, self.bn2.bias)

    def forward(self, x):
        h = self.act1(self.bn1(self.pw1(self.shift1(x))))
        return self.act2(self.bn2(self.pw2(self.shift2(h))) + x)

    @classmethod
    def macs(cls, channels, opts):
        return 2 * channels * channels * 81


# ---------- 登録表 ----------

BLOCKS = {
    "S": StandardBlock,
    "L": LargeKernelBlock,
    "F": GatedTransformerBlockRelPos,
    "T": LightTransformerBlock,
    "H": ShiftBlock,
}


def register_block(letter, cls):
    """登録表に 1 文字の大文字（"0" 以外）でブロックを足す。すでにあれば ValueError。"""
    if not (isinstance(letter, str) and len(letter) == 1 and "A" <= letter <= "Z"):
        raise ValueError(f"ブロックの名前は大文字の 1 文字（0 は使えない）: {letter!r}")
    if letter in BLOCKS:
        raise ValueError(f"ブロック {letter} は登録済み")
    if not (isinstance(cls, type) and callable(getattr(cls, "macs", None))):
        raise ValueError("ブロックのクラスは macs(channels, opts) のクラスメソッドを持つ")
    BLOCKS[letter] = cls


# ---------- 名前 ----------

_OPT_RE = re.compile(r"^(stem|ffn|hd|vh)(\d+)$")


def _check_option(key, v):
    if key == "stem" and not 1 <= v <= 4:
        raise ValueError(f"stem は 1〜4: {v}")
    if key == "ffn" and not 1 <= v <= 8:
        raise ValueError(f"ffn は 1〜8: {v}")
    if key == "hd" and v not in (16, 32, 64):
        raise ValueError(f"hd は 16・32・64: {v}")
    if key == "vh" and not (8 <= v <= 2048 and v % 8 == 0):
        raise ValueError(f"vh は 8 の倍数で 8〜2048: {v}")


def parse_dl(name):
    """構造の名前を読む。正規の形・既定の値・ValueError の規則は、docstring の先頭を参照。"""
    if not isinstance(name, str):
        raise ValueError(f"構造の名前は文字列: {name!r}")
    parts = name.strip().lower().split("_")
    if len(parts) < 5 or parts[0] != "dl":
        raise ValueError(f"構造の名前は dl_<C>_<trunk>_<policy>_<value>: {name}")
    try:
        channels = int(parts[1])
    except ValueError:
        raise ValueError(f"幅（C）が整数でない: {name}") from None
    if channels % 32 or not 32 <= channels <= 1024:
        raise ValueError(f"幅（C）は 32 の倍数で 32〜1024: {channels}")
    trunk, policy, value = (s.upper() for s in parts[2:5])
    if not trunk or trunk == "0":
        raise ValueError(f"trunk は 1 文字以上のブロック: {name}")
    for label, letters in (("trunk", trunk), ("policy", policy), ("value", value)):
        if letters == "0" and label != "trunk":
            continue
        for ch in letters:
            if ch not in BLOCKS:
                raise ValueError(f"未知のブロック {ch!r}（{label}）: {name}")
    options = dict(DEFAULT_OPTS)
    seen = set()
    for part in parts[5:]:
        m = _OPT_RE.match(part)
        if not m:
            raise ValueError(f"未知のオプション {part!r}: {name}")
        key, v = m.group(1), int(m.group(2))
        if key in seen:
            raise ValueError(f"オプション {key} が 2 回ある: {name}")
        seen.add(key)
        _check_option(key, v)
        options[key] = v
    all_blocks = trunk + policy + value
    if any(ch in all_blocks for ch in "FT") and channels % options["hd"]:
        raise ValueError(f"幅 {channels} は hd {options['hd']} で割り切れない: {name}")
    canonical = f"dl_{channels}_{trunk}_{policy}_{value}"
    for key in OPTION_ORDER:
        if options[key] != DEFAULT_OPTS[key]:
            canonical += f"_{key}{options[key]}"
    return {"canonical": canonical, "channels": channels, "trunk": trunk, "policy": policy,
            "value": value, "options": options}


# ---------- ネットワーク ----------

class DLNetwork(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        c, o = spec["channels"], spec["options"]
        self.channels = c
        stem = []
        cin = IN_CHANNELS
        for _ in range(o["stem"]):
            conv, bn = _conv_bn(cin, c, 3)
            stem += [conv, bn, nn.ReLU()]
            cin = c
        self.stem = nn.Sequential(*stem)
        self.trunk = nn.Sequential(*[BLOCKS[b](c, nn.ReLU, o) for b in spec["trunk"]])
        self.policy_blocks = nn.Sequential(*[BLOCKS[b](c, nn.ReLU, o) for b in spec["policy"].replace("0", "")])
        self.policy_conv = nn.Conv2d(c, POLICY_DIRS, 1)
        self.value_blocks = nn.Sequential(*[BLOCKS[b](c, nn.SiLU, o) for b in spec["value"].replace("0", "")])
        self.value_conv, self.value_bn = _conv_bn(c, VALUE_CH, 1)
        self.value_act = nn.SiLU()
        flat = VALUE_CH * N_TOKENS
        self.value_fc1 = nn.Linear(flat, o["vh"])
        self.value_fc2 = nn.Linear(o["vh"], N_VALUE_BINS)
        self.result_fc1 = nn.Linear(flat, N_VALUE_BINS)
        self.result_fc2 = nn.Linear(N_VALUE_BINS, 1)
        _no_decay(self.value_fc1.bias, self.value_fc2.bias, self.result_fc1.bias, self.result_fc2.bias,
                  self.policy_conv.bias)
        self.register_buffer("bin_centers", (torch.arange(N_VALUE_BINS, dtype=torch.float32) + 0.5) / N_VALUE_BINS)

    def forward(self, x):
        h = self.trunk(self.stem(x))
        p = self.policy_conv(self.policy_blocks(h)).flatten(1)
        v = self.value_blocks(h)
        v = self.value_act(self.value_bn(self.value_conv(v))).flatten(1)
        value_logits = self.value_fc2(self.value_act(self.value_fc1(v)))
        result_logit = self.result_fc2(F.silu(self.result_fc1(v)))
        if self.training:
            return p, value_logits, result_logit
        value = (torch.softmax(value_logits, dim=1) * self.bin_centers).sum(dim=1, keepdim=True)
        return p, value, torch.sigmoid(result_logit)


def build_dl(name):
    """構造の名前から、学習するネットワークを作る。"""
    return DLNetwork(parse_dl(name))


def param_groups(model, weight_decay):
    """重み減衰を掛けるものと掛けないもの（BN・LN の重み、バイアス、相対位置バイアス、位置の埋め込み、スカラーのゲート）。"""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (no_decay if getattr(p, "_no_decay", False) else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def _count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def describe_dl(name):
    """構造の概要。params は学習するパラメータの数、macs は 1 局面の順伝播の掛け算の回数（conv・linear・注意の行列積）。"""
    spec = parse_dl(name)
    c, o = spec["channels"], spec["options"]
    macs = 9 * IN_CHANNELS * c * 81 + (o["stem"] - 1) * 9 * c * c * 81
    for b in spec["trunk"]:
        macs += BLOCKS[b].macs(c, o)
    for b in spec["policy"].replace("0", ""):
        macs += BLOCKS[b].macs(c, o)
    macs += c * POLICY_DIRS * 81
    for b in spec["value"].replace("0", ""):
        macs += BLOCKS[b].macs(c, o)
    flat = VALUE_CH * N_TOKENS
    macs += c * VALUE_CH * 81                         # value_conv（1×1）
    macs += flat * o["vh"] + o["vh"] * N_VALUE_BINS  # value_head
    macs += flat * N_VALUE_BINS + N_VALUE_BINS       # result_head
    return {"canonical": spec["canonical"], "params": _count_params(build_dl(spec["canonical"])), "macs": int(macs)}
