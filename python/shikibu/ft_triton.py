"""第1層（特徴量の変換）の Triton のカーネル（M9.1）。設定 train.ft_kernel = "triton" のときだけ使う（既定は "torch"）。

順伝播（局面ごと、手番側・相手側の 2 本）：
    out[b] = bias + Σ_{有効な特徴 r} W[r]            （因数分解つき：+ Σ W[n + r mod P]）
  W_eff も、大きな中間データ（バッチ × 特徴 × 幅）も作らない。
逆伝播：
    gW[r] += gout[b]（有効な特徴ごとに、fp32 の原子的な加算）。因数分解の行 n + r mod P にも同じように足す。
    gbias = Σ_b gout_stm[b] + Σ_b gout_nstm[b]
勾配は今と同じ密な形（重みと同じ大きさ）で返すので、オプティマイザ（AdamW・fused）は今のまま使える。

Triton は必要になったときだけ import する（既定の経路は triton が無くても動く）。
GPU が無いときは、解釈モード（環境変数 TRITON_INTERPRET=1）でだけ動く。
"""
import os

import torch

_TRITON = None


def load_triton():
    """Triton を読む。使えなければ ValueError（日本語）。CPU では解釈モードが必要。"""
    global _TRITON
    if _TRITON is None:
        try:
            import triton  # noqa: F401
            import triton.language  # noqa: F401
        except Exception as e:  # pragma: no cover - 環境による
            raise ValueError(f"ft_kernel = \"triton\" には triton が必要（{e}）") from None
        _TRITON = triton
    return _TRITON


def check_available(device):
    """学習の前に確かめる。CUDA では Triton があればよい。CPU では TRITON_INTERPRET=1 が必要。"""
    load_triton()
    if device.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
        raise ValueError('ft_kernel = "triton" は CUDA か、解釈モード（環境変数 TRITON_INTERPRET=1）が必要')


def _kernels():
    """カーネルを 1 回だけ定義する（triton を読んでから）。"""
    if hasattr(_kernels, "cache"):
        return _kernels.cache
    triton = load_triton()
    tl = triton.language

    @triton.jit
    def _fwd(W, BIAS, IDX, STARTS, ENDS, OUT, H, n_rows, P,
             FACT: tl.constexpr, BLOCK_H: tl.constexpr):
        b = tl.program_id(0)
        hs = tl.arange(0, BLOCK_H)
        mask = hs < H
        acc = tl.load(BIAS + hs, mask=mask, other=0.0)
        start = tl.load(STARTS + b)
        end = tl.load(ENDS + b)
        for j in range(start, end):
            r = tl.load(IDX + j)
            acc += tl.load(W + r * H + hs, mask=mask, other=0.0)
            if FACT:
                acc += tl.load(W + (n_rows + r % P) * H + hs, mask=mask, other=0.0)
        tl.store(OUT + b * H + hs, acc, mask=mask)

    @triton.jit
    def _bwd(GOUT, IDX, STARTS, ENDS, GW, H, n_rows, P,
             FACT: tl.constexpr, BLOCK_H: tl.constexpr):
        b = tl.program_id(0)
        hs = tl.arange(0, BLOCK_H)
        mask = hs < H
        g = tl.load(GOUT + b * H + hs, mask=mask, other=0.0)
        start = tl.load(STARTS + b)
        end = tl.load(ENDS + b)
        for j in range(start, end):
            r = tl.load(IDX + j)
            tl.atomic_add(GW + r * H + hs, g, mask=mask)
            if FACT:
                tl.atomic_add(GW + (n_rows + r % P) * H + hs, g, mask=mask)

    _kernels.cache = (_fwd, _bwd, triton)
    return _kernels.cache


def _block_h(h):
    return max(16, 1 << (int(h) - 1).bit_length())


class FtBagPair(torch.autograd.Function):
    """手番側・相手側の第1層の和（バイアス込み）を、2 本まとめて計算する。"""

    @staticmethod
    def forward(ctx, weight, bias, si, so, ni, no, n_rows, P, fact):
        fwd, _, triton = _kernels()
        W = weight.contiguous()
        H = W.shape[1]
        bs = bias.contiguous()
        outs = []
        for idx, off in ((si, so), (ni, no)):
            idx = idx.contiguous()
            off = off.contiguous()
            ends = torch.cat([off[1:], torch.tensor([idx.numel()], dtype=off.dtype, device=off.device)])
            out = torch.empty((off.numel(), H), dtype=torch.float32, device=W.device)
            fwd[(off.numel(),)](W, bs, idx, off, ends, out, H, n_rows, P,
                                FACT=fact, BLOCK_H=_block_h(H))
            outs.append(out)
        ctx.save_for_backward(si, so, ni, no)
        ctx.meta = (n_rows, P, fact, H, W.shape[0])
        return outs[0], outs[1]

    @staticmethod
    def backward(ctx, ga, gb):
        _, bwd, triton = _kernels()
        si, so, ni, no = ctx.saved_tensors
        n_rows, P, fact, H, R = ctx.meta
        gW = torch.zeros((R, H), dtype=torch.float32, device=si.device)
        for idx, off, g in ((si, so, ga), (ni, no, gb)):
            idx = idx.contiguous()
            off = off.contiguous()
            ends = torch.cat([off[1:], torch.tensor([idx.numel()], dtype=off.dtype, device=off.device)])
            g = g.contiguous()
            bwd[(off.numel(),)](g, idx, off, ends, gW, H, n_rows, P,
                                FACT=fact, BLOCK_H=_block_h(H))
        gbias = ga.sum(0) + gb.sum(0)
        return gW, gbias, None, None, None, None, None, None, None


def ft_pair(net, si, so, ni, no, n_rows, P, fact):
    """net.ft の第1層（バイアス込み）を、Triton のカーネルで、手番側・相手側の 2 本まとめて計算する。"""
    return FtBagPair.apply(net.ft.weight, net.ft_bias, si, so, ni, no, n_rows, P, fact)
