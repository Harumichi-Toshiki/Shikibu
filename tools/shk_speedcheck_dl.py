"""DL の学習の速さを、段階ごとに分けて測る（M11）。Colab の L4 で持ち主が測る想定。

使い方（Colab のセル。リポジトリと wheel の準備が済んだ後）:
    !python /content/Shikibu/tools/shk_speedcheck_dl.py --repo /content/Shikibu --glob "<学習データ>/*.psv" \
        --presets A,B,X2 --batch 4096 --threads 11

測るもの（各行は独立。失敗した行は理由を出して次へ進む。単位は局面/秒）：
  L  読み込みだけ：BatchLoader(output="dl") が用意できる局面数（CPU。--threads で作業係の数）
  X  展開だけ：詰めた局面（同じバッチ）→ 55 チャンネル（dl_features.expand）。device は --device
  T  学習の 1 ステップ（プリセットごと）：順伝播・損失・逆伝播・クリップ・AdamW。同じバッチを繰り返す
     amp・compile・channels_last は学習の設定と同じ既定（bf16・compile・channels_last）。--no-compile で外せる

注意：CPU では amp の bf16 は遅くなることがある。速さの判断は CUDA（L4）で行う。compile の初回は時間がかかるので、
最初の数ステップは数えない（--warmup）。
"""
import argparse
import glob
import os
import sys
import time


def _sync(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_loader(paths, batch, threads, seconds_budget):
    import shikibu_feat
    loader = shikibu_feat.BatchLoader(paths, batch, 0, num_threads=threads, output="dl")
    n = 0
    t0 = time.perf_counter()
    for b in loader:
        n += len(b["move"])
        if time.perf_counter() - t0 >= seconds_budget:
            break
    dt = time.perf_counter() - t0
    return n / dt, n, dt


def first_batch(paths, batch, threads):
    import shikibu_feat
    loader = shikibu_feat.BatchLoader(paths, batch, 0, num_threads=threads, output="dl")
    b = next(loader)
    return b


def to_device_batch(b, device):
    import numpy as np
    import torch
    out = {}
    for k in ("board", "hands", "check", "move", "score", "result"):
        out[k] = torch.from_numpy(np.ascontiguousarray(b[k])).to(device)
    out["score"] = out["score"].float()
    out["result"] = out["result"].float()
    return out


def measure_expand(x, device, reps):
    from shikibu import dl_features
    import torch
    for _ in range(3):
        dl_features.expand(x["board"], x["hands"], x["check"])
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        y = dl_features.expand(x["board"], x["hands"], x["check"])
    _sync(device)
    dt = time.perf_counter() - t0
    del y
    return x["move"].shape[0] * reps / dt


def measure_train(name, x, args, device, eval_scale):
    import torch
    from shikibu import dl_networks, train_dl
    model = dl_networks.build_dl(name).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    fwd = torch.compile(model) if args.compile else model
    opt = train_dl.make_optimizer("adamw", model, 4e-4, 1e-3, fused=device.type == "cuda")
    amp = args.amp

    def step():
        feats = _expand_for_train(x, args)
        opt.zero_grad(set_to_none=True)
        with train_dl.autocast(device, amp):
            p, vl, rl = fwd(feats)
        lp, lv, lr_ = train_dl.loss_terms(p, vl, rl, x["move"], x["score"], x["result"], eval_scale, 0.5, 0.03)
        train_dl.total_loss(lp, lv, lr_, 0.5, 1.5).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    for _ in range(args.warmup):
        step()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        step()
    _sync(device)
    dt = time.perf_counter() - t0
    n = x["move"].shape[0] * args.steps
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n / dt, params


def _expand_for_train(x, args):
    import torch
    from shikibu import dl_features
    feats = dl_features.expand(x["board"], x["hands"], x["check"])
    if args.channels_last:
        feats = feats.contiguous(memory_format=torch.channels_last)
    return feats


def measure_shift(batch, channels, device, reps):
    """shift9 の旧（depthwise conv・one-hot の重み）と新（パディングと切り出し）の順伝播の速さ（局面/秒）。"""
    import torch
    import torch.nn.functional as F
    from shikibu import dl_networks as N
    x = torch.randn(batch, channels, 9, 9, device=device)
    w = N._shift_kernel(channels).to(device)
    funcs = [("旧 conv", lambda: F.conv2d(x, w, padding=1, groups=channels)),
             ("新 切り出し", lambda: N.shift9(x))]
    out = []
    for label, fn in funcs:
        for _ in range(3):
            fn()
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        _sync(device)
        out.append((label, batch * reps / (time.perf_counter() - t0)))
    return out


def measure_blocks(name, batch, device, reps):
    """構造の中の各ブロック（順伝播＋逆伝播）の 1 呼び出しの時間（ms）。C・オプションは構造の名前のとおり。"""
    import torch
    from shikibu import dl_networks as N
    spec = N.parse_dl(name)
    c, o = spec["channels"], spec["options"]
    out = []
    for letter in dict.fromkeys(spec["trunk"] + spec["policy"].replace("0", "") + spec["value"].replace("0", "")):
        blk = N.BLOCKS[letter](c, torch.nn.ReLU, o).to(device)
        x = torch.randn(batch, c, 9, 9, device=device, requires_grad=True)
        for _ in range(2):
            blk(x).sum().backward()
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(reps):
            blk(x).sum().backward()
        _sync(device)
        out.append((letter, (time.perf_counter() - t0) / reps * 1000))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Shikibu のリポジトリ（python/ を import する）")
    ap.add_argument("--glob", required=True, help="PSV の学習データ（最初の数ファイルを使う）")
    ap.add_argument("--presets", default="A", help="測るプリセット（A,B,X2 のカンマ区切り。名前を直接書いてもよい）")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, default=20, help="T で数えるステップ数")
    ap.add_argument("--warmup", type=int, default=3, help="T で数えない最初のステップ数（compile の初回を除く）")
    ap.add_argument("--loader-seconds", type=float, default=20.0, help="L を測る秒数")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--no-channels-last", action="store_true")
    ap.add_argument("--networks", default="", help="測る構造の名前（カンマ区切り）。指定すると --presets より優先（M11.1）")
    ap.add_argument("--blocks", action="store_true", help="ブロックごとの順伝播＋逆伝播の時間を測る（M11.1）")
    ap.add_argument("--shift", action="store_true", help="shift9 の旧（depthwise conv）と新（切り出し）を比べる（M11.1）")
    args = ap.parse_args(argv)
    args.compile = not args.no_compile
    args.channels_last = not args.no_channels_last

    sys.path.insert(0, os.path.join(args.repo, "python"))
    import torch
    from shikibu import dl_networks
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if args.device == "auto" else args.device))
    print(f"device: {device}  batch: {args.batch}  threads: {args.threads}  amp: {args.amp}  "
          f"compile: {args.compile}  channels_last: {args.channels_last}", flush=True)
    paths = sorted(glob.glob(args.glob))[:4]
    if not paths:
        print("学習データが無い（--glob）", flush=True)
        return 1

    try:
        rate, n, dt = measure_loader(paths, args.batch, args.threads, args.loader_seconds)
        print(f"L 読み込みだけ：{rate:,.0f} 局面/秒（{n:,} 局面、{dt:.1f} 秒）", flush=True)
        b = first_batch(paths, args.batch, args.threads)
        x = to_device_batch(b, device)
    except Exception as e:  # 失敗した行は理由を出して次へ
        print(f"L 読み込みだけ：失敗（{e}）", flush=True)
        return 1

    try:
        rate = measure_expand(x, device, reps=20)
        print(f"X 展開だけ：{rate:,.0f} 局面/秒", flush=True)
    except Exception as e:
        print(f"X 展開だけ：失敗（{e}）", flush=True)

    if args.shift:
        try:
            for label, rate in measure_shift(args.batch, 192, device, reps=20):
                print(f"S shift9 [{label}]（B={args.batch}, C=192）：{rate:,.0f} 局面/秒（順伝播のみ）", flush=True)
        except Exception as e:
            print(f"S shift9：失敗（{e}）", flush=True)

    names = [s.strip() for s in args.networks.split(",") if s.strip()] or [
        s.strip() for s in args.presets.split(",") if s.strip()]
    if args.blocks:
        for name in names:
            try:
                for letter, ms in measure_blocks(name, args.batch, device, reps=5):
                    print(f"B 1 ブロック [{letter}] in {name}：{ms:.2f} ms/呼び出し（順伝播＋逆伝播、B={args.batch}）", flush=True)
            except Exception as e:
                print(f"B ブロック：失敗（{name}：{e}）", flush=True)

    for key in names:
        name = dl_networks.PRESETS.get(key, key)
        try:
            rate, params = measure_train(name, x, args, device, eval_scale=600.0)
            print(f"T 学習の 1 ステップ [{key}] {name}：{rate:,.0f} 局面/秒（パラメータ {params:,}）", flush=True)
        except Exception as e:
            print(f"T 学習の 1 ステップ [{key}] {name}：失敗（{type(e).__name__}: {e}）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
