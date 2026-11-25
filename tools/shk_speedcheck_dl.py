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

    for key in [s.strip() for s in args.presets.split(",") if s.strip()]:
        name = dl_networks.PRESETS.get(key, key)
        try:
            rate, params = measure_train(name, x, args, device, eval_scale=600.0)
            print(f"T 学習の 1 ステップ [{key}] {name}：{rate:,.0f} 局面/秒（パラメータ {params:,}）", flush=True)
        except Exception as e:
            print(f"T 学習の 1 ステップ [{key}] {name}：失敗（{type(e).__name__}: {e}）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
