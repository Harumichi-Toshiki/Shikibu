"""Shikibu の学習の速さの診断（Colab 用）。どこが遅いのかを切り分ける。

使い方（Colab のセルで。リポジトリと wheel の準備が済んだ後）:
    !python /content/drive/MyDrive/<置き場所>/shk_speedcheck.py --repo <REPO> --glob "<学習データ>/*.psv" \
        --network halfkp_256x2-32-32_f --batch 65536 --threads 11

測るもの:
  A. 読み込みだけ（Rust の BatchLoader）：毎秒何局面を用意できるか
  B. GPU への転送だけ（to_device）
  C. GPU の計算だけ（同じバッチを繰り返し。前向き＋逆伝播＋AdamW＋クリップ）
     C1 今のまま / C2 fused AdamW / C3 第1層を「局面数 × 特徴数」の表で計算（特徴の数が一定の構造だけ）
  D. 全部込み（今の学習と同じ流れ）
"""
import argparse
import glob
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--glob", required=True)
    ap.add_argument("--network", default="halfkp_256x2-32-32_f")
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--threads", type=int, default=11)
    ap.add_argument("--steps", type=int, default=40)
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(args.repo, "python"))

    import torch
    import torch.nn.functional as F
    import shikibu_feat
    from shikibu import networks, train

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")
    paths = sorted(glob.glob(args.glob))[:4]
    model = networks.build(args.network).to(dev)
    ver = model.FEATURE_VERSION
    B = args.batch

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    def loader():
        return shikibu_feat.BatchLoader(paths, B, ver, shuffle_buffer=0, seed=0, num_threads=args.threads)

    # A. 読み込みだけ
    it = loader()
    batches = [next(it) for _ in range(3)]  # 立ち上がりを除く
    t = time.perf_counter()
    n = 0
    for _ in range(args.steps):
        b = next(it, None)
        if b is None:
            break
        n += len(b["score"])
    ta = time.perf_counter() - t
    print(f"A 読み込みだけ        : {n / ta / 1e4:8.1f} 万局面/秒")
    del it

    # B. 転送だけ
    sync(); t = time.perf_counter()
    for i in range(args.steps):
        x = train.to_device(batches[i % 3], dev)
    sync(); tb = time.perf_counter() - t
    print(f"B 転送だけ            : {args.steps * B / tb / 1e4:8.1f} 万局面/秒")

    x = train.to_device(batches[0], dev)

    def step_loop(m, opt, fwd, label):
        for _ in range(3):  # ウォームアップ
            z = fwd(m, x)
            loss = train.compute_loss(z, x["score"], x["result"], 600.0, 0.75)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); train.clip_weights_(m)
        sync(); t = time.perf_counter()
        for _ in range(args.steps):
            z = fwd(m, x)
            loss = train.compute_loss(z, x["score"], x["result"], 600.0, 0.75)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); train.clip_weights_(m)
        sync(); tc = time.perf_counter() - t
        print(f"{label:22s}: {args.steps * B / tc / 1e4:8.1f} 万局面/秒  ({tc / args.steps * 1000:.1f} ms/step)")

    def fwd_bag(m, x):
        return m(x["stm_indices"], x["stm_offsets"], x["nstm_indices"], x["nstm_offsets"], kings=x["kings"])

    # C1 今のまま
    opt = train.build_optimizer(model, 1e-3, 1e-4)
    step_loop(model, opt, fwd_bag, "C1 GPUの計算（今）")

    # C2 fused AdamW
    if dev.type == "cuda":
        try:
            m2 = networks.build(args.network).to(dev)
            decay = m2.decay_parameters(); ids = {id(p) for p in decay}
            opt2 = torch.optim.AdamW([{"params": decay, "weight_decay": 1e-4},
                                      {"params": [p for p in m2.parameters() if id(p) not in ids], "weight_decay": 0.0}],
                                     lr=1e-3, fused=True)
            step_loop(m2, opt2, fwd_bag, "C2 fused AdamW")
        except Exception as e:
            print("C2 fused AdamW         : 失敗", e)

    # C3 第1層を表で計算（特徴の数が全局面で同じときだけ）
    so, si = x["stm_offsets"], x["stm_indices"]
    k = si.numel() // so.numel()
    same = (k * so.numel() == si.numel()) and bool((so == torch.arange(so.numel(), device=dev) * k).all())
    if same and model.STACKS == 1 and not model.PSQT:
        sidx = si.view(-1, k).int(); nidx = x["nstm_indices"].view(-1, k).int()

        def fwd_tab(m, x):
            w = m.ft.weight
            a = F.embedding(sidx, w).sum(1) + m.ft_bias
            b = F.embedding(nidx, w).sum(1) + m.ft_bias
            h = torch.clamp(torch.cat([a, b], 1), 0, 1)
            h = torch.clamp(m.l1(h), 0, 1)
            h = torch.clamp(m.l2(h), 0, 1)
            return m.out(h).squeeze(1)
        m3 = networks.build(args.network).to(dev)
        step_loop(m3, train.build_optimizer(m3, 1e-3, 1e-4), fwd_tab, f"C3 表で計算（{k}個）")
    else:
        print("C3 表で計算           : この構造は特徴の数が一定でないので省略")

    # D. 全部込み
    it = loader()
    for _ in range(3):
        next(it)
    sync(); t = time.perf_counter(); n = 0
    for _ in range(args.steps):
        b = next(it, None)
        if b is None:
            break
        xx = train.to_device(b, dev)
        z = fwd_bag(model, xx)
        loss = train.compute_loss(z, xx["score"], xx["result"], 600.0, 0.75)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); train.clip_weights_(model)
        n += len(b["score"])
    sync(); td = time.perf_counter() - t
    print(f"D 全部込み            : {n / td / 1e4:8.1f} 万局面/秒")


if __name__ == "__main__":
    main()
