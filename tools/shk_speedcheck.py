"""Shikibu の学習の速さの診断（Colab 用）。どこが遅いのかを切り分ける。

使い方（Colab のセルで。リポジトリと wheel の準備が済んだ後）:
    !python /content/drive/MyDrive/<置き場所>/shk_speedcheck.py --repo <REPO> --glob "<学習データ>/*.psv" \
        --network halfkp_256x2-32-32_f --batch 65536 --threads 11

測るもの（M8 の新しい 1 ステップ＝学習と同じ処理で測る）:
  A. 読み込みだけ（Rust の BatchLoader）：毎秒何局面を用意できるか
  B. GPU への転送だけ（train.Prefetcher：固定メモリ＋転送のストリーム）
  C. GPU の計算だけ（同じバッチを繰り返し。前向き（base=True）＋逆伝播＋AdamW＋クリップ）
     C1 AdamW（fused を使わない）/ C2 fused AdamW（CUDA のときだけ。学習の既定）
  D. 全部込み（学習と同じ流れ：次のバッチの読み込みと転送を、今のステップの計算と重ねる）

注意：CPU では C2 と D の転送は従来どおり（fused と非同期転送は CUDA だけ）。速さは CUDA で測ること。
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
    import shikibu_feat
    from shikibu import networks, train

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fused = dev.type == "cuda"
    print("device:", dev, torch.cuda.get_device_name(0) if fused else "")
    paths = sorted(glob.glob(args.glob))[:4]
    B = args.batch
    ver = networks.build(args.network).INPUT_VERSION  # 学習と同じ：ローダーは因数分解なしの版

    def sync():
        if fused:
            torch.cuda.synchronize()

    def loader():
        return shikibu_feat.BatchLoader(paths, B, ver, shuffle_buffer=0, seed=0, num_threads=args.threads)

    def new_model():
        return networks.build(args.network).to(dev)

    def step(m, opt, x):
        """学習の 1 ステップ（Trainer._train_step と同じ計算）。"""
        z = m(x["stm_indices"], x["stm_offsets"], x["nstm_indices"], x["nstm_offsets"], kings=x["kings"], base=True)
        loss = train.compute_loss(z, x["score"], x["result"], 600.0, 0.75)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        train.clip_weights_(m)

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

    # B. 転送だけ（学習と同じ Prefetcher）
    pf = train.Prefetcher(dev)
    sync(); t = time.perf_counter()
    for i in range(args.steps):
        x, ev = pf.send(batches[i % 3])
        pf.ready(x, ev)
    sync(); tb = time.perf_counter() - t
    print(f"B 転送だけ            : {args.steps * B / tb / 1e4:8.1f} 万局面/秒")

    x0, ev0 = pf.send(batches[0])
    x0 = pf.ready(x0, ev0)

    def timed(label, m, opt):
        for _ in range(3):  # ウォームアップ
            step(m, opt, x0)
        sync(); t = time.perf_counter()
        for _ in range(args.steps):
            step(m, opt, x0)
        sync(); tc = time.perf_counter() - t
        print(f"{label:24s}: {args.steps * B / tc / 1e4:8.1f} 万局面/秒  ({tc / args.steps * 1000:.1f} ms/step)")

    # C1 fused を使わない / C2 学習の既定（CUDA なら fused）
    m1 = new_model()
    timed("C1 GPUの計算（fusedなし）", m1, train.build_optimizer(m1, 1e-3, 1e-4, fused=False))
    m2 = new_model()
    timed("C2 GPUの計算（学習の既定）", m2, train.build_optimizer(m2, 1e-3, 1e-4, fused=fused))

    # D. 全部込み：学習の run() と同じ流れ（次のバッチを、今のステップの計算より前に読んで送る）
    m3 = new_model()
    opt3 = train.build_optimizer(m3, 1e-3, 1e-4, fused=fused)
    it = loader()
    for _ in range(3):
        next(it)

    def fetch():
        return pf.send(next(it))

    pend = fetch()
    sync(); t = time.perf_counter(); n = 0
    for _ in range(args.steps):
        cur, pend = pend, fetch()
        x, ev = cur
        x = pf.ready(x, ev)
        step(m3, opt3, x)
        n += int(x["score"].shape[0])
    sync(); td = time.perf_counter() - t
    print(f"D 全部込み            : {n / td / 1e4:8.1f} 万局面/秒")


if __name__ == "__main__":
    main()
