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
  E. M9 の組み合わせごとの計算（第1層の順伝播・勾配・遅延型 Adam・bf16。C と同じく同じバッチで測る）
  F. 学習の開始前の待ち時間（データの確認 sample・probe、シャッフルの充填 buffer・warm）

注意：CPU では C2 と D の転送は従来どおり（fused と非同期転送は CUDA だけ）。速さは CUDA で測ること。
CPU では bf16 は無視される（E4 は E2 と同じ速さになる）。
F は、データ（Drive 上の *.psv）の全件の大きさに比例するものを含むので、1 回だけ回す。
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
    ap.add_argument("--shuffle-buffer", type=int, default=20_000_000, help="F3・F4 のシャッフルバッファ（局面数）")
    ap.add_argument("--shuffle-warm", type=int, default=2_000_000, help="F4 の立ち上がりの件数")
    ap.add_argument("--cache-dir", default="", help="F3・F4 の先読み先（学習の cache_dir と同じ場所。空なら使わない）")
    ap.add_argument("--skip-startup", action="store_true", help="F（開始前の待ち時間）を測らない")
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

    # E. M9：第1層の疎な計算・遅延型 Adam・bf16（同じ x0 を繰り返す。C と同じ測り方）
    def step_amp(m, opt, x, fast, lazy):
        with train.autocast_bf16(dev, fast):
            z = m(x["stm_indices"], x["stm_offsets"], x["nstm_indices"], x["nstm_offsets"], kings=x["kings"], base=True)
        z = z.float()
        loss = train.compute_loss(z, x["score"], x["result"], 600.0, 0.75)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        train.clip_weights_(m, ft=not lazy)

    combos = [
        ("E0 既定（従来の計算）", dict(ft_forward="dense", ft_grad="dense", ft_optimizer="adamw", fast=False)),
        ("E1 順伝播 sparse", dict(ft_forward="sparse", ft_grad="dense", ft_optimizer="adamw", fast=False)),
        ("E2 順伝播・勾配 sparse＋遅延型", dict(ft_forward="sparse", ft_grad="sparse", ft_optimizer="lazy_adam", fast=False)),
        ("E3 遅延型のみ（順伝播 dense）", dict(ft_forward="dense", ft_grad="dense", ft_optimizer="lazy_adam", fast=False)),
        ("E4 E2 ＋ bf16（fast）", dict(ft_forward="sparse", ft_grad="sparse", ft_optimizer="lazy_adam", fast=True)),
    ]
    for label, f in combos:
        m = new_model()
        m.FT_FORWARD_SPARSE = f["ft_forward"] == "sparse"
        m.FT_GRAD_SPARSE = f["ft_grad"] == "sparse"
        opt = train.build_optimizer(m, 1e-3, 1e-4, fused=fused, ft_optimizer=f["ft_optimizer"])
        lazy = f["ft_optimizer"] == "lazy_adam"
        for _ in range(3):
            step_amp(m, opt, x0, f["fast"], lazy)
        sync(); t = time.perf_counter()
        for _ in range(args.steps):
            step_amp(m, opt, x0, f["fast"], lazy)
        sync(); tc = time.perf_counter() - t
        note = "" if (dev.type == "cuda" or not f["fast"]) else "  （CPU では bf16 は無視）"
        print(f"{label:30s}: {args.steps * B / tc / 1e4:8.1f} 万局面/秒  ({tc / args.steps * 1000:.1f} ms/step){note}")

    if args.skip_startup:
        return

    # F. M9：学習の開始前の待ち時間（全件の Drive のファイルで測る）
    import tempfile
    all_paths = sorted(glob.glob(args.glob))
    tmp = os.path.join(tempfile.gettempdir(), "shk_speedcheck_sample.psv")
    t = time.perf_counter()
    files, count = train.sample_scale_positions(all_paths, 10_000_000, 0, tmp)
    t_pick = time.perf_counter() - t
    t = time.perf_counter()
    shikibu_feat.psv_stats(files)
    shikibu_feat.estimate_eval_scale(files, sample=count, seed=0)
    t_stat = time.perf_counter() - t
    print(f"F1 データの確認（sample・既定）: 抜き出し {t_pick:6.1f} 秒 ＋ 統計・推定 {t_stat:6.1f} 秒（局面 {count:,}）")
    if files != all_paths and os.path.exists(tmp):
        os.remove(tmp)

    t = time.perf_counter()
    files, count, skipped = train.probe_positions(all_paths, 3000, tmp)
    t_pick = time.perf_counter() - t
    t = time.perf_counter()
    shikibu_feat.psv_stats(files)
    shikibu_feat.estimate_eval_scale(files, sample=count, seed=0)
    t_stat = time.perf_counter() - t
    print(f"F2 データの確認（probe）      : 抜き出し {t_pick:6.1f} 秒 ＋ 統計・推定 {t_stat:6.1f} 秒（局面 {count:,}、飛ばし {skipped:,}）")
    if os.path.exists(tmp):
        os.remove(tmp)

    for label, kw in (
        ("F3 シャッフル（既定：バッファを満たす）", dict(shuffle_buffer=args.shuffle_buffer)),
        ("F4 シャッフル（shuffle_warm：少しで開始）", dict(shuffle_buffer=args.shuffle_buffer, shuffle_warm=args.shuffle_warm)),
    ):
        t = time.perf_counter()
        it = shikibu_feat.BatchLoader(all_paths, B, ver, seed=0, num_threads=args.threads,
                                      cache_dir=args.cache_dir or None, **kw)
        next(it)
        print(f"{label}: 最初のバッチまで {time.perf_counter() - t:6.1f} 秒")
        del it


if __name__ == "__main__":
    main()
