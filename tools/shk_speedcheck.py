"""Shikibu の学習の速さの診断（Colab 用）。どこが遅いのかを切り分ける。

使い方（Colab のセルで。リポジトリと wheel の準備が済んだ後）:
    !python /content/drive/MyDrive/<置き場所>/shk_speedcheck.py --repo <REPO> --glob "<学習データ>/*.psv" \
        --network halfkp_256x2-32-32_f --batch 65536 --threads 11

測るもの（学習と同じ処理で測る。各行は独立。失敗した行は理由を出して、次の行へ進む）:
  A. 読み込みだけ（Rust の BatchLoader）：毎秒何局面を用意できるか
  B. GPU への転送だけ（train.Prefetcher：固定メモリ＋転送のストリーム）
  C. GPU の計算だけ（同じバッチを繰り返し。前向き＋逆伝播＋AdamW＋クリップ）
     C1 AdamW（fused を使わない）/ C2 fused AdamW（CUDA のときだけ。学習の既定）
  D. 全部込み（学習と同じ流れ：次のバッチの読み込みと転送を、今のステップの計算と重ねる）
  E. 第1層のオプションの組み合わせごとの計算（同じバッチを繰り返す）
     E0 既定 / E1 順伝播 sparse / E2 順伝播・勾配 sparse＋遅延型 / E3 遅延型のみ / E4 E2＋bf16 / E5 Triton（ft_kernel）
  P. 時間の内訳（既定 E0 の 1 ステップを torch.profiler で数ステップ測る）：
     順伝播・逆伝播・オプティマイザ・勾配の 0 埋めなどの合計と、時間の長い処理の上位 15 件
  F. 学習の開始前の待ち時間（数分〜数十分かかることがある。最後に回す）
     F1 データの確認 sample（既定）/ F2 probe / F3 シャッフルの充填（既定）/ F4 shuffle_warm

注意：CPU では C2 と D の転送は従来どおり（fused と非同期転送は CUDA だけ）。速さは CUDA で測ること。
CPU では bf16 は無視される（E4 は E2 と同じ速さになる）。E5 は CUDA か、解釈モード（TRITON_INTERPRET=1）でだけ動く。
"""
import argparse
import gc
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
    ap.add_argument("--skip-profile", action="store_true", help="P（時間の内訳）を測らない")
    ap.add_argument("--check-threads", type=int, default=16, help="F1・F2 の読みを並列に行う本数（学習の check_threads と同じ）")
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(args.repo, "python"))

    import torch
    import shikibu_feat
    from shikibu import networks, train

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fused = dev.type == "cuda"
    print("device:", dev, torch.cuda.get_device_name(0) if fused else "", flush=True)
    paths = sorted(glob.glob(args.glob))[:4]
    B = args.batch
    ver = networks.build(args.network).INPUT_VERSION  # 学習と同じ：ローダーは因数分解なしの版

    def sync():
        if fused:
            torch.cuda.synchronize()

    def cleanup():
        gc.collect()
        if fused:
            torch.cuda.empty_cache()

    def row(label, fn):
        """1 行ぶんの測定。失敗したら理由を出し、メモリを片付けて次の行へ進む。"""
        try:
            fn()
        except Exception as e:  # メモリ不足などは、この行だけの失敗にする
            msg = str(e).strip().splitlines()[0][:200] if str(e).strip() else ""
            print(f"{label:34s}: 失敗（{type(e).__name__}: {msg}）", flush=True)
        finally:
            cleanup()

    def loader():
        return shikibu_feat.BatchLoader(paths, B, ver, shuffle_buffer=0, seed=0, num_threads=args.threads)

    def new_model(flags=None):
        m = networks.build(args.network).to(dev)
        flags = flags or {}
        m.FT_FORWARD_SPARSE = flags.get("ft_forward", "dense") == "sparse"
        m.FT_GRAD_SPARSE = flags.get("ft_grad", "dense") == "sparse"
        m.FT_KERNEL = flags.get("ft_kernel", "torch")
        return m

    def step(m, opt, x, fast=False, lazy=False):
        """学習の 1 ステップ（Trainer._train_step と同じ計算）。"""
        with train.autocast_bf16(dev, fast):
            z = m(x["stm_indices"], x["stm_offsets"], x["nstm_indices"], x["nstm_offsets"], kings=x["kings"], base=True)
        z = z.float()
        loss = train.compute_loss(z, x["score"], x["result"], 600.0, 0.75)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        train.clip_weights_(m, ft=not lazy)

    # 最初のバッチ（A・B・C・E・P で同じものを使う）
    it0 = loader()
    batches = [next(it0) for _ in range(3)]  # 立ち上がりを除く
    del it0

    # A. 読み込みだけ
    def row_a():
        it = loader()
        for _ in range(3):
            next(it)
        t = time.perf_counter()
        n = 0
        for _ in range(args.steps):
            b = next(it, None)
            if b is None:
                break
            n += len(b["score"])
        ta = time.perf_counter() - t
        print(f"A 読み込みだけ        : {n / ta / 1e4:8.1f} 万局面/秒", flush=True)
    row("A 読み込みだけ", row_a)

    # B. 転送だけ（学習と同じ Prefetcher）
    def row_b():
        pf = train.Prefetcher(dev)
        sync(); t = time.perf_counter()
        for i in range(args.steps):
            x, ev = pf.send(batches[i % 3])
            pf.ready(x, ev)
        sync(); tb = time.perf_counter() - t
        print(f"B 転送だけ            : {args.steps * B / tb / 1e4:8.1f} 万局面/秒", flush=True)
    row("B 転送だけ", row_b)

    pf0 = train.Prefetcher(dev)  # 同じバッチを、学習と同じ形（torch テンソル）にして使う（CPU でも必要）
    x0, ev0 = pf0.send(batches[0])
    x0 = pf0.ready(x0, ev0)

    def timed(label, flags, optimizer, fast=False, lazy=False):
        m = new_model(flags)
        opt = optimizer(m)
        for _ in range(3):  # ウォームアップ
            step(m, opt, x0, fast, lazy)
        sync(); t = time.perf_counter()
        for _ in range(args.steps):
            step(m, opt, x0, fast, lazy)
        sync(); tc = time.perf_counter() - t
        note = "" if (dev.type == "cuda" or not fast) else "  （CPU では bf16 は無視）"
        print(f"{label:34s}: {args.steps * B / tc / 1e4:8.1f} 万局面/秒  ({tc / args.steps * 1000:.1f} ms/step){note}",
              flush=True)

    # C1 fused を使わない / C2 学習の既定（CUDA なら fused）
    row("C1 GPUの計算（fusedなし）", lambda: timed(
        "C1 GPUの計算（fusedなし）", {}, lambda m: train.build_optimizer(m, 1e-3, 1e-4, fused=False)))
    row("C2 GPUの計算（学習の既定）", lambda: timed(
        "C2 GPUの計算（学習の既定）", {}, lambda m: train.build_optimizer(m, 1e-3, 1e-4, fused=fused)))

    # D. 全部込み：学習の run() と同じ流れ（次のバッチを、今のステップの計算より前に読んで送る）
    def row_d():
        m3 = new_model()
        opt3 = train.build_optimizer(m3, 1e-3, 1e-4, fused=fused)
        pf = train.Prefetcher(dev)
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
        print(f"D 全部込み            : {n / td / 1e4:8.1f} 万局面/秒", flush=True)
    row("D 全部込み", row_d)

    # E. M9 / M9.1：第1層のオプションの組み合わせごと（同じ x0 を繰り返す）
    combos = [
        ("E0 既定（従来の計算）", {}, False, False, "adamw"),
        ("E1 順伝播 sparse", {"ft_forward": "sparse"}, False, False, "adamw"),
        ("E2 順伝播・勾配 sparse＋遅延型", {"ft_forward": "sparse", "ft_grad": "sparse"}, False, True, "lazy_adam"),
        ("E3 遅延型のみ（順伝播 dense）", {}, False, True, "lazy_adam"),
        ("E4 E2 ＋ bf16（fast）", {"ft_forward": "sparse", "ft_grad": "sparse"}, True, True, "lazy_adam"),
        ("E5 Triton（ft_kernel）", {"ft_kernel": "triton"}, False, False, "adamw"),
    ]
    for label, flags, fast, lazy, opt_name in combos:
        def run(label=label, flags=flags, fast=fast, lazy=lazy, opt_name=opt_name):
            ft_opt = "lazy_adam" if opt_name == "lazy_adam" else "adamw"
            timed(label, flags, lambda m: train.build_optimizer(m, 1e-3, 1e-4, fused=fused, ft_optimizer=ft_opt),
                  fast=fast, lazy=lazy)
        row(label, run)

    # P. 時間の内訳（既定 E0 の 1 ステップを torch.profiler で測る）
    def row_p():
        from torch.profiler import ProfilerActivity, profile, record_function
        m = new_model()
        opt = train.build_optimizer(m, 1e-3, 1e-4, fused=fused)

        def step_labeled():
            with record_function("1 順伝播"):
                z = m(x0["stm_indices"], x0["stm_offsets"], x0["nstm_indices"], x0["nstm_offsets"],
                      kings=x0["kings"], base=True)
                loss = train.compute_loss(z.float(), x0["score"], x0["result"], 600.0, 0.75)
            with record_function("2 勾配の初期化"):
                opt.zero_grad(set_to_none=True)
            with record_function("3 逆伝播"):
                loss.backward()
            with record_function("4 オプティマイザ"):
                opt.step()
            with record_function("5 クリップ"):
                train.clip_weights_(m)

        for _ in range(3):
            step_labeled()
        sync()
        n_prof = 5
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if fused else [])
        with profile(activities=acts) as prof:
            for _ in range(n_prof):
                step_labeled()
            sync()
        ka = prof.key_averages()

        def t_self(e):  # 自分の時間（µs）。CUDA では GPU の時間、CPU では CPU の時間
            for name in ("self_device_time_total", "self_cuda_time_total") if fused else ("self_cpu_time_total",):
                v = getattr(e, name, None)
                if v is not None:
                    return float(v)
            return 0.0

        def t_total(e):
            for name in ("device_time_total", "cuda_time_total") if fused else ("cpu_time_total",):
                v = getattr(e, name, None)
                if v is not None:
                    return float(v)
            return 0.0

        kind = "GPU" if fused else "CPU"
        print(f"P 時間の内訳（E0 の 1 ステップ・{kind} の時間）：", flush=True)
        for e in ka:
            if e.key.startswith(tuple(f"{i} " for i in range(1, 6))):
                print(f"   {e.key:22s} {t_total(e) / n_prof / 1000:8.2f} ms/step", flush=True)
        total_self = sum(t_self(e) for e in ka) or 1.0
        top = sorted((e for e in ka if not e.key[:1].isdigit()), key=t_self, reverse=True)[:15]
        print(f"   上位 15 件（自分の{kind}時間）：", flush=True)
        for e in top:
            print(f"   {e.key[:44]:44s} {t_self(e) / n_prof / 1000:8.2f} ms/step  {100 * t_self(e) / total_self:5.1f}%",
                  flush=True)
        zero = sum(t_self(e) for e in ka if any(s in e.key for s in ("zero_", "fill_", "copy_")))
        print(f"   0 埋め・コピー系の合計（zero_/fill_/copy_）：{zero / n_prof / 1000:.2f} ms/step "
              f"（{100 * zero / total_self:.1f}%）", flush=True)
        del m, opt
    if not args.skip_profile:
        row("P 時間の内訳", row_p)

    if args.skip_startup:
        return

    # F. M9：学習の開始前の待ち時間（全件の Drive のファイルで測る。数分〜数十分かかることがある）
    print("ここから開始前の時間を測る（数分〜数十分かかることがある）", flush=True)
    import tempfile
    all_paths = sorted(glob.glob(args.glob))
    tmp = os.path.join(tempfile.gettempdir(), "shk_speedcheck_sample.psv")

    def row_f1():
        t = time.perf_counter()
        files, count = train.sample_scale_positions(all_paths, 10_000_000, 0, tmp, threads=args.check_threads)
        t_pick = time.perf_counter() - t
        t = time.perf_counter()
        shikibu_feat.psv_stats(files)
        shikibu_feat.estimate_eval_scale(files, sample=count, seed=0)
        t_stat = time.perf_counter() - t
        print(f"F1 データの確認（sample・既定・並列 {args.check_threads}）: 抜き出し {t_pick:6.1f} 秒 ＋ 統計・推定 {t_stat:6.1f} 秒（局面 {count:,}）",
              flush=True)
        if files != all_paths and os.path.exists(tmp):
            os.remove(tmp)
    row("F1 データの確認（sample）", row_f1)

    def row_f2():
        t = time.perf_counter()
        files, count, skipped = train.probe_positions(all_paths, 3000, tmp, threads=args.check_threads)
        t_pick = time.perf_counter() - t
        t = time.perf_counter()
        shikibu_feat.psv_stats(files)
        shikibu_feat.estimate_eval_scale(files, sample=count, seed=0)
        t_stat = time.perf_counter() - t
        print(f"F2 データの確認（probe・並列 {args.check_threads}）: 抜き出し {t_pick:6.1f} 秒 ＋ 統計・推定 {t_stat:6.1f} 秒（局面 {count:,}、飛ばし {skipped:,}）",
              flush=True)
        if os.path.exists(tmp):
            os.remove(tmp)
    row("F2 データの確認（probe）", row_f2)

    for label, kw in (
        ("F3 シャッフル（既定：バッファを満たす）", dict(shuffle_buffer=args.shuffle_buffer)),
        ("F4 シャッフル（shuffle_warm：少しで開始）", dict(shuffle_buffer=args.shuffle_buffer, shuffle_warm=args.shuffle_warm)),
    ):
        def run_f(label=label, kw=kw):
            t = time.perf_counter()
            it = shikibu_feat.BatchLoader(all_paths, B, ver, seed=0, num_threads=args.threads,
                                          cache_dir=args.cache_dir or None, **kw)
            next(it)
            print(f"{label}: 最初のバッチまで {time.perf_counter() - t:6.1f} 秒", flush=True)
            del it
        row(label, run_f)


if __name__ == "__main__":
    main()
