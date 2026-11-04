# Shikibu M9.1 の確かめ（新規）：第1層の Triton のカーネル（ft_kernel = "triton"）が、今の "torch" の計算と一致する。
# GPU が無い環境では、解釈モード（TRITON_INTERPRET=1）で動かす。この環境変数は、triton を読む前に設定する。
import os

os.environ["TRITON_INTERPRET"] = "1"

import pytest  # noqa: E402
import torch  # noqa: E402

from shikibu import networks  # noqa: E402

pytest.importorskip("triton")


def _batch(counts_s, counts_n, n_rows, seed):
    g = torch.Generator().manual_seed(seed)

    def side(counts):
        counts = torch.tensor(counts, dtype=torch.long)
        offs = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)[:-1]])
        idx = torch.randint(0, n_rows, (int(counts.sum()),), generator=g)
        return idx, offs

    si, so = side(counts_s)
    ni, no = side(counts_n)
    kings = torch.randint(0, 81, (len(counts_s), 2), generator=g)
    return si, so, ni, no, kings


def _pair(name, seed=0):
    torch.manual_seed(seed)
    m = networks.build(name)
    return m


def _compare(name, counts_s, counts_n, n_rows, seed=1):
    a = _pair(name, seed)
    b = _pair(name, seed)
    b.load_state_dict(a.state_dict())
    b.FT_KERNEL = "triton"
    si, so, ni, no, kings = _batch(counts_s, counts_n, n_rows, seed + 10)
    z1 = a(si, so, ni, no, kings=kings, base=True)
    z2 = b(si, so, ni, no, kings=kings, base=True)
    assert torch.allclose(z1, z2, atol=1e-5, rtol=1e-5), float((z1 - z2).abs().max())
    z1.sum().backward()
    z2.sum().backward()
    assert torch.allclose(a.ft.weight.grad, b.ft.weight.grad, atol=1e-5, rtol=1e-5), \
        float((a.ft.weight.grad - b.ft.weight.grad).abs().max())
    assert torch.allclose(a.ft_bias.grad, b.ft_bias.grad, atol=1e-5, rtol=1e-5)
    return a, b


def test_triton_factorized_forward_and_grads_match_torch():
    # 因数分解つき（HalfKP^）：本体の行と、共通の行（番号 mod P）の両方が効く
    counts = [3, 7, 1, 5, 2, 9]
    _compare("halfkp_32x2-8-8_f", counts, counts[::-1], 125388)


def test_triton_plain_forward_and_grads_match_torch():
    counts = [4, 2, 6, 3]
    _compare("halfkp_32x2-8-8", counts, [5, 1, 2, 4], 125388)


def test_triton_handles_empty_bags_and_wider_rows():
    # 空の局面（特徴が 0 個）は、和が 0 になる。幅 64 も確かめる。
    counts_s = [0, 4, 0, 2]
    counts_n = [3, 0, 1, 0]
    _compare("halfkp_64x2-8-8_f", counts_s, counts_n, 125388, seed=5)


def test_ft_kernel_option_is_refused_when_it_cannot_run():
    from shikibu import ft_triton
    # CPU でも解釈モードなら通る（この環境変数は上でセットしてある）
    ft_triton.check_available(torch.device("cpu"))
    old = os.environ.pop("TRITON_INTERPRET")
    try:
        with pytest.raises(ValueError, match="TRITON_INTERPRET"):
            ft_triton.check_available(torch.device("cpu"))
    finally:
        os.environ["TRITON_INTERPRET"] = old


def _cfg(path, data_dir, out_dir, kernel, steps=3):
    import json
    data = {"train_glob": f"{data_dir}/*.psv", "val_path": "", "cache_dir": "", "max_cached": 3,
            "shuffle_buffer": 0, "num_threads": 2}
    tr = {"batch_size": 64, "lr": 1e-3, "weight_decay": 1e-4, "warmup_steps": 0, "total_steps": steps,
          "max_steps": steps, "min_lr_ratio": 0.0, "lam": 0.75, "eval_scale": 600.0, "scale_sample": 10_000_000,
          "allow_missing_results": False, "seed": 0, "device": "cpu", "torch_threads": 1, "deterministic": False,
          "save_every_steps": 0, "save_every_minutes": 30.0, "val_every_steps": 0, "log_every_steps": 1,
          "progress_every_sec": 1000.0, "ft_kernel": kernel}
    lines = []
    for sec, body in (("data", data), ("model", {"network": "halfkp_32x2-8-8_f", "ft_width": 0}),
                      ("train", tr), ("output", {"out_dir": str(out_dir)})):
        lines.append(f"[{sec}]")
        lines += [f"{k} = {json.dumps(v)}" for k, v in body.items()]
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return str(path)


def test_training_with_triton_kernel_matches_torch_checkpoint(tmp_path):
    # 学習の全体（読み込み・損失・AdamW・クリップ・保存）で、重みが torch の計算と一致する
    from m2_data import make_dataset
    from shikibu import train
    data_dir = str(tmp_path / "d")
    make_dataset(data_dir, n_files=1, per_file=600, seed=12)
    ckpts = {}
    for kernel in ("torch", "triton"):
        out = tmp_path / kernel
        os.makedirs(out)
        cfg = _cfg(tmp_path / f"{kernel}.toml", data_dir, out, kernel)
        assert train.main(["--config", cfg]) == 0
        ckpts[kernel] = torch.load(os.path.join(out, "checkpoint.pt"), map_location="cpu", weights_only=False)["model"]
    for k in ckpts["torch"]:
        a, b = ckpts["torch"][k], ckpts["triton"][k]
        assert torch.allclose(a, b, atol=1e-4, rtol=1e-4), (k, float((a - b).abs().max()))
