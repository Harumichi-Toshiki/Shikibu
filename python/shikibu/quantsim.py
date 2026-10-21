"""量子化の確認用ツール（設計書 §E4・§S7）。

§E4 の量子化と推論を Python で再現し、float32 の重みとの差を測る。
エンジン（Nuenagon）と同じ規則を書いたもので、確認用の道具であり凍結区画ではない。

使い方：python -m shikibu.quantsim --shk <file.shk> --psv <file.psv> [--n 100000]

- 丸めは「0から遠い方への四捨五入」（C++ の std::lround と同じ）。偶数への丸めは使わない。
"""
import argparse
import sys

import numpy as np
import torch

import shikibu_feat
from shikibu import export, networks
from shikibu.train import compute_loss

PSV_BYTES = 40
FEATURE_VERSIONS = {v: k for k, v in export.FEATURE_SETS.items()}


def rnd(x):
    """0から遠い方への四捨五入（std::lround）。例：2.5 → 3、−2.5 → −3。"""
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    f = np.floor(a)
    return (np.sign(x) * (f + (a - f >= 0.5))).astype(np.int64)


def quantize(tensors):
    """float32 の重み（§F4.2 の名前）を §E4 の整数の重みにする。"""
    q = {"ft_weight": rnd(tensors["ft_weight"].astype(np.float64) * 127).astype(np.int16),
         "ft_bias": rnd(tensors["ft_bias"].astype(np.float64) * 127).astype(np.int16)}
    for k in ("l1", "l2", "out"):
        q[f"{k}_weight"] = np.clip(rnd(tensors[f"{k}_weight"].astype(np.float64) * 64), -127, 127).astype(np.int8)
        q[f"{k}_bias"] = rnd(tensors[f"{k}_bias"].astype(np.float64) * 127 * 64).astype(np.int32)
    return q


def _accumulate(q, feats):
    acc = q["ft_bias"].astype(np.int64)
    if len(feats):
        acc = acc + q["ft_weight"][np.asarray(feats, dtype=np.int64)].astype(np.int64).sum(axis=0)  # 重複は回数ぶん
    if acc.min() < -32768 or acc.max() > 32767:
        raise ValueError("蓄積値が int16 の範囲を超えた")
    return acc


def qforward(q, stm_features, nstm_features):
    """量子化した推論。返り値：(手番側の蓄積値, 非手番側の蓄積値, 最終出力 raw の int)。z = raw / 8128。"""
    a, b = _accumulate(q, stm_features), _accumulate(q, nstm_features)
    h = np.clip(np.concatenate([a, b]), 0, 127)
    for k in ("l1", "l2"):
        s = q[f"{k}_weight"].astype(np.int64) @ h + q[f"{k}_bias"].astype(np.int64)
        h = np.clip(s >> 6, 0, 127)  # 算術右シフト6（負は直後の clamp で0になる）
    raw = int((q["out_weight"].astype(np.int64) @ h)[0] + int(q["out_bias"][0]))
    return a, b, raw


def cp_of(raw, eval_scale):
    """評価値 cp = round(raw × eval_scale / 8128)（倍精度、0から遠い方への四捨五入、クリップなし）。"""
    return int(rnd(raw * float(np.float32(eval_scale)) / 8128.0))


def _psv_records(paths, n):
    recs = []
    for p in paths:
        with open(p, "rb") as f:
            while len(recs) < n:
                rec = f.read(PSV_BYTES)
                if len(rec) < PSV_BYTES:
                    break
                recs.append(rec)
        if len(recs) >= n:
            break
    return recs


def _float_z(m, fs_list, fn_list):
    def bag(lists):
        idx = [i for feats in lists for i in feats]
        off = np.cumsum([0] + [len(feats) for feats in lists[:-1]])
        return torch.tensor(idx, dtype=torch.int64), torch.tensor(off, dtype=torch.int64)

    si, so = bag(fs_list)
    ni, no = bag(fn_list)
    with torch.no_grad():
        return m(si, so, ni, no)


def compare(shk_path, psv_paths, n):
    """先頭 n 局面で float 版と量子化版を比べる（§S7）。"""
    h, t = export.read_shk(shk_path)
    version = FEATURE_VERSIONS[h["feature_set"]]
    q = quantize(t)
    m = networks.build(h["network"], ft_width=h["W"])
    m.set_from_tensors({k: torch.from_numpy(v) for k, v in t.items()})
    m.eval()
    es, lam = h["eval_scale"], h["lam"]

    recs = _psv_records(psv_paths, n)
    if not recs:
        raise ValueError("局面が1つも読めなかった")
    z_f, z_q, cp_q, scores, results = [], [], [], [], []
    chunk = 8192
    for start in range(0, len(recs), chunk):
        fs_list, fn_list = [], []
        for rec in recs[start:start + chunk]:
            sfen, score, _, _, res = shikibu_feat.decode_psv(rec)
            stm = 0 if sfen.split()[1] == "b" else 1
            fs_list.append(shikibu_feat.extract_psv(rec, stm, version))
            fn_list.append(shikibu_feat.extract_psv(rec, 1 - stm, version))
            scores.append(float(score))
            results.append((res + 1) / 2.0)  # 勝 = 1、引分 = 0.5、負 = 0（§S3.3）
            _, _, raw = qforward(q, fs_list[-1], fn_list[-1])
            z_q.append(raw / 8128.0)
            cp_q.append(cp_of(raw, es))
        z_f.extend(_float_z(m, fs_list, fn_list).tolist())

    z_f = np.asarray(z_f, dtype=np.float64)
    z_q = np.asarray(z_q, dtype=np.float64)
    cp_f = z_f * es
    cp_qa = np.asarray(cp_q, dtype=np.float64)
    score_t = torch.tensor(scores, dtype=torch.float32)
    result_t = torch.tensor(results, dtype=torch.float32)
    loss_f = compute_loss(torch.tensor(z_f, dtype=torch.float32), score_t, result_t, es, lam)
    loss_q = compute_loss(torch.tensor(z_q, dtype=torch.float32), score_t, result_t, es, lam)
    return {
        "n": len(recs),
        "mean_abs_z": float(np.mean(np.abs(z_f - z_q))),
        "max_abs_z": float(np.max(np.abs(z_f - z_q))),
        "mean_abs_cp": float(np.mean(np.abs(cp_f - cp_qa))),
        "max_abs_cp": float(np.max(np.abs(cp_f - cp_qa))),
        "loss_float": float(loss_f.item()),
        "loss_quant": float(loss_q.item()),
    }


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m shikibu.quantsim")
    p.add_argument("--shk", required=True, help="確認する .shk")
    p.add_argument("--psv", required=True, nargs="+", help="検証に使う PSV（先頭から読む）")
    p.add_argument("--n", type=int, default=100_000, help="局面数（既定 10 万）")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        r = compare(args.shk, args.psv, args.n)
    except (ValueError, OSError, RuntimeError) as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    print(f"局面数: {r['n']}")
    print(f"z の差（float と量子化）  平均 {r['mean_abs_z']:.6f} / 最大 {r['max_abs_z']:.6f}")
    print(f"評価値の差（cp）          平均 {r['mean_abs_cp']:.2f} / 最大 {r['max_abs_cp']:.2f}")
    print(f"損失  float {r['loss_float']:.6f} / 量子化 {r['loss_quant']:.6f}  （差 {r['loss_quant'] - r['loss_float']:+.6f}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
