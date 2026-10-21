"""Shikibu の学習を Colab で裏で走らせ、進み具合を表示する道具。

使い方（Colab のセル）：
    import sys; sys.path.insert(0, "/content/drive/MyDrive/shikibu/tools")
    import shk_monitor as mon
    mon.start(CONFIG)          # 学習を裏で開始（checkpoint があれば自動で --resume）
    mon.watch(CONFIG)          # 進み具合を表示（■停止ボタンで表示だけ止まる。学習は続く）
    mon.stop()                 # 学習を止める（最後の保存より後の分は失われる）

学習は裏のプロセスで動くので、表示のセルを止めても、ブラウザを再読み込みしても学習は続く。
ただし Colab のランタイム自体が切れると学習も止まる（そのときは start をもう一度実行すれば続きから再開する）。
"""
import glob
import json
import os
import subprocess
import sys
import time
import tomllib

REPO = "/content/Shikibu"
STDOUT = "/content/shk_train_stdout.txt"
_proc = None
_manual_stop = False


def _cfg(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def _out_dir(cfg):
    return cfg["output"]["out_dir"]


def start(config_path, repo=REPO):
    """学習を裏で開始する。すでに動いていれば何もしない。"""
    global _proc, _manual_stop
    if _proc is not None and _proc.poll() is None:
        print("学習はすでに動いています（PID %d）。" % _proc.pid)
        return
    cfg = _cfg(config_path)
    ck = os.path.join(_out_dir(cfg), "checkpoint.pt")
    cmd = [sys.executable, "-u", "-m", "shikibu.train", "--config", config_path]
    if os.path.exists(ck):
        cmd.append("--resume")
    pp = os.path.join(repo, "python")
    env = dict(os.environ, PYTHONPATH=pp + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""))
    f = open(STDOUT, "a", encoding="utf-8")
    f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
    f.flush()
    _proc = subprocess.Popen(cmd, cwd=repo, env=env, stdout=f, stderr=subprocess.STDOUT)
    _manual_stop = False
    print(("再開" if "--resume" in cmd else "新規") + f"で学習を開始しました（PID {_proc.pid}）。")


def stop():
    """学習を止める。"""
    global _proc, _manual_stop
    if _proc is None or _proc.poll() is not None:
        print("動いている学習はありません。")
        return
    _manual_stop = True
    _proc.terminate()
    try:
        _proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        _proc.kill()
    print("学習を止めました。次に start を実行すると、最後に保存した所から再開します。")


def read_log(out_dir):
    p = os.path.join(out_dir, "log.jsonl")
    rows = []
    if not os.path.exists(p):
        return rows
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass        # 書きかけの最後の行は読み飛ばす
    return rows


def _fmt_sec(s):
    if s is None or s != s or s < 0:
        return "--"
    s = int(s)
    h, m = divmod(s // 60, 60)
    return f"{h}時間{m:02d}分" if h else f"{m}分{s % 60:02d}秒"


def _bar(frac, width=40):
    frac = max(0.0, min(1.0, frac))
    n = int(round(frac * width))
    return "█" * n + "░" * (width - n)


def _cache_info(cfg):
    d = cfg["data"].get("cache_dir") or ""
    if not d or not os.path.isdir(d):
        return None
    files = [(os.path.basename(p), os.path.getsize(p)) for p in glob.glob(os.path.join(d, "*"))]
    return files


def status(config_path, now=None):
    """進み具合をまとめた dict を返す（表示とは別に、数字だけ欲しいとき用）。"""
    cfg = _cfg(config_path)
    T = cfg["train"]
    rows = read_log(_out_dir(cfg))
    target = int(T["max_steps"])
    total = int(T["total_steps"])
    batch = int(T["batch_size"])
    trains = [r for r in rows if r.get("type") == "train"]
    vals = [r for r in rows if r.get("type") == "val"]
    saves = [r for r in rows if r.get("type") == "save"]
    starts = [r for r in rows if r.get("type") == "start"]
    ends = [r for r in rows if r.get("type") == "end"]
    st = {"target_steps": target, "total_steps": total, "batch": batch, "trains": trains, "vals": vals,
          "eval_scale": starts[-1].get("eval_scale") if starts else None}
    step = trains[-1]["step"] if trains else 0
    st["step"] = step
    st["frac"] = step / target if target else 0.0
    # 速度：直近の train 行から（局面/秒）
    recent = [r.get("positions_per_sec") for r in trains[-20:] if r.get("positions_per_sec")]
    pps = sum(recent) / len(recent) if recent else None
    st["pps"] = pps
    st["eta"] = (target - step) * batch / pps if pps else None
    st["positions"] = step * batch
    st["epoch"] = trains[-1].get("epoch") if trains else None
    st["last_save"] = saves[-1]["step"] if saves else None
    alive = _proc is not None and _proc.poll() is None
    st["alive"] = alive
    if ends and (not trains or ends[-1].get("step", -1) >= step) and not alive:
        st["phase"] = "完了" if step >= target else "停止"
    elif not rows or not starts:
        st["phase"] = "準備中：データの確認・eval_scale の推定"
    elif not trains or (starts and trains and rows.index(starts[-1]) > rows.index(trains[-1])):
        info = _cache_info(cfg)
        part = [f for f in (info or []) if f[0].endswith(".part")]
        if part:
            st["phase"] = f"準備中：Drive から先読み中（{part[0][1] / 1e9:.1f} GB）"
        else:
            st["phase"] = "準備中：シャッフルバッファを充填中"
    else:
        st["phase"] = "学習中"
    # M4 以降の学習器は status.json に今の段階を書く。あればそれを優先する
    sp = os.path.join(_out_dir(cfg), "status.json")
    if os.path.exists(sp):
        try:
            with open(sp, encoding="utf-8") as f:
                sj = json.load(f)
            names = {"data_check": "準備中：データの確認（勝敗の記録・局面の抜き出し）",
                     "eval_scale": "準備中：eval_scale を推定中",
                     "prefetch": "準備中：Drive から先読み中",
                     "buffer_fill": "準備中：シャッフルバッファを充填中",
                     "train": "学習中", "done": "完了", "error": "エラーで終了"}
            ph = names.get(sj.get("phase"), sj.get("phase"))
            if sj.get("phase") == "prefetch":
                info = _cache_info(cfg)
                part = [f for f in (info or []) if f[0].endswith(".part")]
                if part:
                    ph += f"（{part[0][1] / 1e9:.1f} GB）"
            st["phase"] = ph
            if st["eta"] is None and sj.get("eta_sec") is not None:
                st["eta"] = sj["eta_sec"]
        except (json.JSONDecodeError, OSError):
            pass
    if _proc is not None and not alive:
        if _manual_stop:
            st["phase"] = "停止（手動）"
        elif _proc.returncode == 0:
            st["phase"] = "完了" if step >= target else "停止（max_steps に到達）"
        else:
            st["phase"] = f"異常終了（終了コード {_proc.returncode}）"
    elif _proc is None and st["phase"] not in ("完了", "停止", "エラーで終了"):
        # このノートブックから起動していない（ランタイムの再起動後など）。ログの更新時刻で判断する
        p = sp if os.path.exists(sp) else os.path.join(_out_dir(cfg), "log.jsonl")
        if os.path.exists(p) and st["phase"] not in ("完了", "エラーで終了"):
            idle = time.time() - os.path.getmtime(p)
            if idle > 600:
                st["phase"] += f"？（ログが {int(idle // 60)} 分更新されていません。止まっている可能性があります）"
    return st


def render(config_path, fig_path=None):
    """進み具合の文字列を返し、fig_path を渡せば学習曲線の画像も保存する。"""
    st = status(config_path)
    lines = [
        f"状態：{st['phase']}",
        f"進捗：{_bar(st['frac'])} {100 * st['frac']:5.1f}%  （{st['step']:,} / {st['target_steps']:,} ステップ）",
        f"学習した局面：{st['positions'] / 1e8:.2f} 億局面   周回：{st['epoch'] if st['epoch'] is not None else '--'}",
        f"速度：{st['pps'] / 1e4:.1f} 万局面/秒" if st["pps"] else "速度：--",
        f"残り時間：{_fmt_sec(st['eta'])}   最後の保存：{st['last_save'] if st['last_save'] is not None else '--'} ステップ",
    ]
    if st["trains"]:
        lines.append(f"直近の損失：{st['trains'][-1]['loss']:.4f}   学習率：{st['trains'][-1].get('lr', 0):.2e}")
    if st["vals"]:
        v = st["vals"][-1]
        lines.append(f"検証：損失 {v['val_loss']:.4f} / MSE {v['val_mse']:.5f}（{v['step']:,} ステップ時点）")
    if st["eval_scale"]:
        lines.append(f"eval_scale：{st['eval_scale']:.1f}")
    if fig_path and st["trains"]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
        xs = [r["step"] for r in st["trains"]]
        ys = [r["loss"] for r in st["trains"]]
        k = max(1, len(ys) // 200)
        sm = [sum(ys[max(0, i - k):i + 1]) / len(ys[max(0, i - k):i + 1]) for i in range(len(ys))]
        ax[0].plot(xs, ys, alpha=0.25, lw=0.8, label="学習の損失")
        ax[0].plot(xs, sm, lw=1.6, label="学習の損失（平滑化）")
        if st["vals"]:
            ax[0].plot([v["step"] for v in st["vals"]], [v["val_loss"] for v in st["vals"]], "o-", ms=3, label="検証の損失")
        ax[0].set_xlabel("ステップ"); ax[0].set_title("損失"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
        pp = [r.get("positions_per_sec", 0) / 1e4 for r in st["trains"]]
        ax[1].plot(xs, pp, lw=1.0)
        ax[1].set_xlabel("ステップ"); ax[1].set_title("速度（万局面/秒）"); ax[1].grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(fig_path, dpi=90); plt.close(fig)
    return "\n".join(lines), st


def tail_stdout(n=15):
    if not os.path.exists(STDOUT):
        return ""
    with open(STDOUT, encoding="utf-8", errors="replace") as f:
        return "".join(f.readlines()[-n:])


def watch(config_path, interval=15):
    """進み具合を interval 秒ごとに更新して表示する。■停止ボタンで表示だけ止まる（学習は続く）。"""
    from IPython.display import Image, clear_output, display
    _japanese_font()
    fig = "/content/shk_progress.png"
    try:
        while True:
            text, st = render(config_path, fig)
            clear_output(wait=True)
            print(text)
            if st["trains"] and os.path.exists(fig):
                display(Image(fig))
            if st["phase"].startswith("異常終了"):
                print("\n--- 学習プロセスの最後の出力 ---\n" + tail_stdout())
                break
            if st["phase"] in ("完了", "エラーで終了") or st["phase"].startswith("停止"):
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n表示を止めました（学習は裏で続いています）。もう一度 watch を実行すると表示を再開します。")


def _japanese_font():
    """グラフの日本語が文字化けしないよう、日本語フォントを用意する（初回だけ少し時間がかかる）。"""
    try:
        import matplotlib
        from matplotlib import font_manager
        cands = glob.glob("/usr/share/fonts/**/*CJK*", recursive=True) + glob.glob("/usr/share/fonts/**/*IPA*", recursive=True)
        if not cands:
            subprocess.run("apt-get -qq install -y fonts-noto-cjk > /dev/null 2>&1", shell=True)
            cands = glob.glob("/usr/share/fonts/**/*CJK*", recursive=True)
        if cands:
            font_manager.fontManager.addfont(cands[0])
            matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=cands[0]).get_name()
    except Exception:
        pass
