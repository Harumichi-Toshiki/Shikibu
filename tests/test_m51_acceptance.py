# 変更禁止（PROTECTED）。Shikibu M5.1（汎用化と、運用上の不具合の修正）の受け入れテスト。
#
# ■ 1. リポジトリは特定の利用者・データ・保存場所に依存しない
#   README.md、docs/（DESIGN.md を除く）、configs/、notebooks/、tools/、python/ に、特定の個人のフォルダ名や
#   特定の教師データの名前を書かない（FORBIDDEN を参照）。
#   configs/ の設定ファイルは「ひな形」とし、利用者が自分の環境に合わせて書き換える値（データの場所、出力先など）は
#   "<…>" の形の目印で書く（例："/content/drive/MyDrive/<データのフォルダ>/*.psv"）。
#   学習器は、パス（train_paths・train_glob・val_path・cache_dir・out_dir）に "<" と ">" で囲まれた部分が残っていたら、
#   「設定のひな形のまま」である旨のエラーで、何も書き出さずに終了する（終了コード 1）。
#   notebooks/ は、取得するブランチを "main" にし、作業用のブランチ（"claude/" で始まるもの）を書かない。
#
# ■ 2. 同じ出力フォルダでの二重起動を防ぐ（ロック）
#   学習器（python -m shikibu.train）は、出力フォルダに何かを書く前に <out_dir>/train.lock を作る。中身は JSON：
#     {"pid": int, "host": str（socket.gethostname()）, "started": UNIX 時刻, "heartbeat": UNIX 時刻}
#   学習中は 30 秒以内の間隔で heartbeat とファイルの更新時刻を新しくする。正常終了・エラー終了のときは消す。
#   すでに train.lock があるときは、次の場合に「学習が実行中」という語を含むエラーで、何も書き出さずに終了する（終了コード 1）：
#     - host が自分と同じで、その pid のプロセスが生きている
#     - host が自分と違い、ファイルの更新時刻が 600 秒以内
#   それ以外（同じ host で pid が死んでいる、違う host で 600 秒より古い）は、古いロックとみなして引き継ぎ、標準エラーに警告を出して続ける。
#   --force-unlock を付けると、ロックの状態にかかわらず引き継ぐ。
#   tools/shk_monitor.py の start も、force=True で --force-unlock を付けて起動できること。
#
# ■ 3. 設定の検査
#   train.log_every_steps は 1 以上、train.save_every_steps と train.val_every_steps は 0 以上（0 は無効）、
#   train.progress_every_sec は 0 以上。違反したら、キーの名前を含むエラーで終了する（終了コード 1、トレースバックを出さない）。
#
# ■ 4. tools/shk_tools.py shuffle の出力先が空でないとき
#   "エラー:" で始まる1行を標準エラーに出し、終了コード 1 で終わる（トレースバックを出さない）。
#
# ■ 5. .gitignore に .venv/ を含める。
import json
import os
import signal
import socket
import subprocess
import sys
import time

import pytest

from m3_data import make_material_dataset
from test_m3_acceptance import make_config, read_log

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORBIDDEN = ["DainagonProject", "匠生用", "データセット/psv", "PonkotsuWCSC", "Ponkotsu", "ponkotsu", "水匠", "Suisho", "suisho"]
SCAN_DIRS = ["docs", "configs", "notebooks", "tools", "python"]
SKIP = {os.path.join("docs", "DESIGN.md")}


def env():
    e = dict(os.environ)
    e["PYTHONPATH"] = os.path.join(ROOT, "python") + os.pathsep + e.get("PYTHONPATH", "")
    return e


def train_proc(cfg, *extra):
    return subprocess.Popen([sys.executable, "-m", "shikibu.train", "--config", str(cfg), *extra],
                            cwd=ROOT, env=env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def train_run(cfg, *extra, timeout=300):
    r = subprocess.run([sys.executable, "-m", "shikibu.train", "--config", str(cfg), *extra],
                       cwd=ROOT, env=env(), capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    return make_material_dataset(str(tmp_path_factory.mktemp("m51data")))


# ---------- 1. 汎用化 ----------

def test_no_personal_strings():
    files = [os.path.join(ROOT, "README.md")]
    for d in SCAN_DIRS:
        for dp, _, fs in os.walk(os.path.join(ROOT, d)):
            if "__pycache__" in dp:
                continue
            files += [os.path.join(dp, f) for f in fs]
    hits = []
    for p in files:
        rel = os.path.relpath(p, ROOT)
        if rel in SKIP:
            continue
        try:
            text = open(p, encoding="utf-8").read()
        except (UnicodeDecodeError, IsADirectoryError):
            continue
        for w in FORBIDDEN:
            if w in text:
                hits.append(f"{rel}: {w}")
    assert not hits, "特定の利用者・データに依存する記述が残っている:\n" + "\n".join(hits)


def test_notebooks_use_main():
    for f in os.listdir(os.path.join(ROOT, "notebooks")):
        if not f.endswith(".ipynb"):
            continue
        nb = json.load(open(os.path.join(ROOT, "notebooks", f), encoding="utf-8"))
        src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
        assert "claude/" not in src, f"{f} に作業用のブランチが書かれている"
        assert '"main"' in src or "'main'" in src, f"{f} が main を取得していない"


def test_configs_are_templates(tmp_path):
    cdir = os.path.join(ROOT, "configs")
    cfgs = [f for f in os.listdir(cdir) if f.endswith(".toml")]
    assert cfgs
    for f in cfgs:
        text = open(os.path.join(cdir, f), encoding="utf-8").read()
        assert "<" in text and ">" in text, f"{f} に書き換え用の目印 <…> が無い"
        rc, out, err = train_run(os.path.join(cdir, f), timeout=120)
        assert rc != 0 and "ひな形" in err, f"{f} をそのまま使ったのにエラーにならない（{err[-300:]}）"
        assert "Traceback" not in err


def test_placeholder_rejected_without_writing(tmp_path, data):
    out = tmp_path / "<出力先>"
    cfg = make_config(tmp_path / "p.toml", data, out, train__max_steps=2, train__total_steps=2)
    rc, _, err = train_run(cfg)
    assert rc != 0 and "ひな形" in err
    assert not out.exists(), "ひな形のままの出力先に書き出した"


# ---------- 2. 二重起動の防止 ----------

def _wait(cond, timeout=120):
    t = time.time()
    while time.time() - t < timeout:
        if cond():
            return True
        time.sleep(0.5)
    return False


def _starts(out):
    p = out / "log.jsonl"
    return sum(1 for r in read_log(str(out)) if r.get("type") == "start") if p.exists() else 0


def test_lock_running_same_host(tmp_path, data):
    out = tmp_path / "run"
    cfg = make_config(tmp_path / "c.toml", data, out, train__max_steps=10**7, train__total_steps=10**7,
                      train__log_every_steps=1000, train__save_every_steps=50)
    a = train_proc(cfg)
    try:
        assert _wait(lambda: (out / "checkpoint.pt").exists()), "学習 A が始まらない"
        lk = json.load(open(out / "train.lock"))
        assert lk["pid"] == a.pid and lk["host"] == socket.gethostname()
        n0 = _starts(out)
        rc, _, err = train_run(cfg, "--resume", timeout=60)
        assert rc != 0 and "学習が実行中" in err, f"実行中なのに2つ目が始まった（{err[-300:]}）"
        assert _starts(out) == n0, "断った側が log.jsonl に書いた"
        st = json.load(open(out / "status.json"))
        assert st["phase"] != "error", "断った側が status.json を書き換えた"
    finally:
        a.send_signal(signal.SIGKILL)
        a.wait()
    # A が異常終了した（ロックは残り、pid は死んでいる）→ 引き継いで再開できる
    assert (out / "train.lock").exists()
    cfg2 = make_config(tmp_path / "c2.toml", data, out, train__max_steps=json.load(open(out / "status.json"))["step"] + 5,
                       train__total_steps=10**7, train__log_every_steps=1000, train__save_every_steps=50)
    rc, _, err = train_run(cfg2, "--resume")
    assert rc == 0, err[-500:]
    assert not (out / "train.lock").exists(), "正常終了したのにロックが残っている"


def _fake_lock(out, host, pid, age):
    out.mkdir(parents=True, exist_ok=True)
    p = out / "train.lock"
    now = time.time()
    p.write_text(json.dumps({"pid": pid, "host": host, "started": now - age, "heartbeat": now - age}))
    os.utime(p, (now - age, now - age))


def test_lock_other_host_and_force(tmp_path, data):
    out = tmp_path / "run2"
    cfg = make_config(tmp_path / "c.toml", data, out, train__max_steps=3, train__total_steps=3)
    _fake_lock(out, "some-other-host-xyz", 999999, age=5)
    rc, _, err = train_run(cfg)
    assert rc != 0 and "学習が実行中" in err
    assert not (out / "log.jsonl").exists()
    _fake_lock(out, "some-other-host-xyz", 999999, age=3600)
    rc, _, err = train_run(cfg)
    assert rc == 0, err[-500:]
    _fake_lock(out, "some-other-host-xyz", 999999, age=5)
    rc, _, err = train_run(cfg, "--resume", "--force-unlock")
    assert rc == 0, err[-500:]


def test_lock_dead_pid_same_host(tmp_path, data):
    p = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
    dead = int(p.stdout)
    out = tmp_path / "run3"
    _fake_lock(out, socket.gethostname(), dead, age=5)
    cfg = make_config(tmp_path / "c.toml", data, out, train__max_steps=3, train__total_steps=3)
    rc, _, err = train_run(cfg)
    assert rc == 0, err[-500:]


def test_monitor_force_parameter():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import inspect
    import shk_monitor
    assert "force" in inspect.signature(shk_monitor.start).parameters


# ---------- 3. 設定の検査 ----------

@pytest.mark.parametrize("key,val", [("log_every_steps", 0), ("log_every_steps", -1), ("save_every_steps", -1),
                                     ("val_every_steps", -5), ("progress_every_sec", -1.0)])
def test_config_validation(tmp_path, data, key, val):
    cfg = make_config(tmp_path / "v.toml", data, tmp_path / "out", train__max_steps=2, train__total_steps=2,
                      **{f"train__{key}": val})
    rc, _, err = train_run(cfg)
    assert rc != 0 and key in err and "Traceback" not in err, err[-500:]


# ---------- 4. shuffle の出力先 ----------

def test_shuffle_nonempty_out(tmp_path, data):
    out = tmp_path / "shards"
    out.mkdir()
    (out / "shard_00000.psv").write_bytes(b"\0" * 40)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "shk_tools.py"), "shuffle", data[0],
                        "--out", str(out)], cwd=ROOT, env=env(), capture_output=True, text=True, timeout=120)
    assert r.returncode == 1
    assert "Traceback" not in r.stderr
    assert any(line.startswith("エラー:") for line in r.stderr.splitlines()), r.stderr[-300:]


# ---------- 5. .gitignore ----------

def test_gitignore_venv():
    lines = [l.strip() for l in open(os.path.join(ROOT, ".gitignore"), encoding="utf-8")]
    assert ".venv/" in lines or ".venv" in lines
