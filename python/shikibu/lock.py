"""同じ出力フォルダでの二重起動を防ぐロック（<out_dir>/train.lock）。

中身は JSON：{"pid", "host", "started", "heartbeat"}。
学習中は HEARTBEAT_SEC ごとに書き直す（heartbeat と、ファイルの更新時刻が新しくなる）。
正常終了・エラー終了のときは、自分のロックだけを消す。

判定（同じ出力フォルダにロックがあるとき）：
- host が自分と同じ：pid のプロセスが生きていれば「実行中」。死んでいれば古いロック。
- host が違う（または読めない）：更新時刻が STALE_SEC 秒以内なら「実行中」。それより古ければ古いロック。
古いロックは、警告を出して引き継ぐ。--force-unlock なら、実行中でも引き継ぐ。
"""
import json
import os
import socket
import sys
import threading
import time

LOCK_NAME = "train.lock"
STALE_SEC = 600.0      # 別のホストのロックを「古い」とみなす更新時刻の差（秒）
HEARTBEAT_SEC = 10.0   # 心拍の間隔（30 秒より十分短く）


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # 別のユーザーのプロセス。生きている
        return True
    return True


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _is_running(holder, mtime, host):
    if holder is not None and holder.get("host") == host and isinstance(holder.get("pid"), int):
        return _pid_alive(holder["pid"])
    return time.time() - mtime <= STALE_SEC


class TrainLock:
    def __init__(self, out_dir):
        self.path = os.path.join(out_dir, LOCK_NAME)
        self.pid = os.getpid()
        self.host = socket.gethostname()
        self.started = time.time()
        self._stop = threading.Event()
        self._thread = None

    @classmethod
    def acquire(cls, out_dir, force=False):
        """ロックを取って返す。実行中の学習があれば ValueError（「学習が実行中」を含む）。"""
        os.makedirs(out_dir, exist_ok=True)
        lock = cls(out_dir)
        if os.path.exists(lock.path):
            holder = _read(lock.path)
            running = _is_running(holder, os.path.getmtime(lock.path), lock.host)
            who = f"host={holder.get('host')} pid={holder.get('pid')}" if holder else "読めないロック"
            if running and not force:
                raise ValueError(
                    f"学習が実行中です（{who}）。同じ出力フォルダで2つは動かせません。"
                    "止めてから再実行するか、--force-unlock で引き継いでください。")
            if running:
                print(f"警告: 実行中かもしれないロック（{who}）を --force-unlock で引き継ぎます。", file=sys.stderr)
            else:
                print(f"警告: 古いロック（{who}）を引き継ぎます。", file=sys.stderr)
        lock._write()
        lock._thread = threading.Thread(target=lock._beat, daemon=True)
        lock._thread.start()
        return lock

    def _write(self):
        data = {"pid": self.pid, "host": self.host, "started": self.started, "heartbeat": time.time()}
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, self.path)  # 書き直すと更新時刻も新しくなる

    def _beat(self):
        while not self._stop.wait(HEARTBEAT_SEC):
            try:
                self._write()
            except OSError:
                pass

    def release(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        holder = _read(self.path)
        if holder is not None and holder.get("pid") == self.pid and holder.get("host") == self.host:
            try:
                os.remove(self.path)
            except OSError:
                pass
