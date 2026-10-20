"""変更禁止ファイルが書き換えられていないかを確認する（変更禁止）。"""
import hashlib, os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
bad = 0
for line in open(os.path.join(ROOT, "PROTECTED.sha256"), encoding="utf-8"):
    h, path = line.strip().split("  ", 1)
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        print("MISSING ", path); bad += 1; continue
    got = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if got != h:
        print("CHANGED ", path); bad += 1
print("OK" if bad == 0 else f"NG: {bad} file(s)")
sys.exit(1 if bad else 0)
