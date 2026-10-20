# 変更禁止（PROTECTED）。Shikibu M1 の受け入れテスト。
# 実装（shikibu_feat モジュール）が満たすべき API：
#   num_features(version: int) -> int
#   extract_sfen(sfen: str, perspective: int, version: int) -> list[int]   # 昇順・重複込み
#   decode_psv(rec: bytes) -> tuple[str, int, int, int, int]             # (sfen, score, move, gamePly, game_result)
#   extract_psv(rec: bytes, perspective: int, version: int) -> list[int]  # 昇順・重複込み
# perspective: 0 = 先手視点, 1 = 後手視点 / version: 0 = RayPair_v0, 1 = RayPair_v1
import json
import os
import random

import pytest

from conftest import N_RANDOM, flip_sfen, make_psv, normalize, random_boards, ref

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = json.load(open(os.path.join(HERE, "golden", "raypair_golden.json"), encoding="utf-8"))


def section_counts(idx, version):
    n = GOLDEN["num_features"][str(version)]
    b1hi = n - 112
    return (sum(0 <= i < 2268 for i in idx), sum(2268 <= i < 2344 for i in idx),
            sum(2344 <= i < b1hi for i in idx), sum(b1hi <= i < n for i in idx))


# ---------- テスト2：正解データ（§F7・付録B・§F2.3） ----------

@pytest.mark.parametrize("v", [0, 1])
def test_num_features(feat, v):
    assert feat.num_features(v) == GOLDEN["num_features"][str(v)]


@pytest.mark.parametrize("case", GOLDEN["cases"],
                         ids=lambda c: f"{c['id']}-v{c['version']}-p{c['perspective']}")
def test_golden(feat, case):
    got = list(feat.extract_sfen(case["sfen"], case["perspective"], case["version"]))
    assert got == sorted(got), "昇順に並んでいない"
    assert len(got) == case["count"], "総数が違う"
    assert section_counts(got, case["version"]) == section_counts(case["indices"], case["version"]), "区間ごとの個数が違う"
    assert sum(got) == case["sum"], "総和が違う"
    assert ref.digest(got) == case["sha16"], "ハッシュが違う"
    assert got == case["indices"], "全リストが違う"


# ---------- テスト1：デコーダ照合（cshogi） ----------

@pytest.fixture(scope="module")
def boards():
    return random_boards(N_RANDOM)


def test_decoder_vs_cshogi(feat, boards):
    rng = random.Random(1)
    for b in boards:
        score = rng.randrange(-32000, 32001)
        move = rng.randrange(0, 65536)
        ply = rng.randrange(0, 1000)
        result = rng.choice([-1, 0, 1])
        rec = make_psv(b, score, move, ply, result)
        assert len(rec) == 40
        sfen, s2, m2, p2, r2 = feat.decode_psv(rec)
        assert normalize(sfen) == normalize(b.sfen()), f"局面が違う: 期待 {b.sfen()} / 実際 {sfen}"
        assert (s2, m2, p2, r2) == (score, move, ply, result)


# ---------- テスト3：参照実装との照合（ランダム局面） ----------

@pytest.mark.parametrize("v", [0, 1])
def test_random_vs_reference(feat, boards, v):
    for b in boards:
        s = b.sfen()
        for p in (0, 1):
            assert list(feat.extract_sfen(s, p, v)) == ref.features(s, p, v), f"不一致: {s} p={p} v={v}"


def test_psv_path_matches_sfen_path(feat, boards):
    for b in boards[:2000]:
        rec = make_psv(b, 0, 0, 0, 0)
        for v in (0, 1):
            for p in (0, 1):
                assert list(feat.extract_psv(rec, p, v)) == list(feat.extract_sfen(b.sfen(), p, v))


# ---------- テスト4：視点対称性 ----------

@pytest.mark.parametrize("v", [0, 1])
def test_perspective_symmetry(feat, boards, v):
    for b in boards:
        s = b.sfen()
        fs = flip_sfen(s)
        assert list(feat.extract_sfen(s, 0, v)) == list(feat.extract_sfen(fs, 1, v))
        assert list(feat.extract_sfen(s, 1, v)) == list(feat.extract_sfen(fs, 0, v))


# ---------- テスト5：範囲と個数の上限 ----------

@pytest.mark.parametrize("v", [0, 1])
def test_range_and_bound(feat, boards, v):
    n = GOLDEN["num_features"][str(v)]
    for b in boards:
        s = b.sfen()
        for p in (0, 1):
            got = list(feat.extract_sfen(s, p, v))
            assert all(0 <= i < n for i in got)
            assert len(got) <= 208
