# 変更禁止（PROTECTED）。M7 の独自の特徴「玉の周りのブロック（区間 C）」の参照実装（設計書 §S12.3）。
# RayPair の特徴（reference/raypair_ref.py）の後ろに付け足して、特徴の版 12（raypair0k）・13（raypair1k）を作る。
#
# ■ 座標と駒の番号は RayPair と同じ（設計書 §F1）。視点座標 (y, x)、駒番号 pc（視点側 0〜13、相手側 +14）。
#   cshogi のマスの番号 s（= (筋 − 1) × 9 + 段の番号）からは、先手視点で y = s % 9、x = 8 − s // 9。
#   後手視点では (8 − y, 8 − x) とし、先後を入れ替える。
#
# ■ 区間 C：玉の周り（5×5 から中心を除いた 24 マス）にあるもの
#   kk = 0（視点側の玉）、1（相手の玉）。玉のマスを (ky, kx) とする。
#   dy, dx = −2〜2 で (0, 0) 以外の 24 通りについて、o = (dy + 2) × 5 + (dx + 2)（12 より大きければ 1 を引く。0〜23）。
#   マス (ky + dy, kx + dx) が
#     盤の外      → c = 28
#     駒がある    → c = その駒の pc（0〜27。相手の玉・自分の玉もそのまま）
#     空いている  → 特徴なし
#   idx = 基準 ＋ kk × 696 ＋ o × 29 ＋ c       （基準：版 12 は 15000、版 13 は 115352）
#   区間 C の次元は 2 × 24 × 29 = 1392。版 12 は 15000 + 1392 = 16392、版 13 は 115352 + 1392 = 116744 次元。
#   1局面1視点の個数は、RayPair の分（最大 208）＋ 区間 C（最大 48）で、最大 256。
PC = {1: 0, 2: 1, 3: 2, 4: 3, 7: 4, 5: 5, 6: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11, 13: 12, 14: 13}  # cshogi の駒種 → pc
BASE = {12: 15000, 13: 115352}
DIMS = {12: 16392, 13: 116744}
C_DIMS = 1392


def view_board(board, persp):
    """{(y, x): pc} の辞書（視点座標）。"""
    out = {}
    for s, p in enumerate(board.pieces):
        if p == 0:
            continue
        color = 0 if p < 16 else 1
        y, x = s % 9, 8 - s // 9
        if persp == 1:
            y, x = 8 - y, 8 - x
        out[(y, x)] = PC[p & 15] + (0 if color == persp else 14)
    return out


def zone_features(board, persp, base):
    vb = view_board(board, persp)
    out = []
    for kk, kpc in ((0, 7), (1, 21)):
        (ky, kx), = [sq for sq, pc in vb.items() if pc == kpc]
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dy == 0 and dx == 0:
                    continue
                o = (dy + 2) * 5 + (dx + 2)
                if o > 12:
                    o -= 1
                y, x = ky + dy, kx + dx
                if not (0 <= y < 9 and 0 <= x < 9):
                    c = 28
                elif (y, x) in vb:
                    c = vb[(y, x)]
                else:
                    continue
                out.append(base + kk * 696 + o * 29 + c)
    return out


def king_squares(board, stm_first=True):
    """(手番側の玉のマス（手番側の視点）, 相手の玉のマス（相手の視点）)。マスは視点座標の sq = y × 9 + x。"""
    stm = 0 if board.turn == 0 else 1
    res = []
    for persp in (stm, 1 - stm):
        vb = view_board(board, persp)
        (ky, kx), = [sq for sq, pc in vb.items() if pc == 7]
        res.append(ky * 9 + kx)
    return tuple(res)


def zone(sq):
    """玉の段の区分（その玉の側の視点）：七〜九段 0、四〜六段 1、一〜三段 2。"""
    y = sq // 9
    return 0 if y >= 6 else (1 if y >= 3 else 2)


def k3k3_bucket(board):
    a, b = king_squares(board)
    return zone(a) * 3 + zone(b)
