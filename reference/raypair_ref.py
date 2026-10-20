# RayPair reference implementation (readable, slow). Frozen-contract oracle.
import hashlib

PIECE_ORDER = ["P","L","N","S","G","B","R","K","+P","+L","+N","+S","+B","+R"]
HAND_ORDER  = ["P","L","N","S","G","B","R"]
HAND_MAX    = [18,4,4,4,4,2,2]
HAND_OFF    = [0,18,22,26,30,34,36]
A1_BASE, A2_BASE, B1_BASE = 0, 2268, 2344
B1_LOCAL = 12544  # 28*28*4*4

def parse_sfen(sfen):
    parts = sfen.split()
    rows = parts[0].split("/"); assert len(rows) == 9
    board = [[None]*9 for _ in range(9)]
    for y, row in enumerate(rows):
        x = 0; promo = False
        for ch in row:
            if ch.isdigit(): x += int(ch); continue
            if ch == "+": promo = True; continue
            color = 0 if ch.isupper() else 1
            board[y][x] = (color, ("+" if promo else "") + ch.upper()); x += 1; promo = False
        assert x == 9, row
    stm = 0 if parts[1] == "b" else 1
    hands = [[0]*7, [0]*7]
    if parts[2] != "-":
        n = 0
        for ch in parts[2]:
            if ch.isdigit(): n = n*10 + int(ch); continue
            hands[0 if ch.isupper() else 1][HAND_ORDER.index(ch.upper())] += (n if n else 1); n = 0
    return board, stm, hands

def view(board, hands, persp):
    vb = [[None]*9 for _ in range(9)]
    for y in range(9):
        for x in range(9):
            p = board[y][x]
            if p is None: continue
            color, pt = p
            yy, xx = (y, x) if persp == 0 else (8-y, 8-x)
            vb[yy][xx] = PIECE_ORDER.index(pt) + (0 if color == persp else 14)
    return vb, [hands[persp], hands[1-persp]]

def lines(axis):
    if axis == 0:
        for x in range(9): yield [(y, x) for y in range(9)]
    elif axis == 1:
        for y in range(9): yield [(y, x) for x in range(9)]
    elif axis == 2:
        for c in range(-8, 9): yield [(y, y+c) for y in range(9) if 0 <= y+c < 9]
    else:
        for s in range(17): yield [(y, s-y) for y in range(9) if 0 <= s-y < 9]

def db_of(d): return 0 if d == 1 else 1 if d == 2 else 2 if d <= 4 else 3

def num_features(version): return 2344 + (1 if version == 0 else 9)*B1_LOCAL + 112

def features(sfen, persp, version):
    board, stm, hands = parse_sfen(sfen)
    vb, vh = view(board, hands, persp)
    out = []
    for y in range(9):
        for x in range(9):
            if vb[y][x] is not None: out.append(A1_BASE + vb[y][x]*81 + y*9 + x)
    for side in range(2):
        for h in range(7):
            assert vh[side][h] <= HAND_MAX[h]
            for k in range(vh[side][h]): out.append(A2_BASE + side*38 + HAND_OFF[h] + k)
    for axis in range(4):
        for ln in lines(axis):
            occ = [(y, x) for (y, x) in ln if vb[y][x] is not None]
            for (ya, xa), (yb, xb) in zip(occ, occ[1:]):
                d = (xb - xa) if axis == 1 else (yb - ya)
                assert d >= 1
                local = vb[ya][xa]*448 + vb[yb][xb]*16 + axis*4 + db_of(d)
                region = 0 if version == 0 else (ya//3)*3 + (xa//3)
                out.append(B1_BASE + region*B1_LOCAL + local)
    b2 = B1_BASE + (1 if version == 0 else 9)*B1_LOCAL
    for y in range(9):
        for x in range(9):
            pc = vb[y][x]
            if pc not in (2, 16): continue
            s = 0 if pc == 2 else 1
            ty = y-2 if s == 0 else y+2
            for lr, tx in ((0, x-1), (1, x+1)):
                if 0 <= ty < 9 and 0 <= tx < 9 and vb[ty][tx] is not None:
                    out.append(b2 + s*56 + vb[ty][tx]*2 + lr)
    return sorted(out)

def digest(idx): return hashlib.sha256(",".join(map(str, idx)).encode()).hexdigest()[:16]
