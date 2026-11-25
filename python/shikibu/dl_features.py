"""DL の入力（M11）：詰めた局面（Rust の dl_record・BatchLoader(output="dl")）から、55 チャンネルの特徴を GPU で作る。

正解は Dainagon の features.py（reference/dainagon_ref.py の make_features_v16_fast）。展開の結果は、それとビット単位で同じ。

- expand(board, hands, check)：バッチを 55 チャンネル（B, 55, 9, 9）の float32 にする（Python のループは回さない）。
  board・hands・check は torch のテンソル（uint8 か整数型）。出力は board と同じ device。
- hl_gauss(t)：Binned Value の教師（128 ビン、σ = 0.75 / 128 の HL-Gauss）。Dainagon の make_hl_gauss_target と同じ。
- from_hcpe(arr)：cshogi の hcpe を、expand の入力と同じ形の dict にする（検証用。遅くてよい）。

チャンネル（Dainagon と同じ）：
  0〜13 手番側の駒、14〜27 相手の駒（駒の種類の順は §F1.3）、28〜34 手番側の持ち駒、35〜41 相手の持ち駒、
  42 手番側の飛系の利き、43 手番側の角系、44 相手の飛系、45 相手の角系、
  46 手番側の玉の 5×5、47 手番側の玉の 7×7、48 相手の玉の 5×5、49 相手の玉の 7×7、
  50 王手、51 0、52 筋（x / 8）、53 段（y / 8）。
"""
import math

import numpy as np
import torch

NUM_CHANNELS = 55
EMPTY = 28          # 盤の空き（チャンネルにしない）
N_HAND = 7
N_SQ = 81
NUM_VALUE_BINS = 128
VALUE_SIGMA = 0.75 * (1.0 / NUM_VALUE_BINS)

# 持ち駒の枚数 → 特徴の値。Dainagon は「枚数 / 10.0」を float32 の配列に入れる（float64 で割ってから丸める）。
HAND_TABLE = (np.arange(19, dtype=np.float64) / 10.0).astype(np.float32)
RAMP = np.arange(9, dtype=np.float32) / 8.0

# 盤の駒の種類（board の値）
OWN_BISHOP, OWN_ROOK, OWN_HORSE, OWN_DRAGON, OWN_KING = 5, 6, 12, 13, 7
OPP = 14  # 相手の駒は +14

ROOK_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BISHOP_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

_TABLES = {}


def _ray_table(dirs):
    """(4, 81, 8)：マス t から方向 d に k 歩戻ったマス（盤の外は 81）。大駒の光線は、ここから占有の累積積で求める。"""
    idx = np.full((len(dirs), N_SQ, 8), N_SQ, dtype=np.int64)
    for di, (dy, dx) in enumerate(dirs):
        for t in range(N_SQ):
            ty, tx = divmod(t, 9)
            for k in range(1, 9):
                sy, sx = ty - k * dy, tx - k * dx
                if 0 <= sy < 9 and 0 <= sx < 9:
                    idx[di, t, k - 1] = sy * 9 + sx
    return idx


def _box_table(r):
    """(82, 81)：玉のマス k（行）から、チェビシェフ距離 r 以内のマス（列）。行 81 は玉が無いときの 0 行。"""
    m = np.zeros((N_SQ + 1, N_SQ), dtype=np.float32)
    for k in range(N_SQ):
        ky, kx = divmod(k, 9)
        for t in range(N_SQ):
            ty, tx = divmod(t, 9)
            if max(abs(ty - ky), abs(tx - kx)) <= r:
                m[k, t] = 1.0
    return m


def _tables(device):
    key = str(device)
    if key not in _TABLES:
        _TABLES[key] = {
            "rook": torch.from_numpy(_ray_table(ROOK_DIRS)).to(device),
            "bishop": torch.from_numpy(_ray_table(BISHOP_DIRS)).to(device),
            "box2": torch.from_numpy(_box_table(2)).to(device),
            "box3": torch.from_numpy(_box_table(3)).to(device),
            "hand": torch.from_numpy(HAND_TABLE).to(device),
            "ramp": torch.from_numpy(np.tile(RAMP, 9)).to(device),        # 筋（t % 9）
            "ramp_y": torch.from_numpy(np.repeat(RAMP, 9)).to(device),    # 段（t // 9）
        }
    return _TABLES[key]


def _ray(src, empty, idx):
    """大駒の光線。src（B, 82）は利きの元になる駒の 0/1、empty は空きの 0/1（どちらも末尾の列は 0）。
    各マスについて、「元の駒から手前（元と自分の間）がすべて空き」なら 1。その結果を方向の和（OR）にする。
    返り値は (B, 81)。"""
    s = src[:, idx]                   # (B, 4, 81, 8)
    e = empty[:, idx]
    cum = torch.cumprod(e, dim=3)
    excl = torch.cat([torch.ones_like(cum[..., :1]), cum[..., :-1]], dim=3)
    return (s * excl).amax(dim=3).amax(dim=1)


def _step(src, idx):
    """1 歩先（馬・竜の隣の 1 マス）。駒があっても付ける。返り値は (B, 81)。"""
    return src[:, idx[:, :, 0]].amax(dim=1)


def _pad(x):
    return torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)


def _king_index(mask):
    """玉のマス（無ければ 81）。mask は (B, 81) の 0/1。"""
    has = mask.any(dim=1)
    idx = mask.long().argmax(dim=1)
    return torch.where(has, idx, torch.full_like(idx, N_SQ))


def expand(board, hands, check):
    """詰めた局面のバッチを 55 チャンネルにする。

    board：(B, 81) uint8（ty × 9 + tx。値は 0〜27 の駒、28 は空き）。
    hands：(B, 14) uint8（手番側の 7 種、相手の 7 種）。check：(B,) 0/1。
    返り値：(B, 55, 9, 9) float32（board と同じ device）。dainagon_ref.make_features_v16_fast とビット単位で同じ。
    """
    device = board.device
    tb = _tables(device)
    b = board.to(torch.long)
    B = b.shape[0]
    x = torch.zeros((B, NUM_CHANNELS, N_SQ), dtype=torch.float32, device=device)

    # 0〜27：駒の one-hot（空きは捨てる）
    ch = torch.arange(EMPTY, device=device).view(1, EMPTY, 1)
    x[:, :EMPTY] = (b.unsqueeze(1) == ch).to(torch.float32)

    # 28〜41：持ち駒の枚数（0 枚は 0）
    h = hands.to(torch.long)
    x[:, 28:35] = tb["hand"][h[:, :N_HAND]].unsqueeze(2).expand(B, N_HAND, N_SQ)
    x[:, 35:42] = tb["hand"][h[:, N_HAND:]].unsqueeze(2).expand(B, N_HAND, N_SQ)

    # 42〜45：大駒の利き。占有は盤のすべての駒（両方の色）
    empty = _pad((b == EMPTY).to(torch.float32))
    rook_src = lambda c: _pad((b == c).to(torch.float32))  # noqa: E731

    own_rook = rook_src(OWN_ROOK)
    own_dragon = rook_src(OWN_DRAGON)
    own_bishop = rook_src(OWN_BISHOP)
    own_horse = rook_src(OWN_HORSE)
    opp_rook = rook_src(OWN_ROOK + OPP)
    opp_dragon = rook_src(OWN_DRAGON + OPP)
    opp_bishop = rook_src(OWN_BISHOP + OPP)
    opp_horse = rook_src(OWN_HORSE + OPP)

    x[:, 42] = torch.maximum(_ray(torch.maximum(own_rook, own_dragon), empty, tb["rook"]),
                             _step(own_dragon, tb["bishop"]))
    x[:, 43] = torch.maximum(_ray(torch.maximum(own_bishop, own_horse), empty, tb["bishop"]),
                             _step(own_horse, tb["rook"]))
    x[:, 44] = torch.maximum(_ray(torch.maximum(opp_rook, opp_dragon), empty, tb["rook"]),
                             _step(opp_dragon, tb["bishop"]))
    x[:, 45] = torch.maximum(_ray(torch.maximum(opp_bishop, opp_horse), empty, tb["bishop"]),
                             _step(opp_horse, tb["rook"]))

    # 46〜49：玉の周り（玉が無ければ 0）
    k_own = _king_index(b == OWN_KING)
    k_opp = _king_index(b == OWN_KING + OPP)
    x[:, 46] = tb["box2"][k_own]
    x[:, 47] = tb["box3"][k_own]
    x[:, 48] = tb["box2"][k_opp]
    x[:, 49] = tb["box3"][k_opp]

    # 50：王手。51：0。52・53：筋と段（回転しない）
    x[:, 50] = check.to(torch.float32).view(B, 1)
    x[:, 52] = tb["ramp"]
    x[:, 53] = tb["ramp_y"]
    return x.view(B, NUM_CHANNELS, 9, 9)


def hl_gauss(t):
    """Binned Value の教師。t は勝率（0〜1）、返り値は (B, 128) の float32。

    各ビットの区間 [i/128, (i+1)/128] に、σ = 0.75/128 の正規分布の質量を割り当て、正規化する。
    Dainagon の make_hl_gauss_target と同じ（float64 で計算してから float32 にする）。
    """
    t = torch.as_tensor(t, dtype=torch.float64).reshape(-1)
    edges = torch.arange(NUM_VALUE_BINS + 1, dtype=torch.float64, device=t.device) / NUM_VALUE_BINS
    cdf = 0.5 * (1.0 + torch.erf((edges.unsqueeze(0) - t.unsqueeze(1)) / (VALUE_SIGMA * math.sqrt(2.0))))
    probs = cdf[:, 1:] - cdf[:, :-1]
    probs = probs / probs.sum(dim=1, keepdim=True)
    return probs.to(torch.float32)


# ---------- 検証用：hcpe から（cshogi、遅い） ----------

# cshogi の駒の種類（1〜14）→ チャンネル（0〜13）。Dainagon の CSHOGI_PT_TO_SHOGI_IDX と同じ。
_CSHOGI_PT_TO_CH = [0, 0, 1, 2, 3, 5, 6, 4, 7, 8, 9, 10, 11, 12, 13]
_RMAP_NUM = str.maketrans("123456789", "987654321")
_RMAP_ALP = str.maketrans("abcdefghi", "ihgfedcba")
_DROP_IDX = {"P": 0, "L": 1, "N": 2, "S": 3, "G": 4, "B": 5, "R": 6}


def _rotate_usi(u):
    if "*" in u:
        return u[0] + "*" + u[2:].translate(_RMAP_NUM).translate(_RMAP_ALP)
    return u[:2].translate(_RMAP_NUM).translate(_RMAP_ALP) + u[2:4].translate(_RMAP_NUM).translate(_RMAP_ALP) + u[4:]


def _usi_to_action(usi):
    """指し手（USI、手番側の座標）を 2187 の番号にする。変換できなければ None。"""
    if usi[1] == "*":
        if usi[0] not in _DROP_IDX:
            return None
        dir_idx = 20 + _DROP_IDX[usi[0]]
        to_x, to_y = 9 - int(usi[2]), ord(usi[3]) - ord("a")
    else:
        from_x, from_y = 9 - int(usi[0]), ord(usi[1]) - ord("a")
        to_x, to_y = 9 - int(usi[2]), ord(usi[3]) - ord("a")
        dy, dx = to_y - from_y, to_x - from_x   # 距離は問わない（香・角・飛は何マスでも同じ方向）
        if dy < 0 and dx == 0:
            d = 0
        elif dy < 0 and dx < 0 and dy == dx:
            d = 1
        elif dy < 0 and dx > 0 and -dy == dx:
            d = 2
        elif dy == 0 and dx < 0:
            d = 3
        elif dy == 0 and dx > 0:
            d = 4
        elif dy > 0 and dx == 0:
            d = 5
        elif dy > 0 and dx < 0 and dy == -dx:
            d = 6
        elif dy > 0 and dx > 0 and dy == dx:
            d = 7
        elif dy == -2 and dx == -1:
            d = 8
        elif dy == -2 and dx == 1:
            d = 9
        else:
            return None
        dir_idx = d + (10 if len(usi) == 5 and usi[4] == "+" else 0)
    return dir_idx * 81 + to_y * 9 + to_x


def _compact_cshogi(cb):
    """cshogi の局面 → (盤 81, 持ち駒 14, 王手)。dainagon_ref.compact_ref と同じ。"""
    import cshogi
    turn = cb.turn
    flip = turn == cshogi.WHITE
    board = np.full(N_SQ, EMPTY, dtype=np.uint8)
    for sq in range(N_SQ):
        p = cb.pieces[sq]
        if p == 0:
            continue
        y, x = sq % 9, 8 - sq // 9
        ty, tx = (8 - y, 8 - x) if flip else (y, x)
        idx = _CSHOGI_PT_TO_CH[cshogi.piece_to_piece_type(p)]
        own = (1 if p >= 17 else 0) == turn
        board[ty * 9 + tx] = idx if own else idx + OPP
    hands = np.array(list(cb.pieces_in_hand[turn]) + list(cb.pieces_in_hand[1 - turn]), dtype=np.uint8)
    return board, hands, int(bool(cb.is_check()))


def from_hcpe(arr):
    """cshogi の HuffmanCodedPosAndEval の配列を dict にする（検証用）。

    "board" uint8 (n, 81)、"hands" uint8 (n, 14)、"check" uint8 (n,)、"move" int64 (n,)（2187 の番号）、
    "score" float32 (n,)（eval そのまま。手番側の視点）、"result" float32 (n,)（手番から見た 0・0.5・1）。
    指し手が変換できない局面は除く。gameResult は 0 引き分け・1 先手勝ち・2 後手勝ち。
    """
    import cshogi
    boards, hands, checks, moves, scores, results = [], [], [], [], [], []
    cb = cshogi.Board()
    for i in range(len(arr)):
        cb.set_hcp(arr["hcp"][i])
        m = cb.move_from_move16(int(arr["bestMove16"][i]))
        try:
            usi = cshogi.move_to_usi(m)
        except Exception:
            continue
        usi_rot = _rotate_usi(usi) if cb.turn == cshogi.WHITE else usi
        label = _usi_to_action(usi_rot)
        if label is None:
            continue
        b, h, c = _compact_cshogi(cb)
        gr = int(arr["gameResult"][i])
        if gr == 0:
            r = 0.5
        else:
            r = 1.0 if (gr == 1) == (cb.turn == cshogi.BLACK) else 0.0
        boards.append(b)
        hands.append(h)
        checks.append(c)
        moves.append(label)
        scores.append(float(arr["eval"][i]))
        results.append(r)
    n = len(moves)
    return {
        "board": np.array(boards, dtype=np.uint8).reshape(n, N_SQ),
        "hands": np.array(hands, dtype=np.uint8).reshape(n, 2 * N_HAND),
        "check": np.array(checks, dtype=np.uint8),
        "move": np.array(moves, dtype=np.int64),
        "score": np.array(scores, dtype=np.float32),
        "result": np.array(results, dtype=np.float32),
    }
