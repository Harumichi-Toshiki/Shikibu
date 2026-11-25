"""変更禁止（PROTECTED）。Shikibu M11（DL の学習、設計書 v1.9 §S14 DL1 の前半）の参照実装。

Dainagon の学習器（持ち主の作。公開してよいと持ち主が決めた）から、55 チャンネルの特徴量・指し手の 2187 の番号・
Binned Value（HL-Gauss）の教師を、そのまま写したもの。遅い（Python のループ）。正しさの基準にだけ使う。

  make_features_v16_fast(cb, is_white) : Dainagon の features.py と同じ（ビット単位で同じ出力）
  usi_to_action(usi)                  : Dainagon の train_rocm.py と同じ（27 方向 × 81 マス）
  rotate_usi(usi)                     : 後手番の指し手を 180° 回す（train_rocm.py の _rotate_move）
  make_hl_gauss_target(p)             : train_rocm.py と同じ（128 ビン、σ = 0.75 / 128）

M11 で足す「詰めた局面」（compact）の正解：
  compact_ref(cb) -> (board, hands, check)
    board : uint8 (81,)。手番側から見た座標（後手番なら盤を 180° 回す）のマス ty*9+tx に、
            make_features_v16_fast のチャンネル番号 0〜27（手番側の駒 0〜13、相手の駒 14〜27）。空きは 28。
    hands : uint8 (14,)。[手番側の 歩 香 桂 銀 金 角 飛, 相手の 歩 香 桂 銀 金 角 飛] の枚数。
    check : 0 か 1。手番側が王手されているか（cb.is_check()）。
  label_from_psv_move(cb, move16_psv) -> int。PSV の指し手（やねうら王の 16 ビット）を 2187 の番号にする。
            変換できない指し手（指し手なし・方向が不正など）は None。
"""
import math

import numpy as np
import cshogi

INPUT_CHANNELS = 55
EMPTY = 28

CSHOGI_PT_TO_SHOGI_IDX = np.array([0, 0, 1, 2, 3, 5, 6, 4, 7, 8, 9, 10, 11, 12, 13], dtype=np.int32)

ROOK_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BISHOP_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

CSHOGI_SQ_TO_Y = np.zeros(81, dtype=np.int32)
CSHOGI_SQ_TO_X = np.zeros(81, dtype=np.int32)
for _sq in range(81):
    _f, _r = divmod(_sq, 9)
    CSHOGI_SQ_TO_Y[_sq] = _r
    CSHOGI_SQ_TO_X[_sq] = 8 - _f

RAMP = np.arange(9, dtype=np.float32) / 8.0


def make_features_v16_fast(cb, is_white):
    """Dainagon の features.py と同じ。"""
    features = np.zeros((INPUT_CHANNELS, 9, 9), dtype=np.float32)
    turn = cb.turn
    opp_turn = 1 - turn
    pieces = cb.pieces
    occupied = np.zeros((9, 9), dtype=bool)

    for cshogi_sq in range(81):
        p = pieces[cshogi_sq]
        if p == 0:
            continue
        y = CSHOGI_SQ_TO_Y[cshogi_sq]
        x = CSHOGI_SQ_TO_X[cshogi_sq]
        occupied[y, x] = True
        p_color = 1 if p >= 17 else 0
        is_self = (p_color == turn)
        pt_idx = CSHOGI_PT_TO_SHOGI_IDX[cshogi.piece_to_piece_type(p)]
        idx = pt_idx if is_self else (pt_idx + 14)
        ty = (8 - y) if is_white else y
        tx = (8 - x) if is_white else x
        features[idx, ty, tx] = 1.0

    hand_self = cb.pieces_in_hand[turn]
    for i in range(7):
        if hand_self[i] > 0:
            features[28 + i, :, :] = hand_self[i] / 10.0
    hand_opp = cb.pieces_in_hand[opp_turn]
    for i in range(7):
        if hand_opp[i] > 0:
            features[35 + i, :, :] = hand_opp[i] / 10.0

    for cshogi_sq in range(81):
        p = pieces[cshogi_sq]
        if p == 0:
            continue
        pt = cshogi.piece_to_piece_type(p)
        if pt not in (5, 6, 13, 14):
            continue
        is_self = ((1 if p >= 17 else 0) == turn)
        is_rook = (pt == 6 or pt == 14)
        ch = (42 if is_rook else 43) if is_self else (44 if is_rook else 45)
        y0 = CSHOGI_SQ_TO_Y[cshogi_sq]
        x0 = CSHOGI_SQ_TO_X[cshogi_sq]
        if pt == 6:
            dirs, single = ROOK_DIRS, ()
        elif pt == 5:
            dirs, single = BISHOP_DIRS, ()
        elif pt == 14:
            dirs, single = ROOK_DIRS, BISHOP_DIRS
        else:
            dirs, single = BISHOP_DIRS, ROOK_DIRS
        for dy, dx in dirs:
            y, x = y0 + dy, x0 + dx
            while 0 <= y < 9 and 0 <= x < 9:
                ty = (8 - y) if is_white else y
                tx = (8 - x) if is_white else x
                features[ch, ty, tx] = 1.0
                if occupied[y, x]:
                    break
                y += dy
                x += dx
        for dy, dx in single:
            y, x = y0 + dy, x0 + dx
            if 0 <= y < 9 and 0 <= x < 9:
                ty = (8 - y) if is_white else y
                tx = (8 - x) if is_white else x
                features[ch, ty, tx] = 1.0

    for k_sq, base_ch in ((cb.king_square(turn), 46), (cb.king_square(opp_turn), 48)):
        if 0 <= k_sq < 81:
            ky = CSHOGI_SQ_TO_Y[k_sq]
            kx = CSHOGI_SQ_TO_X[k_sq]
            for r, r_ch in ((3, 1), (2, 0)):
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        ny, nx = ky + dy, kx + dx
                        if 0 <= ny < 9 and 0 <= nx < 9:
                            ty = (8 - ny) if is_white else ny
                            tx = (8 - nx) if is_white else nx
                            features[base_ch + r_ch, ty, tx] = 1.0

    if cb.is_check():
        features[50, :, :] = 1.0
    features[52, :, :] = RAMP
    features[53, :, :] = RAMP[:, None]
    return features


_TBL_NUM = str.maketrans("123456789", "987654321")
_TBL_ALP = str.maketrans("abcdefghi", "ihgfedcba")


def rotate_usi(u):
    if "*" in u:
        return u[0] + "*" + u[2:].translate(_TBL_NUM).translate(_TBL_ALP)
    return u[:2].translate(_TBL_NUM).translate(_TBL_ALP) + u[2:4].translate(_TBL_NUM).translate(_TBL_ALP) + u[4:]


def usi_to_action(usi):
    """Dainagon の train_rocm.py と同じ。不正な方向は ValueError。"""
    if usi[1] == '*':
        drop_dict = {'P': 0, 'L': 1, 'N': 2, 'S': 3, 'G': 4, 'B': 5, 'R': 6}
        dir_idx = 20 + drop_dict[usi[0]]
        to_x = 9 - int(usi[2])
        to_y = ord(usi[3]) - ord('a')
    else:
        from_x = 9 - int(usi[0])
        from_y = ord(usi[1]) - ord('a')
        to_x = 9 - int(usi[2])
        to_y = ord(usi[3]) - ord('a')
        dx = to_x - from_x
        dy = to_y - from_y
        if dy < 0 and dx == 0: dir_idx = 0
        elif dy < 0 and dx < 0 and dy == dx: dir_idx = 1
        elif dy < 0 and dx > 0 and -dy == dx: dir_idx = 2
        elif dy == 0 and dx < 0: dir_idx = 3
        elif dy == 0 and dx > 0: dir_idx = 4
        elif dy > 0 and dx == 0: dir_idx = 5
        elif dy > 0 and dx < 0 and dy == -dx: dir_idx = 6
        elif dy > 0 and dx > 0 and dy == dx: dir_idx = 7
        elif dy == -2 and dx == -1: dir_idx = 8
        elif dy == -2 and dx == 1: dir_idx = 9
        else:
            raise ValueError(f"Invalid move direction: {usi}")
        if len(usi) == 5 and usi[4] == '+':
            dir_idx += 10
    return dir_idx * 81 + to_y * 9 + to_x


def label_of_usi(cb, usi):
    """局面 cb の手番で指す usi を 2187 の番号に。変換できなければ None。"""
    if cb.turn == cshogi.WHITE:
        usi = rotate_usi(usi)
    try:
        return usi_to_action(usi)
    except (ValueError, KeyError, IndexError):
        return None


def label_from_psv_move(cb, move16_psv):
    m = cshogi.move16_from_psv(int(move16_psv))
    if m == 0:
        return None
    try:
        usi = cshogi.move_to_usi(m)
    except Exception:
        return None
    if not usi or usi in ("none", "resign", "win", "null"):
        return None
    return label_of_usi(cb, usi)


def compact_ref(cb):
    is_white = cb.turn == cshogi.WHITE
    f = make_features_v16_fast(cb, is_white)
    pieces = f[:28].reshape(28, 81)
    board = np.full(81, EMPTY, dtype=np.uint8)
    occ = pieces.sum(0) > 0
    board[occ] = pieces[:, occ].argmax(0).astype(np.uint8)
    hands = np.array(list(cb.pieces_in_hand[cb.turn]) + list(cb.pieces_in_hand[1 - cb.turn]), dtype=np.uint8)
    return board, hands, int(bool(cb.is_check()))


NUM_VALUE_BINS = 128
_VALUE_BIN_EDGES = [i / NUM_VALUE_BINS for i in range(NUM_VALUE_BINS + 1)]
_VALUE_BIN_SIGMA = 0.75 * (1.0 / NUM_VALUE_BINS)


def make_hl_gauss_target(target_prob):
    cdf = [0.5 * (1.0 + math.erf((e - target_prob) / (_VALUE_BIN_SIGMA * math.sqrt(2.0)))) for e in _VALUE_BIN_EDGES]
    probs = [cdf[i + 1] - cdf[i] for i in range(NUM_VALUE_BINS)]
    total = sum(probs)
    return np.array([p / total for p in probs], dtype=np.float32)
