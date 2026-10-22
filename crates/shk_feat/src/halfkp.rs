//! HalfKP 特徴量（比較実験用、§S11）。`tests/halfkp_ref.py` と同じ結果を出す。
//!
//! 版 2 は HalfKP（125,388 次元）、版 3 はその因数分解つき（126,936 次元）。
//! 版 3 は、版 2 の各特徴 f に、玉の位置に関係ない特徴 125,388 + (f mod 1,548) を足したもの。
//!
//! マス番号は `筋の番号 × 9 + 段の番号`（1一 = 0、9九 = 80）。`Position` の `board[y][x]` は
//! y が段、x が 9筋 = 0 の並びなので、マスは `(8 − x) × 9 + y`。後手の視点では盤を 180° 回して
//! `80 − マス` を使う。

use crate::position::Position;

/// 玉の位置ごとの駒の番号の数（盤上の駒 1,466 種＋持ち駒 82 種）
const FE_END: u32 = 1548;
/// 版 2 の次元（81 × 1548）
const HALFKP_DIM: u32 = 81 * FE_END;
/// 玉以外の駒の総数（盤上・持ち駒・駒箱の合計）
const NON_KING_PIECES: usize = 38;
/// 駒種 KING（`position::PIECE_SFEN` の添字）
const KING: u8 = 7;
/// 持ち駒の上限枚数（歩 18、香・桂・銀・金 4、角・飛 2）。RayPair と同じ（§F1.4）。
const HAND_MAX: [u8; 7] = [18, 4, 4, 4, 4, 2, 2];
/// 駒種 pt → 駒の種類 k（歩0 香1 桂2 銀3 金4 角5 馬6 飛7 龍8）。玉（pt 7）は使わない。
const BOARD_K: [u32; 14] = [0, 1, 2, 3, 4, 5, 7, u32::MAX, 4, 4, 4, 4, 6, 8];
/// 持ち駒の番号の基準。f は自分の駒、e は相手の駒（歩香桂銀金角飛の順）。
const HAND_F: [u32; 7] = [1, 39, 49, 59, 69, 79, 85];
const HAND_E: [u32; 7] = [20, 44, 54, 64, 74, 82, 88];

/// 特徴の次元数。`factorized` は版 3（因数分解つき）。
pub fn num_features(factorized: bool) -> u32 {
    HALFKP_DIM + if factorized { FE_END } else { 0 }
}

/// 視点 `persp`（0 = 先手、1 = 後手）から見た特徴の多重集合を昇順で返す（重複を含む）。
pub fn features(pos: &Position, persp: u8, factorized: bool) -> Result<Vec<u32>, String> {
    let mut out = Vec::with_capacity(160);
    features_into(pos, persp, factorized, &mut out)?;
    Ok(out)
}

/// `features` と同じ結果を `out` に書く（`out` の中身は消える）。
pub fn features_into(pos: &Position, persp: u8, factorized: bool, out: &mut Vec<u32>) -> Result<(), String> {
    let kings = king_squares(pos)?;
    // 玉のマス（視点の座標）。特徴の番号は玉のマス × 1548 ＋ 駒の番号。
    let ksq = if persp == 0 { kings[0] } else { 80 - kings[1] };
    let base = ksq * FE_END;
    out.clear();

    // 盤上の玉以外の駒
    for (y, row) in pos.board.iter().enumerate() {
        for (x, cell) in row.iter().enumerate() {
            let Some(p) = cell else { continue };
            if p.pt == KING {
                continue;
            }
            let id = board_id(p.color, BOARD_K[p.pt as usize], sq_of(y, x), persp);
            out.push(base + id);
        }
    }
    let mut n = out.len();

    // 持ち駒：1枚目〜n枚目のそれぞれを1つの特徴とする
    for color in 0..2 {
        for h in 0..7 {
            let count = pos.hands[color][h];
            if count > HAND_MAX[h] {
                return Err(format!("持ち駒の枚数が上限を超えている: h={h}"));
            }
            let start = if color == persp as usize { HAND_F[h] } else { HAND_E[h] };
            for i in 0..count as u32 {
                out.push(base + start + i);
            }
            n += count as usize;
        }
    }

    // 盤上にも持ち駒にも無い駒（駒箱の駒）は、駒の番号 0 として数える（38 枚まで）
    if n > NON_KING_PIECES {
        return Err(format!("玉以外の駒が38枚を超えている: {n}"));
    }
    out.extend(std::iter::repeat_n(base, NON_KING_PIECES - n));

    if factorized {
        let dim = out.len();
        for i in 0..dim {
            let f = out[i];
            out.push(HALFKP_DIM + f % FE_END);
        }
    }
    out.sort_unstable();
    Ok(())
}

/// 盤上の先手玉・後手玉のマス（参照実装のマス番号）。玉が無い・2枚以上は不正。
fn king_squares(pos: &Position) -> Result<[u32; 2], String> {
    let mut kings: [Option<u32>; 2] = [None, None];
    for (y, row) in pos.board.iter().enumerate() {
        for (x, cell) in row.iter().enumerate() {
            let Some(p) = cell else { continue };
            if p.pt != KING {
                continue;
            }
            let slot = &mut kings[p.color as usize];
            if slot.is_some() {
                return Err("同じ色の玉が2枚以上ある".into());
            }
            *slot = Some(sq_of(y, x));
        }
    }
    match kings {
        [Some(b), Some(w)] => Ok([b, w]),
        _ => Err("玉が盤上に無い局面は特徴にできない".into()),
    }
}

/// `Position` の `(y, x)` を参照実装のマス番号にする（1一 = 0、9九 = 80）。
fn sq_of(y: usize, x: usize) -> u32 {
    ((8 - x) * 9 + y) as u32
}

/// 盤上の駒の番号（玉の位置は含まない）。`color` 0 = 先手、`k` は駒の種類、`sq` は盤の番号。
fn board_id(color: u8, k: u32, sq: u32, persp: u8) -> u32 {
    let f = 90 + 162 * k;
    if persp == 0 {
        (if color == 0 { f } else { f + 81 }) + sq
    } else {
        (if color == 1 { f } else { f + 81 }) + (80 - sq)
    }
}
