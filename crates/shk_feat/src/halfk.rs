//! HalfK 系の特徴量（§S11・§S12）。やねうら王の HalfKP・HalfKA 系と同じ番号（tests/nnue_ref.py）。
//!
//! | 版    | 特徴              | 玉を含む | 玉の番号の統一 | 左右反転 |
//! |-------|-------------------|----------|----------------|----------|
//! | 2・3  | HalfKP            | いいえ   | —              | いいえ   |
//! | 4・5  | HalfKA1           | はい     | いいえ         | いいえ   |
//! | 6・7  | HalfKA2           | はい     | はい           | いいえ   |
//! | 8・9  | HalfKA_hm1        | はい     | いいえ         | はい     |
//! | 10・11| HalfKA_hm2        | はい     | はい           | はい     |
//!
//! 奇数の版は因数分解つき。偶数の版の各特徴 f に、玉の位置に関係ない特徴 `N + (f mod P)` を足したもの
//! （N は元の次元、P は 1 つの玉の位置あたりの駒の番号の数）。
//!
//! マス番号は `筋の番号 × 9 + 段の番号`（1一 = 0、9九 = 80）。`Position` の `board[y][x]` は
//! y が段、x が 9筋 = 0 の並びなので、マスは `(8 − x) × 9 + y`。後手の視点では盤を 180° 回して
//! `80 − マス` を使う。

use crate::position::Position;

/// 盤上の駒と持ち駒の番号の数（玉の位置ごと）
const FE_END: u32 = 1548;
/// 味方の玉の番号の基準（HalfKA 系）
const F_KING: u32 = 1548;
/// 相手の玉の番号の基準（HalfKA 系）。HalfKA2 系では F_KING と同じ番号にまとめる
const E_KING: u32 = 1629;
/// 玉以外の駒の総数（盤上・持ち駒・駒箱の合計）
const NON_KING_PIECES: usize = 38;
/// 駒種 KING（`position::PIECE_SFEN` の添字）
const KING: u8 = 7;
/// 駒種 pt → 駒の種類 k（歩0 香1 桂2 銀3 金4 角5 馬6 飛7 龍8）。玉（pt 7）は使わない。
const BOARD_K: [u32; 14] = [0, 1, 2, 3, 4, 5, 7, u32::MAX, 4, 4, 4, 4, 6, 8];
/// 持ち駒の番号の基準。味方は歩香桂銀金角飛の順に 1・39・49・59・69・79・85、相手は 20・44・54・64・74・82・88。
const HAND_F: [u32; 7] = [1, 39, 49, 59, 69, 79, 85];
const HAND_E: [u32; 7] = [20, 44, 54, 64, 74, 82, 88];
/// 持ち駒の上限枚数（歩 18、香・桂・銀・金 4、角・飛 2）。RayPair と同じ（§F1.4）。
const HAND_MAX: [u8; 7] = [18, 4, 4, 4, 4, 2, 2];

/// 特徴の種類（版から決まる）。
#[derive(Clone, Copy)]
struct Kind {
    /// 1 つの玉の位置あたりの駒の番号の数（P）
    plane: u32,
    /// 玉の位置の数（81、左右反転するものは 45）
    squares: u32,
    /// 玉を特徴に含むか（HalfKA 系）
    with_king: bool,
    /// 相手の玉を味方の玉と同じ番号にするか（HalfKA2 系）
    merge_kings: bool,
    /// 玉が 6〜9 筋なら盤を左右に反転するか（hm）
    mirror: bool,
    /// 因数分解つき（奇数の版）
    factorized: bool,
}

fn kind(version: u8) -> Result<Kind, String> {
    let (plane, squares, with_king, merge_kings, mirror) = match version {
        2 | 3 => (FE_END, 81, false, false, false),
        4 | 5 => (E_KING + 81, 81, true, false, false),
        6 | 7 => (E_KING, 81, true, true, false),
        8 | 9 => (E_KING + 81, 45, true, false, true),
        10 | 11 => (E_KING, 45, true, true, true),
        _ => return Err(format!("未知の version: {version}")),
    };
    Ok(Kind { plane, squares, with_king, merge_kings, mirror, factorized: version % 2 == 1 })
}

/// 特徴の次元数（版 2〜11）。
pub fn num_features(version: u8) -> Result<u32, String> {
    let k = kind(version)?;
    Ok(k.plane * k.squares + if k.factorized { k.plane } else { 0 })
}

/// 視点 `persp`（0 = 先手、1 = 後手）から見た特徴を `out` に昇順で書く（`out` の中身は消える）。
pub fn features_into(pos: &Position, persp: u8, version: u8, out: &mut Vec<u32>) -> Result<(), String> {
    let k = kind(version)?;
    // 駒の番号（玉の位置を含まない）。玉以外は 38 枚、HalfKA 系は玉を含めて 40 枚。
    let mut bps = [0u32; 40];
    let mut n = 0usize;
    let mut ksq = None;
    for (y, row) in pos.board.iter().enumerate() {
        for (x, cell) in row.iter().enumerate() {
            let Some(p) = cell else { continue };
            let sq = sq_of(y, x);
            let vsq = if persp == 0 { sq } else { 80 - sq };
            let mine = p.color == persp;
            let id = if p.pt == KING {
                if mine {
                    ksq = Some(vsq);
                }
                if !k.with_king {
                    continue;
                }
                (if mine { F_KING } else { E_KING }) + vsq
            } else {
                board_id(p.color, persp, BOARD_K[p.pt as usize], sq)
            };
            push_piece(&mut bps, &mut n, id)?;
        }
    }
    let ksq = ksq.ok_or("玉が盤上に無い局面は特徴にできない")?;

    // 持ち駒：1枚目〜n枚目のそれぞれを1つの特徴とする
    for color in 0..2u8 {
        for h in 0..7 {
            let count = pos.hands[color as usize][h];
            if count > HAND_MAX[h] {
                return Err(format!("持ち駒の枚数が上限を超えている: h={h}"));
            }
            let start = if color == persp { HAND_F[h] } else { HAND_E[h] };
            for i in 0..count as u32 {
                push_piece(&mut bps, &mut n, start + i)?;
            }
        }
    }

    // 盤上にも持ち駒にも無い駒（駒箱の駒）は、駒の番号 0 として数える
    let total = if k.with_king { 40 } else { NON_KING_PIECES };
    if n > total {
        return Err(format!("玉以外の駒が多すぎる: {n}"));
    }
    bps[n..total].fill(0);

    let flip = k.mirror && ksq / 9 >= 5;
    let kk = if flip { mir(ksq) } else { ksq };
    out.clear();
    for &bp in &bps[..total] {
        let mut b = bp;
        if flip && b >= 90 {
            b = 90 + ((b - 90) / 81) * 81 + mir((b - 90) % 81);
        }
        if k.merge_kings && b >= E_KING {
            b -= 81;
        }
        out.push(kk * k.plane + b);
        if k.factorized {
            out.push(k.plane * k.squares + b);
        }
    }
    out.sort_unstable();
    Ok(())
}

/// 駒を 1 つ数える（番号を入れる）。40 枚を超えたら誤り。
fn push_piece(bps: &mut [u32; 40], n: &mut usize, id: u32) -> Result<(), String> {
    if *n >= bps.len() {
        return Err("駒が多すぎる".into());
    }
    bps[*n] = id;
    *n += 1;
    Ok(())
}

/// 盤上の玉以外の駒の番号。`color` 0 = 先手、`k` は駒の種類、`sq` は盤の番号（cshogi と同じ）。
fn board_id(color: u8, persp: u8, k: u32, sq: u32) -> u32 {
    let f = 90 + 162 * k;
    if persp == 0 {
        (if color == 0 { f } else { f + 81 }) + sq
    } else {
        (if color == 1 { f } else { f + 81 }) + (80 - sq)
    }
}

/// 筋の番号 × 9 ＋ 段の番号（1一 = 0、9九 = 80）。
fn sq_of(y: usize, x: usize) -> u32 {
    ((8 - x) * 9 + y) as u32
}

/// 左右反転のマス（筋の番号を 8 − 筋 にする）。
fn mir(sq: u32) -> u32 {
    (8 - sq / 9) * 9 + sq % 9
}
