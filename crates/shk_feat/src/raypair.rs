//! RayPair 特徴量（設計書 §F2）。`reference/raypair_ref.py` と同じ結果を出す。

use crate::position::Position;

const A2_BASE: u32 = 2268;
const B1_BASE: u32 = 2344;
/// 1 領域ぶんの B1 の大きさ（28*28*4*4）
const B1_LOCAL: u32 = 12544;
/// 駒箱ではなく持ち駒の上限枚数（§F1.4）
const HAND_MAX: [u8; 7] = [18, 4, 4, 4, 4, 2, 2];
const HAND_OFF: [u32; 7] = [0, 18, 22, 26, 30, 34, 36];

/// version の領域数。0 = RayPair_v0（1 領域）、1 = RayPair_v1（9 領域）。
fn region_count(version: u8) -> Result<u32, String> {
    match version {
        0 => Ok(1),
        1 => Ok(9),
        _ => Err(format!("未知の version: {version}")),
    }
}

/// 特徴量の次元数（§F2.1 の 15,000、§F2.2 の 115,352）。
pub fn num_features(version: u8) -> Result<u32, String> {
    let regions = region_count(version)?;
    Ok(B1_BASE + regions * B1_LOCAL + 112)
}

/// 視点 `persp`（0 = 先手、1 = 後手）から見た特徴の多重集合を昇順で返す（重複を含む）。
pub fn features(pos: &Position, persp: u8, version: u8) -> Result<Vec<u32>, String> {
    let regions = region_count(version)?;
    let board = view_board(pos, persp);
    let hands = [pos.hands[persp as usize], pos.hands[1 - persp as usize]];
    let mut out = Vec::with_capacity(208);

    // A1：盤上の駒（玉を含む）
    for (y, row) in board.iter().enumerate() {
        for (x, cell) in row.iter().enumerate() {
            if let Some(pc) = cell {
                out.push(*pc as u32 * 81 + (y * 9 + x) as u32);
            }
        }
    }

    // A2：持ち駒（side 0 = 視点側、1 = 相手側）
    for (side, hand) in hands.iter().enumerate() {
        for h in 0..7 {
            if hand[h] > HAND_MAX[h] {
                return Err(format!("持ち駒の枚数が上限を超えている: h={h}"));
            }
            for k in 0..hand[h] as u32 {
                out.push(A2_BASE + side as u32 * 38 + HAND_OFF[h] + k);
            }
        }
    }

    // B1：ライン上で隣接する駒の対（§F2.1 の走査規則）
    for axis in 0..4u32 {
        for line in lines(axis) {
            let occ: Vec<(usize, usize, u32)> = line
                .iter()
                .filter_map(|&(y, x)| board[y][x].map(|pc| (y, x, pc as u32)))
                .collect();
            for pair in occ.windows(2) {
                let (ya, xa, pa) = pair[0];
                let (yb, xb, pb) = pair[1];
                let d = if axis == 1 { xb - xa } else { yb - ya };
                let local = pa * 448 + pb * 16 + axis * 4 + db_of(d);
                let region = if version == 0 {
                    0
                } else {
                    ((ya / 3) * 3 + xa / 3) as u32
                };
                out.push(B1_BASE + region * B1_LOCAL + local);
            }
        }
    }

    // B2：成っていない桂の利き先にある駒
    let b2 = B1_BASE + regions * B1_LOCAL;
    for (y, row) in board.iter().enumerate() {
        for (x, cell) in row.iter().enumerate() {
            let pc = match cell {
                Some(pc) if *pc == 2 || *pc == 16 => *pc,
                _ => continue,
            };
            let s = if pc == 2 { 0 } else { 1 };
            let ty = if s == 0 { y as i32 - 2 } else { y as i32 + 2 };
            for (lr, tx) in [(0u32, x as i32 - 1), (1u32, x as i32 + 1)] {
                if !(0..9).contains(&ty) || !(0..9).contains(&tx) {
                    continue;
                }
                if let Some(pt) = board[ty as usize][tx as usize] {
                    out.push(b2 + s * 56 + pt as u32 * 2 + lr);
                }
            }
        }
    }

    out.sort_unstable();
    Ok(out)
}

/// 視点座標に変換した盤面。駒番号 `pc` は §F1.3（相手側は +14）。
fn view_board(pos: &Position, persp: u8) -> [[Option<u8>; 9]; 9] {
    let mut vb = [[None; 9]; 9];
    for y in 0..9 {
        for x in 0..9 {
            if let Some(p) = pos.board[y][x] {
                let (yy, xx) = if persp == 0 { (y, x) } else { (8 - y, 8 - x) };
                let side = if p.color == persp { 0 } else { 14 };
                vb[yy][xx] = Some(p.pt + side);
            }
        }
    }
    vb
}

/// 軸ごとのライン。各ラインのマスは走査方向（手前 → 奥）に並ぶ。
fn lines(axis: u32) -> Vec<Vec<(usize, usize)>> {
    let mut out = Vec::new();
    match axis {
        0 => {
            for x in 0..9usize {
                out.push((0..9usize).map(|y| (y, x)).collect());
            }
        }
        1 => {
            for y in 0..9usize {
                out.push((0..9usize).map(|x| (y, x)).collect());
            }
        }
        2 => {
            for c in -8i32..=8 {
                out.push(
                    (0..9i32)
                        .filter(|&y| (0..9).contains(&(y + c)))
                        .map(|y| (y as usize, (y + c) as usize))
                        .collect(),
                );
            }
        }
        _ => {
            for s in 0..=16i32 {
                out.push(
                    (0..9i32)
                        .filter(|&y| (0..9).contains(&(s - y)))
                        .map(|y| (y as usize, (s - y) as usize))
                        .collect(),
                );
            }
        }
    }
    out
}

/// 距離 d（1 以上）を距離区分 db に変換する（§F2.1 の表）。
fn db_of(d: usize) -> u32 {
    match d {
        1 => 0,
        2 => 1,
        3 | 4 => 2,
        _ => 3,
    }
}
