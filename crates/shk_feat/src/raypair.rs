//! RayPair 特徴量（設計書 §F2）。`reference/raypair_ref.py` と同じ結果を出す。
//!
//! 計算は作業領域を使い回す（`features_into`）。ラインは定数表で持ち、局面ごとに確保しない。

use crate::position::Position;

const A2_BASE: u32 = 2268;
const B1_BASE: u32 = 2344;
/// 1 領域ぶんの B1 の大きさ（28*28*4*4）
const B1_LOCAL: u32 = 12544;
/// 駒箱ではなく持ち駒の上限枚数（§F1.4）
const HAND_MAX: [u8; 7] = [18, 4, 4, 4, 4, 2, 2];
const HAND_OFF: [u32; 7] = [0, 18, 22, 26, 30, 34, 36];

/// 視点座標の盤で、駒のないマス
const EMPTY: u8 = u8::MAX;

/// 距離 d（1..=8）を距離区分 db に変換する表（§F2.1）。添字 0 は使わない。
const DB: [u32; 9] = [0, 0, 1, 2, 2, 3, 3, 3, 3];

/// ラインの数（行 9、列 9、斜め 17、斜め 17）
const N_LINES: usize = 52;

/// 軸ごとのライン（§F2.1 の走査規則）。マスは `y * 9 + x` で、並びは手前 → 奥。
struct Lines {
    cells: [[u8; 9]; N_LINES],
    len: [u8; N_LINES],
    axis: [u8; N_LINES],
}

static LINES: Lines = build_lines();

const fn build_lines() -> Lines {
    let mut out = Lines {
        cells: [[0; 9]; N_LINES],
        len: [0; N_LINES],
        axis: [0; N_LINES],
    };
    let mut li = 0;
    // 軸 0：x を固定し、y を 0..9 で走る
    let mut x = 0;
    while x < 9 {
        let mut y = 0;
        while y < 9 {
            out.cells[li][y] = (y * 9 + x) as u8;
            y += 1;
        }
        out.len[li] = 9;
        out.axis[li] = 0;
        li += 1;
        x += 1;
    }
    // 軸 1：y を固定し、x を 0..9 で走る
    let mut y = 0;
    while y < 9 {
        let mut x = 0;
        while x < 9 {
            out.cells[li][x] = (y * 9 + x) as u8;
            x += 1;
        }
        out.len[li] = 9;
        out.axis[li] = 1;
        li += 1;
        y += 1;
    }
    // 軸 2：x − y = c（c は −8..=8）。y の増える向きに並べる
    let mut c: i32 = -8;
    while c <= 8 {
        let mut k = 0;
        let mut y: i32 = 0;
        while y < 9 {
            let x = y + c;
            if x >= 0 && x < 9 {
                out.cells[li][k] = (y * 9 + x) as u8;
                k += 1;
            }
            y += 1;
        }
        out.len[li] = k as u8;
        out.axis[li] = 2;
        li += 1;
        c += 1;
    }
    // 軸 3：x + y = s（s は 0..=16）。y の増える向きに並べる
    let mut s: i32 = 0;
    while s <= 16 {
        let mut k = 0;
        let mut y: i32 = 0;
        while y < 9 {
            let x = s - y;
            if x >= 0 && x < 9 {
                out.cells[li][k] = (y * 9 + x) as u8;
                k += 1;
            }
            y += 1;
        }
        out.len[li] = k as u8;
        out.axis[li] = 3;
        li += 1;
        s += 1;
    }
    out
}

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
    let mut out = Vec::with_capacity(208);
    features_into(pos, persp, version, &mut out)?;
    Ok(out)
}

/// `features` と同じ結果を `out` に書く（`out` の中身は消える）。`out` を使い回せば確保は起きない。
pub fn features_into(pos: &Position, persp: u8, version: u8, out: &mut Vec<u32>) -> Result<(), String> {
    let regions = region_count(version)?;
    let board = view_of(pos, persp);
    out.clear();

    // A1：盤上の駒（玉を含む）
    for (sq, &pc) in board.iter().enumerate() {
        if pc != EMPTY {
            out.push(pc as u32 * 81 + sq as u32);
        }
    }

    // A2：持ち駒（side 0 = 視点側、1 = 相手側）
    let hands = [pos.hands[persp as usize], pos.hands[1 - persp as usize]];
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

    // B1：ライン上で隣接する駒の対（§F2.1 の走査規則）。
    // 駒のあるマスだけを順に見て、直前の駒との組を出す。
    let v1 = version == 1;
    for li in 0..N_LINES {
        let axis = LINES.axis[li] as u32;
        let cells = &LINES.cells[li][..LINES.len[li] as usize];
        let mut prev: Option<(usize, u32)> = None;
        for &sq in cells {
            let sq = sq as usize;
            let pb = board[sq];
            if pb == EMPTY {
                continue;
            }
            if let Some((sa, pa)) = prev {
                let (ya, xa) = (sa / 9, sa % 9);
                let d = if axis == 1 {
                    sq % 9 - xa
                } else {
                    sq / 9 - ya
                };
                let local = pa * 448 + pb as u32 * 16 + axis * 4 + DB[d];
                let region = if v1 { ((ya / 3) * 3 + xa / 3) as u32 } else { 0 };
                out.push(B1_BASE + region * B1_LOCAL + local);
            }
            prev = Some((sq, pb as u32));
        }
    }

    // B2：成っていない桂の利き先にある駒
    let b2 = B1_BASE + regions * B1_LOCAL;
    for (sq, &pc) in board.iter().enumerate() {
        if pc != 2 && pc != 16 {
            continue;
        }
        let s: u32 = if pc == 2 { 0 } else { 1 };
        let (y, x) = ((sq / 9) as i32, (sq % 9) as i32);
        let ty = if s == 0 { y - 2 } else { y + 2 };
        for (lr, tx) in [(0u32, x - 1), (1u32, x + 1)] {
            if !(0..9).contains(&ty) || !(0..9).contains(&tx) {
                continue;
            }
            let target = board[(ty * 9 + tx) as usize];
            if target != EMPTY {
                out.push(b2 + s * 56 + target as u32 * 2 + lr);
            }
        }
    }

    out.sort_unstable();
    Ok(())
}

/// 視点座標に変換した盤面（`sq = y * 9 + x`）。駒番号 `pc` は §F1.3（相手側は +14）。
fn view_of(pos: &Position, persp: u8) -> [u8; 81] {
    let mut vb = [EMPTY; 81];
    for y in 0..9 {
        for x in 0..9 {
            if let Some(p) = pos.board[y][x] {
                // 視点が後手のときは盤を 180° 回す（(y, x) → (8 − y, 8 − x)）
                let sq = if persp == 0 { y * 9 + x } else { 80 - (y * 9 + x) };
                let side = if p.color == persp { 0 } else { 14 };
                vb[sq] = p.pt + side;
            }
        }
    }
    vb
}
