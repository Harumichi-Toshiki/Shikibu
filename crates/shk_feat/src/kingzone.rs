//! 玉の周りのブロック（区間 C、§S12.3）。版 12（raypair0k）・13（raypair1k）で RayPair の後ろに付け足す。
//!
//! 両方の玉の周りの 5×5 から中心を除いた 24 マスについて、その中身を番号にする（tests/kingzone_ref.py）。
//! 座標と駒の番号は RayPair と同じ（視点座標 `y × 9 + x`、駒番号 pc は視点側 0〜13、相手側 +14）。
//! 盤の外は pc 28、空きマスは何も足さない。

use crate::position::Position;

/// 区間 C の次元（2 玉 × 24 マス × 29 通り）
pub const DIM: u32 = 2 * 24 * 29;
/// 空きマス
const EMPTY: u8 = u8::MAX;
/// 盤の外
const OFF_BOARD: u32 = 28;
/// 相手側の駒番号の加算（視点側 0〜13 → 相手側 14〜27）
const ENEMY: u8 = 14;
/// 玉の 1 種類あたりの要素数（24 マス × 29）
const PER_KING: u32 = 24 * 29;

/// 視点 `persp` から見た玉の周りのブロックの番号を `out` に足す（`base` は区間 C の先頭の番号）。
pub fn features_into(pos: &Position, persp: u8, base: u32, out: &mut Vec<u32>) -> Result<(), String> {
    let mut vb = [EMPTY; 81];
    let mut kings = [None; 2];
    for (y, row) in pos.board.iter().enumerate() {
        for (x, cell) in row.iter().enumerate() {
            let Some(p) = cell else { continue };
            let (vy, vx) = if persp == 0 { (y, x) } else { (8 - y, 8 - x) };
            let vsq = vy * 9 + vx;
            let own = p.color == persp;
            vb[vsq] = p.pt + if own { 0 } else { ENEMY };
            if p.pt == 7 {
                // 0 = 味方の玉、1 = 相手の玉
                kings[(!own) as usize] = Some(vsq as i32);
            }
        }
    }
    for (kk, king) in kings.iter().enumerate() {
        let ksq = king.ok_or("玉が盤上に無い局面は特徴にできない")?;
        let (ky, kx) = (ksq / 9, ksq % 9);
        // o は 0〜23（中心を除いて、dy → dx の順）
        let mut o = 0u32;
        for dy in -2..=2i32 {
            for dx in -2..=2i32 {
                if dy == 0 && dx == 0 {
                    continue;
                }
                let (y, x) = (ky + dy, kx + dx);
                let c = if !(0..9).contains(&y) || !(0..9).contains(&x) {
                    OFF_BOARD
                } else {
                    match vb[(y * 9 + x) as usize] {
                        EMPTY => {
                            o += 1;
                            continue;
                        }
                        pc => pc as u32,
                    }
                };
                out.push(base + kk as u32 * PER_KING + o * 29 + c);
                o += 1;
            }
        }
    }
    Ok(())
}
