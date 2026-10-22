//! 特徴の版の振り分け（§F2・§S11・§S12）。
//!
//! 版 0・1 は RayPair（§F2）、2〜11 は HalfK 系（やねうら王と同じ。halfk.rs）、
//! 12・13 は RayPair ＋ 玉の周りのブロック（kingzone.rs）。

use crate::halfk;
use crate::kingzone;
use crate::position::Position;
use crate::raypair;

/// 特徴量の次元数。
pub fn num_features(version: u8) -> Result<u32, String> {
    match version {
        0 | 1 => raypair::num_features(version),
        2..=11 => halfk::num_features(version),
        12 | 13 => Ok(raypair::num_features(version - 12)? + kingzone::DIM),
        _ => Err(format!("未知の version: {version}")),
    }
}

/// 視点 `persp` から見た特徴の多重集合を昇順で返す（重複を含む）。
pub fn features(pos: &Position, persp: u8, version: u8) -> Result<Vec<u32>, String> {
    let mut out = Vec::with_capacity(256);
    side_features(pos, &raypair::Pieces::new(pos), persp, version, &mut out)?;
    Ok(out)
}

/// 1 視点ぶんの特徴を `buf` に書く（`buf` の中身は消える）。
/// `pieces` は `pos` の駒の一覧で、局面ごとに 1 回作って両視点で使い回す。
pub fn side_features(
    pos: &Position,
    pieces: &raypair::Pieces,
    persp: u8,
    version: u8,
    buf: &mut Vec<u32>,
) -> Result<(), String> {
    match version {
        0 | 1 => raypair::features_from(pieces, persp, version, buf),
        2..=11 => halfk::features_into(pos, persp, version, buf),
        12 | 13 => {
            raypair::features_from(pieces, persp, version - 12, buf)?;
            kingzone::features_into(pos, persp, raypair::num_features(version - 12)?, buf)?;
            buf.sort_unstable();
            Ok(())
        }
        _ => Err(format!("未知の version: {version}")),
    }
}

/// 玉のマス（RayPair の視点座標 `y × 9 + x`）。(手番側の玉（手番側の視点）, 相手の玉（相手の視点）)。
/// k3k3 の組の選び方（§S12.4）と、バッチの "kings" に使う。
pub fn king_squares(pos: &Position) -> Result<(u32, u32), String> {
    let mut res: [Option<u32>; 2] = [None, None];
    for (y, row) in pos.board.iter().enumerate() {
        for (x, cell) in row.iter().enumerate() {
            let Some(p) = cell else { continue };
            if p.pt != 7 {
                continue;
            }
            let (vy, vx) = if p.color == 0 { (y, x) } else { (8 - y, 8 - x) };
            let i = (p.color != pos.stm) as usize;
            if res[i].is_some() {
                return Err("同じ色の玉が2枚以上ある".into());
            }
            res[i] = Some((vy * 9 + vx) as u32);
        }
    }
    match res {
        [Some(a), Some(b)] => Ok((a, b)),
        _ => Err("玉が盤上に無い局面は特徴にできない".into()),
    }
}
