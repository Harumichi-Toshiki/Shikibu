//! 特徴の版の振り分け。版 0・1 は RayPair（§F2）、版 2・3 は HalfKP（§S11）。

use crate::halfkp;
use crate::position::Position;
use crate::raypair;

/// 特徴量の次元数。
pub fn num_features(version: u8) -> Result<u32, String> {
    match version {
        0 | 1 => raypair::num_features(version),
        2 | 3 => Ok(halfkp::num_features(version == 3)),
        _ => Err(format!("未知の version: {version}")),
    }
}

/// 視点 `persp` から見た特徴の多重集合を昇順で返す（重複を含む）。
pub fn features(pos: &Position, persp: u8, version: u8) -> Result<Vec<u32>, String> {
    match version {
        0 | 1 => raypair::features(pos, persp, version),
        2 | 3 => halfkp::features(pos, persp, version == 3),
        _ => Err(format!("未知の version: {version}")),
    }
}

/// 1 視点ぶんの特徴を `buf` に書く（`buf` の中身は消える）。
/// 版 0・1 は、局面ごとに1回作った `pieces` を使い回す。版 2・3 は `pos` を直接使う。
pub fn side_features(
    pos: &Position,
    pieces: Option<&raypair::Pieces>,
    persp: u8,
    version: u8,
    buf: &mut Vec<u32>,
) -> Result<(), String> {
    match pieces {
        Some(p) => raypair::features_from(p, persp, version, buf),
        None => halfkp::features_into(pos, persp, version == 3, buf),
    }
}
