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
/// ライン上に直前の駒がないことを表す
const NONE: u8 = u8::MAX;

/// 距離 d（1..=8）を距離区分 db に変換する表（§F2.1）。添字 0 は使わない。
const DB: [u32; 9] = [0, 0, 1, 2, 2, 3, 3, 3, 3];

/// ラインの数（行 9、列 9、斜め 17、斜め 17）
const N_LINES: usize = 52;

/// 軸ごとのライン（§F2.1 の走査規則）。マスは `y * 9 + x` で、並びは手前 → 奥。
struct Lines {
    cells: [[u8; 9]; N_LINES],
    len: [u8; N_LINES],
    axis: [u8; N_LINES],
    /// `of_sq[axis][sq]` は、そのマスを通る軸 `axis` のライン番号
    of_sq: [[u8; 81]; 4],
}

static LINES: Lines = build_lines();

const fn build_lines() -> Lines {
    let mut out = Lines {
        cells: [[0; 9]; N_LINES],
        len: [0; N_LINES],
        axis: [0; N_LINES],
        of_sq: [[0; 81]; 4],
    };
    let mut li = 0;
    // 軸 0：x を固定し、y を 0..9 で走る
    let mut x = 0;
    while x < 9 {
        let mut y = 0;
        while y < 9 {
            out.cells[li][y] = (y * 9 + x) as u8;
            out.of_sq[0][y * 9 + x] = li as u8;
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
            out.of_sq[1][y * 9 + x] = li as u8;
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
                out.of_sq[2][(y * 9 + x) as usize] = li as u8;
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
                out.of_sq[3][(y * 9 + x) as usize] = li as u8;
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
    // 駒のあるマスを、視点座標の昇順に並べたもの（ラインの走査順と同じ向き）
    let mut occ = [(0u8, 0u8); 81];
    let mut n_occ = 0;
    for (sq, &pc) in board.iter().enumerate() {
        if pc != EMPTY {
            occ[n_occ] = (sq as u8, pc);
            n_occ += 1;
        }
    }
    let occ = &occ[..n_occ];
    out.clear();

    // A1：盤上の駒（玉を含む）
    for &(sq, pc) in occ {
        out.push(pc as u32 * 81 + sq as u32);
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
    // 占有マスを昇順に見ると、どの軸のラインも手前 → 奥の順に現れる。
    // 各ラインについて直前の駒を覚えておき、今の駒との組を出す。
    let v1 = version == 1;
    let mut last = [NONE; N_LINES];
    for &(sq, pb) in occ {
        let s = sq as usize;
        for axis in 0..4 {
            let line = LINES.of_sq[axis][s] as usize;
            let prev = last[line];
            if prev != NONE {
                let sa = prev as usize;
                let pa = board[sa] as u32;
                let (ya, xa) = (sa / 9, sa % 9);
                let d = if axis == 1 { s % 9 - xa } else { s / 9 - ya };
                let local = pa * 448 + pb as u32 * 16 + axis as u32 * 4 + DB[d];
                let region = if v1 { ((ya / 3) * 3 + xa / 3) as u32 } else { 0 };
                out.push(B1_BASE + region * B1_LOCAL + local);
            }
            last[line] = sq;
        }
    }

    // B2：成っていない桂の利き先にある駒
    let b2 = B1_BASE + regions * B1_LOCAL;
    for &(sq, pc) in occ {
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

    sort_features(out);
    Ok(())
}

/// 並べ替えの作業用の大きさ。これを超える特徴数は通常の並べ替えを使う。
const SORT_STACK: usize = 256;

/// 特徴の値は 2^18 未満なので、6 ビットずつ 3 回の基数ソートで昇順にする。
fn sort_features(v: &mut Vec<u32>) {
    let n = v.len();
    if n > SORT_STACK || v.iter().any(|&x| x >= 1 << 18) {
        v.sort_unstable();
        return;
    }
    let mut tmp = [0u32; SORT_STACK];
    let tmp = &mut tmp[..n];
    radix_pass(v, tmp, 0);
    radix_pass(tmp, v, 6);
    radix_pass(v, tmp, 12);
    v.copy_from_slice(tmp);
}

/// `src` を `(x >> shift) & 63` の順に `dst` へ安定に並べ替える。
fn radix_pass(src: &[u32], dst: &mut [u32], shift: u32) {
    let mut start = [0usize; 64];
    for &x in src {
        start[((x >> shift) & 63) as usize] += 1;
    }
    let mut sum = 0;
    for c in start.iter_mut() {
        let k = *c;
        *c = sum;
        sum += k;
    }
    for &x in src {
        let b = ((x >> shift) & 63) as usize;
        dst[start[b]] = x;
        start[b] += 1;
    }
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
