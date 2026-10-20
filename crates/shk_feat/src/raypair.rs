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

/// 距離 d（1..=8）を距離区分 db に変換する表（§F2.1）。添字は `d & 15` で引く（9 以上は使わない）。
const DB: [u32; 16] = [0, 0, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3];

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

/// 局面の駒の一覧。盤の昇順（`sq = y * 9 + x` の小さい順）に並ぶ。
/// 両視点の特徴を作るとき、局面ごとに1回だけ作って使い回す。
pub struct Pieces {
    /// `(sq, color, pt)`。`sq` は手番に依らない盤の番号
    items: [(u8, u8, u8); 81],
    n: usize,
    hands: [[u8; 7]; 2],
}

impl Pieces {
    pub fn new(pos: &Position) -> Self {
        let mut items = [(0, 0, 0); 81];
        let mut n = 0;
        for y in 0..9 {
            for x in 0..9 {
                if let Some(p) = pos.board[y][x] {
                    items[n] = ((y * 9 + x) as u8, p.color, p.pt);
                    n += 1;
                }
            }
        }
        Pieces {
            items,
            n,
            hands: pos.hands,
        }
    }
}

/// `features` と同じ結果を `out` に書く（`out` の中身は消える）。`out` を使い回せば確保は起きない。
pub fn features_into(pos: &Position, persp: u8, version: u8, out: &mut Vec<u32>) -> Result<(), String> {
    features_from(&Pieces::new(pos), persp, version, out)
}

/// `Pieces` から視点 `persp` の特徴を `out` に書く（`features_into` と同じ結果）。
pub fn features_from(pieces: &Pieces, persp: u8, version: u8, out: &mut Vec<u32>) -> Result<(), String> {
    let regions = region_count(version)?;
    let n = pieces.n;

    // 視点座標の駒。後手視点は盤を 180° 回すので、昇順に並べ直すには列を逆から読む。
    let mut occ = [(0u8, 0u8); 81];
    let mut board = [EMPTY; 81];
    for i in 0..n {
        let (sq, color, pt) = if persp == 0 {
            pieces.items[i]
        } else {
            pieces.items[n - 1 - i]
        };
        let vsq = if persp == 0 { sq } else { 80 - sq };
        // 駒番号 `pc` は §F1.3（相手側は +14）
        let code = pt + if color == persp { 0 } else { 14 };
        occ[i] = (vsq, code);
        board[vsq as usize] = code;
    }
    let occ = &occ[..n];
    out.clear();

    // A1：盤上の駒（玉を含む）
    for &(sq, pc) in occ {
        out.push(pc as u32 * 81 + sq as u32);
    }

    // A2：持ち駒（side 0 = 視点側、1 = 相手側）
    let hands = [pieces.hands[persp as usize], pieces.hands[1 - persp as usize]];
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
    // 駒を昇順に見ると、どの軸のラインも手前 → 奥の順に現れる。
    // 各ラインについて直前の駒を覚えておき、今の駒との組を出す。
    // 組があるかどうかで分岐すると予測が外れやすいので、組でない場合も書いて、件数だけ数えない。
    let v1 = version == 1;
    let mut pairs = [0u32; 4 * 81];
    let mut np = 0;
    let mut last = [NONE; N_LINES];
    for &(sq, pb) in occ {
        let s = sq as usize;
        for axis in 0..4 {
            let line = LINES.of_sq[axis][s] as usize;
            let prev = last[line];
            let valid = prev != NONE;
            // 直前の駒がないときは自分自身を参照する（書いた値は数えないので使わない）
            let sa = if valid { prev as usize } else { s };
            let (ya, xa) = (sa / 9, sa % 9);
            let d = if axis == 1 {
                (s % 9).wrapping_sub(xa)
            } else {
                (s / 9).wrapping_sub(ya)
            };
            let pa = board[sa] as u32;
            let local = pa * 448 + pb as u32 * 16 + axis as u32 * 4 + DB[d & 15];
            let region = if v1 { ((ya / 3) * 3 + xa / 3) as u32 } else { 0 };
            pairs[np] = B1_BASE + region * B1_LOCAL + local;
            np += valid as usize;
            last[line] = sq;
        }
    }
    out.extend_from_slice(&pairs[..np]);

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
