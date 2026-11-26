//! DL の学習用の「詰めた局面」（M11）。
//!
//! 1 局面を、盤 81 マス（手番側から見た座標の駒のチャンネル 0〜28）・持ち駒 14・王手 1・指し手の番号（2187）に
//! 詰める。55 チャンネルへの展開は Python の `shikibu.dl_features.expand`（GPU）で行う。
//! 正解は `reference/dainagon_ref.py`（`compact_ref`・`label_from_psv_move`）。
//!
//! 座標：`Position` の `board[y][x]`（y = 段、x = SFEN の並び）。手番側が後手なら 180° 回す。
//! 盤のマス番号は `ty × 9 + tx`（ty・tx は手番側から見た座標）。
//!
//! 指し手の番号：`dir × 81 + to_y × 9 + to_x`（dir は 0〜9 が移動、10〜19 が成り、20〜26 が打ち）。
//! PSV の指し手（16 ビット）は、cshogi の `move16_from_psv` と同じ読み方をする：
//! 下位 7 ビット＝移動先、次の 7 ビット＝移動元（打つときは駒の種類 1〜7）、bit 14＝打つ、bit 15＝成る。
//! マスの番号は PackedSfen と同じ（`(筋 − 1) × 9 + (段 − 1)`）。
//! 全 65536 通りのうち、移動先が盤内で、移動元が盤内か打ち駒の種類が 1〜7 のものについて、cshogi と一致することを確かめた（M11）。

use crate::position::{self, Piece, Position};
use crate::psv;

/// 盤の空きマスのチャンネル番号。
pub const EMPTY: u8 = 28;
/// 持ち駒の種類数（歩 香 桂 銀 金 角 飛）。
const N_HAND: usize = 7;

/// 1 局面を詰めたもの。
pub struct Compact {
    pub board: [u8; 81],
    pub hands: [u8; 2 * N_HAND],
    pub check: u8,
}

/// 盤のマス `(y, x)` を、手番側から見た座標の番号（ty × 9 + tx）にする。
fn view_index(y: usize, x: usize, stm: u8) -> usize {
    if stm == 0 {
        y * 9 + x
    } else {
        (8 - y) * 9 + (8 - x)
    }
}

/// PSV のマス番号 `s`（`(筋 − 1) × 9 + (段 − 1)`）を、手番側から見た `(y, x)` にする。
fn view_sq(s: usize, stm: u8) -> (i32, i32) {
    let y = (s % 9) as i32;
    let x = (8 - s / 9) as i32;
    if stm == 0 {
        (y, x)
    } else {
        (8 - y, 8 - x)
    }
}

/// 局面を詰める（盤・持ち駒・王手）。
pub fn compact(pos: &Position) -> Compact {
    let stm = pos.stm;
    let mut board = [EMPTY; 81];
    for (y, row) in pos.board.iter().enumerate() {
        for (x, cell) in row.iter().enumerate() {
            if let Some(Piece { color, pt }) = *cell {
                let ch = if color == stm { pt } else { pt + 14 };
                board[view_index(y, x, stm)] = ch;
            }
        }
    }
    let mut hands = [0u8; 2 * N_HAND];
    for h in 0..N_HAND {
        hands[h] = pos.hands[stm as usize][h];
        hands[N_HAND + h] = pos.hands[(1 - stm) as usize][h];
    }
    Compact {
        board,
        hands,
        check: is_in_check(pos) as u8,
    }
}

/// 手番側の玉が、相手の駒に利かれているか（王手）。
fn is_in_check(pos: &Position) -> bool {
    let stm = pos.stm;
    let mut king = None;
    for (y, row) in pos.board.iter().enumerate() {
        for (x, cell) in row.iter().enumerate() {
            if let Some(Piece { color, pt: 7 }) = *cell {
                if color == stm {
                    king = Some((y as i32, x as i32));
                }
            }
        }
    }
    let Some((ky, kx)) = king else {
        return false;
    };
    for (y, row) in pos.board.iter().enumerate() {
        for (x, cell) in row.iter().enumerate() {
            if let Some(Piece { color, pt }) = *cell {
                if color != stm && attacks(pos, y as i32, x as i32, color, pt, ky, kx) {
                    return true;
                }
            }
        }
    }
    false
}

const ORTHO: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
const DIAG: [(i32, i32); 4] = [(-1, -1), (-1, 1), (1, -1), (1, 1)];
const KING_STEPS: [(i32, i32); 8] = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)];

fn in_board(y: i32, x: i32) -> bool {
    (0..9).contains(&y) && (0..9).contains(&x)
}

/// 駒（色 `color`・種類 `pt`、位置 `(y, x)`）が `(ty, tx)` に利いているか。
/// 利きの方向は、先手が y 減少（一段目の方）へ進む。
fn attacks(pos: &Position, y: i32, x: i32, color: u8, pt: u8, ty: i32, tx: i32) -> bool {
    let fwd = if color == 0 { -1 } else { 1 };
    let hits = |dy: i32, dx: i32| (y + dy, x + dx) == (ty, tx);
    let ray = |dirs: &[(i32, i32)]| {
        dirs.iter().any(|&(dy, dx)| {
            let (mut cy, mut cx) = (y, x);
            loop {
                cy += dy;
                cx += dx;
                if !in_board(cy, cx) {
                    return false;
                }
                if (cy, cx) == (ty, tx) {
                    return true;
                }
                if pos.board[cy as usize][cx as usize].is_some() {
                    return false;
                }
            }
        })
    };
    let steps = |list: &[(i32, i32)]| list.iter().any(|&(dy, dx)| hits(dy, dx));
    match pt {
        0 => hits(fwd, 0),                                   // 歩
        1 => ray(&[(fwd, 0)]),                               // 香
        2 => hits(2 * fwd, -1) || hits(2 * fwd, 1),          // 桂
        3 => steps(&[(fwd, 0), (fwd, -1), (fwd, 1), (-fwd, -1), (-fwd, 1)]), // 銀
        4 | 8..=11 => steps(&[(fwd, 0), (fwd, -1), (fwd, 1), (0, -1), (0, 1), (-fwd, 0)]), // 金（と・成香・成桂・成銀も金）
        5 => ray(&DIAG),                                     // 角
        6 => ray(&ORTHO),                                    // 飛
        7 => steps(&KING_STEPS),                             // 玉
        12 => ray(&DIAG) || steps(&ORTHO),                   // 馬
        13 => ray(&ORTHO) || steps(&DIAG),                   // 竜
        _ => false,
    }
}

/// 指し手（PSV の 16 ビット）を、Dainagon の指し手の番号（0〜2186）にする。変換できなければ −1。
pub fn label(pos: &Position, mv: u16) -> i64 {
    let stm = pos.stm;
    let to = (mv & 0x7f) as usize;
    if to >= 81 {
        return -1;
    }
    let (ty, tx) = view_sq(to, stm);
    let cell = (ty * 9 + tx) as i64;
    if (mv >> 14) & 1 == 1 {
        // 打つ：駒の種類（1〜7）→ 持ち駒の番号（歩 0・香 1・桂 2・銀 3・金 4・角 5・飛 6）
        let idx = match (mv >> 7) & 0x7f {
            1 => 0,
            2 => 1,
            3 => 2,
            4 => 3,
            5 => 5,
            6 => 6,
            7 => 4,
            _ => return -1,
        };
        return (20 + idx) * 81 + cell;
    }
    let from = ((mv >> 7) & 0x7f) as usize;
    if from >= 81 {
        return -1;
    }
    let (fy, fx) = view_sq(from, stm);
    let (dy, dx) = (ty - fy, tx - fx);
    let dir: i64 = match (dy, dx) {
        (dy, 0) if dy < 0 => 0,
        (dy, dx) if dy < 0 && dx < 0 && dy == dx => 1,
        (dy, dx) if dy < 0 && dx > 0 && -dy == dx => 2,
        (0, dx) if dx < 0 => 3,
        (0, dx) if dx > 0 => 4,
        (dy, 0) if dy > 0 => 5,
        (dy, dx) if dy > 0 && dx < 0 && dy == -dx => 6,
        (dy, dx) if dy > 0 && dx > 0 && dy == dx => 7,
        (-2, -1) => 8,
        (-2, 1) => 9,
        _ => return -1,
    };
    let dir = if (mv >> 15) & 1 == 1 { dir + 10 } else { dir };
    dir * 81 + cell
}

/// 指し手の形式（M11.1）。PSV はやねうら王の 16 ビット、hcpe は cshogi の 16 ビット。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fmt {
    Psv,
    Hcpe,
}

/// cshogi の 16 ビットの指し手を、PSV の 16 ビットの形に直す（`label` に渡すため）。
/// cshogi は、打つときは移動元に `81 + 駒（歩 81・香 82・桂 83・銀 84・角 85・飛 86・金 87）`、成りは bit 14。
/// 変換できないもの（移動先が盤外、打つ駒が不正）は 0 を返す（= 指し手なし）。
pub fn cshogi_move_to_psv(m: u16) -> u16 {
    let to = m & 0x7f;
    if to >= 81 {
        return 0;
    }
    let from = (m >> 7) & 0x7f;
    if from >= 81 {
        // PSV の打つ：移動元の欄に駒の種類（1 歩 2 香 3 桂 4 銀 5 角 6 飛 7 金）、bit 14 が打つ
        let pt = match from {
            81 => 1,
            82 => 2,
            83 => 3,
            84 => 4,
            85 => 5,
            86 => 6,
            87 => 7,
            _ => return 0,
        };
        return to | (pt << 7) | (1 << 14);
    }
    let promote = (m >> 14) & 1;
    to | (from << 7) | (promote << 15)
}

/// 1 局面を読んで詰める。PSV（40 バイト）と hcpe（38 バイト）を、形式で分ける。
/// 戻り値は (詰めた局面, 指し手の番号（−1 は変換できない）, 評価値, 勝敗（手番側から見た −1・0・1）)。
/// 読み解けない・勝敗が範囲外は `Err`（理由の区別は `dl_of` の中で行う）。
fn read_record(rec: &[u8], fmt: Fmt) -> Result<(Position, i64, i16, i8), Skip> {
    match fmt {
        Fmt::Psv => {
            let (pos, score, mv, _ply, result) = position::decode_psv(rec).map_err(|_| Skip::Unreadable)?;
            if !(-1..=1).contains(&result) {
                return Err(Skip::Result);
            }
            let l = if mv == 0 { -1 } else { label(&pos, mv) };
            Ok((pos, l, score, result))
        }
        Fmt::Hcpe => {
            let (pos, score, mv16, gr) = position::decode_hcpe(rec).map_err(|_| Skip::Unreadable)?;
            let stm = pos.stm;
            let result: i8 = match gr {
                0 => 0,
                1 => {
                    if stm == 0 {
                        1
                    } else {
                        -1
                    }
                }
                2 => {
                    if stm == 1 {
                        1
                    } else {
                        -1
                    }
                }
                _ => return Err(Skip::Result),
            };
            let mv = cshogi_move_to_psv(mv16);
            let l = if mv == 0 { -1 } else { label(&pos, mv) };
            Ok((pos, l, score, result))
        }
    }
}

/// 飛ばしたレコードの理由（`dl_of` の戻り値の添字）。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Skip {
    /// 読み解けない
    Unreadable = 0,
    /// 勝敗が −1〜1（hcpe は 0・1・2）の外
    Result = 1,
    /// 指し手が −1（`require_move` のときだけ飛ばす）
    NoMove = 2,
}

/// 理由の名前（Python の `skip_reasons()` のキー）。
pub const SKIP_NAMES: [&str; 3] = ["unreadable", "result_range", "no_move"];

/// 1 レコードを詰める（テスト用・公開用）。PSV は 40 バイト、hcpe は 38 バイト。
/// 戻り値は (盤, 持ち駒, 王手, 指し手の番号, 評価値, 勝敗（0〜1））。
pub fn dl_record(rec: &[u8], fmt: Fmt) -> Result<(Compact, i64, i16, f32), String> {
    let (pos, l, score, result) = read_record(rec, fmt).map_err(|e| format!("読み解けない（{:?}）", e))?;
    Ok((compact(&pos), l, score, (result as f32 + 1.0) * 0.5))
}

/// バッチの作業結果（DL 形式）。各配列は局面ごとに並ぶ（盤・持ち駒は連結）。
pub struct DlBatch {
    pub board: Vec<u8>,
    pub hands: Vec<u8>,
    pub check: Vec<u8>,
    pub moves: Vec<i64>,
    pub score: Vec<f32>,
    pub result: Vec<f32>,
}

impl DlBatch {
    pub fn len(&self) -> usize {
        self.moves.len()
    }

    pub fn is_empty(&self) -> bool {
        self.moves.is_empty()
    }
}

/// 1 バッチの DL 形式の作業。飛ばしたレコードは理由別に数えて返す（`[読み解けない, 勝敗, 指し手]`）。
/// `require_move` が真なら、指し手が変換できない局面は飛ばす。偽なら、`moves` を −1 にして残す（M11.1）。
pub fn dl_of(raw: &[u8], fmt: Fmt, require_move: bool) -> (DlBatch, [u64; 3]) {
    let rec_len = match fmt {
        Fmt::Psv => psv::REC,
        Fmt::Hcpe => position::HCPE_BYTES,
    };
    let n = raw.len() / rec_len;
    let mut b = DlBatch {
        board: Vec::with_capacity(n * 81),
        hands: Vec::with_capacity(n * 2 * N_HAND),
        check: Vec::with_capacity(n),
        moves: Vec::with_capacity(n),
        score: Vec::with_capacity(n),
        result: Vec::with_capacity(n),
    };
    let mut skipped = [0u64; 3];
    for rec in raw.chunks_exact(rec_len) {
        let (pos, l, score, result) = match read_record(rec, fmt) {
            Ok(v) => v,
            Err(reason) => {
                skipped[reason as usize] += 1;
                continue;
            }
        };
        if l < 0 && require_move {
            skipped[Skip::NoMove as usize] += 1;
            continue;
        }
        let c = compact(&pos);
        b.board.extend_from_slice(&c.board);
        b.hands.extend_from_slice(&c.hands);
        b.check.push(c.check);
        b.moves.push(l);
        b.score.push(score as f32);
        b.result.push((result as f32 + 1.0) * 0.5);
    }
    (b, skipped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::parse_sfen;

    #[test]
    fn startpos_compact() {
        let pos = parse_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1").unwrap();
        let c = compact(&pos);
        assert_eq!(c.check, 0);
        assert_eq!(c.hands, [0; 14]);
        // 先手の玉は 5i（y = 8, x = 4）→ 番号 8*9+4 = 76、チャンネル 7
        assert_eq!(c.board[76], 7);
        // 後手の玉は 5a（y = 0, x = 4）→ 番号 4、相手の玉のチャンネル 14 + 7 = 21
        assert_eq!(c.board[4], 21);
        assert_eq!(c.board[40], EMPTY);
    }

    #[test]
    fn white_is_rotated() {
        // 後手番：後手の玉（5a）は手番側の玉として、番号 80 − 4 = 76 ではなく 180° 回した 76 に来る
        let pos = parse_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1").unwrap();
        let c = compact(&pos);
        assert_eq!(c.board[76], 7);
        assert_eq!(c.board[4], 21);
    }

    #[test]
    fn check_detection() {
        // 先手番の玉 5i、後手の飛車 5a：縦に通っているので王手
        let pos = parse_sfen("4r4/9/9/9/9/9/9/9/4K4 b - 1").unwrap();
        assert_eq!(compact(&pos).check, 1);
        // 間に歩があれば王手ではない
        let pos = parse_sfen("4r4/9/9/9/4P4/9/9/9/4K4 b - 1").unwrap();
        assert_eq!(compact(&pos).check, 0);
        // 後手の桂 6g（x = 3, y = 6）が先手の玉 5i（y = 8, x = 4）に利く
        let pos = parse_sfen("9/9/9/9/9/9/3n5/9/4K4 b - 1").unwrap();
        assert_eq!(compact(&pos).check, 1);
        // 玉が無い局面は王手にしない（学習データには出ない形）
        let pos = parse_sfen("4r4/9/9/9/9/9/9/9/9 b - 1").unwrap();
        assert_eq!(compact(&pos).check, 0);
    }

    #[test]
    fn move_numbers() {
        // 先手の 7g7f：to = 7f（file 7, rank f）→ s = 6*9 + 5 = 59、from = 7g → s = 6*9+6 = 60
        let pos = parse_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1").unwrap();
        let mv = (60u16 << 7) | 59;
        // 7g7f：dy = −1、dx = 0 → dir 0。to の番号は (y=5, x=2)（7f）→ 5*9+2 = 47
        assert_eq!(label(&pos, mv), 47);
        // 打つ：P*5e（歩打ち、5e＝ y 4, x 4 → 番号 40）。dir = 20 + 0。
        let drop = (1u16 << 14) | (1u16 << 7) | 40;
        assert_eq!(label(&pos, drop), 20 * 81 + 40);
        assert_eq!(label(&pos, 0), -1);
    }
}
