//! 局面の表現と、SFEN・PackedSfen（PSV の先頭32バイト）からの読み取り。
//!
//! 座標は設計書 §F1 に従う。`board[y][x]` で、`y` は段（0 = 一段目）、
//! `x` は SFEN の並び順（0 = 9筋、8 = 1筋）。

/// 駒種の番号 `pt`（§F1.3 の順。0..=7 は基本駒、8..=13 は成駒。金に成駒はない）。
pub const PIECE_SFEN: [&str; 14] = [
    "P", "L", "N", "S", "G", "B", "R", "K", "+P", "+L", "+N", "+S", "+B", "+R",
];

/// 持ち駒の駒種番号 `h`（§F1.4）。
const HAND_LETTER: [char; 7] = ['P', 'L', 'N', 'S', 'G', 'B', 'R'];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Piece {
    /// 0 = 先手、1 = 後手
    pub color: u8,
    /// `PIECE_SFEN` の添字
    pub pt: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Position {
    pub board: [[Option<Piece>; 9]; 9],
    /// 手番（0 = 先手、1 = 後手）
    pub stm: u8,
    /// `hands[color][h]`
    pub hands: [[u8; 7]; 2],
}

impl Position {
    fn empty() -> Self {
        Position {
            board: [[None; 9]; 9],
            stm: 0,
            hands: [[0; 7]; 2],
        }
    }
}

fn piece_type_of(base: char, promoted: bool) -> Option<u8> {
    let text = if promoted {
        format!("+{base}")
    } else {
        base.to_string()
    };
    PIECE_SFEN.iter().position(|s| *s == text).map(|i| i as u8)
}

/// SFEN（`盤面 手番 持ち駒 手数`）を読む。手数は使わない。
pub fn parse_sfen(sfen: &str) -> Result<Position, String> {
    let mut fields = sfen.split_whitespace();
    let (rows, turn, hands) = match (fields.next(), fields.next(), fields.next()) {
        (Some(r), Some(t), Some(h)) => (r, t, h),
        _ => return Err(format!("SFEN の形式が不正: {sfen}")),
    };

    let rows: Vec<&str> = rows.split('/').collect();
    if rows.len() != 9 {
        return Err(format!("SFEN の段数が9ではない: {sfen}"));
    }

    let mut pos = Position::empty();
    for (y, row) in rows.iter().enumerate() {
        let mut x = 0usize;
        let mut promoted = false;
        for ch in row.chars() {
            if let Some(d) = ch.to_digit(10) {
                x += d as usize;
                continue;
            }
            if ch == '+' {
                promoted = true;
                continue;
            }
            if x >= 9 {
                return Err(format!("SFEN の段が長すぎる: {row}"));
            }
            let color = if ch.is_ascii_uppercase() { 0 } else { 1 };
            let pt = piece_type_of(ch.to_ascii_uppercase(), promoted)
                .ok_or_else(|| format!("駒文字が不正: {ch}"))?;
            pos.board[y][x] = Some(Piece { color, pt });
            x += 1;
            promoted = false;
        }
        if x != 9 {
            return Err(format!("SFEN の段の桁数が9ではない: {row}"));
        }
    }

    pos.stm = match turn {
        "b" => 0,
        "w" => 1,
        _ => return Err(format!("手番が不正: {turn}")),
    };

    if hands != "-" {
        let mut n = 0u32;
        for ch in hands.chars() {
            if let Some(d) = ch.to_digit(10) {
                n = n * 10 + d;
                continue;
            }
            let color = if ch.is_ascii_uppercase() { 0 } else { 1 };
            let h = HAND_LETTER
                .iter()
                .position(|c| *c == ch.to_ascii_uppercase())
                .ok_or_else(|| format!("持ち駒の文字が不正: {ch}"))?;
            let count = n.max(1);
            let total = pos.hands[color][h] as u32 + count;
            if total > 255 {
                return Err(format!("持ち駒が多すぎる: {hands}"));
            }
            pos.hands[color][h] = total as u8;
            n = 0;
        }
    }
    Ok(pos)
}

/// 局面を SFEN 文字列にする。手数は 1 以上にする。
pub fn to_sfen(pos: &Position, ply: u16) -> String {
    let rows: Vec<String> = pos
        .board
        .iter()
        .map(|row| {
            let mut s = String::new();
            let mut empty = 0;
            for cell in row.iter() {
                match cell {
                    None => empty += 1,
                    Some(p) => {
                        if empty > 0 {
                            s.push_str(&empty.to_string());
                            empty = 0;
                        }
                        let text = PIECE_SFEN[p.pt as usize];
                        if p.color == 0 {
                            s.push_str(&text.to_ascii_uppercase());
                        } else {
                            s.push_str(&text.to_ascii_lowercase());
                        }
                    }
                }
            }
            if empty > 0 {
                s.push_str(&empty.to_string());
            }
            s
        })
        .collect();

    let mut hand = String::new();
    for color in 0..2 {
        for h in (0..7).rev() {
            let n = pos.hands[color][h];
            if n == 0 {
                continue;
            }
            if n > 1 {
                hand.push_str(&n.to_string());
            }
            let c = HAND_LETTER[h];
            hand.push(if color == 0 {
                c
            } else {
                c.to_ascii_lowercase()
            });
        }
    }
    if hand.is_empty() {
        hand.push('-');
    }

    format!(
        "{} {} {} {}",
        rows.join("/"),
        if pos.stm == 0 { "b" } else { "w" },
        hand,
        ply.max(1)
    )
}

/// PackedSfen（32バイト）を局面に戻す。やねうら王の PackedSfen 形式。
///
/// ビット列は各バイトの下位ビットから順に読む。
/// 1. 手番（1bit）
/// 2. 先手玉の位置、後手玉の位置（各7bit。マス番号は下記 `cshogi_sq_to_yx`）
/// 3. 玉以外の盤上の各マス（マス番号の昇順。玉のマスは飛ばす）：ハフマン符号の駒
/// 4. 残りのビット：持ち駒（と駒箱にある駒）の符号を 256bit の終わりまで
pub fn decode_psfen(data: &[u8]) -> Result<Position, String> {
    if data.len() != 32 {
        return Err(format!(
            "PackedSfen は32バイトのはずが {} バイト",
            data.len()
        ));
    }
    let mut r = BitReader { data, pos: 0 };
    let mut pos = Position::empty();

    pos.stm = r.bits(1)? as u8;
    let king_sq = [r.bits(7)? as usize, r.bits(7)? as usize];
    for (color, &sq) in king_sq.iter().enumerate() {
        if sq >= 81 {
            return Err(format!("玉の位置が不正: {sq}"));
        }
        let (y, x) = cshogi_sq_to_yx(sq);
        pos.board[y][x] = Some(Piece {
            color: color as u8,
            pt: 7,
        });
    }

    for sq in 0..81usize {
        if king_sq.contains(&sq) {
            continue;
        }
        if let Some((color, pt)) = r.read_lut(&BOARD_LUT)? {
            let (y, x) = cshogi_sq_to_yx(sq);
            pos.board[y][x] = Some(Piece { color, pt });
        }
    }

    while r.pos < 256 {
        if let Some((color, h)) = r.read_lut(&HAND_LUT)? {
            let n = &mut pos.hands[color as usize][h as usize];
            *n = n.checked_add(1).ok_or("持ち駒が多すぎる")?;
        }
    }
    Ok(pos)
}

/// PSV（40バイト）を読む。戻り値は (局面, 評価値, 指し手, 手数, 結果)。
pub fn decode_psv(rec: &[u8]) -> Result<(Position, i16, u16, u16, i8), String> {
    if rec.len() != 40 {
        return Err(format!("PSV は40バイトのはずが {} バイト", rec.len()));
    }
    let pos = decode_psfen(&rec[..32])?;
    let score = i16::from_le_bytes([rec[32], rec[33]]);
    let mv = u16::from_le_bytes([rec[34], rec[35]]);
    let ply = u16::from_le_bytes([rec[36], rec[37]]);
    let result = rec[38] as i8;
    Ok((pos, score, mv, ply, result))
}

/// PackedSfen のマス番号（1一 = 0、1九 = 8、9一 = 72）を `(y, x)` にする。
/// マス番号は筋ごとに段の順で並ぶ（`筋の番号 − 1` が 9 で割った商、`段` が余り）。
fn cshogi_sq_to_yx(sq: usize) -> (usize, usize) {
    let y = sq % 9;
    let x = 8 - sq / 9;
    (y, x)
}

/// 符号表の 1 行：（値、ビット数、駒）。駒が None の符号は「駒なし」「駒箱」を表す。
type Code = (u8, u8, Option<(u8, u8)>);

/// 盤上の駒の符号（§F の PackedSfen 形式）。駒は (color, pt)。
const BOARD_CODES: &[Code] = &[
    (0, 1, None),
    (1, 4, Some((0, 0))),    // 先手 歩
    (3, 6, Some((0, 1))),    // 先手 香
    (11, 6, Some((0, 2))),   // 先手 桂
    (7, 6, Some((0, 3))),    // 先手 銀
    (31, 8, Some((0, 5))),   // 先手 角
    (63, 8, Some((0, 6))),   // 先手 飛
    (15, 6, Some((0, 4))),   // 先手 金
    (5, 4, Some((0, 8))),    // 先手 と
    (19, 6, Some((0, 9))),   // 先手 成香
    (27, 6, Some((0, 10))),  // 先手 成桂
    (23, 6, Some((0, 11))),  // 先手 成銀
    (95, 8, Some((0, 12))),  // 先手 馬
    (127, 8, Some((0, 13))), // 先手 竜
    (9, 4, Some((1, 0))),    // 後手 歩
    (35, 6, Some((1, 1))),   // 後手 香
    (43, 6, Some((1, 2))),   // 後手 桂
    (39, 6, Some((1, 3))),   // 後手 銀
    (159, 8, Some((1, 5))),  // 後手 角
    (191, 8, Some((1, 6))),  // 後手 飛
    (47, 6, Some((1, 4))),   // 後手 金
    (13, 4, Some((1, 8))),   // 後手 と
    (51, 6, Some((1, 9))),   // 後手 成香
    (59, 6, Some((1, 10))),  // 後手 成桂
    (55, 6, Some((1, 11))),  // 後手 成銀
    (223, 8, Some((1, 12))), // 後手 馬
    (255, 8, Some((1, 13))), // 後手 竜
];

/// 持ち駒の符号。`(color, h)` は持ち駒、None は駒箱の駒（持ち駒に数えない）。
const HAND_CODES: &[Code] = &[
    (0, 3, Some((0, 0))),
    (4, 3, Some((1, 0))),
    (2, 3, None),
    (1, 5, Some((0, 1))),
    (17, 5, Some((1, 1))),
    (9, 5, None),
    (5, 5, Some((0, 2))),
    (21, 5, Some((1, 2))),
    (13, 5, None),
    (3, 5, Some((0, 3))),
    (19, 5, Some((1, 3))),
    (11, 5, None),
    (7, 5, Some((0, 4))),
    (23, 5, Some((1, 4))),
    (27, 5, None),
    (15, 7, Some((0, 5))),
    (79, 7, Some((1, 5))),
    (47, 7, None),
    (31, 7, Some((0, 6))),
    (95, 7, Some((1, 6))),
    (63, 7, None),
];

/// 符号の引き表の1項目。`len` は符号のビット数（0 は、8 ビット読んでも一致しないこと）。
#[derive(Clone, Copy)]
struct Lut {
    len: u8,
    piece: Option<(u8, u8)>,
}

/// 符号表から 8 ビットの先読みで引く表を作る。`lut[v]` は、下位ビットから読んだ `v` に対して
/// 最も短く一致する符号。これは 1 ビットずつ照合する読み方と同じ結果になる。
const fn build_lut(table: &[Code]) -> [Lut; 256] {
    let mut lut = [Lut { len: 0, piece: None }; 256];
    let mut len = 1u8;
    while len <= 8 {
        let mut i = 0;
        while i < table.len() {
            let (value, l, piece) = table[i];
            if l == len {
                let mut v = value as usize;
                while v < 256 {
                    if lut[v].len == 0 {
                        lut[v] = Lut { len, piece };
                    }
                    v += 1 << len;
                }
            }
            i += 1;
        }
        len += 1;
    }
    lut
}

static BOARD_LUT: [Lut; 256] = build_lut(BOARD_CODES);
static HAND_LUT: [Lut; 256] = build_lut(HAND_CODES);

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl BitReader<'_> {
    fn bit(&mut self) -> Result<u8, String> {
        if self.pos >= 256 {
            return Err("PackedSfen のビット列が途中で終わった".into());
        }
        let b = (self.data[self.pos / 8] >> (self.pos % 8)) & 1;
        self.pos += 1;
        Ok(b)
    }

    /// `n` ビットを下位から順に読んで整数にする。
    fn bits(&mut self, n: u32) -> Result<u32, String> {
        let mut v = 0u32;
        for i in 0..n {
            v |= (self.bit()? as u32) << i;
        }
        Ok(v)
    }

    /// 8 ビット先読みで符号を1つ読む。符号は下位ビットから並ぶ。
    #[inline(always)]
    fn read_lut(&mut self, lut: &[Lut; 256]) -> Result<Option<(u8, u8)>, String> {
        if self.pos >= 256 {
            return Err("PackedSfen のビット列が途中で終わった".into());
        }
        let avail = 256 - self.pos;
        let e = lut[self.peek8()];
        if e.len == 0 {
            // 8 ビット読んでも一致しない（残りが 8 ビット未満なら、途中で終わったのと同じ）
            return Err(if avail >= 8 {
                "不正なハフマン符号".into()
            } else {
                "PackedSfen のビット列が途中で終わった".into()
            });
        }
        if e.len as usize > avail {
            return Err("PackedSfen のビット列が途中で終わった".into());
        }
        self.pos += e.len as usize;
        Ok(e.piece)
    }

    /// 次の 8 ビットを、下位から並べた整数として返す。256 ビットの外は 0 とみなす。
    #[inline(always)]
    fn peek8(&self) -> usize {
        let byte = self.pos / 8;
        let lo = self.data[byte] as u32;
        let hi = if byte + 1 < self.data.len() {
            self.data[byte + 1] as u32
        } else {
            0
        };
        (((hi << 8) | lo) >> (self.pos % 8) & 0xFF) as usize
    }
}
