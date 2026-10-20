//! PSV ファイル（1局面40バイト、§S2）の読み取りの共通部品。

use std::fs::File;
use std::io::{BufReader, Read};

pub const REC: usize = 40;

/// score（手番側視点）。
pub fn score(rec: &[u8]) -> i16 {
    i16::from_le_bytes([rec[32], rec[33]])
}

/// gamePly。
pub fn ply(rec: &[u8]) -> u16 {
    u16::from_le_bytes([rec[36], rec[37]])
}

/// game_result（手番側視点。1 = 勝、0 = 引分、−1 = 負）。
pub fn result(rec: &[u8]) -> i8 {
    rec[38] as i8
}

/// `buf` を満たすまで読む。EOF で途中まで読めた場合は、読めた量を返す。
pub fn read_full(reader: &mut impl Read, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..])? {
            0 => break,
            n => filled += n,
        }
    }
    Ok(filled)
}

/// ファイルサイズを返す。40の倍数でなければエラー。
pub fn checked_size(path: &str) -> Result<u64, String> {
    let len = std::fs::metadata(path)
        .map_err(|e| format!("{path}: {e}"))?
        .len();
    if len % REC as u64 != 0 {
        return Err(format!("{path}: サイズが40バイトの倍数でない"));
    }
    Ok(len)
}

/// `path` の全レコードを先頭から順に `f` へ渡す。
pub fn for_each_record(
    path: &str,
    mut f: impl FnMut(&[u8]) -> Result<(), String>,
) -> Result<(), String> {
    let file = File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut reader = BufReader::with_capacity(1 << 20, file);
    let mut rec = [0u8; REC];
    loop {
        let n = read_full(&mut reader, &mut rec).map_err(|e| format!("{path}: {e}"))?;
        match n {
            0 => return Ok(()),
            REC => f(&rec)?,
            _ => return Err(format!("{path}: 末尾が40バイトの倍数でない")),
        }
    }
}
