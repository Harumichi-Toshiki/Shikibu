//! 計測用（`bench_extract`）。Python を介さず、「PSV の解読 → 両視点の特徴抽出」だけを時間計測する。
//!
//! 作業は `loader::features_of` で行う。ローダーの作業係と同じ計算を測るため。

use std::hint::black_box;
use std::thread;
use std::time::Instant;

use crate::loader::features_of;
use crate::psv::{self, REC};

/// 1回の `features_of` に渡す局面数（ローダーのバッチに近い大きさ）
const CHUNK: usize = 16_384;

/// `paths` の局面を先頭から `n` 件読み、その範囲を `num_threads` 本で分けて特徴抽出する。
/// 足りないときはファイルを循環して `n` 件にする。戻り値は計測した秒数（読み込みは含まない）。
pub fn bench_extract(paths: &[String], n: usize, version: u8, num_threads: usize) -> Result<f64, String> {
    let raw = load_records(paths, n)?;
    let per_thread = n.div_ceil(num_threads) * REC;

    let start = Instant::now();
    thread::scope(|s| {
        let handles: Vec<_> = raw
            .chunks(per_thread)
            .map(|part| {
                s.spawn(move || -> Result<u64, String> {
                    let mut check = 0u64;
                    for batch in part.chunks(CHUNK * REC) {
                        let f = features_of(batch, version)?;
                        check = check
                            .wrapping_add(f.stm_idx.iter().map(|&i| i as u64).sum::<u64>())
                            .wrapping_add(f.nstm_idx.iter().map(|&i| i as u64).sum::<u64>());
                    }
                    Ok(check)
                })
            })
            .collect();
        for h in handles {
            black_box(h.join().map_err(|_| "計測のスレッドが異常終了".to_string())??);
        }
        Ok::<(), String>(())
    })?;
    Ok(start.elapsed().as_secs_f64())
}

/// `n` 件のレコードを先頭から読む。ファイルを使い切ったら先頭へ戻って循環する。
fn load_records(paths: &[String], n: usize) -> Result<Vec<u8>, String> {
    let mut buf = Vec::with_capacity(n * REC);
    while buf.len() < n * REC {
        let before = buf.len();
        for p in paths {
            psv::for_each_record(p, |rec| {
                if buf.len() < n * REC {
                    buf.extend_from_slice(rec);
                }
                Ok(())
            })?;
        }
        if buf.len() == before {
            return Err("bench_extract: 局面が1件もない".into());
        }
    }
    Ok(buf)
}
