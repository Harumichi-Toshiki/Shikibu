//! 事前シャッフルとシャード分割（§S3.1）。2パス方式：
//!
//! 1. 全局面を乱数で K 個のバケツ（一時ファイル）へ振り分ける。
//!    K は、1つのバケツをメモリに載せられる大きさ（mem_limit_mb の 1/4）になるよう決める。
//! 2. バケツの順を乱数で並べ、バケツを1つずつ読んでメモリ上でシャッフルし、
//!    records_per_shard 局面ごとに shard_XXXXX.psv へ連続して書く。
//!
//! バケツは乱数で作った全データの一様な部分集合なので、シャードの中身も一様に混ざる。

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::psv::{self, REC};
use crate::rng::Rng;

/// バケツの数の上限（同時に開くファイルの数を抑える）。
const MAX_BUCKETS: usize = 1024;
const BUCKET_WRITE_BUF: usize = 64 << 10;
const SHARD_WRITE_BUF: usize = 1 << 20;
const TMP_DIR: &str = "_buckets_tmp";
const PASS1_SALT: u64 = 0x5EED_0001;
const PASS2_SALT: u64 = 0x5EED_0002;

fn io_err(path: &Path, e: std::io::Error) -> String {
    format!("{}: {e}", path.display())
}

fn is_shard_name(name: &str) -> bool {
    name.starts_with("shard_") && name.ends_with(".psv")
}

fn swap_records(data: &mut [u8], i: usize, j: usize) {
    for k in 0..REC {
        data.swap(i * REC + k, j * REC + k);
    }
}

struct ShardWriter {
    dir: PathBuf,
    per_shard: u64,
    index: usize,
    count: u64,
    cur: Option<BufWriter<File>>,
    paths: Vec<String>,
}

impl ShardWriter {
    fn new(dir: &Path, per_shard: u64) -> Self {
        ShardWriter {
            dir: dir.to_path_buf(),
            per_shard,
            index: 0,
            count: 0,
            cur: None,
            paths: Vec::new(),
        }
    }

    fn close(&mut self) -> Result<(), String> {
        if let Some(mut w) = self.cur.take() {
            w.flush().map_err(|e| format!("シャードの書き出し: {e}"))?;
        }
        Ok(())
    }

    fn write(&mut self, rec: &[u8]) -> Result<(), String> {
        if self.cur.is_none() {
            let p = self.dir.join(format!("shard_{:05}.psv", self.index));
            let f = File::create(&p).map_err(|e| io_err(&p, e))?;
            self.paths.push(p.to_string_lossy().into_owned());
            self.cur = Some(BufWriter::with_capacity(SHARD_WRITE_BUF, f));
            self.count = 0;
        }
        let w = self.cur.as_mut().expect("直前に開いた");
        w.write_all(rec).map_err(|e| format!("シャードの書き出し: {e}"))?;
        self.count += 1;
        if self.count == self.per_shard {
            self.close()?;
            self.index += 1;
        }
        Ok(())
    }

    fn finish(mut self) -> Result<Vec<String>, String> {
        self.close()?;
        Ok(self.paths)
    }
}

/// 全局面をシャッフルして `out_dir/shard_XXXXX.psv` に分割し、そのパスを昇順で返す。
pub fn shuffle_shards(
    inputs: &[String],
    out_dir: &str,
    records_per_shard: u64,
    seed: u64,
    mem_limit_mb: u64,
) -> Result<Vec<String>, String> {
    if records_per_shard == 0 {
        return Err("records_per_shard は1以上".into());
    }
    if mem_limit_mb == 0 {
        return Err("mem_limit_mb は1以上".into());
    }
    let out = Path::new(out_dir);
    fs::create_dir_all(out).map_err(|e| io_err(out, e))?;
    for entry in fs::read_dir(out).map_err(|e| io_err(out, e))? {
        let name = entry.map_err(|e| io_err(out, e))?.file_name();
        if is_shard_name(&name.to_string_lossy()) {
            return Err(format!("出力先に {} が既にある。空のフォルダを指定する", name.to_string_lossy()));
        }
    }

    let mut total_bytes = 0u64;
    for p in inputs {
        total_bytes += psv::checked_size(p)?;
    }
    if total_bytes == 0 {
        return Ok(Vec::new());
    }

    let budget = ((mem_limit_mb << 20) / 4).max(REC as u64);
    let buckets = total_bytes.div_ceil(budget) as usize;
    if buckets > MAX_BUCKETS {
        return Err(format!(
            "バケツ数 {buckets} が上限 {MAX_BUCKETS} を超える。mem_limit_mb を大きくする"
        ));
    }

    let tmp = out.join(TMP_DIR);
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).map_err(|e| io_err(&tmp, e))?;
    let bucket_path = |b: usize| tmp.join(format!("bucket_{b:05}.bin"));

    // パス1：全局面を乱数でバケツへ振り分ける
    let mut rng = Rng::new(seed ^ PASS1_SALT);
    {
        let mut writers = Vec::with_capacity(buckets);
        for b in 0..buckets {
            let p = bucket_path(b);
            let f = File::create(&p).map_err(|e| io_err(&p, e))?;
            writers.push(BufWriter::with_capacity(BUCKET_WRITE_BUF, f));
        }
        for p in inputs {
            psv::for_each_record(p, |rec| {
                let b = rng.below(buckets);
                writers[b].write_all(rec).map_err(|e| format!("バケツの書き出し: {e}"))
            })?;
        }
        for w in writers.iter_mut() {
            w.flush().map_err(|e| format!("バケツの書き出し: {e}"))?;
        }
    }

    // パス2：バケツの順を乱数で決め、各バケツをシャッフルして連続してシャードへ書く
    let mut order: Vec<usize> = (0..buckets).collect();
    let mut rng = Rng::new(seed ^ PASS2_SALT);
    for i in (1..buckets).rev() {
        order.swap(i, rng.below(i + 1));
    }
    let mut shards = ShardWriter::new(out, records_per_shard);
    for b in order {
        let p = bucket_path(b);
        let mut data = fs::read(&p).map_err(|e| io_err(&p, e))?;
        let n = data.len() / REC;
        for i in (1..n).rev() {
            let j = rng.below(i + 1);
            if i != j {
                swap_records(&mut data, i, j);
            }
        }
        for rec in data.chunks_exact(REC) {
            shards.write(rec)?;
        }
        drop(data);
        fs::remove_file(&p).map_err(|e| io_err(&p, e))?;
    }
    let paths = shards.finish()?;
    fs::remove_dir_all(&tmp).map_err(|e| io_err(&tmp, e))?;
    Ok(paths)
}
