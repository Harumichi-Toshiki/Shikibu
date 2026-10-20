//! 学習用バッチのローダー（§S3.2・§S3.3）。
//!
//! 局面の並びは読み係が1か所で決め、特徴抽出だけを作業係が並列に行う。
//! 呼び出し側は、バッチに付けた番号の順に並べ直して返すので、並びは num_threads に依らない。
//!
//! ```text
//! [コピー係] --cache_dir へ先読み--> [読み係：順に読んで、シャッフル] --40バイトの束-->
//!     [作業係 × num_threads：特徴抽出] --番号つき--> [呼び出し側：番号順に並べ直す]
//! ```
//!
//! 再開：バッチを返すたびに、そのバッチを切り出した時点の読み位置（ファイル番号・バイト位置・
//! 乱数の状態）を覚えておく。`state()` はその位置を返し、`resume_state` でそこから読み直す。
//! シャッフルなしなら、読み位置の先が次の局面なので続きは完全に一致する。
//! シャッフルありなら、シャッフル用のバッファに残っていた局面は失うが、重複はしない。

use std::collections::{BTreeMap, VecDeque};
use std::fs::{self, File};
use std::io::{BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use numpy::IntoPyArray;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::position;
use crate::psv::{self, REC};
use crate::raypair;
use crate::rng::Rng;

/// コピーしたファイルの名前の接頭辞。cache_dir の中で、この接頭辞のファイルだけを消す。
const CACHE_PREFIX: &str = "shk_cache_";
/// 1局面の特徴数の目安（作業領域と出力の確保に使う）
const FEATS_PER_POS: usize = 128;
/// 条件変数を待つ間隔（停止の合図を取りこぼさないため）。
const POLL: Duration = Duration::from_millis(50);

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn wait<'a, T>(cv: &Condvar, g: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
    cv.wait_timeout(g, POLL).unwrap_or_else(PoisonError::into_inner).0
}

/// 読み位置。`offset` は `file` の中で、次に読むレコードのバイト位置。
#[derive(Clone, Copy, Debug)]
struct Cursor {
    file: usize,
    offset: u64,
    /// シャッフル用の乱数の状態
    rng: u64,
}

/// 読み係から作業係へ渡す、番号つきの束。
struct Work {
    seq: u64,
    cursor: Cursor,
    /// `Ok(None)` は終わり、`Err` は読み込みの失敗。
    raw: Result<Option<Vec<u8>>, String>,
}

/// 作業係から呼び出し側へ返す、番号つきの結果。
struct Done {
    seq: u64,
    cursor: Cursor,
    /// `Ok(None)` は終わり。
    body: Result<Option<Features>, String>,
}

/// 1バッチぶんの出力（nn.EmbeddingBag の形式、§S3.3）。
pub(crate) struct Features {
    pub(crate) stm_idx: Vec<i64>,
    stm_off: Vec<i64>,
    pub(crate) nstm_idx: Vec<i64>,
    nstm_off: Vec<i64>,
    score: Vec<f32>,
    result: Vec<f32>,
}

pub(crate) fn features_of(raw: &[u8], version: u8) -> Result<Features, String> {
    let n = raw.len() / REC;
    // 1局面あたり両視点で約 130 個（盤上の駒と持ち駒、隣接する駒の対）
    let mut f = Features {
        stm_idx: Vec::with_capacity(n * FEATS_PER_POS),
        stm_off: Vec::with_capacity(n),
        nstm_idx: Vec::with_capacity(n * FEATS_PER_POS),
        nstm_off: Vec::with_capacity(n),
        score: Vec::with_capacity(n),
        result: Vec::with_capacity(n),
    };
    let mut buf: Vec<u32> = Vec::with_capacity(FEATS_PER_POS);
    for rec in raw.chunks_exact(REC) {
        let (pos, score, _, _, result) = position::decode_psv(rec)?;
        if !(-1..=1).contains(&result) {
            return Err(format!("game_result が −1〜1 の外: {result}"));
        }
        let stm = pos.stm;
        let nstm = 1 - stm;
        let pieces = raypair::Pieces::new(&pos);
        f.stm_off.push(f.stm_idx.len() as i64);
        raypair::features_from(&pieces, stm, version, &mut buf)?;
        f.stm_idx.extend(buf.iter().map(|&i| i as i64));
        f.nstm_off.push(f.nstm_idx.len() as i64);
        raypair::features_from(&pieces, nstm, version, &mut buf)?;
        f.nstm_idx.extend(buf.iter().map(|&i| i as i64));
        f.score.push(score as f32);
        f.result.push((result as f32 + 1.0) * 0.5);
    }
    Ok(f)
}

// ---------- コピー係（cache_dir への先読み） ----------

struct CacheState {
    /// 予約済みのファイル数（コピー中・コピー済みで未読・読んでいる最中のもの）
    in_use: usize,
    /// コピーが済んで、まだ開かれていないファイルの番号（昇順）
    ready: VecDeque<usize>,
    error: Option<String>,
    stop: bool,
}

struct CacheHub {
    dir: PathBuf,
    /// 同時にあってよいファイル数（max_cached + 1）
    cap: usize,
    state: Mutex<CacheState>,
    cv: Condvar,
}

impl CacheHub {
    fn new(dir: PathBuf, cap: usize) -> Self {
        CacheHub {
            dir,
            cap,
            state: Mutex::new(CacheState {
                in_use: 0,
                ready: VecDeque::new(),
                error: None,
                stop: false,
            }),
            cv: Condvar::new(),
        }
    }

    fn file_name(idx: usize, src: &Path) -> String {
        let base = src
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        format!("{CACHE_PREFIX}{idx:06}_{base}")
    }

    /// 読み終えたファイルの枠を空ける。
    fn release(&self) {
        lock(&self.state).in_use -= 1;
        self.cv.notify_all();
    }

    fn stop(&self) {
        lock(&self.state).stop = true;
        self.cv.notify_all();
    }
}

/// ファイルを `first` から順に cache_dir へコピーする。枠（max_cached + 1）が空くまで待つ。
fn copy_ahead(paths: Arc<Vec<PathBuf>>, first: usize, hub: Arc<CacheHub>) {
    for idx in first..paths.len() {
        {
            let mut st = lock(&hub.state);
            while !st.stop && st.in_use >= hub.cap {
                st = wait(&hub.cv, st);
            }
            if st.stop {
                return;
            }
            st.in_use += 1;
        }
        let name = CacheHub::file_name(idx, &paths[idx]);
        let dst = hub.dir.join(&name);
        // 途中のファイルは .part の名前で作り、揃ってから本名にする
        let part = hub.dir.join(format!("{name}.part"));
        let res = fs::copy(&paths[idx], &part).and_then(|_| fs::rename(&part, &dst));
        let mut st = lock(&hub.state);
        match res {
            Ok(()) => st.ready.push_back(idx),
            Err(e) => st.error = Some(format!("{}: コピー失敗: {e}", paths[idx].display())),
        }
        hub.cv.notify_all();
        if st.error.is_some() {
            return;
        }
    }
}

/// `idx` のコピーが揃うまで待ち、読む順番を受け取る。
fn wait_for_copy(hub: &CacheHub, idx: usize) -> Result<(), String> {
    let mut st = lock(&hub.state);
    loop {
        if let Some(e) = &st.error {
            return Err(e.clone());
        }
        if st.stop {
            return Err("ローダーが止められた".into());
        }
        if st.ready.front() == Some(&idx) {
            st.ready.pop_front();
            return Ok(());
        }
        st = wait(&hub.cv, st);
    }
}

// ---------- 読み係 ----------

/// ファイルを順に開き、40バイトずつレコードを返す。
struct Source {
    paths: Arc<Vec<PathBuf>>,
    cache: Option<Arc<CacheHub>>,
    file: usize,
    offset: u64,
    reader: Option<BufReader<File>>,
    /// 開いているコピーのパス（読み終えたら消して枠を空ける）
    copied: Option<PathBuf>,
}

impl Source {
    fn new(paths: Arc<Vec<PathBuf>>, cache: Option<Arc<CacheHub>>, start: Cursor) -> Self {
        Source {
            paths,
            cache,
            file: start.file,
            offset: start.offset,
            reader: None,
            copied: None,
        }
    }

    fn open(&mut self) -> Result<(), String> {
        let path = match &self.cache {
            Some(hub) => {
                wait_for_copy(hub, self.file)?;
                let p = hub.dir.join(CacheHub::file_name(self.file, &self.paths[self.file]));
                self.copied = Some(p.clone());
                p
            }
            None => self.paths[self.file].clone(),
        };
        let mut f = File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        f.seek(SeekFrom::Start(self.offset))
            .map_err(|e| format!("{}: {e}", path.display()))?;
        self.reader = Some(BufReader::with_capacity(1 << 20, f));
        Ok(())
    }

    fn close_file(&mut self) {
        self.reader = None;
        if let Some(p) = self.copied.take() {
            let _ = fs::remove_file(&p);
            if let Some(hub) = &self.cache {
                hub.release();
            }
        }
    }

    /// 次のレコードを返す。全ファイルを読み終えたら `None`。
    fn next(&mut self) -> Result<Option<[u8; REC]>, String> {
        loop {
            if self.file >= self.paths.len() {
                return Ok(None);
            }
            if self.reader.is_none() {
                self.open()?;
            }
            let reader = self.reader.as_mut().expect("直前に開いた");
            let mut rec = [0u8; REC];
            let n = psv::read_full(reader, &mut rec)
                .map_err(|e| format!("{}: {e}", self.paths[self.file].display()))?;
            match n {
                REC => {
                    self.offset += REC as u64;
                    return Ok(Some(rec));
                }
                0 => {
                    self.close_file();
                    self.file += 1;
                    self.offset = 0;
                }
                _ => {
                    return Err(format!(
                        "{}: 末尾が40バイトの倍数でない",
                        self.paths[self.file].display()
                    ))
                }
            }
        }
    }

    fn cursor(&self, rng: &Rng) -> Cursor {
        Cursor {
            file: self.file,
            offset: self.offset,
            rng: rng.state(),
        }
    }
}

impl Drop for Source {
    fn drop(&mut self) {
        self.close_file();
    }
}

/// シャッフル用のバッファ（`shuffle` 件）から1件選んで返す。`shuffle == 0` ならそのまま返す。
fn next_record(
    src: &mut Source,
    pool: &mut Vec<[u8; REC]>,
    rng: &mut Rng,
    shuffle: usize,
) -> Result<Option<[u8; REC]>, String> {
    if shuffle == 0 {
        return src.next();
    }
    while pool.len() < shuffle {
        match src.next()? {
            Some(rec) => pool.push(rec),
            None => break,
        }
    }
    if pool.is_empty() {
        return Ok(None);
    }
    let i = rng.below(pool.len());
    Ok(Some(pool.swap_remove(i)))
}

/// 作業係へバッチを渡す。枠（credit）が空くまで待つ。受け手が消えていたら `false`。
fn emit(credit: &Receiver<()>, work: &Sender<Work>, seq: u64, cursor: Cursor, data: Vec<u8>) -> bool {
    credit.recv().is_ok()
        && work
            .send(Work {
                seq,
                cursor,
                raw: Ok(Some(data)),
            })
            .is_ok()
}

fn produce(
    paths: Arc<Vec<PathBuf>>,
    cache: Option<Arc<CacheHub>>,
    start: Cursor,
    batch_size: usize,
    shuffle: usize,
    credit: Receiver<()>,
    work: Sender<Work>,
) {
    let mut src = Source::new(paths, cache, start);
    let mut rng = Rng::from_state(start.rng);
    let mut pool: Vec<[u8; REC]> = Vec::with_capacity(shuffle);
    let mut raw: Vec<u8> = Vec::with_capacity(batch_size * REC);
    let mut seq = 0u64;
    loop {
        match next_record(&mut src, &mut pool, &mut rng, shuffle) {
            Ok(Some(rec)) => {
                raw.extend_from_slice(&rec);
                if raw.len() == batch_size * REC {
                    let data = std::mem::replace(&mut raw, Vec::with_capacity(batch_size * REC));
                    if !emit(&credit, &work, seq, src.cursor(&rng), data) {
                        return;
                    }
                    seq += 1;
                }
            }
            Ok(None) => {
                if !raw.is_empty() {
                    if !emit(&credit, &work, seq, src.cursor(&rng), raw) {
                        return;
                    }
                    seq += 1;
                }
                let _ = work.send(Work {
                    seq,
                    cursor: src.cursor(&rng),
                    raw: Ok(None),
                });
                return;
            }
            Err(msg) => {
                let _ = work.send(Work {
                    seq,
                    cursor: src.cursor(&rng),
                    raw: Err(msg),
                });
                return;
            }
        }
    }
}

// ---------- 作業係 ----------

fn work_loop(version: u8, rx: Arc<Mutex<Receiver<Work>>>, done: Sender<Done>) {
    loop {
        let w = match lock(&rx).recv() {
            Ok(w) => w,
            Err(_) => return,
        };
        let body = match w.raw {
            Ok(Some(raw)) => features_of(&raw, version).map(Some),
            Ok(None) => Ok(None),
            Err(msg) => Err(msg),
        };
        if done
            .send(Done {
                seq: w.seq,
                cursor: w.cursor,
                body,
            })
            .is_err()
        {
            return;
        }
    }
}

// ---------- 呼び出し側（Python へ渡す） ----------

fn clear_cache(dir: &Path) {
    if let Ok(entries) = fs::read_dir(dir) {
        for e in entries.flatten() {
            if e.file_name().to_string_lossy().starts_with(CACHE_PREFIX) {
                let _ = fs::remove_file(e.path());
            }
        }
    }
}

fn parse_state(d: &Bound<'_, PyDict>, n_files: usize) -> PyResult<Cursor> {
    let get = |key: &str| -> PyResult<u64> {
        d.get_item(key)?
            .ok_or_else(|| PyValueError::new_err(format!("resume_state に {key} がない")))?
            .extract::<u64>()
    };
    if let Some(n) = d.get_item("n_files")? {
        if n.extract::<usize>()? != n_files {
            return Err(PyValueError::new_err("resume_state とファイル数が違う"));
        }
    }
    Ok(Cursor {
        file: get("file")? as usize,
        offset: get("offset")?,
        rng: get("rng")?,
    })
}

fn to_dict<'py>(py: Python<'py>, f: Features) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("stm_indices", f.stm_idx.into_pyarray(py))?;
    d.set_item("stm_offsets", f.stm_off.into_pyarray(py))?;
    d.set_item("nstm_indices", f.nstm_idx.into_pyarray(py))?;
    d.set_item("nstm_offsets", f.nstm_off.into_pyarray(py))?;
    d.set_item("score", f.score.into_pyarray(py))?;
    d.set_item("result", f.result.into_pyarray(py))?;
    Ok(d)
}

/// 学習用バッチのイテレータ。全ファイルを1周したら終わる。
#[pyclass(module = "shikibu_feat")]
pub struct BatchLoader {
    n_files: usize,
    result_rx: Mutex<Receiver<Done>>,
    /// 枠を返すための送信側。Drop で閉じて、読み係を止める。
    credit_tx: Option<Sender<()>>,
    pending: BTreeMap<u64, Done>,
    next_seq: u64,
    /// 直前に返したバッチの直後の位置
    cursor: Cursor,
    finished: bool,
    cache: Option<Arc<CacheHub>>,
    threads: Vec<JoinHandle<()>>,
}

#[pymethods]
impl BatchLoader {
    #[new]
    #[pyo3(signature = (paths, batch_size, version, shuffle_buffer=0, seed=0, num_threads=1,
                        cache_dir=None, max_cached=3, resume_state=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        paths: Vec<String>,
        batch_size: usize,
        version: i64,
        shuffle_buffer: usize,
        seed: u64,
        num_threads: usize,
        cache_dir: Option<String>,
        max_cached: usize,
        resume_state: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let version = crate::check_version(version)?;
        if batch_size == 0 {
            return Err(PyValueError::new_err("batch_size は1以上"));
        }
        if num_threads == 0 {
            return Err(PyValueError::new_err("num_threads は1以上"));
        }
        if cache_dir.is_some() && max_cached == 0 {
            return Err(PyValueError::new_err("max_cached は1以上"));
        }
        let mut sizes = Vec::with_capacity(paths.len());
        for p in &paths {
            sizes.push(psv::checked_size(p).map_err(PyValueError::new_err)?);
        }
        let paths: Arc<Vec<PathBuf>> = Arc::new(paths.iter().map(PathBuf::from).collect());
        let n_files = paths.len();

        let start = match &resume_state {
            None => Cursor {
                file: 0,
                offset: 0,
                rng: Rng::new(seed).state(),
            },
            Some(d) => parse_state(d, n_files)?,
        };
        if start.file > n_files || start.offset % REC as u64 != 0 {
            return Err(PyValueError::new_err("resume_state の位置が不正"));
        }
        if start.file < n_files && start.offset > sizes[start.file] {
            return Err(PyValueError::new_err("resume_state の位置がファイルの外"));
        }

        let cache = match cache_dir {
            Some(dir) => {
                let dir = PathBuf::from(dir);
                fs::create_dir_all(&dir).map_err(|e| PyValueError::new_err(e.to_string()))?;
                clear_cache(&dir);
                Some(Arc::new(CacheHub::new(dir, max_cached + 1)))
            }
            None => None,
        };

        // 作業係への枠は 2 × num_threads + 2 個。未回収のバッチがこれを超えないようにする。
        let (work_tx, work_rx) = mpsc::channel::<Work>();
        let (done_tx, done_rx) = mpsc::channel::<Done>();
        let (credit_tx, credit_rx) = mpsc::channel::<()>();
        for _ in 0..(2 * num_threads + 2) {
            let _ = credit_tx.send(());
        }
        let work_rx = Arc::new(Mutex::new(work_rx));
        let mut threads = Vec::new();
        if let Some(hub) = &cache {
            let (p, h) = (paths.clone(), hub.clone());
            threads.push(thread::spawn(move || copy_ahead(p, start.file, h)));
        }
        {
            let (p, c) = (paths.clone(), cache.clone());
            threads.push(thread::spawn(move || {
                produce(p, c, start, batch_size, shuffle_buffer, credit_rx, work_tx)
            }));
        }
        for _ in 0..num_threads {
            let (rx, tx) = (work_rx.clone(), done_tx.clone());
            threads.push(thread::spawn(move || work_loop(version, rx, tx)));
        }

        Ok(BatchLoader {
            n_files,
            result_rx: Mutex::new(done_rx),
            credit_tx: Some(credit_tx),
            pending: BTreeMap::new(),
            next_seq: 0,
            cursor: start,
            finished: false,
            cache,
            threads,
        })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        if self.finished {
            return Ok(None);
        }
        // 番号の順に取り出す（作業係の終わる順は問わない）
        let done = loop {
            if let Some(d) = self.pending.remove(&self.next_seq) {
                break d;
            }
            let rx = &self.result_rx;
            match py.allow_threads(|| lock(rx).recv()) {
                Ok(d) => {
                    self.pending.insert(d.seq, d);
                }
                Err(_) => {
                    self.finished = true;
                    return Err(PyRuntimeError::new_err("ローダーの作業が途中で止まった"));
                }
            }
        };
        self.next_seq += 1;
        match done.body {
            Err(msg) => {
                self.finished = true;
                Err(PyRuntimeError::new_err(msg))
            }
            Ok(None) => {
                self.cursor = done.cursor;
                self.finished = true;
                Ok(None)
            }
            Ok(Some(f)) => {
                self.cursor = done.cursor;
                if let Some(tx) = &self.credit_tx {
                    let _ = tx.send(());
                }
                Ok(Some(to_dict(py, f)?))
            }
        }
    }

    /// 直前に返したバッチの直後の位置。JSON にできる dict で、`resume_state` に渡せる。
    fn state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("file", self.cursor.file)?;
        d.set_item("offset", self.cursor.offset)?;
        d.set_item("rng", self.cursor.rng)?;
        d.set_item("n_files", self.n_files)?;
        Ok(d)
    }
}

impl Drop for BatchLoader {
    fn drop(&mut self) {
        if let Some(hub) = &self.cache {
            hub.stop();
        }
        // 枠の送信側を閉じると、読み係が待ちをやめて終わる。その後に作業係も終わる。
        drop(self.credit_tx.take());
        for h in self.threads.drain(..) {
            let _ = h.join();
        }
        if let Some(hub) = &self.cache {
            clear_cache(&hub.dir);
        }
    }
}
