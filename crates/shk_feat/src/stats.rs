//! データ統計（§S4-1・§S4-2）と eval_scale の推定（§S4-3）。

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::psv;
use crate::rng::Rng;

/// `game_result == 0` の割合がこれを超えると、勝敗は記録されていないとみなす（§S4-2）。
const MISSING_RATE: f64 = 0.95;
/// 推定から除く評価値の絶対値（詰みなど、§S4-3）。
const MATE_LIMIT: i32 = 30000;
/// 黄金分割探索の範囲（eval_scale、§S4-3）。
const SCALE_LO: f64 = 50.0;
const SCALE_HI: f64 = 5000.0;

fn check_result(res: i8) -> Result<(), String> {
    if (-1..=1).contains(&res) {
        Ok(())
    } else {
        Err(format!("game_result が −1〜1 の外: {res}"))
    }
}

struct Stats {
    records: u64,
    /// 添字 0 / 1 / 2 = game_result −1 / 0 / 1
    result_counts: [u64; 3],
    score_min: i64,
    score_max: i64,
    score_sum: i64,
    ply_min: u16,
    ply_max: u16,
    ply_sum: u64,
}

impl Stats {
    fn new() -> Self {
        Stats {
            records: 0,
            result_counts: [0; 3],
            score_min: i64::MAX,
            score_max: i64::MIN,
            score_sum: 0,
            ply_min: u16::MAX,
            ply_max: 0,
            ply_sum: 0,
        }
    }

    fn add(&mut self, rec: &[u8]) -> Result<(), String> {
        let res = psv::result(rec);
        check_result(res)?;
        self.result_counts[(res + 1) as usize] += 1;
        let score = psv::score(rec) as i64;
        self.score_min = self.score_min.min(score);
        self.score_max = self.score_max.max(score);
        self.score_sum += score;
        let ply = psv::ply(rec);
        self.ply_min = self.ply_min.min(ply);
        self.ply_max = self.ply_max.max(ply);
        self.ply_sum += ply as u64;
        self.records += 1;
        Ok(())
    }

    fn result_missing(&self) -> bool {
        self.records > 0 && self.result_counts[1] as f64 / self.records as f64 > MISSING_RATE
    }
}

fn collect_stats(paths: &[String]) -> Result<Stats, String> {
    let mut st = Stats::new();
    for p in paths {
        psv::for_each_record(p, |rec| st.add(rec))?;
    }
    Ok(st)
}

/// データの統計を返す。`result_missing` は勝敗が記録されていないとみなされたとき True。
#[pyfunction]
pub fn psv_stats<'py>(py: Python<'py>, paths: Vec<String>) -> PyResult<Bound<'py, PyDict>> {
    let st = py
        .allow_threads(|| collect_stats(&paths))
        .map_err(PyValueError::new_err)?;
    if st.records == 0 {
        return Err(PyValueError::new_err("局面が1つもない"));
    }
    let counts = PyDict::new(py);
    counts.set_item("-1", st.result_counts[0])?;
    counts.set_item("0", st.result_counts[1])?;
    counts.set_item("1", st.result_counts[2])?;

    let d = PyDict::new(py);
    d.set_item("records", st.records)?;
    d.set_item("result_counts", counts)?;
    d.set_item("score_min", st.score_min)?;
    d.set_item("score_max", st.score_max)?;
    d.set_item("score_mean", st.score_sum as f64 / st.records as f64)?;
    d.set_item("result_missing", st.result_missing())?;
    d.set_item("ply_min", st.ply_min)?;
    d.set_item("ply_max", st.ply_max)?;
    d.set_item("ply_mean", st.ply_sum as f64 / st.records as f64)?;
    Ok(d)
}

/// 推定に使う1局面ぶん。
struct Sample {
    score: i16,
    /// game_result（−1 / 0 / 1）
    res: i8,
}

struct Estimate {
    scale: f64,
    bce: f64,
    used: usize,
}

fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// 平均 BCE(sigmoid(score / scale), r)。r は勝 = 1、引分 = 0.5、負 = 0。
/// log(sigmoid) を softplus で書いて、評価値が大きくても発散しないようにしている。
fn mean_bce(samples: &[Sample], scale: f64) -> f64 {
    let sum: f64 = samples
        .iter()
        .map(|s| {
            let z = s.score as f64 / scale;
            let r = (s.res as f64 + 1.0) * 0.5;
            r * softplus(-z) + (1.0 - r) * softplus(z)
        })
        .sum();
    sum / samples.len() as f64
}

/// 全データから `sample` 局面を一様に抽出し（抽出器、seed で決まる）、
/// BCE を最小にする eval_scale を log S について黄金分割探索で求める。
fn estimate(paths: &[String], sample: usize, seed: u64) -> Result<Estimate, String> {
    if sample == 0 {
        return Err("sample は1以上".into());
    }
    let mut rng = Rng::new(seed);
    let mut total = 0u64;
    let mut zeros = 0u64;
    let mut valid = 0usize;
    let mut picked: Vec<Sample> = Vec::with_capacity(sample.min(1 << 24));
    for p in paths {
        psv::for_each_record(p, |rec| {
            let res = psv::result(rec);
            check_result(res)?;
            total += 1;
            if res == 0 {
                zeros += 1;
            }
            let score = psv::score(rec);
            if (score as i32).abs() >= MATE_LIMIT {
                return Ok(());
            }
            // 復元抽出なしの一様抽出（Algorithm R）
            valid += 1;
            let s = Sample { score, res };
            if picked.len() < sample {
                picked.push(s);
            } else {
                let j = rng.below(valid);
                if j < sample {
                    picked[j] = s;
                }
            }
            Ok(())
        })?;
    }
    if total == 0 {
        return Err("局面が1つもない".into());
    }
    if zeros as f64 / total as f64 > MISSING_RATE {
        return Err(
            "勝敗が記録されていないとみなした（game_result == 0 が95%超）。\
             eval_scale は推定できない。手動指定（--eval-scale）と λ = 1.0 で学習する"
                .into(),
        );
    }
    if picked.is_empty() {
        return Err("評価値が ±30000 未満の局面がない".into());
    }

    // 黄金分割探索（t = log S を [log 50, log 5000] で最小化）
    let f = |t: f64| mean_bce(&picked, t.exp());
    let gr = (5f64.sqrt() - 1.0) / 2.0;
    let (mut a, mut b) = (SCALE_LO.ln(), SCALE_HI.ln());
    let mut c = b - gr * (b - a);
    let mut d = a + gr * (b - a);
    let mut fc = f(c);
    let mut fd = f(d);
    while b - a > 1e-4 {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - gr * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + gr * (b - a);
            fd = f(d);
        }
    }
    let scale = ((a + b) / 2.0).exp();
    Ok(Estimate {
        scale,
        bce: mean_bce(&picked, scale),
        used: picked.len(),
    })
}

/// 勝敗から eval_scale を推定する（§S4-3）。勝敗が記録されていなければ ValueError。
#[pyfunction]
#[pyo3(signature = (paths, sample=10_000_000, seed=0))]
pub fn estimate_eval_scale<'py>(
    py: Python<'py>,
    paths: Vec<String>,
    sample: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let est = py
        .allow_threads(|| estimate(&paths, sample, seed))
        .map_err(PyValueError::new_err)?;
    let d = PyDict::new(py);
    d.set_item("eval_scale", est.scale)?;
    d.set_item("bce", est.bce)?;
    d.set_item("used", est.used)?;
    Ok(d)
}
